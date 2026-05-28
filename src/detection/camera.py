"""
src/detection/camera.py
───────────────────────
Real-time camera pipeline with a 3-state UX flow:

  READY      → camera connection is open, waiting for a client start/retry command
  CAPTURING  → streams raw frames for CAPTURE_SECONDS (default 5s)
               accumulates detections across all frames
  ANALYSING  → briefly shown while recommendations are computed
  RESULTS    → sends final detections + recommendations, then waits
               for a client command:
                 { "cmd": "start" }      → begin the first capture cycle
                 { "cmd": "retry" }      → restart capture
                 { "cmd": "more_recs" }  → keep same detections, get new recs

WebSocket message types sent to client:
  { "type": "ready",   "message": "camera_ready" }
  { "type": "frame",   "frame": <b64>, "countdown": int, "phase": "capturing" }
  { "type": "frame",   "frame": <b64>, "countdown": 0,   "phase": "analysing" }
  { "type": "results", "detections": [...], "recommendations": [...] }
  { "type": "error",   "message": "..." }
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import time
from typing import Callable, Dict, List, Optional

import cv2
import numpy as np

from src.detection.detector import BaseDetector, Detection, DetectionResult
from src.recommendations.engine import RecommendationEngine


# ── Constants ──────────────────────────────────────────────────────────────
CAPTURE_SECONDS = 5      # how long to scan the user
FRAME_INTERVAL  = 0.08   # ~12 fps — feels live, light on CPU


BODY_OVERLAY_INTERVAL = 1.5  # seconds between expensive body-analysis preview refreshes


class CameraStream:
    """
    Manages a camera capture loop with a controlled UX state machine.
    One instance is shared across WebSocket connections via main.py.
    """

    def __init__(
        self,
        detector:    BaseDetector,
        recommender: RecommendationEngine,
        recommendation_resolver: Optional[Callable[[List[str]], List[dict]]] = None,
        body_analysis_resolver: Optional[Callable[[np.ndarray], tuple[dict | None, str | None]]] = None,
        source:      int | str = 0,
        width:       int = 1280,
        height:      int = 720,
    ):
        self.detector    = detector
        self.recommender = recommender
        self.recommendation_resolver = recommendation_resolver
        self.body_analysis_resolver = body_analysis_resolver
        self.source      = source
        self.width       = width
        self.height      = height
        self._cap: Optional[cv2.VideoCapture] = None
        self._is_warm = False
        self._last_body_overlay_time: float = 0.0
        self._cached_body_overlay: Optional[np.ndarray] = None

    # ── Public API ─────────────────────────────────────────────────────────

    async def run_session(self, send, receive, user_profile=None):
        """
        Full session loop for one WebSocket connection.

        send         : async callable — sends a JSON string to the client
        receive      : async callable — returns next raw client message string
        user_profile : optional dict with age_group, gender, preferences
        """
        self._open()
        try:
            await send(json.dumps({"type": "ready", "message": "camera_ready"}))
            started = False
            while True:
                if not started:
                    while True:
                        await self._send_preview_frame(send)
                        cmd = await self._wait_for_command(receive, timeout=FRAME_INTERVAL)
                        if cmd == "start":
                            started = True
                            break
                        if cmd == "disconnect":
                            return

                # Phase 1 — capture & accumulate detections
                accumulated, last_frame = await self._phase_capture(send)

                # Phase 2 — brief "analysing" pause
                await self._phase_analyse(send)

                # Phase 3 — filter accumulated by current conf threshold, send results
                threshold = self.detector.conf_thres
                filtered  = {
                    cat: conf for cat, conf in accumulated.items()
                    if conf >= threshold
                }
                dominant_cats = self._dominant_categories(filtered)
                recs = self._resolve_recommendations(dominant_cats, user_profile=user_profile)
                final_detections, final_b64, dom_color = self._build_final_results_frame(last_frame, filtered)

                # Print the aggregated dominant color once for this session results
                try:
                    if dom_color:
                        print(f"[DETECT-AVG] color={dom_color.get('rgb')} name='{dom_color.get('name')}'")
                    else:
                        print("[DETECT-AVG] color=<none>")
                except Exception:
                    pass

                body_analysis, body_annotated_frame = self._resolve_body_analysis(last_frame)

                await send(json.dumps({
                    "type": "results",
                    "detections": self._serialize_detections(final_detections),
                    "dominant":        dominant_cats,
                    "dominant_color":  dom_color,
                    "recommendations": recs,
                    "annotated_frame": final_b64,
                    "body_analysis": body_analysis,
                    "body_annotated_frame": body_annotated_frame,
                }))

                # Wait for user action
                cmd = await self._wait_for_command(receive)

                if cmd == "retry":
                    continue                  # restart full capture cycle

                elif cmd == "more_recs":
                    # Keep same outfit detections, cycle through new recs
                    while True:
                        new_recs = self._resolve_recommendations(dominant_cats, user_profile=user_profile)
                        await send(json.dumps({
                            "type": "results",
                            "detections": self._serialize_detections(final_detections),
                            "dominant":        dominant_cats,
                            "recommendations": new_recs,
                            "dominant_color":  dom_color,
                            "body_analysis": body_analysis,
                            "body_annotated_frame": body_annotated_frame,
                        }))
                        inner_cmd = await self._wait_for_command(receive)
                        if inner_cmd == "retry":
                            break           # break inner → restart capture
                        elif inner_cmd == "more_recs":
                            continue        # get yet more recs
                        else:
                            return          # disconnected
                    continue                # restart capture after retry

                else:
                    return                  # disconnected / unknown

        except Exception as e:
            try:
                await send(json.dumps({"type": "error", "message": str(e)}))
            except Exception:
                pass

    def run_local(self):
        """Blocking OpenCV window for local testing without WebSocket."""
        self._open()
        print("Camera open — press Q to quit")
        try:
            while True:
                frame = self._read_frame()
                if frame is None:
                    continue
                result    = self.detector.detect(frame)
                annotated = self.detector.draw(frame, result)
                cv2.imshow("FashionSense", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        finally:
            self._close()
            cv2.destroyAllWindows()

    # ── Phases ─────────────────────────────────────────────────────────────

    def warmup(self, frames: int = 8):
        """Open the camera early and discard a few frames to reduce first-scan latency."""
        self._open()
        for _ in range(max(frames, 1)):
            frame = self._read_frame()
            if frame is not None:
                self._is_warm = True

    def shutdown(self):
        self._close()

    async def _phase_capture(self, send) -> tuple[Dict[str, float], Optional[np.ndarray]]:
        """
        Stream annotated frames for CAPTURE_SECONDS.
        Returns averaged confidence per detected class plus the last raw frame.
        """
        t_start     = time.perf_counter()
        conf_totals: Dict[str, float] = {}
        conf_counts: Dict[str, int]   = {}
        last_frame: Optional[np.ndarray] = None

        while True:
            elapsed   = time.perf_counter() - t_start
            remaining = max(0.0, CAPTURE_SECONDS - elapsed)

            frame = self._read_frame()
            if frame is None:
                await asyncio.sleep(0.01)
                continue
            last_frame = frame.copy()

            result = self.detector.detect(frame)
            pose_preview = self._resolve_body_overlay_frame(frame)
            preview_base = pose_preview if pose_preview is not None else frame
            annotated = self._draw_capture_overlay(preview_base, result, int(remaining) + 1)

            # Accumulate
            for det in result.detections:
                conf_totals[det.class_name] = conf_totals.get(det.class_name, 0.0) + det.confidence
                conf_counts[det.class_name] = conf_counts.get(det.class_name, 0)   + 1

            _, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 75])
            b64    = base64.b64encode(buf).decode("utf-8")

            is_last = elapsed >= CAPTURE_SECONDS
            await send(json.dumps({
                "type":      "frame",
                "phase":     "capturing",
                "frame":     b64,
                "countdown": int(remaining) + 1,
                "fps":       round(1000 / max(result.inference_ms, 1), 1),
                "flash":     is_last,   # frontend triggers camera flash on this frame
            }))

            if is_last:
                break

            await asyncio.sleep(FRAME_INTERVAL)

        return (
            {
                cat: conf_totals[cat] / conf_counts[cat]
                for cat in conf_totals
            },
            last_frame,
        )

    async def _phase_analyse(self, send):
        """Flash an 'Analysing' overlay while recommendations are generated."""
        frame = self._read_frame()
        if frame is not None:
            h, w  = frame.shape[:2]
            dark  = frame.copy()
            cv2.rectangle(dark, (0, 0), (w, h), (10, 10, 20), -1)
            out   = cv2.addWeighted(dark, 0.55, frame, 0.45, 0)

            label = "ANALYSING YOUR OUTFIT..."
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
            cv2.putText(out, label, ((w - tw) // 2, (h + th) // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (232, 255, 71), 2, cv2.LINE_AA)

            _, buf = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, 75])
            b64    = base64.b64encode(buf).decode("utf-8")
            await send(json.dumps({"type": "frame", "phase": "analysing", "frame": b64, "countdown": 0}))

        await asyncio.sleep(0.5)

    async def _send_preview_frame(self, send):
        """Stream a raw preview frame while the app is waiting for the user to start scanning."""
        frame = self._read_frame()
        if frame is None:
            await asyncio.sleep(0.01)
            return

        preview_frame = self._resolve_body_overlay_frame(frame)
        if preview_frame is None:
            preview_frame = frame
        _, buf = cv2.imencode(".jpg", preview_frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
        b64 = base64.b64encode(buf).decode("utf-8")
        await send(json.dumps({
            "type": "frame",
            "phase": "preview",
            "frame": b64,
            "countdown": 0,
            "flash": False,
        }))

    def _build_final_results_frame(
        self,
        frame: Optional[np.ndarray],
        filtered: Dict[str, float],
    ) -> tuple[List[Detection], Optional[str], Optional[dict]]:
        if frame is None:
            return [], None, None

        result = self.detector.detect(frame)
        kept = [
            det for det in result.detections
            if det.class_name in filtered and det.confidence >= self.detector.conf_thres
        ]
        if not kept:
            return [], None, None

        annotated = self.detector.draw(
            frame,
            DetectionResult(
                detections=kept,
                frame_shape=frame.shape,
                inference_ms=result.inference_ms,
                timestamp=result.timestamp,
            ),
        )
        _, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])

        # Compute an aggregated dominant color across kept detections.
        # Average the RGB tuples; choose the most common non-empty color_name if available.
        dom_color = None
        try:
            rgbs = [getattr(d, 'color', None) for d in kept if getattr(d, 'color', None)]
            names = [getattr(d, 'color_name', '') for d in kept if getattr(d, 'color_name', '')]
            if rgbs:
                n = len(rgbs)
                avg_r = round(sum([int(c[0]) for c in rgbs]) / n)
                avg_g = round(sum([int(c[1]) for c in rgbs]) / n)
                avg_b = round(sum([int(c[2]) for c in rgbs]) / n)
                # majority color name
                cname = ''
                if names:
                    from collections import Counter
                    cnt = Counter(names)
                    # pick the most common name
                    cname = cnt.most_common(1)[0][0]
                dom_color = { 'rgb': [avg_r, avg_g, avg_b], 'name': str(cname) if cname else '' }
        except Exception:
            dom_color = None

        return kept, base64.b64encode(buf).decode("utf-8"), dom_color

    # ── Helpers ────────────────────────────────────────────────────────────

    def _serialize_detections(self, detections: List[Detection]) -> List[dict]:
        return [
            {
                "class_id": det.class_id,
                "class_name": det.class_name,
                "confidence": round(det.confidence, 3),
                "bbox": det.bbox,
                "color": list(det.color) if getattr(det, "color", None) else None,
                "color_name": getattr(det, "color_name", ""),
            }
            for det in detections
        ]

    async def _wait_for_command(self, receive, timeout: Optional[float] = 120) -> str:
        """Wait for a client command and return its cmd value."""
        try:
            if timeout is None:
                msg = await receive()
            else:
                msg = await asyncio.wait_for(receive(), timeout=timeout)
            if msg is None:
                return "disconnect"
            data = json.loads(msg) if isinstance(msg, str) else msg
            return data.get("cmd", "disconnect")
        except asyncio.TimeoutError:
            return "timeout"
        except Exception:
            return "disconnect"

    def _dominant_categories(self, accumulated: Dict[str, float]) -> List[str]:
        """Top 5 categories by average confidence."""
        return [
            cat for cat, _ in
            sorted(accumulated.items(), key=lambda x: x[1], reverse=True)[:5]
        ]

    def _resolve_recommendations(self, categories: List[str], user_profile=None) -> List[dict]:
        if self.recommendation_resolver is not None:
            try:
                resolved = self.recommendation_resolver(categories, user_profile)
                if isinstance(resolved, list):
                    return resolved
            except Exception as exc:
                print(f"Warning: DB recommendation resolver failed: {exc}")
        return self.recommender.recommend(categories)

    def _resolve_body_analysis(self, frame: Optional[np.ndarray]) -> tuple[dict | None, str | None]:
        if frame is None or self.body_analysis_resolver is None:
            return None, None
        try:
            return self.body_analysis_resolver(frame.copy())
        except Exception as exc:
            print(f"Warning: body analysis resolver failed: {exc}")
            return None, None

    def _resolve_body_overlay_frame(self, frame: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if frame is None or self.body_analysis_resolver is None:
            return None
        now = time.perf_counter()
        if now - self._last_body_overlay_time < BODY_OVERLAY_INTERVAL:
            return self._cached_body_overlay
        try:
            analysis, encoded_frame = self.body_analysis_resolver(frame.copy())
            if not analysis or not analysis.get("landmarks_detected") or not encoded_frame:
                self._last_body_overlay_time = now
                return self._cached_body_overlay
            buffer = np.frombuffer(base64.b64decode(encoded_frame), dtype=np.uint8)
            decoded = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
            self._cached_body_overlay = decoded
            self._last_body_overlay_time = now
            return decoded
        except Exception as exc:
            print(f"Warning: body overlay preview failed: {exc}")
            self._last_body_overlay_time = now
            return self._cached_body_overlay

    def _draw_capture_overlay(
        self, frame: np.ndarray, result: DetectionResult, countdown: int
    ) -> np.ndarray:
        out  = self.detector.draw(frame, result)
        h, w = out.shape[:2]

        # Countdown circle — top right
        cx, cy, r = w - 60, 60, 44
        cv2.circle(out, (cx, cy), r, (20, 20, 30), -1)
        cv2.circle(out, (cx, cy), r, (232, 255, 71), 3)
        cnt_str = str(countdown)
        (tw, th), _ = cv2.getTextSize(cnt_str, cv2.FONT_HERSHEY_SIMPLEX, 1.6, 3)
        cv2.putText(out, cnt_str, (cx - tw // 2, cy + th // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.6, (232, 255, 71), 3, cv2.LINE_AA)

        # Bottom label
        label = "SCANNING YOUR OUTFIT"
        (lw, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
        cv2.putText(out, label, ((w - lw) // 2, h - 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (232, 255, 71), 2, cv2.LINE_AA)
        return out

    def _open(self):
        if self._cap is not None and self._cap.isOpened():
            return

        backend = cv2.CAP_ANY
        if os.name == "nt" and isinstance(self.source, int):
            backend = cv2.CAP_DSHOW

        self._cap = cv2.VideoCapture(self.source, backend)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self._cap.set(cv2.CAP_PROP_FPS, 30)
        if not self._cap.isOpened():
            self._cap.release()
            self._cap = None
            raise RuntimeError(f"Cannot open camera source: {self.source}")

    def _close(self):
        if self._cap:
            self._cap.release()
            self._cap = None
            self._is_warm = False

    def _read_frame(self) -> Optional[np.ndarray]:
        if self._cap is None:
            return None
        ret, frame = self._cap.read()
        return frame if ret else None
