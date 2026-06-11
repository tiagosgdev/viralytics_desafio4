"""
scripts/app/start_full_app.py
--------------------------------
Start the full FashionSense application with the integrated search stack.

This launcher checks the pieces the merged app needs to work end-to-end:
- Python dependencies import cleanly
- Ollama is reachable (or can be started)
- the expected LLM model exists locally
- the local Qdrant collection already exists

If the prerequisites are satisfied, it starts uvicorn in the foreground.

Usage:
    python scripts/app/start_full_app.py
    python scripts/app/start_full_app.py --reload
"""

from __future__ import annotations

import argparse
import os
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

import requests


REPO_ROOT = Path(__file__).resolve().parents[2]
OLLAMA_URL = "http://127.0.0.1:11434"
DEFAULT_OLLAMA_REFINER_MODEL = "qwen2.5:7b-instruct-q3_K_M"
DEFAULT_OLLAMA_ROUTER_MODEL = "qwen2.5:7b-instruct-q3_K_M"

# Backward-compatible alias used in legacy messages.
EXPECTED_OLLAMA_MODEL = DEFAULT_OLLAMA_REFINER_MODEL


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Start the full FashionSense app")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true", help="Run uvicorn with autoreload")
    parser.add_argument(
        "--detector-backend",
        default="yolov8",
        choices=["yolov8"],
        help="Detector backend to expose through the API",
    )
    parser.add_argument(
        "--model-weights",
        default="",
        help="Optional model weights folder name under models/weights (Cruella/YOLO)",
    )
    parser.add_argument(
        "--edna-weights",
        default="",
        help="Optional model weights folder name under models/weights for Edna (FashionNet). E.g. edna_1.2m",
    )
    parser.add_argument(
        "--skip-ollama",
        action="store_true",
        help="Skip the Ollama availability/model checks",
    )
    parser.add_argument(
        "--skip-vector-check",
        action="store_true",
        help="Skip the vector collection check",
    )
    parser.add_argument(
        "--auto-pull-model",
        action="store_true",
        help="Automatically pull the required Ollama model if it is missing",
    )
    parser.add_argument(
        "--skip-xmpp",
        action="store_true",
        help="Skip starting the XMPP broker (multi-agent recommendations will be unavailable)",
    )
    parser.add_argument(
        "--refiner-model",
        default="",
        help="Optional Ollama model for parser/refinement (defaults to env or legacy model)",
    )
    parser.add_argument(
        "--router-model",
        default="",
        help="Optional Ollama model for interaction tasks (defaults to env value or the lightweight router model)",
    )
    return parser.parse_args()


def ensure_imports() -> None:
    try:
        import fastapi  # noqa: F401
        import uvicorn  # noqa: F401
        import ultralytics  # noqa: F401
    except Exception as exc:  # pragma: no cover - environment-dependent
        raise SystemExit(
            "Python dependencies are missing or broken.\n"
            "Run: pip install -r requirements.txt\n"
            f"Details: {exc}"
        ) from exc


def ollama_ready() -> bool:
    try:
        response = requests.get(f"{OLLAMA_URL}/api/tags", timeout=2)
        return response.ok
    except requests.RequestException:
        return False


def find_ollama_exe() -> str | None:
    exe = shutil.which("ollama")
    if exe:
        return exe

    candidates = [
        Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Ollama" / "ollama.exe",
        Path("C:/Program Files/Ollama/ollama.exe"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return None


def start_ollama_if_needed() -> None:
    if ollama_ready():
        print("Ollama server is already reachable.")
        return

    ollama_exe = find_ollama_exe()
    if not ollama_exe:
        raise SystemExit(
            "Ollama is not running and the 'ollama' command was not found.\n"
            "Install Ollama and run `ollama serve`, then retry."
        )

    print("Ollama is not reachable. Trying to start `ollama serve`...")

    creationflags = 0
    if os.name == "nt":
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS

    subprocess.Popen(  # pragma: no cover - process management is environment-dependent
        [ollama_exe, "serve"],
        cwd=str(REPO_ROOT),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=creationflags,
    )

    for _ in range(20):
        time.sleep(1)
        if ollama_ready():
            print("Ollama server started successfully.")
            return

    raise SystemExit(
        "Tried to start Ollama, but it is still not reachable.\n"
        "Start it manually with `ollama serve` and retry."
    )


def _resolve_ollama_models(args: argparse.Namespace) -> dict[str, str | list[str]]:
    legacy_model = (os.getenv("OLLAMA_MODEL") or "").strip()

    refiner_model = (
        (args.refiner_model or "").strip()
        or (os.getenv("OLLAMA_REFINER_MODEL") or "").strip()
        or legacy_model
        or DEFAULT_OLLAMA_REFINER_MODEL
    )
    router_model = (
        (args.router_model or "").strip()
        or (os.getenv("OLLAMA_ROUTER_MODEL") or "").strip()
        or DEFAULT_OLLAMA_ROUTER_MODEL
    )

    required: list[str] = []
    for model_name in (refiner_model, router_model):
        if model_name and model_name not in required:
            required.append(model_name)

    return {
        "refiner": refiner_model,
        "router": router_model,
        "required": required,
    }


def _query_ollama_models() -> set[str]:
    try:
        response = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise SystemExit(f"Could not query Ollama models: {exc}") from exc

    models = response.json().get("models", [])
    return {
        str(model.get("name") or "").strip()
        for model in models
        if str(model.get("name") or "").strip()
    }


def ensure_ollama_models(required_models: list[str]) -> list[str]:
    available = _query_ollama_models()
    missing = []

    for model_name in required_models:
        if model_name in available:
            print(f"Found Ollama model: {model_name}")
        else:
            missing.append(model_name)

    return missing


def pull_ollama_models(model_names: list[str]) -> None:
    if not model_names:
        return

    ollama_exe = find_ollama_exe()
    if not ollama_exe:
        raise SystemExit(
            "Required Ollama model(s) are missing, and `ollama.exe` could not be found.\n"
            "Add Ollama to PATH or reinstall it, then retry."
        )

    for model_name in model_names:
        print(f"Pulling Ollama model `{model_name}`...")
        completed = subprocess.run([ollama_exe, "pull", model_name], cwd=str(REPO_ROOT))
        if completed.returncode != 0:
            raise SystemExit(
                f"Failed to pull `{model_name}`.\n"
                f"You can try manually with:\n\"{ollama_exe}\" pull {model_name}"
            )

    missing_after_pull = ensure_ollama_models(model_names)
    if missing_after_pull:
        missing_text = ", ".join(missing_after_pull)
        raise SystemExit(f"Model pull completed, but these model(s) are still missing: {missing_text}.")


def ensure_vector_collection() -> None:
    sys.path.insert(0, str(REPO_ROOT))
    try:
        from LNIAGIA.DB.vector.VectorDBManager import COLLECTION_NAME, _collection_exists, _get_client
    except Exception as exc:  # pragma: no cover - environment-dependent
        raise SystemExit(
            "Could not import the vector DB manager.\n"
            "Make sure the search dependencies are installed.\n"
            f"Details: {exc}"
        ) from exc

    try:
        client = _get_client()
        try:
            exists = _collection_exists(client)
        finally:
            client.close()
    except Exception as exc:  # pragma: no cover - environment-dependent
        raise SystemExit(f"Could not inspect the local vector collection: {exc}") from exc

    if exists:
        print(f"Found local vector collection: {COLLECTION_NAME}")
        return

    raise SystemExit(
        "The integrated search collection has not been built yet.\n"
        "Build it first with the LNIAGIA tooling, then rerun this launcher."
    )


def _docker_available() -> bool:
    try:
        result = subprocess.run(
            ["docker", "info"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=5,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _xmpp_container_running() -> bool:
    try:
        result = subprocess.run(
            ["docker", "inspect", "--format", "{{.State.Running}}", "viralytics_xmpp"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.stdout.strip() == "true"
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _wait_for_port(host: str, port: int, retries: int = 20, delay: float = 1.5) -> bool:
    for i in range(retries):
        try:
            with socket.create_connection((host, port), timeout=2):
                return True
        except OSError:
            if i < retries - 1:
                print(".", end="", flush=True)
                time.sleep(delay)
    return False


def ensure_xmpp_running() -> None:
    if not _docker_available():
        raise SystemExit(
            "Docker is not running or not installed.\n"
            "Start Docker Desktop, then retry.\n"
            "Or skip with --skip-xmpp (multi-agent recommendations will be unavailable)."
        )

    if _xmpp_container_running():
        print("XMPP broker (Prosody) is already running.")
        return

    print("Starting XMPP broker (Prosody)...")
    result = subprocess.run(
        ["docker", "compose", "up", "-d", "--build", "xmpp"],
        cwd=str(REPO_ROOT),
    )
    if result.returncode != 0:
        raise SystemExit(
            "Failed to start the XMPP broker.\n"
            "Check: docker compose logs xmpp"
        )

    print("Waiting for XMPP port 5222", end="", flush=True)
    if not _wait_for_port("127.0.0.1", 5222):
        raise SystemExit(
            "\nXMPP broker did not become ready in time.\n"
            "Check: docker logs viralytics_xmpp"
        )
    print(" ready.")


def build_env(args: argparse.Namespace, refiner_model: str, router_model: str) -> dict:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    env["DETECTOR_BACKEND"] = args.detector_backend
    env["OLLAMA_REFINER_MODEL"] = refiner_model
    env["OLLAMA_ROUTER_MODEL"] = router_model
    env["OLLAMA_MODEL"] = refiner_model
    if args.model_weights:
        env["MODEL_WEIGHTS"] = args.model_weights
    if args.edna_weights:
        env["FASHIONNET_WEIGHTS"] = args.edna_weights
    return env


def run_uvicorn(args: argparse.Namespace, env: dict) -> int:
    command = [
        sys.executable,
        "-m",
        "uvicorn",
        "src.api.main:app",
        "--host",
        args.host,
        "--port",
        str(args.port),
    ]
    if args.reload:
        command.append("--reload")

    print("\nStarting FashionSense...")
    print("  Bootstrapping services (models, vectors, camera, speech)...")
    print("  URL will be announced by Uvicorn after startup completes.")
    print(f"  Detector backend: {args.detector_backend}")
    print(f"  Parser/refiner model: {env.get('OLLAMA_REFINER_MODEL')}")
    print(f"  Interaction model : {env.get('OLLAMA_ROUTER_MODEL')}")
    if args.model_weights:
        print(f"  MODEL_WEIGHTS: {args.model_weights}")
    if args.edna_weights:
        print(f"  FASHIONNET_WEIGHTS: {args.edna_weights}")
    print("Press Ctrl+C to stop.\n")

    completed = subprocess.run(command, cwd=str(REPO_ROOT), env=env)
    return completed.returncode


def main() -> int:
    args = parse_args()
    ensure_imports()
    ollama_models = _resolve_ollama_models(args)
    required_models = list(ollama_models.get("required", []))
    refiner_model = str(ollama_models.get("refiner") or DEFAULT_OLLAMA_REFINER_MODEL)
    router_model = str(ollama_models.get("router") or DEFAULT_OLLAMA_ROUTER_MODEL)

    if not args.skip_ollama:
        start_ollama_if_needed()
        missing_models = ensure_ollama_models(required_models)
        if missing_models:
            if args.auto_pull_model:
                pull_ollama_models(missing_models)
            else:
                ollama_exe = find_ollama_exe()
                missing_text = ", ".join(missing_models)

                if ollama_exe:
                    commands_text = "\n".join(f"\"{ollama_exe}\" pull {name}" for name in missing_models)
                else:
                    commands_text = "\n".join(f"ollama pull {name}" for name in missing_models)

                if ollama_exe:
                    raise SystemExit(
                        f"Required Ollama model(s) are missing: {missing_text}.\n"
                        f"Run:\n{commands_text}\n"
                        "Or rerun this launcher with --auto-pull-model."
                    )
                raise SystemExit(
                    f"Required Ollama model(s) are missing: {missing_text}.\n"
                    f"Run:\n{commands_text}\n"
                    "Or rerun this launcher with --auto-pull-model."
                )
    else:
        print("Skipping Ollama checks.")

    if not args.skip_vector_check:
        ensure_vector_collection()
    else:
        print("Skipping vector collection check.")

    if not args.skip_xmpp:
        ensure_xmpp_running()
    else:
        print("Skipping XMPP broker check.")

    env = build_env(args, refiner_model=refiner_model, router_model=router_model)
    return run_uvicorn(args, env)


if __name__ == "__main__":
    raise SystemExit(main())
