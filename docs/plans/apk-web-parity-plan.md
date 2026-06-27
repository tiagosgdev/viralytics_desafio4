# APK ↔ Web Parity — Implementation Plan (4 gaps)

> Status: **PLANNED & REVIEWED — not yet implemented.** Resume here.
> Branch: `sensores`. Authored 2026-06-26.
> Pipeline so far: history-archaeology ✅ → plan ✅ → plan-review ✅ (NEEDS-REWORK, folded in below) → **implement (next)** → review-implementation (after).

## Goal
Bring the Android APK (`android_app/`, Kotlin, `com.viralytics.mobile`) to parity with the web base app on four capabilities. **All four are build-fresh Kotlin ports** — verified they never existed in the APK on any branch (see memory `apk-missing-features`). The web app is the authoritative reference.

## CONFIRMED ARCHITECTURE (decides where everything goes)
- The robot is a **split**: a **phone** (AppMode.PHONE_CAMERA) is **camera-only** — it takes the photo and uploads to `/api/mobile/scan` (the robot tablet has no usable camera). The **tablet** (AppMode.TABLET, detected via `Robot.globalContext() != null`, `MainActivity.kt:322`) is the robot's screen: it **displays recommendations, hosts the chat, and shows the 1-5 feedback** — fed by **MQTT** (`handleScanResult` → `injectScanResult`).
- The tablet **has HTTP access** to the PC server (it already calls `/api/robot/navigate-by-category` and chat over HTTP), so it can call `/api/recommend` and `/api/feedback`.
- Server detects body type / color **server-side from the phone's uploaded photo**; results reach the tablet over MQTT.
- **Gap 4 latency decision: ACCEPTED** — enable pose on every `/api/mobile/scan` (one-liner), no mitigation for v1.

### The pivotal correction (from plan-review)
The original plan wired the new logic into the **phone's** HTTP path. WRONG. The phone is camera-only; the **tablet** displays. **Current bug:** `uploadScan` (phone) calls `fetchAgentRecommendations` (`MainViewModel.kt:100`) but `AgentRecsComplete` only renders in TABLET mode (`MainActivity.kt:308`) — so today the phone fetches agent recs that are discarded and the tablet never fetches. **Fix: MOVE the agent fetch from the phone path to the tablet MQTT path; remove it from the phone path.**

---

## APK tech facts (match this style)
- HTTP: one shared OkHttp `OkHttpClient` built in `MainViewModel.kt:43-58` (trust-all TLS), passed to repos (`:59-62`).
- JSON: `org.json` (`JSONObject`/`JSONArray`) everywhere. No Gson/Moshi/Retrofit.
- Threading: repos are `suspend fun … = withContext(Dispatchers.IO) { runCatching { … } }` returning `kotlin.Result<T>`. ViewModel calls from `viewModelScope.launch` (Main dispatcher) and posts `UiEvent` to `MutableLiveData _events`; `MainActivity.observeViewModel()` (`MainActivity.kt:278-313`) renders on main.
- State (`MainViewModel.kt:67-76`): `selectedPersona, currentSessionId, detectedCategories, currentRecommendations, currentConversationState, currentIncludeFilters, chatHistory`. **Missing today:** detectedColor, detectedBodyType, currentRoundId, a chat-intent buffer, gender.
- `RecommendationItem` (`MainActivity.kt:851-920`): `id,name,category,price,reason,imageUrl,brand,description,sku,stockStatus,sizes:List<String>,metadata:LinkedHashMap` + `toJson()`/`fromJson()`. **No `itemId:Int` / `size:String`.**
- Detail dialog: `dialog_recommendation_detail.xml`, inflated in `showRecommendationDetail(item)` (`MainActivity.kt:561-617`); rows built programmatically (`addDetailRow`, `dp(...)`).
- Tablet scan entry: MQTT `handleScanResult` (`MainActivity.kt:1254-1282`) → `injectScanResult` (`MainViewModel.kt:189-209`).
- Server scan publish: `src/mqtt_scan.py:17-24` publishes `session_id, persona, detections, recommendations, annotated_frame` — **NOT body_analysis**.

## Server contracts (confirmed by review)
- `/api/feedback` (`schemas.py:164-175`, `main.py:1274-1300`): `FeedbackRequest{round_id:str,item_id:int,size:str="",rating:int 1..5}` → `FeedbackResponse{ok,applied,reason,policy}`. Never raises; returns `ok=False` (HTTP 200) when rec system down or round/item unknown. **Best-effort — never surface ok=False as an error.**
- `round_id`: returned by `/api/recommend` (`RecommendResponse`), consumed by `submit_feedback`→`rl_store.add_reward(round_id,"item_id:size",reward)`. **Eviction:** a round survives only ~`PPO_ROLLOUT_ROUNDS=8` later rounds or `RL_ROUND_CACHE=200` LRU; late ratings silently drop to ok=False. Acceptable.
- `/api/recommend` `RecommendRequest` accepts `detected_color/detected_type/detected_body_type/user_gender/user_answer` (+ conf fields). Returns `{recommendations, round_id}`. Agent items have fields `rank,item_id,size,color,type,brand,price,agent_scores{...}` (`multi_agent/run.py:123-126`).
- `/api/chat` `ChatResponse.action` (`schemas.py:97`, set `main.py:1161`): `action == "searched"` is the unique signal for a completed real search (`search_app.py:2201`). `active_filters` is **nested**: `{"include":{"type":[...],"color":[...]},"exclude":{...}}` (`main.py:1137`).
- Gap 4: flipping `mobile_scan` `run_body_analysis=False`→`True` (`main.py:846`) makes `/api/mobile/scan` return `body_analysis.body_shape`; does NOT affect `/api/detect/image` (already True). Body-shape labels (`hourglass/pear/triangle/rectangle/inverted_triangle/apple/oval/trapezoid`) match `detected_body_type` vocab the body agent expects (`body_agent.py:93,104`). Pose-unavailable → null → `detected_body_type=""`, no crash. **Cost: heavy MediaPipe pose runs synchronously on every scan (hundreds of ms+).** Accepted.

---

## Implementation order (interlocks: escalation → user_answer → recommend → round_id → feedback)
0. Shared model (RecommendationItem + fromAgentJson)
1. Server: Gap 4 (`main.py:846`) + publish `body_analysis` over MQTT (`mqtt_scan.py`)
2. Gap 2: AgentRepository payload (+user_answer) + capture round_id + map via fromAgentJson
3. Tablet wiring: move agent fetch into the MQTT path; source color/body; reset round_id + intent buffer
4. Gap 3: chat escalation on `action == "searched"`
5. Gap 1: FeedbackRepository + ViewModel.submitFeedback + dialog feedback UI

---

## Step 0 — Shared model (`MainActivity.kt`, `RecommendationItem` 851-920)
- Add `val itemId: Int? = null,` and `val size: String? = null,`.
- `toJson()`: `itemId?.let{put("item_id",it)}`, `size?.let{put("size",it)}`.
- `fromJson()`: `itemId = if(json.has("item_id")) json.optInt("item_id") else null`; `size = json.optString("size").takeIf{it.isNotBlank()}` (DB-scan recs lack these → stay null; back-compat preserved).
- Add companion mapper `fromAgentJson(json)` mirroring web `formatAgentRec` (`frontend/js/ui/recommendations.js:4-24`):
  - `itemId = json.optInt("item_id")`; `id = itemId.toString()`.
  - name = `[brand, type.replace('_',' ')]` non-blank joined by space, else `"Item <itemId>"` (NOT "Unnamed item").
  - price: guard `if(json.has("price")) "EUR %.2f".format(json.optDouble("price")) else "N/A"` (handle non-numeric safely → avoid `EUR NaN`).
  - metadata: pull non-blank `color,style,pattern,material,fit,season,occasion,gender,age_group,size`.
  - `size = json.optString("size").takeIf{it.isNotBlank()}`; `sizes = listOfNotNull(size)`.
  - reason: short agent-score summary (rank + body/clothing/colour/stock from `agent_scores`).
- **Why:** agent items are currently mismapped through `fromJson` (reads id/name/category which agent items lack) → render as "Unnamed item / EUR 0". This mapper fixes it AND yields the `item_id`/`size` feedback needs.

## Step 1 — Server (Python)
- `src/api/main.py:846`: `run_body_analysis=False` → `True`.
- `src/mqtt_scan.py:17-24`: add `"body_analysis": <result>.get("body_analysis")` to the published payload (else the tablet never gets body_shape; color is already in published `detections`).

## Step 2 — Gap 2: `AgentRepository.recommend` (`AgentRepository.kt:13-43`)
- NOTE: it ALREADY sends `detected_color`, `detected_body_type`, `user_gender:""`. **Only genuinely new: `user_answer`.** Don't re-add existing keys.
- Signature: add `userAnswer: String` (and ensure real `userGender` passed, default "").
- Payload: `put("user_answer", userAnswer)`.
- Return: `Result<Pair<String?, List<RecommendationItem>>>` — capture `round_id = json.optString("round_id").takeIf{it.isNotBlank()}`; map items via `RecommendationItem.fromAgentJson(...)` (NOT fromJson).

## Step 3 — Tablet wiring (the key correction)
- `ScanResult` / MQTT parse: get color from `detections[0].color_name`, body from `body_analysis.body_shape`.
- In the **tablet** path (`MainActivity.handleScanResult` → `MainViewModel.injectScanResult` `:189-209`):
  - store `detectedType`(detections[0]), `detectedColor`, `detectedBodyType`;
  - **reset `currentRoundId = null` and clear `searchIntentMessages`** here;
  - call `fetchAgentRecommendations(baseUrl, detectedType, detectedColor, detectedBodyType, userAnswer="")`, capture `currentRoundId`, render via `AgentRecsComplete`.
  - (tablet has baseUrl from server_url SharedPreferences — pass it through `handleScanResult`.)
- **Remove** the agent fetch from the **phone** path (`uploadScan` `MainViewModel.kt:100`) — phone is camera-only; avoids a wasted/double round.
- `fetchAgentRecommendations` new signature: `(baseUrl, type, color, bodyType, userAnswer)`; on success set `currentRoundId` + `currentRecommendations` + emit `AgentRecsComplete`.
- MainViewModel new state: `detectedType, detectedColor, detectedBodyType, currentRoundId:String?, searchIntentMessages:MutableList<String>`. Also reset round_id + clear buffer in `clearSession` (`:235-242`).

## Step 4 — Gap 3: chat escalation
- `ChatRepository.ChatResult` (`:12-18`): add `action:String?` and `activeFilters:JSONObject?`. Parse `action = json.optString("action").takeIf{isNotBlank}` and `activeFilters = json.optJSONObject("active_filters")` **directly** — do NOT reuse the buggy `extractIncludeFilters(wholeResponse)` (`ChatRepository.kt:87-96`) which mistakes the `results` array for filters.
- `MainViewModel`:
  - confirm-word set + `accumulatedUserIntent()` mirroring web (`chat.js:13-19, 114-120`): trim → drop blanks & confirm words → `takeLast(6)` → join `". "`.
  - in `sendChat` (`:144-187`): push each user message to `searchIntentMessages` (`:151`).
  - onSuccess, if `result.action == "searched"`: derive `type`/`color` from `activeFilters.include` (`{type:[],color:[]}` nested) with fallback to `detectedType`/`detectedColor`; `intent = accumulatedUserIntent()`; **clear `searchIntentMessages`**; `fetchAgentRecommendations(baseUrl, type, color, detectedBodyType, intent)`.
  - add small `extractInclude(src)` reading `src.optJSONObject("include")`.
- Decide how escalation recs surface on the tablet (web auto-opens detail; here at minimum re-render via `AgentRecsComplete`).

## Step 5 — Gap 1: feedback
- New `FeedbackRepository.kt` (OkHttp + org.json, same style): `suspend submit(baseUrl, roundId, itemId, size, rating): Result<JSONObject>` POSTing `/api/feedback` `{round_id,item_id,size,rating}`.
- `MainViewModel`: instantiate it; `submitFeedback(baseUrl, item, rating, onResult)` — guard `currentRoundId != null && item.itemId != null` else `onResult("Feedback unavailable…")`; launch in `viewModelScope`; **invoke `onResult` from the coroutine continuation (main thread)**, NOT inside the IO block. Map response: ok+applied→"Thanks — the agent is learning 🎓", ok+!applied→"Thanks! Neutral — no change.", else reason. Never treat ok=False as an error.
- `dialog_recommendation_detail.xml`: append a `detailFeedbackSection` (label "Rate this recommendation" + horizontal `detailFeedbackRow` + `detailFeedbackStatus`).
- `MainActivity.showRecommendationDetail` (`:561-617`): gate `section.isVisible = (currentRoundId != null && item.itemId != null && baseUrl != null)`; build 5 emoji `TextView`s 😣😕😐🙂😍 (ratings 1-5), each `setOnClickListener { viewModel.submitFeedback(baseUrl, item, rating){ msg -> statusView.text = msg } }`. (Optional: highlight selected, like web `.selected`.)

---

## Decisions a human may still want to override
- Rating widget: plain emoji TextViews vs Material buttons; selected-state highlight (web highlights selected — not yet replicated).
- Gender source: hardcoded `""` (no gender field exists in the APK) → agent gender signal dead in v1. Could add a selector or profile later.
- Gap 4 latency: accepted as-is (heavy pose on every scan). Mitigations available later (lite model / off-critical-path).
- How chat-escalation recs surface on the tablet (re-render vs auto-open detail).

## Risks / watch-items
- Pose on every scan = added scan latency (heavy MediaPipe, synchronous). Won't time out (90s read), but user-perceived slower scan.
- round_id eviction: ratings >~8 rounds later silently drop (ok=False) — surface as best-effort only.
- `active_filters` is nested `{include:{...}}` — parse explicitly; avoid the buggy root-scan extractor.
- Fixing the agent-rec remap (Step 0) changes how agent cards render — visual check on the tablet.
- No full gradle/Android build assumed in this env; rely on careful self-review + the implementation-review agent.

## Reference (web sources to port)
- Feedback: `frontend/js/ui/recommendations.js:293-345`
- Recommend payload: `frontend/js/api.js:60-77`
- Chat escalation: `frontend/js/ui/chat.js:13-19, 114-147`
- Agent item mapper: `frontend/js/ui/recommendations.js:4-24`

## Files to touch
- Python: `src/api/main.py` (1 line), `src/mqtt_scan.py`
- Kotlin: `MainActivity.kt`, `MainViewModel.kt`, `AgentRepository.kt`, `ChatRepository.kt`, `ScanRepository.kt`, new `FeedbackRepository.kt`
- Layout: `res/layout/dialog_recommendation_detail.xml`
