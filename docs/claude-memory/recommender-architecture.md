---
name: recommender-architecture
description: The app has two separate recommenders (camera→SPADE vs chat→search); chat→agent wiring added 2026-06-23
metadata: 
  node_type: memory
  type: project
  originSessionId: 3d388e58-6fca-476f-b7b3-bb57e026d8ad
---

The Viralytics app (`src/api/main.py` + `frontend/index.html`) has **two independent recommenders**
that were historically disjoint (true on ALL branches — main, filipe, serra, Robot, reinforcement-
learning, etc.; verified 2026-06-23):

1. **Multi-agent SPADE** (`rec_system` = `RecommendationSystem`): POST `/api/recommend`. The
   colour/body/clothing/stock/rl debate + weighted Borda (Parts A–F). Returns `agent_scores`/
   `agent_weights`/`round_id`.
2. **Conversational search** (`search_service`): POST `/api/chat`. Persona text reply + embedding/
   vector search via its OWN intent parser (`llm_query_parser`). Produces filters + a reply, but
   NO agent weights and NO debate. This is what the Cruella/Edna chat box used.

**Historically the two never exchanged data.** The frontend's `triggerAgentRecommendations()`
(→ `/api/recommend`) was called ONLY from the camera handler and never sent `user_answer`, so
`FeatureWeightAgent` always took its fast-path fallback (fixed 30/30/25/15). The Part A conversation-
driven weighting (`analyze_intent`, `LNIAGIA/query_parsing/feature_weighting.py:543` — ONE Ollama
call: detected_* + user_answer → filters + 4 importance weights summing to 100) was therefore
**dormant in the live UI**. NOTE: this also means the live FE never exercised the path the Part E
harness drives — see [[spade-experiments-state]].

**Fix committed `156ced3` (2026-06-23, frontend-only — backend `/api/recommend` already accepted
`user_answer` via `schemas.py`):** the chat box now drives a multi-agent round.
`triggerAgentRecommendations(...)` gained a `userAnswer` arg; `sendChat()` + the voice handler call
it with the message after showing the `/api/chat` persona reply. The `/api/chat` search results
render first as a fast fallback; the agent round overwrites them on success and leaves them intact if
the broker is down. RL emoji feedback (`currentRoundId`) attaches to the chat-driven round.

**The agreed model (user's words):** *confidence* comes from the camera detection and **persists
unchanged until a re-scan**; *importance* is re-derived by the chat LLM (`analyze_intent`) each turn.
Both carry through the full interaction. Implemented via module vars
`currentDetectedTypeColorConf`/`currentDetectedBodyConf` (set on scan, reset only in `startScan`,
reused on chat-after-camera rounds; 1.0 when chat-first / no scan). Three supported flows:
chat-first, camera-first (fallback weights), chat-after-camera (refines importance, keeps scan conf).
