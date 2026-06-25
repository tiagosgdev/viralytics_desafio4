# Running the SPADE experiment harness on another machine

Quick guide to pull this branch on a second PC (the Linux + NVIDIA box) and run the
Part E experiment harness, plus how to restore the Claude Code memory so an assistant
session continues with full context.

Branch: `spade-dynamic-weights-experiments`.

---

## 1. Get the code

```bash
git clone <repo-url> viralytics_desafio3      # or: git pull
cd viralytics_desafio3
git checkout spade-dynamic-weights-experiments
git pull
```

## 2. Python environment

- Use a recent `python3` (3.11–3.13 all fine on Linux). A venv is recommended:
  ```bash
  python3 -m venv .venv && source .venv/bin/activate
  ```
- **Do NOT run `pip install -r requirements.txt` blindly** — it pins `numpy>=1.26.4,<2`,
  which is stale and fails to build on newer Pythons. Install what the harness needs directly:
  ```bash
  pip install spade pandas ollama          # core harness + agents
  pip install 'mediapipe>=0.10.14'         # only if you also run camera/pose; not needed for the harness
  ```
  (`spade` 4.x works even though the project pins `>=3.2.3`.)

## 3. XMPP broker (required for any live round)

```bash
docker compose up -d xmpp        # Prosody on port 5222
docker compose ps xmpp           # confirm "Up"
```

## 4. Ollama + model (required — both the weight agent and the LLM shopper use it)

```bash
# install/start ollama (see ollama.com), then pull the model the pipeline uses:
ollama pull qwen2.5:7b-instruct-q3_K_M
curl -s http://127.0.0.1:11434/api/version    # confirm reachable
```
On the NVIDIA box the 7B q3 model (~3.8 GB) fits entirely in 16 GB VRAM, so episodes
run materially faster than on the Mac (the per-episode cost is ~80% Ollama).
**Optional quality bump:** the shopper LLM hallucinated a bit on q3 — consider a higher
quant / larger model (e.g. `qwen2.5:7b-instruct` or `:14b-instruct`) now that VRAM allows it.
The model id lives in `LNIAGIA/query_parsing/llm_query_parser.py` (`OLLAMA_REFINER_MODEL`).

## 5. (Optional) detection model weights

Not needed for the harness (personas supply faked detections). Only needed if you run the
live camera app: download the team SharePoint weights bundle into `models/weights/`
(see `README.md:41` and `docs/claude-memory/spade-env-setup.md`).

## 6. Run the experiment

```bash
# IMPORTANT: stop any running uvicorn/FastAPI app first — the harness starts its OWN
# RecommendationSystem on the SAME XMPP JIDs, so they collide on the broker if both run.
python3 -m multi_agent.experiments.run_experiment
```
Default is the OFAT slice: 9 combos × 3 personas × 1 repeat = **27 episodes**. Progress prints
per episode (`▸ customer=… combo=… repeat=…` then `review = N`), and a `Mean review per combo`
table prints at the end. Results go to `multi_agent/experiments/results.db` (gitignored).

- Scale up: bump `repeats` in `multi_agent/experiments/spec.py` (→3 for signal), or move to the
  full factorial grid (Part E follow-up).
- Quick unit check (no broker/Ollama needed):
  ```bash
  python3 -m pytest tests/test_retrieval.py tests/test_agent_weights.py tests/test_strategies.py -q
  ```

## 7. Restore the Claude Code memory (so an assistant session continues with context)

The memory lives OUTSIDE the repo in `~/.claude/projects/<encoded-project-path>/memory/`,
where `<encoded-project-path>` is the project's absolute path with `/` replaced by `-`.
It can't be pre-placed because the path differs per machine. The files are checked in under
`docs/claude-memory/`. To restore:

```bash
# 1. Launch Claude Code once inside the project dir so it creates the project folder, then:
ls ~/.claude/projects/                       # find the folder for THIS project's abs path
PROJ=~/.claude/projects/<the-folder-you-see>
mkdir -p "$PROJ/memory"
cp docs/claude-memory/*.md "$PROJ/memory/"
```
`MEMORY.md` is the index loaded each session; `spade-experiments-state.md` is the START-HERE
file; `spade-experiment-run1-findings.md` has the run-1 diagnosis + the fixes applied.
`CLAUDE.md` (repo root) travels with git automatically. `.claude/settings.local.json` is
gitignored and trivial (a single `git fetch` permission) — recreate if you want it.

---

## What changed since run 1 (already on this branch)

Two fixes were applied after the first OFAT run scored low (see
`docs/claude-memory/spade-experiment-run1-findings.md`):

1. **Distinct-item retrieval** (`multi_agent/retrieval.py` + `multi_agent/aggregator.py`) — the
   candidate pool and top-k were flooded with the same garment across sizes (~7 real items for
   n=40). Now retrieval returns n distinct items (best-stocked size each) and the Borda top-k
   dedups by item_id. A live round now returns 10 distinct, colour/type-correct items.
2. **Accumulated shopper intent** (`multi_agent/experiments/run_experiment.py`) — the harness was
   sending only the latest shopper message, so the colour anchor drifted away by the final turn
   (which is the turn that gets reviewed). It now accumulates the last 6 shopper utterances like
   the real frontend (`accumulatedUserIntent()`), keeping intent anchored across turns.

**Still open (not fixed):** price / soft-attribute preferences (style/occasion/price) are dropped
at retrieval because `price` isn't in the stock query keys — see the run-1 findings memo.
