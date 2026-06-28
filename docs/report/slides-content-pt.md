# Apresentação técnica — conteúdo dos slides (PT)

---

## Slide 1 — Design experimental (duas simulações isoladas)

**Explicação.** Duas simulações isoladas: **A** varia as personalidades dos agentes (RL fixo); **B** deixa o RL aprender
(personalidades fixas). O *shopper* LLM avalia **cada** item 1–5.

**Código-chave** — `multi_agent/experiments/run_experiment.py`, `multi_agent/experiments/shopper.py`
```python
if   mode == "full":  combos = list(full_factorial_combos())  # Sim A: 81 personality combos
elif mode == "curve": combos = [baseline_combo()]; feed_review = True  # Sim B: RL learns online

# per-item feedback: each top-10 item gets ITS OWN 1–5 rating → its own RL reward
for item in last_recs[:config.TOP_K]:
    r = item_ratings.get((int(item["item_id"]), str(item["size"])), rating)
    system.submit_feedback(round_id=last_round_id, item_id=int(item["item_id"]), rating=int(r))
```

---

## Slide 2 — Estratégia de recolha: focada vs ampla (ambas agregam com Borda)

| Métrica                          | Recolha focada (escolhida) |   Ampla + veto   |
| -------------------------------- | :------------------------: | :--------------: |
| Avaliação média (1–5)            |          **1,57**          |       1,28       |
| Itens distintos apresentados     |            1258            |   8978 (7,1×)    |
| Personalidades alteram avaliação?|        não (Δ ≤ 0,06)      |  sim (Δ = 0,16)  |

*(Ambas agregam os votos dos agentes com a mesma contagem de Borda ponderada e sensível a empates.)*

**Explicação.** Muda só a **recolha de candidatos**, não a agregação — ambas terminam com **Borda**. Focada (cor ∧ tipo)
ganha em satisfação; ampla+veto dá ~7× mais variedade e faz as personalidades pesarem (Δ = 0,16).

**Código-chave** — `multi_agent/agents/orchestrator.py`, `multi_agent/aggregator.py`
```python
# orchestrator.py:146 — the arms differ in RETRIEVAL (+ veto), NOT aggregation
if SELECTION_MODE == "veto_batch":           # broad retrieval (~1,300 candidates)
    # agents veto weak items: drop when Σ weight(vetoing agents) ≥ τ (=0.5)
    pool = [it for it in survivors if reject_mass(it, vetoes, agent_weights) < τ]
    top_k = borda_aggregate(pool, agent_weights, k=TOP_K)
else:                                        # focused retrieval (~45, colour ∧ type)
    top_k = borda_aggregate(proposals, agent_weights, k=TOP_K)
# → BOTH finish with the SAME weighted, tie-aware Borda count (aggregator.py:95)
```

---

## Slide 3 — Simulação A: personalidades dos agentes (Borda)

| Personalidade de cor    | Avaliação média (1–5) |
| ----------------------- | :-------------------: |
| purista (no tema)       |     `[TBD-grid]`      |
| harmonizador            |     `[TBD-grid]`      |
| aventureiro (contraste) |     `[TBD-grid]`      |
| **amplitude**           |   **`[TBD-grid]`**    |

Média por persona: escritório `[TBD-grid]` / casual `[TBD-grid]` / festa `[TBD-grid]`.

**Explicação.** Com o RL fixo, varremos todas as combinações de personalidades. A amplitude `[TBD-grid]` mostra se os
agentes `[são / não são]` determinantes na avaliação.

**Código-chave** — `multi_agent/strategies/colour.py`, `multi_agent/strategies/registry.py`
```python
# colour.py — one scorer; the score depends only on its params
if   item_color == detected:        s = p["exact"]
elif item_color in COMPATIBLE[det]: s = p["compatible"]
else:                               s = p["unrelated"]

# registry.py — a "personality" = a params set (+ veto strictness)
"purist":      {"exact":1.0, "compatible":0.65,"unrelated":0.20}  # veto 0.7
"harmonizer":  {"exact":0.85,"compatible":1.0, "unrelated":0.30}  # veto 0.3
"adventurous": {"exact":0.40,"compatible":0.70,"unrelated":1.0}   # no veto
```

---

## Slide 4 — Simulação B: aprendizagem RL (Borda, final)

| Fase de treino          | Avaliação média (1–5) | Retorno médio |
| ----------------------- | :-------------------: | :-----------: |
| Início (primeiros 25%)  |         2,44          |     0,185     |
| Fim (últimos 25%)       |         2,48          |     0,078     |
| **Δ (fim − início)**    |       **+0,04**       |  **−0,107**   |

**Explicação.** Em 300 episódios o RL **não aprendeu** a melhorar a satisfação (Δ +0,04). O limite é o **sinal**: voto
diluído, recompensa premeia o consenso, e avaliações com pouca variância.

**Código-chave** — `multi_agent/rl/policy.py`, `multi_agent/rl/store.py`, `multi_agent/config.py`
```python
# policy.py — 9-feature per-item vector (now incl. style/occasion)
FEATURE_NAMES = ("bias","color_match","type_match","gender_match",
                 "push_norm","price_norm","stock_norm","style_match","occasion_match")

# store.py:84 — per-item 1–5 rating → reward in [-1,+1]
def rating_reward(r): return (clamp(r,1,5) - 3) / 2.0      # 1→-1 … 3→0 … 5→+1

# config.py:79 — RL is ONE fixed slice of the vote (the "dilution")
RL_WEIGHT = float(os.environ.get("RL_WEIGHT", "0.15"))
```

---

## Slide 5 — Conclusões / trabalho futuro

**Explicação.** O recomendador multi-agente baseado em Borda oferece a melhor satisfação do cliente e corre de ponta a
ponta no robô físico. As personalidades dos agentes `[alteram / não alteram]` de forma mensurável as recomendações
(Sim A); o agente RL ainda não aprende com o feedback (Sim B), limitado pela diluição da recompensa e por um sinal de
avaliação de baixa variância. O passo mais promissor é promover o RL de simples votante para o **agregador
orquestrador** — que consome as pontuações dos quatro agentes e produz ele próprio o top-10, fazendo com que a avaliação
se torne a consequência direta e atribuível da sua própria decisão — a par de treino com **feedback real de produção** e
um catálogo coerente para alargar o sinal aprendível.
