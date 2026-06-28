"""
PPOPolicy — the RLRecommenderAgent's learnable policy (PyTorch actor-critic).
──────────────────────────────────────────────────────────────────────────────
Each recommendation round is treated as a **one-step contextual-bandit episode**.
The policy acts **per candidate item**: a shared-trunk actor-critic network maps an
item's feature vector x to

    actor  : μ(x)              → a Gaussian over a score-logit  z ~ N(μ(x), σ)
    critic : V(x)             → expected return baseline

The proposed Borda score is ``s = sigmoid(z)``.  Sampling z is what makes the
policy stochastic — that is the exploration (an entropy bonus keeps σ from
collapsing), so no ε-greedy is needed.

Learning is **Proximal Policy Optimisation** (clipped surrogate objective):

    ratio   = exp(logπ_new(z|x) - logπ_old(z|x))
    L_clip  = E[ min(ratio · A,  clip(ratio, 1-ε, 1+ε) · A) ]
    L_value = E[ (V(x) - return)² ]
    loss    = -L_clip + c_v · L_value - c_e · entropy

with advantage A = return - V_old(x).  Because each episode is a single step,
there is no temporal discounting: the return is just the per-item reward
(pass-rate + emoji), and the critic is the only baseline.

ONE global policy is shared by all customers and checkpointed (network + Adam
state) to disk, so the learned pattern survives an app restart.
"""

from __future__ import annotations

import logging
import threading

import torch
import torch.nn as nn
from torch.distributions import Normal

from multi_agent.config import (
    PPO_CLIP_EPS,
    PPO_ENTROPY_COEF,
    PPO_EPOCHS,
    PPO_HIDDEN,
    PPO_INIT_LOG_STD,
    PPO_LR,
    PPO_MAX_GRAD_NORM,
    PPO_MINIBATCH,
    PPO_VALUE_COEF,
    RL_CHECKPOINT_PATH,
    RL_FRESH_START,
)
from multi_agent.rl.store import Transition

logger = logging.getLogger(__name__)

# Ordered feature schema — each name's index is its position in the input vector.
# Persisted with the checkpoint so a saved policy is validated against the code.
FEATURE_NAMES: tuple[str, ...] = (
    "bias",            # constant 1.0 (harmless; the linear layers also have biases)
    "color_match",     # 1.0 if item colour matches include.color (else detected colour)
    "type_match",      # 1.0 if item type   matches include.type  (else detected type)
    "gender_match",    # 1.0 if item gender ∈ {user_gender, "unisex"}
    "push_norm",       # push_score   min-max normalised across the candidate set
    "price_norm",      # price        min-max normalised across the candidate set
    "stock_norm",      # stock_count  min-max normalised across the candidate set
    "style_match",     # 1.0 if item style    ∈ include.style    (refined conversation)
    "occasion_match",  # 1.0 if item occasion ∈ include.occasion (refined conversation)
)
N_FEATURES = len(FEATURE_NAMES)


# ── Feature extraction (unchanged from the bandit version) ──────────────────────

def _norm_map(rows: list[dict], key: str) -> dict[int, float]:
    """Min-max normalise the numeric field `key` across `rows` → {index: value∈[0,1]}."""
    vals: list[float] = []
    for r in rows:
        try:
            vals.append(float(r.get(key) or 0.0))
        except (TypeError, ValueError):
            vals.append(0.0)
    lo, hi = (min(vals), max(vals)) if vals else (0.0, 0.0)
    span = (hi - lo) or 1.0
    return {i: (v - lo) / span for i, v in enumerate(vals)}


def _item_key(item_id, size) -> str:
    return f"{item_id}:{size}"


def _include_values(include: dict, axis: str) -> set[str]:
    """Case-insensitive set of include-filter values for `axis` (empty if absent)."""
    return {str(v).lower().strip() for v in (include.get(axis) or [])}


def extract_features(
    candidates: list[dict],
    context: dict,
    weights_result: dict | None = None,
) -> dict[str, list[float]]:
    """
    Build the per-item feature vector for every candidate, using only fields
    available during a sealed bid (CFP payload + round context).

    `weights_result` carries the refined conversation filters
    (`filters.include`). When an axis is present there it drives the match flags
    (style/occasion, and refined color/type); otherwise color/type fall back to
    the raw scan context. All membership tests are case-insensitive.
    """
    detected_color = str(context.get("detected_color") or "").lower().strip()
    detected_type  = str(context.get("detected_type")  or "").lower().strip()
    user_gender    = str(context.get("user_gender")    or "").lower().strip()

    include = ((weights_result or {}).get("filters") or {}).get("include") or {}
    inc_color    = _include_values(include, "color")
    inc_type     = _include_values(include, "type")
    inc_style    = _include_values(include, "style")
    inc_occasion = _include_values(include, "occasion")

    push  = _norm_map(candidates, "push_score")
    price = _norm_map(candidates, "price")
    stock = _norm_map(candidates, "stock_count")

    features: dict[str, list[float]] = {}
    for i, c in enumerate(candidates):
        item_color    = str(c.get("color")    or "").lower().strip()
        item_type     = str(c.get("type")     or "").lower().strip()
        item_gender   = str(c.get("gender")   or "").lower().strip()
        item_style    = str(c.get("style")    or "").lower().strip()
        item_occasion = str(c.get("occasion") or "").lower().strip()

        # Refined filter takes precedence; fall back to the raw scan when absent.
        if inc_color:
            color_match = 1.0 if item_color in inc_color else 0.0
        else:
            color_match = 1.0 if (detected_color and item_color == detected_color) else 0.0
        if inc_type:
            type_match = 1.0 if item_type in inc_type else 0.0
        else:
            type_match = 1.0 if (detected_type and item_type == detected_type) else 0.0

        gender_match   = 1.0 if (user_gender and item_gender in (user_gender, "unisex")) else 0.0
        style_match    = 1.0 if (inc_style    and item_style    in inc_style)    else 0.0
        occasion_match = 1.0 if (inc_occasion and item_occasion in inc_occasion) else 0.0

        features[_item_key(c["item_id"], c["size"])] = [
            1.0, color_match, type_match, gender_match,
            push[i], price[i], stock[i], style_match, occasion_match,
        ]
    return features


# ── Network ─────────────────────────────────────────────────────────────────────

class ActorCritic(nn.Module):
    """Shared MLP trunk with a Gaussian actor head and a scalar value head."""

    def __init__(self, n_features: int, hidden: int, init_log_std: float) -> None:
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(n_features, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden),     nn.Tanh(),
        )
        self.actor_mu  = nn.Linear(hidden, 1)   # mean of the score-logit
        self.critic    = nn.Linear(hidden, 1)   # state-value baseline
        # State-independent log σ, learnable (standard PPO continuous-control choice).
        self.log_std = nn.Parameter(torch.full((1,), float(init_log_std)))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.trunk(x)
        mu    = self.actor_mu(h).squeeze(-1)   # [N]
        value = self.critic(h).squeeze(-1)     # [N]
        return mu, value


# ── Policy ──────────────────────────────────────────────────────────────────────

class PPOPolicy:
    """Thread-safe PPO actor-critic with checkpoint persistence."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.net = ActorCritic(N_FEATURES, PPO_HIDDEN, PPO_INIT_LOG_STD)
        self.optim = torch.optim.Adam(self.net.parameters(), lr=PPO_LR)
        self._update_count = 0
        self.load()

    # ── Acting ──────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def act(
        self,
        features: dict[str, list[float]],
        deterministic: bool = False,
    ) -> tuple[dict[str, float], dict[str, Transition]]:
        """
        Score every candidate.  Returns (scores, transitions):
          scores       : item_key → score∈[0,1] (= sigmoid of the sampled logit)
          transitions  : item_key → Transition(x, action, logπ_old, V, score)

        With `deterministic=False` (serving + training) the score-logit is sampled
        from N(μ, σ); that sampling is the exploration.  `deterministic=True` uses
        the mean (handy for tests / eval).
        """
        keys = list(features.keys())
        X = torch.tensor([features[k] for k in keys], dtype=torch.float32)

        mu, value = self.net(X)
        std = self.net.log_std.exp()
        dist = Normal(mu, std)
        z = mu if deterministic else dist.sample()
        logp = dist.log_prob(z)
        scores = torch.sigmoid(z)

        out_scores: dict[str, float] = {}
        out_trans:  dict[str, Transition] = {}
        for i, k in enumerate(keys):
            s = float(scores[i])
            out_scores[k] = round(s, 6)
            out_trans[k] = Transition(
                item_key=k,
                x=features[k],
                action=float(z[i]),
                logprob=float(logp[i]),
                value=float(value[i]),
                score=s,
            )
        return out_scores, out_trans

    # ── Learning (PPO) ────────────────────────────────────────────────────────────

    def learn(self, transitions: list[Transition]) -> dict:
        """Run one PPO update over a collected on-policy rollout, then checkpoint."""
        if not transitions:
            return {"updated": False, "reason": "empty rollout"}

        with self._lock:
            X        = torch.tensor([t.x for t in transitions], dtype=torch.float32)
            actions  = torch.tensor([t.action for t in transitions], dtype=torch.float32)
            old_logp = torch.tensor([t.logprob for t in transitions], dtype=torch.float32)
            old_val  = torch.tensor([t.value for t in transitions], dtype=torch.float32)
            returns  = torch.tensor([t.reward for t in transitions], dtype=torch.float32)

            # One-step episodes ⇒ no temporal discounting; advantage = return - baseline.
            adv = returns - old_val
            # Guard against near-constant returns: when the spread is tiny, only
            # center (dividing by ~0 std would blow tiny differences into noise).
            std = adv.std()
            adv = (adv - adv.mean()) / std if std > 1e-3 else (adv - adv.mean())

            n = len(transitions)
            mb = min(PPO_MINIBATCH, n)
            last = {}
            for _ in range(PPO_EPOCHS):
                perm = torch.randperm(n)
                for start in range(0, n, mb):
                    idx = perm[start:start + mb]
                    mu, value = self.net(X[idx])
                    dist = Normal(mu, self.net.log_std.exp())
                    new_logp = dist.log_prob(actions[idx])

                    ratio = torch.exp(new_logp - old_logp[idx])
                    surr1 = ratio * adv[idx]
                    surr2 = torch.clamp(ratio, 1 - PPO_CLIP_EPS, 1 + PPO_CLIP_EPS) * adv[idx]
                    policy_loss = -torch.min(surr1, surr2).mean()
                    value_loss  = (value - returns[idx]).pow(2).mean()
                    entropy     = dist.entropy().mean()
                    loss = policy_loss + PPO_VALUE_COEF * value_loss - PPO_ENTROPY_COEF * entropy

                    self.optim.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.net.parameters(), PPO_MAX_GRAD_NORM)
                    self.optim.step()
                    last = {
                        "policy_loss": policy_loss.detach().item(),
                        "value_loss":  value_loss.detach().item(),
                        "entropy":     entropy.detach().item(),
                    }

            self._update_count += 1
            self._save_locked()
            stats = {
                "updated": True,
                "update_count": self._update_count,
                "rollout_size": n,
                "mean_return": round(float(returns.mean()), 4),
                "sigma": round(float(self.net.log_std.detach().exp()), 4),
                **{k: round(v, 4) for k, v in last.items()},
            }
        logger.info(f"[rl-ppo] PPO update #{stats['update_count']}: {stats}")
        return stats

    # ── Persistence ────────────────────────────────────────────────────────────

    def load(self) -> None:
        path = RL_CHECKPOINT_PATH
        if RL_FRESH_START:
            logger.info(
                f"[rl-ppo] RL_FRESH_START set; ignoring any checkpoint at {path} "
                "and starting from a fresh policy (update_count=0). Will save here."
            )
            return
        if not path.exists():
            logger.info(f"[rl-ppo] No checkpoint at {path}; starting from a fresh policy.")
            return
        try:
            ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
        except Exception as exc:
            logger.warning(f"[rl-ppo] Could not read checkpoint ({exc}); fresh policy.")
            return
        if tuple(ckpt.get("feature_names", ())) != FEATURE_NAMES:
            logger.warning("[rl-ppo] Checkpoint feature schema mismatch; starting fresh.")
            return
        try:
            self.net.load_state_dict(ckpt["model"])
            self.optim.load_state_dict(ckpt["optim"])
            self._update_count = int(ckpt.get("update_count", 0))
        except Exception as exc:
            logger.warning(f"[rl-ppo] Failed to restore checkpoint ({exc}); fresh policy.")
            return
        logger.info(f"[rl-ppo] Loaded checkpoint from {path} (updates={self._update_count}).")

    def save(self) -> None:
        with self._lock:
            self._save_locked()

    def _save_locked(self) -> None:
        """Persist network + optimizer state.  Caller must hold the lock."""
        try:
            RL_CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "feature_names": list(FEATURE_NAMES),
                    "model": self.net.state_dict(),
                    "optim": self.optim.state_dict(),
                    "update_count": self._update_count,
                },
                str(RL_CHECKPOINT_PATH),
            )
        except Exception as exc:
            logger.warning(f"[rl-ppo] Failed to persist checkpoint: {exc}")

    # ── Introspection ────────────────────────────────────────────────────────────

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "update_count": self._update_count,
                "sigma": round(float(self.net.log_std.detach().exp()), 4),
                "params": sum(p.numel() for p in self.net.parameters()),
            }

    @property
    def update_count(self) -> int:
        return self._update_count


# ── Module-level singleton ──────────────────────────────────────────────────────
# ONE global policy shared by the RLRecommenderAgent (acting + PPO updates) and the
# API feedback path (emoji reward attribution).  Named `rl_policy` (not `policy`) so
# it does not shadow this `policy` submodule when re-exported from the package.
rl_policy: PPOPolicy = PPOPolicy()
