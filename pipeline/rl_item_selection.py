"""
Deep CAT — Reinforcement Learning-based Computerized Adaptive Testing.

Upgrades the Questions Agent's item selection from rule-based IRT
(maximum Fisher information criterion) to a learned policy that balances
information gain, user engagement (via response-time modeling), and
fatigue management across sessions.

Architecture:
  - DeepCATSelector: Dueling DQN that learns which item to present next
  - ResponseTimeModel: Log-normal response-time model for engagement detection
  - AdaptiveItemSelector: Blended selector that transitions from IRT to RL

The RL selector is a DROP-IN addition alongside the existing IRT-based
selection in `selection.py`.  It does NOT modify existing code — it wraps
the IRT selector and gradually takes over as the session progresses.

References:
  - Ghosh et al. (2021). BOBCAT: Bilevel Optimization-Based CAT. ICML.
  - Zhuang et al. (2022). Fully Adaptive Framework for CAT. AAAI.
  - Wang et al. (2020). Deep RL for CAT. NeurIPS Workshop.
  - van der Linden (2007). Handbook of Modern IRT.
  - Wise & Kong (2005). Response time effort measurement.
  - De Boeck & Jeon (2019). Joint response time and accuracy models.
"""

from __future__ import annotations

import logging
import math
import os
import random
from collections import deque
from dataclasses import asdict, dataclass, field
from typing import Any, Deque, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

logger = logging.getLogger("questions_agent.rl_item_selection")

_ALLOW_UNSAFE_CHECKPOINT_LOAD_ENV = "QUESTIONS_AGENT_ALLOW_UNSAFE_CHECKPOINT_LOAD"


def _env_truthy(name: str) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return False
    return str(raw).strip().lower() not in {"", "0", "false", "no", "off"}


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DeepCATConfig:
    """Configuration for the Deep CAT RL selector."""

    # DQN architecture
    hidden_dim: int = 128
    latent_dim: int = 8
    n_dueling_layers: int = 2

    # Training
    learning_rate: float = 1e-3
    gamma: float = 0.99
    batch_size: int = 32
    replay_capacity: int = 10_000
    target_update_freq: int = 100
    min_replay_size: int = 64

    # Exploration
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 5000

    # Reward shaping
    se_reduction_weight: float = 1.0
    engagement_bonus_weight: float = 0.3
    fatigue_penalty_weight: float = 0.2
    fatigue_decay_rate: float = 0.1

    # Blending schedule (AdaptiveItemSelector)
    irt_only_items: int = 5
    blend_start_items: int = 6
    blend_end_items: int = 15
    irt_weight_at_blend_start: float = 0.7
    irt_weight_at_blend_end: float = 0.3


# ---------------------------------------------------------------------------
# State representation
# ---------------------------------------------------------------------------

@dataclass
class CATState:
    """Encodes the adaptive testing session state for the DQN.

    All fields are numeric (floats) so they can be stacked into a tensor.
    The state is a fixed-size vector regardless of how many items exist.
    """

    theta_estimate: float = 0.0
    theta_se: float = 1.0
    n_items_answered: int = 0
    total_information: float = 0.0
    mean_response_time: float = 0.0
    rt_std: float = 0.0
    engagement_score: float = 1.0
    session_duration_minutes: float = 0.0
    instrument_coverage_ratio: float = 0.0
    recent_se_trend: float = 0.0  # negative = improving

    def to_tensor(self) -> torch.Tensor:
        """Convert state to a 1D float tensor of shape (state_dim,)."""
        return torch.tensor(
            [
                self.theta_estimate,
                self.theta_se,
                float(self.n_items_answered) / 50.0,  # normalize
                self.total_information,
                self.mean_response_time / 60.0,  # normalize to minutes
                self.rt_std / 30.0,
                self.engagement_score,
                self.session_duration_minutes / 30.0,
                self.instrument_coverage_ratio,
                self.recent_se_trend,
            ],
            dtype=torch.float32,
        )

    @staticmethod
    def state_dim() -> int:
        """Dimensionality of the state vector."""
        return 10


# ---------------------------------------------------------------------------
# Item feature encoder
# ---------------------------------------------------------------------------

class ItemFeatureEncoder(nn.Module):
    """Encodes per-item features into a fixed-size embedding.

    Item features include:
      - Fisher information at current theta
      - discrimination parameter
      - n_categories (normalized)
      - n_scales_shared (multiplex count)
      - time_since_last_asked (normalized)
      - difficulty (mean threshold)
    """

    ITEM_FEATURE_DIM = 6

    def __init__(self, embed_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(self.ITEM_FEATURE_DIM, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, item_features: torch.Tensor) -> torch.Tensor:
        """Encode item features.

        Args:
            item_features: shape (batch, n_items, ITEM_FEATURE_DIM)
                           or (n_items, ITEM_FEATURE_DIM)

        Returns:
            Item embeddings of shape (..., n_items, embed_dim)
        """
        return self.net(item_features)


# ---------------------------------------------------------------------------
# Dueling DQN
# ---------------------------------------------------------------------------

class DuelingDQN(nn.Module):
    """Dueling Deep Q-Network for item selection.

    Architecture (Wang et al. 2016 — Dueling Network Architectures):
      - Shared state encoder
      - Value stream: V(s) — how good is the current testing state
      - Advantage stream: A(s, a) — relative value of selecting item a

    Q(s, a) = V(s) + A(s, a) - mean(A(s, .))
    """

    def __init__(
        self,
        state_dim: int,
        n_items: int,
        hidden_dim: int = 128,
        item_embed_dim: int = 32,
    ):
        super().__init__()
        self.n_items = n_items

        # Item feature encoder
        self.item_encoder = ItemFeatureEncoder(embed_dim=item_embed_dim)

        # State encoder: state_dim -> hidden_dim
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Combine state + aggregated item info
        combined_dim = hidden_dim + item_embed_dim

        # Value stream: scalar V(s)
        self.value_stream = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        # Advantage stream: per-item A(s, a)
        # Takes state + individual item embedding -> advantage
        self.advantage_fc1 = nn.Linear(hidden_dim + item_embed_dim, hidden_dim // 2)
        self.advantage_fc2 = nn.Linear(hidden_dim // 2, 1)

    def forward(
        self,
        state: torch.Tensor,
        item_features: torch.Tensor,
        available_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute Q-values for all items.

        Args:
            state: (batch, state_dim) or (state_dim,)
            item_features: (batch, n_items, item_feat_dim) or (n_items, item_feat_dim)
            available_mask: (batch, n_items) or (n_items,) — 1 for available, 0 for not

        Returns:
            Q-values: (batch, n_items) or (n_items,)
        """
        squeeze = False
        if state.dim() == 1:
            state = state.unsqueeze(0)
            item_features = item_features.unsqueeze(0)
            if available_mask is not None:
                available_mask = available_mask.unsqueeze(0)
            squeeze = True

        batch_size = state.shape[0]
        n_items = item_features.shape[1]

        # Encode state
        state_enc = self.state_encoder(state)  # (batch, hidden_dim)

        # Encode items
        item_enc = self.item_encoder(item_features)  # (batch, n_items, item_embed_dim)

        # Aggregate item info (mean pooling over available items)
        if available_mask is not None:
            mask_expanded = available_mask.unsqueeze(-1)  # (batch, n_items, 1)
            masked_items = item_enc * mask_expanded
            item_count = available_mask.sum(dim=-1, keepdim=True).clamp(min=1.0)
            item_agg = masked_items.sum(dim=1) / item_count  # (batch, item_embed_dim)
        else:
            item_agg = item_enc.mean(dim=1)  # (batch, item_embed_dim)

        # Combined state representation
        combined = torch.cat([state_enc, item_agg], dim=-1)  # (batch, hidden+item_embed)

        # Value stream: V(s)
        value = self.value_stream(combined)  # (batch, 1)

        # Advantage stream: per-item
        state_expanded = state_enc.unsqueeze(1).expand(-1, n_items, -1)  # (batch, n, hidden)
        adv_input = torch.cat([state_expanded, item_enc], dim=-1)  # (batch, n, hidden+embed)
        adv_hidden = F.relu(self.advantage_fc1(adv_input))
        advantages = self.advantage_fc2(adv_hidden).squeeze(-1)  # (batch, n_items)

        # Dueling combination: Q = V + (A - mean(A))
        if available_mask is not None:
            # Compute mean only over available items
            masked_adv = advantages.clone()
            masked_adv[available_mask == 0] = 0.0
            n_avail = available_mask.sum(dim=-1, keepdim=True).clamp(min=1.0)
            adv_mean = masked_adv.sum(dim=-1, keepdim=True) / n_avail
        else:
            adv_mean = advantages.mean(dim=-1, keepdim=True)

        q_values = value + (advantages - adv_mean)

        # Mask unavailable items with very negative Q-value
        if available_mask is not None:
            q_values = q_values.masked_fill(available_mask == 0, -1e9)

        if squeeze:
            q_values = q_values.squeeze(0)

        return q_values


# ---------------------------------------------------------------------------
# Experience replay buffer
# ---------------------------------------------------------------------------

@dataclass
class Transition:
    """Single experience tuple for replay."""
    state: torch.Tensor
    item_features: torch.Tensor
    available_mask: torch.Tensor
    action: int
    reward: float
    next_state: torch.Tensor
    next_item_features: torch.Tensor
    next_available_mask: torch.Tensor
    done: bool


class ReplayBuffer:
    """Fixed-size circular experience replay buffer."""

    def __init__(self, capacity: int = 10_000):
        self.capacity = capacity
        self.buffer: Deque[Transition] = deque(maxlen=capacity)

    def push(self, transition: Transition) -> None:
        """Add a transition to the buffer."""
        self.buffer.append(transition)

    def sample(self, batch_size: int) -> List[Transition]:
        """Sample a random batch of transitions."""
        return random.sample(list(self.buffer), min(batch_size, len(self.buffer)))

    def __len__(self) -> int:
        return len(self.buffer)


# ---------------------------------------------------------------------------
# DeepCATSelector
# ---------------------------------------------------------------------------

class DeepCATSelector:
    """RL-based item selection using Dueling DQN.

    State: (theta_estimate, theta_se, n_items_answered,
            total_information, response_time_features,
            instrument_coverage)
    Action: select next item index from available pool
    Reward: -SE(theta) reduction + engagement bonus - fatigue penalty

    The DQN learns which item maximises long-term measurement precision
    while keeping the user engaged and avoiding fatigue.
    """

    def __init__(
        self,
        n_items: int,
        config: Optional[DeepCATConfig] = None,
    ):
        self.n_items = n_items
        self.config = config or DeepCATConfig()
        self.state_dim = CATState.state_dim()

        # Networks
        self.policy_net = DuelingDQN(
            state_dim=self.state_dim,
            n_items=n_items,
            hidden_dim=self.config.hidden_dim,
            item_embed_dim=32,
        )
        self.target_net = DuelingDQN(
            state_dim=self.state_dim,
            n_items=n_items,
            hidden_dim=self.config.hidden_dim,
            item_embed_dim=32,
        )
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(
            self.policy_net.parameters(),
            lr=self.config.learning_rate,
        )

        # Replay buffer
        self.replay_buffer = ReplayBuffer(capacity=self.config.replay_capacity)

        # Step counter for epsilon decay and target updates
        self.step_count = 0

    @property
    def epsilon(self) -> float:
        """Current exploration rate (linearly decayed)."""
        progress = min(1.0, self.step_count / max(1, self.config.epsilon_decay_steps))
        return self.config.epsilon_start + progress * (
            self.config.epsilon_end - self.config.epsilon_start
        )

    def select_next_item(
        self,
        state: CATState,
        item_features: torch.Tensor,
        available_mask: torch.Tensor,
        epsilon: Optional[float] = None,
    ) -> Tuple[int, Dict[str, Any]]:
        """Select next item using epsilon-greedy DQN policy.

        Args:
            state: current CATState of the testing session
            item_features: (n_items, ItemFeatureEncoder.ITEM_FEATURE_DIM)
            available_mask: (n_items,) — 1.0 for available items, 0.0 for taken/excluded
            epsilon: override exploration rate (uses scheduled epsilon if None)

        Returns:
            (selected_item_index, metadata_dict)
        """
        eps = epsilon if epsilon is not None else self.epsilon
        available_indices = torch.where(available_mask > 0.5)[0].tolist()

        if not available_indices:
            raise ValueError("No available items to select from")

        # Epsilon-greedy exploration
        if random.random() < eps:
            action = random.choice(available_indices)
            selection_mode = "explore"
        else:
            with torch.no_grad():
                state_tensor = state.to_tensor()
                q_values = self.policy_net(state_tensor, item_features, available_mask)
                action = int(q_values.argmax().item())
                selection_mode = "exploit"

        # Compute Q-values for metadata
        with torch.no_grad():
            state_tensor = state.to_tensor()
            q_values = self.policy_net(state_tensor, item_features, available_mask)

        metadata = {
            "action": action,
            "selection_mode": selection_mode,
            "epsilon": round(eps, 4),
            "q_value": round(float(q_values[action].item()), 4),
            "q_mean_available": round(
                float(q_values[available_mask > 0.5].mean().item()), 4
            ),
            "n_available": len(available_indices),
            "step_count": self.step_count,
        }

        self.step_count += 1
        return action, metadata

    def compute_reward(
        self,
        se_before: float,
        se_after: float,
        engagement_score: float,
        n_items_answered: int,
    ) -> float:
        """Compute shaped reward for a state transition.

        Reward = w1 * SE_reduction + w2 * engagement_bonus - w3 * fatigue_penalty

        Args:
            se_before: SE(theta) before this item
            se_after: SE(theta) after this item
            engagement_score: [0, 1] from ResponseTimeModel
            n_items_answered: items answered so far (for fatigue)

        Returns:
            Scalar reward value
        """
        cfg = self.config

        # SE reduction reward (positive when SE decreases)
        se_reduction = max(0.0, se_before - se_after)
        se_reward = cfg.se_reduction_weight * se_reduction

        # Engagement bonus (higher for engaged respondents)
        engagement_bonus = cfg.engagement_bonus_weight * engagement_score

        # Fatigue penalty (exponentially increasing with items answered)
        fatigue_penalty = cfg.fatigue_penalty_weight * (
            1.0 - math.exp(-cfg.fatigue_decay_rate * n_items_answered)
        )

        return se_reward + engagement_bonus - fatigue_penalty

    def update(
        self,
        transition: Transition,
    ) -> Optional[float]:
        """Store transition and update DQN from experience replay.

        Args:
            transition: experience tuple (s, item_feats, mask, a, r, s', item_feats', mask', done)

        Returns:
            Training loss if an update was performed, None otherwise.
        """
        self.replay_buffer.push(transition)

        # Don't train until we have enough samples
        if len(self.replay_buffer) < self.config.min_replay_size:
            return None

        # Sample mini-batch
        batch = self.replay_buffer.sample(self.config.batch_size)

        # Stack batch tensors
        states = torch.stack([t.state for t in batch])
        item_feats = torch.stack([t.item_features for t in batch])
        masks = torch.stack([t.available_mask for t in batch])
        actions = torch.tensor([t.action for t in batch], dtype=torch.long)
        rewards = torch.tensor([t.reward for t in batch], dtype=torch.float32)
        next_states = torch.stack([t.next_state for t in batch])
        next_item_feats = torch.stack([t.next_item_features for t in batch])
        next_masks = torch.stack([t.next_available_mask for t in batch])
        dones = torch.tensor([float(t.done) for t in batch], dtype=torch.float32)

        # Current Q-values: Q(s, a)
        q_values = self.policy_net(states, item_feats, masks)
        q_selected = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        # Target Q-values: r + gamma * max_a' Q_target(s', a') * (1 - done)
        with torch.no_grad():
            next_q = self.target_net(next_states, next_item_feats, next_masks)
            next_q_max = next_q.max(dim=1).values
            target = rewards + self.config.gamma * next_q_max * (1.0 - dones)

        # Huber loss (smooth L1)
        loss = F.smooth_l1_loss(q_selected, target)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        # Periodic target network update
        if self.step_count % self.config.target_update_freq == 0:
            self.sync_target_network()

        return float(loss.item())

    def sync_target_network(self) -> None:
        """Hard copy policy network weights to target network."""
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def save(self, path: str) -> None:
        """Save model weights and optimizer state."""
        torch.save(
            {
                "format_version": 2,
                "n_items": self.n_items,
                "policy_net": self.policy_net.state_dict(),
                "target_net": self.target_net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "step_count": self.step_count,
                # Store pure primitives to avoid pickle-backed object loading.
                "config": asdict(self.config),
            },
            path,
        )

    def load(self, path: str) -> None:
        """Load model weights and optimizer state."""
        checkpoint: dict[str, Any]
        try:
            raw = torch.load(path, map_location="cpu", weights_only=True)
        except TypeError as exc:
            if not _env_truthy(_ALLOW_UNSAFE_CHECKPOINT_LOAD_ENV):
                raise RuntimeError(
                    "Refusing unsafe checkpoint load because this torch runtime "
                    "does not support weights_only=True. "
                    f"Set {_ALLOW_UNSAFE_CHECKPOINT_LOAD_ENV}=1 only for trusted legacy checkpoints."
                ) from exc
            raw = torch.load(path, map_location="cpu", weights_only=False)
        except Exception as exc:
            if not _env_truthy(_ALLOW_UNSAFE_CHECKPOINT_LOAD_ENV):
                raise RuntimeError(
                    "Checkpoint uses an unsafe legacy format or unsupported payload. "
                    f"Set {_ALLOW_UNSAFE_CHECKPOINT_LOAD_ENV}=1 only for trusted legacy checkpoints."
                ) from exc
            raw = torch.load(path, map_location="cpu", weights_only=False)

        if not isinstance(raw, dict):
            raise ValueError("Invalid checkpoint payload: expected a dictionary.")
        checkpoint = raw

        required_keys = {"policy_net", "target_net", "optimizer", "step_count", "config"}
        missing = required_keys - set(checkpoint.keys())
        if missing:
            raise ValueError(f"Invalid checkpoint payload: missing keys {sorted(missing)}")

        ckpt_n_items = int(checkpoint.get("n_items", self.n_items))
        if ckpt_n_items != self.n_items:
            raise ValueError(
                f"Checkpoint n_items={ckpt_n_items} does not match current model n_items={self.n_items}"
            )

        cfg_raw = checkpoint.get("config")
        if isinstance(cfg_raw, dict):
            # Parse for schema validation; runtime architecture remains the current instance.
            DeepCATConfig(**cfg_raw)

        self.policy_net.load_state_dict(checkpoint["policy_net"])
        self.target_net.load_state_dict(checkpoint["target_net"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.step_count = int(checkpoint["step_count"])


# ---------------------------------------------------------------------------
# Response Time Model
# ---------------------------------------------------------------------------

class ResponseTimeModel:
    """Log-normal response time model for engagement detection.

    Engagement scoring based on response latency:
      - Very fast (< 2s): likely careless -> low score
      - Fast (2-5s): quick but plausible -> moderate score
      - Optimal (5-30s): thoughtful response -> high score
      - Slow (30-60s): deliberating or reading carefully -> moderate score
      - Very slow (> 60s): likely distracted -> low score

    The model maintains an online estimate of the log-normal parameters
    (mu, sigma) using Bayesian updates.

    References:
      - Wise & Kong (2005). Response time effort in psychometric measurement.
      - van der Linden (2006). A lognormal model for response times on test items.
      - De Boeck & Jeon (2019). Joint response time and accuracy models.
    """

    def __init__(
        self,
        mu: float = 2.5,
        sigma: float = 0.8,
        prior_strength: float = 5.0,
    ):
        """Initialize the response time model.

        Args:
            mu: log-normal location parameter (log seconds).
                Default 2.5 -> median ~12s.
            sigma: log-normal scale parameter.
            prior_strength: pseudo-count for Bayesian prior (higher = more stable).
        """
        self.mu = mu
        self.sigma = sigma
        self.prior_mu = mu
        self.prior_sigma = sigma
        self.prior_strength = prior_strength
        self.n_observations = 0
        self._log_rt_sum = 0.0
        self._log_rt_sq_sum = 0.0

    def engagement_score(self, response_time_seconds: float) -> float:
        """Compute engagement score in [0, 1] from response time.

        Uses a smooth function that peaks in the optimal range and falls
        off for very fast (careless) or very slow (distracted) responses.

        Args:
            response_time_seconds: time in seconds from item display to submission.

        Returns:
            Score in [0, 1] where 1.0 = maximal engagement.
        """
        if response_time_seconds <= 0:
            return 0.0

        log_rt = math.log(max(response_time_seconds, 0.1))

        # Compute z-score relative to the learned distribution
        z = (log_rt - self.mu) / max(self.sigma, 0.1)

        # Smooth bell-shaped engagement curve centered on the population mean
        # exp(-z^2 / 2) gives a Gaussian shape in log-RT space
        raw_score = math.exp(-0.5 * z * z)

        # Penalty for extremely fast responses (likely careless)
        if response_time_seconds < 2.0:
            fast_penalty = response_time_seconds / 2.0  # linear 0->1 over 0-2s
            raw_score *= fast_penalty

        return max(0.0, min(1.0, raw_score))

    def classify_response(self, response_time_seconds: float) -> str:
        """Classify response time into engagement categories.

        Args:
            response_time_seconds: response latency in seconds.

        Returns:
            One of: 'careless', 'quick', 'optimal', 'deliberate', 'distracted'
        """
        if response_time_seconds < 2.0:
            return "careless"
        elif response_time_seconds < 5.0:
            return "quick"
        elif response_time_seconds < 30.0:
            return "optimal"
        elif response_time_seconds < 60.0:
            return "deliberate"
        else:
            return "distracted"

    def update_parameters(self, response_times: List[float]) -> None:
        """Online Bayesian update of log-normal parameters.

        Uses conjugate normal-inverse-gamma updates for the log-normal.
        The prior acts as a regularizer to prevent wild swings from
        small samples.

        Args:
            response_times: list of response times in seconds.
        """
        if not response_times:
            return

        valid_rts = [rt for rt in response_times if rt > 0]
        if not valid_rts:
            return

        # Accumulate sufficient statistics in log space
        for rt in valid_rts:
            log_rt = math.log(rt)
            self._log_rt_sum += log_rt
            self._log_rt_sq_sum += log_rt * log_rt
            self.n_observations += 1

        # Bayesian posterior for mu (conjugate normal with known variance proxy)
        n = self.n_observations
        n0 = self.prior_strength
        posterior_n = n + n0

        # Posterior mean of mu
        self.mu = (n0 * self.prior_mu + self._log_rt_sum) / posterior_n

        # Posterior estimate of sigma (using sample variance + prior)
        if n >= 2:
            sample_mean = self._log_rt_sum / n
            sample_var = max(
                0.01,
                (self._log_rt_sq_sum / n) - sample_mean * sample_mean,
            )
            # Weighted average of prior and sample variance
            prior_var = self.prior_sigma ** 2
            self.sigma = math.sqrt(
                (n0 * prior_var + n * sample_var) / posterior_n
            )

    def expected_response_time(self) -> float:
        """Expected (mean) response time given current parameters.

        For log-normal: E[X] = exp(mu + sigma^2 / 2)
        """
        return math.exp(self.mu + 0.5 * self.sigma ** 2)

    def median_response_time(self) -> float:
        """Median response time given current parameters.

        For log-normal: median = exp(mu)
        """
        return math.exp(self.mu)


# ---------------------------------------------------------------------------
# Adaptive Item Selector (blends IRT + RL)
# ---------------------------------------------------------------------------

@dataclass
class SelectionResult:
    """Result of the adaptive item selection."""
    item_index: int
    item_id: Optional[str] = None
    selection_source: str = "irt"  # "irt", "rl", "blended"
    irt_score: float = 0.0
    rl_score: float = 0.0
    blended_score: float = 0.0
    irl_weight: float = 1.0
    rl_weight: float = 0.0
    engagement_score: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class AdaptiveItemSelector:
    """Unified selector that blends IRT + RL approaches.

    Blending strategy:
      - First N items (default 5): pure IRT (maximum information) for
        initial calibration of theta
      - Items N+1 to M (default 6-15): blend IRT + RL with linearly
        increasing RL weight
      - Items M+1 onward (default 16+): primarily RL, with IRT as a
        safety floor

    Response time always modulates item weights through the engagement
    model.  If a user appears disengaged (very fast or very slow),
    selection pivots to simpler/shorter items regardless of the
    IRT/RL blend.

    This is a DROP-IN wrapper — it accepts IRT information scores
    (as computed by `_compute_irt_information_scores` in selection.py)
    and RL Q-values, and blends them.
    """

    def __init__(
        self,
        n_items: int,
        rl_selector: Optional[DeepCATSelector] = None,
        rt_model: Optional[ResponseTimeModel] = None,
        config: Optional[DeepCATConfig] = None,
    ):
        """Initialize the adaptive selector.

        Args:
            n_items: total number of items in the item pool
            rl_selector: optional pre-trained DeepCATSelector
            rt_model: optional ResponseTimeModel (created fresh if None)
            config: DeepCAT configuration
        """
        self.n_items = n_items
        self.config = config or DeepCATConfig()
        self.rl_selector = rl_selector or DeepCATSelector(n_items, self.config)
        self.rt_model = rt_model or ResponseTimeModel()

    def irt_weight(self, n_items_answered: int) -> float:
        """Compute the IRT weight given the number of items answered.

        Returns a value in [irt_weight_at_blend_end, 1.0] that linearly
        decreases as the session progresses.
        """
        cfg = self.config
        if n_items_answered < cfg.irt_only_items:
            return 1.0
        if n_items_answered >= cfg.blend_end_items:
            return cfg.irt_weight_at_blend_end

        # Linear interpolation
        progress = (n_items_answered - cfg.blend_start_items + 1) / max(
            1, cfg.blend_end_items - cfg.blend_start_items + 1
        )
        progress = max(0.0, min(1.0, progress))
        return cfg.irt_weight_at_blend_start + progress * (
            cfg.irt_weight_at_blend_end - cfg.irt_weight_at_blend_start
        )

    def select(
        self,
        state: CATState,
        irt_scores: Dict[int, float],
        item_features: torch.Tensor,
        available_mask: torch.Tensor,
        item_id_by_index: Optional[Dict[int, str]] = None,
    ) -> SelectionResult:
        """Select next item with metadata about selection rationale.

        Args:
            state: current CATState
            irt_scores: {item_index: Fisher_information} from IRT
            item_features: (n_items, feature_dim) tensor for the DQN
            available_mask: (n_items,) 1.0/0.0 mask
            item_id_by_index: optional mapping from index to item_id

        Returns:
            SelectionResult with selected item and rationale
        """
        n = state.n_items_answered
        w_irt = self.irt_weight(n)
        w_rl = 1.0 - w_irt

        available_indices = torch.where(available_mask > 0.5)[0].tolist()
        if not available_indices:
            raise ValueError("No available items to select from")

        # IRT component: normalize scores to [0, 1]
        irt_vals = {idx: irt_scores.get(idx, 0.0) for idx in available_indices}
        max_irt = max(irt_vals.values()) if irt_vals else 1.0
        max_irt = max(max_irt, 1e-8)
        irt_norm = {idx: v / max_irt for idx, v in irt_vals.items()}

        # RL component: get Q-values (pure IRT if w_rl ~ 0)
        rl_scores: Dict[int, float] = {}
        rl_metadata: Dict[str, Any] = {}
        if w_rl > 0.01:
            # Use greedy (epsilon=0) for the blended score
            with torch.no_grad():
                q_values = self.rl_selector.policy_net(
                    state.to_tensor(), item_features, available_mask
                )
            q_avail = q_values[available_mask > 0.5]
            q_min = float(q_avail.min().item())
            q_max = float(q_avail.max().item())
            q_range = max(q_max - q_min, 1e-8)
            for idx in available_indices:
                rl_scores[idx] = (float(q_values[idx].item()) - q_min) / q_range
            rl_metadata = {
                "q_min": round(q_min, 4),
                "q_max": round(q_max, 4),
            }

        # Blend scores
        blended: Dict[int, float] = {}
        for idx in available_indices:
            irt_s = irt_norm.get(idx, 0.0)
            rl_s = rl_scores.get(idx, 0.0)
            blended[idx] = w_irt * irt_s + w_rl * rl_s

        # Apply engagement modulation
        engagement = state.engagement_score
        if engagement < 0.3:
            # Disengage override: prefer items with highest IRT info
            # (shortest measurement path) when user appears disengaged
            blended = irt_norm

        # Select the item with highest blended score
        best_idx = max(blended, key=lambda k: blended[k])

        # Determine selection source label
        if w_irl_is_one := (w_irt >= 0.99):
            source = "irt"
        elif w_rl >= 0.99:
            source = "rl"
        else:
            source = "blended"

        return SelectionResult(
            item_index=best_idx,
            item_id=(item_id_by_index or {}).get(best_idx),
            selection_source=source,
            irt_score=irt_norm.get(best_idx, 0.0),
            rl_score=rl_scores.get(best_idx, 0.0),
            blended_score=blended.get(best_idx, 0.0),
            irl_weight=round(w_irt, 4),
            rl_weight=round(w_rl, 4),
            engagement_score=round(engagement, 4),
            metadata={
                "n_items_answered": n,
                "n_available": len(available_indices),
                **rl_metadata,
            },
        )

    def record_response_time(self, response_time_seconds: float) -> float:
        """Record a response time and return the engagement score.

        Args:
            response_time_seconds: latency in seconds

        Returns:
            Engagement score in [0, 1]
        """
        score = self.rt_model.engagement_score(response_time_seconds)
        self.rt_model.update_parameters([response_time_seconds])
        return score
