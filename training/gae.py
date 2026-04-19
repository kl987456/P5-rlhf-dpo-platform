"""
Generalized Advantage Estimation (GAE)
=======================================
GAE paper: Schulman et al., 2016 — https://arxiv.org/abs/1506.02438

A_t = sum_{l>=0} (gamma * lambda)^l * delta_{t+l}
delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)

Implementation uses a backward DP pass:
  - Base case at T-1: last_gae = 0
  - Recurrence: last_gae = delta_t + gamma*lambda*last_gae   (if not done)
  - Result: A_t stored backwards, O(N) time and O(N) space
"""

import torch
import numpy as np
from typing import List, Tuple


def compute_gae(
    rewards: List[float],
    values: List[float],
    dones: List[bool],
    gamma: float = 0.99,
    lam: float = 0.95,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute Generalized Advantage Estimation via backward DP.

    Args:
        rewards: r_t for t in [0, T-1]
        values:  V(s_t) for t in [0, T-1]  (bootstrap value at each state)
        dones:   episode-end flags; resets bootstrap when True
        gamma:   discount factor (default 0.99)
        lam:     GAE lambda smoothing parameter (default 0.95)

    Returns:
        advantages: normalized A_t tensor of shape (T,)
        returns:    TD(lambda) returns R_t = A_t + V(s_t), shape (T,)

    Complexity: O(T) time, O(T) space — single backward pass, no nested loops.
    """
    n = len(rewards)
    if n == 0:
        return torch.zeros(0), torch.zeros(0)

    advantages = np.zeros(n, dtype=np.float32)
    rewards_arr = np.array(rewards, dtype=np.float32)
    values_arr = np.array(values, dtype=np.float32)

    last_gae = 0.0  # base case: gae beyond episode end = 0

    for t in reversed(range(n)):
        # Bootstrap V(s_{t+1}); 0 if terminal or last timestep
        if t == n - 1 or dones[t]:
            next_value = 0.0
            carry_gae = 0.0   # no carry across episode boundary
        else:
            next_value = values_arr[t + 1]
            carry_gae = last_gae

        # TD residual: delta_t = r_t + gamma*V(s_{t+1}) - V(s_t)
        delta = rewards_arr[t] + gamma * next_value - values_arr[t]

        # Recurrence: A_t = delta_t + (gamma*lambda) * A_{t+1}
        last_gae = delta + gamma * lam * carry_gae
        advantages[t] = last_gae

    # TD(lambda) returns as targets for the value function
    returns = advantages + values_arr

    # Normalize advantages: zero mean, unit variance (stabilizes policy gradient)
    adv_mean = advantages.mean()
    adv_std = advantages.std()
    advantages = (advantages - adv_mean) / (adv_std + 1e-8)

    return torch.tensor(advantages, dtype=torch.float32), torch.tensor(returns, dtype=torch.float32)


def compute_gae_vectorized(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    gamma: float = 0.99,
    lam: float = 0.95,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Batch GAE for multiple parallel environments.

    Args:
        rewards: (T, B) array of rewards across T steps, B envs
        values:  (T, B) array of value estimates
        dones:   (T, B) bool array of done flags
        gamma, lam: same as compute_gae

    Returns:
        advantages: (T, B) normalized
        returns:    (T, B)
    """
    T, B = rewards.shape
    advantages = np.zeros((T, B), dtype=np.float32)
    last_gae = np.zeros(B, dtype=np.float32)

    for t in reversed(range(T)):
        mask = 1.0 - dones[t].astype(np.float32)  # 0 at terminal
        next_values = values[t + 1] if t < T - 1 else np.zeros(B, dtype=np.float32)
        next_values = next_values * mask

        delta = rewards[t] + gamma * next_values - values[t]
        last_gae = delta + gamma * lam * mask * last_gae
        advantages[t] = last_gae

    returns = advantages + values

    # Per-batch normalization
    adv_mean = advantages.mean(axis=0, keepdims=True)
    adv_std = advantages.std(axis=0, keepdims=True)
    advantages = (advantages - adv_mean) / (adv_std + 1e-8)

    return advantages, returns


def test_gae_correctness():
    """
    Unit test: single episode, known values.
    Manual derivation at T=3, no done flags, gamma=1.0, lam=1.0:
      delta_2 = r2 + 0 - V2
      delta_1 = r1 + V2 - V1
      delta_0 = r0 + V1 - V0
      A_2 = delta_2
      A_1 = delta_1 + A_2
      A_0 = delta_0 + A_1
    """
    rewards = [1.0, 1.0, 1.0]
    values  = [0.5, 0.5, 0.5]
    dones   = [False, False, False]

    adv, ret = compute_gae(rewards, values, dones, gamma=1.0, lam=1.0)

    # With gamma=lam=1, advantages = discounted sum of deltas (unnormalized)
    # delta = [1-0.5+0.5, 1-0.5+0.5, 1+0-0.5] = [1, 1, 0.5]
    # A_raw = [2.5, 1.5, 0.5]
    # returns = A_raw + values = [3, 2, 1]
    raw_adv = np.array([2.5, 1.5, 0.5], dtype=np.float32)
    expected_adv = (raw_adv - raw_adv.mean()) / (raw_adv.std() + 1e-8)
    expected_ret = np.array([3.0, 2.0, 1.0], dtype=np.float32)

    assert np.allclose(adv.numpy(), expected_adv, atol=1e-5), f"Advantage mismatch: {adv}"
    assert np.allclose(ret.numpy(), expected_ret, atol=1e-5), f"Return mismatch: {ret}"
    print("GAE correctness test passed.")


if __name__ == "__main__":
    test_gae_correctness()
