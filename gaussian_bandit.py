"""Thompson sampling for Gaussian rewards with known noise.

Each alternative i has an unknown mean. A play of i returns one reward

    R | A = i ~ Normal(mu_i, sigma^2)

where the noise standard deviation sigma > 0 is known. The learner does not
see mu_i. Every mean starts with the same prior, Normal(0, 25).

A round draws one candidate mean from each current posterior, plays the
largest draw, observes one reward, and updates only that alternative.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

PRIOR_MEAN = 0.0
PRIOR_VARIANCE = 25.0


@dataclass(frozen=True)
class GaussianBanditResult:
    true_means: np.ndarray
    posterior_means: np.ndarray
    posterior_variances: np.ndarray
    selection_counts: np.ndarray
    rewards: np.ndarray
    total_reward: float
    recommendation: int
    sigma: float
    seed: int

    @property
    def n_rounds(self) -> int:
        return int(self.rewards.size)


def run_gaussian_thompson(
    true_means: list[float] | np.ndarray,
    sigma: float,
    n_rounds: int,
    seed: int,
    prior_mean: float = PRIOR_MEAN,
    prior_variance: float = PRIOR_VARIANCE,
) -> GaussianBanditResult:
    """Run Gaussian Thompson sampling and return the final posteriors.

    `true_means` are used only to simulate rewards. Arm selection uses draws
    from the learned Normal posteriors. `seed` makes the run reproducible.
    """
    means = np.asarray(true_means, dtype=float)
    if means.ndim != 1 or means.size < 1 or not np.all(np.isfinite(means)):
        raise ValueError("True means must be a non-empty list of finite numbers.")

    sigma = float(sigma)
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("Known σ must be positive.")

    if isinstance(n_rounds, bool) or not isinstance(n_rounds, (int, np.integer)):
        raise ValueError("Number of rounds must be a positive integer.")
    n_rounds = int(n_rounds)
    if n_rounds < 1:
        raise ValueError("Number of rounds must be a positive integer.")

    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise ValueError("Random seed must be a non-negative integer.")
    seed = int(seed)
    if seed < 0:
        raise ValueError("Random seed must be a non-negative integer.")

    if prior_variance <= 0:
        raise ValueError("Prior variance must be positive.")

    n_arms = int(means.size)
    posterior_means = np.full(n_arms, float(prior_mean), dtype=float)
    posterior_variances = np.full(n_arms, float(prior_variance), dtype=float)
    selection_counts = np.zeros(n_arms, dtype=int)
    rewards = np.empty(n_rounds, dtype=float)

    rng = np.random.default_rng(seed)
    reward_precision = 1.0 / (sigma * sigma)

    for t in range(n_rounds):
        posterior_draws = rng.normal(
            loc=posterior_means,
            scale=np.sqrt(posterior_variances),
        )
        selected = int(np.argmax(posterior_draws))
        reward = float(rng.normal(loc=means[selected], scale=sigma))

        precision = 1.0 / posterior_variances[selected]
        updated_precision = precision + reward_precision
        posterior_means[selected] = (
            precision * posterior_means[selected] + reward_precision * reward
        ) / updated_precision
        posterior_variances[selected] = 1.0 / updated_precision

        selection_counts[selected] += 1
        rewards[t] = reward

    recommendation = int(np.argmax(posterior_means))
    return GaussianBanditResult(
        true_means=means.copy(),
        posterior_means=posterior_means.copy(),
        posterior_variances=posterior_variances.copy(),
        selection_counts=selection_counts,
        rewards=rewards,
        total_reward=float(rewards.sum()),
        recommendation=recommendation,
        sigma=sigma,
        seed=seed,
    )
