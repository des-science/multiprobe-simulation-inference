import numpy as np
from scipy.optimize import linprog
from sklearn.metrics.pairwise import euclidean_distances
import matplotlib.pyplot as plt


def in_convex_hull(point, sample_points, tolerance=1e-12):
    """
    Efficiently checks if a point is in the convex hull of sample_points in high dimensions.

    Parameters:
    - point: (D,) array, the query point.
    - sample_points: (N, D) array, the set of points defining the hull.

    Returns:
    - bool: True if the point is inside the hull.
    """
    n_points, n_dim = sample_points.shape

    # --- PHASE 1: Bounding Box Check (O(N) - Very Fast) ---
    # If the point is outside the min/max of any dimension, it's definitely outside.
    min_bounds = np.min(sample_points, axis=0)
    max_bounds = np.max(sample_points, axis=0)

    if np.any(point < min_bounds - tolerance) or np.any(point > max_bounds + tolerance):
        return False

    # --- PHASE 2: Linear Programming (O(N) - Rigorous) ---
    # We try to solve: sum(alpha_i * sample_i) = point
    # Subject to:      sum(alpha_i) = 1
    #                  alpha_i >= 0

    c = np.zeros(n_points)  # Dummy objective (we just check feasibility)

    # Constraint 1: Weighted coordinates match the point
    A_eq = np.vstack([sample_points.T, np.ones((1, n_points))])
    b_eq = np.concatenate([point, [1]])

    # Solve using 'highs' method (fastest/most robust in modern scipy)
    result = linprog(c, A_eq=A_eq, b_eq=b_eq, method="highs")

    return result.success


class PriorPredictiveMMD:
    def __init__(self, s_sim, s_obs, subsample_size=2000):
        """
        Args:
            s_sim (np.array): Shape (S, N). The full grid of training simulations drawn from the prior.
            s_obs (np.array): Shape (1, N). The single observation summary.
            subsample_size (int): Number of simulations to use for the reference background.
                                  MMD is O(N^2), so using the full grid (e.g., 100k) is too slow.
                                  2000-5000 is usually sufficient for a robust test.
        """
        self.s_obs = s_obs

        # Subsample the simulation grid for the test if it's too large
        if s_sim.shape[0] > subsample_size:
            indices = np.random.choice(s_sim.shape[0], subsample_size, replace=False)
            self.S_ref = s_sim[indices]
            print(f"Subsampled simulation grid from {s_sim.shape[0]} to {subsample_size} for efficiency.")
        else:
            self.S_ref = s_sim

        # --- Bandwidth Heuristic ---
        # We calculate the median distance between simulations in the prior grid.
        # This sets the "length scale" of the test. If s_obs is further than this
        # typical distance, MMD will detect it.
        dists = euclidean_distances(self.S_ref, self.S_ref)
        upper_tri = dists[np.triu_indices_from(dists, k=1)]
        self.sigma = np.median(upper_tri) if len(upper_tri) > 0 else 1.0

        print(f"Kernel Bandwidth (Sigma) set to: {self.sigma:.4f}")

    def rbf_kernel(self, X, Y):
        """Gaussian RBF Kernel: K(x,y) = exp(-||x-y||^2 / 2*sigma^2)"""
        X_norm = np.sum(X**2, axis=1).reshape(-1, 1)
        Y_norm = np.sum(Y**2, axis=1).reshape(1, -1)
        D_sq = X_norm + Y_norm - 2 * np.dot(X, Y.T)
        D_sq = np.maximum(D_sq, 0)
        gamma = 1.0 / (2 * self.sigma**2)
        return np.exp(-gamma * D_sq)

    def compute_mmd_sq(self, X_ref, y_obs):
        """
        Calculates MMD squared between a reference set X_ref and a single observation y_obs.
        """
        m = X_ref.shape[0]

        # 1. Similarity within the simulation grid (E[k(x, x')])
        K_xx = self.rbf_kernel(X_ref, X_ref)
        term_1 = (np.sum(K_xx) - np.trace(K_xx)) / (m * (m - 1))

        # 2. Similarity between grid and observation (E[k(x, y)])
        K_xy = self.rbf_kernel(X_ref, y_obs)
        term_2 = np.mean(K_xy)  # Since y_obs is single, we just average the vector

        # 3. Similarity of observation to itself (k(y,y) = 1 for RBF)
        term_3 = 1.0

        return term_1 - 2 * term_2 + term_3

    def run_check(self, n_permutations=1000, plot=True):
        """
        Performs the hypothesis test.
        Null Hypothesis H0: s_obs is drawn from the marginal distribution of s_sim.

        Args:
            n_permutations (int): Number of permutations for the null distribution.
            plot (bool): If True, creates a plot comparing the null distribution to the observation.
        """
        # 1. Compute MMD for the actual observation
        t_obs = self.compute_mmd_sq(self.S_ref, self.s_obs)

        # 2. Generate Null Distribution
        # We treat individual simulations from S_ref as if they were the observation
        # and compare them to the rest of S_ref.
        null_stats = []
        indices = np.arange(self.S_ref.shape[0])

        for _ in range(n_permutations):
            # Select a random simulation to act as the "fake observation"
            idx_fake = np.random.choice(indices)
            fake_obs = self.S_ref[idx_fake].reshape(1, -1)

            # The rest of the grid acts as the reference
            # (masking ensuring we don't compare the sample to itself in the ref set)
            mask = np.ones(len(indices), dtype=bool)
            mask[idx_fake] = False
            remaining_ref = self.S_ref[mask]

            t_null = self.compute_mmd_sq(remaining_ref, fake_obs)
            null_stats.append(t_null)

        null_stats = np.array(null_stats)

        # 3. Compute P-Value
        p_value = np.mean(null_stats >= t_obs)

        # 4. Print results
        print(f"\n{'='*60}")
        print(f"Prior Predictive Check - MMD Test Results")
        print(f"{'='*60}")
        print(f"Observed MMD²: {t_obs:.6f}")
        print(f"Null Mean MMD²: {np.mean(null_stats):.6f}")
        print(f"Null Std MMD²: {np.std(null_stats):.6f}")
        print(f"P-value: {p_value:.4f}")

        if p_value < 0.001:
            print(f"Result: REJECT H0 (p < 0.001) - Observation is NOT consistent with prior")
        elif p_value < 0.05:
            print(f"Result: REJECT H0 (p < 0.05) - Observation is NOT consistent with prior")
        else:
            print(f"Result: FAIL TO REJECT H0 - Observation is consistent with prior")
        print(f"{'='*60}\n")

        # 5. Create visualization
        if plot:
            fig, ax = plt.subplots(figsize=(10, 6))

            # Plot null distribution histogram
            ax.hist(
                null_stats,
                bins=50,
                density=True,
                alpha=0.7,
                color="skyblue",
                edgecolor="black",
                label="Null Distribution",
            )

            # Add vertical line for observed value
            ax.axvline(t_obs, color="red", linestyle="--", linewidth=2, label=f"Observed MMD² = {t_obs:.6f}")

            # Add vertical line for mean of null distribution
            ax.axvline(
                np.mean(null_stats),
                color="green",
                linestyle=":",
                linewidth=2,
                label=f"Null Mean = {np.mean(null_stats):.6f}",
            )

            # Labels and title
            ax.set_xlabel("MMD² Statistic", fontsize=12)
            ax.set_ylabel("Density", fontsize=12)
            ax.set_title(f"Prior Predictive Check\nP-value = {p_value:.4f}", fontsize=14, fontweight="bold")
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.show()

        return t_obs, p_value, null_stats
