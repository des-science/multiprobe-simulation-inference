import itertools
import unittest

import numpy as np

from msi.utils.grid_weighting import COSMO_PARAMS, design_weights, lookup_coordinates, match_mock_rows, volume_fractions


class GridWeightingTests(unittest.TestCase):
    def setUp(self):
        self.priors = dict(Om=(0.1, 0.5), s8=(0.4, 1.4), w0=(-1.0, -0.5),
                           ns=(0, 1), Ob=(0, 1), H0=(0, 1),
                           Om_s8_border_points=[[0.1, 0.4], [0.5, 0.4], [0.5, 1.4], [0.1, 1.4]])
        self.narrow = {p: (lo + (hi - lo) / 4, hi - (hi - lo) / 4)
                       for p, (lo, hi) in ((p, self.priors[p]) for p in COSMO_PARAMS)}
        self.unit = np.array(list(itertools.product([0.125, 0.375, 0.625, 0.875], repeat=6)))
        bounds = np.array([self.priors[p] for p in COSMO_PARAMS])
        self.wide = bounds[:, 0] + self.unit * (bounds[:, 1] - bounds[:, 0])
        self.inner = ((self.unit > 0.25) & (self.unit < 0.75)).all(axis=1)
        self.coords = np.concatenate([self.wide, np.tile(self.wide[self.inner], (64, 1))])
        self.labels = np.concatenate([np.arange(4096), np.tile(np.flatnonzero(self.inner), 64)])

    def test_volume_matches_rectangular_analytic_result(self):
        np.testing.assert_allclose(volume_fractions(self.priors, self.narrow), [1 / 8, 1 / 64])

    def test_joint_weight_restores_full_distribution(self):
        w, _ = design_weights(self.coords, self.priors, self.narrow, "joint")
        mass = np.bincount(self.labels, weights=w)
        np.testing.assert_allclose(mass, np.full(4096, mass.mean()))

    def test_conditional_preserves_theta_and_corrects_nuisances(self):
        w, _ = design_weights(self.coords, self.priors, self.narrow, "conditional")
        mass = np.bincount(self.labels, weights=w).reshape(64, 64)
        original = np.bincount(self.labels).reshape(64, 64)
        np.testing.assert_allclose(mass.sum(axis=1), original.sum(axis=1))
        np.testing.assert_allclose(mass, np.repeat(mass.mean(axis=1)[:, None], 64, axis=1))

    def test_projected_does_not_correct_nuisance_distribution(self):
        w, _ = design_weights(self.coords, self.priors, self.narrow, "projected")
        mass = np.bincount(self.labels, weights=w).reshape(64, 64)
        np.testing.assert_allclose(mass.sum(axis=1), np.full(64, 128))
        self.assertGreater(np.ptp(mass.sum(axis=0)), 1)

    def test_coordinate_lookup_is_identity_based(self):
        table = np.zeros(3, dtype=[("sobol_index", int)] + [(p, float) for p in COSMO_PARAMS])
        table["sobol_index"] = [9, 2, 18]
        for p in COSMO_PARAMS:
            table[p] = [1, 2, 3]
        np.testing.assert_array_equal(lookup_coordinates(table, [18, 9, 18])[:, 0], [3, 1, 3])
        with self.assertRaises(ValueError):
            lookup_coordinates(table, [19])

    def test_mock_matching_rejects_leakage_and_duplicates(self):
        available = np.array([[9, 78, 0], [2, 79, 1], [18, 78, 0]])
        np.testing.assert_array_equal(match_mock_rows(available, available[[2, 0]]), [2, 0])
        with self.assertRaises(ValueError):
            match_mock_rows(available, [[9, 64, 0]])
        with self.assertRaises(ValueError):
            match_mock_rows(available, available[[0, 0]])


if __name__ == "__main__":
    unittest.main()
