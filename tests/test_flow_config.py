import importlib.util
import unittest


@unittest.skipUnless(importlib.util.find_spec('torch') and importlib.util.find_spec('enflows'),
                     'Requires the existing PyTorch environment')
class FlowConfigTests(unittest.TestCase):
    def test_removed_options_fail_instead_of_being_ignored(self):
        from msi.utils.flow import _extract_train_kwargs
        for settings in ({'theta_jitter': .1}, {'group_design': 'bootstrap'},
                         {'group_bootstrap': True}, {'group_by': 'unknown'},
                         {'train_prior': 'reweight'}):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                _extract_train_kwargs({'training': settings})

    def test_retained_round_configs_have_unchanged_training_defaults(self):
        from msi.utils.flow import _extract_train_kwargs
        for mode in ['all', 'wide', 'reweight_projected', 'reweight_joint', 'reweight_conditional']:
            result = _extract_train_kwargs({'training': {'train_prior': mode, 'group_by': 'signal'}})
            self.assertEqual(result['batch_size'], 10000)
            self.assertEqual(result['scheduler_type'], 'cosine')

    def test_cosmology_split_holds_out_only_wide_cosmologies(self):
        import numpy as np
        from unittest.mock import patch
        from msi.utils.flow import resolve_group_ids
        from msi.flow_conductor.likelihood_flow import group_split_indices
        sobol = np.repeat([10, 20, 30, 40, 50, 60, 70, 80], 3)
        signals = np.tile([0, 1, 2], 8)
        config = {"training": {"group_by": "cosmology", "vali_split": .25}}
        with patch("msi.utils.coverage.wide_prior_sobol_indices", return_value=[10, 20, 30, 40]):
            groups = resolve_group_ids(config, signals, i_sobol=sobol, msfm_conf={})
        train, validation, _, _ = group_split_indices(groups, .25)
        np.testing.assert_array_equal(np.unique(sobol[validation]), [10, 40])
        self.assertEqual(set(sobol[train]) & set(sobol[validation]), set())
        np.testing.assert_array_equal(resolve_group_ids({}, signals), signals)


if __name__ == '__main__':
    unittest.main()
