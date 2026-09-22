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

    def test_extend_params_resolution(self):
        from msi.utils.extended_params import DEFAULT_EXTEND_PARAMS
        from msi.utils.flow import resolve_extend_params
        prod = {'extend_params': ['ns', 'Ob', 'H0']}
        # the config is the production source; no CLI flag means no 'ext' label
        self.assertEqual(resolve_extend_params(None, [prod]), (['ns', 'Ob', 'H0'], False))
        # a config without the key, or with it emptied, trains the unextended flow
        self.assertEqual(resolve_extend_params(None, [{}]), ([], False))
        self.assertEqual(resolve_extend_params(None, [{'extend_params': []}]), ([], False))
        # the CLI wins, and is flagged so run_inference protects the baseline checkpoint
        self.assertEqual(resolve_extend_params([], [prod]), (list(DEFAULT_EXTEND_PARAMS), True))
        self.assertEqual(resolve_extend_params(['ns'], [prod]), (['ns'], True))
        # a heterogeneous ensemble cannot have members with different conditioning vectors
        with self.assertRaises(ValueError):
            resolve_extend_params(None, [prod, {'extend_params': ['ns']}])

    def test_production_flow_config_carries_the_validated_setup(self):
        import pathlib
        import yaml
        root = pathlib.Path(__file__).resolve().parents[1]
        conf = yaml.safe_load((root / 'configs/flow/maf.yaml').read_text())
        self.assertEqual(conf['extend_params'], ['ns', 'Ob', 'H0'])
        # plot_eecp_check slices 100 confidence levels out of n_samples // n_flows; 8 members need
        # >= 800 or the likelihood HPD plot dies on a zero-length slice step
        self.assertGreaterEqual(conf['diagnostics']['n_likelihood_samples'], 800)
        self.assertIsInstance(conf['training']['learning_rate'], float)

    def test_retrain_guard_protects_a_differently_conditioned_run_dir(self):
        import tempfile
        from pathlib import Path
        import yaml
        from msi.utils.flow import refuse_incompatible_retrain

        class FakeFlow:
            def __init__(self, model_dir):
                self.model_dir = model_dir

        with tempfile.TemporaryDirectory() as tmp:
            flow = FakeFlow(tmp)
            # no saved config yet -> a fresh directory, never a conflict
            refuse_incompatible_retrain(flow, {'extend_params': ['ns']})
            # the v18 production dirs have no extend_params key at all
            Path(tmp, 'flow_config.yaml').write_text(yaml.safe_dump({'seed': 7}))
            with self.assertRaisesRegex(ValueError, 'overwrite it in place'):
                refuse_incompatible_retrain(flow, {'extend_params': ['ns', 'Ob', 'H0']})
            refuse_incompatible_retrain(flow, {})
            Path(tmp, 'flow_config.yaml').write_text(yaml.safe_dump({'extend_params': ['ns', 'Ob', 'H0']}))
            refuse_incompatible_retrain(flow, {'extend_params': ['ns', 'Ob', 'H0']})
            with self.assertRaises(ValueError):
                refuse_incompatible_retrain(flow, {})

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
