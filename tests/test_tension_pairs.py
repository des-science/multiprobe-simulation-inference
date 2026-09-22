import unittest

from msi.utils.tensions import build_combinations


class TensionPairTests(unittest.TestCase):
    def setUp(self):
        self.config = {
            'runs': {name: {'lensing': {'pred_dir': f'/runs/{name}/lensing/v1'}}
                     for name in ('maps', 'cls', 'transformer')},
            'comparisons': {'data': True},
        }

    def test_null_and_empty_allowlists_differ(self):
        self.config['comparisons']['data_pairs'] = None
        self.assertEqual(len(build_combinations(self.config)), 3)
        self.config['comparisons']['data_pairs'] = []
        self.assertEqual(build_combinations(self.config), [])

    def test_disabled_representation_is_allowed(self):
        self.config['comparisons']['data_pairs'] = [['maps', 'cls'], ['transformer', 'cls']]
        self.config['runs']['transformer'] = None
        pairs = build_combinations(self.config)
        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0][0], 'maps_lensing_vs_cls_lensing')

    def test_invalid_pairs_fail(self):
        for pairs in ([['typo', 'cls']], [['maps']], [['maps', 'maps']], ['maps'], 'maps'):
            with self.subTest(pairs=pairs):
                self.config['comparisons']['data_pairs'] = pairs
                with self.assertRaises(ValueError):
                    build_combinations(self.config)

    def test_production_config_selects_21_pairs(self):
        from pathlib import Path
        import yaml
        path = Path(__file__).resolve().parents[1] / 'configs/runs/v18/default/prod.yaml'
        config = yaml.safe_load(path.read_text())
        self.assertEqual(len(build_combinations(config)), 21)
        config['runs']['maps_convnext'] = None
        self.assertEqual(len(build_combinations(config)), 15)


if __name__ == '__main__':
    unittest.main()
