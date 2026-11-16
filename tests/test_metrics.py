import unittest

import numpy as np

from evaluation import possession_metrics as pm


def codes(labels):
    return np.array([pm.LABELS.index(c) for c in labels], dtype=np.int8)


class OfficialMetricTests(unittest.TestCase):
    def test_internal_leading_and_dead_carry(self):
        for raw, expected in [('HXXH', 'HHHH'), ('XXHH', 'HHHH'), ('DXXH', 'DDDH'), ('XXXX', 'XXXX')]:
            with self.subTest(raw=raw):
                np.testing.assert_array_equal(pm.fill_abstentions(codes(raw), [(0, 4)]), codes(expected))

    def test_ground_truth_runs_define_fill_boundaries(self):
        gt = codes('HHXXAA')
        pred = codes('HXXXAX')
        np.testing.assert_array_equal(pm.fill_abstentions(pred, pm.valid_runs(gt)), gt)

    def test_abstention_inside_possession_preserves_spell_but_costs_macro(self):
        gt = codes('HHHHDDHHAA')
        result = pm.evaluate(gt, codes('HXXHDDHHAA'))
        self.assertEqual(result['spell_f1_mean'], 1)
        self.assertLess(result['macro_f1'], 1)

    def test_abstention_after_dead_is_not_live_episode_reset(self):
        np.testing.assert_array_equal(pm.fill_abstentions(codes('DDXXHH'), [(0, 6)]), codes('DDDDHH'))

    def test_all_abstention_has_no_possessions(self):
        result = pm.evaluate(codes('HHHDDDAAA'), codes('XXXXXXXXX'))
        self.assertEqual(result['spell_f1_mean'], 0)
        self.assertEqual(result['macro_f1'], 0)
        self.assertEqual(result['n_pred_spells'], 0)

    def test_dead_ball_splits_same_team_and_halves_do_not_merge(self):
        self.assertEqual(len(pm.possession_spells(codes('HDH'), [(0, 3)])), 2)
        self.assertEqual(len(pm.possession_spells(codes('HH'), [(0, 1), (1, 2)])), 2)
