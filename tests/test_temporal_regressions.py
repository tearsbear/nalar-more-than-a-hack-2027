import unittest

import numpy as np

from nalar.data import Event
from evaluation import possession_metrics as pm
from nalar.inference import A, D, H, X, infer
from tests.helpers import missing, scene


class RestartRegressionTests(unittest.TestCase):
    def test_overlapping_wait_cannot_suppress_confirmed_restart(self):
        f, layout, cfg = scene(220)
        f.ball[50:120] = f.ball[50]
        f.ball[120:] = f.ball[50] + np.column_stack((np.arange(1, 101) * 0.12, np.zeros(100)))
        f.distances[120:] = np.inf
        # The short tag ends before motion confirmation, but after its onset.
        events = [Event('set pieces', 50, 160, A), Event('set pieces', 65, 123, -1)]
        expected = np.r_[np.full(50, H), np.full(70, D), np.full(100, A)]
        for ordered in (events, events[::-1]):
            with self.subTest(reversed_order=ordered != events):
                np.testing.assert_array_equal(infer(f, ordered, layout, cfg), expected)

    def test_unresolved_restart_followed_by_exit_preserves_abstention_episode(self):
        for unresolved_frames in (20, 60, 95):
            with self.subTest(unresolved_frames=unresolved_frames):
                restart, exit_start = 120, 120 + unresolved_frames
                resume = exit_start + 30
                f, layout, cfg = scene(resume + 50)
                f.ball[50:restart] = f.ball[50]
                f.ball[restart:] = f.ball[50] + np.column_stack((
                    np.arange(1, layout.n_frames - restart + 1) * 0.12,
                    np.zeros(layout.n_frames - restart)))
                f.ball[exit_start:resume, 0] = -2
                f.distances[restart:resume] = np.inf
                f.distances[resume:] = [6, 0.5]
                events = [Event('set pieces', 50, 130, -1)]
                expected = np.r_[np.full(50, H), np.full(70, D),
                                 np.full(unresolved_frames, X), np.full(30, D), np.full(50, A)]
                labels = infer(f, events, layout, cfg)
                np.testing.assert_array_equal(labels, expected)
                # Spell scoring carries D through this unresolved episode; later
                # control must not backfill across the intervening exit.
                filled = expected.copy()
                filled[filled == X] = D
                np.testing.assert_array_equal(pm.fill_abstentions(labels, layout.blocks), filled)

    def test_old_motion_cannot_restart_after_ball_stops(self):
        f, layout, cfg = scene(220)
        f.ball[50:80, 0] = -2
        f.distances[80:160] = np.inf
        f.ball[90:] = f.ball[89]
        expected = np.r_[np.full(50, H), np.full(170, D)]
        np.testing.assert_array_equal(infer(f, [], layout, cfg), expected)

    def test_new_motion_after_stationary_wait_defines_restart(self):
        f, layout, cfg = scene(300)
        f.ball[50:80, 0] = -2
        f.distances[80:200] = np.inf
        f.ball[90:200] = f.ball[89]
        f.ball[200:] = f.ball[89] + np.column_stack((np.arange(1, 101) * 0.12, np.zeros(100)))
        expected = np.r_[np.full(50, H), np.full(150, D), np.full(100, H)]
        np.testing.assert_array_equal(infer(f, [], layout, cfg), expected)

    def test_tracking_gap_invalidates_old_restart_motion(self):
        f, layout, cfg = scene(220)
        f.ball[50:80, 0] = -2
        f.distances[80:160] = np.inf
        missing(f, 90, 160)
        expected = np.r_[np.full(50, H), np.full(111, D), np.full(59, H)]
        np.testing.assert_array_equal(infer(f, [], layout, cfg), expected)

    def test_control_before_motion_cannot_confirm_a_restart(self):
        f, layout, cfg = scene(220)
        f.ball[50:80, 0] = -2
        f.ball[80:130] = f.ball[80]
        f.ball[130:135] = f.ball[80] + np.column_stack((np.arange(1, 6) * 0.12, np.zeros(5)))
        f.ball[135:] = f.ball[134]
        f.distances[130:] = np.inf
        expected = np.r_[np.full(50, H), np.full(170, D)]
        np.testing.assert_array_equal(infer(f, [], layout, cfg), expected)

    def test_later_stationarity_cannot_reopen_completed_tagged_restart(self):
        for team in (H, A, -1):
            with self.subTest(restart_team=team):
                f, layout, cfg = scene(260)
                f.ball[50:80, 0] = -2
                f.ball[150:210] = f.ball[149]
                f.ball[210:] = f.ball[149] + np.column_stack((np.arange(1, 51) * 0.12, np.zeros(50)))
                events = [Event('set pieces', 50, 250, team)]
                expected = np.r_[np.full(50, H), np.full(30, D), np.full(180, H)]
                np.testing.assert_array_equal(infer(f, events, layout, cfg), expected)

    def test_motion_already_in_progress_consumes_broad_restart_context(self):
        f, layout, cfg = scene(260)
        f.ball[150:210] = f.ball[149]
        events = [Event('set pieces', 50, 250, A)]
        np.testing.assert_array_equal(infer(f, events, layout, cfg), np.full(260, H))

    def test_conflicting_restart_teams_still_require_geometric_control(self):
        f, layout, cfg = scene(220)
        f.ball[50:120] = f.ball[50]
        f.ball[120:] = f.ball[50] + np.column_stack((np.arange(1, 101) * 0.12, np.zeros(100)))
        f.distances[120:] = [6, 0.5]
        events = [Event('set pieces', 50, 155, H), Event('set pieces', 50, 155, A)]
        expected = np.r_[np.full(50, H), np.full(70, D), np.full(100, A)]
        for ordered in (events, events[::-1]):
            np.testing.assert_array_equal(infer(f, ordered, layout, cfg), expected)

    def test_unconfirmed_brief_motion_does_not_consume_restart_context(self):
        f, layout, cfg = scene(180)
        f.ball[24:100] = f.ball[23]
        f.ball[100:] = f.ball[23] + np.column_stack((np.arange(1, 81) * 0.12, np.zeros(80)))
        events = [Event('set pieces', 20, 150, H)]
        expected = np.r_[np.full(20, H), np.full(80, D), np.full(80, H)]
        np.testing.assert_array_equal(infer(f, events, layout, cfg), expected)

    def test_separate_later_restart_keeps_its_own_waiting_period(self):
        f, layout, cfg = scene(280)
        f.ball[50:80, 0] = -2
        f.ball[150:220] = f.ball[150]
        f.ball[220:] = f.ball[150] + np.column_stack((np.arange(1, 61) * 0.12, np.zeros(60)))
        events = [Event('set pieces', 50, 140, H), Event('set pieces', 150, 260, H)]
        expected = np.r_[np.full(50, H), np.full(30, D), np.full(70, H),
                         np.full(70, D), np.full(60, H)]
        np.testing.assert_array_equal(infer(f, events, layout, cfg), expected)


class TakeoverRegressionTests(unittest.TestCase):
    def test_takeover_boundary_does_not_reach_back_to_isolated_touch(self):
        for incumbent in (H, A):
            with self.subTest(incumbent=incumbent):
                challenger = 1 - incumbent
                f, layout, cfg = scene(team=incumbent)
                f.distances[25:50] = [1, 1]
                f.distances[50] = f.distances[53:] = 6
                f.distances[50, challenger] = f.distances[53:, challenger] = 0.5
                expected = np.r_[np.full(53, incumbent), np.full(127, challenger)]
                np.testing.assert_array_equal(infer(f, [], layout, cfg), expected)

    def test_expiring_votes_cannot_backdate_over_latest_incumbent_evidence(self):
        f, layout, cfg = scene(79)
        f.distances[25:] = np.inf
        f.distances[50:53] = f.distances[73] = [0.5, 6]
        f.distances[54:64] = [6, 0.5]
        np.testing.assert_array_equal(infer(f, [], layout, cfg), np.full(79, H))
