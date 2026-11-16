from dataclasses import replace
import unittest

import numpy as np

from nalar.config import Layout, official_layout
from nalar.data import Event, Features
from nalar.inference import A, D, H, X, infer, interpolate_ball, phase_votes, stationary_windows
from tests.helpers import missing, scene


class ControlTests(unittest.TestCase):
    def test_initial_control_backfills_episode(self):
        f, layout, cfg = scene()
        self.assertTrue(np.all(infer(f, [], layout, cfg) == H))

    def test_controlled_takeover_backdated(self):
        f, layout, cfg = scene()
        f.distances[60:] = [6, 0.5]
        labels = infer(f, [], layout, cfg)
        self.assertTrue(np.all(labels[:60] == H))
        self.assertTrue(np.all(labels[60:] == A))

    def test_brief_deflection_is_not_takeover(self):
        f, layout, cfg = scene()
        f.distances[60:64] = [6, 0.5]
        self.assertTrue(np.all(infer(f, [], layout, cfg) == H))

    def test_contested_ball_preserves_owner(self):
        f, layout, cfg = scene()
        f.distances[50:110] = [1, 1]
        self.assertTrue(np.all(infer(f, [], layout, cfg) == H))

    def test_missing_ball_alone_never_dead(self):
        f, layout, cfg = scene()
        missing(f, 50, 160)
        self.assertTrue(np.all(infer(f, [], layout, cfg) == H))

    def test_no_information_remains_X(self):
        f, layout, cfg = scene()
        missing(f, 0, layout.n_frames)
        self.assertTrue(np.all(infer(f, [], layout, cfg) == X))

    def test_initial_distance_tie_remains_X(self):
        f, layout, cfg = scene()
        f.distances[:] = 1
        self.assertTrue(np.all(infer(f, [], layout, cfg) == X))

    def test_geometry_overrides_contradictory_tag(self):
        f, layout, cfg = scene()
        events = [Event('build up', 0, layout.n_frames, A)]
        self.assertTrue(np.all(infer(f, events, layout, cfg) == H))

    def test_sustained_missing_ball_tag_backdates(self):
        f, layout, cfg = scene()
        missing(f, 50, 180)
        events = [Event('progression', 70, 180, A)]
        labels = infer(f, events, layout, cfg)
        self.assertTrue(np.all(labels[:70] == H))
        self.assertTrue(np.all(labels[70:] == A))

    def test_tag_confirmation_requires_consecutive_missing_frames(self):
        f, layout, cfg = scene()
        missing(f, 50, 99)
        events = [Event('progression', 50, 99, A)]
        self.assertTrue(np.all(infer(f, events, layout, cfg) == H))

    def test_unknown_restart_team_does_not_become_away(self):
        f, layout, cfg = scene()
        missing(f, 0, 180)
        self.assertTrue(np.all(infer(f, [Event('set pieces', 0, 100, -1)], layout, cfg) == X))

    def test_team_symmetry(self):
        f, layout, cfg = scene()
        f.distances[60:] = [6, 0.5]
        events = [Event('progression', 10, 120, H)]
        a = infer(f, events, layout, cfg)
        swapped = Features(f.ball, f.distances[:, ::-1], f.observed)
        b = infer(swapped, [replace(e, team=1 - e.team) for e in events], layout, cfg)
        expected = a.copy()
        expected[a == H], expected[a == A] = A, H
        np.testing.assert_array_equal(b, expected)

    def test_equivalent_unit_scaling(self):
        f, layout, cfg = scene()
        factor = 2
        scaled = Features(f.ball * factor, f.distances * factor, f.observed)
        scaled_layout = replace(layout, pitch=tuple(x * factor for x in layout.pitch))
        scaled_cfg = replace(cfg, control_radius_m=cfg.control_radius_m * factor,
                             control_margin_m=cfg.control_margin_m * factor,
                             exit_margin_m=cfg.exit_margin_m * factor,
                             stationary_span_m=cfg.stationary_span_m * factor,
                             restart_speed_mps=cfg.restart_speed_mps * factor)
        np.testing.assert_array_equal(infer(f, [], layout, cfg), infer(scaled, [], scaled_layout, scaled_cfg))


class TagTests(unittest.TestCase):
    def test_duplicate_votes_cap_and_opponents_cancel(self):
        events = [Event('build up', 0, 100, H), Event('progression', 0, 100, H),
                  Event('creating chances', 30, 60, A)]
        votes = phase_votes(events, 100)
        self.assertTrue(np.all(votes[:30] == H))
        self.assertTrue(np.all(votes[30:60] == -1))
        self.assertTrue(np.all(votes[60:] == H))

    def test_defensive_unknown_and_player_count_tags_do_not_vote(self):
        events = [Event(code, 0, 50, team) for code, team in
                  [('defensive transition', H), ('players in the box', A), ('build up', -1)]]
        self.assertTrue(np.all(phase_votes(events, 50) == -1))


class DeadBallTests(unittest.TestCase):
    def test_short_exit_noise_does_not_stop_play(self):
        f, layout, cfg = scene()
        f.ball[50:59, 0] = -2
        self.assertFalse(np.any(infer(f, [], layout, cfg) == D))

    def test_exit_backdates_and_controlled_motion_restarts(self):
        f, layout, cfg = scene()
        f.ball[50:80, 0] = -2
        labels = infer(f, [], layout, cfg)
        self.assertTrue(np.all(labels[:50] == H))
        self.assertTrue(np.all(labels[50:80] == D))
        self.assertTrue(np.all(labels[80:] == H))

    def test_stationary_live_ball_without_restart_not_dead(self):
        f, layout, cfg = scene()
        f.ball[:] = [20, 30]
        self.assertTrue(np.all(infer(f, [], layout, cfg) == H))

    def test_stationary_restart_stops_then_seeds_team(self):
        f, layout, cfg = scene(220)
        f.ball[50:120] = f.ball[50]
        f.ball[120:] = f.ball[50] + np.column_stack((np.arange(1, 101) * 0.12, np.zeros(100)))
        f.distances[120:] = [6, 0.5]
        events = [Event('set pieces', 50, 155, A)]
        labels = infer(f, events, layout, cfg)
        self.assertTrue(np.all(labels[:50] == H))
        self.assertTrue(np.all(labels[50:120] == D))
        self.assertTrue(np.all(labels[120:] == A))

    def test_unknown_restart_backfills_live_prefix_after_D(self):
        f, layout, cfg = scene(220)
        f.ball[50:120] = f.ball[50]
        f.ball[120:] = f.ball[50] + np.column_stack((np.arange(1, 101) * 0.12, np.zeros(100)))
        f.distances[120:] = [6, 0.5]
        events = [Event('set pieces', 50, 155, -1)]
        labels = infer(f, events, layout, cfg)
        self.assertTrue(np.all(labels[50:120] == D))
        self.assertTrue(np.all(labels[120:] == A))

    def test_moving_tag_interval_does_not_create_dead_ball(self):
        f, layout, cfg = scene()
        events = [Event('set pieces', 50, 150, A)]
        self.assertTrue(np.all(infer(f, events, layout, cfg) == H))

    def test_stationary_window_requires_real_observations(self):
        ball = np.full((80, 2), 10.0)
        observed = np.ones(80, dtype=bool)
        observed[40] = False
        self.assertFalse(stationary_windows(ball, observed, 50, 0.75).any())

    def test_published_kickoff_ends_wait_despite_stationary_tag_context(self):
        f, layout, cfg = scene()
        layout = replace(layout, kickoffs=(60,))
        f.ball[:80] = f.ball[0]
        labels = infer(f, [Event('set pieces', 20, 110, H)], layout, cfg)
        self.assertTrue(np.all(labels[:60] == D))
        self.assertTrue(np.all(labels[60:80] == H))


class BoundaryTests(unittest.TestCase):
    def test_official_counts_and_anchors(self):
        layout = official_layout()
        self.assertEqual(layout.kickoffs, (11185, 113742))
        self.assertEqual(layout.counts(), {'video_frames': 199319, 'graded_frames': 158857,
                         'ungraded_frames': 40462, 'pre_kickoff_graded_frames': 2934,
                         'kickoff_to_half_end_frames': 155923})

    def test_half_reset_and_waiting_mask(self):
        f, _, cfg = scene(180)
        layout = Layout(180, 25, (105.3, 68), ((10, 70), (100, 170)), (10, 120))
        missing(f, 120, 180)
        labels = infer(f, [], layout, cfg)
        self.assertTrue(np.all(labels[10:70] == H))
        self.assertTrue(np.all(labels[100:120] == D))
        self.assertTrue(np.all(labels[120:170] == X))
        self.assertTrue(np.all(labels[np.r_[0:10, 70:100, 170:180]] == X))

    def test_short_interpolation_never_establishes_control(self):
        f, layout, cfg = scene()
        f.distances[:] = np.inf
        missing(f, 40, 45)
        ball = interpolate_ball(f, layout, cfg.interpolation_frames)
        self.assertTrue(np.isfinite(ball[40:45]).all())
        self.assertTrue(np.all(infer(f, [], layout, cfg) == X))

    def test_long_gap_and_half_gap_not_interpolated(self):
        f, layout, cfg = scene()
        missing(f, 40, 46)
        self.assertTrue(np.isnan(interpolate_ball(f, layout, cfg.interpolation_frames)[40:46]).all())
        f, _, cfg = scene()
        missing(f, 48, 52)
        layout = replace(layout, blocks=((0, 50), (50, 180)), kickoffs=(0, 50))
        self.assertTrue(np.isnan(interpolate_ball(f, layout, cfg.interpolation_frames)[48:52]).all())

    def test_interpolation_does_not_bridge_inferred_dead_ball(self):
        f, layout, cfg = scene()
        missing(f, 40, 45)
        blocked = np.zeros(180, dtype=bool)
        blocked[39] = True
        self.assertTrue(np.isnan(interpolate_ball(f, layout, 5, blocked=blocked)[40:45]).all())
