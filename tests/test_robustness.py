from dataclasses import replace
import unittest

import numpy as np

from nalar.data import Event, Features
from nalar.__main__ import diagnose
from nalar.inference import A, H, infer
from tests.helpers import scene


class RobustnessTests(unittest.TestCase):
    def test_diagnostics_report_actual_perturbed_coverage(self):
        f, layout, cfg = scene()
        report = diagnose(f, [], layout, cfg)
        self.assertLess(report['variants']['ten_percent_ball_dropout']['diagnostics']['observed_ball_frames'],
                        report['base']['observed_ball_frames'])

    def test_dropout_and_small_distance_noise_preserve_clear_takeover(self):
        f, layout, cfg = scene(300)
        f.distances[150:] = [6, 0.5]
        rng = np.random.default_rng(2027)
        observed = rng.random(300) >= 0.1
        observed[150] = True
        distances = np.maximum(0, f.distances + rng.normal(0, 0.1, (300, 2)))
        ball = f.ball.copy()
        ball[~observed], distances[~observed] = np.nan, np.inf
        labels = infer(Features(ball, distances, observed), [], layout, cfg)
        np.testing.assert_array_equal(labels, np.r_[np.full(150, H), np.full(150, A)])

    def test_irrelevant_component_removals_preserve_clear_control(self):
        f, layout, cfg = scene()
        events = [Event('build up', 0, 180, H)]
        base = infer(f, events, layout, cfg)
        for kwargs, config in [({'use_tags': False}, cfg), ({'use_dead_ball': False}, cfg),
                               ({}, replace(cfg, interpolation_frames=0))]:
            np.testing.assert_array_equal(infer(f, events, layout, config, **kwargs), base)
