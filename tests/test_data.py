from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

import numpy as np

from nalar.config import Config, Layout
from nalar.data import DataError, read_events, read_metadata, read_tracking


META = '<root><Team id="h"><Name>Home</Name></Team><Team id="a"><Name>Away</Name></Team><Player id="1" teamId="h"/><Player id="2" teamId="a"/></root>'


class DataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)
        self.layout = Layout(3, 25, (100, 50), ((0, 3),), (0,))
        self.slots = np.array([0, 1], dtype=np.int8)

    def write(self, name, content):
        p = self.path / name
        p.write_text(content, encoding='utf-8')
        return p

    def raw(self, rows):
        return self.write('synthetic.txt', '\n'.join(rows) + '\n')

    def test_metadata_and_unknown_team(self):
        slots, names = read_metadata(self.write('meta.xml', META))
        self.assertEqual(slots.tolist(), [0, 1])
        self.assertEqual(names, ('Home', 'Away'))
        with self.assertRaises(DataError):
            read_metadata(self.write('bad.xml', META.replace('teamId="a"', 'teamId="unknown"')))

    def test_duplicate_track_and_team_ids(self):
        for text in [META.replace('Player id="2"', 'Player id="1"'), META.replace('Team id="a"', 'Team id="h"')]:
            with self.subTest(text=text), self.assertRaises(DataError):
                read_metadata(self.write('bad.xml', text))

    def test_normalized_coordinates_and_slot_permutation(self):
        rows = [f'{i}:0.1,0.2,1;0.8,0.2,1:0.11,0.2,1' for i in range(1, 4)]
        f = read_tracking(self.raw(rows), self.slots, self.layout)
        np.testing.assert_allclose(f.ball[0], [11, 10])
        np.testing.assert_allclose(f.distances[0], [1, 69])
        reverse = [f'{i}:0.8,0.2,1;0.1,0.2,1:0.11,0.2,1' for i in range(1, 4)]
        other = read_tracking(self.raw(reverse), self.slots[::-1], self.layout)
        np.testing.assert_array_equal(f.distances, other.distances)

    def test_missing_coordinates_and_outside_ball_preserved(self):
        rows = ['1:NaN,NaN,NaN;NaN,NaN,NaN:NaN,NaN,NaN',
                '2:0.1,0.2,1;0.8,0.2,1:-0.1,0.2,1',
                '3:0.1,0.2,1;NaN,NaN,NaN:0.1,0.2,1']
        f = read_tracking(self.raw(rows), self.slots, self.layout)
        self.assertFalse(f.observed[0])
        self.assertEqual(f.ball[1, 0], -10)
        self.assertEqual(f.distances[2, 1], np.inf)

    def test_malformed_records_fail_without_echoing_content(self):
        normal = [f'{i}:0.1,0.2,1;0.8,0.2,1:0.11,0.2,1' for i in range(1, 4)]
        cases = [normal[:2], normal + [normal[-1].replace('3:', '4:')],
                 [normal[0], normal[0], normal[2]], [normal[0], normal[2], normal[1]],
                 [normal[0].replace('1:', '0:'), *normal[1:]],
                 [normal[0].replace('0.1,0.2,1;', ''), *normal[1:]],
                 [normal[0].replace('0.11,0.2,1', 'NaN,0.2,1'), *normal[1:]],
                 [normal[0].replace('0.1,0.2,1', '0.1,NaN,1'), *normal[1:]],
                 [normal[0].replace('0.11,0.2,1', 'inf,0.2,1'), *normal[1:]],
                 ['1:RESTRICTED_SENTINEL', *normal[1:]], [' ', *normal[1:]]]
        for rows in cases:
            with self.subTest(case=cases.index(rows)), self.assertRaises(DataError) as caught:
                read_tracking(self.raw(rows), self.slots, self.layout)
            self.assertNotIn('RESTRICTED_SENTINEL', str(caught.exception))

    def test_short_broadcast_like_count_rejected(self):
        with self.assertRaises(DataError):
            read_tracking(self.raw(['1:0,0,0;1,1,0:0,0,0']), self.slots, self.layout)

    def test_finite_coordinates_that_overflow_fail_without_echoing_data(self):
        for player, ball in [('0.1', '1e308'), ('1e308', '0.1'), ('1e200', '0.1')]:
            with self.subTest(player=player, ball=ball):
                rows = [f'{i}:{player},0.2,1;0.8,0.2,1:{ball},0.2,1' for i in range(1, 4)]
                with self.assertRaises(DataError) as caught:
                    read_tracking(self.raw(rows), self.slots, self.layout)
                self.assertNotIn('1e', str(caught.exception))

    def event(self, team='Home', code=' BUILD UP ', start='0', stop='0.12'):
        return f'<root><instance><start>{start}</start><end>{stop}</end><code>{code}</code><label><group> Team </group><text>{team}</text></label></instance></root>'

    def test_tag_casing_rounding_unknown_and_NA(self):
        events = read_events(self.write('events.xml', self.event('hOmE')), ('Home', 'Away'), self.layout)
        self.assertEqual((events[0].team, events[0].code, events[0].start, events[0].stop), (0, 'build up', 0, 3))
        self.assertEqual(read_events(self.write('na.xml', self.event('N/A')), ('Home', 'Away'), self.layout)[0].team, -1)
        with self.assertRaises(DataError):
            read_events(self.write('unknown.xml', self.event('Other')), ('Home', 'Away'), self.layout)

    def test_invalid_tag_intervals(self):
        for start, stop in [('0', '0'), ('-1', '0.1'), ('0', '1'), ('NaN', '0.1')]:
            with self.subTest(start=start, stop=stop), self.assertRaises(DataError):
                read_events(self.write('events.xml', self.event(start=start, stop=stop)), ('Home', 'Away'), self.layout)

    def test_configuration_rejects_invalid_values(self):
        for changes in [{'control_observations': 26}, {'control_agreement': 0.5},
                        {'control_radius_m': float('nan')}, {'interpolation_frames': -1},
                        {'control_window_frames': 2.5}, {'stationary_frames': True}]:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                replace(Config(), **changes)

    def test_configuration_requires_complete_schema(self):
        with self.assertRaises(ValueError):
            Config.load(self.write('config.json', '{"unexpected": 1}'))
