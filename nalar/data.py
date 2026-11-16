from dataclasses import dataclass
import hashlib
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np


class DataError(ValueError):
    """An input contract failure whose message contains no restricted record contents."""


@dataclass
class Features:
    ball: np.ndarray
    distances: np.ndarray
    observed: np.ndarray


@dataclass(frozen=True)
class Event:
    code: str
    start: int
    stop: int
    team: int


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read_metadata(path):
    try:
        root = ET.parse(path).getroot()
    except (ET.ParseError, OSError):
        raise DataError('cannot read tracking metadata XML') from None
    teams = list(root.iter('Team'))
    ids = [t.get('id') for t in teams]
    names = tuple((t.findtext('Name') or '').strip() for t in teams)
    if len(ids) != 2 or any(not x for x in ids + list(names)) or len(set(ids)) != 2:
        raise DataError('tracking metadata must define two distinct named teams')
    players = list(root.iter('Player'))
    player_ids = [p.get('id') for p in players]
    if not players or any(not x for x in player_ids) or len(set(player_ids)) != len(players):
        raise DataError('tracking metadata must define distinct track slots')
    if any(p.get('teamId') not in ids for p in players):
        raise DataError('track slot references an unknown team')
    slots = np.array([ids.index(p.get('teamId')) for p in players], dtype=np.int8)
    if set(slots.tolist()) != {0, 1}:
        raise DataError('tracking must define slots for both teams')
    return slots, names


def read_tracking(path, slots, layout):
    n, p = layout.n_frames, len(slots)
    ball = np.full((n, 2), np.nan, dtype=np.float64)
    distances = np.full((n, 2), np.inf, dtype=np.float64)
    pitch = np.asarray(layout.pitch)
    count = 0
    with open(path, encoding='utf-8') as handle:
        for count, line in enumerate(handle, 1):
            try:
                fields = line.rstrip('\r\n').split(':')
                if len(fields) != 3 or int(fields[0]) != count or count > n:
                    raise ValueError
                groups = fields[1].split(';')
                if len(groups) != p:
                    raise ValueError
                players = np.array([list(map(float, g.split(','))) for g in groups])
                point = np.array(list(map(float, fields[2].split(','))))
                if players.shape != (p, 3) or point.shape != (3,):
                    raise ValueError
                if np.isinf(players).any() or np.isinf(point).any():
                    raise ValueError
                if np.any(np.isnan(players[:, 0]) != np.isnan(players[:, 1])):
                    raise ValueError
                if np.isnan(point[0]) != np.isnan(point[1]):
                    raise ValueError
            except (ValueError, OverflowError):
                raise DataError(f'invalid tracking structure or frame sequence at record {count}') from None
            i = count - 1
            try:
                with np.errstate(over='raise', invalid='raise'):
                    ball[i] = point[:2] * pitch
                    player_positions = players[:, :2] * pitch
                    valid = np.isfinite(player_positions).all(axis=1)
                    delta = player_positions - ball[i]
                    d = np.sqrt(np.sum(delta * delta, axis=1))
            except FloatingPointError:
                raise DataError(f'tracking coordinate arithmetic overflow at record {count}') from None
            if np.isfinite(ball[i]).all():
                for team in (0, 1):
                    use = valid & (slots == team)
                    if use.any():
                        distances[i, team] = np.min(d[use])
    if count != n:
        raise DataError(f'tracking record count {count} does not match expected {n}')
    return Features(ball, distances, np.isfinite(ball).all(axis=1))


def normalize(value):
    return ' '.join(value.casefold().split())


def read_events(path, names, layout):
    try:
        root = ET.parse(path).getroot()
    except (ET.ParseError, OSError):
        raise DataError('cannot read automatic tagging XML') from None
    lookup = {normalize(name): i for i, name in enumerate(names)}
    events = []
    for index, node in enumerate(root.iter('instance'), 1):
        try:
            start_s, stop_s = float(node.findtext('start')), float(node.findtext('end'))
            if not np.isfinite([start_s, stop_s]).all():
                raise ValueError
            start, stop = round(start_s * layout.fps), round(stop_s * layout.fps)
            if not 0 <= start < stop <= layout.n_frames:
                raise ValueError
            tags = {}
            for label in node.findall('label'):
                group, value = label.findtext('group'), label.findtext('text')
                if not group or value is None or normalize(group) in tags:
                    raise ValueError
                tags[normalize(group)] = normalize(value)
            name = tags.get('team', '')
            if name not in lookup and name not in ('n/a', ''):
                raise ValueError
            code = normalize(node.findtext('code') or '')
            if not code:
                raise ValueError
        except (ValueError, TypeError, OverflowError):
            raise DataError(f'invalid automatic tagging structure at instance {index}') from None
        events.append(Event(code, start, stop, lookup.get(name, -1)))
    if not events:
        raise DataError('automatic tagging contains no instances')
    return sorted(events, key=lambda e: (e.start, e.stop, e.code, e.team))


def load_inputs(data_dir, layout):
    from games import asset_paths
    paths = {key: Path(path) for key, path in asset_paths(data_dir=str(data_dir)).items()
             if key in ('atd_meta', 'atd_raw', 'smart_tagging')}
    for key, path in paths.items():
        if not path.is_file():
            raise DataError(f'missing tactical input asset: {key}; check --data-dir')
    slots, names = read_metadata(paths['atd_meta'])
    if len(slots) != 46 or tuple(map(normalize, names)) != ('barcelona', 'atletic club'):
        raise DataError('metadata does not match the documented tactical source')
    features = read_tracking(paths['atd_raw'], slots, layout)
    events = read_events(paths['smart_tagging'], names, layout)
    return features, events, paths
