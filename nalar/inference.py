from collections import deque

import numpy as np

H, A, D, X = 0, 1, 2, 3
LABELS = np.array(['H', 'A', 'D', 'X'])
ATTACK_CODES = frozenset(('build up', 'progression', 'creating chances', 'attacking transition'))


def runs(mask):
    edges = np.diff(np.r_[0, np.asarray(mask, dtype=np.int8), 0])
    return zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1))


def phase_votes(events, n):
    delta = np.zeros((n + 1, 2), dtype=np.int32)
    for event in events:
        if event.code in ATTACK_CODES and event.team in (H, A):
            delta[event.start, event.team] += 1
            delta[event.stop, event.team] -= 1
    present = np.cumsum(delta[:-1], axis=0) > 0
    votes = np.full(n, -1, dtype=np.int8)
    votes[present[:, 0] & ~present[:, 1]] = H
    votes[present[:, 1] & ~present[:, 0]] = A
    return votes


def interpolate_ball(features, layout, limit, blocked=None):
    ball = features.ball.copy()
    blocked = np.zeros(layout.n_frames, dtype=bool) if blocked is None else blocked
    if limit:
        for start, stop in layout.blocks:
            for a, b in runs(~features.observed[start:stop]):
                a, b = int(a) + start, int(b) + start
                if b - a <= limit and a > start and b < stop and not np.any(blocked[a - 1:b + 1]):
                    fraction = np.arange(1, b - a + 1)[:, None] / (b - a + 1)
                    ball[a:b] = ball[a - 1] + fraction * (ball[b] - ball[a - 1])
    return ball


def stationary_windows(ball, observed, window, span):
    """Observed rolling windows whose bounding-box diagonal does not exceed span."""
    n = len(ball)
    result = np.zeros(n, dtype=bool)
    queues = [[deque(), deque()] for _ in range(2)]
    length = 0
    for i in range(n):
        if not observed[i]:
            length = 0
            for pair in queues:
                for q in pair:
                    q.clear()
            continue
        length += 1
        extent = []
        for axis in (0, 1):
            low, high = queues[axis]
            for q, lower in ((low, True), (high, False)):
                while q and (ball[q[-1], axis] >= ball[i, axis] if lower
                             else ball[q[-1], axis] <= ball[i, axis]):
                    q.pop()
                q.append(i)
                while q[0] <= i - window:
                    q.popleft()
            extent.append(ball[high[0], axis] - ball[low[0], axis])
        if length >= window:
            result[i] = np.hypot(*extent) <= span
    return result


def temporal_evidence(features, events, layout, config):
    n = layout.n_frames
    ball = features.ball
    inbounds = features.observed & (features.ball >= 0).all(axis=1) & (
        features.ball <= np.asarray(layout.pitch)).all(axis=1)
    speed = np.full(n, np.nan)
    for start, stop in layout.blocks:
        speed[start + 1:stop] = np.linalg.norm(np.diff(ball[start:stop], axis=0), axis=1) * layout.fps
    previous_observed = np.r_[False, features.observed[:-1]]
    moving = (speed > config.restart_speed_mps) & inbounds & previous_observed
    motion_onset = np.full(n, -1, dtype=np.int64)
    for a, b in runs(moving):
        confirmed = a + config.restart_motion_frames - 1
        if confirmed < b:
            motion_onset[confirmed:b] = a

    outside = features.observed & ((features.ball < -config.exit_margin_m).any(axis=1) |
              (features.ball > np.asarray(layout.pitch) + config.exit_margin_m).any(axis=1))
    exits = np.zeros(n, dtype=bool)
    stationary = np.zeros(n, dtype=bool)
    for start, stop in layout.blocks:
        for a, b in runs(outside[start:stop]):
            if b - a >= config.exit_confirmation_frames:
                exits[int(a) + start:int(b) + start] = True
        stationary[start:stop] = stationary_windows(
            features.ball[start:stop], features.observed[start:stop],
            config.stationary_frames, config.stationary_span_m)

    waiting_periods = []
    restart = np.zeros(n, dtype=bool)
    team_mask = np.zeros(n, dtype=np.int8)
    for event in events:
        if event.code != 'set pieces':
            continue
        # An interval that crosses a half boundary cannot connect the two episodes.
        for half_start, half_stop in layout.blocks:
            a, b = max(event.start, half_start), min(event.stop, half_stop)
            if a >= b:
                continue
            search = a
            dead_start = None
            onset = None
            for i in range(a, b):
                if i in layout.kickoffs:
                    onset = i
                    break
                if motion_onset[i] >= 0:
                    candidate = max(search, int(motion_onset[i]))
                    if i - candidate + 1 >= config.restart_motion_frames:
                        onset = candidate
                        break  # A completed restart cannot acquire a later waiting period.
                if dead_start is None and i - a + 1 >= config.stationary_frames and stationary[i]:
                    dead_start = i - config.stationary_frames + 1
                    search = i + 1
            if onset is not None:
                restart[onset] = True
                if event.team in (H, A):
                    team_mask[onset] |= 1 << event.team
            if dead_start is not None:
                end = onset if onset is not None else b
                for kickoff in layout.kickoffs:
                    if event.start <= kickoff < event.stop:
                        end = min(end, kickoff)
                if dead_start < end:
                    waiting_periods.append((dead_start, end))

    # Only these published anchors are exceptions to inferred restart timing.
    for kickoff in layout.kickoffs:
        restart[kickoff] = True
        for event in events:
            if event.code == 'set pieces' and event.start <= kickoff < event.stop and event.team in (H, A):
                team_mask[kickoff] |= 1 << event.team
    # A confirmed restart ends all waits already in progress, including waits
    # from shorter overlapping tags that could not observe motion confirmation.
    restart_frames = np.flatnonzero(restart)
    dead_delta = np.zeros(n + 1, dtype=np.int32)
    for start, stop in waiting_periods:
        next_restart = np.searchsorted(restart_frames, start)
        if next_restart < len(restart_frames):
            stop = min(stop, int(restart_frames[next_restart]))
        if start < stop:
            dead_delta[start] += 1
            dead_delta[stop] -= 1
    restart_team = np.full(n, -1, dtype=np.int8)
    restart_team[team_mask == 1] = H
    restart_team[team_mask == 2] = A
    return exits, np.cumsum(dead_delta[:-1]) > 0, restart, restart_team, motion_onset


def infer(features, events, layout, config, *, use_tags=True, use_dead_ball=True):
    n = layout.n_frames
    labels = np.full(n, X, dtype=np.int8)
    distances = features.distances
    geometry = np.full(n, -1, dtype=np.int8)
    for team in (H, A):
        good = features.observed & (distances[:, team] <= config.control_radius_m) & (
            distances[:, team] + config.control_margin_m <= distances[:, 1 - team])
        geometry[good] = team
    tags = phase_votes(events, n) if use_tags else np.full(n, -1, dtype=np.int8)
    exits, stationary, restart, restart_team, motion = temporal_evidence(
        features, events if use_tags else [], layout, config)
    if not use_dead_ball:
        exits[:] = False
        stationary[:] = False
    if not use_tags:
        restart_team[:] = -1

    for (start, stop), kickoff in zip(layout.blocks, layout.kickoffs):
        labels[start:kickoff] = D
        state, episode = X, kickoff
        window = deque()
        totals = [0, 0]
        tag_team, tag_start = -1, kickoff
        dead_since = kickoff
        for i in range(kickoff, stop):
            if exits[i] or stationary[i]:
                if state != D:
                    dead_since = i
                state = D
                labels[i] = D
                window.clear()
                totals[:] = [0, 0]
                tag_team, tag_start = -1, i + 1
                continue
            if restart[i] and (state == D or i == kickoff):
                episode = i
                state = int(restart_team[i]) if restart_team[i] >= 0 else X
                window.clear()
                totals[:] = [0, 0]
                tag_team, tag_start = -1, i
            moving_since = int(motion[i]) if state == D and motion[i] >= dead_since else None

            oldest = i - config.control_window_frames
            if moving_since is not None:
                # Control before this motion episode cannot establish a restart.
                oldest = max(oldest, moving_since - 1)
            while window and window[0][0] <= oldest:
                _, old = window.popleft()
                totals[old] -= 1
            vote = int(geometry[i])
            if vote >= 0:
                window.append((i, vote))
                totals[vote] += 1
            confirmed = -1
            for team in (H, A):
                if totals[team] >= config.control_observations and (
                        totals[team] >= config.control_agreement * sum(totals)) and window[-1][1] == team:
                    confirmed = team

            tag = int(tags[i]) if not features.observed[i] else -1
            if tag < 0 or tag != tag_team:
                tag_team, tag_start = tag, i
            tag_confirmed = tag >= 0 and i - tag_start + 1 >= config.tag_confirmation_frames

            target, boundary = -1, i
            if confirmed >= 0 and (state != D or moving_since is not None):
                target = confirmed
                last_opponent = next((t for t, v in reversed(window) if v != confirmed), episode - 1)
                boundary = next(t for t, v in window if v == confirmed and t > last_opponent)
                if state == D:
                    episode = moving_since
            elif tag_confirmed:
                target, boundary = tag, tag_start
                if state == D:
                    episode = tag_start
            if target >= 0 and target != state:
                boundary = episode if state in (X, D) else max(episode, boundary)
                labels[boundary:i + 1] = target
                state = target
                window.clear()
                totals[:] = [0, 0]
                tag_team, tag_start = -1, i + 1
            labels[i] = state
    return labels


def summarize(labels, features, layout):
    counts = layout.counts()
    graded = np.zeros(layout.n_frames, dtype=bool)
    possessions = 0
    for start, stop in layout.blocks:
        graded[start:stop] = True
        part = labels[start:stop]
        boundaries = np.r_[True, part[1:] != part[:-1]]
        possessions += int(np.sum(boundaries & ((part == H) | (part == A))))
    counts.update({'observed_ball_frames': int(features.observed.sum()),
                   'observed_ball_graded_frames': int(np.sum(features.observed & graded)),
                   'label_counts': {label: int(np.sum(labels == i)) for i, label in enumerate(LABELS)},
                   'graded_label_counts': {label: int(np.sum((labels == i) & graded)) for i, label in enumerate(LABELS)},
                   'predicted_possessions_raw_X_as_gap': possessions})
    return counts
