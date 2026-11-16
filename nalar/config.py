from dataclasses import asdict, dataclass
import json
import math

from games import GAME, GAMES, half_video_blocks
from starter.load_smart_tagging import KICKOFF_VIDEO_FRAMES


@dataclass(frozen=True)
class Config:
    control_radius_m: float = 2.5
    control_margin_m: float = 1.0
    control_observations: int = 10
    control_window_frames: int = 25
    control_agreement: float = 0.8
    interpolation_frames: int = 5
    tag_confirmation_frames: int = 50
    exit_margin_m: float = 1.0
    exit_confirmation_frames: int = 10
    stationary_frames: int = 50
    stationary_span_m: float = 0.75
    restart_speed_mps: float = 2.0
    restart_motion_frames: int = 5

    def __post_init__(self):
        for name, value in asdict(self).items():
            if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
                raise ValueError(f"invalid configuration value: {name}")
            if name.endswith('frames') or name == 'control_observations':
                if type(value) is not int or value < (0 if name == 'interpolation_frames' else 1):
                    raise ValueError(f"invalid frame/observation count: {name}")
            elif value <= 0:
                raise ValueError(f"configuration must be positive: {name}")
        if self.control_agreement <= 0.5 or self.control_agreement > 1:
            raise ValueError('control_agreement must be in (0.5, 1]')
        if self.control_observations > self.control_window_frames:
            raise ValueError('control_observations exceeds control_window_frames')

    @classmethod
    def load(cls, path):
        with open(path, encoding='utf-8') as handle:
            values = json.load(handle)
        if not isinstance(values, dict) or set(values) != set(asdict(cls())):
            raise ValueError('configuration must specify exactly the documented parameters')
        return cls(**values)


@dataclass(frozen=True)
class Layout:
    n_frames: int
    fps: int
    pitch: tuple
    blocks: tuple
    kickoffs: tuple

    def __post_init__(self):
        if self.n_frames <= 0 or self.fps <= 0 or len(self.blocks) != len(self.kickoffs):
            raise ValueError('invalid layout')
        previous = 0
        for (start, stop), kickoff in zip(self.blocks, self.kickoffs):
            if not (previous <= start <= kickoff < stop <= self.n_frames):
                raise ValueError('invalid half/kickoff boundaries')
            previous = stop

    def counts(self):
        graded = sum(stop - start for start, stop in self.blocks)
        waiting = sum(kickoff - start for (start, _), kickoff in zip(self.blocks, self.kickoffs))
        return {'video_frames': self.n_frames, 'graded_frames': graded,
                'ungraded_frames': self.n_frames - graded,
                'pre_kickoff_graded_frames': waiting,
                'kickoff_to_half_end_frames': graded - waiting}


def official_layout():
    game = GAMES[GAME]
    blocks = tuple((start, last + 1) for _, start, last in half_video_blocks(GAME))
    return Layout(game['n_video_frames'], game['fps'], tuple(game['pitch']), blocks,
                  (KICKOFF_VIDEO_FRAMES['1st Half'], KICKOFF_VIDEO_FRAMES['2nd Half']))
