import numpy as np

from nalar.config import Config, Layout
from nalar.data import Features


def scene(n=180, team=0):
    layout = Layout(n, 25, (105.3, 68.0), ((0, n),), (0,))
    ball = np.column_stack((10 + np.arange(n) * 0.12, np.full(n, 30.0)))
    distances = np.full((n, 2), 6.0)
    distances[:, team] = 0.5
    return Features(ball, distances, np.ones(n, dtype=bool)), layout, Config()


def missing(features, start, stop):
    features.ball[start:stop] = np.nan
    features.distances[start:stop] = np.inf
    features.observed[start:stop] = False
