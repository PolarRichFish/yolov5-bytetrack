import numpy as np
from collections import OrderedDict

"""
 定义`TrackState`类：

   - `TrackState`是一个枚举类，定义了目标追踪的状态。它包括以下几种状态：
     - `New`：新目标，刚开始被追踪。
     - `Tracked`：追踪中的目标。
     - `Lost`：丢失的目标，无法被正确追踪。
     - `Removed`：移除的目标，不再进行追踪。
"""
class TrackState(object):
    New = 0
    Tracked = 1
    Lost = 2
    Removed = 3


class BaseTrack(object):
    _count = 0

    track_id = 0
    is_activated = False
    state = TrackState.New

    history = OrderedDict()
    features = []
    curr_feature = None
    score = 0
    start_frame = 0
    frame_id = 0
    time_since_update = 0

    # multi-camera
    location = (np.inf, np.inf)

    @property
    def end_frame(self):
        return self.frame_id

    @staticmethod
    def next_id():
        BaseTrack._count += 1
        return BaseTrack._count

    def activate(self, *args):
        raise NotImplementedError

    def predict(self):
        raise NotImplementedError

    def update(self, *args, **kwargs):
        raise NotImplementedError

    def mark_lost(self):
        self.state = TrackState.Lost

    def mark_removed(self):
        self.state = TrackState.Removed