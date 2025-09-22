import time


class ThrottledExecution:
    """
    This object evaluates to True if the last executed time is more than the
    throttled threshold (in seconds)
    """

    def __init__(self, throttled_threshold: float):
        super().__init__()
        self.throttled_threshold = throttled_threshold
        self.last_executed = time.time() - throttled_threshold * 2

    def __bool__(self):
        _now = time.time()
        if (_now - self.last_executed) >= self.throttled_threshold:
            self.last_executed = _now
            return True
        return False
