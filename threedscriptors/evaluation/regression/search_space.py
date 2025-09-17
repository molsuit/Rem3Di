import numpy as np
from sklearn.utils import check_random_state
from typing import Sequence, Optional, Any, Dict

class Dist:
    def rvs(self, size: Optional[int]=None, random_state: Optional[int]=None):
        raise NotImplementedError

class Choice(Dist):
    def __init__(self, values: Sequence[Any]):
        self.values = list(values)
    def rvs(self, size=None, random_state=None):
        rng = check_random_state(random_state)
        if size is None:
            return rng.choice(self.values)
        idx = rng.randint(0, len(self.values), size=size)
        return [self.values[i] for i in idx]

class IntRange(Dist):
    def __init__(self, low: int, high: int, *, inclusive: bool = False):
        self.low, self.high, self.inclusive = int(low), int(high), inclusive
    def rvs(self, size=None, random_state=None):
        rng = check_random_state(random_state)
        high = self.high + (1 if self.inclusive else 0)
        return rng.randint(self.low, high, size=size)

class FloatRange(Dist):
    def __init__(self, low: float, high: float):
        self.low, self.high = float(low), float(high)
    def rvs(self, size=None, random_state=None):
        rng = check_random_state(random_state)
        return rng.uniform(self.low, self.high, size=size)

class LogUniform(Dist):
    def __init__(self, low: float, high: float, base: float = np.e):
        assert low > 0 and high > low
        self.low, self.high, self.base = float(low), float(high), float(base)
        self._loglow = np.log(self.low) / np.log(self.base)
        self._loghigh = np.log(self.high) / np.log(self.base)
    def rvs(self, size=None, random_state=None):
        rng = check_random_state(random_state)
        s = rng.uniform(self._loglow, self._loghigh, size=size)
        return np.power(self.base, s)