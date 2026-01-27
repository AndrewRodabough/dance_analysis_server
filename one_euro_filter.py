# -------------------- 1. ONE EURO FILTER CLASS (Self-Contained) -------------------- #
# This replaces: from mmpose.functional import OneEuroFilter
import numpy as np
import math

class LowPassFilter:
    def __init__(self, alpha):
        self.__setAlpha(alpha)
        self.y = None
        self.s = None

    def __setAlpha(self, alpha):
        # Handle array alpha values by clipping element-wise
        if isinstance(alpha, np.ndarray):
            alpha = np.clip(alpha, 0.0, 1.0)
        else:
            alpha = float(alpha)
            if alpha < 0 or alpha > 1:
                alpha = 0.85
        self.alpha = alpha

    def __call__(self, value, timestamp=None, alpha=None):
        if alpha is not None:
            self.__setAlpha(alpha)
        if self.y is None:
            s = value
        else:
            s = self.alpha * value + (1.0 - self.alpha) * self.s
        self.y = value
        self.s = s
        return s

class OneEuroFilter:
    def __init__(self, min_cutoff=1.0, beta=0.0, d_cutoff=1.0, freq=30.0):
        self.freq = freq
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x_filter = LowPassFilter(self.alpha(min_cutoff))
        self.dx_filter = LowPassFilter(self.alpha(d_cutoff))
        self.t_prev = None

    def alpha(self, cutoff):
        te = 1.0 / self.freq
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / te)

    def __call__(self, x, t=None):
        # Handle numpy arrays for x
        if t is None:
            if self.t_prev is None:
                t = 0
            else:
                t = self.t_prev + (1.0 / self.freq)
        
        if self.t_prev is None:
            self.t_prev = t
            self.x_filter.s = x
            self.dx_filter.s = np.zeros_like(x)
            return x

        te = t - self.t_prev
        self.alpha = lambda cutoff: 1.0 / (1.0 + (1.0 / (2 * math.pi * np.clip(cutoff, 0.1, 1e6))) / te)

        ad = self.dx_filter( (x - self.x_filter.s) / te, t, alpha=self.alpha(self.d_cutoff))
        ed = np.abs(ad)
        cutoff = self.min_cutoff + self.beta * ed
        # Ensure cutoff is scalar or broadcast-compatible
        cutoff = np.asarray(cutoff).mean() if isinstance(cutoff, np.ndarray) else cutoff
        result = self.x_filter(x, t, alpha=self.alpha(cutoff))
        
        self.t_prev = t
        return result