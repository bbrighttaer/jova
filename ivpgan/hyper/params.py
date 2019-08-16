# Author: bbrighttaer
# Project: ivpgan
# Date: 6/27/19
# Time: 10:33 AM
# File: params.py


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import abc

import numpy as np
import numpy.random as rand


def _get_size(size):
    if isinstance(size, DiscreteParam):
        s = size.sample()
    else:
        s = size
    return s


class Param(abc.ABC):
    def __init__(self, min, max, choices, size):
        self.min = min
        self.max = max
        self.choices = choices
        self.size = size

    @abc.abstractmethod
    def sample(self):
        pass


class DiscreteParam(Param):
    """
    Random sampling of digits between min and max (inclusive).
    """
    __name__ = "discrete_param"

    def __init__(self, min, max, size=1):
        super(DiscreteParam, self).__init__(min, max, None, size)

    def sample(self):
        size = _get_size(self.size)
        val = rand.randint(self.min, self.max + 1, size).tolist()
        if len(val) == 1 and not isinstance(self.size, DiscreteParam):
            return val[0]
        else:
            return val


class CategoricalParam(Param):
    """
    Random selection of a list of options.
    """
    __name__ = "categorical_param"

    def __init__(self, choices, size=1):
        super(CategoricalParam, self).__init__(None, None, choices, size)

    def sample(self):
        size = _get_size(self.size)
        sel = []
        for i in range(size):
            probs = rand.uniform(0., 1., len(self.choices))
            idx = np.argmax(probs)
            sel.append(self.choices[idx])
        if len(sel) == 1 and not isinstance(self.size, DiscreteParam):
            return sel[0]
        else:
            return sel


class LogRealParam(Param):
    """
    Random sampling of real values using the log-scale.
    """
    __name__ = "continuous_param"

    def __init__(self, min=-4, max=0, size=1):
        super(LogRealParam, self).__init__(min, max, None, size)

    def sample(self):
        size = _get_size(self.size)
        vals = []
        for i in range(size):
            r = self.min * rand.rand() + self.max
            v = pow(10, r)
            vals.append(v)

        if len(vals) == 1 and not isinstance(self.size, DiscreteParam):
            return vals[0]
        else:
            return vals


class ConstantParam(Param):
    """A wrapper for constant values"""

    __name__ = "constant_param"

    def __init__(self, c):
        super(ConstantParam, self).__init__(None, None, None, None)
        self._c = c

    def sample(self):
        return self._c


class RealParam(Param):
    __name__ = "real_range_param"

    def __init__(self, min=1e-5, max=1., size=1):
        super(RealParam, self).__init__(min, max, None, size)

    def sample(self):
        size = _get_size(self.size)
        vals = rand.uniform(self.min, self.max, size)
        if len(vals) == 1 and not isinstance(self.size, DiscreteParam):
            return vals[0]
        return vals


class DictParam(Param):
    __name__ = "dict_param"

    def __init__(self, dict_params):
        super(DictParam, self).__init__(None, None, None, None)
        self.p_dict = dict_params

    def keys(self):
        return self.p_dict.keys()

    def values(self):
        return self.p_dict.values()

    def sample(self):
        params_dict = dict()
        for param in self.p_dict:
            params_dict[param] = self.p_dict[param].sample()
        return params_dict
