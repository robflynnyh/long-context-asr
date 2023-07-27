import torch, torch.nn as nn
from torch import Tensor
from typing import Optional, Tuple, Union, Callable, List, Any
import random

'''
taken from: https://github.com/k2-fsa/icefall/blob/master/egs/librispeech/ASR/zipformer/scaling.py
'''

class PiecewiseLinear(object):
    """
    Piecewise linear function, from float to float, specified as nonempty list of (x,y) pairs with
    the x values in order.  x values <[initial x] or >[final x] are map to [initial y], [final y]
    respectively.
    """
    def __init__(self, *args):
        assert len(args) >= 1, len(args)
        if len(args) == 1 and isinstance(args[0], PiecewiseLinear):
            self.pairs = list(args[0].pairs)
        else:
            self.pairs = [ (float(x), float(y)) for x,y in args ]
        for (x,y) in self.pairs:
            assert isinstance(x, (float, int)), type(x)
            assert isinstance(y, (float, int)), type(y)

        for i in range(len(self.pairs) - 1):
            assert self.pairs[i + 1][0] > self.pairs[i][0], (i, self.pairs[i], self.pairs[i + 1])

    def __str__(self):
        # e.g. 'PiecewiseLinear((0., 10.), (100., 0.))'
        return f'PiecewiseLinear({str(self.pairs)[1:-1]})'

    def __call__(self, x):
        if x <= self.pairs[0][0]:
            return self.pairs[0][1]
        elif x >= self.pairs[-1][0]:
            return self.pairs[-1][1]
        else:
            cur_x, cur_y = self.pairs[0]
            for i in range(1, len(self.pairs)):
                next_x, next_y = self.pairs[i]
                if x >= cur_x and x <= next_x:
                    return cur_y + (next_y - cur_y) * (x - cur_x) / (next_x - cur_x)
                cur_x, cur_y = next_x, next_y
            assert False

    def __mul__(self, alpha):
        return PiecewiseLinear(
            * [(x, y * alpha) for x, y in self.pairs])

    def __add__(self, x):
        if isinstance(x, (float, int)):
            return PiecewiseLinear(
                * [(p[0], p[1] + x) for p in self.pairs])
        s, x = self.get_common_basis(x)
        return PiecewiseLinear(
            * [(sp[0], sp[1] + xp[1]) for sp, xp in zip(s.pairs, x.pairs)])

    def max(self, x):
        if isinstance(x, (float, int)):
            x = PiecewiseLinear( (0, x) )
        s, x = self.get_common_basis(x, include_crossings=True)
        return PiecewiseLinear(
            * [(sp[0], max(sp[1], xp[1])) for sp, xp in zip(s.pairs, x.pairs)])

    def min(self, x):
        if isinstance(x, float) or isinstance(x, int):
            x = PiecewiseLinear( (0, x) )
        s, x = self.get_common_basis(x, include_crossings=True)
        return PiecewiseLinear(
            * [ (sp[0], min(sp[1], xp[1])) for sp, xp in zip(s.pairs, x.pairs)])

    def __eq__(self, other):
        return self.pairs == other.pairs

    def get_common_basis(self,
                         p: 'PiecewiseLinear',
                         include_crossings: bool = False):
        """
        Returns (self_mod, p_mod) which are equivalent piecewise linear
        functions to self and p, but with the same x values.

          p: the other piecewise linear function
          include_crossings: if true, include in the x values positions
              where the functions indicate by this and p crosss.
        """
        assert isinstance(p, PiecewiseLinear), type(p)

        # get sorted x-values without repetition.
        x_vals = sorted(set([ x for x, _ in self.pairs ] + [ x for x, _ in p.pairs ]))
        y_vals1 = [ self(x) for x in x_vals ]
        y_vals2 = [ p(x) for x in x_vals ]

        if include_crossings:
            extra_x_vals = []
            for i in range(len(x_vals) - 1):
                if (y_vals1[i] > y_vals2[i]) != (y_vals1[i+1] > y_vals2[i+1]):
                    # if the two lines in this subsegment potentially cross each other..
                    diff_cur = abs(y_vals1[i] - y_vals2[i])
                    diff_next = abs(y_vals1[i+1] - y_vals2[i+1])
                    # `pos`, between 0 and 1, gives the relative x position,
                    # with 0 being x_vals[i] and 1 being x_vals[i+1].
                    pos = diff_cur / (diff_cur + diff_next)
                    extra_x_val = x_vals[i] + pos * (x_vals[i+1] - x_vals[i])
                    extra_x_vals.append(extra_x_val)
            if len(extra_x_vals) > 0:
                x_vals = sorted(set(x_vals + extra_x_vals))
        y_vals1 = [ self(x) for x in x_vals ]
        y_vals2 = [ p(x) for x in x_vals ]
        return ( PiecewiseLinear(* zip(x_vals, y_vals1)),
                 PiecewiseLinear(* zip(x_vals, y_vals2)) )

class ScheduledFloat(torch.nn.Module):
    """
    This object is a torch.nn.Module only because we want it to show up in [top_level module].modules();
    it does not have a working forward() function.  You are supposed to cast it to float, as
    in, float(parent_module.whatever), and use it as something like a dropout prob.

    It is a floating point value whose value changes depending on the batch count of the
    training loop.  It is a piecewise linear function where you specify the (x,y) pairs
    in sorted order on x; x corresponds to the batch index.  For batch-index values before the
    first x or after the last x, we just use the first or last y value.

    Example:
       self.dropout = ScheduledFloat((0.0, 0.2), (4000.0, 0.0), default=0.0)

    `default` is used when self.batch_count is not set or not in training mode or in
     torch.jit scripting mode.
    """
    def __init__(self,
                 *args,
                 default: float = 0.0):
        super().__init__()
        # self.batch_count and self.name will be written to in the training loop.
        self.batch_count = None
        self.name = None
        self.default = default
        self.schedule = PiecewiseLinear(*args)

    def extra_repr(self) -> str:
        return f'batch_count={self.batch_count}, schedule={str(self.schedule.pairs[1:-1])}'

    def __float__(self):
        batch_count = self.batch_count
        if batch_count is None or not self.training or torch.jit.is_scripting() or torch.jit.is_tracing():
            return float(self.default)
        else:
            ans = self.schedule(self.batch_count)
            if random.random() < 0.0002:
                logging.info(f"ScheduledFloat: name={self.name}, batch_count={self.batch_count}, ans={ans}")
            return ans

    def __add__(self, x):
        if isinstance(x, float) or isinstance(x, int):
            return ScheduledFloat(self.schedule + x,
                                  default=self.default)
        else:
            return ScheduledFloat(self.schedule + x.schedule,
                                  default=self.default+x.default)

    def max(self, x):
        if isinstance(x, float) or isinstance(x, int):
            return ScheduledFloat(self.schedule.max(x),
                                  default=self.default)
        else:
            return ScheduledFloat(self.schedule.max(x.schedule),
                                  default=max(self.default, x.default))


FloatLike = Union[float, ScheduledFloat]

class CutoffEstimator:
    """
    Estimates cutoffs of an arbitrary numerical quantity such that a specified
    proportion of items will be above the cutoff on average.

      p is the proportion of items that should be above the cutoff.
    """
    def __init__(self, p: float):
        self.p = p
        # total count of items
        self.count = 0
        # total count of items that were above the cutoff
        self.count_above = 0
        # initial cutoff value
        self.cutoff = 0

    def __call__(self, x: float) -> bool:
        """
        Returns true if x is above the cutoff.
        """
        ans = (x > self.cutoff)
        self.count += 1
        if ans:
            self.count_above += 1
        cur_p = self.count_above / self.count
        delta_p = cur_p - self.p
        if (delta_p > 0) == ans:
            q = abs(delta_p)
            self.cutoff = x * q + self.cutoff * (1-q)
        return ans

def _no_op(x: Tensor) -> Tensor:
    if torch.jit.is_scripting() or torch.jit.is_tracing():
        return x
    else:
        # a no-op function that will have a node in the autograd graph,
        # to avoid certain bugs relating to backward hooks
        return x.chunk(1, dim=-1)[0]

class Balancer(torch.nn.Module):
    """
    Modifies the backpropped derivatives of a function to try to encourage, for
    each channel, that it is positive at least a proportion `threshold` of the
    time.  It does this by multiplying negative derivative values by up to
    (1+max_factor), and positive derivative values by up to (1-max_factor),
    interpolated from 1 at the threshold to those extremal values when none
    of the inputs are positive.

    Args:
           num_channels: the number of channels
           channel_dim: the dimension/axis corresponding to the channel, e.g.
               -1, 0, 1, 2; will be interpreted as an offset from x.ndim if negative.
           min_positive: the minimum, per channel, of the proportion of the time
               that (x > 0), below which we start to modify the derivatives.
           max_positive: the maximum, per channel, of the proportion of the time
               that (x > 0), above which we start to modify the derivatives.
           scale_gain_factor: determines the 'gain' with which we increase the
              change in gradient once the constraints on min_abs and max_abs
              are violated.
           min_abs:  the minimum average-absolute-value difference from the mean
               value per channel, which we allow, before we start to modify
               the derivatives to prevent this.
           max_abs:  the maximum average-absolute-value difference from the mean
               value per channel, which we allow, before we start to modify
               the derivatives to prevent this.
         prob: determines the minimum probability with which we modify the
             gradients for the {min,max}_positive and {min,max}_abs constraints,
             on each forward().  This is done randomly to prevent all layers
             from doing it at the same time.
    """
    def __init__(
            self,
            num_channels: int,
            channel_dim: int,
            min_positive: FloatLike = 0.05,
            max_positive: FloatLike = 0.95,
            min_abs: FloatLike = 0.2,
            max_abs: FloatLike = 100.0,
            grad_scale: FloatLike = 0.04,
            prob: Optional[FloatLike] = None,
    ):
        super().__init__()

        if prob is None:
            prob = ScheduledFloat((0.0, 0.5), (8000.0, 0.125), default=0.4)
        self.prob = prob
        # 5% of the time we will return and do nothing because memory usage is
        # too high.
        self.mem_cutoff = CutoffEstimator(0.05)

        # actually self.num_channels is no longer needed except for an assertion.
        self.num_channels = num_channels
        self.channel_dim = channel_dim
        self.min_positive = min_positive
        self.max_positive = max_positive
        self.min_abs = min_abs
        self.max_abs = max_abs
        self.grad_scale = grad_scale

    def forward(self, x: Tensor) -> Tensor:
        if (torch.jit.is_scripting() or not x.requires_grad or
           (x.is_cuda and self.mem_cutoff(torch.cuda.memory_allocated()))):
            return _no_op(x)

        prob = float(self.prob)
        if random.random() < prob:
            # The following inner-functions convert from the way we historically specified
            # these limitations, as limits on the absolute value and the proportion of positive
            # values, to limits on the RMS value and the (mean / stddev).
            def _abs_to_rms(x):
                # for normally distributed data, if the expected absolute value is x, the
                # expected rms value will be sqrt(pi/2) * x.
                return 1.25331413732 * x

            def _proportion_positive_to_mean(x):
                def _atanh(x):
                    eps = 1.0e-10
                    # eps is to prevent crashes if x is exactly 0 or 1.
                    # we'll just end up returning a fairly large value.
                    return (math.log (1+x+eps) - math.log (1-x+eps)) / 2.

                def _approx_inverse_erf(x):
                    # 1 / (sqrt(pi) * ln(2)),
                    # see https://math.stackexchange.com/questions/321569/approximating-the-error-function-erf-by-analytical-functions
                    # this approximation is extremely crude and gets progressively worse for
                    # x very close to -1 or +1, but we mostly care about the "middle" region
                    # e.g. _approx_inverse_erf(0.05) = 0.0407316414078772,
                    # and math.erf(0.0407316414078772) = 0.045935330944660666,
                    # which is pretty close to 0.05.
                    return 0.8139535143 * _atanh(x)
                # first convert x from the range 0..1 to the range -1..1 which the error
                # function returns
                x = -1 + (2 * x)
                return _approx_inverse_erf(x)

            min_mean = _proportion_positive_to_mean(float(self.min_positive))
            max_mean = _proportion_positive_to_mean(float(self.max_positive))
            min_rms = _abs_to_rms(float(self.min_abs))
            max_rms = _abs_to_rms(float(self.max_abs))
            grad_scale = float(self.grad_scale)

            assert x.shape[self.channel_dim] == self.num_channels

            return BalancerFunction.apply(
                x, min_mean, max_mean, min_rms, max_rms, grad_scale, self.channel_dim
            )
        else:
            return _no_op(x)