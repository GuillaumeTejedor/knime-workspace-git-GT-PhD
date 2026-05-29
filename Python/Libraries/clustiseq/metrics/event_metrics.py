import numpy as np
import warnings


class em_base:
    """Default metric between event types"""

    def __call__(self, evt1, evt2) -> float:
        try:
            return np.abs((evt1 - evt2))
        except:
            warnings.warn(f"Impossible to substract '{evt2}' to '{evt1}'.")
            return 0.0


class em_euclidean(em_base):
    """Euclidean metric: 

    Warning
    -------
    Require numerical events or probabilistic events.
    """

    def __call__(self, evt1, evt2) -> float:
        return np.sqrt((evt1 - evt2)**2)
        
class em_hamming(em_base):
    """Hamming metric: it equals 1 if different, 0 otherwise

    Warning
    -------
    Require the event types to be equipped with `__eq__()` function.
    """

    def __call__(self, evt1, evt2) -> float:
        try:
            if evt1 == evt2:
                return 0.0
            else:
                return 1.0
        except:
            warnings.warn(
                f"Impossible to compare '{evt2}' to '{evt1}': no __eq__ function."
            )
            return 0.0


class em_mdist(em_base):
    """Metric based on a distance matrix between event types."""

    def __init__(self, distmat) -> float:
        self.mat = distmat

    def __call__(self, evt1, evt2):
        try:
            return self.mat[evt1, evt2]
        except:
            warnings.warn("Index out of the index of distance matrix")
            return 0.0


class em_set:
    """Metrics between set of events"""

    def __init__(self, base=em_base(), agg=np.sum):
        """
        Parameters
        ==========
        base: em_base
            pairwise event metric
        param agg functor
            aggregation function (default is a sum)"""
        self.em = base
        self.agg = agg

    def __call__(self, evts1, evts2) -> float:
        """
        Parameters
        ==========
        evts1,evts2:
            list of events to compare.

        Returns
        =======
        A float that corresponds to the aggregation of the pairwise metrics."""
        if not isinstance(evts1, list):
            evts1 = [evts1]
        if not isinstance(evts2, list):
            evts2 = [evts2]
        return self.agg([self.em(ev1, ev2) for ev1 in evts1 for ev2 in evts2])
