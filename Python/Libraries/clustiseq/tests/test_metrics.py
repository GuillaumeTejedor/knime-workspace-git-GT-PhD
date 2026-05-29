import sys

sys.path.append(".")

from clustiseq.metrics.event_metrics import em_set, em_base, em_hamming
from clustiseq.timedsequence import TimedSequence
import numpy as np

from clustiseq.metrics.dtw import euclidean, lev, lcss, drop_dtw

"""
TODO: make assertions
"""


def test_eventmetrics():
    # comparison of a collection of event types
    evts1 = [1, 4, 6]
    evts2 = [6]
    dist = em_set()
    ret = dist(evts1, evts2)
    print(ret)

    # we modify the aggregation function
    dist = em_set(agg=np.mean)
    ret = dist(evts1, evts2)
    print(ret)

    dist = em_set(agg=np.min)
    ret = dist(evts1, evts2)
    print(ret)


def test_sim():
    seq1 = [(1, 1.0), (3, 2.0), (2, 3.0), (1, 8.0), (1, 10.0), (2, 12.0), (1, 15.0)]

    seq2 = [(3, 17.0), (2, 20.0), (3, 23.0), (3, 25.0), (2, 26.0), (3, 28.0), (2, 30.0)]
    dates = np.array([e[1] for e in seq1], dtype=float)
    data = np.array([e[0] for e in seq1])
    ts1 = TimedSequence(dates, data)

    dates = np.array([e[1] for e in seq2], dtype=float)
    data = np.array([e[0] for e in seq2])
    ts2 = TimedSequence(dates, data)

    d = euclidean()
    ret = d(ts1, ts2)
    print(ret)

    ret = d(ts1, ts1)
    print(ret)

def test_ddtw():
    seq1 = [(1, 1.0), (3, 2.0), (2, 3.0), (1, 8.0), (1, 10.0), (2, 12.0), (1, 15.0)]

    seq2 = [(3, 17.0), (2, 20.0), (3, 23.0), (3, 25.0), (2, 26.0), (3, 28.0), (2, 30.0)]
    dates = np.array([e[1] for e in seq1], dtype=float)
    data = np.array([e[0] for e in seq1])
    ts1 = TimedSequence(dates, data)

    dates = np.array([e[1] for e in seq2], dtype=float)
    data = np.array([e[0] for e in seq2])
    ts2 = TimedSequence(dates, data)

    d = drop_dtw(mode=1)
    ret = d(ts1, ts2)
    print(ret)
    
def test_lev():
    seq1 = [(1, 1.0), (3, 2.0), (2, 3.0), (1, 8.0), (1, 10.0), (2, 12.0), (1, 15.0)]

    seq2 = [(3, 17.0), (3, 25.0), (2, 26.0), (3, 28.0), (2, 30.0)]
    dates = np.array([e[1] for e in seq1], dtype=float)
    data = np.array([e[0] for e in seq1])
    ts1 = TimedSequence(dates, data)

    dates = np.array([e[1] for e in seq2], dtype=float)
    data = np.array([e[0] for e in seq2])
    ts2 = TimedSequence(dates, data)

    d = lev()
    ret = d(ts1, ts2)
    print(ret)

    d = lev(em=em_base())
    ret = d(ts1, ts2)
    print(ret)

    d = lev(costs=(1, 1, 1))  # similar as hamming
    ret = d(ts1, ts2)
    print(ret)

    d = lev(costs=(1, 1, 0.5))
    ret = d(ts1, ts2)
    print(ret)

    ret = d(ts1, ts1)
    print(ret)


def test_lcss():
    seq1 = [(1, 1.0), (3, 2.0), (2, 3.0), (1, 8.0), (1, 10.0), (2, 12.0), (1, 15.0)]

    seq2 = [(2, 17.0), (3, 25.0), (2, 26.0), (3, 28.0), (1, 30.0)]
    dates = np.array([e[1] for e in seq1], dtype=float)
    data = np.array([e[0] for e in seq1])
    ts1 = TimedSequence(dates, data)

    dates = np.array([e[1] for e in seq2], dtype=float)
    data = np.array([e[0] for e in seq2])
    ts2 = TimedSequence(dates, data)

    d = lcss()
    ret = d.lcss(ts1, ts2)
    print(ret)

    ret = d(ts1, ts2)
    print(ret)

    d = lcss(delta=15.0)
    ret = d(ts1, ts2)
    print(ret)


def test_dtw():
    pass


if __name__ == "__main__":
    #test_eventmetrics()
    #test_sim()
    #test_lev()
    #test_lcss()
    test_ddtw()
