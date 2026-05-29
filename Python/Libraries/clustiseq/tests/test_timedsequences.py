import sys

sys.path.append(".")

import numpy as np
from clustiseq.timedsequence import TimedSequence


def test_timedsequences():
    seq = [
        ("a", 1),
        ("c", 2),
        ("b", 3),
        ("a", 8),
        ("a", 10),
        ("b", 12),
        ("a", 15),
        ("c", 17),
        ("b", 20),
        ("c", 23),
        ("c", 25),
        ("b", 26),
        ("c", 28),
        ("b", 30),
    ]

    dates = np.array(
        [np.datetime64("1970-01-01") + np.timedelta64(e[1], "D") for e in seq],
        dtype="datetime64",
    )
    data = np.array([e[0] for e in seq])

    ts = TimedSequence(dates, data)
    print(ts)
    print("---- time based selection ------")
    tssel = ts[ts < np.datetime64("1970-01-07")]
    print(tssel)

    print("----- item based selection ------")
    tssel = ts[ts == "a"]
    print(tssel)

    print("----- index based selection ------")
    evt = ts[3]
    print(evt)

    print("----- start -----")
    print(tssel.start())

    print("----- at ------")
    print(ts.at(np.datetime64("1970-01-02")))
    print(ts.at(np.datetime64("1970-01-08")))

    #############################""
    print("================================")
    dates = np.array([float(e[1]) for e in seq], dtype="float")
    data = np.array([e[0] for e in seq])

    ts = TimedSequence(dates, data)
    print(ts)
    print("---- time based selection ------")
    tssel = ts[ts < 6.0]
    print(tssel)

    try:
        tssel = ts[ts < 6]
    except ValueError:
        print("Floats are mandatory")

    print("----- item based selection ------")
    tssel = ts[ts == "a"]
    print(tssel)

    print("----- start -----")
    print(tssel.start())

    print("----- at ------")
    print(ts.at(2))
    print(ts.at(7.0))

    print("---- iteration ----")
    for i in ts:
        print(i)


if __name__ == "__main__":
    test_timedsequences()