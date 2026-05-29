"""
This test file illustrates how to use the KMeans algorithm
of `sklearn` with metrics computed between timed sequences.
"""
import sys

sys.path.append(".")

from clustiseq.metrics.event_metrics import em_set, em_base, em_hamming
from clustiseq.timedsequence import TimedSequence
import numpy as np

from clustiseq.metrics.dtw import euclidean, lev, lcss

from scipy.sparse import csr_matrix
from sklearn.cluster import KMeans

if __name__ == "__main__":
    # creation of two timed sequences
    seq1 = [(1, 1.0), (3, 2.0), (2, 3.0), (1, 8.0), (1, 10.0), (2, 12.0), (3, 15.0)]
    seq2 = [(3, 1.0), (2, 3.0), (3, 6.0), (3, 8.0), (2, 9.0), (1, 10.0), (2, 12.0)]
    rng = np.random.default_rng(12345)

    # creation of a dataset with noisy copies of the two sequences
    n = 20
    dataset = []
    for i in range(n):
        dates = np.array([e[1] + rng.standard_normal() for e in seq1], dtype=float)
        data = np.array([e[0] for e in seq1])
        dataset.append(TimedSequence(dates, data))

        dates = np.array([e[1] + rng.standard_normal() for e in seq2], dtype=float)
        data = np.array([e[0] for e in seq2])
        dataset.append(TimedSequence(dates, data))

    # creation of distance matrix
    metric = lev()
    row = []
    col = []
    data = []
    for i in range(len(dataset)):
        for j in range(i + 1, len(dataset)):
            row.append(i)
            col.append(j)
            data.append(metric(dataset[i], dataset[j]))
    X = csr_matrix((data, (row, col)), shape=(len(dataset), len(dataset)))

    # set up and run the KMean algorithm
    kmeans = KMeans(2)
    labels = kmeans.fit_predict(X)

    # the expected result is an alternate sequence of 0 and 1
    print(labels)
