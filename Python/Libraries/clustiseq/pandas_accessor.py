"""

@author: Thomas Guyet
@date: 05/2023
@institution: Inria
"""


import pandas as pd
from pandas.api.extensions import register_dataframe_accessor
import numpy as np

from clustiseq.timedsequence import TimedSequence
from clustiseq.metrics.dtw import euclidean

from scipy.sparse import csr_matrix
from sklearn.cluster import KMeans

## typing features
from typing import TypeVar, Union, Dict, Mapping, Tuple, Sequence, Any


@register_dataframe_accessor("tpattern")
class TPatternAccessor:
    """Pandas accessor

    This class registers a new extension keyword to manipulate pandas
    dataframe with personal function.
    """

    def __init__(self, df: pd.DataFrame):
        self._validate(df)
        self._df = df

    @staticmethod
    def _validate(df):
        # verify there is a no MultiIndex, and that the Index is made of Integers or Timestamps
        if isinstance(df.index, pd.MultiIndex):
            raise AttributeError("Can not handle multi-indexed dataframes.")
        if (
            df.index.dtype != float
            and df.index.dtype != int
            and df.index.dtype != np.dtype("datetime64[ns]")
        ):
            raise AttributeError("Dataframe index has to be convertible in float.")

    @staticmethod
    def __TSExtractor__(df: pd.DataFrame, event):
        """
        df: pandas.dataframe
            dataframe indexed with time
        event:
            name of the column
        """
        dates = df.index.to_numpy()
        if dates.dtype == "int":
            dates = dates.astype("float")
        data = df[event].tolist()

        return TimedSequence(dates, data)

    def clustering(
        self, k: int, event: str = None, groupby: str = None, metric=euclidean()
    ):
        """
        Parameters
        ----------
        k: int
            Number of clusters to create
        event: str
            Name of the dataframe column to use as event (must contains integers or str)
        groupby: str or None
            name of the column used to identify events belonging to the same sequence. If None, the dataset is assumed to describe a unique sequence.
        metric:
            definition of the metric (dissimilarity) to compare sequences

        Warning
        -------
        An definition of the sequence identifier is required. It must be done through the use of
        a multiple index or through the definition or a grouping attribute (`groupby`). Both can
        not be set.
        """
        if isinstance(self._df.index, pd.MultiIndex) and (groupby is not None):
            raise AttributeError(
                "Multiple index is not compatible with the explicite definition of a grouping attribute."
            )
        elif not isinstance(self._df.index, pd.MultiIndex) and (groupby is None):
            raise AttributeError(
                "A grouping attribute or an multiple index is missing."
            )

        if event == groupby:
            raise AttributeError(
                "The grouping attribute and the event attribute can not be the same."
            )

        # collect the TimedSequences from the dataframe
        if groupby is None:
            if self._df.index._is_multi:
                groupby = self._df.index.names[0]
                tss = (
                    self._df.groupby(groupby)
                    .apply(
                        lambda d: TPatternAccessor.__TSExtractor__(
                            d.droplevel(0), event
                        )
                    )
                    .tolist()
                )
            else:
                tss = [TPatternAccessor.__TSExtractor__(self._df, event)]
        else:
            tss = (
                self._df.groupby(groupby)
                .apply(lambda d: TPatternAccessor.__TSExtractor__(d, event))
                .tolist()
            )

        row = []
        col = []
        data = []
        for i in range(len(tss)):
            for j in range(i + 1, len(tss)):
                row.append(i)
                col.append(j)
                data.append(metric(tss[i], tss[j]))
        X = csr_matrix((data, (row, col)), shape=(len(tss), len(tss)))

        kmeans = KMeans(k)
        labels = kmeans.fit_predict(X)

        # the expected result is an alternate sequence of 0 and 1
        return labels


if __name__ == "__main__":
    #################################
    # Example of sequence
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

    df = pd.DataFrame(
        {
            "label": [e[0] for e in seq],
            "str_val": [
                e[0] * 2 for e in seq
            ],  # illustration of another columns than "label"
            "num_val": np.random.randint(
                10, size=len(seq)
            ),  # illustration of another columns than "label"
        },
        index=[np.datetime64("1970-01-01") + np.timedelta64(e[1], "D") for e in seq],
    )
    print("----------------")

    ##########################################################################
    # Use with a dataframe representing a collection of sequences

    # Create a dataframe representing several sequences with complex events,
    # each sequence having its own id
    grpdf = pd.DataFrame(
        {
            "label": [e[0] for e in seq] * 3,
            "str_val": [e[0] * 2 for e in seq]
            * 3,  # illustration of another columns than "label"
            "num_val": np.random.randint(
                10, size=3 * len(seq)
            ),  # illustration of another columns than "label"
            "id": [1] * len(seq) + [2] * len(seq) + [3] * len(seq),
        },
        index=[np.datetime64("1970-01-01") + np.timedelta64(e[1], "D") for e in seq]
        * 3,
    )

    print(grpdf)

    labels = grpdf.tpattern.clustering(2, event="num_val", groupby="id")
    print(labels)
