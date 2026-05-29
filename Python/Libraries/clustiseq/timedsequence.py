#!/bin/python3
# -*- coding: utf-8 -*-
"""
Timed sequence

@author: Thomas Guyet
@date: 05/2023
@institution: Inria
"""

import warnings
import numpy as np
from datetime import datetime as dt

## typing features
from typing import TypeVar, Union, Dict, Mapping, Tuple, Sequence, Any

#TimedSequence = TypeVar("clustiseq.timedsequence.TimedSequence")

TYPE_DELTATIME = 1
TYPE_FLOAT = 2


class TimedSequence:
    def __init__(
        self,
        dates: Union[Sequence[np.datetime64], Sequence[float]],
        data: Union[Sequence[str], Sequence[int]],
    ):
        """ """
        if len(dates) != len(data):
            raise ValueError("dates and data must have the same length")
        if not isinstance(dates, np.ndarray):
            raise ValueError("Dates must be a numpy.ndarray")

        if not np.issubdtype(dates.dtype, np.datetime64) and not dates.dtype == float and not dates.dtype == np.float32:
            raise ValueError("Dates elements must be of kind datetime64 or float")
        if np.issubdtype(dates.dtype, np.datetime64):
            self.dtype = TYPE_DELTATIME
        else:
            self.dtype = TYPE_FLOAT

        self._dates = dates
        self._data = np.array(data)
        # ensure that it is ordered by dates
        reorder = np.argsort(self._dates)
        self._dates = self._dates[reorder]
        self._data = self._data[reorder]
        self._pdata = None #hot-encoding of _data
        self._use_prob = False

    def __lt__(self, dt: Union[np.datetime64, float]) -> Sequence[bool]:
        if not isinstance(dt, np.datetime64) and not isinstance(dt, float):
            raise ValueError("Datetime/float expected")
        return self._dates.__lt__(dt)

    def __le__(self, dt: Union[np.datetime64, float]) -> Sequence[bool]:
        if not isinstance(dt, np.datetime64) and not isinstance(dt, float):
            raise ValueError("Datetime/float expected")
        return self._dates.__le__(dt)

    def __gt__(self, dt: Union[np.datetime64, float]) -> Sequence[bool]:
        if not isinstance(dt, np.datetime64) and not isinstance(dt, float):
            raise ValueError("Datetime/float expected")
        return self._dates.__gt__(dt)

    def __ge__(self, dt: Union[np.datetime64, float]) -> Sequence[bool]:
        if not isinstance(dt, np.datetime64) and not isinstance(dt, float):
            raise ValueError("Datetime/float expected")
        return self._dates.__ge__(dt)

    def __eq__(self, dt: Union[int, str, np.datetime64, float]) -> Sequence[bool]:
        if isinstance(dt, np.datetime64):
            return self._dates.__eq__(dt)
        elif isinstance(dt, str):
            return self._data.__eq__(dt)
        elif isinstance(dt, int):
            return self._data.__eq__(dt)
        elif isinstance(dt, float):
            return self._dates.__eq__(dt)
        raise ValueError("Datetime, str or int expected")

    def start(self) -> Union[np.datetime64, float]:
        return self._dates[0]

    def end(self) -> Union[np.datetime64, float]:
        return self._dates[-1]

    def len(self) -> int:
        return len(self._dates)

    def __len__(self) -> int:
        return self.len()

    def at(self, dt: Union[np.datetime64, float]) -> Union[int, str]:
        return self._data[self._dates.__eq__(dt)]

    def __str__(self, sep: str = "\n") -> str:
        if not self._use_prob:
            elems = []
            for k, v in zip(self._dates, self._data):
                elems.append(str(k) + ":" + str(v))
        else:
            elems = []
            for k, v in zip(self._dates, self._pdata):
                elems.append(str(k) + ":" + str(v))
        return sep.join(elems)

    def __getitem__(self, selection):
        if isinstance(selection, int):
            return (self._data[selection], self._dates[selection])
        else:
            return TimedSequence(self._dates[selection], self._data[selection])

    class iterator:
        def __init__(self, itdates, itdata):
            self.itdates = itdates
            self.itdata = itdata

        def __next__(self):
            d = self.itdates.__next__()
            e = self.itdata.__next__()
            return (e.item(), d.item())

    def __iter__(self):
        return TimedSequence.iterator(np.nditer(self._dates), np.nditer(self._data))
