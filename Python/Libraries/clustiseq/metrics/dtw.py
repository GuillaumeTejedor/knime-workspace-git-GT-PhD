"""
In this module, we define metrics between timed sequences that 
only consider the sequential nature of the data.
"""
import warnings
import numpy as np

from .event_metrics import em_set, em_base, em_hamming, em_euclidean
from clustiseq import TimedSequence

from typing import Sequence

__author__ = "Thomas Guyet thomas.guyet[at]inria.fr"


class euclidean:
    def __init__(self, em=em_base()):
        self.em = em

    def __call__(self, s1: TimedSequence, s2: TimedSequence) -> float:
        """Euclidean distance between the two sequences.

        Let :math:`S=\langle s_1, \dots, s_n\rangle` and
        :math:`T=\langle t_1, \dots, t_m\rangle` be two
        sequences. The Euclidean distance between two sequences is given by:

        .. math::

            euclidean(S,T)= \sqrt{\sum_{i=1}^{min(n,m)} d(s_i,t_i)^2}

        where :math:`d(s,t)` is a metric between two events. This metric can be
        set up in the parameters.

        Warning
        -------
        This metric is made for time sequences having the same length (same number
        of events). If it is not the case, then the end of the longest sequence is
        simply ignore.

        Parameters
        -----------
        s1, s2: TimedSequence
            Two timed sequences to compare.
        em:
            Event metric
        """
        return np.sqrt(np.sum([self.em(e1[0], e2[0]) ** 2 for e1, e2 in zip(s1, s2)]))


class lev:
    def __init__(self, em=em_hamming(), costs=(1, 1, None)):
        self.costs = costs
        self.em = em

    def __call__(self, s1: TimedSequence, s2: TimedSequence) -> float:
        """Levenshtein distance between two sequences

        The default behavior (when `costs[2]=None`) is to assign the distance
        value computed by `em` to the substitution. `em` must be definite (return 0
        if the two event types are the same)

        Note
        -----
        We suggest to use `em_hamming` as event type metric.

        Parameters
        ----------
        s1, s2: TimedSequence
            Two timed sequences to compare.
        em:
            Event metric
        costs: tuple
            Costs of insertion, deletion or substitution (typle in that order).
            Default is (1, 1, None).

        References
        ----------
        - `Wikipedia <https://en.wikipedia.org/wiki/Levenshtein_distance>`_
        """

        # creation of two lines of the distance matrix
        D = np.zeros((2, len(s2) + 1))

        # initial costs
        for t2 in range(len(s2) + 1):
            D[0][t2] = t2

        for t1 in range(1, len(s1) + 1):
            D[1][0] = t1
            for t2 in range(1, len(s2) + 1):
                diff = self.em(s1._data[t1 - 1], s2._data[t2 - 1])

                if self.costs[2] is not None and diff != 0:
                    diff = self.costs[2]

                D[1][t2] = np.min(
                    [
                        D[1][t2 - 1] + self.costs[0],  # insertion
                        D[0][t2] + self.costs[1],  # deletion
                        D[0][t2 - 1] + diff,
                    ]  # substitution
                )
            D[0][:] = D[1][:]
        return D[0][len(s2)]


class lcss:
    def __init__(self, em=em_hamming(), epsilon: float = 0, delta: float = None):
        """
        Parameters
        ----------
        em: event metric
            Event metric
        espilon: float
            Threshold to consider two event types as similar
        delta: None, float or np.TimeDelta64
            Maximum time decay between two event date to consider them as similar
        """
        self.em = em
        self.delta = delta
        self.epsilon = epsilon

    def lcss(self, s1: TimedSequence, s2: TimedSequence) -> float:
        """Longuest common subsequence

        The LCSS is a similarity measure that captures the longuest common sub-sequences.
        Two events are similar when `em(ev1,ev2)=0̀ : using an Hamming event metric and
        `\epsilon=0` means to have the exact same events.

        The constant :math:`\delta` enforces to match two events only if there are not too far
        in the timed sequence. If None, this constraint is ignored.

        Warning
        --------
        `delta` must be specified as a `np.TimeDelta64` when timed sequences are indexed with
        `np.Datetime64`.

        Parameters
        ----------
        s1, s2: TimedSequence
            Two timed sequences to compare.

        References
        ----------
        - `Wikipedia <https://en.wikipedia.org/wiki/Longest_common_subsequence>`_
        """

        # creation of two lines of the distance matrix
        D = np.zeros((2, len(s2) + 1))

        for t1 in range(1, len(s1) + 1):
            D[1, 0] = 0
            for t2 in range(1, len(s2) + 1):
                if self.em(s1._data[t1 - 1], s2._data[t2 - 1]) <= self.epsilon and (
                    self.delta is None
                    or abs(s1._dates[t1 - 1] - s2._dates[t2 - 1]) < self.delta
                ):
                    D[1, t2] = D[0, t2 - 1] + 1
                else:
                    D[1, t2] = max(D[0, t2], D[1, t2 - 1])
            D[0, :] = D[1, :]
        return D[0, len(s2)]

    def __call__(self, s1: TimedSequence, s2: TimedSequence) -> float:
        """
        Metric based on the LCSS similarity measure.

        The domain of this metric is [0,1]. It is null when one sequence is
        included in the other. It is 1.0 when there is no common event.

        Warning
        -------
        This metric is not a distance. The triangular inegality does not hold.

        Reference
        ---------
        :ref: lcss
        """
        return 1 - float(self.lcss(s1, s2)) / float(min(len(s1), len(s2)))


class dtw:
    def __init__(self, tc_param : float = None, sc_param : int =  None, em=em_hamming()):
        """
        Parameters
        ----------
        tc_param : Float
            Maximum time between two events for time_constraint. Default is None.
        sc_param : Int
            Number of comparisons you want to put at 'inf' in the corners of the matrix. Default is None.
        em: event metric
            Event metric for events.
        """
        self.vem = np.vectorize(em)

        if (sc_param != None) and (sc_param < 0): # warn if tc_param is a negative value and set it to 0 if it is the case
            warnings.warn(f"You entered a negative parameter, only positive parameters are allowed, sc_params has been set to 0.")
            sc_param = 0

        if (tc_param != None) and (tc_param < 0): # warn if tc_ param is a negative value
            warnings.warn(f"You chose a negative value for sc_param, it has been replaced by its absolute value")
            tc_param = abs(tc_param)

        self.tc_param=tc_param
        self.sc_param=sc_param

    def __call__(self, s1: TimedSequence, s2: TimedSequence, return_matrix : bool = False):
        """Dynamic Time Warping
        
        The Dynamic Time Warping is an algorithm used to measure the similarity 
        between two temporal sequences, which may vary in speed.
        Two events are similar when `em(ev1,ev2)=0̀. The Sakoe-Chiba and a time constraint can be used:

        * Sakoe-Chiba's DTW put at 'inf' the up-right and under-left corners of the matrix with the parameter sc_param.
        * Time Constraint DTW put at 'inf' all the comparisons between events to far in time with parameter tc_param.
        
        Parameters
        ----------
        s1, s2: TimedSequence
            Two timed sequences to compare.
        return_matrix : Boolean
            True to return the entire DTW matrix, False either. Default is False.
        References
        ----------
        - Wikipedia <https://en.wikipedia.org/wiki/Dynamic_time_warping>
        """

        n = len(s1._dates)
        m = len(s2._dates)

        if (self.sc_param != None) and (self.sc_param >= min(n,m)): # warn if tc_param is to high and set it to the maximum value if it is the case
            warnings.warn(f"Sakoe-Chiba parameter is too high and has been replaced by the maximum value.")
            sc_param = min(n,m)-1
        else:
            sc_param = self.sc_param
        
        if (sc_param != None):
            n_sc = n-sc_param
            m_sc = m-sc_param
        else:
            n_sc = n
            m_sc = m
        
        #use vectorize distance function to compute event distances
        C = self.vem( s1._data[np.arange(n)[:, np.newaxis]],s2._data[np.arange(m)] ) 

        DTW = np.full((n+1,m+1), float('inf')) # initialize distance matrix to infinite
        DTW[0,0] = 0
        for i in range(1,n+1): # compute the cost for each comparison of events and add the minimum between DTW[i-1,j], DTW[i,j-1] and DTW[i-1,j-1]
            for j in range(max(1,i-n_sc+1),min(m,i+m_sc-1)+1):
                if self.tc_param is None or (s1._dates[i-1]-s2._dates[j-1])^2 <= self.tc_param: #TODO TG: use the order of events to prune when tc is no more satisfied
                    DTW[i,j] = C[i-1,j-1] + min(DTW[i-1,j],DTW[i,j-1],DTW[i-1,j-1])

        if(DTW[n,m] == float('inf')): # warn when the last value is 'inf'
            warnings.warn(f"The value of tc_param seems not adapted to the situation. You should try with an higher one.")
        if (return_matrix == False): # return the matrix or the last value
            return DTW[n,m]
        else:
            return DTW

class drop_dtw:

    MODE_FULL=0
    MODE_PARTIAL=1

    def __init__(self, tc_param : float = None, sc_param : int =  None, drop_cost : float = None, 
                 t_weight : float = 1, ev_weight : float = 1, mode = MODE_FULL):
        """
        
        The Drop-Dynamic Time Warping is an algorithm used to measure the similarity 
        between two temporal sequences, which may vary in speed.
        The Sakoe-Chiba, a time constraint and the Drop-DTW can be used.
        Sakoe-Chiba's DTW put at 'inf' the up-right and under-left corners of the matrix with the parameter sc_param.
        Time Constraint DTW put at 'inf' all the comparisons between events to far in time with parameter tc_param.
        The Drop-DTW will put a value of a drop cost if the classic DTW value is too high.

        Parameters
        ----------
        tc_param : Float
            Maximum time between two events for time_constraint. Default is None.
        sc_param : Int
            Number of comparisons you want to put at 'inf' in the corners of the matrix. Default is None.
        drop_cost : Float[]
            A list of drop cost, setting as low will lead to high frequency of drop. Default is None.
        t_weight : Float
            Time weight in the distance
        ev_weight : Float
            Event weigth in the distance
        """

        if (sc_param != None) and (sc_param < 0): # warn if tc_param is a negative value and set it to 0 if it is the case
            warnings.warn(f"Only positive values are allowed for parameter sc_param, it has been set to 0.")
            sc_param = 0

        if (tc_param != None) and (tc_param < 0): # warn if tc_ param is a negative value
            warnings.warn(f"Only positive values are allowed for parameter sc_param, it has been replaced by its absolute value")
            tc_param = abs(tc_param)

        self.tc_param=tc_param
        self.sc_param=sc_param
        self.drop_cost=drop_cost
        self.t_weight=t_weight
        self.ev_weight=ev_weight
        self.__mode=mode

    def __call__(self, s1: TimedSequence, s2: TimedSequence, return_matrix : bool = False):
        if self.__mode==drop_dtw.MODE_FULL:
            return self.full(s1,s2,return_matrix)
        else:
            return self.partial(s1,s2,return_matrix)

    def partial(self, s1: TimedSequence, s2: TimedSequence, return_matrix : bool = False):
        """Drop-DTW (Drop Dynamic Time Warping)

        This version of the drop DTW align the sequence s1 to the sequence s2 allowing 
        to remove events in the sequence s2 (only). 
        See *full* function for removing events in both sequences.
        
        Parameters
        ----------
        s1, s2: TimedSequence
            Two timed sequences to compare.
        return_matrix : Boolean
            True to return the entire DTW matrix and the alignment path, False either. Default is False.

        Return
        -------
        * float: Drop-DTW value in case
        * DTW matrix and Path 
            
        Warning
        --------
        This function assumes that the events are distributions of event types (with same dimensions), or are integer events (not strings).
        Otherwise, the **_pdata of the sequences are modified**

        References
        ----------
        - "Drop-DTW: Aligning Common Signal Between Sequences While Dropping Outliers", Nikita Dvornik, Isma Hadji, Konstantinos Derpanis, Animesh Garg, Allan D. Jepson
        """
        print("s1 =", s1)
        print("s2 =", s2)
        # Compute the onehot encoding version of the data
        #   The two sequences encodings must share the same dimension
        if not (s1._pdata is not None and not s2._pdata is None and s1._pdata.shape[1]==s2._pdata.shape[1]):
            onehotdim= max(s1._data.max(), s2._data.max())
            if s1._pdata is None or s1._pdata.shape[1]!=onehotdim:
                s1._pdata = np.zeros( (s1._data.size, onehotdim + 1) )
                s1._pdata[np.arange(s1._data.size), s1._data] = 1
            if s2._pdata is None or s2._pdata.shape[1]!=onehotdim:
                s2._pdata = np.zeros( (s2._data.size, onehotdim + 1) )
                s2._pdata[np.arange(s2._data.size), s2._data] = 1

        print("s1._pdata =", s1._pdata)
        print("s2._pdata =", s2._pdata)

        n = len(s1._dates)
        m = len(s2._dates)

        print("(m, n) =", (m, n))

        if (self.sc_param != None) and (self.sc_param >= min(n,m)): # warn if tc_param is to high and set it to the maximum value if it is the case
            warnings.warn(f"Sakoe-Chiba parameter is too high. It has been replaced by the maximum value.")
            sc_param = min(n,m)-1
        else:
            sc_param = self.sc_param
        
        if (sc_param != None):
            n_sc = n-sc_param
            m_sc = m-sc_param
        else :
            n_sc = n
            m_sc = m

        D_plus = np.full((n+1,m+1), float('inf')) # initialize distance matrix to infinite
        D_plus[0,0]=0.0
        D_minus = np.full((n+1,m+1), float('inf')) # initialize distance matrix to infinite
        D_minus[0,0]=0.0

        if (self.drop_cost is None):
            drop_cost = float("inf")
        else:
            D_minus[0,1:] = (np.arange(m)+1)*self.drop_cost
            drop_cost=self.drop_cost

        print("D_plus =", D_plus)
        print("D_minus =", D_minus)
    
        DTW = D_minus.copy()
        if return_matrix:
            PathMatrix = np.full((n+1,m+1,3), np.nan, dtype=np.int16) # initialize path matrix to NaN

        print("s1._dates =", s1._dates)
        print("s2._dates =", s2._dates)
        C_date = np.vectorize(lambda x,y:(x-y)**2)(s1._dates[:, np.newaxis], s2._dates )
        print("C_date =", C_date)

        print("s1._pdata =", s1._pdata)
        print("s2._pdata =", s2._pdata)
        C_event = np.apply_along_axis(np.sum,-1, np.vectorize(lambda x,y:(x-y)**2)(s1._pdata[:,np.newaxis], s2._pdata[np.newaxis,:] )) 
        print("C_event =", C_event)

        C = np.abs(self.t_weight*C_date + self.ev_weight*C_event)

        print("C =", C)

        for i in range(1,n+1): # compute the cost for each comparison of events and add the minimum between DTW[i-1,j], DTW[i,j-1] and DTW[i-1,j-1]
            for j in range(max(1,i-n_sc+1),min(m,i+m_sc-1)+1):
                if self.tc_param is None or C_date[i-1,j-1] <= self.tc_param:
                    if return_matrix:
                        if DTW[i-1, j-1]< DTW[i, j-1]:
                            a = DTW[i-1, j-1]
                            p= (i-1, j-1, 0)
                        else:
                            a = DTW[i, j-1]
                            p= (i, j-1, 0)
                        if D_plus[i-1, j] < a:
                            a = D_plus[i-1, j]
                            p= (i-1, j, 0)
                        D_plus[i,j]=C[i-1,j-1] + a
                        D_minus[i,j]=drop_cost + DTW[i, j-1]
                        if D_plus[i,j]<D_minus[i,j]:
                            DTW[i,j] = D_plus[i,j]
                            PathMatrix[i,j] = p
                        else: #drop
                            DTW[i,j] = D_minus[i,j]
                            PathMatrix[i,j] = (i, j-1, 1)
                    else:
                        D_plus[i,j] = C[i-1,j-1] + min( DTW[i-1, j-1], DTW[i, j-1], D_plus[i-1, j] ) 
                        D_minus[i,j] = drop_cost + DTW[i, j-1]
                        DTW[i,j] = min(D_plus[i,j],D_minus[i,j])
                elif C_date[i-1,j-1] > self.tc_param and j>i: #prune when tc is no more satisfied (assume that events are temporaly ordered)
                    break

        if(DTW[n,m] == float('inf')): # warn if the last value is 'inf'
            warnings.warn(f"tc_param or sc_param value is not adapted to the situation, please try with an higher one.")
        if (return_matrix == False): # return the matrix or the last value
            return DTW[n,m]
        else:
            #reconstruct the path, with indices starting with 0
            Path = []
            pos = (n,m)
            while pos[0]!=0 and pos[1]!=0:
                if PathMatrix[pos][2]==0: #if not dropped
                    Path.append( (pos[0]-1, pos[1]-1) )
                pos = tuple( PathMatrix[pos][:2] )
            Path.reverse()
            return DTW, Path
        

    def full(self, s1: TimedSequence, s2: TimedSequence, return_matrix : bool = False):
        """Drop-DTW (Drop Dynamic Time Warping)

        Compute the dissimilarity with event dropping on both s1 and s2.
        
        Parameters
        ----------
        s1, s2: TimedSequence
            Two timed sequences to compare.
        return_matrix : Boolean
            True to return the entire DTW matrix and the alignment path, False either. Default is False.

        Return
        -------
        * float: Drop-DTW value in case
        * DTW matrix and Path 
            
        Warning
        --------
        This function assumes that the events are distributions of event types (with same dimensions), or are integer events (not strings).
        Otherwise, the **_pdata of the sequences are modified**

        References
        ----------
        - "Drop-DTW: Aligning Common Signal Between Sequences While Dropping Outliers", Nikita Dvornik, Isma Hadji, Konstantinos Derpanis, Animesh Garg, Allan D. Jepson
        """

        # Compute the onehot encoding version of the data
        #   The two sequences encodings must share the same dimension
        if not (s1._pdata is not None and not s2._pdata is None and s1._pdata.shape[1]==s2._pdata.shape[1]):
            onehotdim= max(s1._data.max(), s2._data.max())
            if s1._pdata is None or s1._pdata.shape[1]!=onehotdim:
                s1._pdata = np.zeros( (s1._data.size, onehotdim + 1) )
                s1._pdata[np.arange(s1._data.size), s1._data] = 1
            if s2._pdata is None or s2._pdata.shape[1]!=onehotdim:
                s2._pdata = np.zeros( (s2._data.size, onehotdim + 1) )
                s2._pdata[np.arange(s2._data.size), s2._data] = 1

        n = len(s1._dates)
        m = len(s2._dates)

        if (self.sc_param != None) and (self.sc_param >= min(n,m)): # warn if tc_param is to high and set it to the maximum value if it is the case
            warnings.warn(f"Sakoe-Chiba parameter is too high. It has been replaced by the maximum value.")
            sc_param = min(n,m)-1
        else:
            sc_param = self.sc_param
        
        if (sc_param != None):
            n_sc = n-sc_param
            m_sc = m-sc_param
        else :
            n_sc = n
            m_sc = m

        D_zx = np.full((n+1,m+1), float('inf')) # initialize distance matrix to infinite
        D_zminus = np.full((n+1,m+1), float('inf')) # initialize distance matrix to infinite
        D_minusx = np.full((n+1,m+1), float('inf')) # initialize distance matrix to infinite
        D_minus = np.full((n+1,m+1), float('inf')) # initialize distance matrix to infinite
        D_zx[0,0]=D_zminus[0,0]=D_minusx[0,0]=D_minus[0,0]=0

        if (self.drop_cost is None):
            drop_cost = float("inf")
        else:
            D_zminus[0,1:] = (np.arange(m)+1)*self.drop_cost
            D_minusx[1:,0] = (np.arange(n)+1)*self.drop_cost
            D_minus[0,1:] = (np.arange(m)+1)*self.drop_cost
            D_minus[1:,0] = (np.arange(n)+1)*self.drop_cost
            drop_cost=self.drop_cost
        DTW = D_minus.copy()
        if return_matrix:
            PathMatrix = np.full((n+1,m+1,3), np.nan, dtype=np.int16) # initialize path matrix to NaN

        C_date = np.vectorize(lambda x,y:(x-y)**2)(s1._dates[:, np.newaxis], s2._dates )
        C_event = np.apply_along_axis(np.sum,-1, np.vectorize(lambda x,y:(x-y)**2)(s1._pdata[:,np.newaxis], s2._pdata[np.newaxis,:] ))
        C = np.abs(self.t_weight*C_date + self.ev_weight*C_event)

        for i in range(1,n+1): # compute the cost for each comparison of events and add the minimum between DTW[i-1,j], DTW[i,j-1] and DTW[i-1,j-1]
            for j in range(max(1,i-n_sc+1),min(m,i+m_sc-1)+1):
                if self.tc_param is None or C_date[i-1,j-1] <= self.tc_param:
                    D_zx[i,j]= C[i-1,j-1] +  min( D_zx[i-1,j-1], D_zminus[i-1,j-1], D_minusx[i-1,j-1], D_minus[i-1,j-1])
                    D_zminus[i,j] = drop_cost + min(D_zx[i,j-1], D_zminus[i,j-1])
                    D_minusx[i,j] = drop_cost + min(D_zx[i-1,j], D_minusx[i-1,j])
                    
                    if return_matrix:
                        tp = np.argmin([D_minusx[i,j-1], D_minus[i,j-1], D_zminus[i-1,j], D_minus[i-1,j]])
                        D_minus[i,j] = drop_cost + min(D_minusx[i,j-1], D_minus[i,j-1], D_zminus[i-1,j], D_minus[i-1,j])
                        vals=[D_zx[i,j],D_zminus[i,j],D_minusx[i,j],D_minus[i,j]]
                        mp=np.argmin(vals)
                        DTW[i,j] = vals[mp]
                        if mp==0:
                            PathMatrix[i,j] = (i-1, j-1, 0)
                        elif mp==1:
                            PathMatrix[i,j] = (i, j-1, 1)
                        elif mp==2:
                            PathMatrix[i,j] = (i-1, j, 1)
                        else:
                            if tp<=1:
                                PathMatrix[i,j] = (i, j-1, 1)
                            else:
                                PathMatrix[i,j] = (i-1, j, 1)
                    else:
                        D_minus[i,j] = drop_cost + min(D_minusx[i,j-1], D_minus[i,j-1], D_zminus[i-1,j], D_minus[i-1,j])
                        DTW[i,j]=min(D_zx[i,j],D_zminus[i,j],D_minusx[i,j],D_minus[i,j])
                    
                elif C_date[i-1,j-1] > self.tc_param and j>i: #prune when tc is no more satisfied (assume that events are temporaly ordered)
                    break

        if(DTW[n,m] == float('inf')): # warn if the last value is 'inf'
            warnings.warn(f"tc_param or sc_param value is not adapted to the situation, please try with an higher one.")
        if (return_matrix == False): # return the matrix or the last value
            return DTW[n,m]
        else:
            #reconstruct the path, with indices starting with 0
            Path = []
            pos = (n,m)
            while pos[0]!=0 and pos[1]!=0:
                if PathMatrix[pos][2]==0: #if not dropped
                    Path.append( (pos[0]-1, pos[1]-1) )
                pos = tuple( PathMatrix[pos][:2] )
            Path.reverse()
            return DTW, Path
        

    def average(self, sequences: Sequence[TimedSequence], itmax: int=1, preparesequences: bool = True) -> TimedSequence:
        """
        Parameters
        -----------
        sequences: list[TimedSequences]
            List of timed sequences

        Return:
            An average timed sequence 
        """

        #find largest sequences
        largests=[]
        maxlen=0
        vmax=0
        for s in sequences:
            if len(s)>maxlen:
                largests=[s]
            elif len(s)==maxlen:
                largests.append(s)
            lmax= np.max(s._data)
            vmax = max(lmax, vmax)
        sr = largests[ np.random.choice(np.arange( len(largests) )) ] #random choice among the largest sequences
        
        if preparesequences:
            #create distributions with all the same dimensions
            for s in sequences:
                s._pdata = np.zeros( (s._data.size, vmax + 1) )
                s._pdata[np.arange(s._data.size), s._data] = 1
        else:
            assert(len(sequences)>0 and sequences[0]._pdata is not None)
            vmax = sequences[0]._pdata.shape[1]-1

        for _ in range(itmax): #repetition of the refinement process
            E = np.full( (len(sr),vmax+1), 0, dtype=np.float32)
            T = np.full( len(sr), 0, dtype=np.float32)
            N = np.full( len(sr), 0, dtype=np.float32)

            for s in sequences:
                # compute the Drop-DTW and get the path
                _, p = self.__call__(sr, s, True)
                for pos in p:
                    E[ pos[0], : ] += s._pdata[ pos[1] ]
                    T[ pos[0] ] += s._dates[ pos[1] ]
                    N[ pos[0] ] += 1
                    
            #remove empty events
            E=E[ N>0, :]
            T=T[ N>0 ]
            N=N[ N>0 ]
            #normalize
            E[ :, : ] /= N[:,np.newaxis]
            T /= N
            #create the new mean sequence
            sr=TimedSequence(T, np.full(len(N),0))
            sr._use_prob=True
            sr._pdata = E
        return sr

