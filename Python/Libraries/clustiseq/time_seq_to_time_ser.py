import numpy as np
import pandas as pd

class time_seq_to_time_ser:
    """
    Time sequence to time series converter
    Return the time serie corresponding to the time sequence according to the accuracy chosen.

    Parameters
    ----------
    list_seq : List of timedsequence
        Time sequence to convert.
    list_event : Dict or List
        List in the order from the less to the most important event.
        The first element of the list have to be " ".
    granularity : Int
        Number of day for each event to fill the serie.    
    """
    def __call__(self,seq,list_event,path,granularity = 1):
        if (type(list_event) is not dict): # change the type from list to dictionnary if needed
            list_event = {k:i for i, k in enumerate(list_event)}
        n = len(seq)
        res = pd.DataFrame()
        _size = 0
        _min = float('inf')
        _max = -float('inf')
        for i in range(n):
            if seq[i]._dates[0] < _min:
                _min = seq[i]._dates[0]
            if seq[i]._dates[-1] > _max:
                _max = seq[i]._dates[-1]
        a = np.int16(_max-_min+1)
        _size = np.int16(a/granularity)+1 # size of the serie
        _min = np.int16(np.floor(_min))
        for i in range(n):
            serie = np.full(_size,20,dtype = int) # initialize the new serie with previous size
            serie = list(serie)
            for j in range(len(seq[i])):
                ind = int((np.int16(np.floor(seq[i]._dates[j]))-_min)/granularity) # index in the serie of the begining of the ith element of the sequence
                data = list_event[seq[i]._data[j]]+1 # data of the ith element of the sequence
                if (ind < _size): # check if the index is in the serie
                    if (serie[ind] == 20) or (data > serie[ind]): # check the importance in case there are two elements in the same case
                        serie[ind] = data
            res = pd.concat([res,pd.DataFrame(serie)],axis = 1 ,ignore_index=True)
        res.transpose().to_csv(path,index=False) 
        # res.to_csv('series.csv',index=False)