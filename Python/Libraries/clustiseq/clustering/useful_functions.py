import numpy as np
import pandas as pd


from clustiseq.timedsequence import TimedSequence
from clustiseq.metrics import dtw


def csv_to_pd(csv,delimiter = ';',object = ['num_patient','event_name','difftime']):
    """CSV file to Pandas Dataframe
    Return a pandas dataframe from the CSV file. This dataframe is the one used in the clustering.

    Parameters
    ----------
    csv : Str
        access path of the csv file with the list of events.
    delimiter : Str
        delimiter used in the file. Default is ;.
    object : Str[]
        name of the columns to use in the clustering. Default is ['num_patient','event_name','difftime'].
    """
    df = pd.read_csv(r''+csv, delimiter = delimiter) # read the original dataframe
    df = df.sort_values(object[2]) # sort values by time
    data_set_pds = [list(p_df[[object[1], object[2]]].itertuples(index=False, name=None)) for p_id, p_df in df.groupby(object[0])] # extract for each patient_num tuples with the events and their times

    return data_set_pds,df

