from snorkel.labeling.model import LabelModel
from snorkel.labeling import labeling_function, PandasLFApplier
from snorkel.labeling.apply.dask import DaskLFApplier

from pyspark.sql import SparkSession
from snorkel.labeling.apply.spark import SparkLFApplier
import os


from utils import get_quartiles
import numpy as np
import dask.dataframe as dd
from dask import delayed, compute
import pandas as pd
from scipy import sparse


TOGETHER = 0
SEPARATED = 1
ABSTAIN = -1

def label_pairs(df_train, lfs_names):
    """
    Train Snorkel on pairs of patients.

    :param df_train: Pandas DataFrame of pairs with features.
    :param lfs_names: List of labeling function names.
    :return:
        - df_train with predicted labels
        - conditional probabilities
    """
    global df_global
    df_global = df_train.copy()
    # Create labeling functions
    lfs = [] 
    for lf_name in lfs_names:
        lf = make_quartile_lf(lf_name) 
        lfs.append(lf)

    lfs = [make_quartile_lf(name) for name in lfs_names]

    # Apply labeling functions (returns dense NumPy array)
    applier = PandasLFApplier(lfs)
    L_train = applier.apply(df_global)

    # Ensure dense NumPy array (important)
    L_train = np.asarray(L_train)

    # Train LabelModel (must be dense)
    label_model = LabelModel()
    label_model.fit(L_train, seed=42)

    # Predict
    df_train["LABEL"] = label_model.predict(
        L=L_train,
        tie_break_policy="abstain"
    )

    cond_proba = label_model.get_conditional_probs()

    return df_train, cond_proba

def label_pairs_dask(df_train, lfs_names):
    """
    Train Snorkel on pairs of patients to label them.

    :param df_train: Training dataframe of pairs that contains features to train Snorkel.
    :param lfs_names: List of labeling function names.
    :return: Training dataframe with predicted labels and 3D numpy array with conditional probabilities of predicted labels.
    """

    global df_global

    # --- Create labeling functions ---
    lfs = [make_quartile_lf(lf_name) for lf_name in lfs_names]

    # --- Copy dataframe ---
    df_global = df_train.copy()

    # --- Convert pandas DataFrame to Dask DataFrame ---
    df_global_dask = dd.from_pandas(df_global, npartitions=40)  # adjust npartitions as needed

    # --- Apply labeling functions ---
    applier = DaskLFApplier(lfs)
    L_train = applier.apply(df_global_dask)  # returns a NumPy array

    # --- Convert to sparse matrix for memory efficiency ---
    L_train_sparse = sparse.csr_matrix(L_train)

    # --- Train Snorkel LabelModel ---
    label_model = LabelModel()
    label_model.fit(L_train_sparse.toarray(), seed=42)  # LabelModel.fit expects dense array

    # --- Predict labels ---
    df_train["LABEL"] = label_model.predict(L=L_train_sparse.toarray(), tie_break_policy="abstain")

    # --- Get conditional probabilities ---
    cond_proba = label_model.get_conditional_probs()

    return df_train, cond_proba

def label_pairs_spark(df_train, lfs_names):
    """
    Train Snorkel on pairs of patients using Spark for LF application,
    with a custom temp folder to avoid PermissionError on Windows.
    * It doesn't work for now ... *
    """

    global df_global
    df_global = df_train.copy()

    # --- Create Spark session ---
    spark = SparkSession.builder \
        .appName("SnorkelSparkLabeling") \
        .master("local[*]") \
        .getOrCreate()

    # --- Create labeling functions ---
    lfs = [make_quartile_lf(name) for name in lfs_names]

    # --- Convert pandas DataFrame to Spark DataFrame ---
    sdf = spark.createDataFrame(df_global)

    # --- Convert Spark DataFrame to RDD ---
    rdd = sdf.rdd

    # --- Apply labeling functions using SparkLFApplier ---
    applier = SparkLFApplier(lfs)
    L_train = applier.apply(rdd)  # returns NumPy array

    # --- Convert to sparse matrix for memory efficiency ---
    L_train_sparse = sparse.csr_matrix(L_train)

    # --- Train Snorkel LabelModel ---
    label_model = LabelModel()
    label_model.fit(L_train_sparse.toarray(), seed=42)

    # --- Predict labels ---
    df_train["LABEL"] = label_model.predict(L=L_train_sparse.toarray(), tie_break_policy="abstain")

    # --- Get conditional probabilities ---
    cond_proba = label_model.get_conditional_probs()

    return df_train, cond_proba

def make_quartile_lf(feature_name):

    """
    Create labeling functions to provide intermediate label for each pair of patients.
    :param feature_name: Name of the features
    :return: Label that state if pair of patients should be clustered together (TOGETHER), not together (SEPARATED) or remain undertemined (ABSTAIN). 
    """

    @labeling_function(name=f"lf_{feature_name}")
    def lf(x):
        """
        Label a given pair of patients if it should be clustered together of not.
        :param x: Pandas Series that contains descriptive variables values for a corresponding pair of patients.
        :return: Label that state if pair of patients should be clustered together (TOGETHER), not together (SEPARATED) or remain undertemined (ABSTAIN). 
        """
        #print("--------------------------------------------------")
        #print("feature_name=",feature_name)
        #print("x shape =", x.shape)
        #print("x=",x)
        #print("df_global shape=", df_global.shape)
        #print("df_global=",df_global)
        feature = x[feature_name]
        #print("Variable called: ", feature_name)
        q1, _, q3 = get_quartiles(df_global[feature_name].dropna())
        #print("(q1, q3)=",(q1,q3))
        #print("feature=",feature)
        #print(type(feature))
        if feature is None or pd.isna(feature):
            return ABSTAIN
        if feature <= q1:
            return TOGETHER
        if (q1 < feature < q3):
            return ABSTAIN
        if feature >= q3:
            return SEPARATED

    return lf