from sklearn.metrics import silhouette_samples
from sklearn.metrics import calinski_harabasz_score
from sklearn.neighbors import NearestNeighbors
from collections import Counter
import itertools
import numpy as np
import pandas as pd
from lifelines.statistics import pairwise_logrank_test
import statistics

def internal(X, cluster_labels, metric="euclidean", n_iter_hopkins=100, snorkel_labels=None):

    """
    Evaluate clusters by differents internal quality metrics.
    :param X: 2D array that represent either features for each sample or a measure matrix.
    :param cluster_labels: 1D array of clusters labels for each patient.
    :param metric: The metric to use when calculating distance between instances in a feature array.
    :param n_iter_hopkins: Number of times to compute Hopkins statistic before to compute avg and std.
    :param snorkel_labels: 1D array of snorkel labels for each pair of patients.
    :return: silhouette avg, silhouette std, hopkins avg, hopkins std, clusters distribution
    """
    m = int(np.ceil(0.1 * len(X)))
    max_value = np.max(X)
    hopkins_values = [compute_hopkins(X, m=m, low=0, high=max_value, metric=metric) for i in range(n_iter_hopkins)]

    silhouette_avg, silhouette_std = compute_silhouette(X, cluster_labels, metric=metric)
    hopkins_avg, hopkins_std = np.average(hopkins_values), np.std(hopkins_values)
    entropy, entropy_normalized = compute_clusters_distribution(cluster_labels)
    
    if metric=="euclidean":
        ch_score = calinski_harabasz_score(X, cluster_labels)
    else:
        ch_score = -1.0

    if snorkel_labels is not None:
        matching_percentage = compute_matching_percentage(snorkel_labels, cluster_labels)
    else:
        matching_percentage = -1.0

    return silhouette_avg, silhouette_std, ch_score, hopkins_avg, hopkins_std, entropy_normalized, matching_percentage

def compute_matching_percentage(snorkel_labels, cluster_labels):
    """
    Compute percentage of matching pairs of patients between Snorkel and clustering method 
    (i.e. are pairs together based on Snorkel when they are clustered together).

    :param snorkel_labels: 1D array of snorkel labels for each pair of patients.
    :param cluster_labels: 1D array of clusters labels for each patient.
    :return: Percentage of matching pairs between Snorkel and clustering algorithm.
    """

    # Prepare labels of pairs to specify if patients are in same cluster or not
    n = len(cluster_labels)
    pairs = list(itertools.combinations(range(n), 2))
    pairwise_list = [(i, j, int(cluster_labels.iloc[i] != cluster_labels.iloc[j])) for i, j in pairs]
    pairwise_array = np.array(pairwise_list)
    cluster_label_pairs = pairwise_array[:,-1]
    
    labels_state_clustering = cluster_label_pairs[snorkel_labels != -1]
    labels_state_snorkel = snorkel_labels[snorkel_labels != -1]

    matching_count = np.sum(labels_state_clustering == labels_state_snorkel)
    total_labels = labels_state_snorkel.size
    matching_percentage = (matching_count / total_labels) * 100
    return matching_percentage

def compute_silhouette(X, cluster_labels, metric="euclidean"):
    
    """
    Compute silhouette average and standard deviation from a 2D 
    array where each sample is labeled.

    :param X: 2D array that represent either features for each sample or a precomputed matrix.
    :param cluster_labels: 1D array of cluster cluster_labels.
    :param metric: The metric to use when calculating distance between instances in a feature array.
    :return: silhouette average, silhouette std
    """

    silhouette_values = silhouette_samples(X, cluster_labels, metric=metric)

    silhouette_avg = np.mean(silhouette_values)
    silhouette_std = np.std(silhouette_values)

    return silhouette_avg, silhouette_std

def compute_hopkins(X, m=5, low=0, high=1, metric="minkowski"):

    """
    Compute Hopkins statistic.

    :param X: 2D array that represent either features for each sample or a measure matrix.
    :param m: Integer that represent the number of data points to create for uniform random generation. 
    It has to be lesser than length of X.
    :param low: Real value that represent minimum thresold for uniform random values generation.
    :param max: Real value that represent maximum thresold for uniform random values generation.
    :param metric: Metric used to compute separation between individuals.
    :return: Hopkins statistic
    """

    if metric=="precomputed":

        # Define the number of additional rows/columns
        n_new = m
        #print("n_new=",n_new)
        # Generate new indexes
        new_indexes = [-(i+1) for i in range(n_new)]
        all_indexes = X.index.tolist() + new_indexes
        #print("X=",X)
        #print("X_indexes=", X.index.tolist())
        #print("X_all_indexes=", all_indexes)
        # Expand the DataFrame to include new rows and columns
        expanded_X = X.reindex(index=all_indexes, columns=all_indexes, fill_value=np.nan)
        #print("expanded_X=", expanded_X)
        # Fill new cells with random values between low and high in a symmetric way
        for new_label in new_indexes:
            for existing_label in all_indexes:
                if new_label != existing_label:
                    random_value = np.random.uniform(low=low, high=high)
                    expanded_X.loc[new_label, existing_label] = random_value
                    expanded_X.loc[existing_label, new_label] = random_value
        #print("expanded_X_filled=", expanded_X)
        # Set diagonal to 0 to maintain distance matrix properties
        np.fill_diagonal(expanded_X.values, 0)
                        
        indexes = expanded_X.index
        negative_indexes = indexes[indexes < 0].tolist()
        positive_indexes = indexes[indexes >= 0].tolist()
        # Compute u_distances
        Y = expanded_X.drop(positive_indexes, axis=0).drop(negative_indexes, axis=1)
        #print("Y=",Y)
        u_distances = np.array(Y.min(axis=1))
        #print("u_distances=",u_distances)
        # Compute w_distances
        #print("X=",X)
        
        n, d = X.shape
        #print("(n,m)=",(n,m))
        random_indices = np.random.choice(n, size=m, replace=False)
        #print("random_indices=", random_indices)
        X_tilt_indexes = X.index[random_indices]
        #print("X_tilt_indexes=", X_tilt_indexes)
        X_tilt = X.loc[X_tilt_indexes].drop(X_tilt_indexes, axis=1)
        #print("X_tilt=", X_tilt)
        w_distances = np.array(X_tilt.min(axis=1))
        #print("w_distances=", w_distances)
    else: # X is a matrix with n samples and is d dimensional
        X = np.array(X)
        #print("test")
        print(X)
        n, d = X.shape
        #print("test")
        random_indices = np.random.choice(n, size=m, replace=False)
        X_tilt = X[random_indices]
        X_remaining = np.delete(X, random_indices, axis=0)
        Y = np.random.uniform(low=low, high=high, size=(m,d))
        # Compute u_distances
        neigh_u = NearestNeighbors(n_neighbors=1, metric=metric).fit(X)
        u_distances, _ = neigh_u.kneighbors(Y)
        # Compute w_distances
        neigh_y = NearestNeighbors(n_neighbors=1, metric=metric).fit(X_remaining)
        w_distances, _ = neigh_y.kneighbors(X_tilt)

    H = np.sum(u_distances) / (np.sum(u_distances) + np.sum(w_distances))

    return H

def compute_clusters_distribution(cluster_labels):

    """
    :param cluster_labels: 1D array of cluster cluster_labels.
    :return: Raw entropy, Normalized entropy.
    """

    counts = Counter(cluster_labels)
    patient_counts = np.array(list(counts.values()))
    #print("Patient count: ", patient_counts)

    nb_patients_per_cluster = patient_counts
    nb_clusters = len(nb_patients_per_cluster)
    proba_per_cluster = [x/sum(nb_patients_per_cluster) for x in nb_patients_per_cluster]
    #print("Proba per cluster:", proba_per_cluster)
    #nb_patients_per_cluster/sum(nb_patients_per_cluster)
    entropy = -np.sum(proba_per_cluster * np.log2(proba_per_cluster))
    entropy_normalized = entropy/np.log2(nb_clusters)
    
    return entropy, entropy_normalized

def compute_pairwise_logrank(df, cluster_label_columns, duration_col="EVOLUTION_DURATION", event_col="IS_UNCENSORED"):
    """
    Computes pairwise Log-Rank and Wilcoxon tests for multiple clustering label columns.

    Parameters:
        df (pd.DataFrame): Input DataFrame containing survival data and cluster labels.
        cluster_label_columns (list): List of column names for cluster labels.
        duration_col (str): Column containing survival duration.
        event_col (str): Column containing event occurrence (1 = event, 0 = censored).

    Returns:
        pd.DataFrame: Sorted summary DataFrame with log-rank and Wilcoxon test results.
    """
    df_logrank = pd.DataFrame(index=cluster_label_columns)

    durations = np.array(df[duration_col])
    events = np.array(df[event_col])

    for cluster_label_column in cluster_label_columns:
        groups = np.array(df[cluster_label_column])

        # Log-Rank test
        result_logrank = pairwise_logrank_test(durations, groups, events)
        statistic_min_log = np.min(result_logrank.summary.test_statistic.tolist())
        pval_max_log = np.max(result_logrank.summary.p)
        print("result_logrank=",result_logrank.summary)

        # Wilcoxon (Breslow) test
        result_wilcoxon = pairwise_logrank_test(durations, groups, events, weightings="wilcoxon")
        statistic_min_wil = np.min(result_wilcoxon.summary.test_statistic.tolist())
        pval_max_wil = np.max(result_wilcoxon.summary.p)

        # Store results
        df_logrank.loc[cluster_label_column, "LOGRANK-PVAL_MAX"] = pval_max_log
        df_logrank.loc[cluster_label_column, "LOGRANK-STAT_MIN"] = statistic_min_log
        df_logrank.loc[cluster_label_column, "WILCOXON-PVAL_MAX"] = pval_max_wil
        df_logrank.loc[cluster_label_column, "WILCOXON-STAT_MIN"] = statistic_min_wil

    # Sort and format output
    df_logrank_sorted = df_logrank.sort_values(by="LOGRANK-PVAL_MAX", ascending=True)
    df_logrank_sorted["LOGRANK-PVAL_MAX"] = df_logrank_sorted["LOGRANK-PVAL_MAX"].apply(lambda x: f"{x:.3e}")
    df_logrank_sorted["WILCOXON-PVAL_MAX"] = df_logrank_sorted["WILCOXON-PVAL_MAX"].apply(lambda x: f"{x:.3e}")

    return df_logrank_sorted

from lifelines.statistics import pairwise_logrank_test
from lifelines import KaplanMeierFitter
import pandas as pd
import numpy as np


import pandas as pd
import numpy as np

from lifelines.statistics import pairwise_logrank_test
from lifelines import KaplanMeierFitter
from lifelines.utils import median_survival_times


def compute_survival_statistics(
        df,
        cluster_label_columns,
        duration_col="EVOLUTION_DURATION",
        event_col="IS_UNCENSORED"
    ):
    """
    Computes pairwise Log-Rank and Wilcoxon tests for multiple clustering label columns
    and summarizes survival statistics (n patients, median survival, 95% CI) per cluster.

    Returns
    -------
    df_logrank_sorted : pd.DataFrame
        Summary of log-rank and Wilcoxon statistics for each clustering method.

    survival_table : pd.DataFrame
        Survival statistics per cluster and clustering method.
    """

    df_logrank = pd.DataFrame(index=cluster_label_columns)
    survival_summary = []

    durations = np.array(df[duration_col])
    events = np.array(df[event_col])

    for cluster_label_column in cluster_label_columns:

        groups = np.array(df[cluster_label_column])

        # -----------------------------
        # Pairwise Log-Rank test
        # -----------------------------
        result_logrank = pairwise_logrank_test(durations, groups, events)

        statistic_min_log = np.min(result_logrank.summary.test_statistic.tolist())
        pval_max_log = np.max(result_logrank.summary.p)

        # -----------------------------
        # Pairwise Wilcoxon test
        # -----------------------------
        result_wilcoxon = pairwise_logrank_test(
            durations,
            groups,
            events,
            weightings="wilcoxon"
        )

        statistic_min_wil = np.min(result_wilcoxon.summary.test_statistic.tolist())
        pval_max_wil = np.max(result_wilcoxon.summary.p)

        df_logrank.loc[cluster_label_column, "LOGRANK-PVAL_MAX"] = pval_max_log
        df_logrank.loc[cluster_label_column, "LOGRANK-STAT_MIN"] = statistic_min_log
        df_logrank.loc[cluster_label_column, "WILCOXON-PVAL_MAX"] = pval_max_wil
        df_logrank.loc[cluster_label_column, "WILCOXON-STAT_MIN"] = statistic_min_wil

        # -----------------------------
        # Survival statistics per cluster
        # -----------------------------
        for cluster_id in np.unique(groups):

            subset = df[df[cluster_label_column] == cluster_id]

            kmf = KaplanMeierFitter()

            kmf.fit(
                durations=subset[duration_col],
                event_observed=subset[event_col]
            )

            median_survival = kmf.median_survival_time_

            # Compute CI of median survival
            try:
                ci_df = median_survival_times(kmf.confidence_interval_)
                ci_lower = ci_df.iloc[0, 0]
                ci_upper = ci_df.iloc[0, 1]
            except:
                ci_lower = np.nan
                ci_upper = np.nan

            survival_summary.append({
                "method": cluster_label_column,
                "cluster": cluster_id,
                "n_patients": len(subset),
                "median_survival": median_survival,
                "CI_lower": ci_lower,
                "CI_upper": ci_upper
            })

    # -----------------------------
    # Format statistical results
    # -----------------------------
    df_logrank_sorted = df_logrank.sort_values(
        by="LOGRANK-PVAL_MAX",
        ascending=True
    )

    df_logrank_sorted["LOGRANK-PVAL_MAX"] = \
        df_logrank_sorted["LOGRANK-PVAL_MAX"].apply(lambda x: f"{x:.3e}")

    df_logrank_sorted["WILCOXON-PVAL_MAX"] = \
        df_logrank_sorted["WILCOXON-PVAL_MAX"].apply(lambda x: f"{x:.3e}")

    # -----------------------------
    # Survival summary table
    # -----------------------------
    survival_table = pd.DataFrame(survival_summary)

    survival_table["Median survival (95% CI)"] = survival_table.apply(
        lambda row: (
            f"{row['median_survival']:.2f} "
            f"({row['CI_lower']:.2f}-{row['CI_upper']:.2f})"
            if not pd.isna(row["median_survival"])
            else "NA"
        ),
        axis=1
    )

    survival_table = survival_table.sort_values(
        by=["method", "cluster"]
    )

    return df_logrank_sorted, survival_table