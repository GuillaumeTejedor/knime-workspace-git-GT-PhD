import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy import stats
from utils import cramers_v


def principal_feature_analysis(df, variability_percent=80):

    """
    Select best subset of features based on PCA algorithm.
    :param df: DataFrame that contains data.
    :param variability_percent: Percentage of variability we want when do the sum of eigenvalues.
    :return: Selected features, cumulated variability
    """

    X = StandardScaler().fit_transform(df)
    pca = PCA()
    pca.fit(X)

    eigenvalues = pca.explained_variance_
    eigenvectors = pca.components_.T  # A matrix
    print("eigenvalues=",eigenvalues, eigenvalues.shape)
    print("eigenvectors=",eigenvectors, eigenvectors.shape)

    # Step 3: retained variability → choose q
    sorted_eigvals = np.sort(eigenvalues)[::-1]
    print("sorted_eig=",eigenvalues)
    cumvar = np.cumsum(sorted_eigvals) / np.sum(sorted_eigvals)
    print("cumvar=",cumvar, len(cumvar))
    q = np.argmax(cumvar >= variability_percent/100) + 1

    Aq = eigenvectors[:, :q]
    print("Aq=",Aq, Aq.shape)

    # Step 4: K- means clustering
   
    p = q

    kmeans = KMeans(n_clusters=p, random_state=42)
    cluster_labels = kmeans.fit_predict(Aq)
    print("cluster_labels=", cluster_labels, cluster_labels.shape)
    # Step 5: select principal features
    feature_names = df.columns.tolist()
    print("feature_names=", feature_names, len(feature_names))
    principal_features = select_principal_features(Aq, cluster_labels, feature_names)

    return principal_features, cumvar

def select_principal_features(Aq, cluster_labels, feature_names):

    """
    Step 5 of PFA: select the feature nearest to each cluster centroid
    
    Parameters
    ----------
    Aq : np.array, shape (n_features, q)
        Feature vectors in PCA subspace
    cluster_labels : np.array, shape (n_features,)
        Cluster assignments for each feature
    feature_names : list of str
        Original feature names
    
    Returns
    -------
    principal_features : list of str
        Selected features (length = number of clusters nb_clusters)
    """
    
    print("\n#### FINDING CENTROIDS ##############################\n")

    nb_clusters = np.max(cluster_labels) + 1  # number of clusters

    principal_features = []

    for c in range(nb_clusters):
        print("# CLUSTER ", c, "#")
        # Indices of features in cluster c
        cluster_idx = np.where(cluster_labels == c)[0]
        print("cluster_idx=", cluster_idx)
        # Cluster feature vectors
        cluster_vectors = Aq[cluster_idx]
        print("cluster_vectors=", cluster_vectors)
        # Cluster centroid
        centroid = np.mean(cluster_vectors, axis=0)
        print("centroid=",centroid)
        # Compute distances to centroid
        diff_matrix = cluster_vectors - centroid
        print("diff_matrix=", diff_matrix)
        distances = np.linalg.norm(diff_matrix, axis=1)
        print("distances=",distances)
        
        # Index of feature nearest to centroid
        nearest_idx_in_cluster = np.argmin(distances)
        print("nearest_idx_in_cluster=",nearest_idx_in_cluster)
        feature_idx = cluster_idx[nearest_idx_in_cluster]
        print("feature_idx=", feature_idx)
        
        # Append original feature name
        principal_features.append(feature_names[feature_idx])
    
    return principal_features

def select_nominal_features_not_correlated_to_numerical_features(
        
        df, nominal_features_names, numerical_features_names):

    """
    Filter out nominal features correlated to numerical features from the provided dataframe.

    :param df: Dataframe that contains features.
    :param nominal_features_names: List of nominal features names.
    :param numerical_features_names: List of numerical features names.
    :return: Dataframe with filtered out nominal features.
    """

    # List to store nominal features to keep
    kept_nominal_features = []

    for bool_feature_name in nominal_features_names:
        print("#####################")
        print("bool_feature_name=", bool_feature_name)
        var_bool = df[bool_feature_name].to_numpy()
        print("var_bool=", var_bool)
        keep_feature = True

        distinct_nominal_values = np.unique(var_bool)
        print("len(distinct_nominal_values)=", len(distinct_nominal_values))

        for numerical_feature_name in numerical_features_names:
            var_numerical = df[numerical_feature_name].to_numpy()
            print("var_numerical=", var_numerical)
            
            groups = []
            for nominal_value in distinct_nominal_values:

                group = var_numerical[var_bool == nominal_value]
                if len(group) < 2:
                    continue
                groups.append(group)

            print("*groups=",*groups)
            stat, p_value = stats.kruskal(*groups, nan_policy='omit')
            print(f"Checking {bool_feature_name} vs {numerical_feature_name}: p={p_value}")

            if p_value < 0.05:
                print(f"-> Dropping {bool_feature_name} (correlated with {numerical_feature_name})")
                keep_feature = False
                break

        if keep_feature:
            kept_nominal_features.append(bool_feature_name)

    print("Kept nominal features:", kept_nominal_features)

    # --- NEW PART: get remaining (non-nominal, non-numerical) columns ---
    used_columns = set(nominal_features_names) | set(numerical_features_names)
    other_columns = [col for col in df.columns if col not in used_columns]

    # Build df_output with correct ordering:
    # 1) other columns
    # 2) kept nominal features
    # 3) all numerical features
    print("other_columns=", other_columns)
    df_output = df.loc[:, other_columns + kept_nominal_features + numerical_features_names].copy()

    return df_output

def select_nominal_features_not_correlated(
        df, nominal_features_names
):
    cols_to_remove = set()

    # --- Evaluate pairwise Cramér’s V ---
    for i in range(len(nominal_features_names)):
        col1 = nominal_features_names[i]
        if col1 in cols_to_remove:
            continue
        for j in range(i + 1, len(nominal_features_names)):
            col2 = nominal_features_names[j]
            if col2 in cols_to_remove:
                continue

            v = cramers_v(df[col1], df[col2])
            print("cramer(",col1,",",col2,")=", v)

            if v >= 0.7:
                # remove the *first* feature in the pair
                cols_to_remove.add(col1)
                break  # no need to compare col1 further

    # --- Filter the dataframe ---
    df_filtered = df.drop(columns=list(cols_to_remove))

    return df_filtered

def select_numerical_features_not_correlated_to_other_features(df, features_names1, features_names2, metric="Spearman"):

    """
    Drop features from first set that are correlated to features from second set 2.
    :param features_names1: 1d array first set of features where filtering will be applied.
    :param features_names2: 1d array second set of features.
    :return: Dataframe with dropped correlated features.
    """

    # List to store non-boolean features to drop
    features_to_drop = []

    # Only test NON-BOOLEAN features
    for feature_name1 in features_names1:

        var1 = df[feature_name1].to_numpy()

        for feature_name2 in features_names2:
            print("###############################")
            var2 = df[feature_name2].to_numpy()

            # Create mask for valid (non-NaN) pairs
            mask = ~np.isnan(var1) & ~np.isnan(var2)

            # Filter both variables
            v1 = var1[mask]
            v2 = var2[mask]
            
            if metric=="Spearman":
                corr = np.abs(stats.spearmanr(v1, v2).statistic)
                print(f"Checking {feature_name1} vs {feature_name2}: corr={corr}")
            elif metric=="Pearson":
                corr = np.abs(stats.pearsonr(v1, v2).statistic)
                print(f"Checking {feature_name1} vs {feature_name2}: corr={corr}")
            else:
                raise ValueError('Specified metric "', metric, '" do not exist.',)

            print("Number of patients =", len(v1))

            if corr >= 0.7:
                features_to_drop.append(feature_name1)
                print(f"Marking for removal: {feature_name1}")
                break

    # Drop correlated NON-BOOLEAN features
    df.drop(columns=features_to_drop, inplace=True)

    print("Dropped numerical features:", features_to_drop)

    return df