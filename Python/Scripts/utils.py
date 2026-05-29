from scipy.spatial.distance import euclidean
from scipy.interpolate import interp1d
from sklearn.metrics import r2_score
from matplotlib.colors import to_hex
import pandas as pd
import numpy as np
import matplotlib
import itertools
import math
from scipy.spatial.distance import cdist
from scipy.optimize import curve_fit
from sklearn.metrics import make_scorer, confusion_matrix
from scipy.interpolate import interp1d
from scipy.stats import chi2_contingency

### SIGMOID LEARNING PER ALSFRS-R SEQUENCE ##################################
# Defining initial parameters for sigmoid
p0 = [48, 0.00001, 0, 0]
# Defining bounds for sigmoid
bounds=[[38, 0, 0, -10],[48, 10, np.inf, 10]]
# Defining number of iterations to learn sigmoid parameters
maxfev = 10000000
#############################################################################

nb_days_per_year = 365.25

def specificity_score(y_true, y_pred, average='macro'):

    """
    Compute specificity score (true negative rate).
    :param y_true: 1D array data that contains true labels.
    :param y_pred: 1D array data that contains predicted labels.
    :param average: String that set the average method to compute metric.
    """

    cm = confusion_matrix(y_true, y_pred)
    TN, FP, FN, TP = [], [], [], []
    total = np.sum(cm)
    n_classes = cm.shape[0]

    for i in range(n_classes):
        TP_i = cm[i, i]
        FP_i = cm[:, i].sum() - TP_i
        FN_i = cm[i, :].sum() - TP_i
        TN_i = total - TP_i - FP_i - FN_i
        TP.append(TP_i)
        FP.append(FP_i)
        FN.append(FN_i)
        TN.append(TN_i)

    specificity = np.array(TN) / (np.array(TN) + np.array(FP))

    if average == 'macro':
        return np.mean(specificity)
    elif average == 'micro':
        return np.mean(specificity)
    elif average == 'weighted':
        support = np.array(TP) + np.array(FN)
        return np.average(specificity, weights=support)
    else:
        return specificity

def npv_score(y_true, y_pred, average='macro'):

    """
    Compute negative predictive value.
    :param y_true: 1D array data that contains true labels.
    :param y_pred: 1D array data that contains predicted labels.
    :param average: String that set the average method to compute metric.
    """

    cm = confusion_matrix(y_true, y_pred)
    TN, FP, FN, TP = [], [], [], []
    total = np.sum(cm)
    n_classes = cm.shape[0]

    for i in range(n_classes):
        TP_i = cm[i, i]
        FP_i = cm[:, i].sum() - TP_i
        FN_i = cm[i, :].sum() - TP_i
        TN_i = total - TP_i - FP_i - FN_i
        TP.append(TP_i)
        FP.append(FP_i)
        FN.append(FN_i)
        TN.append(TN_i)

    npv = np.array(TN) / (np.array(TN) + np.array(FN))

    if average == 'macro':
        return np.mean(npv)
    elif average == 'micro':
        return np.mean(npv)
    elif average == 'weighted':
        support = np.array(TP) + np.array(FN)
        return np.average(npv, weights=support)
    else:
        return npv

def sigmoid(x, b, k, a, c):
    """
    Sigmoid function.
    :param x: 1D array data.
    :param b: first hyperparameter.
    :param k: second hyperparameter.
    :param a: third hyperparameter.
    :return: 1D array y data.
    """
    return (b) / (1 + np.exp((k * (x - a)))) + c

def get_interpolated_value_after_x_days(timestamps, values, x_to_predict=183):

    if x_to_predict > timestamps[-1]:
        x_to_predict = timestamps[-1]

    f_linear = interp1d(timestamps, values, kind='linear')
    y_pred = f_linear(x_to_predict)
    
    return y_pred.flat[0]

def get_D50(timestamps, values, id_patient=None):
    """
    Get timestamp where ALSFRS-R value is equal to 24
    :param timestamps: 1D array of timestamps in days.
    :param values: 1D array of ALSFRS-R values
    :return: timestamp where ALSFRS-R score is equal to 24.
    """

    if values[0] < 24: return 0, pd.NA

    for timestamp, value in zip(timestamps, values):
        if value == 24: return timestamp/nb_days_per_year, pd.NA

    bounds[1][2] = timestamps[-1]

    popt, pcov, infodict, i1, i2 = curve_fit(
        sigmoid,
        timestamps,
        values,
        bounds=bounds,
        p0=p0,
        full_output=True,
        maxfev=maxfev
        )
    
    b, k, a, c = popt[0], popt[1], popt[2], popt[3]
    residuals = infodict["fvec"]
    error = sum(np.power(residuals,2))

    def solve_for_x(b, k, a, c, y=24):
        term = (b / (y - c)) - 1
        x = (np.log(term) / k) + a
        return x
    
    x_solution = solve_for_x(*popt, y=24)
    x_solution_in_years = x_solution/nb_days_per_year

    if id_patient is not None:
        if id_patient==297:
            print("------------------------------")
            print("timestamps=", timestamps)
            print("values=", values)
            print("bounds=", bounds)
            print("(b,k,a,c)=", (b,k,a,c))

    return x_solution_in_years, error

def get_percentage_decrease_after_x_days(timestamps, values, x_to_predict=183, interpolation="sigmoid", parameter="ALSFRS-R"):
    
    """
    Get percentage of decrease after x days since first score.
    :param timestamps: 1D array of timestamps in days.
    :param values: 1D array of ALSFRS-R values.
    :param x_to_predict: Number of days elapsed to get predicted value.
    :param interpolation: Type of interpolation (i.e. linear, sigmoid)
    :param parameter: Clinical parameter (i.e. ALSFRS-R, WEIGHT, FVC)
    :return: Predicted Percentage of decrease.
    """

    if interpolation == "sigmoid":
        bounds[1][2] = timestamps[-1]
        popt, pcov, infodict, i1, i2 = curve_fit(
        sigmoid, 
        timestamps, 
        values,
        p0, 
        bounds=bounds, 
        full_output=True,
        maxfev=maxfev
        )

        y_init = sigmoid(0, *popt)
        y_pred = sigmoid(x_to_predict, *popt)
        if y_pred < 0 and parameter == "ALSFRS-R": y_pred = 0

    elif interpolation == "linear":
        if x_to_predict > timestamps[-1]: x_to_predict = timestamps[-1]
        f_linear = interp1d(timestamps, values, kind='linear')

        y_init = values[0]
        y_pred = f_linear(x_to_predict)

    percentage = ((y_init - y_pred)/y_init) * 100

    return percentage

def get_score_after_x_days(timestamps, values, x_to_predict=183, interpolation="sigmoid", parameter="ALSFRS-R"):
    
    """
    Get score after x days since first record.
    :param timestamps: 1D array of timestamps in days.
    :param values: 1D array of ALSFRS-R values
    :param x_to_predict: Number of days elapsed to get predicted value.
    :param interpolation: Type of interpolation (i.e. linear, sigmoid)
    :param parameter: Clinical parameter (i.e. ALSFRS-R, WEIGHT, FVC)
    :return: Predicted ALSFRS-R score.
    """

    if interpolation == "sigmoid":
        bounds[1][2] = timestamps[-1]

        popt, pcov, infodict, i1, i2 = curve_fit(
            sigmoid, 
            timestamps, 
            values, 
            p0, 
            bounds=bounds, 
            full_output=True,
            maxfev=maxfev
            )
        
        y_pred = sigmoid(x_to_predict, *popt)
        if y_pred < 0: y_pred = 0
    elif interpolation == "linear":
        if x_to_predict > timestamps[-1]: x_to_predict = timestamps[-1]
        f_linear = interp1d(timestamps, values, kind='linear')

        y_pred = f_linear(x_to_predict)
        
    return y_pred

def closest_timestamp_index(timestamps, target, threshold):
    """
    Finds the index of the closest timestamp in the list to the given target timestamp.
    If the difference is greater than the threshold, returns None.
    
    :param timestamps: List of timestamps (elapsed days), first timestamp starts at 0
    :param target: The target timestamp to compare
    :param threshold: The maximum allowed difference
    :return: The index of the closest timestamp or None if the difference exceeds the threshold
    """
    closest_index = min(range(len(timestamps)), key=lambda i: abs(timestamps[i] - target))
    closest_diff = abs(timestamps[closest_index] - target)

    return closest_index if closest_diff <= threshold else pd.NA

def get_label_colors(labels):

    """
    Return a list that contains a distinct color for each distinct label.

    :param labels: 1D array of labels
    :return: List of colors
    """

    # Get the colormap
    colors = matplotlib.colormaps.get_cmap('tab20')
    # Generate colors by evenly spacing them across the colormap for each unique cluster label
    label_colors = [to_hex(colors(i / len(labels))) for i in range(len(labels))]
    #label_colors = [to_hex(colors(i/len(labels))) for i in range(len(labels))]
    return label_colors



def get_label_colors(labels, is_returning_color_map=False):
    """
    Return a fixed list of colors for a given set of labels, with predefined palettes
    for 2 to 10 clusters. Falls back to tab20 if more than 10 clusters.

    :param labels: 1D array-like of labels (must be sortable)
    :param is_returning_color_map: Is the function return a color map (True: yes; False: Return list of colors)
    :return: List of hex color strings corresponding to each label
    """
    # Sorted unique labels to assign fixed colors in a consistent order
    unique_labels = sorted(set(labels))
    num_clusters = len(unique_labels)

    # Define fixed color palettes for 2 to 10 clusters
    predefined_palettes = {
        2: ['#1f77b4', '#ff7f0e'],
        3: ['#1f77b4', '#ff7f0e', '#2ca02c'],
        4: ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'],
        5: ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'],
        6: ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b'],
        7: ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2'],
        8: ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f'],
        9: ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22'],
        10:['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    }

    # Use predefined palette if number of clusters is between 2 and 10
    if 2 <= num_clusters <= 10:
        palette = predefined_palettes[num_clusters]
    else:
        # Fallback to a generic colormap (e.g., tab20)
        cmap = matplotlib.colormaps.get_cmap('tab20')
        palette = [to_hex(cmap(i % 20)) for i in range(num_clusters)]

    if is_returning_color_map:
        return {label: palette[i] for i, label in enumerate(unique_labels)}
    else:
        # Map each unique label to a color
        label_to_color = {label: palette[i] for i, label in enumerate(unique_labels)}
        # Return color list corresponding to input label order
        return [label_to_color[label] for label in labels]

def find_medoid(cluster_points):
    """Finds the medoid of a given cluster of points."""
    distances = cdist(cluster_points, cluster_points, metric='euclidean')  # Compute pairwise distances
    total_distances = distances.sum(axis=1)  # Sum of distances for each point
    medoid_index = np.argmin(total_distances)  # Index of the medoid
    return medoid_index  # Return the medoid

def get_matrix_pairs(df_pairs, normalization="none"):

    """
    Get matrix of pairs of individuals for corresponding measure.
    :param df_pairs: Dataframe that contains measures for each pair of patients.
    :param normalization: none|sigmoid|minmax
    """

    unique_patients = pd.unique(df_pairs[['ID1_PATIENT', 'ID2_PATIENT']].values.ravel('K'))
    df_measures_matrix = pd.DataFrame(np.nan, index=unique_patients, columns=unique_patients)
    for _, row in df_pairs.iterrows():
        df_measures_matrix.loc[row['ID1_PATIENT'], row['ID1_PATIENT']] = 0
        df_measures_matrix.loc[row['ID2_PATIENT'], row['ID2_PATIENT']] = 0
        if normalization=="none":
            df_measures_matrix.loc[row['ID1_PATIENT'], row['ID2_PATIENT']] = row['MEASURE']
            df_measures_matrix.loc[row['ID2_PATIENT'], row['ID1_PATIENT']] = row['MEASURE']
        elif normalization=="sigmoid":
            df_measures_matrix.loc[row['ID1_PATIENT'], row['ID2_PATIENT']] = row['SIGMOID']
            df_measures_matrix.loc[row['ID2_PATIENT'], row['ID1_PATIENT']] = row['SIGMOID']
        elif normalization=="minmax":
            df_measures_matrix.loc[row['ID1_PATIENT'], row['ID2_PATIENT']] = row['MINMAX']
            df_measures_matrix.loc[row['ID2_PATIENT'], row['ID1_PATIENT']] = row['MINMAX']

    return df_measures_matrix

def get_percentile(A, p):

    """
    :param A: Array of k dimensions
    :param p: Percentile
    :return: the pth percentile from A
    """

    array = A.flatten()
    array = np.sort(array)
    return np.percentile(array, p)

def get_euclidean_cost_matrix(A, B):
    """
    :param A: 2d array
    :param B: 2d array
    :return: Cost matrix
    """

    C = np.zeros([len(A), len(B)])
    for i in range(0,len(A)):
        for j in range(0,len(B)):
            C[i,j] = euclidean(A[i], B[j])
    return C

def get_HATS_cost_matrix(sv1, sv2, st1, st2, pe=1, pt=1):

    """
    :param sv1: 2d array    (sequence of vectors)
    :param sv2: 2d array    (sequence of vectors)
    :param st1: 1d array    (sequence of values that correspond to the number of days elapsed for each vector since onset)
    :param st2: 1d array    (sequence of values that correspond to the number of days elapsed for each vector since onset)
    :return: Cost matrix
    """

    C_date = np.zeros([len(sv1), len(sv2)])
    C_event = np.zeros([len(sv1), len(sv2)])
    for i in range(0,len(sv1)):
        for j in range(0,len(sv2)):
            C_date[i,j] = (st1[i] - st2[j])**2
            C_event[i,j] = euclidean(sv1[i], sv2[j])

    return pe * C_event + pt * C_date, C_event, C_date

def get_quartiles(values):
    q1 = np.percentile(values, 25)
    q2 = np.percentile(values, 50)
    q3 = np.percentile(values, 75)
    return q1, q2, q3

def convert_distance_matrix_to_similarity_matrix(d_matrix, scaling_factor):

    """
    :d_matrix: 2D numpy array that contains distances between patients.
    :scaling_factor: User-defined constant.
    :return: Similarity matrix.
    """

    d_median = np.median(d_matrix[d_matrix > 0])
    #print(d_median)
    gamma =  1 / (2 * d_median ** 2)

    return np.array([ (np.exp(-gamma*np.power(r, 2))) for r in np.array(d_matrix)])

def find_best_kneighbors_from_spectral_clustering(embedding, n_clusters):
    """

    Return the best number of neighbors for the spectral clustering.

    :embedding: 2D numpy array where rows represent patients and columns features.
    :return: Best number of neighbors
    """

    from sklearn.cluster import SpectralClustering
    from sklearn.metrics import silhouette_score

    n_patients = len(embedding)
    
    if n_patients < 50: neighbors_thresold = n_patients
    else: neighbors_thresold = 50
    
    neighbors_grid = range(2, neighbors_thresold)
    best_score = -1

    for n_neighbors in neighbors_grid:
        model = SpectralClustering(
            n_clusters=n_clusters,
            affinity="nearest_neighbors",
            n_neighbors=n_neighbors
        )
        labels = model.fit_predict(embedding)

        score = silhouette_score(embedding, labels, metric="euclidean")
        
        #print(f"n_neighbors={n_neighbors}, silhouette_score={score:.4f}")
        
        if score > best_score:
            best_score = score
            best_labels = labels
            best_n = n_neighbors

    return best_n




def interpolate_sequence(timestamps, values, interval_in_days=90, nb_intervals=5):

    """
    Align given sequence with 90 days regularity
    """

    interp_function = interp1d(timestamps, values, kind='linear', fill_value=0, bounds_error=False)
    interpolated_timestamps = np.linspace(0, (nb_intervals) * interval_in_days, nb_intervals+1)
    interpolated_values = [np.round(interpolated_value, 1) for interpolated_value in interp_function(interpolated_timestamps)]

    return interpolated_timestamps, interpolated_values

def align_sequences(values1, timestamps1, values2, timestamps2):

    """
    Align pair of sequences with 90 days regularity
    """

    interp_function1 = interp1d(timestamps1, values1, kind='linear', fill_value=0, bounds_error=False)
    interp_function2 = interp1d(timestamps2, values2, kind='linear', fill_value=0, bounds_error=False)
    num_intervals = math.ceil(max(timestamps1[-1], timestamps2[-1])/90)
    timestamps_common = np.linspace(0, (num_intervals) * 90, num_intervals+1)

    interp_values1 = [x if x >= 0 else 0 for x in interp_function1(timestamps_common)]
    interp_values2 = [x if x >= 0 else 0 for x in interp_function2(timestamps_common)]

    return timestamps_common, interp_values1, interp_values2

def get_slope(values, timestamps):
    return (values[-1]-values[0])/(timestamps[-1]-timestamps[0])

def get_slope_difference(values_p1, timestamps_p1, values_p2, timestamps_p2):
    slope_p1 = get_slope(values_p1, timestamps_p1)
    slope_p2 = get_slope(values_p1, timestamps_p2)
    slope_difference = np.abs(slope_p1-slope_p2)
    return slope_difference

def get_variation(values):
    return ((values[-1]-values[0])/values[0])

def get_auc(values, timestamps):

    auc_list = []

    for i in range(0, len(values)):
        if i+1 < len(values):
            ts_diff = timestamps[i+1] - timestamps[i]
            val_diff = values[i+1] - values[i]
            value = min(values[i], values[i+1])

            auc = np.abs((ts_diff * value) + ((ts_diff * val_diff)/2))
            auc_list.append(auc)

    return sum(auc_list)

def get_r2(s1_values, s2_values):

    is_inverse_best = False

    x = s1_values
    y = s2_values
    min_xval = int(min(x))
    max_xval = math.ceil(max(x))
    trend_y_values = np.linspace(min_xval, max_xval, num=len(x))
    r_squared1 = r2_score(trend_y_values, sorted(y))

    x = s2_values
    y = s1_values
    min_xval = int(min(x))
    max_xval = math.ceil(max(x))
    trend_y_values = np.linspace(min_xval, max_xval, num=len(x))
    r_squared2 = r2_score(trend_y_values, sorted(y))

    if r_squared2 > r_squared1:
        is_inverse_best = True

    return is_inverse_best, max(r_squared1, r_squared2)

def get_angle(s1_values, s2_values):
    x = s1_values
    y = s2_values
    coefficients = np.polyfit(x=x, y=y, deg=1)
    trend_y_values = np.polyval(coefficients, x)
    adjacent_val = np.abs(trend_y_values[-1] - trend_y_values[0])
    axiside_val = max(x) - min(x)
    hypothenus_val = np.sqrt(np.power(adjacent_val, 2) + np.power(axiside_val, 2))
    angle = math.degrees(math.asin(adjacent_val/hypothenus_val))

    return angle

def get_pairs(df):
    ids = np.array(df["ID_PATIENT"])
    pairs = list(itertools.combinations(ids, 2))
    pairs_df = pd.DataFrame(pairs, columns=['ID1_PATIENT', 'ID2_PATIENT'])
    return pairs_df

def get_highest_consecutive_slope(values, timestamps):
    highest_slope = np.finfo(np.float32).max
    for i, (value, timestamp) in enumerate(zip(values, timestamps)):
        if (i + 1) < len(values):
            consecutive_slope = (values[i+1] - values[i])/(timestamps[i+1] - timestamps[i])
            if consecutive_slope < highest_slope and consecutive_slope != -np.inf:
                highest_slope = consecutive_slope
    return highest_slope

def get_highest_consecutive_slope_difference(values_p1, timestamps_p1, values_p2, timestamps_p2):
    highest_consecutive_slope_p1 = get_highest_consecutive_slope(values_p1, timestamps_p1)
    highest_consecutive_slope_p2 = get_highest_consecutive_slope(values_p2, timestamps_p2)
    return np.abs(highest_consecutive_slope_p1 - highest_consecutive_slope_p2)

def get_highest_consecutive_value_difference(values):
    highest_consecutive_value_difference = np.finfo(np.float32).min
    for i, value in enumerate(values):
        if (i + 1) < len(values):
            consecutive_value_difference = np.abs(values[i] - values[i+1])
            if consecutive_value_difference > highest_consecutive_value_difference:
                highest_consecutive_value_difference = consecutive_value_difference
    return highest_consecutive_value_difference

def get_highest_consecutive_timestamp_difference(timestamps):
    highest_consecutive_timestamp_difference = np.finfo(np.float32).min
    for i, value in enumerate(timestamps):
        if (i + 1) < len(timestamps):
            consecutive_timestamp_difference = np.abs(timestamps[i] - timestamps[i+1])
            if consecutive_timestamp_difference > highest_consecutive_timestamp_difference:
                highest_consecutive_timestamp_difference = consecutive_timestamp_difference
    return highest_consecutive_timestamp_difference


#def get_median_after_x_days(list_timestamps, list_values, x_to_predict=183):

    predicted_values = []

    for (timestamps, values) in zip(list_timestamps, list_values):
        
        bounds[1][2] = timestamps[-1]

        popt, pcov, infodict, i1, i2 = curve_fit(
            sigmoid,
            timestamps,
            values,
            bounds=bounds,
            p0=p0,
            full_output=True,
            maxfev=maxfev
        )

        y_pred = sigmoid(x_to_predict, *popt)
        if y_pred < 0: y_pred = 0
        predicted_values.append(y_pred)

    return np.percentile(predicted_values, 90)

def cramers_v(x, y):
    """
    Compute Cramér's V for two boolean/nominal Series.
    :param x: First 1d array
    :param y: Second 1d array
    
    """
    confusion_matrix = pd.crosstab(x, y)
    #print("confusion_matrix=\n", confusion_matrix)
    chi2, p, dof, expected = chi2_contingency(confusion_matrix)
    #print("chi2=",chi2)
    n = confusion_matrix.sum().sum()
    #print("n=",n)
    r, k = confusion_matrix.shape
    #print("(r, k)=", (r, k))
    return np.sqrt(chi2 / (n * (min(r - 1, k - 1))))


#################################
# - PROPOSED FEATURES BY LLMS - #
#################################

def compute_plateau_duration(values, timestamps, delta=1, window=90):
    """
    Compute total duration of plateau periods.

    Plateau Definition
    ------------------
    Consecutive visits where:
        |score_change| ≤ delta
        AND
        time_difference ≤ window days

    Parameters
    ----------
    values : List of values
    timestamps : List of timestamps in days

    delta : float, default=1
        Maximum allowed change in ALSFRS-R score
        between consecutive visits to be considered stable.
        Units: ALSFRS-R points.

    window : int, default=90
        Maximum allowed time difference between visits
        to consider them contiguous.
        Units: days.

    Returns
    -------
    duration : float
        Total plateau duration in days.

    Interpretation
    --------------
    Larger value → longer functional stability periods.
    """

    duration = 0

    for i in range(len(values) - 1):
        if (
            abs(values[i+1] - values[i]) <= delta and
            (timestamps[i+1] - timestamps[i]) <= window
        ):
            duration += timestamps[i+1] - timestamps[i]

    return duration

def compute_early_slope(values, timestamps, anchor_days=365):
    """
    Compute early disease progression slope.

    Parameters
    ----------
       values : List of values
       timestamps : List of timestamps in days

    anchor_days : int, default=365
        Time window (days) to compute early slope.

    Returns
    -------
    slope : float
        Linear regression slope (points/day).
    """

    mask = timestamps <= anchor_days

    if np.sum(mask) > 1:
        slope = np.polyfit(timestamps[mask], values[mask], 1)[0]
    else:
        slope = np.nan

    return slope

def compute_curvature_index(group):
    t = group['TIMESTAMP'].values
    y = group['ALSFRS-R'].values

    if len(y) < 3:
        return np.nan

    curvature = np.sum(
        np.abs(y[2:] - 2*y[1:-1] + y[:-2]) /
        np.mean(np.diff(t))**2
    )

    return curvature

def change_point_time(time, score, min_points=2):
    """
    Estimate change-point time via piecewise linear regression.
    Returns time (days) when slope changes.
    """
    if len(time) < 2 * min_points:
        return np.nan

    best_t = None
    best_error = np.inf

    for i in range(min_points, len(time) - min_points):
        t1, y1 = time[:i], score[:i]
        t2, y2 = time[i:], score[i:]

        c1 = np.polyfit(t1, y1, 1)
        c2 = np.polyfit(t2, y2, 1)

        err = np.sum((np.polyval(c1, t1) - y1)**2) + np.sum((np.polyval(c2, t2) - y2)**2)

        if err < best_error:
            best_error = err
            best_t = time[i]

    return best_t