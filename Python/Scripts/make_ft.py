from utils import *
import pandas as pd
import numpy as np
from rdp import rdp


import os
os.environ["NUMBA_DISABLE_CUDA"] = "1"

def add_ft(df_patients, ft_activation_dict = {
        "duration": True,
        "auc": False,
        "mean_diff": False,
        "lv": False,
        "fv": True,
        "percentage_change_M6": True,
        "percentage_change_M12": False,
        "score_M6":False,
        "score_M12":True,
        "D50":True,
        "variation_diff": False,
        "nb_breakpoint_diff": False,
        "r2": False,
        "angle": False,
        "slope": True,
        "slope_M3": False,
        "slope_M6": False,
        "slope_M3M6": False,
        "hcs": True,
        "hcs_M3": False,
        "hcs_M6": False,
        "hcs_M3M6": False}, parameter="ALSFRS-R", interpolation="sigmoid"):

    """
    Compute features for each patient.
    :param df_patients: Dataframe where each row represent a patient.
    :return: Dataframe with patients and computed features.
    """

    duration_values = []
    auc_values = []
    slope_values = []
    hcs_values = []
    first_values = []
    last_values = []
    pc_changes_M6 = []
    pc_changes_M12 = []
    scores_M12 = []
    scores_M6 = []
    D50_values = []
    slopes_M3 = []
    slopes_M3M6 = []

    three_month_limit = 91
    six_month_limit = 182

    for _, row in df_patients.iterrows():
        
        if row["ID_PATIENT"] is not None:
            id = row["ID_PATIENT"]
        timestamps = row["List(TIMESTAMP)"]
        values = row["List(VALUE)"]
    
        if ft_activation_dict["duration"]: 
            duration_values.append(timestamps[-1])
        if ft_activation_dict["slope"]: 
            slope_values.append(get_slope(values, timestamps))
        if ft_activation_dict["auc"]:
            auc_values.append(get_auc(values, timestamps))
        if ft_activation_dict["fv"]: 
            first_values.append(values[0])
        if ft_activation_dict["lv"]: 
            last_values.append(values[-1])
        if ft_activation_dict["hcs"]: 
            hcs_values.append(get_highest_consecutive_slope(values, timestamps))
        if ft_activation_dict["percentage_change_M6"]:
            pc_change = get_percentage_decrease_after_x_days(timestamps, values, x_to_predict=183, parameter=parameter, interpolation=interpolation)
            pc_changes_M6.append(pc_change)
        if ft_activation_dict["percentage_change_M12"]:
            pc_change = get_percentage_decrease_after_x_days(timestamps, values, x_to_predict=365, parameter=parameter, interpolation=interpolation)
            pc_changes_M12.append(pc_change)
        if ft_activation_dict["score_M12"]:
            score_M12 = get_score_after_x_days(timestamps, values, x_to_predict=365, parameter=parameter, interpolation=interpolation)
            scores_M12.append(score_M12)
        if ft_activation_dict["score_M6"]:
            score_M6 = get_score_after_x_days(timestamps, values, x_to_predict=183, parameter=parameter, interpolation=interpolation)
            scores_M6.append(score_M6)
        if ft_activation_dict["D50"]:
            if pd.notna(row["ID_PATIENT"]):
                d50_value, _ = get_D50(timestamps, values, row["ID_PATIENT"])
            else:
                d50_value, _ = get_D50(timestamps, values)
            D50_values.append(d50_value)
        if ft_activation_dict["slope_M3"]:
            timestamps_M3 = [timestamp for timestamp in timestamps if timestamp <= three_month_limit]
            values_M3 = [value for timestamp, value in zip(timestamps, values) if timestamp <= three_month_limit]
            if len(timestamps_M3) >= 2: slopes_M3.append(get_slope(values_M3, timestamps_M3))
            else: slopes_M3.append(pd.NA)
        if ft_activation_dict["slope_M3M6"]:
            timestamps_M3M6 = [timestamp for timestamp in timestamps if timestamp > three_month_limit and timestamp <= six_month_limit]
            values_M3M6 = [value for timestamp, value in zip(timestamps, values) if timestamp > three_month_limit and timestamp <= six_month_limit]
            if len(timestamps_M3M6) >= 2: slopes_M3M6.append(get_slope(values_M3M6, timestamps_M3M6))
            else: slopes_M3M6.append(pd.NA)
            
    df_patients_features = df_patients.copy()
    if ft_activation_dict["duration"]:
        df_patients_features["DURATION"] = duration_values
    if ft_activation_dict["slope"]:
        df_patients_features["SLOPE"] = slope_values
    if ft_activation_dict["fv"]:
        df_patients_features["FV"] = first_values
    if ft_activation_dict["lv"]:
        df_patients_features["LV"] = last_values
    if ft_activation_dict["auc"]:
        df_patients_features["AUC"] = auc_values
    if ft_activation_dict["hcs"]:
        df_patients_features["HCS"] = hcs_values
    if ft_activation_dict["percentage_change_M6"]:
        df_patients_features["PC_CHANGE_M6"] = pc_changes_M6
    if ft_activation_dict["percentage_change_M12"]:
        df_patients_features["PC_CHANGE_M12"] = pc_changes_M12
    if ft_activation_dict["score_M12"]:
        df_patients_features["SCORE_M12"] = np.array(scores_M12)
    if ft_activation_dict["score_M6"]:
        df_patients_features["SCORE_M6"] = np.array(scores_M6)
    if ft_activation_dict["D50"]:
        df_patients_features["D50"] = D50_values
    if ft_activation_dict["slope_M3"]:
        df_patients_features["SLOPE_M3"] = slopes_M3
    if ft_activation_dict["slope_M3M6"]:
        df_patients_features["SLOPE_M3M6"] = slopes_M3M6

    return df_patients_features

def add_ft_tsfresh(df_input):

    """
    Generate features from sequences.
    :param df_input: Contains temporal sequences with following columns: id of patient, 
                     index of record, parameter value
    :return: Dataframe with extracted features for each patient.
    """

    from tsfresh import extract_features
    import pandas as pd

    column_name_id = df_input.columns.values[0]
    column_name_index = df_input.columns.values[1]
    column_name_parameter = df_input.columns.values[2]

    extracted_features = extract_features(
        timeseries_container=df_input, 
        column_id=column_name_id, 
        column_sort=column_name_index, 
        column_value=column_name_parameter
    )

    # Identify boolean-like columns (only 0 and 1)
    bool_cols = [
        col for col in extracted_features.columns
        if extracted_features[col].dropna().isin([0, 1]).all()
    ]

    # Rename them by adding "_bool" at the end
    extracted_features = extracted_features.rename(
        columns={col: f"{col}_bool" for col in bool_cols}
    )

    return extracted_features

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import warnings
warnings.filterwarnings("ignore")


def add_ft_chatgpt(df,
                   id_col="ID_PATIENT",
                   time_col="TIMESTAMP",
                   value_col="VALUE",
                   early_window=365,
                   plateau_delta=1,
                   plateau_window=90):
    """
    Extract advanced ALSFRS-R longitudinal features per patient.

    Each row in df must represent one measurement:
    ID_PATIENT | TIMESTAMP | VALUE

    Returns:
        DataFrame with one row per patient and extracted features.
    """

    # =========================
    # Logistic decline model
    # =========================
    # A  = theoretical maximum score
    # t0 = inflection point (maximal decline speed)
    # s  = steepness parameter
    def logistic(t, A, t0, s):
        return A / (1 + np.exp((t - t0) / s))

    # =========================
    # Weibull decline model
    # =========================
    # A     = initial level
    # scale = time scale parameter
    # k     = shape parameter (decline acceleration pattern)
    def weibull_decline(t, A, scale, k):
        return A * np.exp(-(t / scale) ** k)
    
    features = {}

    for pid, group in df.groupby(id_col):
        
        group = group.sort_values(time_col)
        
        #print("########################################")
        #print("group[List(TIMESTAMP)]=",group[time_col].values[0])
        #print("group[List(VALUE)]=",group[value_col].values[0])

        t = group[time_col].values[0].astype(float)
        y = group[value_col].values[0].astype(float)

        if len(t) < 3:
            continue

        # =========================
        # 1️⃣ Early slope (first-year slope with post-365 adjustment)
        # =========================
        # Linear regression slope during first year (early_window = 365 days)
        # Captures early disease aggressiveness.
        # If there is exactly one measurement in the first year,
        # compute slope using the **closest measurement after 365 days**.
        # 366 916 1773 => nothing before 365 days

        mask = t <= early_window  # measurements within first year

        if np.sum(mask) > 1:
            # Standard case: multiple points in first year → linear regression
            early_slope = np.polyfit(t[mask], y[mask], 1)[0]
        elif np.sum(mask) == 1:
            # Exactly one point in the first year
            baseline_idx = np.where(mask)[0][0]  # index of the single early-year point
            after_mask = t > early_window       # measurements after 365 days
            if np.any(after_mask):
                # Take the first measurement after 365 days
                next_idx = np.argmin(t[after_mask] - early_window)  # index of closest after-365 point
                after_indices = np.where(after_mask)[0]
                closest_idx = after_indices[next_idx]

                # Slope = (y_after - y_early) / (t_after - t_early)
                delta_y = y[closest_idx] - y[baseline_idx]
                delta_t = t[closest_idx] - t[baseline_idx]
                early_slope = delta_y / delta_t
            else:
                # No measurement after 365 days → cannot compute slope
                early_slope = np.nan
        

        # =========================
        # 2️⃣ Curvature index
        # =========================
        # Sum of absolute discrete second derivatives.
        # Measures deviation from linear progression.
        # High value = nonlinear or multi-phase disease.
        dt = np.mean(np.diff(t))
        curvature = np.sum(
            np.abs(y[2:] - 2*y[1:-1] + y[:-2]) / (dt**2)
        )

        # =========================
        # 3️⃣ Plateau duration
        # =========================
        # Total time where score change is small
        # (<= plateau_delta) over short intervals.
        # Captures temporary stabilization phases.
        plateau_duration = 0
        for i in range(len(y)-1):
            if abs(y[i+1] - y[i]) <= plateau_delta \
               and (t[i+1] - t[i]) <= plateau_window:
                plateau_duration += (t[i+1] - t[i])

        

        # =========================
        # 4️⃣ Logistic model parameters
        # =========================
        # Captures sigmoidal disease progression shape.
        #try:
        print("test")
        popt_log, _ = curve_fit(
            logistic,
            t,
            y,
            p0=[max(y), np.median(t), 100],
            # Original proposed by chatgpt: maxfev=10000
            maxfev=100000
        )
        A_log, t0_log, s_log = popt_log
        print("A_log=", A_log)
        print("t0_log=", t0_log)
        print("s_log=", s_log)

        # Steepness proxy: inverse of |s|
        # Higher value = sharper decline
        logistic_steepness = 1 / abs(s_log)

        # Time when 75% functional loss is reached
        # (trajectory exhaustion timing)
        t75 = t0_log + s_log * np.log(A_log/(0.25*A_log) - 1)

        #except:
            #A_log = t0_log = s_log = logistic_steepness = t75 = np.nan

        # =========================
        # 5️⃣ Weibull model parameters
        # =========================
        # Flexible monotonic decline model.
        # Shape parameter indicates acceleration pattern.
        try:
            popt_weib, _ = curve_fit(
                weibull_decline,
                t,
                y,
                p0=[max(y), np.median(t), 1.5],
                maxfev=10000
            )
            A_w, scale_w, k_w = popt_weib

            # k < 1  → decelerating decline
            # k = 1  → exponential
            # k > 1  → accelerating decline
            weibull_shape = k_w
            weibull_scale = scale_w

        except:
            weibull_shape = weibull_scale = np.nan

        # =========================
        # 6️⃣ Total decline magnitude
        # =========================
        # Observed functional loss over follow-up.
        total_decline = y[0] - y[-1]

        # =========================
        # 7️⃣ Progression variability
        # =========================
        # Standard deviation of successive score differences.
        # Captures instability or fluctuating progression.
        variability = np.std(np.diff(y))

        # ============================================================
        # 8️⃣ Change-point time
        # ============================================================
        # Definition:
        #   Time when ALSFRS-R slope changes significantly
        #   (approximates piecewise linear regime transition)
        # Clinical interpretation:
        #   Hybrid feature: possible acceleration phase
        #   or transition from plateau to rapid decline
        #
        cp_time = change_point_time(t, y)        

        # =========================
        # Store patient-level features
        # =========================
        features[pid] = {
            "early_slope": early_slope,
            "curvature_index": curvature,
            "plateau_duration": plateau_duration,
            "logistic_A": A_log,
            "logistic_t0": t0_log,
            "logistic_s": s_log,
            "logistic_steepness": logistic_steepness,
            "t75_logistic": t75,
            "weibull_shape_k": weibull_shape,
            "weibull_scale": weibull_scale,
            "total_decline": total_decline,
            "progression_variability": variability
        }

    return pd.DataFrame.from_dict(features, orient="index")


def add_pairs_ft(df_patients):

    """
    Compute features for each pair of data points.
    :param df_patients: Dataframe where each row represent a pair of data points.
    :return: Dataframe with pairs and computed features.
    """

    ft_activation_dict = get_ft_activation_dict()

    duration_differences = []
    slope_differences = []
    hcs_differences = []
    r2_values = []
    auc_differences = []
    mean_differences = []
    variation_differences = []
    fv_differences = []
    lv_differences = []
    nb_breakpoint_differences = []
    angle_values = []

    slope_differences_6M = []
    hcs_differences_6M = []
    slope_differences_3M = []
    hcs_differences_3M = []
    slope_differences_3M6M = []
    hcs_differences_3M6M = []

    for row in df_patients.values:

        id1 = row[0]
        id2 = row[1]

        timestamps_p1 = row[2]
        values_p1 = row[3]
        timestamps_p2 = row[4]
        values_p2 = row[5]
        
        timestamps_common, new_values_first_patient, new_values_second_patient = align_sequences(values_p1, timestamps_p1, values_p2, timestamps_p2)

        # -- COMPUTE DURATION DIFFERENCE BETWEEN TWO SEQUENCES
        if ft_activation_dict["duration"]: duration_differences.append(np.abs(timestamps_p1[-1] - timestamps_p2[-1]))

        # -- COMPUTE SLOPE DIFFERENCE BETWEEN TWO SEQUENCES
        if ft_activation_dict["slope"]: slope_differences.append(get_slope_difference(values_p1, timestamps_p1, values_p2, timestamps_p2))

        # -- COMPUTE HIGHEST CONSECUTIVE SLOPE DIFFERENCE BETWEEN TWO SEQUENCES
        if ft_activation_dict["hcs"]: hcs_differences.append(get_highest_consecutive_slope_difference(values_p1, timestamps_p1, values_p2, timestamps_p2))
        
        # -- COMPUTE AUC DIFFERENCE BETWEEN TWO SYNCHRONOUS SEQUENCES OF VALUES OF SAME LENGTH
        if ft_activation_dict["auc_diff"]:
            auc1 = get_auc(new_values_first_patient, timestamps_common)
            auc2 = get_auc(new_values_second_patient, timestamps_common)
            auc_differences.append(np.abs(auc1 - auc2))

        # -- COMPUTE MEAN DIFFERENCE BETWEEN TWO SEQUENCES OF VALUES
        if ft_activation_dict["mean_diff"]:
            mean_differences.append(np.abs(np.mean(values_p1) - np.mean(values_p2)))

        # -- COMPUTE FIRST VALUE DIFFERENCE BETWEEN TWO SEQUENCES
        if ft_activation_dict["fv"]:
            fv_differences.append(np.abs(values_p1[0] - values_p2[0]))

        # -- COMPUTE LAST VALUE DIFFERENCE BETWEEN TWO SEQUENCES
        if ft_activation_dict["lv"]:
            lv_differences.append(np.abs(values_p1[-1] - values_p2[-1]))

        # -- COMPUTE VARIATION DIFFERENCE BETWEEN TWO SEQUENCES OF VALUES
        if ft_activation_dict["variation_diff"]:
            var1 = get_variation(values_p1)
            var2 = get_variation(values_p2)
            variation_differences.append(np.abs(var1-var2))

        # -- COMPUTE NUMBER OF BREAKPOINTS DIFFERENCE BETWEEN TWO SEQUENCES
        if ft_activation_dict["nb_breakpoint_diff"]:
            simplified1 = rdp([(v, t) for v, t in zip(values_p1, timestamps_p1)], epsilon=1.35)
            simplified2 = rdp([(v, t) for v, t in zip(values_p2, timestamps_p2)], epsilon=1.35)
            original2 = [(v, t) for v, t in zip(values_p2, timestamps_p2)]
            nb_breakpoints1 = len(simplified1)
            nb_breakpoints2 = len(simplified2)
            nb_breakpoint_differences.append(np.abs(nb_breakpoints1-nb_breakpoints2))

        # -- COMPUTE R-SQUARED BETWEEN TWO SYNCHRONOUS SEQUENCES OF VALUES OF SAME LENGTH
        if ft_activation_dict["r2"]:
            is_inverse_best, r2 = get_r2(new_values_first_patient, new_values_second_patient)
            r2_values.append(r2)

        # -- COMPUTE ANGLE OF PREDICTED TREND BETWEEN TWO SYNCHRONOUS SEQUENCES OF VALUES OF SAME LENGTH
        if ft_activation_dict["angle"]:
            if is_inverse_best == True: x, y = new_values_second_patient, new_values_first_patient
            else: x, y = new_values_first_patient, new_values_second_patient
            angle = get_angle(x, y)
            angle_values.append(angle)


        # -- COMPUTE SLOPE AND DURATION AT 6 MONTH INTERVAL
        six_month_limit = 182

        if ft_activation_dict["slope_M6"] or ft_activation_dict["hcs_M6"]:
            timestamps_6M_p1 = [timestamp for timestamp in timestamps_p1 if timestamp <= six_month_limit]
            timestamps_6M_p2 = [timestamp for timestamp in timestamps_p2 if timestamp <= six_month_limit]
            values_6M_p1 = [value for timestamp, value in zip(timestamps_p1, values_p1) if timestamp <= six_month_limit]
            values_6M_p2 = [value for timestamp, value in zip(timestamps_p2, values_p2) if timestamp <= six_month_limit]

            if len(timestamps_6M_p1) >= 2 and len(timestamps_6M_p2) >= 2:
                if ft_activation_dict["slope_M6"]: slope_differences_6M.append(get_slope_difference(values_6M_p1, timestamps_6M_p1, values_6M_p2, timestamps_6M_p2))
                if ft_activation_dict["hcs_M6"]: hcs_differences_6M.append(get_highest_consecutive_slope_difference(values_6M_p1, timestamps_6M_p1, values_6M_p2, timestamps_6M_p2))    
            else:
                slope_differences_6M.append(pd.NA)
                hcs_differences_6M.append(pd.NA)

        # -- COMPUTE SLOPE AND DURATION AT 3 MONTH INTERVAL
        three_month_limit = 91

        if ft_activation_dict["slope_M3"] or ft_activation_dict["hcs_M3"]:
            timestamps_3M_p1 = [timestamp for timestamp in timestamps_p1 if timestamp <= three_month_limit]
            timestamps_3M_p2 = [timestamp for timestamp in timestamps_p2 if timestamp <= three_month_limit]
            values_3M_p1 = [value for timestamp, value in zip(timestamps_p1, values_p1) if timestamp <= three_month_limit]
            values_3M_p2 = [value for timestamp, value in zip(timestamps_p2, values_p2) if timestamp <= three_month_limit]
            
            if len(timestamps_3M_p1) >= 2 and len(timestamps_3M_p2) >= 2:
                if ft_activation_dict["slope_M3"]: slope_differences_3M.append(get_slope_difference(values_3M_p1, timestamps_3M_p1, values_3M_p2, timestamps_3M_p2))
                if ft_activation_dict["hcs_M3"]: hcs_differences_3M.append(get_highest_consecutive_slope_difference(values_3M_p1, timestamps_3M_p1, values_3M_p2, timestamps_3M_p2))    
            else:
                slope_differences_3M.append(pd.NA)
                hcs_differences_3M.append(pd.NA)

        # -- COMPUTE SLOPE AND DURATION BETWEEN 3-6 MONTH INTERVAL
        if ft_activation_dict["slope_M3M6"] or ft_activation_dict["hcs_M3M6"]:
            timestamps_3M6M_p1 = [timestamp for timestamp in timestamps_p1 if timestamp > three_month_limit and timestamp <= six_month_limit]
            timestamps_3M6M_p2 = [timestamp for timestamp in timestamps_p2 if timestamp > three_month_limit and timestamp <= six_month_limit]
            values_3M6M_p1 = [value for timestamp, value in zip(timestamps_p1, values_p1) if timestamp > three_month_limit and timestamp <= six_month_limit]
            values_3M6M_p2 = [value for timestamp, value in zip(timestamps_p2, values_p2) if timestamp > three_month_limit and timestamp <= six_month_limit]
    
            if len(timestamps_3M_p1) >= 2 and len(timestamps_3M_p2) >= 2:
                if ft_activation_dict["slope_M3M6"]: slope_differences_3M6M.append(get_slope_difference(values_3M6M_p1, timestamps_3M6M_p1, values_3M6M_p2, timestamps_3M6M_p2))
                if ft_activation_dict["hcs_M3M6"]: hcs_differences_3M6M.append(get_highest_consecutive_slope_difference(values_3M6M_p1, timestamps_3M6M_p1, values_3M6M_p2, timestamps_3M6M_p2))    
            else:
                slope_differences_3M6M.append(pd.NA)
                hcs_differences_3M6M.append(pd.NA)

    df_pairs_with_features = df_patients.copy()

    if ft_activation_dict["duration"]:
        df_pairs_with_features["DURATION_DIFF"] = duration_differences

    if ft_activation_dict["slope"]:
        df_pairs_with_features["SLOPE_DIFF"] = slope_differences
    if ft_activation_dict["slope_M3"]:
        df_pairs_with_features["SLOPE_DIFF_3M"] = slope_differences_3M
    if ft_activation_dict["slope_M6"]:
        df_pairs_with_features["SLOPE_DIFF_6M"] = slope_differences_6M
    if ft_activation_dict["slope_M3M6"]:
        df_pairs_with_features["SLOPE_DIFF_3M6M"] = slope_differences_3M6M

    if ft_activation_dict["fv"]:
        df_pairs_with_features["FV_DIFF"] = fv_differences
    if ft_activation_dict["lv"]:
        df_pairs_with_features["LV_DIFF"] = lv_differences

    if ft_activation_dict["hcs"]:
        df_pairs_with_features["HCS_DIFF"] = hcs_differences
    if ft_activation_dict["hcs_M3"]:
        df_pairs_with_features["HCS_DIFF_3M"] = hcs_differences_3M
    if ft_activation_dict["hcs_M6"]: 
        df_pairs_with_features["HCS_DIFF_6M"] = hcs_differences_6M
    if ft_activation_dict["hcs_M3M6"]: 
        df_pairs_with_features["HCS_DIFF_3M6M"] = hcs_differences_3M6M

    if ft_activation_dict["auc_diff"]:
        df_pairs_with_features["AUC_DIFF"] = auc_differences
    if ft_activation_dict["mean_diff"]: 
        df_pairs_with_features["MEAN_DIFF"] = mean_differences
    
    if ft_activation_dict["variation_diff"]: 
        df_pairs_with_features["VARIATION_DIFF"] = variation_differences
    if ft_activation_dict["nb_breakpoint_diff"]:
        df_pairs_with_features["NB_BREAKPOINT_DIFF"] = nb_breakpoint_differences
    if ft_activation_dict["r2"]:
        df_pairs_with_features["R2"] = r2_values
    if ft_activation_dict["angle"]:
        df_pairs_with_features['ANGLE_VALUES'] = angle_values

    return df_pairs_with_features