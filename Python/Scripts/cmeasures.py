
from utils import *
import warnings
from scipy.spatial import distance

def weighted_measure(coefficients, values, use_intercept=False):
    """
    Method that compute weighted measure between 1D list of coefficients 
    and 1D list of features value.
    
    :param coefficients: 1D list of coefficients
    :param values: 1D list of feature values
    :param use_intercept: Specify if intercept from SVM is used as a coefficient to compute measure.
    :return: Metric measure
    """
    
    weights = coefficients[0:-1]
    print(len(weights))
    #print("weights=", weights)
    if use_intercept:
        intercept = coefficients[-1]
    else:
        intercept = 0

    weighted_values = []

    for value, weight in zip(values, weights):
        if np.isnan(value) == False:
            weighted_values.append(value * weight)

    measure = np.sum(weighted_values) + intercept

    """if use_intercept:
        # Correct measure
        if measure<intercept:
            diff_intercept_and_measure = intercept - measure
            corrected_measure = intercept+diff_intercept_and_measure
        else: 
            corrected_measure = measure

        # Shift measure
        if intercept < 0:
            final_measure = corrected_measure + np.abs(intercept)
        elif intercept > 0:
            final_measure = corrected_measure - intercept
        else:
            final_measure = corrected_measure
    else:
        final_measure = measure
    """
    
    is_displayed = False
    if is_displayed:
        print("-----------------------------------")
        print("intercept=", intercept)
        print("values=", values)
        print("coefficients=", coefficients)
        print("weights=", coefficients[0:-1])
        print("measure=",measure)
        """if use_intercept:
            print("corrected_measure=",corrected_measure)
        print("final_measure=",final_measure)"""
        
    return measure

def manhattan_pairs(vector):
    """
    Method that compute classic manhattan distance on features pairs of patients.
    :param X: 1D array where each element represent a feature value.
    :return: Manhattan metric.
    """
    measure = np.sum(vector)
    return measure

def euclidean_pairs(vector):
    """
    Method that compute classic euclidean distance on features pairs of patients.
    :param X: 1D array where each element represent a feature value.
    :return: Manhattan metric.
    """
    measure = np.sqrt(np.sum(np.power(vector, 2)))
    return measure

def cosine_distance(vector1, vector2):
    
    """
    Compute cosine distance.
    
    :param vector1: First sequence 1D array of values
    :param vector2: Second sequence 1D array of values
    :return: Cosine distance measure
    """

    dot_product = np.dot(vector1, vector2)
    magnitude_A = np.linalg.norm(vector1)
    magnitude_B = np.linalg.norm(vector2)
    cosine_similarity = dot_product / (magnitude_A * magnitude_B)
    cosine_dist = 1 - cosine_similarity
    print(cosine_similarity, cosine_dist)
    return cosine_dist

def DTW(s1, s2):

    """
    Compute Dynamic Time Warping (DTW). Return a dissimilarity measure.
    
    :param s1: First sequence 1D array of values
    :param s1: Second sequence 1D array of values
    :return: Dissimilarity measure
    """
    #print("-----------------")
    n = len(s1)
    m = len(s2)
    DTW = np.full((n + 1, m + 1), np.inf)
    DTW[0, 0] = 0
    #print("s1: ", s1)
    #print("s2: ", s2)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            #print("(s1[i-1], s2[j-1]) =", (s1[i-1], s2[j-1]))
            cost = euclidean([s1[i-1]], [s2[j-1]])
            DTW[i, j] = cost + min(DTW[i - 1, j],     # insertion
                                   DTW[i, j - 1],     # deletion
                                   DTW[i - 1, j - 1]) # match
    
    #print("measure: ", DTW[n, m])

    return DTW[n, m]

def DTW_with_window(s, t, w):

    n = len(s)
    m = len(t)
    w = max(w, abs(n-m))

    DTW = np.full((n, m), np.inf)
    DTW[0, 0] = 0

    for i in range(1, n):
        for j in range(max(1,i-w), min(m ,i+w)):
            DTW[i, j] = 0
    
    for i in range(1, n):
        for j in range(max(1,i-w), min(m,i+w)):
            cost = euclidean(s[i], t[j])
            DTW[i, j] = cost + min(DTW[i - 1, j],     # insertion
                                   DTW[i, j - 1],     # deletion
                                   DTW[i - 1, j - 1]) # match
            
    return DTW[n-1, m-1]

def lcs(X, Y, m, n): 
	if m == 0 or n == 0: 
		return 0
	elif X[m-1] == Y[n-1]: 
		return 1 + lcs(X, Y, m-1, n-1) 
	else:
		return max(lcs(X, Y, m, n-1), lcs(X, Y, m-1, n)) 

def Dlp(A, B, p=2):
    cost = np.sum(np.power(np.abs(A - B), p))
    return np.power(cost, 1 / p)

def twed(A, timeSA, B, timeSB, nu, _lambda):
    # [distance, DP] = TWED( A, timeSA, B, timeSB, lambda, nu )
    # Compute Time Warp Edit Distance (TWED) for given time series A and B
    #
    # A      := Time series A (e.g. [ 10 2 30 4])
    # timeSA := Time stamp of time series A (e.g. 1:4)
    # B      := Time series B
    # timeSB := Time stamp of time series B
    # lambda := Penalty for deletion operation
    # nu     := Elasticity parameter - nu >=0 needed for distance measure
    # Reference :
    #    Marteau, P.; F. (2009). "Time Warp Edit Distance with Stiffness Adjustment for Time Series Matching".
    #    IEEE Transactions on Pattern Analysis and Machine Intelligence. 31 (2): 306–318. arXiv:cs/0703033
    #    http://people.irisa.fr/Pierre-Francois.Marteau/

    # Check if input arguments
    if len(A) != len(timeSA):
        print("The length of A is not equal length of timeSA")
        return None, None

    if len(B) != len(timeSB):
        print("The length of B is not equal length of timeSB")
        return None, None

    if nu < 0:
        print("nu is negative")
        return None, None

    # Add padding
    A = np.array([0] + list(A))
    timeSA = np.array([0] + list(timeSA))
    B = np.array([0] + list(B))
    timeSB = np.array([0] + list(timeSB))

    n = len(A)
    m = len(B)
    # Dynamical programming
    DP = np.zeros((n, m))

    # Initialize DP Matrix and set first row and column to infinity
    DP[0, :] = np.inf
    DP[:, 0] = np.inf
    DP[0, 0] = 0

    # Compute minimal cost
    for i in range(1, n):
        for j in range(1, m):
            # Calculate and save cost of various operations
            C = np.ones((3, 1)) * np.inf
            # Deletion in A
            C[0] = (
                DP[i - 1, j]
                + Dlp(A[i - 1], A[i])
                + nu * (timeSA[i] - timeSA[i - 1])
                + _lambda
            )
            # Deletion in B
            C[1] = (
                DP[i, j - 1]
                + Dlp(B[j - 1], B[j])
                + nu * (timeSB[j] - timeSB[j - 1])
                + _lambda
            )
            # Keep data points in both time series
            C[2] = (
                DP[i - 1, j - 1]
                + Dlp(A[i], B[j])
                + Dlp(A[i - 1], B[j - 1])
                + nu * (abs(timeSA[i] - timeSB[j]) + abs(timeSA[i - 1] - timeSB[j - 1]))
            )
            # Choose the operation with the minimal cost and update DP Matrix
            DP[i, j] = np.min(C)
    distance = DP[n - 1, m - 1]
    return distance, DP

def drop_dtw_HATS(sv1, sv2, st1, st2, pe=1, pt=1, drop_cost_method=None, sc_param=None, tc_param=None, return_matrix : bool = False):

    C, C_event, C_date = get_HATS_cost_matrix(sv1, sv2, st1, st2, pe, pt)

    if drop_cost_method is not None and "percentile" in drop_cost_method:
        percentile = float(drop_cost_method.split("percentile")[1])
        drop_cost = get_percentile(C, percentile)
    else:
        drop_cost = None

    n = len(sv1)
    m = len(sv2)

    if (sc_param != None) and (sc_param >= min(n,m)): # warn if tc_param is to high and set it to the maximum value if it is the case
        warnings.warn(f"Sakoe-Chiba parameter is too high. It has been replaced by the maximum value.")
        sc_param = min(n,m)-1
    
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

    if (drop_cost is None):
        drop_cost = float("inf")
    else:
        D_zminus[0,1:] = (np.arange(m)+1)*drop_cost
        D_minusx[1:,0] = (np.arange(n)+1)*drop_cost
        D_minus[0,1:] = (np.arange(m)+1)*drop_cost
        D_minus[1:,0] = (np.arange(n)+1)*drop_cost

    DTW = D_minus.copy()
    if return_matrix:
        PathMatrix = np.full((n+1,m+1,3), np.nan, dtype=np.int16) # initialize path matrix to NaN

    for i in range(1,n+1): # compute the cost for each comparison of events and add the minimum between DTW[i-1,j], DTW[i,j-1] and DTW[i-1,j-1]
        for j in range(max(1,i-n_sc+1),min(m,i+m_sc-1)+1):
            if tc_param is None or C_date[i-1,j-1] <= tc_param:
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
            elif C_date[i-1,j-1] > tc_param and j>i: #prune when tc is no more satisfied (assume that events are temporaly ordered)
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

def HATS_distance(v1, v2, t1, t2, pe=1, pt=1):
    """
    Thomas Guyet's distance between events for HIERASTISEQ algorithm
    :param v1: First vector
    :param v2: Second vector
    :param t1: Moment that   occured (for us, the moment correspond to the number of days elapsed since onset)
    :param t2: Moment that v2 occured (for us, the moment correspond to the number of days elapsed since onset)
    :param pe: Event weighting
    :param pt: temporal weighting
    """
    return pe * euclidean(v1, v2) + pt * np.power(t2-t1, 2)