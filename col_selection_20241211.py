# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 10:49:42 2024

@author: Johnson
"""

!pip install numpy matplotlib scikit-learn pandas


import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.impute import SimpleImputer
import pandas as pd


# Helper function to compute f(i, A)
def compute_f(i, A):
    """
    Computes the value of f(i, A) as defined in Equation (8):
    f(i, A) = -||A[:, i]||^2 / A[i, i] * I(A[i, i] > 0)
    """
    if A[i, i] > 0:
        norm_squared = np.linalg.norm(A[:, i])**2
        return -norm_squared / A[i, i]
    else:
        return float('inf')  # Set to infinity to avoid selecting this index

# Function to compute the residual covariance matrix directly
def compute_residual_covariance(Sigma, S):
    """
    Computes the residual covariance Σ_R(X, X_S) directly:
    Σ_R(X, X_S) = Σ - Σ_U Σ_U^+ Σ_U^T
    where Σ_U is the submatrix of Σ corresponding to subset S.
    
    Parameters:
    - Sigma: Original covariance matrix
    - S: Subset of indices
    
    Returns:
    - Residual covariance matrix
    """
    if not S:  # If S is empty, return the original covariance matrix
        return Sigma
    Sigma_S = Sigma[np.ix_(S, S)]  # Submatrix Σ_S
    Sigma_S_inv = np.linalg.pinv(Sigma_S)  # Pseudo-inverse of Σ_S
    Sigma_S_cols = Sigma[:, S]  # Columns of Σ corresponding to S
    return Sigma - Sigma_S_cols @ Sigma_S_inv @ Sigma_S_cols.T

# Algorithm 1: Greedy Subset Selection
def greedy_subset_selection(Sigma, k):
    """
    Implements Algorithm 1: Greedy Subset Selection.
    
    Parameters:
    - Sigma: Covariance matrix
    - k: Number of variables to select
    
    Returns:
    - Selected subset of indices
    """
    n = Sigma.shape[0]
    S = []  # Initialize empty set
    Sigma_residual = Sigma.copy()  # Initialize residual covariance matrix
    
    for t in range(k):
        scores = [compute_f(i, Sigma_residual) for i in range(n) if i not in S]
        candidates = [i for i in range(n) if i not in S]
        i_star = candidates[np.argmin(scores)]  # Find the index of the minimum score
        S.append(i_star)
        Sigma_residual = compute_residual_covariance(Sigma, S)  # Recompute residual covariance
    
    return S

# Algorithm 2: Subset Selection by Swapping
def swapping_subset_selection(Sigma, k, S_initial):
    """
    Implements Algorithm 2: Subset Selection by Swapping.
    
    Parameters:
    - Sigma: Covariance matrix
    - k: Number of variables to select
    - S_initial: Initial subset of indices
    
    Returns:
    - Final refined subset of indices
    """
    S = S_initial.copy()
    n = Sigma.shape[0]
    Sigma_residual = compute_residual_covariance(Sigma, S)  # Initial residual covariance matrix
    
    for iterations in range(100):
        S_copy = S.copy()
        
        for j in range(k):
            # Remove the j-th variable from S
            U = S[:j] + S[j+1:]  # Subset without S[j]
            Sigma_residual_U = compute_residual_covariance(Sigma, U)
            
            # Find the best variable to add to U
            scores = [compute_f(i, Sigma_residual_U) for i in range(n) if i not in U]
            candidates = [i for i in range(n) if i not in U]
            i_star = candidates[np.argmin(scores)]
            
            # Swap variables if it improves the score
            if i_star not in S:
                S[j] = i_star
        
        # Terminate when no changes are made
        if S == S_copy:
            break
    return S

# Generate the linear synthetic data with missing values
def generate_data(n_samples=40, n_features=20, missing_rate=0.1):
    X = np.random.randn(n_samples, n_features)
    true_coefficients = np.random.randn(n_features)
    y = X @ true_coefficients + np.random.randn(n_samples) * 0.1  # Add noise
    
    # Introduce missing values
    mask = np.random.rand(*X.shape) < missing_rate
    X[mask] = np.nan  # Set some entries to NaN
    return X, y, true_coefficients

# Generating the dataset as described in Appendix A.1
def generate_dataset(n, p, k, sigma=1.0, random_seed=None):

    if random_seed is not None:
        np.random.seed(random_seed)
    
    # Step 1: Generate X_S (k principal variables) from some distribution F
    # Here, we assume F is a standard normal distribution
    X_S = np.random.normal(0, 1, size=(n, k))
    
    # Step 2: Define W (coefficient matrix) and mu_{-S}
    # W is a (p-k) x k matrix, randomly initialized
    W = np.array([
        [np.sqrt(17/90), np.sqrt(17/90), np.sqrt(17/90), 0],
        [np.sqrt(17/50), np.sqrt(17/50), -np.sqrt(17/50), 0],
        [np.sqrt(17/50), -np.sqrt(17/50), np.sqrt(17/50), 0],
        [np.sqrt(17/50), -np.sqrt(17/50), -np.sqrt(17/50), 0],
        [-np.sqrt(17/50), np.sqrt(17/50), np.sqrt(17/50), 0],
        [-np.sqrt(17/50), np.sqrt(17/50), -np.sqrt(17/50), 0],
        [-np.sqrt(17/50), -np.sqrt(17/50), np.sqrt(17/50), 0],
        [-np.sqrt(17/90), -np.sqrt(17/90), -np.sqrt(17/90), 0],
        [0,np.sqrt(17/90), np.sqrt(17/90), np.sqrt(17/90)],
        [0,np.sqrt(17/50), np.sqrt(17/50), -np.sqrt(17/50)],
        [0,np.sqrt(17/50), -np.sqrt(17/50), np.sqrt(17/50)],
        [0,np.sqrt(17/50), -np.sqrt(17/50), -np.sqrt(17/50)],
        [0,-np.sqrt(17/50), np.sqrt(17/50), np.sqrt(17/50)],
        [0,-np.sqrt(17/50), np.sqrt(17/50), -np.sqrt(17/50)],
        [0,-np.sqrt(17/50), -np.sqrt(17/50), np.sqrt(17/50)],
        [0,-np.sqrt(17/90), -np.sqrt(17/90), -np.sqrt(17/90)],
    ])

    
    # mu_{-S} is the mean vector for the remaining variables
    mu_minus_S = np.random.normal(0, 1, size=(p-k,))
    
    # Step 3: Generate X_{-S} given X_S
    # E_F[X_S] is the mean of X_S, which is zero for a standard normal distribution
    # Add Gaussian noise with variance sigma^2
    noise = np.random.normal(0, sigma, size=(n, p-k))
    X_minus_S = mu_minus_S + X_S @ W.T + noise
    
    # Step 4: Combine X_S and X_{-S} to form the full dataset X
    X = np.hstack([X_S, X_minus_S])
    
    # Optionally, define y (labels for supervised tasks) as a function of X
    # Here, we generate y as a linear function of X_S plus noise
    beta = np.random.normal(0, 1, size=(k,))
    y = X_S @ beta + np.random.normal(0, sigma, size=n)
    
    return X, y

# Calculate Covariance Matrix with Missing Values
def covariance_with_missing(X):
    # Number of samples and features
    n_samples, n_features = X.shape
    # Initialize covariance matrix
    Sigma = np.zeros((n_features, n_features))

    # Calculate covariance using available data
    for i in range(n_features):
        for j in range(n_features):
            # Get indices of non-missing data
            valid_indices = ~np.isnan(X[:, i]) & ~np.isnan(X[:, j])
            if np.sum(valid_indices) > 1:  # Ensure there are enough samples
                Sigma[i, j] = np.cov(X[valid_indices, i], X[valid_indices, j])[0, 1]

    # Project onto the positive semi-definite cone
    # Calculate eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(Sigma)
    # Set negative eigenvalues to zero
    eigenvalues[eigenvalues < 0] = 0
    # Reconstruct the covariance matrix
    Sigma_positive = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T

    return Sigma_positive

# Principal Component Analysis (PCA)
def pca_selection(X, k):
    pca = PCA(n_components=k)
    X_reduced = pca.fit_transform(X)
    return X_reduced

# Function for evaluating both PCA, Greedy and Swapping CSS given features, labels and number of variables.
def evaluate_methods(X, y, n_variables_list):
    r2_scores = {"Greedy": [], "Swapping": [], "PCA": []}
    
    for k in n_variables_list:
        # Create a mask for non-NaN values
        mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
        X_masked = X[mask]
        y_masked = y[mask]

        if X_masked.shape[0] == 0:
            print(f"No valid samples for k={k}. Skipping.")
            continue

        Sigma = covariance_with_missing(X_masked)

        # Greedy Subset Selection
        greedy_indices = greedy_subset_selection(Sigma, k)
        X_greedy = X_masked[:, greedy_indices]

        imputer = SimpleImputer(strategy='mean')
        X_greedy_imputed = imputer.fit_transform(X_greedy)

        model_greedy = LinearRegression().fit(X_greedy_imputed, y_masked)
        y_pred_greedy = model_greedy.predict(X_greedy_imputed)
        r2_scores["Greedy"].append(r2_score(y_masked, y_pred_greedy))
        
        # Subset Selection by Swapping
        initial_indices = list(range(k))
        swapping_indices = swapping_subset_selection(Sigma, k, initial_indices)
        X_swapping = X_masked[:, swapping_indices]

        X_swapping_imputed = imputer.fit_transform(X_swapping)
        model_swapping = LinearRegression().fit(X_swapping_imputed, y_masked)
        y_pred_swapping = model_swapping.predict(X_swapping_imputed)
        r2_scores["Swapping"].append(r2_score(y_masked, y_pred_swapping))
        
        # PCA
        X_pca = pca_selection(X_masked, k)
        X_pca_imputed = imputer.fit_transform(X_pca)
        model_pca = LinearRegression().fit(X_pca_imputed, y_masked)
        y_pred_pca = model_pca.predict(X_pca_imputed)
        r2_scores["PCA"].append(r2_score(y_masked, y_pred_pca))
      

    return r2_scores

# function for evaluating the accuracy of algorithms in terms of number of correct columns picked
def evaluate_column_accuracy(X, y, n_variables_list):
    r2_scores = {"Greedy": [], "Swapping": []}
    
    for k in n_variables_list:
        # Create a mask for non-NaN values
        mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
        X_masked = X[mask]

        if X_masked.shape[0] == 0:
            print(f"No valid samples for k={k}. Skipping.")
            continue

        Sigma = covariance_with_missing(X_masked)

        # Greedy Subset Selection
        greedy_indices = greedy_subset_selection(Sigma, k)
        
        r2_scores["Greedy"].append(len(set([0,1,2,3]) & set(greedy_indices)))
        
        # Subset Selection by Swapping
        initial_indices = list(np.random.randint(20, size = 4)) #list(range(k))
        swapping_indices = swapping_subset_selection(Sigma, k, initial_indices)

        r2_scores["Swapping"].append(len(set([0,1,2,3]) & set(swapping_indices)))
        

      

    return r2_scores

# Plot R-squared results
def plot_results(n_variables_list, r2_scores, title):
    plt.figure(figsize=(10, 6), dpi = 300)
    plt.plot(n_variables_list, r2_scores["PCA"], label="PCA", marker='^')
    plt.plot(n_variables_list, r2_scores["Greedy"], label="Greedy CSS", marker='o')
    plt.plot(n_variables_list, r2_scores["Swapping"], label="Swapping CSS", marker='s')

    plt.xlabel("Number of Variables/Components", fontweight = "bold")
    plt.ylabel("R-squared", fontweight = "bold")
    plt.title(title, fontweight = "bold")
    plt.legend()
    plt.grid()
    plt.show()



'''
##################################
#####       Evaluation       #####
##################################

'''


if True: # 1. Evaluation based on the California housing dataset
    np.random.seed(0)

    n_tests = 1
    
    n_variables_list = range(1, 13)

    r2_scores = {"Greedy": np.zeros(len(n_variables_list)),
                         "Swapping": np.zeros(len(n_variables_list)),
                         "PCA": np.zeros(len(n_variables_list))}

    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/housing.csv"
    data = pd.read_csv(url)    
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    r2_scores = evaluate_methods(X, y, n_variables_list)


    plot_results(n_variables_list, r2_scores, "R-squared Performance Comparison - Housing dataset")

    print("Results for housing data:")        

    for method in r2_scores.keys():
        print(f"Average R-squared for {method}: {r2_scores[method]}")
        
    print("")        

 

if True: # 2.  Evaluation using simulation study
    np.random.seed(0)

    # Number of tests to perform
    n_tests = 100
    # Define the range of variables/components to evaluate
    n_variables_list = list(range(1,31, 1))
    
    # Initialize a dictionary to accumulate R-squared scores
    average_r2_scores = {"Greedy": np.zeros(len(n_variables_list)),
                         "Swapping": np.zeros(len(n_variables_list)),
                         "PCA": np.zeros(len(n_variables_list))}
    
    n_test = 0
    
    while n_test < n_tests:
        # Generate synthetic data with missing values
        
        try:
            X, y, true_coefficients = generate_data(n_samples=200, n_features=30, missing_rate=0.05)
    
            # Evaluate methods
            r2_scores = evaluate_methods(X, y, n_variables_list)
            n_test += 1
        except:
            continue

    
        # Accumulate R-squared scores
        for method in average_r2_scores.keys():
            average_r2_scores[method] += np.array(r2_scores[method])
            
        plot_results(n_variables_list, average_r2_scores, "R-squared Performance Comparison")

    print("Results for simulation study:")        
    
    # Calculate the average R-squared scores
    for method in average_r2_scores.keys():
        average_r2_scores[method] /= n_tests
        print(f"Average R-squared for {method}: {average_r2_scores[method]}")
        
    plot_results(n_variables_list, average_r2_scores, "R-squared Performance Comparison - Simulation Study")
    print("")        



        
        
if True: # 3. Evaluation of number of correct columns selected
   np.random.seed(0)
    
   n = 100  # Number of samples
   p = 20   # Total number of variables
   k = 4    # Number of principal variables
   sigma = np.sqrt(0.15)  # Residual variance

   num_cor_cols_all = {"Greedy": 0, "Swapping": 0}
   
   correct_subset_greedy = 0
   correct_subset_swapping = 0
   
   for n_tries in range(100):
       
       X, y = generate_dataset(n, p, k, sigma)
       mask = np.random.rand(n, p) > 0.05  # Create a mask
       X_missing = X * mask  # Apply the mask to X
    
       num_cor_cols = evaluate_column_accuracy(X_missing, y, [4])
       
                            
       
       for method in num_cor_cols.keys():
           num_cor_cols_all[method] += num_cor_cols[method][0]     
           
       if num_cor_cols["Greedy"][0] == 4:
           correct_subset_greedy += 1
       if num_cor_cols["Swapping"][0] == 4:
           correct_subset_swapping += 1



   print("Greedy CSS selected " + str(num_cor_cols_all["Greedy"] / 100) + " correct columns on average" )
   print("Swapping CSS selected " + str(num_cor_cols_all["Swapping"] / 100) + " correct columns on average" )
   
   print("Greedy CSS selected the correct subset " + str(correct_subset_greedy) + " % of the time" )
   print("Swapping CSS selected the correct subset " + str(correct_subset_swapping) + " % of the time" )




