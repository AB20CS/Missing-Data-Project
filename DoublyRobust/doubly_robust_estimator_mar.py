### Doubly Robust Model to Estimate Mean With Missing Data

from cmath import nan
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.utils import check_random_state
from scipy.optimize import fsolve

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def MCAR_one_column(X, p, random_state):
    """
    Missing completely at random mechanism.
    Parameters
    ----------
    X : array-like, shape (n, d)
        Data for which missing values will be simulated.
    p : float
        Proportion of missing values to generate for variables which will have
        missing values.
    random_state: int, RandomState instance or None, optional, default None
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.
    Returns
    -------
    mask : array-like, shape (n, d)
        Mask of generated missing values (True if the value is missing).
    """

    rng = check_random_state(random_state)

    n = X.size
    mask = np.zeros((n, 1))

    ber = rng.rand(n, 1)
    mask = ber < p

    return mask

def MAR_logistic(X, p, Y, random_state):
    """
    Missing at random mechanism with a logistic masking model. First, a subset
    of variables with *no* missing values is randomly selected. The remaining
    variables have missing values according to a logistic model with random
    weights, but whose intercept is chosen so as to attain the desired
    proportion of missing values on those variables.
    Parameters
    ----------
    X : array-like, shape (n, d)
        Data for which missing values will be simulated.
    p : float
        Proportion of missing values to generate for variables which will have
        missing values.
    Y : variable in X with missing subjects
    random_state: int, RandomState instance or None, optional, default None
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.
    Returns
    -------
    mask : array-like, shape (n, d)
        Mask of generated missing values (True if the value is missing).
    """

    rng = check_random_state(random_state)

    n, d = X.shape
    mask = np.zeros((n, d))
    # number of variables that will have no missing values
    # (at least one variable)
    d_obs = d-1
    # number of variables that will have missing values
    d_na = d - d_obs

    # Sample variables that will all be observed, and those with missing values
    idxs_obs = []
    for col in X.columns:
        if col != Y:
            idxs_obs += [X.columns.get_loc(col)]
    idxs_nas = [X.columns.get_loc(Y)]

    # Other variables will have NA proportions that depend on those observed
    # variables, through a logistic model. The parameters of this logistic
    # model are random, and adapted to the scale of each variable.
    X = X.to_numpy()
    mu = X.mean(0)
    cov = (X-mu).T.dot(X-mu)/n
    cov_obs = cov[np.ix_(idxs_obs, idxs_obs)]
    coeffs = rng.randn(d_obs, d_na)

    v = np.array([coeffs[:, j].dot(cov_obs).dot(
        coeffs[:, j]) for j in range(d_na)])
    steepness = rng.uniform(low=0.1, high=0.5, size=d_na)
    coeffs /= steepness*np.sqrt(v)

    # Move the intercept to have the desired amount of missing values
    intercepts = np.zeros((d_na))
    for j in range(d_na):
        w = coeffs[:, j]

        def f(b):
            s = sigmoid(X[:, idxs_obs].dot(w) + b) - p
            return s.mean()

        res = fsolve(f, x0=0)
        intercepts[j] = res[0]

    ps = sigmoid(X[:, idxs_obs].dot(coeffs) + intercepts)
    ber = rng.rand(n, d_na)
    mask[:, idxs_nas] = ber < ps

    return mask


def horvitz_thompson(df, X, Y):
    DELTA = 'Delta'

    # model for propensity score
    pi = LogisticRegression(C=1e6, max_iter=1000).fit(df[X], df[DELTA]).predict_proba(df[X])[:, 1]
    
    mu_ht = (np.mean((df[DELTA] * df[Y]) / pi))
    return mu_ht


def outcome_regression(df, X, Y):
    Delta_1 = df # will hold only rows with no missing values

    # delete rows with missing values
    for i, row in df.iterrows():
        if np.isnan(row[Y]):
            Delta_1 = Delta_1.drop(labels=i, axis=0)
    
    # regression model trained on observed outcomes
    s = LinearRegression().fit(Delta_1[X], Delta_1[Y]).predict(Delta_1[X])
    
    mu_or = np.mean(s)
    return mu_or


def doubly_robust(df, X, Y):
    DELTA = 'Delta'
    Delta_1 = df # will hold only rows with no missing values
    # delete rows with missing values
    for i, row in df.iterrows():
        if np.isnan(row[Y]):
            Delta_1 = Delta_1.drop(labels=i, axis=0)

    # model for propensity score
    pi = LogisticRegression(C=1e6, max_iter=1000).fit(df[X], df[DELTA]).predict_proba(df[X])[:, 1]
    
    # regression model trained on observed outcomes
    s = LinearRegression().fit(Delta_1[X], Delta_1[Y]).predict(df[X])
    mu_dr = np.mean(s + df[DELTA] * (df[Y]-s) * (1 / pi))
    return mu_dr


def doubly_robust_incorrect_pi(df, df_incorrect, X, Y):
    DELTA = 'Delta'
    Delta_1 = df # will hold only rows with no missing values
    # delete rows with missing values
    for i, row in df.iterrows():
        if np.isnan(row[Y]):
            Delta_1 = Delta_1.drop(labels=i, axis=0)

    # model for propensity score
    pi = LogisticRegression(C=1e6, max_iter=1000).fit(df_incorrect[X], df_incorrect[DELTA]).predict_proba(df[X])[:, 1]
    
    # regression model trained on observed outcomes
    s = LinearRegression().fit(Delta_1[X], Delta_1[Y]).predict(df[X])
    mu_dr = np.mean(s + df[DELTA] * (df[Y]-s) * (1 / pi))
    return mu_dr


def doubly_robust_incorrect_s(df, df_incorrect, X, Y):
    DELTA = 'Delta'
    Delta_1 = df_incorrect # will hold only rows with no missing values in incorrectly masked data
    # delete rows with missing values
    for i, row in df_incorrect.iterrows():
        if np.isnan(row[Y]):
            Delta_1 = Delta_1.drop(labels=i, axis=0)

    # model for propensity score
    pi = LogisticRegression(C=1e6, max_iter=1000).fit(df[X], df[DELTA]).predict_proba(df[X])[:, 1]
    
    # regression model trained on observed outcomes
    s = LinearRegression().fit(Delta_1[X], Delta_1[Y]).predict(df[X])
    mu_dr = np.mean(s + df[DELTA] * (df[Y]-s) * (1 / pi))
    return mu_dr


def doubly_robust_incorrect_pi_and_s(df, df_incorrect, X, Y):
    DELTA = 'Delta'
    Delta_1 = df_incorrect # will hold only rows with no missing values in incorrectly masked data
    # delete rows with missing values
    for i, row in df_incorrect.iterrows():
        if np.isnan(row[Y]):
            Delta_1 = Delta_1.drop(labels=i, axis=0)

    # model for propensity score
    pi = LogisticRegression(C=1e6, max_iter=1000).fit(df_incorrect[X], df_incorrect[DELTA]).predict_proba(df[X])[:, 1]
    
    # regression model trained on observed outcomes
    s = LinearRegression().fit(Delta_1[X], Delta_1[Y]).predict(df[X])
    mu_dr = np.mean(s + df[DELTA] * (df[Y]-s) * (1 / pi))
    return mu_dr


if __name__ == "__main__":
    T = 'intervention'
    Y = 'achievement_score'

    data = pd.read_csv("./learning_mindset.csv")
    categ = ["ethnicity", "gender", "school_urbanicity"]
    cont = ["school_mindset", "school_achievement", "school_ethnic_minority", "school_poverty", "school_size"]

    data_with_categ = pd.concat([
        data.drop(columns=categ), # dataset without the categorical features
        pd.get_dummies(data[categ], columns=categ, drop_first=False) # categorical features converted to dummies
    ], axis=1)

    all_columns = data_with_categ.columns
    Y_loc = all_columns.get_loc(Y)

    X = data_with_categ.columns.drop(['schoolid', T, Y])

    # save data without mask for later
    data_without_mask = data_with_categ

    # apply MAR mask   
    M = MAR_logistic(data_with_categ, 0.4, Y, None) 
    data_with_categ = data_with_categ.to_numpy()
    np.putmask(data_with_categ, M, np.nan)

    Delta = []
    # Create vector Delta of Indicator values (0 if NaN and 1 otherwise)
    for i in data_with_categ[:,Y_loc]:
        if np.isnan(i):
            Delta += [0]
        else:
            Delta += [1]

    data_with_categ = pd.DataFrame(data_with_categ, columns=all_columns)

    # Add column Delta
    data_with_categ['Delta'] = Delta

    data_with_categ.to_csv("./learning_mindset_masked.csv")

    naive_mu = data_with_categ[Y].mean()
    print('Naive Calculation of Mean: mu = {}'.format(naive_mu))

    # Correctly specified models
    print("\nCorrectly specified models")

    mu_ht = horvitz_thompson(data_with_categ, X, Y)
    print('\tHorvitz-Thompson Esimator: mu_HT = {}, bias = {}'.format(mu_ht, naive_mu - mu_ht))

    mu_or = outcome_regression(data_with_categ, X, Y)
    print('\tOutcome Regression Esimator: mu_OR = {}, bias = {}'.format(mu_or, naive_mu - mu_or))

    mu_dr = doubly_robust(data_with_categ, X, Y)
    print('\tDoubly Robust Esimator: mu_dr = {}, bias = {}'.format(mu_dr, naive_mu - mu_dr))

    # Generate incorrectly masked data
    incorrecly_masked_data = data_without_mask
    incorrect_Y = incorrecly_masked_data[Y].to_numpy()
    incorrect_Y_mask = MCAR_one_column(incorrect_Y, 0.8, None)
    np.putmask(incorrect_Y, incorrect_Y_mask, np.nan)
    incorrecly_masked_data[Y] = incorrect_Y.tolist()

    Delta_incorrect = []
    # Create vector Delta of Indicator values (0 if NaN and 1 otherwise)
    
    for i in incorrect_Y_mask:
        if np.isnan(i):
            Delta_incorrect += [0]
        else:
            Delta_incorrect += [1]

    incorrecly_masked_data = pd.DataFrame(incorrecly_masked_data, columns=all_columns)

    # Add column Delta_incorrect
    incorrecly_masked_data['Delta'] = Delta

    incorrecly_masked_data.to_csv("./learning_mindset_masked_incorrect.csv")

    # Incorrectly specified Horvitz-Thompson Estimator
    print("\nIncorrectly specified Horvitz-Thompson Estimator")

    mu_ht = horvitz_thompson(incorrecly_masked_data, X, Y)
    print('\tHorvitz-Thompson Esimator: mu_HT = {}, bias = {}'.format(mu_ht, naive_mu - mu_ht))

    # Incorrectly specified Outcome Regression Estimator
    print("\nIncorrectly specified Outcome Regression Estimator")
    mu_or = outcome_regression(incorrecly_masked_data, X, Y)
    print('\tOutcome Regression Esimator: mu_OR = {}, bias = {}'.format(mu_or, naive_mu - mu_or))

    # Incorrectly specified Propensity Score Model
    print("\nIncorrectly specified Propensity Score Model (pi)")
    mu_dr = doubly_robust_incorrect_pi(data_with_categ, incorrecly_masked_data, X, Y)
    print('\tDoubly Robust Esimator: mu_dr = {}, bias = {}'.format(mu_dr, naive_mu - mu_dr))

    # Incorrectly specified Linear Regression Model
    print("\nIncorrectly specified Linear Regression Model (s)")
    mu_dr = doubly_robust_incorrect_s(data_with_categ, incorrecly_masked_data, X, Y)
    print('\tDoubly Robust Esimator: mu_dr = {}, bias = {}'.format(mu_dr, naive_mu - mu_dr))

    # Incorrectly specified pi and s models
    print("\nIncorrectly specified pi and s models")
    mu_dr = doubly_robust_incorrect_pi_and_s(data_with_categ, incorrecly_masked_data, X, Y)
    print('\tDoubly Robust Esimator: mu_dr = {}, bias = {}'.format(mu_dr, naive_mu - mu_dr))

