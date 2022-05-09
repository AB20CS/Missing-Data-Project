### Doubly Robust Model to Estimate Mean With Missing Data

from cmath import nan
import pandas as pd
import numpy as np
from math import sqrt
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.utils import check_random_state
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from scipy.optimize import fsolve


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def MCAR(X, p, random_state):
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

    n, d = X.shape
    mask = np.zeros((n, d))

    ber = rng.rand(n, d)
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

def gen_params(n_features, missing_rate, prop_latent, snr, masking,
               prop_for_masking=None, random_state=None):
    """Creates parameters for generating multivariate Gaussian data.

    Parameters
    ----------
    n_features: int
        The number of features desired.

    missing_rate: float
        The percentage of missing entries for each incomplete feature.
        Entries should be between 0 and 1.

    prop_latent: float
        The number of latent factors used to generate the covariance matrix is
        prop_latent*n_feature. The less factors the higher the correlations.
        Should be between 0 and 1.

    snr: float
        The desired signal to noise ratio.

    masking: str
        The desired masking type. One of 'MCAR', 'MAR_logistic'.

    prop_for_masking: float, default None
        The proportion of variables used in the logistic function for masking.
        It is not relevant if `masking == 'MCAR'` or
        `masking == 'MNAR_logistic'`.

    random_state : int, RandomState instance or None, optional, default None
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.
    """

    if missing_rate > 1 or missing_rate < 0:
        raise ValueError("missing_rate must be >= 0 and <= 1, got %s" %
                         missing_rate)

    if prop_latent > 1 or prop_latent < 0:
        raise ValueError("prop_latent should be between 0 and 1")

    if prop_for_masking and (prop_for_masking > 1 or prop_for_masking < 0):
        raise ValueError("prop_for_masking should be between 0 and 1")

    rng = check_random_state(random_state)

    # Generate covariance and mean
    # ---------------------------
    B = rng.randn(n_features, int(prop_latent*n_features))
    cov = B.dot(B.T) + np.diag(
        rng.uniform(low=0.01, high=0.1, size=n_features))

    mean = rng.randn(n_features)

    # Generate beta
    beta = np.repeat(1., n_features + 1)

    # Convert the desired signal-to-noise ratio to a noise variance
    var_Y = beta[1:].dot(cov).dot(beta[1:])
    sigma2_noise = var_Y/snr

    return (n_features, mean, cov, beta, sigma2_noise, masking, missing_rate,
            prop_for_masking)


def gen_data(n_sizes, data_params, random_state=None):

    rng = check_random_state(random_state)

    (n_features, mean, cov, beta, sigma2_noise, masking, missing_rate,
     prop_for_masking) = data_params

    X = np.empty((0, n_features))
    X_no_mask = np.empty((0, n_features))
    y = np.empty((0, ))
    current_size = 0

    for _, n_samples in enumerate(n_sizes):

        current_X = rng.multivariate_normal(
                mean=mean, cov=cov,
                size=n_samples-current_size,
                check_valid='raise')

        noise = rng.normal(
            loc=0, scale=sqrt(sigma2_noise), size=n_samples-current_size)
        current_y = beta[0] + current_X.dot(beta[1:]) + noise

        if masking == 'MCAR':
            current_M = MCAR(current_X, missing_rate, rng)
        elif masking == 'MAR_logistic':
            current_M = MAR_logistic(current_X, missing_rate, prop_for_masking,
                                     rng)

        X_no_mask = np.vstack((X_no_mask, current_X))

        np.putmask(current_X, current_M, np.nan)

        X = np.vstack((X, current_X))
        y = np.hstack((y, current_y))

        current_size = n_samples

        yield X, X_no_mask, y



def bayes_approx_Neumann(sigma, mu, beta, X, depth, typ='mcar', k=None,
                         tsigma2=None, gm_approx=None, init=None):
    pred = []
    for x in X:
        obs = np.where(~np.isnan(x))[0]
        mis = np.where(np.isnan(x))[0]

        n_obs = len(obs)

        sigma_obs = sigma[np.ix_(obs, obs)]
        sigma_mis_obs = sigma[np.ix_(mis, obs)]

        if depth >= 0:
            # Ensure convergence
            if len(obs) > 0:
                L = np.linalg.norm(sigma_obs, ord=2)
                sigma_obs /= L

            # Initialisation
            if init is not None:
                sigma_obs_inv = init[np.ix_(obs, obs)]
            else:
                sigma_obs_inv = np.eye(n_obs)

            for i in range(1, depth+1):
                sigma_obs_inv = (np.eye(n_obs) - sigma_obs).dot(
                    sigma_obs_inv) + np.eye(n_obs)

            # To counterbalance the fact that we divided sigma by L
            if len(obs) > 0:
                sigma_obs_inv /= L
        else:
            sigma_obs_inv = np.linalg.inv(sigma_obs)

        if typ == 'mcar':
            E_x_mis = mu[mis] + sigma_mis_obs.dot(
                sigma_obs_inv).dot(x[obs] - mu[obs])
        elif typ == 'gm':
            sigma_mis = sigma[np.ix_(mis, mis)]
            sigma_misobs = sigma_mis - sigma_mis_obs.dot(
                sigma_obs_inv).dot(sigma_mis_obs.T)
            sigma_misobs_inv = np.linalg.inv(sigma_misobs)
            D_mis = np.diag(tsigma2[mis])
            A = D_mis.dot(sigma_misobs_inv)

            alpha = np.mean(np.diag(A))
            D_A = np.diag(np.diag(A))
            D_A_inv = np.linalg.inv(D_A)
            tmu = mu + k*np.sqrt(np.diag(sigma))

            # Identity approx
            if gm_approx == 'Id':
                E_x_mis = (0.5*(tmu[mis] + mu[mis] + sigma_mis_obs.dot(
                    sigma_obs_inv).dot(x[obs] - mu[obs])))

            # alpha Id approx
            elif gm_approx == 'alphaId':
                E_x_mis = 1/(1+alpha)*(tmu[mis] + alpha*(
                    mu[mis] + sigma_mis_obs.dot(sigma_obs_inv).dot(
                        x[obs] - mu[obs])))

            # Diagonal matrix approx
            elif gm_approx == 'diagonal':
                E_x_mis = D_A_inv.dot(tmu[mis]) + mu[mis] + sigma_mis_obs.dot(
                    sigma_obs_inv).dot(x[obs] - mu[obs])

            # No approx
            else:
                E_x_mis = tmu[mis] + A.dot(mu[mis] + sigma_mis_obs.dot(
                    sigma_obs_inv).dot(x[obs] - mu[obs]))
                E_x_mis = np.linalg.inv(np.eye(len(mis)) + A).dot(E_x_mis)

        predx = beta[0] + beta[mis+1].dot(E_x_mis) + beta[obs+1].dot(x[obs])
        pred.append(predx)

    return np.array(pred)


def doubly_robust(df, df_pred, X, Y):
    DELTA = 'Delta_' + str(Y)

    if len(np.unique(df[DELTA].to_numpy())) == 1:
        return np.mean(df[Y].to_numpy())

    # model for propensity score
    pi = LogisticRegression(C=1e6, max_iter=1000).fit(df_pred[X], df[DELTA])
    pi = pi.predict_proba(df_pred[X])[:, 1]

    # regression model trained on observed outcomes
    s = LinearRegression().fit(df_pred[X], df_pred[Y]).predict(df_pred[X])
    mu_dr = np.mean(s + df[DELTA] * (df[Y]-s) * (1 / pi))
    return mu_dr



if __name__ == "__main__":
    n_features = 5
    # Generate data
    params = gen_params(
        n_features=n_features, missing_rate=0.5, prop_latent=0.5, snr=10,
        masking='MCAR', prop_for_masking=None, random_state=None)

    (n_features, mean, cov, beta, sigma2_noise, masking, missing_rate,
     prop_for_masking) = params

    gen = gen_data([100], params, random_state=None)
    X, X_no_mask, y = next(gen)


    X = pd.DataFrame(X)
    y = pd.DataFrame(y)
    X[len(X.columns)] = y

    X_neumiss = pd.DataFrame()
    all_columns = X.columns
    for i in X.columns:
        curr_columns = all_columns.drop(i)
        X_neumiss[i] = bayes_approx_Neumann(sigma=cov, mu=mean, beta=beta, X=X[curr_columns].to_numpy(), depth=9)

    pd.DataFrame(X_neumiss).to_csv('./synthetic_X_pred.csv', index=False, index_label=False)

    pd.DataFrame(X_no_mask).to_csv('./synthetic_X_no_mask.csv', index=False, index_label=False)
    pd.DataFrame(X).to_csv('./synthetic_X.csv', index=False, index_label=False)

    # Add Delta Columns
    for i in X.columns:
        Delta = []
        # Create vector Delta of Indicator values (0 if NaN and 1 otherwise)
        for j in X[i]:
            if np.isnan(j):
                Delta += [0]
            else:
                Delta += [1]
        Delta = np.array(Delta)
        X['Delta_' + str(i)] = Delta
        X_neumiss['Delta_' + str(i)] = Delta

    pd.DataFrame(X).to_csv('./synthetic_X_deltas.csv', index=False, index_label=False)
    
    # Impute with MICE
    imp = IterativeImputer(random_state=0)
    imp.fit(X.to_numpy())
    X_mice = pd.DataFrame(imp.transform(X.to_numpy()), columns=X.columns)

    neumiss_dr_means = []
    mice_dr_means = []
    for i in all_columns:
        y_col = i
        X_cols = all_columns.drop(i)

        neumiss_dr_means.append(doubly_robust(X, X_neumiss, X_cols, y_col))
        mice_dr_means.append(doubly_robust(X, X_mice, X_cols, y_col))
   
    naive_mean = np.mean(X_no_mask)

    neumiss_mean = np.mean(X_neumiss.to_numpy())
    mice_mean = np.mean(X_mice.to_numpy())

    neumiss_dr_mean = np.mean(neumiss_dr_means)
    mice_dr_mean = np.mean(mice_dr_means)

    print('Naive Calculation of Mean (on unmasked data): {}'.format(naive_mean))

    print('\nMean of All Data with NeuMiss (Bayes\' Predictor) Imputation - No Doubly Robust:\n\tmean={}, bias={}'.format(neumiss_mean, naive_mean - neumiss_mean))
    print('Mean of All Data with MICE Imputation - No Doubly Robust:\n\tmean={}, bias={}'.format(mice_mean, naive_mean - mice_mean))

    print('\nMean of All Data with NeuMiss (Bayes\' Predictor) Imputation and Doubly Robust:\n\tmean={}, bias={}'.format(neumiss_dr_mean, naive_mean - neumiss_dr_mean))
    print('Mean of All Data with MICE Imputation and Doubly Robust:\n\tmean={}, bias={}'.format(mice_dr_mean, naive_mean - mice_dr_mean))

