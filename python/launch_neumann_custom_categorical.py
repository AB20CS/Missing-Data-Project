# Same as launch_simu_neumann_single.py except data is not simulated
# for categorical y values

import numpy as np
from neumannS0_mlp_categorical import Neumann_mlp, Neumann
from ground_truth import gen_params, gen_data
from ground_truth import BayesPredictor_MCAR_MAR
import pandas as pd
from collections import namedtuple
from sklearn.metrics import log_loss
import torch
import torch.nn as nn

from joblib import Memory, Parallel, delayed
location = './cachedir'
memory = Memory(location, verbose=0)

fields = ['iter', 'method', 'train_test', 'log_loss', 'depth']
ResultItem = namedtuple('ResultItem', fields)
ResultItem.__new__.__defaults__ = (np.nan, )*len(ResultItem._fields)

DATASET_FILENAME = '../datasets/diabetes.csv' # change filename depending on dataset used
RESULTS_OUTPUT_FILENAME = '../results/diabetes_results.csv'


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


def get_score(pred, y):
    loss = nn.BCELoss()
    return loss(torch.tensor(pred).float(), torch.tensor(y).float()).item()


@memory.cache
def run_one_iter(it, n_features):
    result_iter = []
    # generate parameters
    # params = gen_params(
    #     n_features=n_features, missing_rate=0.5, prop_latent=0.5, snr=10,
    #     masking='MCAR', prop_for_masking=None, random_state=it)
    #
    # (n_features, mean, cov, beta, sigma2_noise, masking, missing_rate,
    #  prop_for_masking) = params

    # load data from dataset
    data = np.genfromtxt(DATASET_FILENAME, delimiter=',')
    data = data[1:]  # remove header

    X = []
    y = np.empty(len(data))
    for i in range(len(data)):
        y[i] = data[i][-1:]
        X.append(np.delete(data[i], -1))  # remove last element

    X = np.asarray(X)
    y = np.asarray(y)

    X_test = X[0:n_test]
    y_test = y[0:n_test]
    X_val = X[n_test:(n_test + n_val)]
    y_val = y[n_test:(n_test + n_val)]
    X_train = X[(n_test + n_val):]
    y_train = y[(n_test + n_val):]

    # Run Neumann with various depths, and with and without residual connection
    d = 9
    for res in [True, False]:
        print('Neumann d={}, res={}'.format(d, res))
        est = Neumann_mlp(
            depth=d, n_epochs=100, batch_size=10, lr=1e-2/n_features,
            early_stopping=False, residual_connection=res, verbose=True)

        est.fit(X_train, y_train, X_val=X_val, y_val=y_val)

        est.net = Neumann(n_features=n_features, depth=d,
                          residual_connection=res,
                          mlp_depth=0, init_type='normal')

        pred_test = est.predict(X_test)
        perf_test = get_score(pred_test, y_test)
        pred_train = est.predict(X)
        perf_train = get_score(pred_train, y)

        if res:
            method = 'Neumann_res'
        else:
            method = 'Neumann'

        res_train = ResultItem(iter=it, method=method, depth=d,
                               train_test="train", log_loss=perf_train)
        res_test = ResultItem(iter=it, method=method, depth=d,
                              train_test="test", log_loss=perf_test)
        result_iter.extend([res_train, res_test])
        print(result_iter)

    # # Run the Bayes predictor approximated with Neumann
    # depths = [0, 1, 2, 3, 4, 5, 10, 15, 20, 30, 40, 50, 100]
    # for d in depths:
    #     pred = bayes_approx_Neumann(sigma=cov, mu=mean, beta=beta,
    #                                 X=X_test, depth=d)
    #     perf = get_score(pred, y_test)
    #     res = ResultItem(iter=it, method='Neumann_analytic', depth=d, log_loss=perf)
    #     result_iter.extend([res])
    #
    # # Compute the Bayes rate
    # bp = BayesPredictor_MCAR_MAR(params)
    # pred_test = bp.predict(X_test)
    # perf_test = get_score(pred_test, y_test)
    # res_test = ResultItem(
    #     iter=it, method='Bayes_rate', train_test="test", log_loss=perf)
    # pred_train = bp.predict(X_train)
    # perf_train = get_score(pred_train, y_train)
    #
    # res_train = ResultItem(
    #     iter=it, method='Bayes_rate', train_test="train", log_loss=perf)
    # result_iter.extend([res_train, res_test])

    return result_iter


if __name__ == '__main__':
    n = int(614)
    n_test = int(77)
    n_val = int(77)
    n_iter = 1
    n_features = 8
    n_jobs = 1

    results = Parallel(n_jobs=n_jobs)(
            delayed(run_one_iter)(it, n_features=n_features)
            for it in range(n_iter)
        )
    results = [item for result_iter in results for item in result_iter]
    results = pd.DataFrame(results)

    results.to_csv(RESULTS_OUTPUT_FILENAME)
