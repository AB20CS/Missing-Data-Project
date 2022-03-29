# Calculates the difference (MSE) between the actual and estimated covariance matrices

import pandas as pd
import numpy as np
from numpy import genfromtxt
import csv
from sklearn.metrics import mean_squared_error

ACTUAL_COV_FILENAME = '../results/diabetes_complete_unmasked_cov.csv'
MCAR_COV_FILENAME = '../results/diabetes_complete_mcar_cov.csv'
MAR_COV_FILENAME = '../results/diabetes_complete_mar_cov.csv'
MNAR_COV_FILENAME = '../results/diabetes_complete_mnar_cov.csv'

OUTPUT_FILENAME = '../results/diabetes_complete_cov_comparisons.csv'

actual_cov = genfromtxt(ACTUAL_COV_FILENAME, delimiter=',')
mcar_cov = genfromtxt(MCAR_COV_FILENAME, delimiter=',')
mar_cov = genfromtxt(MAR_COV_FILENAME, delimiter=',')
mnar_cov = genfromtxt(MNAR_COV_FILENAME, delimiter=',')

mse = [] # mse[0] - MCAR, mse[1] - MAR, mse[2] - MNAR
mse += [mean_squared_error(actual_cov, mcar_cov)]  # Calculate MSE for MCAR covariance matrix
mse += [mean_squared_error(actual_cov, mar_cov)]  # Calculate MSE for MAR covariance matrix
mse += [mean_squared_error(actual_cov, mnar_cov)]  # Calculate MSE for MNAR covariance matrix

with open(OUTPUT_FILENAME, 'w') as output:
    writer = csv.writer(output, delimiter=',')
    writer.writerow(['MCAR', 'MAR', 'MNAR'])  # write headers
    writer.writerow(mse)  # write values

