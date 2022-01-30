# first line: 25
@memory.cache
def run_one(X, y, est, params, method, n_test, n_val):
    n, p = X.shape
    n = n - n_val - n_test
    if method == 'torchMLP':
        print('method: {}, dim: {}, width: {}'.format(
            method, (n, p), params['hidden_layer_sizes'][0]))
    elif method == 'Neumann':
        print('method: {}, dim: {}, depth: {}, early_stop: {}, res: {}'.format(
            method, (n, p), params['depth'], params['early_stopping'],
            params['residual_connection']))
    else:
        print('method: {}, dim: {}'.format(method, (n, p)))

    X_test = X[0:n_test]
    y_test = y[0:n_test]
    X_val = X[n_test:(n_test + n_val)]
    y_val = y[n_test:(n_test + n_val)]
    X_train = X[(n_test + n_val):]
    y_train = y[(n_test + n_val):]

    if n_val > 0 and method in ['Neumann', 'torchMLP']:
        reg = est(**params)
        reg.fit(X_train, y_train, X_val=X_val, y_val=y_val)
    elif method == 'MICEMLP':
        X_train = X[n_test:]
        y_train = y[n_test:]
        reg = est(params)
        reg.fit(X_train, y_train)
    else:
        reg = est(**params)
        reg.fit(X_train, y_train)

    pred_test = reg.predict(X_test)
    pred_train = reg.predict(X_train)
    pred_val = reg.predict(X_val)

    mse_train = ((y_train - pred_train)**2).mean()
    mse_test = ((y_test - pred_test)**2).mean()
    mse_val = ((y_val - pred_val)**2).mean()

    var_train = ((y_train - y_train.mean())**2).mean()
    var_test = ((y_test - y_test.mean())**2).mean()
    var_val = ((y_val - y_val.mean())**2).mean()

    r2_train = 1 - mse_train/var_train
    r2_test = 1 - mse_test/var_test
    r2_val = 1 - mse_val/var_val

    return {'train': {'mse': mse_train, 'r2': r2_train},
            'test': {'mse': mse_test, 'r2': r2_test},
            'val': {'mse': mse_val, 'r2': r2_val}
            }
