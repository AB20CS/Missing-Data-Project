# first line: 102
@memory.cache
def run_one_iter(it, n_features):
    result_iter = []

    # Generate data
    params = gen_params(
        n_features=n_features, missing_rate=0.5, prop_latent=0.5, snr=10,
        masking='MCAR', prop_for_masking=None, random_state=it)

    (n_features, mean, cov, beta, sigma2_noise, masking, missing_rate,
     prop_for_masking) = params

    gen = gen_data([120000], params, random_state=it)
    X, y = next(gen)

    X_test = X[0:n_test]
    y_test = y[0:n_test]
    X_val = X[n_test:(n_test + n_val)]
    y_val = y[n_test:(n_test + n_val)]
    X_train = X[(n_test + n_val):]
    y_train = y[(n_test + n_val):]

    # Run Neumann with various depths, and with and without residual connection
    for d in range(10):
        for res in [True, False]:
            print('Neumann d={}, res={}'.format(d, res))
            est = Neumann_mlp(
                depth=d, n_epochs=100, batch_size=10, lr=1e-2/n_features,
                early_stopping=False, residual_connection=res, verbose=False)

            est.fit(X_train, y_train, X_val=X_val, y_val=y_val)

            pred_test = est.predict(X_test)
            perf_test = get_score(pred_test, y_test)

            pred_train = est.predict(X)
            perf_train = get_score(pred_train, y)

            if res:
                method = 'Neumann_res'
            else:
                method = 'Neumann'

            res_train = ResultItem(iter=it, method=method, depth=d,
                                   train_test="train", r2=perf_train)
            res_test = ResultItem(iter=it, method=method, depth=d,
                                  train_test="test", r2=perf_test)
            result_iter.extend([res_train, res_test])

    # Run the Bayes predictor approximated with Neumann
    depths = [0, 1, 2, 3, 4, 5, 10, 15, 20, 30, 40, 50, 100]
    for d in depths:
        pred = bayes_approx_Neumann(sigma=cov, mu=mean, beta=beta,
                                    X=X_test, depth=d)
        perf = get_score(pred, y_test)
        res = ResultItem(iter=it, method='Neumann_analytic', depth=d, r2=perf)
        result_iter.extend([res])

    # Compute the Bayes rate
    bp = BayesPredictor_MCAR_MAR(params)
    pred_test = bp.predict(X_test)
    perf_test = get_score(pred_test, y_test)
    res_test = ResultItem(
        iter=it, method='Bayes_rate', train_test="test", r2=perf)
    pred_train = bp.predict(X_train)
    perf_train = get_score(pred_train, y_train)
    res_train = ResultItem(
        iter=it, method='Bayes_rate', train_test="train", r2=perf)
    result_iter.extend([res_train, res_test])

    return result_iter
