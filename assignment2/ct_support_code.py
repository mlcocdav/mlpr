# You are expected to use this support code.
# You may want to write:
# from ct_support_code import *
# at the top of your answers

# You will need NumPy and SciPy:
import numpy as np
from scipy.optimize import minimize
from scipy.linalg import cho_factor, cho_solve
from scipy.stats import norm
import matplotlib.pyplot as plt


def params_unwrap(param_vec, shapes, sizes):
    """Helper routine for minimize_list"""
    args = []
    pos = 0
    for i in range(len(shapes)):
        sz = sizes[i]
        args.append(param_vec[pos:pos+sz].reshape(shapes[i]))
        pos += sz
    return args


def params_wrap(param_list):
    """Helper routine for minimize_list"""
    param_list = [np.array(x) for x in param_list]
    shapes = [x.shape for x in param_list]
    sizes = [x.size for x in param_list]
    param_vec = np.zeros(sum(sizes))
    pos = 0
    for param in param_list:
        sz = param.size
        param_vec[pos:pos+sz] = param.ravel()
        pos += sz
    unwrap = lambda pvec: params_unwrap(pvec, shapes, sizes)
    return param_vec, unwrap


def minimize_list(cost, init_list, args):
    """Optimize a list of arrays (wrapper of scipy.optimize.minimize)

    The input function "cost" should take a list of parameters,
    followed by any extra arguments:
        cost(init_list, *args)
    should return the cost of the initial condition, and a list in the same
    format as init_list giving gradients of the cost wrt the parameters.

    The options to the optimizer have been hard-coded. You may wish
    to change disp to True to get more diagnostics. You may want to
    decrease maxiter while debugging. Although please report all results
    in Q2-5 using maxiter=500.

    The Matlab code comes with a different optimizer, so won't give the same
    results.
    """
    opt = {'maxiter': 1000, 'disp': False}
    init, unwrap = params_wrap(init_list)
    def wrap_cost(vec, *args):
        E, params_bar = cost(unwrap(vec), *args)
        vec_bar, _ = params_wrap(params_bar)
        return E, vec_bar
    res = minimize(wrap_cost, init, args, 'L-BFGS-B', jac=True, options=opt)
    return unwrap(res.x)


def linreg_cost(params, X, yy, alpha):
    """Regularized least squares cost function and gradients

    Can be optimized with minimize_list -- see fit_linreg_gradopt for a
    demonstration.

    Inputs:
    params: tuple (ww, bb): weights ww (D,), bias bb scalar
         X: N,D design matrix of input features
        yy: N,  real-valued targets
     alpha: regularization constant

    Outputs: (E, [ww_bar, bb_bar]), cost and gradients
    """
    # Unpack parameters from list
    ww, bb = params

    # forward computation of error
    ff = np.dot(X, ww) + bb
    res = ff - yy
    E = np.dot(res, res) + alpha*np.dot(ww, ww)

    # reverse computation of gradients
    ff_bar = 2*res
    bb_bar = np.sum(ff_bar)
    ww_bar = np.dot(X.T, ff_bar) + 2*alpha*ww

    return E, [ww_bar, bb_bar]


def fit_linreg_gradopt(X, yy, alpha):
    """
    fit a regularized linear regression model with gradient opt

         ww, bb = fit_linreg_gradopt(X, yy, alpha)

     Find weights and bias by using a gradient-based optimizer
     (minimize_list) to improve the regularized least squares cost:

       np.sum(((np.dot(X,ww) + bb) - yy)**2) + alpha*np.dot(ww,ww)

     Inputs:
             X N,D design matrix of input features
            yy N,  real-valued targets
         alpha     scalar regularization constant

     Outputs:
            ww D,  fitted weights
            bb     scalar fitted bias
    """
    D = X.shape[1]
    args = (X, yy, alpha)
    init = (np.zeros(D), np.array(0))
    ww, bb = minimize_list(linreg_cost, init, args)
    return ww, bb


def logreg_cost(params, X, yy, alpha):
    """Regularized logistic regression cost function and gradients

    Can be optimized with minimize_list -- see fit_linreg_gradopt for a
    demonstration of fitting a similar function.

    Inputs:
    params: tuple (ww, bb): weights ww (D,), bias bb scalar
         X: N,D design matrix of input features
        yy: N,  real-valued targets
     alpha: regularization constant

    Outputs: (E, [ww_bar, bb_bar]), cost and gradients
    """
    # Unpack parameters from list
    ww, bb = params

    # Force targets to be +/- 1
    yy = 2*(yy==1) - 1

    # forward computation of error
    aa = yy*(np.dot(X, ww) + bb)
    sigma = 1/(1 + np.exp(-aa))
    E = -np.sum(np.log(sigma)) + alpha*np.dot(ww, ww)

    # reverse computation of gradients
    aa_bar = sigma - 1
    bb_bar = np.dot(aa_bar, yy)
    ww_bar = np.dot(X.T, yy*aa_bar) + 2*alpha*ww

    return E, (ww_bar, bb_bar)


def nn_cost(params, X, yy=None, alpha=None):
    """NN_COST simple neural network cost function and gradients, or predictions

           E, params_bar = nn_cost([ww, bb, V, bk], X, yy, alpha)
                    pred = nn_cost([ww, bb, V, bk], X)

     Cost function E can be minimized with minimize_list

     Inputs:
             params (ww, bb, V, bk), where:
                    --------------------------------
                        ww K,  hidden-output weights
                        bb     scalar output bias
                         V K,D hidden-input weights
                        bk K,  hidden biases
                    --------------------------------
                  X N,D input design matrix
                 yy N,  regression targets
              alpha     scalar regularization for weights

     Outputs:
                     E  sum of squares error
            params_bar  gradients wrt params, same format as params
     OR
               pred N,  predictions if only params and X are given as inputs
    """
    # Unpack parameters from list
    ww, bb, V, bk = params

    # Forwards computation of cost
    A = np.dot(X, V.T) + bk[None,:] # N,K
    P = 1 / (1 + np.exp(-A)) # N,K
    F = np.dot(P, ww) + bb # N,
    if yy is None:
        # user wants prediction rather than training signal:
        return F
    res = F - yy # N,
    E = np.dot(res, res) + alpha*(np.sum(V*V) + np.dot(ww,ww)) # 1x1

    # Reverse computation of gradients
    F_bar = 2*res # N,
    ww_bar = np.dot(P.T, F_bar) + 2*alpha*ww # K,
    bb_bar = np.sum(F_bar) # scalar
    P_bar = np.dot(F_bar[:,None], ww[None,:]) # N,
    A_bar = P_bar * P * (1 - P) # N,
    V_bar = np.dot(A_bar.T, X) + 2*alpha*V # K,
    bk_bar = np.sum(A_bar, 0)

    return E, (ww_bar, bb_bar, V_bar, bk_bar)


def rbf_fn(X1, X2):
    """Helper routine for gp_post_par"""
    return np.exp((np.dot(X1,(2*X2.T))-np.sum(X1*X1,1)[:,None]) - np.sum(X2*X2,1)[None,:])


def gauss_kernel_fn(X1, X2, ell, sigma_f):
    """Helper routine for gp_post_par"""
    return sigma_f**2 * rbf_fn(X1/(np.sqrt(2)*ell), X2/(np.sqrt(2)*ell))


def gp_post_par(X_rest, X_obs, yy, sigma_y=0.01, ell=5.0, sigma_f=0.01):
    """GP_POST_PAR means and covariances of a posterior Gaussian process

         rest_cond_mu, rest_cond_cov = gp_post_par(X_rest, X_obs, yy)
         rest_cond_mu, rest_cond_cov = gp_post_par(X_rest, X_obs, yy, sigma_y, ell, sigma_f)

     Calculate the means and covariances at all test locations of the posterior Gaussian
     process conditioned on the observations yy at observed locations X_obs.

     Inputs:
                 X_rest GP test locations
                  X_obs locations of observations
                     yy observed values
                sigma_y observation noise standard deviation
                    ell kernel function length scale
                sigma_f kernel function standard deviation

     Outputs:
           rest_cond_mu mean at each location in X_rest
          rest_cond_cov covariance matrix between function values at all test locations
    """
    X_rest = X_rest[:, None]
    X_obs = X_obs[:, None]
    K_rest = gauss_kernel_fn(X_rest, X_rest, ell, sigma_f)
    K_rest_obs = gauss_kernel_fn(X_rest, X_obs, ell, sigma_f)
    K_obs = gauss_kernel_fn(X_obs, X_obs, ell, sigma_f)
    M = K_obs + sigma_y**2 * np.eye(yy.size)
    M_cho, M_low = cho_factor(M)
    rest_cond_mu = np.dot(K_rest_obs, cho_solve((M_cho, M_low), yy))
    rest_cond_cov = K_rest - np.dot(K_rest_obs, cho_solve((M_cho, M_low), K_rest_obs.T))

    return rest_cond_mu, rest_cond_cov

# my shit from here
data = np.load('ct_data.npz')
X_train = data['X_train']; X_val = data['X_val']; X_test = data['X_test']
y_train = data['y_train']; y_val = data['y_val']; y_test = data['y_test']
alpha=30

#a = fit_linreg_gradopt(X_train, y_train, alpha)
#exit()


#
# #train_w2, train_b2 = fit_linreg_gradopt(X_train, y_train, alpha)
# K = 20 # number of thresholded classification problems to fit
# mx = np.max(y_train); mn = np.min(y_train); hh = (mx-mn)/(K+1)
# thresholds = np.linspace(mn+hh, mx-hh, num=K, endpoint=True)
#
#
# def fit_logreg_gradopt(X, yy, alpha):
#     D = X.shape[1]
#     args = (X, yy, alpha)
#     init = (np.zeros(D), np.array(0))
#     ww, bb = minimize_list(logreg_cost, init, args)
#     return ww, bb
#
# def logreg_forward(X, ww, bb):
#     aa = (X @ ww) + bb
#     return 1 / (1 + np.exp(-aa))
#
# N = X_train.shape[0]
# D = X_train.shape[1]
# WW = np.empty((D, K))
# BB = np.empty(K)
# LL = np.empty((N, K))
# for kk in range(K):
#     labels = y_train > thresholds[kk]
#     LL[:, kk] = labels
#     WW[:,kk], BB[kk] = fit_logreg_gradopt(X_train, labels, alpha)
#
#
# X_train_logreg = logreg_forward(X_train, WW, BB)
# X_val_logreg = logreg_forward(X_val, WW, BB)
#
# train_w_lr, train_b_lr = fit_linreg_gradopt(X_train_logreg, y_train, alpha)
# print('Training RMSE logreg transformation (fit_linreg_gradopt): ',
#       rmse(train_w_lr, train_b_lr, X_train_logreg, y_train))
# print('Validation RMSE logreg transformation (fit_linreg_gradopt): ',
#       rmse(train_w_lr, train_b_lr, X_val_logreg, y_val))
# """ Lower RMSE than normal linear regression """

#%%
# 4

def fit_nn_gradopt(X, yy, alpha, K=20, init=None):
    D = X.shape[1]
    args = (X, yy, alpha)
    if init == None:
        init = (np.random.randn(K), np.array(0), np.random.randn(K, D), np.random.randn(K))
    ww, bb, V, bk = minimize_list(nn_cost, init, args)
    return ww, bb, V, bk


def nn_rmse(params, XX, yy):
    ww, bb, V, bk = params

    A = np.dot(XX, V.T) + bk[None, :]  # N,K
    P = 1 / (1 + np.exp(-A))  # N,K
    F = np.dot(P, ww) + bb  # N,
    E = np.sqrt(np.mean((F - yy) ** 2))
    return E


# nn_rand_params = fit_nn_gradopt(X_train, y_train, 30)
# q3_params = (train_w_lr, train_b_lr, WW.T, BB)
# nn_q3_params = fit_nn_gradopt(X_train, y_train, 30, init=q3_params)


def train_nn_reg(X_train, y_train, X_val, y_val, alpha):
    params = fit_nn_gradopt(X_train, y_train, alpha)
    return nn_rmse(params, X_val, y_val)

def prob_imp(mu, cov, yy, alphas, alpha):
    a_idx = np.where(alphas==alpha)[0][0]
    #pi = norm.cdf((mu[a_idx]-np.max(yy))/np.sqrt(cov[a_idx,a_idx]))
    pi = (mu[a_idx] - np.max(yy)) / np.sqrt(cov[a_idx, a_idx])
    return pi

alphas = np.arange(0, 50, 0.02)
idx = np.round(len(alphas)*np.array([0.25,0.5,0.75])).astype(int)
y_train_gp = np.array([])
train_alphas = alphas[idx]
for alpha in train_alphas:
    y_train_gp = np.append(y_train_gp, -np.log(train_nn_reg(X_train, y_train, X_val, y_val, alpha)))
nn_rand_params = fit_nn_gradopt(X_train, y_train, 30)
baseline = np.log(nn_rmse(nn_rand_params, X_val, y_val))
y_train_gp = baseline + y_train_gp
test_alphas = np.delete(alphas, idx)

for i in range(5):
    mu, cov = gp_post_par(test_alphas, train_alphas, y_train_gp)

    # plt.plot(test_alphas, mu, '-k', linewidth=2)
    # std = np.sqrt(np.diag(cov))
    # plt.plot(test_alphas, mu + 2 * std, '--k', linewidth=2)
    # plt.plot(test_alphas, mu - 2 * std, '--k', linewidth=2)
    # plt.show()

    best_alpha = test_alphas[0]
    best_pi = - 1e100
    for alpha in test_alphas:
        pi = prob_imp(mu, cov, y_train_gp, test_alphas, alpha)
        if pi > best_pi:
            best_pi = pi
            best_alpha = alpha
    print(best_alpha, best_pi)
    train_alphas = np.append(train_alphas, best_alpha)
    test_alphas = np.delete(test_alphas, np.where(test_alphas==best_alpha))
    #y_train_gp = np.append(y_train_gp, - np.log(train_nn_reg(X_train, y_train, X_val, y_val, best_alpha)))
    y_train_gp = np.append(y_train_gp, baseline -np.log(train_nn_reg(X_train, y_train, X_val, y_val, best_alpha)))
best_alpha = train_alphas[np.argmax(y_train_gp)]

val_rmse = train_nn_reg(X_train, y_train, X_val, y_val, best_alpha)
test_rmse = train_nn_reg(X_train, y_train, X_test, y_test, best_alpha)
print(train_alphas)
print('Best alpha: ', best_alpha)
print('Val RMSE: ', val_rmse)
print('Test RMSE: ', test_rmse)


def nn2_cost(params, X, yy=None, alpha=None):
    """NN_COST simple neural network cost function and gradients, or predictions

           E, params_bar = nn_cost([ww, bb, V, bk], X, yy, alpha)
                    pred = nn_cost([ww, bb, V, bk], X)

     Cost function E can be minimized with minimize_list

     Inputs:
             params (ww, bb, V, bk), where:
                    --------------------------------
                        ww K,  hidden-output weights
                        bb     scalar output bias
                         V K,D hidden-input weights
                        bk K,  hidden biases
                    --------------------------------
                  X N,D input design matrix
                 yy N,  regression targets
              alpha     scalar regularization for weights

     Outputs:
                     E  sum of squares error
            params_bar  gradients wrt params, same format as params
     OR
               pred N,  predictions if only params and X are given as inputs
    """
    # Unpack parameters from list
    ww, bb, V, bk, V2, b2 = params

    # Forwards computation of cost
    A = np.dot(X, V.T) + bk[None, :]  # N,K
    P = 1 / (1 + np.exp(-A))  # N,K
    B = np.dot(P, V2.T) + b2[None, :]  #
    P2 = 1 / (1 + np.exp(-B))  # N,K
    F = np.dot(P2, ww) + bb  # N,
    if yy is None:
        # user wants prediction rather than training signal:
        return F
    res = F - yy  # N,
    E = np.dot(res, res) + alpha * (np.sum(V * V) + np.dot(ww, ww) + np.sum(V2 * V2) )  # 1x1

    # Reverse computation of gradients
    F_bar = 2 * res  # N,
    ww_bar = np.dot(P.T, F_bar) + 2 * alpha * ww  # K,
    bb_bar = np.sum(F_bar)  # scalar
    P_bar = np.dot(F_bar[:, None], ww[None, :])  # N,
    A_bar = P_bar * P * (1 - P)  # N,
    V_bar = np.dot(A_bar.T, X) + 2 * alpha * V  # K,
    bk_bar = np.sum(A_bar, 0)

    return E, (ww_bar, bb_bar, V_bar, bk_bar, V2_bar, b2_bar)


def fit_nn2_gradopt(X, yy, alpha, K=64, H=32, init=None):
    D = X.shape[1]
    args = (X, yy, alpha)
    if init == None:
        init = (np.random.randn(H), np.array(0), np.random.randn(K, D), np.random.randn(K),
                np.random.randn(H, K), np.random.randn(H))
    ww, bb, V, bk, V2, b2 = minimize_list(nn2_cost, init, args)
    return ww, bb, V, bk, V2, b2


def nn2_rmse(params, XX, yy):
    ww, bb, V, bk, V2, b2 = params

    A = np.dot(XX, V.T) + bk[None, :]  # N,H
    P = 1 / (1 + np.exp(-A))  # N,H
    B = np.dot(P, V2.T) + b2[None, :]  #
    P2 = 1 / (1 + np.exp(-B))  # N,K
    F = np.dot(P2, ww) + bb  # N,
    E = np.sqrt(np.mean((F - yy) ** 2))


    return E


K = 64
H = 32
# params = (np.random.randn(H), np.array(0), np.random.randn(K, D), np.random.randn(K),
#                 np.random.randn(H,K), np.random.randn(H))
# print(nn2_rmse(params,X_val, y_val))
#nn_rand_params = fit_nn2_gradopt(X_train, y_train, 30)
