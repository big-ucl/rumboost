import numpy as np

def lin(x, a, b):
    return a*x + b

def sqr(x, a, b, c):
    return a*x**2 + b*x + c

def cub(x, a, b, c, d):
    return a*x**3 + b*x**2 + c*x + d

def exp(x, a, b, c):
    return a*np.exp(-b*x) + c

def log(x, a, b):
    return a*np.log(x) + b

def pow(x, a, b):
    return x**a + b

def logistic(x, a, b, c):
    return c / (1 + b*np.exp(a*x))

def sin(x, a, b, c, d):
    return a*np.sin(b*x +c) + d

def inv(x, a, b, c):
    return a / (x + b) + c

def lin_exp(x, a, b, c, d):
    return a*x + b*np.exp(-c*x) + d

def lin_log(x, a, b, c, d):
    return a*x + b*np.log(c*x) + d

def lin_inv(x, a, b, c, d):
    return a*x + b/(x + c) + d

def exp_log(x, a, b, c, d, e):
    return a*np.exp(-b*x) + c*np.log(d*x) + e

def exp_inv(x, a, b, c, d, e):
    return a*np.exp(-b*x) + c/(x + d) + e

def log_inv(x, a, b, c, d, e):
    return a*np.log(b*x) + c/(x + d) + e

def lin_exp_log(x, a, b, c, d, e, f):
    return a*x + b*np.exp(-c*x) + d*np.log(e*x) + f

def lin_exp_inv(x, a, b, c, d, e, f):
    return a*x + b*np.exp(-c*x) + d/(x + e) + f

def lin_log_inv(x, a, b, c, d, e, f):
    return a*x + b*np.log(c*x) + d/(x + e) + f

def exp_log_inv(x, a, b, c, d, e, f, g):
    return a*np.exp(-b*x) + c*np.log(d*x) + e/(x + f) + g

def lin_exp_log_inv(x, a, b, c, d, e, f, g, h):
    return a*x + b*np.exp(-c*x) + d*np.log(e*x) + f/(x + g) + h

def func_wrapper():
    return {'lin':lin, 'sqr':sqr, 'cub':cub, 'exp':exp, 'log':log, 'logistic':logistic, 'sin':sin, 'inv':inv, 
            'lin_exp':lin_exp, 'lin_log':lin_log, 'lin_inv':lin_inv, 'exp_log':exp_log, 'exp_inv':exp_inv, 
            'log_inv':log_inv, 'lin_exp_log':lin_exp_log, 'lin_exp_inv':lin_exp_inv, 'lin_log_inv':lin_log_inv,
            'exp_log_inv':exp_log_inv, 'lin_exp_log_inv':lin_exp_log_inv}

def all_func(x, a, b, c, d, e, f, g, h, i, j):
    return a*x**3 + b*x**2 + c*x + d*np.exp(-e*x) + f*np.log(g*x) + h/(x + i) + j

def penalty_neg(x, a, b, c, d, e, f, g, h, i, j):
    return (np.sign(np.grad(all_func(x, a, b, c, d, e, f, g, h, i, j))) + 1)*1000000

def penalty_pos(x, a, b, c, d, e, f, g, h, i, j):
    return (np.sign(np.grad(all_func(x, a, b, c, d, e, f, g, h, i, j))) - 1)*1000000

def all_func_neg(x, a, b, c, d, e, f, g, h, i, j):
    return all_func(x, a, b, c, d, e, f, g, h, i, j) + penalty_neg(x, a, b, c, d, e, f, g, h, i, j)

def all_func_pos(x, a, b, c, d, e, f, g, h, i, j):
    return all_func(x, a, b, c, d, e, f, g, h, i, j) + penalty_pos(x, a, b, c, d, e, f, g, h, i, j)

def all_func_1step(x, a, b, c, d, e, f, g, h, i, j, k, l, m):
    return all_func(x, a, b, c, d, e, f, g, h, i, j) + logistic(x, k, l, m)

def all_func_neg_1step(x, a, b, c, d, e, f, g, h, i, j, k, l, m):
    return all_func(x, a, b, c, d, e, f, g, h, i, j) + penalty_neg(x, a, b, c, d, e, f, g, h, i, j) + logistic(x, k, l, m)

def all_func_pos_1step(x, a, b, c, d, e, f, g, h, i, j, k, l, m):
    return all_func(x, a, b, c, d, e, f, g, h, i, j) + penalty_pos(x, a, b, c, d, e, f, g, h, i, j) + logistic(x, k, l, m)

def all_func_2step(x, a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p):
    return all_func(x, a, b, c, d, e, f, g, h, i, j) + logistic(x, k, l, m) + logistic(x, n, o, p)

def all_func_neg_2step(x, a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p):
    return all_func(x, a, b, c, d, e, f, g, h, i, j) + penalty_neg(x, a, b, c, d, e, f, g, h, i, j) + logistic(x, k, l, m) + logistic(x, n, o, p)

def all_func_pos_2step(x, a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p):
    return all_func(x, a, b, c, d, e, f, g, h, i, j) + penalty_pos(x, a, b, c, d, e, f, g, h, i, j) + logistic(x, k, l, m) + logistic(x, n, o, p)

def all_func_3step(x, a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s):
    return all_func(x, a, b, c, d, e, f, g, h, i, j) + logistic(x, k, l, m) + logistic(x, n, o, p) + logistic(x, q, r, s)

def all_func_neg_3step(x, a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s):
    return all_func(x, a, b, c, d, e, f, g, h, i, j) + penalty_neg(x, a, b, c, d, e, f, g, h, i, j) + logistic(x, k, l, m) + logistic(x, n, o, p) + logistic(x, q, r, s)

def all_func_pos_3step(x, a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s):
    return all_func(x, a, b, c, d, e, f, g, h, i, j) + penalty_pos(x, a, b, c, d, e, f, g, h, i, j) + logistic(x, k, l, m) + logistic(x, n, o, p) + logistic(x, q, r, s)