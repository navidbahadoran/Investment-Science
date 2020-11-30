import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import os

yearly_return = np.array([[0.02, 0.025, 0.04]]).T

Table = dict(enumerate(yearly_return))

yearly_cov = np.array(
    [[0.38 ** 2, 0.3 * 0.38 * 0.3, 0.3 * 0.38 * 0.5], [0.3 * 0.38 * 0.3, 0.3 ** 2, 0.1 * 0.3 * 0.5],
     [0.3 * 0.38 * 0.5, 0.1 * 0.3 * 0.5, 0.5 ** 2]])


# finding global minimum variance
def global_min(asset_cov, asset_return):
    one_vector = np.ones((asset_return.shape[0], 1))
    A = float(one_vector.T.dot(np.linalg.inv(asset_cov)).dot(one_vector))
    B = float(one_vector.T.dot(np.linalg.inv(asset_cov)).dot(asset_return))
    C = float(asset_return.T.dot(np.linalg.inv(asset_cov)).dot(asset_return))
    global_portfolio_phi = A * C - (B ** 2)
    global_portfolio_var = 1 / A
    global_portfolio_weight = np.linalg.inv(asset_cov).dot(one_vector) * (1 / A)
    portfolio_mean = B / A

    return global_portfolio_var, portfolio_mean, global_portfolio_weight, global_portfolio_phi


def quad_max_graph(g_var, g_return, g_phi):
    risk_averse = 0.3
    quad_return = g_return + (1 / risk_averse) * g_var * g_phi
    quad_var = g_var + (1 / (risk_averse ** 2)) * g_var * g_phi
    c = quad_return - (risk_averse / 2) * quad_var
    portfolio_std = np.arange(0, 0.6, 0.0001)
    portfolio_mean = (portfolio_std * portfolio_std) * (risk_averse / 2) + np.ones(len(portfolio_std)) * c
    plt.plot(math.sqrt(quad_var), quad_return, 'o', label='tangent point')
    plt.plot(portfolio_std, portfolio_mean, color="r", label="Indifference Curve")
    plt.xlabel('std')
    plt.ylabel('mean')
    plt.grid(True)


def tangent_graph(asset_cov, asset_return):
    risk_free = 0.01
    one_vector = np.ones((asset_return.shape[0], 1))
    excess_return = asset_return - one_vector * risk_free
    risk_averse = float(excess_return.T.dot(np.linalg.inv(asset_cov)).dot(one_vector))
    sharpe = float(excess_return.T.dot(np.linalg.inv(asset_cov)).dot(excess_return))
    tangent_return = risk_free + (1 / risk_averse) * sharpe
    tangent_var = 1 / (risk_averse ** 2) * sharpe
    print(tangent_return,math.sqrt(tangent_var))
    portfolio_mean = np.arange(risk_free, 0.06, 0.0001)
    print(portfolio_mean)
    portfolio_std = (portfolio_mean - (np.ones(len(portfolio_mean)) * risk_free)) / math.sqrt(sharpe)
    print(portfolio_std)
    plt.plot(portfolio_std, portfolio_mean, color='g', label="Tangent portfolio line")
    plt.plot(math.sqrt(tangent_var), tangent_return, 'o', label='Tangent portfolio')




# sketch the graph for mean-variance boundary
def mean_var_graph(g_var, g_return, g_phi, graph_text, graph_color):
    portfolio_mean = np.arange(0, 0.06, 0.00001)
    diff = portfolio_mean - np.ones(len(portfolio_mean)) * g_return
    portfolio_std = np.sqrt(np.ones(len(portfolio_mean)) * g_var + (diff * diff) * (1 / (g_phi * g_var)))
    plt.plot(portfolio_std, portfolio_mean, color=graph_color, label=f"Mean Variance Boundary{graph_text}")
    plt.plot(math.sqrt(g_var), g_return, 'o', label=f'global minimum variance{graph_text}')
    plt.xlabel('std')
    plt.ylabel('mean')
    plt.grid(True)


# sketch the assets points
def plot_points(data_table, asset_cov, asset_return):
    for i, asset in enumerate(data_table.keys()):
        plt.plot(math.sqrt(asset_cov[i, i]), asset_return[i, 0], 'o', label=asset)


global_var, global_return, global_weight, global_phi = global_min(yearly_cov, yearly_return)

plot_points(Table, yearly_cov, yearly_return)
quad_max_graph(global_var, global_return, global_phi)
mean_var_graph(global_var, global_return, global_phi, "", "k")
tangent_graph(yearly_cov, yearly_return)
plt.title('Mean Variance Boundary')
plt.legend(loc="upper left", ncol=2, prop={'size': 6})
plt.tight_layout()
plt.show()
# printing global minimum variance  weight, return and std
for i, asset in enumerate(Table.keys()):
    print("{}:Global wight: {}, Yearly std: {}, Yearly return: {} ".format(asset,
                                                                           global_weight[i, 0],
                                                                           math.sqrt(yearly_cov[i, i]),
                                                                           yearly_return[i, 0]))

print("Global return: ", global_return)
print("Global Std: ", math.sqrt(global_var))

# calculate daily and yearly return and covariance matrix

# one_vector = np.ones((data.shape[0], 1))
# daily_return = data.mean(axis=0).T
# yearly_return = np.power(daily_return + np.ones((10, 1)), one_vector * 252) - one_vector

# daily_cov = np.cov(data, rowvar=False, bias=True)
# yearly_cov = np.power(daily_cov + (1 + daily_return) * (1 + daily_return).T, np.ones((10, 10)) * 252) - \
#              np.power((1 + daily_return) * (1 + daily_return).T, np.ones((10, 10)) * 252)
# # calculate correlation from convarince matrix


# v = np.sqrt(np.diag(yearly_cov))
# outer_v = np.outer(v, v)
# correlation = yearly_cov / outer_v
# correlation[yearly_cov == 0] = 0


## new correlation which has 0.1 difference in cross correlation
# change_factor = -0.1
# change_corr = correlation + np.ones((10, 10)) * change_factor
# np.fill_diagonal(change_corr, val=1)
# new_yearly_cov = np.multiply(change_corr, outer_v)


# new_global_var, new_global_return, new_global_weight, new_phi = global_min(new_yearly_cov, yearly_return)
# print(global_var, global_return, global_phi)
# mean_var_graph(new_global_var, new_global_return, new_phi, " Corr-0.1", "r")


# printing new global minimum variance  weight, return and std after changing the correlation
# for i, asset in enumerate(Table.keys()):
#     print("{}:New Global wight: {}, New Yearly std: {}, New Yearly return: {}".format(asset,
#                                                                                       new_global_weight[i, 0],
#                                                                                       math.sqrt(new_yearly_cov[i, i]),
#                                                                                       yearly_return[i, 0]))

# printing global min variance return and std with and without change in correlation

# print("Global return after change in Corr: ", new_global_return)
# print("Global Std after change in Corr: ", math.sqrt(new_global_var))
