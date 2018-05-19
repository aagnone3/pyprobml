import os
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import (
    inv
)
from scipy.stats import (
    uniform,
    norm,
    multivariate_normal as mvn
)

np.random.seed(0)

# number of samples to draw from the posterior distribution
# of the parameters.
n_test_samples = 10

# the amount of data to represent in each plot of the
# posterior distribution of the parameters.
data_amounts = [0, 1, 2, 100]

# true regression weights in [-1, 1] to estimate.
a0 = -0.3
a1 = 0.5

# number of (x, y) training vectors
n_train_samples = 100
# true noise standard deviation
noise_stdev = 0.2
# alpha, the prior precision.
prior_precision = 2.0
# here, we assume the likelihood precision, beta, is known.
likelihood_stdev = noise_stdev
likelihood_precision = 1.0 / (likelihood_stdev ** 2)

# generate the training data
x = 2 * uniform().rvs(n_train_samples) - 1
y = a0 + a1 * x + norm(0, noise_stdev).rvs(n_train_samples)


def posterior(x, y):
    # given data vectors x and y, compute the posterior mean and covariance.
    X = np.array(
        [
            [1, vec]
            for vec in x
        ]
    )
    precision = np.diag([prior_precision] * 2) + likelihood_precision * X.T.dot(X)
    covariance = inv(precision)
    mean = likelihood_precision * covariance.dot(X.T.dot(y))
    return mean, covariance


def vectorize_mvn_pdf(mean, covariance):
    # given a mean and covariance, return a vectorized Gaussian pdf function.
    def out(w1, w2):
        return mvn.pdf([w1, w2], mean=mean, cov=covariance)
    return np.vectorize(out)


def vectorize_likelihood(x0, y0):
    # given a (x, y) data pair, return a vectorized likelihood function.
    def out(w1, w2):
        err = y0 - (w1 + w2 * x0)
        return norm.pdf(err, loc=0, scale=likelihood_stdev)
    return np.vectorize(out)


def graph_edits(whitemark=False):
    if whitemark:
        plt.ylabel(r'$w_1$')
        plt.xlabel(r'$w_0$')
        plt.scatter(a0, a1, marker='+', color='white', s=100)
    else:
        plt.ylabel('y')
        plt.xlabel('x')
    plt.ylim([-1, 1])
    plt.xlim([-1, 1])
    plt.xticks([-1, 0, 1])
    plt.yticks([-1, 0, 1])


# define the grid over which values will be defined
grid = np.linspace(-1, 1, 50)
Xg = np.array([
    [1, point] for point in grid
])
G1, G2 = np.meshgrid(grid, grid)

# add some transparency to lines if there are several of them.
alpha = 5.0 / n_test_samples if n_test_samples > 50 else 1.0

fig_cnt = 1
fig = plt.figure()

ax = fig.add_subplot(len(data_amounts), 3, fig_cnt)
ax.set_title('Likelihood')
plt.axis('off')

for di in data_amounts:
    if di == 0:
        posterior_mean = [0, 0]
        posterior_covariance = np.diag([1.0 / prior_precision] * 2)
    else:
        posterior_mean, posterior_covariance = posterior(x[:di], y[:di])

        fig_cnt += 1
        fig.add_subplot(len(data_amounts), 3, fig_cnt)
        likelihood_func = vectorize_likelihood(x[di-1], y[di-1])
        plt.contourf(G1, G2, likelihood_func(G1, G2), 100)
        graph_edits(True)


    posterior_func = vectorize_mvn_pdf(posterior_mean, posterior_covariance)
    fig_cnt += 1
    ax = fig.add_subplot(len(data_amounts), 3, fig_cnt)
    plt.contour(G1, G2, posterior_func(G1, G2), 100)
    graph_edits(True)
    if fig_cnt == 2:
        ax.set_title('prior/posterior')

    samples = mvn(posterior_mean, posterior_covariance).rvs(n_test_samples)
    lines = Xg.dot(samples.T)
    fig_cnt += 1
    ax = fig.add_subplot(len(data_amounts), 3, fig_cnt)
    if di != 0:
        plt.scatter(x[:di], y[:di], s=140, facecolors='none', edgecolors='b')
    for j in range(lines.shape[1]):
        plt.plot(grid, lines[:, j], linewidth=2, color='r', alpha=alpha)

    if fig_cnt == 3:
        ax.set_title('data space')
    graph_edits(False)

fig.tight_layout()
plt.show()
