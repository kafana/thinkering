import numpy as np
from scipy import optimize
import cvxopt
from cvxopt import blas, solvers


solvers.options['show_progress'] = False


class PortfolioOptimizer(object):

    @classmethod
    def optimize_weights1(cls, portfolio):
        portfolio_size = len(portfolio.columns)
        returns = portfolio.pct_change()
        mean_return = np.array(returns.mean())
        annualized_return = np.round(mean_return * 252.0, 2)
        cov_matrix = np.multiply(returns.cov(), 252.0)

        def portfolio_return(x):
            return np.array(np.dot(x.T, annualized_return))

        def portfolio_var(x):
            return np.array((np.dot(np.dot(x.T, cov_matrix), x)))

        def target(x):
            return np.array(-1 * (0.1 * portfolio_return(x) - portfolio_var(x)))

        # Optimize
        initial_guess = np.random.random(portfolio_size)
        initial_guess = initial_guess / sum(initial_guess)
        out = optimize.minimize(target, initial_guess, bounds=tuple([(0.05, 1)] * portfolio_size))
        out.x = out.x / np.sum(np.abs(out.x))

        weights = {}
        for i in range(portfolio_size):
            weights[portfolio.columns[i]] = out.x[i]

        return weights

    @classmethod
    def optimize_weights2(cls, portfolio):
        """
        Markowitz-style portfolio optimization
        https://blog.quantopian.com/markowitz-portfolio-optimization-2/
        """
        if len(portfolio.columns) == 1:
            return {portfolio.columns[0]: 1.0}, None, None
        returns = portfolio.pct_change().dropna()
        returns = returns.T.values
        n = len(returns)
        returns = np.asmatrix(returns)

        N = 100
        mus = [10 ** (5.0 * t/N - 1.0) for t in range(N)]

        # Convert to cvxopt matrices
        S = cvxopt.matrix(np.cov(returns))
        pbar = cvxopt.matrix(np.mean(returns, axis=1))

        # Create constraint matrices
        G = -cvxopt.matrix(np.eye(n))   # negative n x n identity matrix
        h = cvxopt.matrix(0.0, (n ,1))
        A = cvxopt.matrix(1.0, (1, n))
        b = cvxopt.matrix(1.0)

        # Calculate efficient frontier weights using quadratic programming
        portfolios = [solvers.qp(mu*S, -pbar, G, h, A, b)['x'] 
                      for mu in mus]
        ## CALCULATE RISKS AND RETURNS FOR FRONTIER
        returns = [blas.dot(pbar, x) for x in portfolios]
        risks = [np.sqrt(blas.dot(x, S * x)) for x in portfolios]
        ## CALCULATE THE 2ND DEGREE POLYNOMIAL OF THE FRONTIER CURVE
        m1 = np.polyfit(returns, risks, 2)
        x1 = np.sqrt(m1[2] / m1[0])
        # CALCULATE THE OPTIMAL PORTFOLIO
        wt = solvers.qp(cvxopt.matrix(x1 * S), -pbar, G, h, A, b)['x']

        weights = {}
        w = np.asarray(wt)
        w = np.around(w)
        for symbol, weight in zip(portfolio.columns, w):
            weights[symbol] = weight[0]

        return weights, returns, risks
