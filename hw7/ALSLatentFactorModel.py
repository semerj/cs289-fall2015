import numpy as np
from numpy.linalg import norm
np.random.seed(289)


class ALSLatentFactorModel:
    '''
    Latent Factor Model with Alternating Least Squares (ALS)
    for Matrix Completion
    '''
    def __init__(self, d, λ=0.01, num_iters=10, mu=0, std=1):
        self.d = d
        self.λ= λ
        self.num_iters = num_iters
        self.mu = mu
        self.std = std

    def _init_weights(self, num_factors):
        return np.random.normal(self.mu, self.std, (num_factors, self.d))

    def fit(self, X):
        self.X = X
        num_users, num_items = X.shape
        self.U = self._init_weights(num_users)
        self.V = self._init_weights(num_items)

        for _ in range(self.num_iters):
            # user factors
            VtV = self.V.T.dot(self.V)
            λI = np.diag(self.λ*np.ones(self.d))
            VtV_λI = np.linalg.inv(VtV + λI)
            for u in range(num_users):
                user_ratings = X[u,:]
                not_nan_ind = np.where(~np.isnan(user_ratings))[0]
                r_ui = user_ratings[not_nan_ind]
                V_i = self.V[not_nan_ind]
                self.U[u] = VtV_λI.dot(r_ui.dot(V_i))

            # item factors
            UtU = self.U.T.dot(self.U)
            λI = np.diag(self.λ*np.ones(self.d))
            UtU_λI = np.linalg.inv(UtU + λI)
            for i in range(num_items):
                item_ratings = X[:,i]
                not_nan_ind = np.where(~np.isnan(item_ratings))[0]
                r_ui = item_ratings[not_nan_ind]
                U_i = self.U[not_nan_ind]
                self.V[i] = UtU_λI.dot(r_ui.dot(U_i))

    def predict(self, binarize=True):
        R_hat = self.U.dot(self.V.T)
        if binarize:
            return np.where(R_hat > 0, 1., 0.)
        else:
            return R_hat
