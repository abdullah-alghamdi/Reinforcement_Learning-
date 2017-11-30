
import numpy as np

class TOTD:

    """
    Represents a true online temporal difference lambda learning agent.
    """

    def __init__(self, alpha, lambda_, gamma, theta, phi):
        """
        Constructs a new agent with the given parameters. Note that a copy of
        phi is created during the construction process.
        """
        self.alpha = np.array(alpha)
        self.old_alpha = np.zeros(self.alpha.shape)
        self.gamma = np.array(gamma)
        self.old_gamma = np.zeros(self.gamma.shape)
        self.lambda_ = np.array(lambda_)
        self.old_lambda = np.zeros(self.lambda_.shape)
        self.theta = np.atleast_2d(theta)
        self._phi = np.array(np.copy(phi)) # use a copy of phi
        self._e = np.zeros(np.shape(self.theta))
        self._V_old = 0

    def update(self, phi_prime, reward, alpha=None, lambda_=None, gamma=None):
        """
        Updates the parameter vector for a new observation. If any optional
        values are set then the new value of the optional is used for this and
        future calls that do not set the same optional value.
        """
        # set optional values
        if alpha is not None:
            self.alpha = np.array(alpha)
        if lambda_ is not None:
            self.lambda_ = np.array(lambda_)
        if gamma is not None:
            self.gamma = np.array(gamma)
        # calculate V and V_prime
        V = np.dot(self.theta, self._phi)
        V_prime = np.dot(self.theta, phi_prime)
        # calculate delta
        delta = reward + self.gamma * V_prime - V
        # update eligibility traces
        e_phi = np.dot(self._e, self._phi)
        self._e *= (self.old_lambda * self.old_gamma)[..., np.newaxis]
        self._e -= np.outer(self.old_lambda * self.old_gamma * self.old_alpha * e_phi, self._phi)
        self._e += self._phi
        # update theta
        self.theta += (self._e*(self.alpha*(delta + V - self._V_old))[..., np.newaxis])
        self.theta -= np.outer(self.alpha * (V - self._V_old), self._phi)
        # update values
        self._V_old = V_prime
        self._phi = np.array(np.copy(phi_prime))
        np.copyto(self.old_gamma, self.gamma)
        np.copyto(self.old_lambda, self.lambda_)
        np.copyto(self.old_alpha, self.alpha)

    def predict(self, phi):
        """
        Returns the current prediction for a given set of features phi.
        """
        return np.dot(self.theta, phi)
