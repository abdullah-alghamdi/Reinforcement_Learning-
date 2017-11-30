
import numpy as np
import theano
import theano.tensor as T

class TOTD:

    """
    Represents a true online temporal difference lambda learning agent.
    """

    def __init__(self, alpha, lambda_, gamma, theta, phi):
        """
        Constructs a new agent with the given parameters. Note that a copy of
        phi is created during the construction process.
        """
        #Default values for update parameters
        self.gamma_default = np.atleast_1d(gamma)
        self.alpha_default = np.atleast_1d(alpha)
        self.lambda_default = np.atleast_1d(lambda_)

        #Input variables
        self.reward = T.fvector('reward')
        self.phi_prime = T.fvector('phi_prime')
        self.alpha = T.fvector('alpha')
        self.lambda_ = T.fvector('lambda')
        self.gamma = T.fvector('gamma')

        #Shared variables
        self.phi = theano.shared(value = np.copy(phi).astype(theano.config.floatX), name='phi', borrow=True)
        theta_val = np.atleast_2d(theta).astype(theano.config.floatX)
        self.num_predictions = theta_val.shape[0]
        self.num_features = theta_val.shape[1]
        self.theta = theano.shared(value = theta_val, name='theta', borrow=True)
        self.V_old = theano.shared(value = np.zeros(self.num_predictions).astype(theano.config.floatX), name='V_old', borrow=True)
        self.e = theano.shared(value = np.zeros(theta_val.shape).astype(theano.config.floatX), name='e', borrow=True)

        V = T.dot(self.theta, self.phi)
        V_prime = T.dot(self.theta, self.phi_prime)

        delta = self.reward + self.gamma * V_prime - V

        e_phi = T.dot(self.e, self.phi)
        e_new = ((self.lambda_ * self.gamma).dimshuffle(0,'x') * (self.e - T.outer(self.alpha * e_phi, self.phi)))+ self.phi

        theta_new = self.theta + (e_new * (self.alpha * (delta + V - self.V_old)).dimshuffle(0,'x')) - T.outer(self.alpha * (V - self.V_old),self.phi)

        self._update = theano.function(
            [self.alpha, self.lambda_, self.gamma, self.phi_prime, self.reward],
            updates = [(self.phi, self.phi_prime), (self.V_old, V_prime), (self.e, e_new), (self.theta, theta_new) ]
            )

        self._predict = theano.function(
            [self.phi_prime],
            T.dot(self.theta, self.phi_prime)
            )



    def update(self, phi_prime, reward, alpha=None, lambda_=None, gamma=None):
        """
        Updates the parameter vector for a new observation. If any optional
        values are set then the new value of the optional is used for this and
        future calls that do not set the same optional value.
        """
        # set optional values
        if alpha is None:
            alpha = self.alpha_default
        if lambda_ is None:
            lambda_ = self.lambda_default
        if gamma is None:
            gamma = self.gamma_default
        alpha = np.broadcast_to(alpha, (self.num_predictions))
        lambda_ = np.broadcast_to(lambda_, (self.num_predictions))
        gamma = np.broadcast_to(gamma, (self.num_predictions))
        self._update(alpha.astype(theano.config.floatX).astype(theano.config.floatX), lambda_.astype(theano.config.floatX).astype(theano.config.floatX), gamma.astype(theano.config.floatX), phi_prime.astype(theano.config.floatX), np.atleast_1d(reward).astype(theano.config.floatX))

    def predict(self, phi):
        """
        Returns the current prediction for a given set of features phi.
        """
        return self._predict(phi)
