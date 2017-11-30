#!/usr/bin/env python

import numpy as np
import unittest

from TOTD import TOTD


class TestTOTD(unittest.TestCase):

    def test_update(self):
        phi = np.zeros(10)
        phi[0] = 1
        phi[1] = 1
        phi[4] = 1
        alpha = np.random.sample((10))
        lambda_ = np.random.sample((10))
        learner = TOTD(0.1 / 3.0, 0.9, 0.9, np.zeros(10), phi)
        for i, value in enumerate([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]):
            self.assertAlmostEqual(value, learner.theta[0,i])
        phi[0] = 0
        phi[5] = 1
        learner.update(phi, 10)
        for i, value in enumerate([0.33333333, 0.33333333, 0.0, 0.0, 0.33333333, 0.0, 0.0, 0.0, 0.0, 0.0]):
            self.assertAlmostEqual(value, learner.theta[0,i])
        phi[4] = 0
        phi[7] = 1
        learner.update(phi, - 4)
        for i, value in enumerate([0.23343333, 0.09453778, 0.0, 0.0, 0.09453778, -0.13889556, 0.0, 0.0, 0.0, 0.0]):
            self.assertAlmostEqual(value, learner.theta[0,i])
        phi[0] = 1
        phi[4] = 1
        phi[5] = 0
        phi[7] = 0
        learner.update(phi, 6.5)
        for i, value in enumerate([0.37661458, 0.61984028, 0.0, 0.0, 0.40494057, 0.24322571, 0.0, 0.21489971, 0.0, 0.0]):
            self.assertAlmostEqual(value, learner.theta[0,i])

        phi = np.random.sample((5))
        phi/=np.sum(phi)
        theta = np.zeros((10,5))
        learner = TOTD(0.1, 0.9, 0.9, theta, phi)
        phi = np.random.sample((5))
        r = np.random.randint(0, 10, (10))
        learner.update(phi, r)

        theta = np.zeros((10,5))
        phi = np.random.sample((5))
        phi/=np.sum(phi)
        single_learners = [TOTD(0.1, 0.9, 0.9, np.copy(theta[i,:]), phi) for i in range(5)]
        multi_learner = TOTD(0.1*np.ones((10)), 0.9*np.ones((10)), 0.9*np.ones((10)), np.copy(theta), phi)
        for i in range(100):
            phi = np.random.sample((5))
            phi/=np.sum(phi)
            r = np.random.randint(0, 10, (10))
            gamma = np.random.sample((10))
            alpha = np.random.sample((10))
            lambda_ = np.random.sample((10))
            for j in range(len(single_learners)):
                single_learners[j].update(phi, r[j], alpha = alpha[j], lambda_ = lambda_[j], gamma = gamma[j])
            multi_learner.update(phi, r, alpha = alpha, lambda_ = lambda_, gamma = gamma)
        for i in range(len(single_learners)):
            self.assertTrue(np.allclose(single_learners[i].theta, multi_learner.theta[i,:]))
            phi = np.random.sample((5))
            phi/=np.sum(phi)
            self.assertAlmostEqual(single_learners[i].predict(phi)[0], multi_learner.predict(phi)[i])

if __name__ == "__main__":
    suite = unittest.TestLoader().loadTestsFromTestCase(TestTOTD)
    unittest.TextTestRunner(verbosity=2).run(suite)
