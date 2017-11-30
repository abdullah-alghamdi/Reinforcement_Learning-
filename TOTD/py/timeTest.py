#!/usr/bin/env python

import numpy as np
import unittest
import time

from BinaryTOTD import BinaryTOTD
from TOTD import TOTD

num_predictions = 1000
num_features = 16380
active_binary_features = 32


class testTime(unittest.TestCase):

    def test_time(self):
        print()
        #Test time for BinaryTOTD
        theta = np.zeros((num_predictions,num_features))
        phi = np.random.randint(0, num_features, (active_binary_features))
        single_learners = [BinaryTOTD(0.1, 0.9, 0.9, np.copy(theta[i,:]), phi) for i in range(num_predictions)]
        multi_learner = BinaryTOTD(0.1*np.ones((num_predictions))/active_binary_features, 0.9*np.ones((num_predictions)), 0.9*np.ones((num_predictions))*np.ones((num_predictions)), np.copy(theta), phi)
        start_time = time.time()
        #Test single learners
        for i in range(100):
            phi = np.random.randint(0, num_features, (active_binary_features))
            r = np.random.randint(-1, 1, (num_predictions))
            gamma = np.random.sample((num_predictions))
            alpha = np.random.sample((num_predictions))/active_binary_features
            lambda_ = np.random.sample((num_predictions))
            for j in range(len(single_learners)):
                single_learners[j].update(phi, r[j], alpha = alpha[j], lambda_ = lambda_[j], gamma = gamma[j])
        print("Binary single time: ", time.time()-start_time)

        start_time = time.time()
        #Test multiple learners
        for i in range(100):
            phi = np.random.randint(0, num_features, (active_binary_features))
            r = np.random.randint(-1, 1, (num_predictions))
            gamma = np.random.sample((num_predictions))
            alpha = np.random.sample((num_predictions))/active_binary_features
            lambda_ = np.random.sample((num_predictions))
            multi_learner.update(phi, r, alpha = alpha, lambda_ = lambda_, gamma = gamma)
        print("Binary multiple time: ", time.time()-start_time)


        #Test time for TOTD
        theta = np.zeros((num_predictions,num_features))
        phi = np.random.sample((num_features))
        single_learners = [TOTD(0.1, 0.9, 0.9, np.copy(theta[i,:]), phi) for i in range(num_predictions)]
        multi_learner = TOTD(0.1*np.ones((num_predictions)), 0.9*np.ones((num_predictions)), 0.9*np.ones((num_predictions)), np.copy(theta), phi)
        start_time = time.time()
        #Test single learners
        for i in range(100):
            phi = np.random.sample((num_features))
            r = np.random.randint(-1, 1, (num_predictions))
            gamma = np.random.sample((num_predictions))
            alpha = np.random.sample((num_predictions))
            alpha /= np.sum(phi)
            lambda_ = np.random.sample((num_predictions))
            for j in range(len(single_learners)):
                single_learners[j].update(phi, r[j], alpha = alpha[j], lambda_ = lambda_[j], gamma = gamma[j])
        print("Nonbinary single time: ", time.time()-start_time)

        start_time = time.time()
        #Test multiple learners
        for i in range(100):
            phi = np.random.sample((num_features))
            r = np.random.randint(-1, 1, (num_predictions))
            gamma = np.random.sample((num_predictions))
            alpha = np.random.sample((num_predictions))
            alpha/=np.sum(phi)
            lambda_ = np.random.sample((num_predictions))
            multi_learner.update(phi, r, alpha = alpha, lambda_ = lambda_, gamma = gamma)
        print("Nonbinary multiple time: ", time.time()-start_time)

if __name__ == "__main__":
    suite = unittest.TestLoader().loadTestsFromTestCase(testTime)
    unittest.TextTestRunner(verbosity=2).run(suite)
