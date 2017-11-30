
/**
 * BinaryTOTD: An implementation of a True Online TD(lambda) learner with binary
 * features.
 *
 * @author Abdullah Alghamdi
 * @author Dylan Ashley
 * @author Richard Sutton
 * @date 2016-05-19
 */

#ifndef BinaryTOTD_H
#define BinaryTOTD_H

#include <stddef.h>

class BinaryTOTD
{

    /**
     * Represents a true online temporal difference lambda learning agent for a
     * task with binary features.
     */

    double V_old;
    size_t theta_size;
    size_t * phi;
    size_t phi_size;
    double * e;

    double index_sum(size_t indices[], size_t indices_size, double v[]);

public:

    double alpha;
    double lambda;
    double gamma;
    double * theta;

    /**
     * @brief Constructs a new agent with the given parameters. Note that a
     *        copy of phi is created during the construction process.
     *
     * @param alpha Step size parameter.
     *
     * @param lambda Eligibility trace parameter.
     *
     * @param gamma Continuation function parameter.
     *
     * @param theta Initial parameter vector.
     *
     * @param theta_size_ Size of theta.
     *
     * @param phi_ Initial features.
     *
     * @param phi_size_ Size of phi_.
     */
    BinaryTOTD(double alpha,
               double lambda,
               double gamma,
               double * theta,
               size_t theta_size_,
               size_t * phi_,
               size_t phi_size_);

    /**
     * @brief Updates the parameter vector for a new observation.
     *
     * @param phi_prime Set of features of length phi_size_.
     *
     * @param reward Reward received by agent.
     *
     * @param alpha_ New value of alpha to use in this and future calls that
     *               do not set alpha
     *
     * @param lambda_ New value of lambda to use in this and future calls that
     *                do not set lambda
     *
     * @param gamma_ New value of gamma to use in this and future calls that
     *               do not set gamma
     */
    void update(size_t * phi_prime,
                double reward,
                double alpha_,
                double lambda_,
                double gamma_);

    /**
     * @brief Updates the parameter vector for a new observation.
     *
     * @param phi_prime Set of features of length phi_size_.
     *
     * @param reward Reward received by agent.
     */
    void update(size_t * phi_prime, double reward);

    /**
     * @brief Returns the current prediction for a given set of features phi.
     *
     * @param phi Set of features of length phi_size_.
     *
     * @return double Predicted value for the set of features.
     */
    double predict(size_t * phi);

    ~BinaryTOTD();
};

#endif
