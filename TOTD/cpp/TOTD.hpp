
/**
 * TOTD: An implementation of a True Online TD(lambda) learner.
 *
 * @author Abdullah Alghamdi
 * @author Dylan Ashley
 * @author Richard Sutton
 * @date 2016-05-19
 */

#ifndef TOTD_H
#define TOTD_H

#include <stddef.h>

class TOTD
{

    /**
     * Represents a true online temporal difference lambda learning agent.
     */

    double V_old;
    size_t size;
    double * phi;
    double * e;

    double dot(double * v_one, double * v_two, size_t size);

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
     * @param size_ Size of theta and phi_.
     *
     * @param phi_ Initial features.
     */
    TOTD(double alpha,
         double lambda,
         double gamma,
         double * theta,
         size_t size_,
         double * phi_);

    /**
     * @brief Updates the parameter vector for a new observation.
     *
     * @param phi_prime Set of new features.
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
    void update(double * phi_prime,
                double reward,
                double alpha_,
                double lambda_,
                double gamma_);

    /**
     * @brief Updates the parameter vector for a new observation.
     *
     * @param phi_prime Set of new features.
     *
     * @param reward Reward received by agent.
     */
    void update(double * phi_prime, double reward);

    /**
     * @brief Returns the current prediction for a given set of features phi.
     *
     * @param phi_prime Set of new features.
     *
     * @return double Predicted value for the set of features.
     */
    double predict(double * phi);

    ~TOTD();
};

#endif
