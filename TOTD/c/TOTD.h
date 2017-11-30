
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

/**
 * Represents a true online temporal difference lambda learning agent.
 */
typedef struct
{
    double * theta, * phi, * e, V_old;
    size_t size;
} TOTD;

/**
 * @brief Obtains the dot product of two vectors.
 *
 * @param v_one First vector in the dot product.
 *
 * @param v_two Second vector in the dot product.
 *
 * @param size Length of both v_one and v_two.
 *
 * @return double Dot product of v_one with v_two.
 */
double TOTD_dot(double * v_one, double * v_two, size_t size);

/**
 * @brief Constructs a new agent with the given parameters. Note that a
 *        copy of phi is created during the construction process.
 *
 * @param theta Initial parameter vector.
 *
 * @param size Size of theta and phi.
 *
 * @param phi Initial features.
 *
 * @return TOTD newly allocated agent
 */
TOTD * TOTD_init(double * theta,
                 size_t size,
                 double * phi);

/**
 * @brief Updates the parameter vector for a new observation.
 *
 * @param agent Agent to update.
 *
 * @param phi_prime Set of new features.
 *
 * @param reward Reward received by agent.
 *
 * @param alpha Step size parameter.
 *
 * @param lambda Eligibility trace parameter.
 *
 * @param gamma Continuation function parameter.
 */
void TOTD_update(TOTD * agent,
                 double * phi_prime,
                 double reward,
                 double alpha,
                 double lambda,
                 double gamma);

/**
 * @brief Returns the current prediction for a given set of features phi.
 *
 * @param agent Agent to predict with.
 *
 * @param phi_prime Set of new features.
 *
 * @return double Predicted value for the set of features.
 */
double TOTD_predict(TOTD * agent, double * phi);

/**
 * @brief Deconstructs an existing agent.
 *
 * @param agent Agent to deconstruct.
 */
void TOTD_destroy(TOTD * agent);

#endif
