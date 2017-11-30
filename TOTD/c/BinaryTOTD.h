
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

/**
 * Represents a true online temporal difference lambda learning agent for a
 * task with binary features.
 */
typedef struct
{
    double * theta, * e, V_old;
    size_t theta_size, * phi, phi_size;
} BinaryTOTD;

/**
 * @brief Obtains the dot product of a binary vector with another vector.
 *
 * @param indices Indices of the ones in the first vector.
 *
 * @param v Second vector in the dot product.
 *
 * @param indices_size Length of v.
 *
 * @return double Dot product of v with a binary vector having ones located
 *                only at indices.
 */
double BinaryTOTD_dot(size_t * indices, size_t indices_size, double * v);

/**
 * @brief Constructs a new agent with the given parameters. Note that a
 *        copy of phi is created during the construction process.
 *
 * @param theta Initial parameter vector.
 *
 * @param theta_size Size of theta.
 *
 * @param phi Indices of initial features.
 *
 * @param phi_size Size of phi.
 *
 * @return BinaryTOTD newly allocated agent
 */
BinaryTOTD * BinaryTOTD_init(double * theta,
                             size_t theta_size,
                             size_t * phi,
                             size_t phi_size);

/**
 * @brief Updates the parameter vector for a new observation.
 *
 * @param agent Agent to update.
 *
 * @param phi_prime Indices of new features.
 *
 * @param reward Reward received by agent.
 *
 * @param alpha Step size parameter.
 *
 * @param lambda Eligibility trace parameter.
 *
 * @param gamma Continuation function parameter.
 */
void BinaryTOTD_update(BinaryTOTD * agent,
                       size_t * phi_prime,
                       double reward,
                       double alpha,
                       double lambda,
                       double gamma);

/**
 * @brief Returns the current prediction for a given indices of features phi.
 *
 * @param agent Agent to predict with.
 *
 * @param phi_prime Indices of new features.
 *
 * @return double Predicted value for the set of features.
 */
double BinaryTOTD_predict(BinaryTOTD * agent, size_t * phi);

/**
 * @brief Deconstructs an existing agent.
 *
 * @param agent Agent to deconstruct.
 */
void BinaryTOTD_destroy(BinaryTOTD * agent);

#endif
