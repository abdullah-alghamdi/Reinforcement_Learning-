
#include "BinaryTOTD.h"

#include <stdio.h>
#include <stdlib.h>

double BinaryTOTD_dot(size_t * indices, size_t indices_size, double * v)
{
    double sum = 0.0;
    for (size_t i = 0; i < indices_size; i++) sum += v[indices[i]];
    return sum;
}

BinaryTOTD * BinaryTOTD_init(double * theta,
                             size_t theta_size,
                             size_t * phi,
                             size_t phi_size)
{
    // create new agent
    BinaryTOTD * agent;
    if ((agent = (BinaryTOTD *) malloc(sizeof(BinaryTOTD))) == NULL)
    {
        perror("malloc for new agent failed");
        return NULL;
    }

    // set passed in parameters
    agent -> theta = theta;
    agent -> theta_size = theta_size;
    agent -> phi_size = phi_size;

    // initialize V_old
    agent -> V_old = 0.0;

    // make phi
    if ((agent -> phi = (size_t *) malloc(phi_size * sizeof(size_t))) == NULL)
    {
        perror("malloc for new agent phi failed");
        free(agent);
        return NULL;
    }
    for (size_t i = 0; i < phi_size; i++) agent -> phi[i] = phi[i];

    // make e
    if ((agent -> e = (double *) calloc(theta_size, sizeof(double))) == NULL)
    {
        perror("calloc for new agent e failed");
        free(agent -> phi);
        free(agent);
        return NULL;
    }

    return agent;
}

void BinaryTOTD_update(BinaryTOTD * agent,
                       size_t * phi_prime,
                       double reward,
                       double alpha,
                       double lambda,
                       double gamma)
{
    // calculate V and V_prime
    double V = BinaryTOTD_dot(agent -> phi, agent -> phi_size, agent -> theta);
    double V_prime = BinaryTOTD_dot(phi_prime, agent -> phi_size, agent -> theta);

    // calculate delta
    double delta = reward + gamma * V_prime - V;

    // update eligibility traces
    double e_phi = BinaryTOTD_dot(agent -> phi, agent -> phi_size, agent -> e);
    for (size_t i = 0; i < agent -> theta_size; i++) agent -> e[i] *= gamma * lambda;
    for (size_t i = 0; i < agent -> phi_size; i++) agent -> e[agent -> phi[i]] += 1 - alpha * gamma * lambda * e_phi;

    // update theta
    for (size_t i = 0; i < agent -> theta_size; i++) agent -> theta[i] += alpha * (delta + V - agent -> V_old) * agent -> e[i];
    for (size_t i = 0; i < agent -> phi_size; i++) agent -> theta[agent -> phi[i]] -= alpha * (V - agent -> V_old);

    // update values
    agent -> V_old = V_prime;
    for (size_t i = 0; i < agent -> phi_size; i++) agent -> phi[i] = phi_prime[i];
}

double BinaryTOTD_predict(BinaryTOTD * agent, size_t * phi)
{
    return BinaryTOTD_dot(phi, agent -> phi_size, agent -> theta);
}

void BinaryTOTD_destroy(BinaryTOTD * agent)
{
    free(agent -> phi);
    free(agent -> e);
    free(agent);
}
