
#include "TOTD.h"

#include <stdio.h>
#include <stdlib.h>

double TOTD_dot(double * v_one, double * v_two, size_t size)
{
    double sum = 0.0;
    for (size_t i = 0; i < size; i++) sum += v_one[i] * v_two[i];
    return sum;
}

TOTD * TOTD_init(double * theta,
                 size_t size,
                 double * phi)
{
    // create new agent
    TOTD * agent;
    if ((agent = (TOTD *) malloc(sizeof(TOTD))) == NULL)
    {
        perror("malloc for new agent failed");
        return NULL;
    }

    // set passed in parameters
    agent -> size = size;
    agent -> theta = theta;

    // initialize V_old
    agent -> V_old = 0.0;

    // make phi
    if ((agent -> phi = (double *) malloc(size * sizeof(double))) == NULL)
    {
        perror("malloc for new agent phi failed");
        free(agent);
        return NULL;
    }
    for (size_t i = 0; i < size; i++) agent -> phi[i] = phi[i];

    // make e
    if ((agent -> e = (double *) calloc(size, sizeof(double))) == NULL)
    {
        perror("calloc for new agent e failed");
        free(agent -> phi);
        free(agent);
        return NULL;
    }

    return agent;
}

void TOTD_update(TOTD * agent,
                 double * phi_prime,
                 double reward,
                 double alpha,
                 double lambda,
                 double gamma)
{
    // calculate V and V_prime
    double V = TOTD_dot(agent -> phi, agent -> theta, agent -> size);
    double V_prime = TOTD_dot(phi_prime, agent -> theta, agent -> size);

    // calculate delta
    double delta = reward + gamma * V_prime - V;

    // update eligibility traces
    double e_phi = TOTD_dot(agent -> phi, agent -> e, agent -> size);
    for (size_t i = 0; i < agent -> size; i++)
    {
        agent -> e[i] = gamma * lambda * agent -> e[i] + agent -> phi[i] -
                         alpha * gamma * lambda * e_phi * agent -> phi[i];
    }

    // update theta
    for (size_t i = 0; i < agent -> size; i++)
    {
        agent -> theta[i] += alpha * (delta + V - agent -> V_old) * agent -> e[i] -
                             alpha * (V - agent -> V_old) * agent -> phi[i];
    }

    // update values
    agent -> V_old = V_prime;
    for (size_t i = 0; i < agent -> size; i++) agent -> phi[i] = phi_prime[i];
}

double TOTD_predict(TOTD * agent, double * phi)
{
    return TOTD_dot(phi, agent -> theta, agent -> size);
}

void TOTD_destroy(TOTD * agent)
{
    free(agent -> phi);
    free(agent -> e);
    free(agent);
}
