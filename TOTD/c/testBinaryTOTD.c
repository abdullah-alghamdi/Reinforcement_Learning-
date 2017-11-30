
#include "BinaryTOTD.h"

#include <assert.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define EPS 1e-8


bool eps_equal(double * a, double * b, size_t size)
{
    for (size_t i = 0; i < size; i++)
    {
        if (fabs(a[i] - b[i]) > EPS) return false;
    }
    return true;
}


int main()
{
    double * theta = (double *) malloc(10 * sizeof(double));
    memset(theta, 0, 10 * sizeof(double));
    size_t * phi = (size_t *) malloc(3 * sizeof(size_t));
    phi[0] = 0;
    phi[1] = 1;
    phi[2] = 4;
    BinaryTOTD * learner = BinaryTOTD_init(theta, 10, phi, 3);
    double result_one[] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    assert(eps_equal(result_one, learner -> theta, 10));
    phi[0] = 1;
    phi[1] = 4;
    phi[2] = 5;
    BinaryTOTD_update(learner, phi, 10.0, 0.1 / 3.0, 0.9, 0.9);
    double result_two[] = {0.33333333, 0.33333333, 0.0, 0.0, 0.33333333, 0.0, 0.0, 0.0, 0.0, 0.0};
    assert(eps_equal(result_two, learner -> theta, 10));
    phi[0] = 1;
    phi[1] = 5;
    phi[2] = 7;
    BinaryTOTD_update(learner, phi, - 4.0, 0.1 / 3.0, 0.9, 0.9);
    double result_three[] = {0.23343333, 0.09453778, 0.0, 0.0, 0.09453778, -0.13889556, 0.0, 0.0, 0.0, 0.0};
    assert(eps_equal(result_three, learner -> theta, 10));
    phi[0] = 0;
    phi[1] = 1;
    phi[2] = 4;
    BinaryTOTD_update(learner, phi, 6.5, 0.1 / 3.0, 0.9, 0.9);
    double result_four[] = {0.37661458, 0.61984028, 0.0, 0.0, 0.40494057, 0.24322571, 0.0, 0.21489971, 0.0, 0.0};
    assert(eps_equal(result_four, learner -> theta, 10));
    free(phi);
    free(theta);
    BinaryTOTD_destroy(learner);
    printf("all tests passed\n");
    return 0;
}
