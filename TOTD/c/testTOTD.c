
#include "TOTD.h"

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
    double * phi = (double *) malloc(10 * sizeof(double));
    memset(phi, 0, 10 * sizeof(double));
    phi[0] = 1;
    phi[1] = 1;
    phi[4] = 1;
    TOTD * learner = TOTD_init(theta, 10, phi);
    double result_one[] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    assert(eps_equal(result_one, learner -> theta, 10));
    phi[0] = 0;
    phi[5] = 1;
    TOTD_update(learner, phi, 10.0, 0.1 / 3.0, 0.9, 0.9);
    double result_two[] = {0.33333333, 0.33333333, 0.0, 0.0, 0.33333333, 0.0, 0.0, 0.0, 0.0, 0.0};
    assert(eps_equal(result_two, learner -> theta, 10));
    phi[4] = 0;
    phi[7] = 1;
    TOTD_update(learner, phi, - 4.0, 0.1 / 3.0, 0.9, 0.9);
    double result_three[] = {0.23343333, 0.09453778, 0.0, 0.0, 0.09453778, -0.13889556, 0.0, 0.0, 0.0, 0.0};
    assert(eps_equal(result_three, learner -> theta, 10));
    phi[0] = 1;
    phi[4] = 1;
    phi[5] = 0;
    phi[7] = 0;
    TOTD_update(learner, phi, 6.5, 0.1 / 3.0, 0.9, 0.9);
    double result_four[] = {0.37661458, 0.61984028, 0.0, 0.0, 0.40494057, 0.24322571, 0.0, 0.21489971, 0.0, 0.0};
    assert(eps_equal(result_four, learner -> theta, 10));
    free(phi);
    free(theta);
	TOTD_destroy(learner);
    printf("all tests passed\n");
    return 0;
}
