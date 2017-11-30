
#include "TOTD.hpp"

double TOTD::dot(double * v_one, double * v_two, size_t size)
{
    double sum = 0.0;
    for (size_t i = 0; i < size; i++) sum += v_one[i] * v_two[i];
    return sum;
}

TOTD::TOTD(double alpha_,
           double lambda_,
           double gamma_,
           double * theta_,
           size_t size_,
           double * phi_) :
           size(size_),
           alpha(alpha_),
           lambda(lambda_),
           gamma(gamma_),
           theta(theta_)
{
    // initialize V_old
    V_old = 0.0;

    // make phi
    phi = new double[size];
    for (size_t i = 0; i < size; i++) phi[i] = phi_[i];

    // make e
    e = new double[size];
    for (size_t i = 0; i < size; i++) e[i] = 0.0;
}

void TOTD::update(double * phi_prime,
                  double reward,
                  double alpha_,
                  double lambda_,
                  double gamma_)
{
    alpha = alpha_;
    lambda = lambda_;
    gamma = gamma_;
    this -> update(phi_prime, reward);
}

void TOTD::update(double * phi_prime, double reward)
{
    // calculate V and V_prime
    double V = dot(phi, theta, size);
    double V_prime = dot(phi_prime, theta, size);

    // calculate delta
    double delta = reward + gamma * V_prime - V;

    // update eligibility traces
    double e_phi = dot(phi, e, size);
    for (size_t i = 0; i < size; i++)
    {
        e[i] = gamma * lambda * e[i] + phi[i] - alpha * gamma * lambda * e_phi * phi[i];
    }

    // update theta
    for (size_t i = 0; i < size; i++)
    {
        theta[i] += alpha * (delta + V - V_old) * e[i] - alpha * (V - V_old) * phi[i];
    }

    // update values
    V_old = V_prime;
    for (size_t i = 0; i < size; i++) phi[i] = phi_prime[i];
}

double TOTD::predict(double * phi)
{
    return dot(phi, theta, size);
}

TOTD::~TOTD()
{
    delete [] phi;
    delete [] e;
}
