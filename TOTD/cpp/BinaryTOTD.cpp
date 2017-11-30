
#include "BinaryTOTD.hpp"

double BinaryTOTD::index_sum(size_t indices[], size_t indices_size, double v[])
{
    double sum = 0.0;
    for (size_t i = 0; i < indices_size; i++) sum += v[indices[i]];
    return sum;
}

BinaryTOTD::BinaryTOTD(double alpha_,
                       double lambda_,
                       double gamma_,
                       double * theta_,
                       size_t theta_size_,
                       size_t * phi_,
                       size_t phi_size_) :
                       theta_size(theta_size_),
                       alpha(alpha_),
                       lambda(lambda_),
                       gamma(gamma_),
                       theta(theta_)
{
    // initialize V_old
    V_old = 0.0;

    // make phi
    phi_size = phi_size_;
    phi = new size_t[phi_size];
    for (size_t i = 0; i < phi_size; i++) phi[i] = phi_[i];

    // make e
    e = new double[theta_size];
    for (size_t i = 0; i < theta_size; i++) e[i] = 0.0;
}

void BinaryTOTD::update(size_t * phi_prime,
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

void BinaryTOTD::update(size_t * phi_prime, double reward)
{
    // calculate V and V_prime
    double V = index_sum(phi, phi_size, theta);
    double V_prime = index_sum(phi_prime, phi_size, theta);

    // calculate delta
    double delta = reward + gamma * V_prime - V;

    // update eligibility traces
    double e_phi = index_sum(phi, phi_size, e);
    for (size_t i = 0; i < theta_size; i++) e[i] *= gamma * lambda;
    for (size_t i = 0; i < phi_size; i++) e[phi[i]] += 1 - alpha * gamma * lambda * e_phi;

    // update theta
    for (size_t i = 0; i < theta_size; i++) theta[i] += alpha * (delta + V - V_old) * e[i];
    for (size_t i = 0; i < phi_size; i++) theta[phi[i]] -= alpha * (V - V_old);

    // update values
    V_old = V_prime;
    for (size_t i = 0; i < phi_size; i++) phi[i] = phi_prime[i];
}

double BinaryTOTD::predict(size_t * phi)
{
    return index_sum(phi, phi_size, theta);
}

BinaryTOTD::~BinaryTOTD()
{
    delete [] phi;
    delete [] e;
}
