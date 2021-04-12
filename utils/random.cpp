#include <mlib/utils/random.h>
namespace mlib{

/**
 * @brief randu integer random value drawn from uniform distribution
 * @param low  - included
 * @param high - included
 * @return random value according to compile time random value strategy
 */
double randu(double low, double high){
    std::uniform_real_distribution<double> rn(low,high);
    return rn(random::generator);
}

/**
 * @brief randn random value drawn from normal distribution
 * @param m
 * @param sigma
 * @return random value drawn from normal distribution
 */
double randn(double mean, double sigma){
    std::normal_distribution<double> rn(mean, sigma);
    return rn(random::generator);
}
/**
 * @brief randui integer random value drawn from uniform distribution
 * @param low  - included
 * @param high - included
 * @return random value according to compile time random value strategy
 */
int randui(int low, int high){
    std::uniform_int_distribution<int> rn(low,high);
    return rn(random::generator);
}
}
