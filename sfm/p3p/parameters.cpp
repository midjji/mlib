#include <mlib/sfm/p3p/parameters.h>
namespace cvl {
int PnpParams::get_iterations(double estimated_inliers,
                   double p_meets_treshold_given_inlier_and_gt){

    double p_inlier=std::min(0.9,estimated_inliers*p_meets_treshold_given_inlier_and_gt);
    p_inlier  = std::min(std::max(p_inlier,1e-2),1-1e-8);
    if(p_inlier<0.01) return max_iterations;

    // this is the range in which the approximation is resonably valid.
    double p_failure = std::min(std::max( 1.0-min_probability,1e-8),0.01);

    // approximate P(inlier|inlier) as 1. note how this differs from P(inlier|inlier,gt)
    double p_good=std::pow(p_inlier,4);
    // approximate hyp as bin
    // approximate bin as norm
    // always draw atleast +50? yeah makes it better
    double iterations=std::ceil((log(p_failure)/log(1.0-p_good))) +50;

    // warning, small number of sample points should increase this further,
    //since bin,norm approxes are bad, this is mostly captured by the +50 and min_iterations, which shouldnt be under 100

    // p_good_given_inlier should be drop with increasing model points too, or noise aswell
    // more iters required if we dismiss solutions
    if(max_angle<2*3.1415)
        iterations= iterations / (max_angle/(2*3.1415));

    if(iterations<min_iterations) return min_iterations;
    if(iterations>max_iterations) return max_iterations;

    return int(std::ceil(iterations));


}
}
