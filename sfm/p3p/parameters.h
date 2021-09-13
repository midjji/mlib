#pragma once
/** \file    parameters.h
 *
 * \brief    This header contains parameter set for the pnp methods
 *
 * \remark
 *  - header only
 *  - c++11
 *  - no dependencies
 * \todo
 *  - improve structure
 *
 * The purpose of sharing one set of parameters for all pnp ransac variants is to reduce the amount of ifdeffing required to include the ip code
 *
 * \author   Mikael Persson
 * \date     2015-10-01
 * \note MIT licence
 *
 *
 *
 ******************************************************************************/

#include <cmath>
#include <mlib/utils/cvl/pose.h>
#include <mlib/param/parametrized.h>

namespace cvl{

/**
 * @brief The PNPParams class
 * common PNP ransac parameters
 */
class PnpParams :public Parametrized{
public:

    /// pixeldist/focallength
    RealParameter* threshold=preal(0.01,"pnp ransac threshold","","pixel_threshold/K(0,0), approx 1-3 pixels",0.0,1.0);
    /// the minimum number of iterations to perform to ensure the required number of iterations is correctly computed
    IntParameter* min_iterations=pint(1000,"minimum ransac iterations","","",100);
    /// a maximum number of iterations to perform to limit the amount of time spent
    IntParameter* max_iterations=pint(2000,"maximum ransac iterations","","",10e6);



    PnpParams(double thr=0.001 /*=pixel_threshold/K(0,0), approx 1-3 pixels*/, std::string name="Pnp ransac");
    PoseD reference;
    double max_angle = 2*3.1415+10; // what is the maximum angle of interest in radians, default any, compared to reference;

    //  parameters
    /// initial inlier ratio estimate, good value = expected inlier ratio/2
    /// should really be estimated online.
    /// double inlier_ratio=0.25;

    /// minimum probability for never findning a single full inlier set
    double min_probability=0.99999; // effectively capped by the max_iterations too...




    /// perform maximum likelihood sampling consensus,
    //bool MLESAC=false;// not supported at this time... its not better in any of my cases anyways...

    /// break early in the ransac loop if a sufficiently good solution is found
    bool early_exit=false;
    uint early_exit_min_iterations =5;
    double early_exit_inlier_ratio=0.8;

    /// if random eval is used, this is the number of samples drawn
    //uint max_support_eval=500; not used...








    /**
     * @brief get_iterations
     * @param estimated_inliers, start with a conservative guess, say expected_inliers/2.0
     * @param p_inlier_given_noise_and_gt
     * @return
     */
    int get_iterations(double estimated_inliers,
                       double p_meets_treshold_given_inlier_and_gt=0.9);


};



}
