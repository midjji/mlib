#pragma once
/* ********************************* FILE ************************************/
/** \file    klas.pnp.h
 *
 * \brief    This header contains a standard RANSAC pnp solver wrapping the p3p solver by klas.
 *
 *
 * Single refinement of the minimal case solution with a maximum number of inliers or lowest total cutoff error(MLESAC)
 *
 *
 * \remark
 * - c++11
 * - can fail
 * - tested by test_pnp.cpp
 *
 * Dependencies:
 * - ceres solver
 *
 * \todo
 * -losac
 * -pose priors
 *
 *
 *
 * \author   Mikael Persson
 * \date     2016-04-01
 * \note MIT licence
 *
 ******************************************************************************/

#include <vector>

#include <mlib/utils/cvl/pose.h>
#include <mlib/sfm/p3p/parameters.h>





namespace cvl {

/**
 * @brief pnp_ransac
 * @param xs, 3d points in world coordinate system
 * @param yns, 2d pinhole normalized points in image
 * @param params, various parameters, threshold is the most important one, threshold=pixel_threshold/minormax(K(0,0),K(1,1))
 * @return Pcam_world such that: x_cam= Pcam_world*x_world
 */
PoseD pnp_ransac(const std::vector<Vector3d>& xs,
                 const std::vector<Vector2d>& yns,
                 PnpParams params=PnpParams());
/**
 * @brief pnp_ransac
 * @param xs, 3d points in world coordinate system in hom representation,
 * @param yns, 2d pinhole normalized points in image
 * @param params, various parameters, threshold is the most important one, threshold=pixel_threshold/minormax(K(0,0),K(1,1))
 * @return Pcam_world such that: x_cam= Pcam_world*x_world
 */
PoseD pnp_ransac(const std::vector<Vector4d>& xs,
                 const std::vector<Vector2d>& yns,
                 PnpParams params=PnpParams());

/**
 * @brief The PNP class A basic RANSAC PNP solver
 *
 * Pcw=est.compute()
 * X_c=Pcw*X_w
 */
class PNP{
public:
    PNP()=default;
    PNP(PnpParams params );
    PoseD operator()(const std::vector<Vector3d>& xs,/// the known 3d positions in world coordinates
    const std::vector<Vector2d>& yns/// the pinhole normalized measurements corresp to xs
    );


    /// attempts to compute the solution
    PoseD compute(const std::vector<Vector3d>& xs,/// the known 3d positions in world coordinates
                  const std::vector<Vector2d>& yns/// the pinhole normalized measurements corresp to xs
                  );

    /// parmeter block
    PnpParams params;
protected:




    /// refine a given pose given the stored data, @param pose
    PoseD refine(const std::vector<Vector3d>& xs,/// the known 3d positions in world coordinates
                 const std::vector<Vector2d>& yns,/// the pinhole normalized measurements corresp to xs
                 PoseD best_pose);







};



} // end namespace cvl



