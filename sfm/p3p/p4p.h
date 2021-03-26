#pragma once
/* ********************************* FILE ************************************/
/** \file    p4p.h
 *
 * \brief    This header contains a wrapper for kneips p3p solver
 *
 * \remark
 * - c++11
 * - no dependencies
 * - can fail, returns identity pose if the data is degenerate, or asserts on incorrect data
 * - no throw
 *
 * \todo
 * - Warn on degenerate input. optional output?
 * - for points that form an orthogonal vector pair, there is a simpler solution,
 *  This configuration may also be a degenerate or semidegenerate case:
 *  the solution comes from simplifying:
 *  L^T(1 b01 b02;0 0 b12; 0 0 0)L =0
 *  which comes from (x2-x0)^T(x1 - x0) = 0
 *  Its unclear if it even is a degenerate case, or if lambdatwist could be modified accordingly since its probably just a permutation of the 3 points,
 *  better yet, since this one takes four points, it should select the best combo of the four.
 *  But detecting semi degeneracy is nearly impossible. Instead, perform the calculation and if it fails,
 *  compute the pose for the special case and see if that works better, then if that fails vary their order.
 *
 *
 *
 *
 *
 * \author   Mikael Persson
 * \date     2016-04-01
 * \note MIT licence
 *
 ******************************************************************************/
#include <set>
#include <vector>
#include "mlib/utils/cvl/pose.h"


namespace cvl{




/**
 * @brief p4p 3 points for the solution and a 4th to distinguish
 * @param xs, at least 4
 * @param yns, size equal to xs
 * @param indexes, elements in xs, yns
 * @return the Pose which best fits all four.
 *
 * this is the simplest imaginable version of this,
 * the pose will fit the first 3 perfectly and the last if it feels like it.
 *
 * This is suitable for low outlieratios, <50%, and low noise. Otherwize use something better...
 *
 * Note, the three first points must be in a non degenerate configuration!
 *
 */
PoseD p4p(const std::vector<cvl::Vector3d>& xs,
         const std::vector<cvl::Vector2d>& yns,
          Vector4<uint> indexes, double max_angle=2*3.1415*2 /* in radians */, PoseD reference =PoseD());



}// end namespace cvl
