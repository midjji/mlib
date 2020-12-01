#pragma once

/** \file    grid.h
 *
 * \brief    This header contains the grid based Adaptive NonMaxima Supression filter, this filter is generally faster than the base version, how much depends on the how distributed the points are
 *
 *
 * \remark
 *  - c++11
 *  - no dependencies
 *
 * ANMS filtering orders measurements by a supplied score and then selects the strongest with a minimum distance in order.
 *
 * The radius is fixed and the filter must be iterated for further reductions.
 *
 *
 * \author   Mikael Persson
 * \date     2015-10-01
 * \note MIT licence

 *
 ******************************************************************************/

#include <mlib/sfm/anms/base.h>
#include <mlib/utils/cvl/matrix.h>
#include <mlib/utils/cvl/matrix_adapter.h>

namespace cvl{


namespace anms{



/**
 * @brief The Gridlsh class a exact lsh
 * A simple local sensitive hash, => toss em in evenly distributed buckets...
 * add to one bucket, query multiple buckets
 *
 * Many speedups are possible,
 * fast enough for now
 *
 * looks in the buckets in order of likelihood
 *
 * \todo
 * - only look in possible buckets
 *
 */
class Gridlsh{
public:
    Gridlsh();

    /**
     * @brief Gridlsh
     * @param minv
     * @param maxv
     * @param buckets per dimension
     */
    Gridlsh(Vector2f minv,
            Vector2f maxv,
            uint buckets);
    /**
     * @brief getGridPos fills the row and col which corresponds to the grid coordinates
     * @param v positon queried
     * @param row
     * @param col
     */
    void getGridPos(const Vector2f& v, int& row, int& col);
    /**
     * @brief getGridPos getGridPos(v.y,row,col)
     * @param v
     * @param row
     * @param col
     */
    void getGridPos(const Data& v, int& row, int& col);

    /**
     * @brief add add a data element to its grid position
     * @param v
     */
    void add(const Data& v);
    /**
     * @brief query looks for a data within radius in each possible grid cell
     * @param v
     * @param radius
     * @return
     */
    bool query(const Data& v,
               float radius);
    // should not be copied!
private:
    /// lowest values
    Vector2f minv;
    /// highest values
    Vector2f maxv;

    /// number of buckets in each direction
    int buckets;
    /// bucket size x direction
    float delta_x;
    /// bucket size y direction
    float delta_y;
    /// grid matrix adapter
    MatrixAdapter<std::vector<Data>> grid;
    /// grid matrix data
    std::vector<std::vector<Data>> griddata;

};



/**
 * @brief The anms::GridSolver class
 *
 * Adaptive NonMaxima Supression filtering is the process by which keypoints or tracks are ordered by a quality value and points too close to a nearby better value are dropped.
 * This improves the conditioning of most algorithms which use keypoints and reduce computational cost with little loss.
 *
 * A gridlsh based exact anms implementation.
 *
 * moderatly fast and certain essentially...,
 * similar to bucketing but exact and examines all possibilites at low cost
 */
class GridSolver : public Solver{
public:
    void init(const std::vector<Data>& datas,const std::vector<Data>& locked);
    void init(const std::vector<Data>& datas,const std::vector<Data>& locked, float minx, float miny, float maxx, float maxy );

    virtual void compute(double minRadius, int minKeep);
    virtual bool exact(){return true;}





private:
/// the data grid
    Gridlsh grid;
};


}
}








