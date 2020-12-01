#pragma once
/** \file    base.h
 *
 * \brief    This header contains the base Adaptive NonMaxima Supression filter
 *
 * \remark
 *  - c++11
 *  - no dependencies
 *  - tested by test_anms.cpp
 *
 * ANMS filtering orders measurements by a supplied score and then selects the strongest with a minimum distance in order.
 *
 * The radius is fixed and the filter must be iterated for further reductions.
 * For a fast exact implementation see anms/grid.h
 *
 *
 *
 * \todo
 * - port the iterative anms with a goal method
 *
 *
 * \author   Mikael Persson
 * \date     2015-10-01
 * \note MIT licence

 *
 ******************************************************************************/






#include <mlib/utils/cvl/matrix.h>
#include <mlib/utils/vector.h>

namespace cvl{
/**
 * \namespace cvl::anms
 * \brief Contains generic ANMS filter implementations see base.h
 *
 */
namespace anms{




/**
 * @brief The anms::Data class
 *  Wraps the values and id necessary to identify which whatever survived the anms filtering (see base_anms.h)
 */
class Data{
public:
    Data();
    /**
     * @brief Data
     * @param str the strength of the point, higher is better=> more likely too be kept
     * @param x coordinate
     * @param y coordinate
     * @param id unique id for the point
     */
    Data(float str, float x, float y, int id);
    template<class V2>
    Data(float str, V2 v2, int id):str(str),y(v2[0],v2[1]),id(id){}
    /**
     * @brief near is this point within radius2 of any in the list
     * @param datas
     * @param radius2 radius squared
     * @return
     */
    bool near(const std::vector<Data>& datas,float radius2) const;
    /// the strength of the point, higher is better
    float str;
    /// the position of the point
    Vector2f y;
    /// the unique id of this point
    uint id;
};


/**
 * @brief The anms::Solver class
 *
 * Adaptive NonMaxima Supression filtering is the process by which keypoints or tracks are ordered by a quality value and points too close to a nearby better value are dropped.
 * This improves the conditioning of most algorithms which use keypoints and reduce computational cost with little loss.
 *
 *
 *
 *
 *
 *
 * A basic exact anms implementation.
 *
 *
 * Slow and certain essentially...
 */
class Solver{
public:
    /**
     * @brief init initialize the anms filter with data
     * @param datas
     */
    void init(const std::vector<Data>& datas);
    /**
     * @brief init initialize the filter with data but assume that all locked are added first with infinite strength
     * @param datas
     * @param locked
     */
    virtual void init(const std::vector<Data>& datas,const std::vector<Data>& locked);
    /**
     * @brief compute
     * @param minRadius
     * @param minKeep if a negative minKeep is provided the system can remove all tracks save one. Otherwize it will stop once only minkeep remain.
     */
    virtual void compute(double minRadius, int minKeep);


    /**
     * @brief exact - is the anms exact, a reflection method for testing
     * @return
     */
    virtual bool exact();
    virtual ~Solver(){}




    template<class T>
    /**
     * @brief filter
     * @param vs
     *
     * once the anms has been inited and run, this method removes the T not of interest.
     * assumes data.id enumerates the input set
     */
    void filter(std::vector<T>& vs) const{
        std::vector<T> tmp=vs;
        vs.clear();
        for(const Data& data:filtered){
            assert(tmp.size()<data.id);
            vs.push_back(tmp[data.id]);
        }
    }

/// the input data
    std::vector<anms::Data> datas;
    /// the filtered data
    std::vector<anms::Data> filtered;
protected:
/// has the initialization been performed
    bool inited=false;
};

/**
 * @brief getIds returns the ids from the datas
 * @param datas
 * @return
 */
std::vector<int> getIds(std::vector<anms::Data>& datas);
/**
 * @brief getStrengths get the strengths from the datas
 * @param datas
 * @return
 */
std::vector<float> getStrengths(std::vector<anms::Data>& datas);
/**
 * @brief issorted is the data vector already correctly sorted
 * @param datas
 * @return
 */
bool issorted(const std::vector<Data>& datas);






















} // end namespace anms
}// end namespace cvl
