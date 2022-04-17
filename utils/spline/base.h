#pragma once
#include <array>
#include <iostream>
#include <map>
#include <vector>

#include <mlib/utils/cvl/matrix.h>
#include <mlib/utils/mlog/log.h>

#include <mlib/utils/spline/coeffs.h>
#include <mlib/utils/spline/control_point.h>

namespace cvl
{

class BaseUniformSpline
{


    double delta_time_=1;
    int degree_=0;


    // extrapolation limits

    int first_control_point_=std::numeric_limits<int>::min() + 100;
    int last_control_point_=std::numeric_limits<int>::max() -100;
protected:
    using ControlPoint_Ptr =ControlPoint_*;
private:
    // map is perfect if slow, consider unordered_map
    // note pointers to the coeffs are not invalidated by
    // coeffs[] and insert.
    std::map<int,ControlPointWrapper> control_points_;


public:
    BaseUniformSpline()=default;
    BaseUniformSpline(double delta_time_, int degree_);
    // the static constexpr matches, but is stronger, so no override or virtual...
    int degree() const{return degree_;}
    void set_delta_time(double dt);
    inline double delta_time() const {return delta_time_;}
    int first_control_point() const;
    int last_control_point() const;
    // without interpolations needed if this is included
    int current_first_control_point() const;
    int current_last_control_point() const;

    double time_of_controlpoint(int i) const;

    int get_first(double time) const ;
    int get_last(double time) const;
    // the first time that can be evaluated without needing to add knots
    // includes a small region which gets affected by extrapolation,
    // assuming contigious knots, inclusive
    double first_valid_time() const;
    // the last time which can be evaluated without needing to add knots
    // includes a small region which gets affected by extrapolation,
    // assuming contigious knots, inclusive
    double last_valid_time() const;
    double get_first_time() const;
    double get_last_time() const;
    bool in_spline(double time) const;

    /// clears control points outside the interval
    void limit_control_points_by_control_point(int cpt0, int cpt1);
    void limit_control_points_by_time(double t0, double t1);
    void set_extrapolation_bounds(int first, int last);
    /**
     * @brief set_extrapolation_bounds_by_observation_times
     * @param t0 first time of an observation
     * @param t1 last time of an observation
     */
    void set_extrapolation_bounds_by_observation_times(double t0, double t1);
    std::vector<double> interior_times(double samples_per_control_point=10, int border=0) const;
    // note will give values beyond the extrapolation points too!
    // range will be [t0,t1+(
    std::vector<double> times_between(double t0,
                                      double t1,
                                      double samples_per_control_point) const;

    SplineBasisKoeffs ccbs(double time) const;


    // [low:samples:high)*delta_time()
    std::vector<double> interior_times_for_control_point_interval(int low, int high, double samples_per_control_point) const;
    std::vector<double> regularization_times(double observation_t0,
                                             double observation_t1,
                                             double samples_per_control_point) const;







    /////////////// CPTS stuff below!
    //std::map<int,ControlPoint_*>& control_points();
    const std::map<int,ControlPointWrapper>& control_points() const;
    std::map<int,ControlPointWrapper>& control_points();

    int size() const;
    bool empty() const;
    bool has_index(int i) const;
    ControlPoint_Ptr get(int i) const;

    /**
     * @brief initialize_control_point
     * @param i
     * @return
     * Shall set the corresponding control point in the map and return that ptr
     */
    virtual ControlPoint_Ptr make_control_point(int i) const=0;
    ControlPoint_Ptr control_point_ptr(int i);
    void initialize_control_points(double start_time,
                                   double end_time);
    /// adds time to interior
    void add_to_interior(double time);


    // for ceres
    double* view_control_point_param(int number);
    std::vector<double*> all_existing_parameters_before(double t1) const;
    std::vector<double*> view_parameter_range(double t0, double t1);
    std::vector<double*> view_parameter_all();
    std::vector<double*> view_parameter_range_by_control_point(int cpt0, int cpt1);
    std::vector<double*> get_control_point_param_pointers();

    // odd stuff
    template<class Obs>
    std::map<int, std::vector<Obs>> group_by_knot(const std::vector<Obs>& obs){
        std::map<int, std::vector<Obs>> ret;
        for(const auto& ob:obs)
        {
            auto& vs=ret[get_first(ob.time)];
            vs.reserve(obs.size());
            vs.push_back(ob);
        }
        return ret;
    }

    std::tuple<double,double>
    integrate_accelleration_squared_time_cap(double t0, double t1) const;
    std::tuple<int,int>
    integrate_accelleration_squared_cpts_needed(double t0, double t1) const;

    std::string display() const;
};


















}
