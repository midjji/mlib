#include <mlib/utils/spline/base.h>
namespace cvl {




BaseUniformSpline::BaseUniformSpline(double delta_time_, int degree_):delta_time_(delta_time_),degree_(degree_){}
void BaseUniformSpline::set_delta_time(double dt) {        delta_time_=dt;    }

int BaseUniformSpline::first_control_point() const{return first_control_point_;}
int BaseUniformSpline::last_control_point() const{return last_control_point_;}
// without interpolations needed if this is included
int BaseUniformSpline::current_first_control_point() const{

    if(empty()){
        mlog()<<"looking for controlpoint in empty spline\n";
        return 0;
    }

    int f= control_points().cbegin()->first;
    if(f<first_control_point()) mlog()<<"check the limiter\n";

    return f;
}
int BaseUniformSpline::current_last_control_point() const{
    if(size()==0){
        mlog()<<"looking for controlpoint in empty spline\n";
        return 0;
    }
    int r= control_points().crbegin()->first;
    if(r>last_control_point()) mlog()<<"check the limiter\n";
    return r;
}

double BaseUniformSpline::time_of_controlpoint(int i) const{
    return i*delta_time();
}

int BaseUniformSpline::get_first(double time) const {
    return get_last(time)-(degree_);
}
int BaseUniformSpline::get_last(double time) const {
    int last=int(std::floor(time / delta_time()));
    if(last>last_control_point_)
        last=last_control_point_;
    if(last<int(first_control_point_+degree_))
        last=int(first_control_point_+degree_);
    return last;
}

double BaseUniformSpline::first_valid_time() const{
    if(size()==0) return 0;
    return delta_time()*(current_first_control_point()+ int(degree_+1));
}


double BaseUniformSpline::last_valid_time() const{
    if(size()==0) return 0;
    return delta_time()*(int(current_last_control_point())) +(1-1e-9)*delta_time();
}
/// time req cpts
double BaseUniformSpline::get_first_time() const
{
    if(empty()) return 0;
    int first=control_points().cbegin()->first;
    return first*delta_time();
}
double BaseUniformSpline::get_last_time() const
{
    if(empty()) return 0;
    int e=control_points().crbegin()->first;
    return e*delta_time() + (1-1e-9)*delta_time();
}
bool BaseUniformSpline::in_spline(double time) const{
    for(int i=get_first(time);i<=get_last(time);++i){
        if(!has_index(i)) return false;
    }
    return true;
}
void BaseUniformSpline::limit_control_points_by_control_point(int cpt0, int cpt1)
{
    for(auto it=control_points().begin();it!=control_points().end();++it)
        if(it->first<cpt0 ||it->first>=cpt1)
            control_points_.erase(it); // does not touch other iterators!

}

void BaseUniformSpline::limit_control_points_by_time(double t0, double t1)
{
    int cpt0=std::floor(t0/delta_time()) - degree_;
    int cpt1=t1/delta_time();
    limit_control_points_by_control_point(cpt0,cpt1);
}

void BaseUniformSpline::set_extrapolation_bounds(int first, int last)
{
    first_control_point_=std::min(first,last - degree_-1);
    last_control_point_=last;
    limit_control_points_by_control_point(first_control_point_,
                                          last_control_point_);
}
/**
 * @brief set_extrapolation_bounds_by_observation_times
 * @param t0 first time of an observation
 * @param t1 last time of an observation
 */
void BaseUniformSpline::set_extrapolation_bounds_by_observation_times(double t0, double t1)
{

    // thefirst and last control point needed to compute the observations.
    // cant use get_first,get_last in case we are changing them...
    // - int(Degree +2)
    // +1
    int cpt0=int(std::floor(t0 / delta_time()))-5;
    int cpt1=int(std::ceil(t1 / delta_time()));

    set_extrapolation_bounds(cpt0, cpt1);
}

std::vector<double> BaseUniformSpline::interior_times(double samples_per_control_point, int border) const{


    int cpt0 = std::floor(first_valid_time()/delta_time()) + border;
    int cpt1 = std::floor(last_valid_time()/delta_time()) - border;
    if(cpt0>=cpt1) return {};

    return interior_times_for_control_point_interval(cpt0, cpt1, samples_per_control_point);
}
// note will give values beyond the extrapolation points too!
// range will be [t0,t1+(
std::vector<double> BaseUniformSpline::times_between(double t0,
                                                     double t1,
                                                     double samples_per_control_point) const {
    return interior_times_for_control_point_interval(
                std::floor(t0/delta_time()),
                std::ceil(t1/delta_time()),
                samples_per_control_point);
}
SplineBasisKoeffs BaseUniformSpline::ccbs(double time) const{
    return SplineBasisKoeffs(time,
                             SplineBasis(delta_time(), degree_,
                                         first_control_point(),
                                         last_control_point()));
}


// [low:samples:high)*delta_time()
std::vector<double>
BaseUniformSpline::interior_times_for_control_point_interval(int low,
                                                             int high,
                                                             double samples_per_control_point)
const {

    double t0 = time_of_controlpoint(low);
    double t1 = time_of_controlpoint(high);
    if(t1<t0) mlog()<<"WFT are you doing ... t1<t0"<<t0<< " "<<t1<<" "<<t1-t0<<"\n";
    if(t0==t1) mlog()<<"Warning, times for cpts are the same: "<<low<< " "<<high<<"\n";
    if(t1<t0) return std::vector<double>({0});
    if(delta_time()==0.0)
        mlog()<<"WTF are you doing ... delta_time()==0"<<"\n";
    int N=samples_per_control_point*(high-low);
    std::vector<double> ts;ts.reserve(N);

    // deal with the special case of less than 1 per interval and odd numbers, which generally should not be used...
    // but which we test anyway to show that is the case...
    int spc=samples_per_control_point;
    if(samples_per_control_point<1 || double(spc)!=samples_per_control_point) {
        double d = (t1-t0)/double(N);
        for(int i=0;i<N;++i)
            ts.push_back(t0+i*d);
        return ts;
    }

    // it is important that each control point interval gets the right number of samples, so simply counting up has potential numerical problems.
    // the first sample should also always be in the knot.
    for(int c=low;c<high;++c){

        for(int i=0;i<spc;++i){


            double time=time_of_controlpoint(c) +i*(delta_time())/double(spc);
            ts.push_back(time);
        }
    }
    return ts;
}
std::vector<double> BaseUniformSpline::regularization_times(
        double observation_t0,
                                                            double observation_t1,
                                                            double samples_per_control_point) const
{

    int cpt0=std::floor(observation_t0/delta_time() -degree_);
    int cpt1=std::ceil(observation_t1/delta_time());
    if(observation_t0==observation_t1 || cpt0==cpt1){
        mlog()<<"warning bad regularization interval"<<observation_t0<< " "<<observation_t1<< " "<<cpt0<< " "<<cpt1<<"\n";
    }
    if(cpt0==cpt1) cpt1++;
    //cpt0 = std::max(cpt0, first_control_point_);
    // reg should only be applied up to and inc the last control point to make it uniformly weighted.
    //cpt1=std::min(cpt1, last_control_point_+1);


    return interior_times_for_control_point_interval(cpt0, cpt1,samples_per_control_point);
}

/////////////////CPTS STUFFF
std::map<int,ControlPointWrapper>& BaseUniformSpline::control_points()
{
    return control_points_;
}
const std::map<int,ControlPointWrapper>& BaseUniformSpline::control_points() const{
    return control_points_;
}

int BaseUniformSpline::size() const{return control_points().size();}
bool BaseUniformSpline::empty() const{return size()==0;}
bool BaseUniformSpline::has_index(int i) const{
    return control_points().find(i)!=control_points().end();
}
BaseUniformSpline::ControlPoint_Ptr BaseUniformSpline::get(int i) const{
    if(i<first_control_point()) mlog()<<"bad user asking about before extrapol first"<<i<<", "<<first_control_point()<<"\n";
    if(i>last_control_point()) mlog()<<"bad user asking about after extrapol last"<<i<<", "<<last_control_point()<<"\n";

    auto it=this->control_points().find(i);
    if(it!=this->control_points().end())
        return it->second.ptr;
    return nullptr;
}
BaseUniformSpline::ControlPoint_Ptr BaseUniformSpline::control_point_ptr(int i)
{
    if(i<first_control_point()) mlog()<<"bad user asking about before extrapol first"<<i<<", "<<first_control_point()<<"\n";
    if(i>last_control_point()) mlog()<<"bad user asking about after extrapol last"<<i<<", "<<last_control_point()<<"\n";

    auto it=control_points().find(i);
    if(it!=control_points().end())
        return it->second.ptr;
    ControlPoint_Ptr x=make_control_point(i);
    auto& cpt=control_points_[i];
    cpt.ptr=x;
    return x;
}
void BaseUniformSpline::initialize_control_points(double start_time,
                                                  double end_time){
    int s=get_first(start_time);
    int e=get_last(end_time);
    for(int i=s;i<=e;++i) control_point_ptr(i);
}

///CERES stuff
double* BaseUniformSpline::view_control_point_param(int number) {
    // should this one insert? yes
    return control_point_ptr(number)->begin();
}



std::vector<double*> BaseUniformSpline::all_existing_parameters_before(double t1) const{
    std::vector<double*> vs;vs.reserve(size());
    for(const auto& [i, cpt]:control_points_){
        if(time_of_controlpoint(i)>=t1) break;
        vs.push_back(cpt.ptr->begin());
    }

    return vs;
}
std::vector<double*> BaseUniformSpline::view_parameter_range(double t0, double t1){
    std::vector<double*> params;
    for(int i=get_first(t0);i<=get_last(t1);++i)
        params.push_back(view_control_point_param(i));
    return params;
}
std::vector<double*> BaseUniformSpline::view_parameter_all(){
    std::vector<double*> params;
    for(auto& [a,b]:control_points())
        params.push_back(b.ptr->begin());
    return params;
}
std::vector<double*> BaseUniformSpline::view_parameter_range_by_control_point(int cpt0, int cpt1){
    std::vector<double*> params;
    for(int i=cpt0;i<=cpt1;++i)
        params.push_back(view_control_point_param(i));
    return params;
}

std::vector<double*> BaseUniformSpline::get_control_point_param_pointers(){
    std::vector<double*> ptrs;ptrs.reserve(size());
    for(auto& [i, c]:control_points()){
        ptrs.push_back(c.ptr->begin());
    }
    return ptrs;
}

/// adds time to interior
void BaseUniformSpline::add_to_interior(double time){
    for(int i=get_first(time);i<=get_last(time);++i)
        control_point_ptr(i);
}
// odd stuff
std::tuple<double,double>
BaseUniformSpline::integrate_accelleration_squared_time_cap(double t0, double t1) const{
    //s''(t) =0 for t not in [B+2, E+Degree]*dt
    double a=std::max(double(first_control_point()+1)*delta_time(),t0);
    double b=std::min(double(last_control_point()+degree_)*delta_time(),t1);
    a=std::min(a,b);
    return std::make_tuple(a,b);
}
std::tuple<int,int>
BaseUniformSpline::integrate_accelleration_squared_cpts_needed(double t0, double t1) const
{
    // I am not sure if the bounds are right. There is an edge problem where the derivative of the
    // cbs isnt right, I think this is a numeric in the integration possibly.
    // Lets test it,
    // accelleration is zero outside:
    auto ab = integrate_accelleration_squared_time_cap(t0,t1);    double a=std::get<0>(ab);    double b=std::get<1>(ab);

    a/=delta_time();
    b/=delta_time();
    int A=std::floor(a+1);
    int B=std::floor(b);
    return std::make_tuple(A,B);
}
std::string BaseUniformSpline::display() const
{

    std::stringstream ss;
    ss<<"delta_time(): "<<delta_time()<<" knots: "<<size()<<"\n";
    for(auto [i,c]: control_points())
        ss<<i<<": "<<c.ptr->str()<<"\n";
    return ss.str();
}

}
