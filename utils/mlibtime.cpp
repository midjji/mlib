#include "mlib/utils/mlibtime.h"
#include <iomanip>
#include <assert.h>
#include <fstream>
#include <sstream>
#include <thread>
#include <mutex>
#include <cmath>
#include <algorithm>


using std::cout; using std::endl;
namespace mlib{

// some implementations may be missing this,
// but it works for clang 9+ and gcc 8+, c++11
static_assert (sizeof(float128) >=16, "long float128 implementation is missing" );

void sleep(float128 seconds){
    if(seconds<0) return;
    std::this_thread::sleep_for(
                std::chrono::milliseconds(int(1000*seconds)));}
void sleep_ms(float128 milliseconds){
    if(milliseconds<0) return;
    std::this_thread::sleep_for(std::chrono::milliseconds(int(milliseconds)));}
void sleep_us(float128 microseconds){
    if(microseconds<0) return;
    std::this_thread::sleep_for(std::chrono::microseconds(int(microseconds)));}

std::string getIsoDate(){
    std::string datetime=getIsoDateTime();
    return datetime.substr(0,10);
}
std::string getIsoTime(){
    std::string datetime=getIsoDateTime();
    return datetime.substr(11,8);
}

std::string getIsoDateTime()
{
    std::stringstream now;
    auto tp = std::chrono::system_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>( tp.time_since_epoch() );
    size_t modulo = ms.count() % 1000;
    time_t seconds = std::chrono::duration_cast<std::chrono::seconds>( ms ).count();

    now << std::put_time( localtime( &seconds ), "%Y-%m-%d %H-%M-%S." );

    // ms
    now.fill( '0' );
    now.width( 3 );
    now << modulo;
    return now.str();
}
std::string getNospaceIsoDateTime(){
    std::string datetime=getIsoDateTime();
for(auto& c:datetime) if(c==' ')c='_';
return datetime;

    //return datetime.substr(0,10)+"_"+datetime.substr(11,8);
}






namespace  {
std::string startstr="";
std::mutex mtx;
auto program_start = std::chrono::system_clock::now();
}

std::string getNospaceIsoDateTimeofStart(){
    std::unique_lock<std::mutex> ul(mtx);// the images are allocated to new memory
    if(startstr.size()>0) return startstr;

    std::stringstream now;
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>( program_start.time_since_epoch() );
    size_t modulo = ms.count() % 1000;
    time_t seconds = std::chrono::duration_cast<std::chrono::seconds>( ms ).count();
    now << std::put_time( localtime( &seconds ), "%Y-%m-%d %H-%M-%S." );
    // ms
    now.fill( '0' );
    now.width( 3 );
    now << modulo;

    startstr=now.str();
    for(auto& c:startstr)
        if(c==' ')
            c='_';

    return startstr;
}

namespace mlibtime{
// for once we actually want this to be created on startup and not on demand
std::chrono::steady_clock clock;

// more or less when the program starts... well not long after atleast
std::chrono::time_point<std::chrono::steady_clock,std::chrono::nanoseconds > start=clock.now();
// this is the only place I will ever allow myself to do this! Never anywhere else.
//Still bad things could creep in, so beware the static initialization order fiasco.

}// end time
float128  get_steady_now(){
     return (mlibtime::clock.now()-mlibtime::start).count();
}





float128 Time::seconds(){return ns/1e9L;}
float128 Time::milli_seconds(){return ns/1e6L;}


Time& Time::operator+=(Time rhs){
    ns+=rhs.ns;
    return *this;
}
Time& Time::operator/=(Time rhs){
    ns/=rhs.ns;
    return *this;
}
bool operator==(Time lhs, Time rhs){ return lhs.ns==rhs.ns; }
bool operator!=(Time lhs, Time rhs){return !operator==(lhs,rhs);}
bool operator< (Time lhs, Time rhs){ return lhs.ns<rhs.ns; }
bool operator> (Time lhs, Time rhs){return  operator< (rhs,lhs);}
bool operator<=(Time lhs, Time rhs){return !operator> (lhs,rhs);}
bool operator>=(Time lhs, Time rhs){return !operator< (lhs,rhs);}

Time operator+ (Time lhs,Time rhs){
    return Time(lhs.ns+rhs.ns);
}
Time operator- (Time lhs,Time rhs){
    return Time(lhs.ns-rhs.ns);
}

std::string Time::str() const{

    std::stringstream ss;
    if(float128(ns)<1e3){    // less than two microseconds as ns
        ss<<std::round(float128(ns))<<"ns";
        return ss.str();
    }
    if(float128(ns)<1e6){ // less than two milliseconds as us
        ss<<std::round(float128(ns)/1e3)<<"us";
        return ss.str();
    }
    if(float128(ns)<1e9){ // less than two seconds as ms
        ss<<std::round(float128(ns)/1e6)<<"ms";
        return ss.str();
    }
    ss<<std::round(float128(ns)/1e9)<<"s";return ss.str();
}


std::ostream& operator<<(std::ostream& os,Time t){
    os<<t.str();
    return os;
}

TimeScope::TimeScope(Timer* timer):timer(timer){    timer->tic();}
TimeScope::~TimeScope(){        timer->toc();    }

ScopedDelay::ScopedDelay(float128 delay_ns){
    if(delay_ns>0 && delay_ns<1e9)
        mark += std::chrono::nanoseconds(int64_t(delay_ns));

}

ScopedDelay::~ScopedDelay(){    
    std::this_thread::sleep_until(mark);
}

Timer::Timer(){
    ts.reserve(8*256);
    dotic=true;
}
Timer::Timer(std::string name,uint capacity){
    ts.reserve(capacity+256);
    this->name=name;
    dotic=true;
}
void Timer::reserve(unsigned int size){
    ts.reserve(size);
}
bool Timer::tickable(){
    return dotic;
}
bool Timer::tockable(){
    return !dotic;
}

void Timer::tic(){
    // cout<<"name"<<name<<endl;
    // informative_assert(dotic);
    // assert(dotic);
    if(!dotic) cout<<"tic whops in timer"<<name<<endl;
    dotic=false;
    mark=mlibtime::clock.now();
}
Time Timer::toc(){
    // informative_assert(!dotic);
    // assert(!dotic);
    if(dotic) cout<<"toc whops in timer"<<name<<endl;
    dotic=true;
    recall=mlibtime::clock.now();
    std::chrono::nanoseconds diff = std::chrono::duration_cast<std::chrono::nanoseconds>(recall - mark);

    Time d(diff.count());
    ts.push_back(d);
    return d;
}





TimeScope Timer::time_scope(){return TimeScope(this);}

namespace local{
template<class T> std::string str(const T& t){
    std::ostringstream ss("");
    ss << t;
    return ss.str();
}
template<class T>
float128 getStringWidth(const T& obj){
    std::stringstream ss;
    ss<<obj;
    return ss.str().size();
}
template<class T>
std::vector<float128> getStringWidths(const std::vector<T>& objs){
    std::vector<float128> ws;ws.reserve(objs.size());
    for(const T& obj:objs)
        ws.push_back(getStringWidth(obj));
    return ws;
}
template<class T>
/**
 * @brief displayTable
 * @param headers
 * @param rows
 * @return a string containing a table convenient for display purposes
 *
 * This class has been copied here in order to minimize dependencies.
 */
std::string DisplayTable(std::vector<std::string> headers,
                         std::vector<std::vector<T>> rows,
                         std::vector<std::string> rownames=std::vector<std::string>()){

    // validate:

    // variable header size, variable content size...
    // variable content size makes this more difficult
    // assume standard int should fit regardless
    // precompute table appearance...
    // is the number of headers sufficient?

    for(auto row:rows){
        if(row.size()>headers.size())
            headers.push_back("    ?    ");
    }
    while(rownames.size()<rows.size()+1)
        rownames.push_back("");

    float128 min_header_width=7;
    std::vector<float128> widths=getStringWidths(headers);
    for(float128& w:widths){w=std::max(min_header_width,w);}



    // header widths fixed, now for row widths
    for(auto row:rows){
        for(uint e=0;e<row.size();++e){
            widths[e]=std::max(getStringWidth(row[e]),widths[e]);
        }
    }
    // now check that no row exceeds 150
    for(auto w:widths)
        if(w>50) return std::string("broken table")+str(w);




    std::stringstream ss0,ss;
    ss0<<"| ";
    for(uint i=0;i<headers.size();++i){
        ss0<<std::left << std::setw(int(widths[i])) << std::setfill(' ')<<headers[i]<<" | ";
    }
    int rowwidth=int(ss0.str().size());
    ss0<<"\n";
    // add roof

    ss<<std::left << std::setw(std::max(rowwidth-2,0)) << std::setfill('-')<<"";
    ss<<"\n";
    ss<<ss0.str();




    // per row
    for(uint row=0;row<rows.size();++row)
    {
        ss<<"| ";
        for(uint element=0;element<rows[row].size();++element)
        {

            ss << std::left << std::setw(int(widths[element])) << std::setfill(' ') << rows[row][element]<<" | ";
        }
        for(auto i=rows[row].size();i<widths.size();++i)
            ss << std::left << std::setw(int(widths[i])) << std::setfill(' ') << ""<<" | ";
        ss<<"\n";
    }
    // add floor
    ss<<std::left << std::setw(std::max(rowwidth-2,0)) << std::setfill('-')<<"";
    ss<<"\n";
    return ss.str();
}
}




















std::vector<std::string> Timer::toStrRow() const{
    std::vector<std::string> row;
    if(ts.size()==0)
        row={name,"-","-","-","-","-"};
    else
        row={name,local::str(sum()),local::str(mean()),
             local::str(median()),local::str(min()),
             local::str(max()),local::str(ts.back()),
             local::str(ts.size())};
    return row;
}


std::string Timer::str() const{
    if(ts.size()==0){
        std::stringstream ss;
        ss<<"Timer: name: "<< "No Beats";
        return ss.str();
    }
    std::vector<std::string> headers=
    {"Timer","Total",  "Mean", "Median","Min", "Max","Latest", "Samples"};
    std::vector<std::string> row=toStrRow();
    std::vector<std::vector<std::string>> rows={row};
    return local::DisplayTable(headers,rows);
}

void Timer::clear(){ts.clear();}

namespace  {
template<class T> T sum_(const std::vector<T>& xs){
    T r=0;       for(auto x:xs) r+=x;               return r;}
template<class T> T mean_(const std::vector<T>& xs){
    T r=0;       for(auto x:xs) r+=x; r/=xs.size(); return r;}
template<class T> T min_(const std::vector<T>& xs){
    T r=xs.at(0);for(auto x:xs) r= (x<r) ? x:r;     return r;}
template<class T> T max_(const std::vector<T>& xs){
    T r=xs.at(0);for(auto x:xs) r= (x>r) ? x:r;     return r;}
template<class T> T median_(std::vector<T> xs){
    if(xs.size()<1) return T(0);
    std::sort(xs.begin(),xs.end()); return xs.at((xs.size() -1)/2);}
}



std::vector<Time> Timer::times() const{    return ts;}
uint64_t Timer::samples() const{return ts.size();}
Time Timer::sum()    const{return sum_<Time>(ts);}
Time Timer::median() const{return median_<Time>(ts);}
Time Timer::mean()   const{return mean_<Time>(ts);}
Time Timer::max()    const{return max_<Time>(ts);}
Time Timer::min()    const{return min_<Time>(ts);}


Timer& NamedTimerPack::make_or_get(std::string name){
    auto it=ts.find(name);
    if(it!=ts.end()) return ts[name];
    ts[name]=Timer(name);
    return ts[name];
}
void NamedTimerPack::tic(std::string name){        make_or_get(name).tic();    }
void NamedTimerPack::toc(std::string name){        make_or_get(name).toc();    }

std::ostream& operator<<(std::ostream &os, NamedTimerPack ntp){
    std::vector<Timer> ts;ts.reserve(ntp.ts.size());
    for(const auto& t:ntp.ts)
        ts.push_back(t.second);
    return os<<ts;
}
std::ostream& operator<<(std::ostream &os, const Timer& t){
    return os<<t.str();
}
std::ostream& operator<<(std::ostream &os,const std::vector<Timer>& ts){
    if(ts.size()==0)
        cout<<"Empty timer list";
    std::vector<std::string> headers=
    {"Timer","Total",  "Mean", "Median","Min", "Max", "Samples"};
    std::vector<std::vector<std::string>> rows;
    for(const Timer& timer:ts){
        rows.push_back(timer.toStrRow());
    }
    return os<< local::DisplayTable(headers,rows);
}
}// end namespace mlib
