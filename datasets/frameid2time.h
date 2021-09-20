#pragma once
#include <map>
#include <set>



namespace cvl
{
/**
 * @brief The FrameidTimeMap interface
 *
 */
class Frameid2TimeMap
{
public:
    Frameid2TimeMap()=default;
    virtual ~Frameid2TimeMap();
    virtual double time_of(int frameid) const=0;
    virtual int frameid_of(double time) const=0;
    virtual double predict_time_of(int frameid) const=0;
    // rounds down
    virtual int predict_frameid_of(double time) const=0;
    virtual std::string str() const=0;
};



class FixedFps :public Frameid2TimeMap
{

public:
    FixedFps()=default;
    FixedFps(double fps);
    ~FixedFps();
    double time_of(int frameid) const override;
    int frameid_of(double time_seconds) const override;
    // these are terrible for the default, override them
    double predict_time_of(int frameid) const override;
    int predict_frameid_of(double time) const override;
    std::string str() const override;

private:
    double fps;
    std::map<int,double> fid2time;
    std::map<double,int> time2fid;
};
/**
 * @brief The Frameid2TimeMapLive class
 * The most generic version takes the samples as they enter, and uses this to predict them.
 */
class Frameid2TimeMapLive :public Frameid2TimeMap
{

public:

    void add(int frameid, double time_seconds);
    double time_of(int frameid) const override;
    int frameid_of(double time_seconds) const override;
    // these are terrible for the default, override them
    virtual double predict_time_of(int frameid) const override;
    virtual int predict_frameid_of(double time) const override;
    std::string str() const override;
private:
    std::map<int,double> fid2time;
    std::map<double,int> time2fid;
};






}
