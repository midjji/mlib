#pragma once
#include <vector>
#include <map>
#include <mlib/opencv_util/cv.h>
#include <iostream>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

namespace cvl{

template<class T>
/**
 * @brief The Histogram class
 * computes and draws histograms of vector elements using opencv
 * \todo: Verify and fix
 */
class Histogram{
public:
    Histogram(const std::vector<T>& ys, std::vector<T> buckets){
        init(ys,buckets);
    }
    Histogram(const std::vector<T>& ys, int buckets, T min, T max){
        init(ys,buckets,min,max);
    }
    Histogram(std::vector<T> ys, int buckets=10){
        init(ys,buckets);
    }
    cv::Mat3b draw(){
        if(buckets.size()==0)
            return cv::Mat3b(100,100);
        cv::Mat3b im(buckets.size()*10, buckets.size());

        //std::cout<<"bucketsize"<<buckets.size()<<std::endl;
        for(int r=0;r<im.rows;++r)
            for(int c=0;c<im.cols;++c)
                im(r,c)=cv::Vec3b(0,0,0);

        for(uint i=0;i<buckets.size();++i){
            double height=buckets.size()*10*elements[i]/total;
            cv::line(im,cv::Point(i,0),cv::Point(i,height),cv::Scalar(255,255,255),1);
        }
        cv::resize(im,im,cv::Size(256,256));
        return im;
    }
    std::vector<T> buckets;
    std::vector<uint> elements;
    uint total;
private:
    void init(const std::vector<T>& ys,
              std::vector<T> boxes){

        buckets=boxes;
        //std::cout<<"buckets.size"<<buckets.size()<<std::endl;
        assert( buckets.size()>1);
        // if the buckets are sorted this can be done faster,
        std::map<T,unsigned int> bucketset; for(uint i=0;i<buckets.size();++i) bucketset[buckets[i]]=i;
        elements.resize(buckets.size(),0);
        total=ys.size();

        for(T y:ys){
            auto it = bucketset.lower_bound(y); // first value greater than value
            if(it!=bucketset.end())
                elements[(*it).second]++;
            else
                elements.back()++;
        }

       // for(int i=0;i<buckets.size();++i)std::cout<<buckets[i]<<" "<<elements[i]<<std::endl;
    }
    void init(const std::vector<T>& ys, int buckets, T min, T max){
        /*
        std::cout<<"histogram:  init"<<std::endl;
        std::cout<<ys.size()<<std::endl;
        std::cout<<min<<" "<<max<<std::endl;
        */
         std::vector<T> bucketvs;bucketvs.reserve(buckets);

        double delta=(max-min)/(double)buckets;

        for(int i=0;i<buckets;++i){
            bucketvs.push_back(delta*i+min);
        }
        init(ys,bucketvs);
    }
    void init(std::vector<T> ys, int buckets=10){
        if(ys.size()==0) return;

        // automatic method should be clever
        sort(ys.begin(),ys.end());
        // bottom limit should be at say 5%
        // top limit at say 5%
        // region in the middle uniformly distributed over the span
        // nans should be ignored?
        T minbucket=ys.at(std::floor(0.05*(ys.size()-1)));
        T maxbucket=ys.at(std::floor(0.95*(ys.size()-1)));
        init(ys,buckets,minbucket,maxbucket);
    }
};



} // end namespace cvl
