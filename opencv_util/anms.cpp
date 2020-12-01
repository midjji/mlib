#include <iostream>

#include <opencv2/imgproc/imgproc.hpp>

#include "cv.h"
#include "anms.h"




template<class T>
class Grid{
public:
    Grid(uint cols, uint rows, uint blocksize){
        bc=ceil((double)cols/(double)blocksize);
        br=ceil((double)rows/(double)blocksize);
        g.resize(bc);
        for(auto& v:g)
            v.resize(br);
    }
    T at(uint colv,uint rowv){
        assert(colv/64<g.size());
        assert(colv/64<g.size());
        return g[colv/64][rowv/64];
    }
    void set(uint colv,uint rowv,const T& v){
        assert(colv/64<g.size());
        assert(colv/64<g.size());
        g[colv/64][rowv/64]=v;
    }
    void setAll(const T& v){
        for(auto& a:g)
            for(auto& b:a)
                b=v;
    }
    void add(uint colv,uint rowv,const T& v){
        assert(colv/64<g.size());
        assert(colv/64<g.size());
        g[colv/64][rowv/64]+=v;
    }
//private:
    std::vector<std::vector<T>> g;

    int bc,br;
    int blocksize;
};

using std::cout;
using std::endl;

namespace mlib {
typedef std::vector<std::vector<std::vector<cv::KeyPoint> > > KPGrid;

inline bool compareKP(const cv::KeyPoint& a,const cv::KeyPoint& b){
    return a.response>b.response;
}
inline double dist2(const cv::KeyPoint& a,const cv::KeyPoint& b){
    return (a.pt.x - b.pt.x )*(a.pt.x - b.pt.x ) + (a.pt.y - b.pt.y )*(a.pt.y - b.pt.y );
}
inline double dist2(const cvl::Vector2d& a,const cv::KeyPoint& b){
    return (a[0] - b.pt.x )*(a[0] - b.pt.x ) + (a[1] - b.pt.y )*(a[1] - b.pt.y );
}

// this is the ideal if slow version, for a fixed radius
std::vector<cv::KeyPoint> anms(std::vector<cv::KeyPoint>& kps, double radius){
    assert(radius>1);
    std::vector<cv::KeyPoint> kpf;kpf.reserve(kps.size());
    // sort them
    sort(kps.begin(),kps.end(),compareKP);
    double R=radius*radius;
    bool test=true;
    for(const cv::KeyPoint& kp:kps){
        test=true;
        for(uint i=0;(i<kpf.size()) && test;++i){
            if(dist2(kpf[i],kp)<R)
                test=false;
        }
        if(test)
            kpf.push_back(kp);
    }
    return kpf;
}
// again slow but pretty good
std::vector<cv::KeyPoint> anms(std::vector<cv::KeyPoint>& kps, double radius, int goal){
    assert(radius>1);
    // the radius is the minimum, it is then increased untill the goal is met
    std::vector<cv::KeyPoint> kpf=anms(kps,radius);
    while(float(kpf.size())>goal*1.2)
        kpf=anms(kpf,radius+=5);
    return kpf;
}


std::vector<cv::KeyPoint>
anms(std::vector<cvl::Vector2d>& lockedkps,
     std::vector<cv::KeyPoint>& kps,
     double radius){
    // first remove anyone within radius of the locked
    double R=radius*radius;
    std::vector<cv::KeyPoint> kpf;kpf.reserve(kps.size());
    for(const cv::KeyPoint& kp:kps){
        bool test=true;
        for(const cvl::Vector2d& v: lockedkps)
            if(dist2(v,kp)<R){
                test=false;
                break;
            }
        if(test)
            kpf.push_back(kp);
    }
    return anms(kpf,radius);
}

// replace with quad tree! && implicit sort, lots of variants already in old trunk
std::pair<int,int> getMax(std::vector<cv::KeyPoint>& kps){
    float maxx=0;
    float maxy=0;
    for(const cv::KeyPoint& kp:kps){
        if(kp.pt.x>maxx)
            maxx=(std::ceil(kp.pt.x));
        if(kp.pt.y>maxy)
            maxy=(std::ceil(kp.pt.y));
    }
    return std::make_pair(int(maxx),int(maxy));
}

// less ideal but faster, still way more to do to do it ideally!( old trunk contains optimized code but this will do for now)
std::vector<cv::KeyPoint> aanms(std::vector<cv::KeyPoint>& kps, double radius){
    std::vector<cv::KeyPoint> kpf;kpf.reserve(kps.size());
    sort(kps.begin(),kps.end(),compareKP);
    double gridsize=radius*2 +4;
    std::pair<int,int> lim= getMax(kps);
    int gridx=int((lim.first/gridsize) +1);
    int gridy=int((lim.second/gridsize) +1);
    KPGrid g;g.resize(gridx);
    for(std::vector<std::vector<cv::KeyPoint> >& gy:g){
        gy.resize(gridy);
        for(std::vector<cv::KeyPoint>& gxy:gy)
            gxy.reserve(kps.size()/10);
    }

    for(const cv::KeyPoint& kp:kps){
        g[int(kp.pt.x/float(gridx))][int(kp.pt.y/float(gridy))].push_back(kp);// grid is sorted!
    }

    for(std::vector<std::vector<cv::KeyPoint> >& gy:g)
        for(std::vector<cv::KeyPoint>& gxy:gy)
            gxy=anms(gxy,radius);

    for(std::vector<std::vector<cv::KeyPoint> >& gy:g)
        for(std::vector<cv::KeyPoint>& gxy:gy)
            for(cv::KeyPoint& kp:gxy)
                kpf.push_back(kp);
    // final check
    kpf=anms(kpf,radius);
    return kpf;
}


// Threshold && pack the source image into a 1-bit-per-pixel bitmap
// The most significant bit is to the left
static cv::Mat1b GrayscaleToBitmap(const cv::Mat1b& src, uchar threshold)
{
    cv::Mat1b dst(src.rows, (src.cols + 7) / 8);

	for (int y = 0; y < src.rows; y++) {

		const uchar *s = src.ptr(y);
		uchar *d = dst.ptr(y);

		for (int x = 0; x < dst.cols; x++) {

			uchar bits = 0;

			bits |= (*s++ < threshold) ? 0 : 0x80;
			bits |= (*s++ < threshold) ? 0 : 0x40;
			bits |= (*s++ < threshold) ? 0 : 0x20;
			bits |= (*s++ < threshold) ? 0 : 0x10;

			bits |= (*s++ < threshold) ? 0 : 0x08;
			bits |= (*s++ < threshold) ? 0 : 0x04;
			bits |= (*s++ < threshold) ? 0 : 0x02;
			bits |= (*s++ < threshold) ? 0 : 0x01;

			*d++ = bits;
		}
	}
	return dst;
}

// Test whether the point pt in the mask is *not* painted
static inline bool IsPixelCleared(const cv::Mat1b& mask, const cv::Point2i& pt)
{
	// test a single bit in the mask bitmap. MSB in a byte = leftmost pixel
    // suggest paren around - in op &, remove if wrong
    return (mask(pt.x / 8, pt.y) & (1 << ((7 - pt.x) & 7))) == 0;
}

// Paint a stamp into the mask at point pt, marking the "occupied" zone around a keypoint
static inline void PaintStamp(cv::Mat1b& mask, const cv::Mat1b& stamp, const cv::Point2i& pt, int radius)
{
    uchar *p = mask.ptr((pt.x - radius) / 8) + pt.y - radius; // Note: x && y swapped

	size_t step = mask.step[0];
	for(int i = 0; i < stamp.cols; i++) {
		for (int j = 0; j < stamp.rows; j++) {
			*(p + j) |= stamp(j,i);
		}
		p += step;
	}
}

std::vector<cv::KeyPoint> FastAnms(std::vector<cv::KeyPoint>& keypoints, int height, int width, float radius)
{
    std::vector<cv::KeyPoint> kept_points;

	// Masking bitmap, with added border. x && y are swapped for better cache friendliness
    cv::Mat1b mask(int((float(width) + radius*2 + 7) / 8), int(float(height) + radius * 2));
    mask.setTo(cv::Scalar(0));

    cv::Point2f rr(radius, radius); // Translation added to point coordinates to compensate for the border.

	// Create a stamp used to mark areas in the image that already have keypoints.
	// FIXME: The diameter must be odd for the stamp to be symmetric.

    cv::Mat1b stamp = cv::Mat1b::zeros(int(radius*2), int(radius*2));
    cv::circle(stamp, rr, int(radius - 0.5), cv::Scalar(255), -1, 4);// LINE_4
	stamp = GrayscaleToBitmap(stamp, 128);

	// Sort keypoints in descending corner response strength order.

	sort(begin(keypoints), end(keypoints),
        [] (const cv::KeyPoint& a, const cv::KeyPoint& b) { return a.response > b.response; });

	kept_points.resize(keypoints.size()); // Preallocation saves time in the loop below

	// Run the filter

	int n = 0;
	for(auto& kpt:keypoints) {
        cv::Point2i pt = kpt.pt + rr;
		if (IsPixelCleared(mask, pt)) {
            PaintStamp(mask, stamp, pt, int(radius));
			kept_points[n] = kpt;
			n++;
		}
	}
	kept_points.resize(n);

	return kept_points;
}






}// end namespace mlib
