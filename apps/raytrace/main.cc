#include <iostream>
#include <cmath>
#include <opencv4/opencv2/highgui.hpp>



#include "sphere.h"
#include "hitable_list.h"


#include "camera.h"
#include "material.h"
#include <mlib/opencv_util/imshow.h>
#include <mlib/opencv_util/cv.h>
#include <thread>
#include <mlib/utils/random.h>
#include <mlib/utils/cvl/matrix.h>
#include <mlib/utils/mlibtime.h>


using namespace cvl;

    template<bool host>
struct Rand{


    __host__ __device__
    float u(float low, float high){
        if constexpr (host) return  mlib::randu(low,high);
        return 0;
    }


};
Rand<true> ra;


template<class T>
Vector3<T> cast_ray(const Ray<T>& r,
                    const hitable_list<T>& world,
                    int depth)
{

    hit_record<T> rec;
    if (world.hit(r, 0.001, 1e7, rec))
    {
        Ray<T> scattered;
        Vector3<T> attenuation;
        // 50 for the test
        if (depth < 2 && rec.mat.scatter(r, rec, attenuation, scattered,ra))
        {
            return point_multiply(attenuation, cast_ray<T>(scattered, world, depth+1));
        }
        return Vector3<T>(0,0,0);
    }

    Vector3<T> unit_direction = r.direction;
    T t = 0.5*(unit_direction[1] + 1.0);
    return (T(1.0f)-t)*Vector3<T>(1.0, 1.0, 1.0) + t*Vector3<T>(0.5, 0.7, 1.0);
}


template<class T>
inline uchar cap(T v)
{
    if(v<0.0f) return 0;
    if(v<255.0f) return v;
    return 255;
}


template<class T>
void raytrace(cv::Mat3f& image)
{
    int cols = image.cols;
    int rows = image.rows;

    int ns = 1;


    std::vector<Sphere<T>> spheres=test_scene<T>();
    //spheres=random_scene<T>();




    hitable_list world(spheres);


    Vector3<T> lookfrom(13,2,3);
    Vector3<T> lookat(0,0,0);
    T dist_to_focus = 10.0;
    T aperture = 0.1;

    Camera<T> cam(lookfrom, lookat, Vector3<T>(0,1,0), 20, T(cols)/T(rows), aperture, dist_to_focus);
    mlib::Timer timer;

    for(int cycles=0;cycles<100000;cycles++)
    {
        timer.tic();
        for (int row = 0; row < rows; ++row)
        {
            for (int col = 0; col < cols; ++col)
            {
                Vector3<T> color(0,0,0);
                int N=1;
                for(int i=0;i<N;++i)
                {
                    // random Ray<T> within the pixel
                    T u = T(col + mlib::randu_<T>(-0.1,1.1)) / T(cols);
                    T v = T(row + mlib::randu_<T>(-0.1,1.1)) / T(rows);
                    Ray<T> r = cam.get_ray(u, v);

                    color+=cast_ray(r, world,0);
                }
                cv::Vec3f& pixel=image(rows-row-1,col);
                for(int pi=0;pi<3;++pi)                    pixel[pi]=(pixel[pi]*cycles + color[pi])/T(cycles +N);
            }
        }
        timer.toc();
        std::cout<<timer<<std::endl;
        mlib::sleep(10);
    }


}

void parse(cv::Mat3f& image, cv::Mat3b& rgb)
{
    if(rgb.rows<image.rows||rgb.cols<image.cols)
        rgb=cv::Mat3b(image.rows, image.cols);
    for(int r=0;r<image.rows;++r)
        for(int c=0;c<image.cols;++c)
        {
            auto& opx=rgb(r,c);
            auto& ipx=image(r,c);

            opx[0] = cap(256.0f*std::sqrt(ipx[2]));
            opx[1] = cap(256.0f*std::sqrt(ipx[1]));
            opx[2] = cap(256.0f*std::sqrt(ipx[0]));
        }
}

int main()
{
    // old vec class, 42, now 72
    // pre matrix refactor 72, now 56,
    // pre material pointer refactor 56, now 45


    int rows=80;
    int cols=120;
    cv::Mat3f image(rows,cols);

    using T=double;
    std::thread thr([&](){
        raytrace<T>(image);
    });

    cv::Mat3b rgb(rows,cols);
    while(true)
    {
        parse(image, rgb);
        cvl::imshow(rgb,"in progress");
        cv::waitKey(0);
    }
    thr.join();
}
