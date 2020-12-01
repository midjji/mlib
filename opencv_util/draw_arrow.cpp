#include <mlib/opencv_util/draw_arrow.h>
#include <opencv2/imgproc/imgproc.hpp>
using namespace cvl;
namespace mlib{

double abs(cv::Point2f p){
    return sqrt(p.x*p.x + p.y*p.y);
}


/**
 * @brief drawArrow
 * @param im
 * @param from
 * @param to
 * @param color
 * @param thick
 * @param length
 * draws a arrow...
 */
void drawArrow(cv::Mat3b im, Vector2d from, Vector2d to, Color color, int thick, int length){

    //assert(in(from,im.size()));
    //assert(in(to,im.size()));
    // opencv uses reverse coordinates for its points, i.e. col,row
    cv::Point2f cfrom(from[1],from[0]);
    cv::Point2f cto(to[1],to[0]);
    cv::Scalar ccolor(color.getB(), color.getG(), color.getR());




    if( (from - to).norm()<15){
        // only draw line
        cv::line(im,cfrom,cto,ccolor,thick);
        return;
    }

    cv::circle(im,cfrom,2,ccolor);
    cv::Point2f v=cto-cfrom;

    float c=float(std::cos(3.1415*(7.0/6.0)));        // same pi just want to get rid of a warning...
    float s=float(std::sin(3.1415*(7.0/6.0)));
    cv::Point2f a,b;
    a.x=c*v.x - s*v.y;
    a.y=s*v.x + c*v.y;



    a.x=(float)((a.x/abs(a))*length);
    a.y=(float)((a.y/abs(a))*length);
    a=a+cto;

    c=float(cos(3.1415*(5.0/6.0)));
    s=float(sin(3.1415*(5.0/6.0)));
    b.x=c*v.x - s*v.y;
    b.y=s*v.x + c*v.y;
    b.x=float((b.x/abs(b))*length);
    b.y=float((b.y/abs(b))*length);
    b=b+cto;

    cv::line(im,cfrom,cto,ccolor,thick);
    cv::line(im,cto,a,ccolor,thick);
    cv::line(im,cto,b,ccolor,thick);

}
}