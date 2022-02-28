#include <mlib/opencv_util/draw_arrow.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <mlib/utils/mlog/log.h>
using namespace cvl;
namespace mlib{

double abs(cv::Point2f p){
    return std::sqrt(p.x*p.x + p.y*p.y);
}

void draw_arrow(cv::Mat3b& im,
               cvl::Vector2d from, // row,col
               cvl::Vector2d to, // row,col
               Color col,
               int thick,
                int length,double arrow_min_dist)
{
    drawArrow(im,from,to, col, thick, length, arrow_min_dist);
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
void drawArrow(cv::Mat3b& im, Vector2d from, Vector2d to, Color color, int thick, int length, double arrow_min_dist){


    if(!(from.isnormal()&& to.isnormal()))
    {
        mlog()<<"trying to draw bad arrow"<<from<< " "<<to<<"\n";
        return;
    }
    //from.cap(Vector2d(0,0),Vector2d(im.rows,im.cols));
    //to.cap(Vector2d(0,0),Vector2d(im.rows,im.cols));

    //assert(in(from,im.size()));
    //assert(in(to,im.size()));
    // opencv uses reverse coordinates for its points, i.e. col,row

    cv::Point2f cfrom(from[1],from[0]);
    cv::Point2f cto(to[1],to[0]);

    cv::Scalar ccolor(color.getB(), color.getG(), color.getR());


    float dist=(from - to).norm();

    if( dist<arrow_min_dist)
    {
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

void draw_legend(cv::Mat3b rgb, std::vector<std::tuple<std::string,Color>> labels){
    int fontface = cv::FONT_HERSHEY_SIMPLEX;
    double scale = 2;
    int thickness = 2;
    cv::Vec2i origin(100,100);
    int i=0;
    for(const auto& [str,col]:labels){
        cv::putText(rgb, str, origin + cv::Vec2i(0,i++*50), fontface, scale, col.fliprb().toScalar<cv::Scalar>(), thickness, 8);
    }
}
cv::Mat3b draw_arrows(const std::vector<std::pair<Vector2d,Vector2d>>& deltas,
                      cv::Mat3b rgb)
{
    rgb=rgb.clone();
    for(auto& delta:deltas)
    {
        mlib::drawArrow(rgb,
                        delta.first,
                        delta.second,
                        mlib::Color::blue(),2);
    }
    return rgb;
}

}
