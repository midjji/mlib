#include <mlib/opencv_util/draw_box.h>
#include <opencv2/imgproc/imgproc.hpp>
namespace mlib{
void drawbox(cv::Mat3b& im,cvl::Vector2d y0,cvl::Vector2d y1, Color color){
    drawbox(im,int(y0[1]),int(y0[0]),int(y1[1]),int(y1[0]),color);
}



void drawbox(cv::Mat3b& im,int row, int col, int rows, int cols, Color color){
    auto c=cv::Scalar(color.getB(),color.getG(),color.getR());

    cv::line(im,cv::Point2i(row,col),cv::Point2i(row+rows,col),c);
    cv::line(im,cv::Point2i(row,col),cv::Point2i(row,col+cols),c);
    cv::line(im,cv::Point2i(row+rows,col+cols),cv::Point2i(row,col+cols),c);
    cv::line(im,cv::Point2i(row+rows,col+cols),cv::Point2i(row+rows,col),c);
}
}
