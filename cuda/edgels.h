

#include <mlib/utils/files.h>
#include <mlib/opencv_util/cv.h>
#include <mlib/utils/string_helpers.h>

#include <mlib/utils/simulator_helpers.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <kitti/odometry/kitti.h>
#include <mlib/opencv_util/cv.h>
#include <mlib/utils/cvl/convertopencv.h>



namespace cvl{




template<class T>

/**
 * @brief getGoodEdgels
 * @param im
 * @param K
 * @param P10
 * @return
 *
 * Speedup later...
 */
cv::Mat1f getGoodEdgels(cv::Mat_<T> im, Matrix3d K, PoseD P10){
    // oki compute the derivatives... shar for simplicity...
    cv::Mat1f drow;
    cv::Mat1f dcol;

    drow=cv::Mat1f(im.rows,im.cols);
    dcol=cv::Mat1f(im.rows,im.cols);

    // dy for cv is dcol and dx is drow
    cv::Scharr(im,drow,im.depth(),0,1,1.0f/32.0f);

    cv::Scharr(im,dcol,im.depth(),1,0,1.0f/32.0f);
    cv::blur(drow,drow,cv::Size(5,5));
    cv::blur(dcol,dcol,cv::Size(5,5));

//std::cout<<"gradients computed!"<<std::endl;

    // oki now get the epiline from the predicted pose...
    P10=PoseD(Vector3d(0,1,0));
    Matrix3d EKinv=P10.essential_matrix()*K.inverse();
    cv::Mat1f edgelscore(im.rows,im.cols);
    cv::Mat1f grad(im.rows,im.cols);
    cv::Mat1f mingrad(im.rows,im.cols);
    for(int r=0;r<im.rows;++r)
        for(int c=0;c<im.cols;++c){
            if(c<10 ||r<10 ||c>im.cols-10||r>im.rows-10){
                edgelscore(r,c)=0;
                grad(r,c)=0;
                mingrad(r,c)=0;
                continue;
            }

            Vector3d l=EKinv*Vector3d(c,r,1);
            l.normalize();

            double val=0;
            for(int wr=-2;wr<3;++wr){
                for(int wc=-2;wc<3;++wc){

                    float dc=dcol(r+wr,c+wc);
                    float dr=drow(r+wr,c+wc);

                    Vector3d g(dc,dr,1); // last value does not matter since we are looking for the angle!
                    g.normalize();

                    double factor=1- std::abs(cvl::dot(l,g)); // from 0 to 1 with the highest when its ortho...
                    //val+=factor;

                    val+=factor*Vector2f(dc,dr).length();
                  //  gval=Vector2d(dcol(r+wr,c+wc),drow(r+wr,c+wc)).length();
                    break;
                }
                break;
            }
            edgelscore(r,c)=float(val);
            edgelscore(r,c)=std::min(dcol(r,c),drow(r,c));

            //grad(r,c)=gval;
            //mingrad(r,c)=std::min(dcol(r,c),drow(r,c));




        }
/*
    cv::namedWindow("edgelscore", cv::WINDOW_GUI_EXPANDED);
    cv::namedWindow("grad", cv::WINDOW_GUI_EXPANDED);
    cv::namedWindow("mingrad", cv::WINDOW_GUI_EXPANDED);
    cv::imshow("edgelscore",normalize01(edgelscore));
    cv::imshow("grad",normalize01(grad));
    cv::imshow("mingrad",normalize01(mingrad));
    */


    return edgelscore;
}
}
