#include <opencv2/opencv.hpp>

#include <mlib/sfm/camera/distortion_map.h>
#include <mlib/utils/string_helpers.h>


#include <mlib/utils/continuous_image.h>
#include <mlib/utils/cvl/convertopencv.h>
using namespace cvl;

cv::Vec3b rgbgradient(double dx){

    dx*=255;

    if(dx>0)
        return cv::Vec3b(dx,0,0);
    return cv::Vec3b(0,0,-dx);
}

void inspect_undistortion_map()
{



    cvl::Matrix3d K = {
        374.57306153566566f, 0., 825.93220071686324,
        0., 374.11549832701735, 639.47445426053685,
        0., 0., 1.
    };


    uint rows = 1200;
    uint cols = 1600;


    cvl::Vector5d d = {        -1.1792581125020355e-02f, 9.9459350504973558e-03, 6.3994114866457787e-05, -2.2272402548758913e-04, -2.5124899416328213e-03   };
    //cvl::Vector5d d ={0,0,0,0,0};
    // slight outwards radial distortion
    //cvl::Vector5d d ={-0.01,0,0,0,0};

    printf("Building undistort function.\n");

    PointUndistorter<double> undistort(K,rows,cols,
                                       rows/2,cols/2,d);    undistort.init();





    printf("Drawing flow from pinhole to d image.\n");
    printf("Drawing flow magnitude image.\n");
    cv::Mat1f drowf(rows, cols);    drowf.setTo(0);
    cv::Mat1f dcolf(rows, cols);    dcolf.setTo(0);


    cv::Mat3b im1(rows, cols);     im1.setTo(cv::Vec3b(0,0,0));
    cv::Mat1f im3(rows, cols);     im3.setTo(0);

double absdrowmax=0;
double absdcolmax=0;

    for (uint row = 0; row < rows; ++row) {
        for (uint col = 0; col < cols; ++col) {
            Vector2d ykd(row,col);
            Vector2d yn(0,0);
            bool ok = undistort(ykd,yn);




            if (ok) {
                Vector2d yk=K*yn;
                Vector2d delta=ykd-yk;
                // say +-25 pixel flow is the maximum

                //delta.cap(Vector2d(-5,-5),Vector2d(5,5));
                //delta*=1.0/10.0;

                float d = (ykd - yk).norm();
                //if(d>10) d=10;                d/=10.0;
                drowf(row,col)=delta(0);
                absdrowmax=std::max(std::abs(delta(0)),absdrowmax);
                dcolf(row,col)=delta(1);
                absdcolmax=std::max(std::abs(delta(1)),absdcolmax);
                im3(row,col)=d*1;

                // pretty picture...

                if (d <= 1) {
                    im1(row,col) = cv::Vec3b(0,d*200,0);
                } else if (d <= 10) {
                    im1(row,col) = cv::Vec3b(d*20,d*20,0);
                } else {
                    im1(row,col) = cv::Vec3b(0,0,d);
                }
            }
        }
    }

    cv::Mat3b drow(rows, cols);
    drow.setTo(cv::Vec3b(0,0,0));
    cv::Mat3b dcol(rows, cols);
    dcol.setTo(cv::Vec3b(0,0,0));
    for (uint row = 0; row < rows; ++row) {
        for (uint col = 0; col < cols; ++col) {
            drow(row,col) = rgbgradient(drowf(row,col));
            dcol(row,col) = rgbgradient(dcolf(row,col));
        }
    }


    cv::imshow("distortion flow drow", drow);
    cv::imshow("distortion flow dcol", dcol);
    cv::imshow("round-trip distance abs", im3);
    cv::imshow("pretty", im1);





    // cx

    {


        cv::Mat1f errim(rows,cols);
        std::vector<double> errs;errs.resize(rows*cols,-1);
        for(uint row=0;row<rows;++row){
            for(uint col=0;col<cols;++col){
                // the average error per pixel? nah
                Vector3d y(row,col,1);
                Vector3d yn_gt=K.inverse()*y;

                Vector2d yd_gt=K*undistort.distort(yn_gt.dehom());
                Vector2d yn(0,0);
                // should always work...
                if(!undistort(yd_gt,yn)){
                    std::cout<<"fail!"<<std::endl;
                }
                double err=(yn-yn_gt.dehom()).length();
                errs.push_back(err);
                if(1600*err>1e-4)
                    std::cout<<err<<std::endl;
                errim(row,col)=err;



            }
        }
        std::cout<< mlib::display(errs,false)<<std::endl;
        cv::imshow("ykd- project(unproject(yn)",errim);
    }
    cv::waitKey(0);
}





void test_warp( ){


    std::vector<double> ss4filter=   {0.011225212611744,
                                      -0.002001772970566,
                                      -0.005122394860402,
                                      -0.008085683001379,
                                      -0.008505944520736,
                                      -0.004785147796319,
                                      0.002792082997062,
                                      0.011657921185942,
                                      0.017610100237213,
                                      0.016570705800009,
                                      0.006535628941174,
                                      -0.010586199931756,
                                      -0.028810385175949,
                                      -0.039487583918833,
                                      -0.034155313941560,
                                      -0.007917259965883,
                                      0.037992960013266,
                                      0.095607259720250,
                                      0.152043298758179,
                                      0.193214364667442,
                                      0.208289205117816,
                                      0.193214364667442,
                                      0.152043298758179,
                                      0.095607259720250,
                                      0.037992960013266,
                                      -0.007917259965883,
                                      -0.034155313941560,
                                      -0.039487583918833,
                                      -0.028810385175949,
                                      -0.010586199931756,
                                      0.006535628941174,
                                      0.016570705800009,
                                      0.017610100237213,
                                      0.011657921185942,
                                      0.002792082997062,
                                      -0.004785147796319,
                                      -0.008505944520736,
                                      -0.008085683001379,
                                      -0.005122394860402,
                                      -0.002001772970566,
                                      0.011225212611744};





    cv::Mat3b tmp=cv::imread("/home/mikael/Downloads/image-3.jpg");
    cv::Mat1f img=rgb2gray<float>(tmp);
    cv::imshow("img",img);




    cvl::Matrix3d K = {
        374.57306153566566, 0., 825.93220071686324,
        0., 374.11549832701735, 639.47445426053685,
        0., 0., 1.
    };


    uint rows = 1200;
    uint cols = 1600;


    cvl::Vector5d d = {        -1.1792581125020355e-02, 9.9459350504973558e-03, 6.3994114866457787e-05, -2.2272402548758913e-04, -2.5124899416328213e-03   };

    // first create a supersampled image
    int ssf=4;
    PointUndistorter<double> undistort(K,rows,cols,rows,cols,d);    undistort.init();
    cout<<"created undist"<<endl;
    ContinuousImage<Bilinear<double>> CI(img);
    cout<<"created CI"<<endl;
    cv::Mat1f ss(img.rows*ssf,img.cols*ssf);
    for(int row=0;row<ss.rows;++row){
        for(int col=0;col<ss.cols;++col){
            Vector2d ykd=(Vector2d(row/(double)ssf,col/(double)ssf));
            Vector2d yn(0,0);
            bool ok=undistort(ykd,yn);
            if(ok){
                ss(row,col)=CI.at(K*yn);
            }
            else
                ss(row,col)=0;
        }
    }
    cout<<"created super res"<<endl;
    // apply separable filter, note that the CI allows continuation fixed value extrapolation

    cv::imshow("the unfiltered ss image",ss);

    cv::sepFilter2D(ss,ss,-1,ss4filter,ss4filter);
    cv::imshow("the filtered ss image",ss);

    cout<<"completed sep filters"<<endl;
    double ssf2=ssf*ssf;
    cv::Mat1f out(img.rows,img.cols);
    for(int row=0;row<img.rows;++row){
        for(int col=0;col<img.cols;++col){
            double v=0;
            for(int r=0;r<ssf;++r)
                for(int c=0;c<ssf;++c)
                    v+=ss(row*ssf +r,col*ssf+c);
            out(row,col)=v/ssf2;
        }
    }
    cv::imshow("out",out);
    cv::waitKey(0);
    // subsample
}








template<class T>
/**
 * @brief The Camera class
 * x_camera : a 3d point in the world coordinate system.
 * Normalized pinhole coordinates:
 * yn  = \wp(x_camera)= (x0/x2,x1/x2,1) - note the loss of depth
 * yd  = distort(yn)
 * ykd = K*yd;
 *
 *
 *
 */
class Camera {
public:

    uint rows = 1200;
    uint cols = 1600;

    cvl::Matrix3d K = {
        374.57306153566566f, 0., 825.93220071686324,
        0., 374.11549832701735, 639.47445426053685,
        0., 0., 1.
    };
    cvl::Vector5d d = {
        -1.1792581125020355e-02f, 9.9459350504973558e-03, 6.3994114866457787e-05, -2.2272402548758913e-04, -2.5124899416328213e-03
    };
    BrownDistortionFunction<T> dister=BrownDistortionFunction<T>(d);




    //d={0,0,0,0,0};



    Camera(){}
    Camera(uint rows, uint cols):rows(rows),cols(cols){}

    /**
     * @brief project
     * @param x_camera a R3 coordinate which will be projected yn=(x0/x2,x1/x2) and distorted yd=d()
     * @return
     */
    Vector<T,2> project(Vector<T,3> x_camera){
        return (K*(dister.value(x_camera.dehom())));
    }
    Vector<T,3> unproject(Vector<T,2> ykd){
        return K.inverse()*ykd.homogeneous();
    }




    void show_error(){
        // for every yd
        // so we can do this with higher precision, which we should!
        cv::Mat1f errorimg=cv::Mat1f::ones(rows,cols);
        cv::Mat1f KDvsK=cv::Mat1f::zeros(rows,cols);
        cv::Mat3b flow=cv::Mat3b::zeros(rows,cols);
        for(uint row=0;row<rows;++row){
            for(uint col=0;col<cols;++col){
                Vector<T,2> ykd0=Vector<T,2>(row,col);
                Vector<T,2> yn=unproject(Vector<T,2>(row,col)).dehom();
                Vector<T,2> ykd_rec = project(yn.homogeneous()); // not all must be valid!

                // show the error:
                Vector<T,2> error=ykd_rec-ykd0;
                // the error may be too large, well we are basically interested in the 0-1 range, but I want to know how big the error gets outside it...
                double err=error.length();
                if(err<10)
                    errorimg(row,col)=err/10.0;

                KDvsK(row,col)=(ykd0-(K*yn)).length();


            }
        }
        cv::imshow("ykd- project(unproject(yn)",errorimg);
        cv::imshow("|KD*yn - K*yn|",KDvsK);
        cv::waitKey(0);
    }
};






int main(int argc, char **argv) {
    Camera<double> cam;    cam.show_error();
    inspect_undistortion_map();
    test_warp();
    return 0;

}




