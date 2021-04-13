#if 0
#include <opencv2/imgproc.hpp>
#include <mlib/vis/CvGL.h>
#include <iostream>

using std::cout;
using std::endl;



osg::Matrixd CvToGl(const Eigen::Matrix4d& m)
{
    return osg::Matrixd(
                m(0,0), m(0,1), m(0,2), m(0,3),
                m(1,0), m(1,1), m(1,2), m(1,3),
                m(2,0), m(2,1), m(2,2), m(2,3),
                m(3,0), m(3,1), m(3,2), m(3,3));
}

osg::Matrixd CvToGl(const Eigen::Matrix4f& m)
{
    return osg::Matrixd(
                m(0,0), m(0,1), m(0,2), m(0,3),
                m(1,0), m(1,1), m(1,2), m(1,3),
                m(2,0), m(2,1), m(2,2), m(2,3),
                m(3,0), m(3,1), m(3,2), m(3,3));
}


Eigen::Matrix4d GlToCv(const osg::Matrixd& m)
{
    Eigen::Matrix4d m2;
    m2 <<
          m(0,0), m(0,1), m(0,2), m(0,3),
            m(1,0), m(1,1), m(1,2), m(1,3),
            m(2,0), m(2,1), m(2,2), m(2,3),
            m(3,0), m(3,1), m(3,2), m(3,3);
    return m2;
}



osg::Matrixd CvPoseToGlPose(const Eigen::Matrix4f& cvPose)
{
    Eigen::Matrix4f glPose = cvPose.transpose();
    return CvToGl(glPose);
}

osg::Matrixd CvPoseToGlView(const Eigen::Matrix4d& cvPose)
{
    Eigen::Matrix3d A;
    A << 1, 0, 0,
            0,-1, 0,
            0, 0,-1;

    Eigen::Matrix4d glPose = cvPose;
    glPose.topLeftCorner<3,3>() = A * glPose.topLeftCorner<3,3>();
    glPose = glPose.transpose();

    return CvToGl(glPose);
}

osg::Matrixd CvPoseToGlView(const Eigen::Matrix4f& cvPose)
{
    Eigen::Matrix3f A;
    A << 1, 0, 0,
            0,-1, 0,
            0, 0,-1;

    Eigen::Matrix4f glPose = cvPose;
    glPose.topLeftCorner<3,3>() = A * glPose.topLeftCorner<3,3>();
    glPose = glPose.transpose();

    return CvToGl(glPose);
}

Eigen::Matrix4d GlViewToCvPose(const osg::Matrixd& glPose)
{
    Eigen::Matrix3d A;
    A << 1, 0, 0,
            0,-1, 0,
            0, 0,-1;

    Eigen::Matrix4d cvPose = GlToCv(glPose).transpose();
    cvPose.topLeftCorner<3,3>() = A * cvPose.topLeftCorner<3,3>();

    return cvPose;
}

osg::Matrixd CvKToGlProjection(const cvl::Matrix3d& K,
                               float viewportWidth, float viewportHeight,
                               float zNear, float zFar)
{
    float depth = zFar - zNear;
    float q = -(zFar + zNear) / depth;
    float qn = -2 * (zFar * zNear) / depth;

    float x0 = 0;
    float y0 = 0;

    osg::Matrixd p;
    float w = viewportWidth;
    float h = viewportHeight;


    p = osg::Matrixd(
                2*K(0,0) / w, -2*K(0,1) / w, (-2*K(0,2) + w + 2*x0) / w, 0,
                0, -2*K(1,1) / h, (-2*K(1,2) + h + 2*y0) / h, 0,
                0,             0,                          q, qn,
                0,             0,                         -1, 0);



    osg::Matrixd pT;
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            pT(i,j) = p(j,i);
        }
    }
    return pT;
}








osg::Image *CvMatToGlImage(const cv::Mat& src)
{
    // convert to cv::Mat3f from whatever...

    cv::Mat3f im3f(src.rows,src.cols);
    if(src.channels()==1)
    {
        switch (src.depth()) {

        case CV_8U: {
            for(int r=0;r<src.rows;++r)
                for(int c=0;c<src.cols;++c){
                    float v=src.at<uchar>(r,c)/255.0f;
                    im3f(r,c)=cv::Vec3f(v,v,v);
                }
            break;
        }

        case CV_16U:
        {
            for(int r=0;r<src.rows;++r)
                for(int c=0;c<src.cols;++c)
                {
                    float v=src.at<unsigned short>(r,c)/(255*255.0f);
                    im3f(r,c)=cv::Vec3f(v,v,v);
                }
            break;
        }
            //case CV_16S: pxtype = GL_SHORT; break;
        case CV_32F:
        {
            for(int r=0;r<src.rows;++r)
                for(int c=0;c<src.cols;++c)
                {
                    float v=src.at<float>(r,c);
                    im3f(r,c)=cv::Vec3f(v,v,v);
                }
            break;
        }
        default:
            assert(false && "Unimplemented pixel type");
            std::cerr<<"Unimplemented pixel type"<<std::endl;
        }
    }

    if(src.channels()==3){
        switch (src.depth()) {

        case CV_8U: {
            for(int r=0;r<src.rows;++r)
                for(int c=0;c<src.cols;++c){
                    cv::Vec3b v=src.at<cv::Vec3b>(r,c);
                    im3f(r,c)=(1.0/255.0)*cv::Vec3f(v[0],v[1],v[2]);
                }
            break;
        }
            //case CV_16S: pxtype = GL_SHORT; break;
        case CV_32F:
        {
            for(int r=0;r<src.rows;++r)
                for(int c=0;c<src.cols;++c)
                {
                    cv::Vec3f v=src.at<cv::Vec3f>(r,c);
                    im3f(r,c)=cv::Vec3f(v[0],v[1],v[2]);
                }
            break;
        }
        default:
            assert(false && "Unimplemented pixel type");
            std::cerr<<"Unimplemented pixel type"<<std::endl;
        }
    }

    osg::Image *dstImg = new osg::Image;
    dstImg->allocateImage(src.cols, src.rows, 1, GL_RGB, GL_FLOAT);
    dstImg->setOrigin(osg::Image::TOP_LEFT); // Possibly not needed

    cv::Mat dst = cv::Mat(dstImg->t(), dstImg->s(), im3f.type(), dstImg->data(), dstImg->getRowStepInBytes());

    for(int r=0;r<src.rows;++r)
        for(int c=0;c<src.cols;++c)
            dst.at<cv::Vec3f>(r,c)=im3f(r,c);
    return dstImg;

}

cv::Mat3b GlImageToCvMat(const osg::Image *src)
{
    assert(src->getDataType() == GL_UNSIGNED_BYTE); // 8-bit color is supported
    assert(src->getPixelFormat() == GL_RGB); // RGB is supported

    cv::Mat dst(src->t(), src->s(), CV_8UC3, const_cast<unsigned char*>(src->data()), src->getRowStepInBytes());
    cv::Mat tmp(dst.size(), CV_8UC3);
    cv::flip(dst, tmp, 0);	// Flip vertically
    cv::cvtColor(tmp, dst, cv::COLOR_RGB2BGR);	// Swap R and B channels

    return dst;
}
#endif
