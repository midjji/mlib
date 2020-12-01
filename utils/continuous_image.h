#pragma once
#include <thread>
#include <mutex>
#include <mlib/utils/cvl/matrix.h>
 
#include <mlib/utils/interpolation.h>

using std::cout;using std::endl;

namespace cvl{


template<class T> inline int getFlooredScalar(const T& v){
    return std::floor(v.a);// works for all the jets!
}
template<> inline int getFlooredScalar<float>(const float& v){
    return std::floor(v);// works for all the jets!
}
template<> inline int getFlooredScalar<double>(const double& v){
    return std::floor(v);// works for all the jets!
}
template<> inline int getFlooredScalar<int>(const int& v){
    return std::floor(v);// works for all the jets!
}





template< class Interpolation> class ContinuousImage{
public:

    ContinuousImage(){}

    template<class T>
    /**
     * @brief ContinuousImage
     * @param im beware of using non cv types!
     */
    explicit ContinuousImage(cv::Mat_<T> im){

        data=MatrixAdapter<Interpolation>::allocate(im.rows,im.cols);
        datap=std::shared_ptr<Interpolation>(data.getData());


        for(uint row=0;row<data.rows;++row)
            for(uint col=0;col<data.cols;++col)
                data(row,col)=Interpolation(row,col,im);

        this->im=im;
        assert(im.isContinuous());
    }
    template<class T>
    /**
     * @brief ContinuousImage
     * @param im beware of using non cv types!
     */
    explicit ContinuousImage(MatrixAdapter<T> im){

        data=MatrixAdapter<Interpolation>::allocate(im.rows,im.cols,false);


        datap=std::shared_ptr<Interpolation>(data.data);


        for(int row=0;row<data.rows;++row)
            for(int col=0;col<data.cols;++col)
                data(row,col)=Interpolation(row,col,im);

    }

    ContinuousImage(cv::Mat1w img,cv::Mat1f drow,cv::Mat1f dcol, cv::Mat1f drowcol){

        data=MatrixAdapter<Interpolation>::allocate(img.rows,img.cols,false);
        datap=std::shared_ptr<Interpolation>(data.data);


        for(int row=0;row<data.rows;++row)
            for(int col=0;col<data.cols;++col)
                data(row,col)=Interpolation(row,col,img,drow,dcol,drowcol);

    }

    ContinuousImage(cv::Mat1f img,cv::Mat1f drow,cv::Mat1f dcol, cv::Mat1f drowcol){

        data=MatrixAdapter<Interpolation>::allocate(img.rows,img.cols);
        datap=std::shared_ptr<Interpolation>(data.getData());
        for(uint row=0;row<data.rows;++row)
            for(uint col=0;col<data.cols;++col)
                data(row,col)=Interpolation(row,col,img,drow,dcol,drowcol);

    }


    template<class T> bool safeat(const T& row, const T& col, T& val) const{

        int frow=getFlooredScalar(row);
        int fcol=getFlooredScalar(col);
        val=T(0);

        if(frow>(int)(data.rows-1) ||fcol>(int)(data.cols-1) ||fcol<0 || frow<0)
            return false;
        val=at(row,col);
        return true;
    }

    template<class T>
    Vector<T,2> deriveAt(const T& row, const T& col) const{
        return data(floor(row),floor(col)).deriveAt(row-floor(row), col-floor(col));
    }

    uint rows(){return data.rows;}
    uint cols(){return data.cols;}



    template<class T> inline T at(const Vector<T,2>& pos) const{    return at(pos(0),pos(1));}
    /*
    // sin(a + h) ~= sin(a) + cos(a) h
    template <typename T, int N> inline
    Jet<T, N> at2(const Jet<T, N>& f) {
      return Jet<T, N>((f.a), cos(f.a) * f.v);
    }
    */
    // automatic type U interpolation
    template<class T> T inline at(const T& row, const T& col) const{
        /*
        {
            double* data=(double* )im.data;
            ceres::Grid2D<double, 1>  array(data, 0, im.rows, 0, im.cols);
            ceres::BiCubicInterpolator<ceres::Grid2D<double, 1>> interpolator(array);
            double f, dfdr, dfdc;
            interpolator.Evaluate(row, col, &f, &dfdr, &dfdc);
        }
        */


        //return data(getFlooredScalar(row),getFlooredScalar(col)).at(row-ceres::floor(row), col-ceres::floor(col));

        int frow=getFlooredScalar(row);
        int fcol=getFlooredScalar(col);

        if(false &&( (frow<0) || (fcol<0)||(frow>=(int)data.rows)||(fcol>=(int)data.cols))){
            //std::unique_lock<std::mutex>(mtx);
            //std::cout<<"CI::at: "<<row<<" "<<col<<" of: "<<data.rows<<", "<<data.cols<<std::endl<<std::endl;
            return T(1);

        }
        // extrapolation...
        if(frow<0 || fcol<0 || frow>=(int)data.rows || fcol>= (int)data.cols){
            if(frow<0){
                frow=0;

            }
            if(fcol<0){
                fcol=0;

            }
            if(frow>=(int)data.rows){
                frow=data.rows-1;

            }
            if(fcol>=(int)data.cols){
                fcol=data.cols-1;

            }
            return data(frow,fcol).at(T(0), T(0));
        }



        assert(frow<(int)data.rows);
        assert(fcol<(int)data.cols);
        assert(frow>=0);
        assert(fcol>=0);
        // with bilinear of double it makes very little difference... perhaps for small
#if 1
        //return data(frow,fcol).at(row-T(frow), col-T(fcol));
        return data(frow,fcol).at(row, col);
#else
        Bilinear<double> interp(frow,fcol,im);
        return interp.at(row-T(frow), col-T(fcol));
#endif

    }














    std::shared_ptr<Interpolation> datap=nullptr;// shared data
    MatrixAdapter<Interpolation> data;
    cv::Mat1f im;





};






}
