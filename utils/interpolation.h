#pragma once
#include <mlib/utils/cvl/pose.h>
#include <opencv2/core.hpp>
//////////// INTERPOLATION METHODS //////////////////////////
namespace cvl{


template< class T>
class Bilinear{
public:
    Vector<T,4> coeffs;


    void init(int row, int col,const T& i00,const T& i01,const T& i10,const T& i11, uint rows, uint cols ){


        coeffs[0]=i00*(col + row + col*row + T(1.0)) - i01*(col + col*row) - i10*(row + col*row) + col*i11*row;
        coeffs[1]=                                      col*i01 - col*i11 - i00*(col + T(1.0)) + i10*(col + T(1.0));
        coeffs[2]=                                      i10*row - i11*row - i00*(row + T(1.0)) + i01*(row + T(1.0));
        coeffs[3]=                                                                  i00 - i01 - i10 + i11;


        /*
        if((row+1<rows) && (col+1<cols)){
            coeffs[0]=i00; //0
            coeffs[1]=i10 - i00; //1,0
            coeffs[2]=i01 - i00; //0,1
            coeffs[3]=i11 + i00 -(i10 + i01); //1,1
        }else{
            // deal with the borders!
            coeffs[0]=T(i00);
            coeffs[1]=T(0); // as big as possible
            coeffs[2]=T(0);
            coeffs[3]=T(0);
        }
        */
    }


    template<class T2>
    // theoretically it could be rewritten since all used things are shared between MatrixAdapter and cv::Mat, but the seg faults in cv::Mat caused by using non standard types need to be kept
    explicit Bilinear(int row, int col, const cv::Mat_<T2>& img){

        // T may be a Vector

        assert(row>=0);
        assert(col>=0);

        T i00,i01,i10,i11;

        i00=img(row,col);
        i01=i10=i11=i00;
        if(col+1<img.cols)
            i01=img(row,col+1);
        if(row+1<img.rows)
            i10=img(row+1,col);
        if(row+1<img.rows && col+1<img.cols)
            i11=img(row+1,col+1);
        init(row,col,i00,i01,i10,i11,img.rows,img.cols);
    }
    template<class T2>
    explicit Bilinear(int row, int col, const MatrixAdapter<T2>& img){

        // T may be a Vector

        T i00=img(row,col);
        T i01=img(row,col+1);
        T i10=img(row+1,col);
        T i11=img(row+1,col+1);
        init(row,col,i00,i01,i10,i11,img.rows,img.cols);
    }

    template<class T2> inline
    T2 at(const T2& row, const T2& col) const{
        //
        return T2(coeffs[0]) +  row*T2(coeffs[1]) + col*T2(coeffs[2]) +row*col*T2(coeffs[3]);
    }





    template<class T2>
    Vector2<T2> deriveAt(const T2& row,const T2& col) const{
        // bilinear variant corresponds to the simple differences!
        return Vector2<T2>(coeffs[1]+coeffs[3]*col, coeffs[2] + coeffs[3]*row);
    }
};












class Bicubic{
public:
    Matrix<double,4,4> coeffs;
    Bicubic(){}
    Bicubic(int row, int col, const cv::Mat1f& img, const cv::Mat1f& drow,const cv::Mat1f& dcol,const cv::Mat1f& drowcol){


        if((row+1<img.rows) && (col+1<img.cols)){
            // speedup later!
#if 0
            { // method 1 takes 68ms
                Vector<double,16>
                        data(img(row,col),img(row+1,col),img(row,col+1),img(row+1,col+1),
                             drow(row,col), drow(row+1,col), drow(row,col+1), drow(row+1,col+1),
                             dcol(row,col), dcol(row+1,col), dcol(row,col+1), dcol(row+1,col+1),
                             drowcol(row,col), drowcol(row+1,col), drowcol(row,col+1), drowcol(row+1,col+1));



                Matrix<double,16,16> A(1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                       0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                       -3, 3, 0, 0, -2, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                       2, -2, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                       0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
                                       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
                                       0, 0, 0, 0, 0, 0, 0, 0, -3, 3, 0, 0, -2, -1, 0, 0,
                                       0, 0, 0, 0, 0, 0, 0, 0, 2, -2, 0, 0, 1, 1, 0, 0,
                                       -3, 0, 3, 0, 0, 0, 0, 0, -2, 0, -1, 0, 0, 0, 0, 0,
                                       0, 0, 0, 0, -3, 0, 3, 0, 0, 0, 0, 0, -2, 0, -1, 0,
                                       9, -9, -9, 9, 6, 3, -6, -3, 6, -6, 3, -3, 4, 2, 2, 1,
                                       -6, 6, 6, -6, -3, -3, 3, 3, -4, 4, -2, 2, -2, -2, -1, -1,
                                       2, 0, -2, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0,
                                       0, 0, 0, 0, 2, 0, -2, 0, 0, 0, 0, 0, 1, 0, 1, 0,
                                       -6, 6, 6, -6, -4, -2, 4, 2, -3, 3, -3, 3, -2, -1, -2, -1,
                                       4, -4, -4, 4, 2, 2, -2, -2, 2, -2, 2, -2, 1, 1, 1, 1);

                Vector<double,16> b=A*data;
                for(int i=0;i<16;++i)
                    coeffs(i)=b(i);
                coeffs=coeffs.transpose();
                // verify=coeffs;
            }
#endif
#if 0

            { // method 2 takes 30 ms
                Matrix<double,4,4>

                        data(img(row,col),img(row,col+1),drow(row,col), drow(row,col+1),
                             img(row+1,col),img(row+1,col+1),drow(row+1,col),drow(row+1,row+1),
                             dcol(row,col),dcol(row,col+1),drowcol(row,col),drowcol(row,col+1),
                             dcol(row+1,col),dcol(row+1,col+1),drowcol(row+1,col),drowcol(row+1,col+1));
                Matrix<double,4,4> a(1,0,0,0,
                                     0,0,1,0,
                                     -3,3,-2,-1,
                                     2,-2,1,1);

                Matrix<double,4,4> b(1,0,-3,2,
                                     0,0,3,-2,
                                     0,1,-2,1,
                                     0,0,-1,1);
                coeffs=a*data*b;
            }
#endif
#if 1
            { // method 3 takes 10 ms
                // should be possible to do using a convolution which would be extremely fast!

                Vector<double,16> data(img(row,col),img(row+1,col),img(row,col+1),img(row+1,col+1),
                                       drow(row,col), drow(row+1,col), drow(row,col+1), drow(row+1,col+1),
                                       dcol(row,col), dcol(row+1,col), dcol(row,col+1), dcol(row+1,col+1),
                                       drowcol(row,col), drowcol(row+1,col), drowcol(row,col+1), drowcol(row+1,col+1));
                int i=0;


                coeffs(i++)= data(0);
                coeffs(i++)= data(4);
                coeffs(i++)= 3*data(1) - 3*data(0) - 2*data(4) - data(5);
                coeffs(i++)= 2*data(0) - 2*data(1) + data(4) + data(5);
                coeffs(i++)= data(8);
                coeffs(i++)= data(12);
                coeffs(i++)= 3*data(9) - data(13) - 3*data(8) - 2*data(12);
                coeffs(i++)= data(12) + data(13) + 2*data(8) - 2*data(9);
                coeffs(i++)= 3*data(2) - 3*data(0) - data(10) - 2*data(8);
                coeffs(i++)= 3*data(6) - data(14) - 3*data(4) - 2*data(12);
                coeffs(i++)= 3*data(10) - 3*data(11) + 4*data(12) + 2*data(13) + 2*data(14) + data(15) + 9*data(0) - 9*data(1) - 9*data(2) + 9*data(3) + 6*data(4) + 3*data(5) - 6*data(6) - 3*data(7) + 6*data(8) - 6*data(9);
                coeffs(i++)= 2*data(11) - 2*data(10) - 2*data(12) - 2*data(13) - data(14) - data(15) - 6*data(0) + 6*data(1) + 6*data(2) - 6*data(3) - 3*data(4) - 3*data(5) + 3*data(6) + 3*data(7) - 4*data(8) + 4*data(9);
                coeffs(i++)= data(10) + 2*data(0) - 2*data(2) + data(8);
                coeffs(i++)= data(12) + data(14) + 2*data(4) - 2*data(6);
                coeffs(i++)= 3*data(11) - 3*data(10) - 2*data(12) - data(13) - 2*data(14) - data(15) - 6*data(0) + 6*data(1) + 6*data(2) - 6*data(3) - 4*data(4) - 2*data(5) + 4*data(6) + 2*data(7) - 3*data(8) + 3*data(9);
                coeffs(i++)= 2*data(10) - 2*data(11) + data(12) + data(13) + data(14) + data(15) + 4*data(0) - 4*data(1) - 4*data(2) + 4*data(3) + 2*data(4) + 2*data(5) - 2*data(6) - 2*data(7) + 2*data(8) - 2*data(9);
                coeffs=coeffs.transpose();

            }
#endif

        }else{

            //cout<<"row,col: "<<row<<", "<<col<<endl;
            for(int i=0;i<16;++i)
                coeffs[i]=0;
            coeffs[0]=img(row,col);
        }


    }
    template<class T>
    T at(T row, T col) const{
        assert(row>=0);
        assert(col>=0);
        assert(row<1);
        assert(col<1);
        Vector<T,4> dr(T(1),row,row*row,row*row*row);
        Vector<T,4> dc(T(1),col,col*col,col*col*col);
        return cvl::dot(dr,Matrix<T,4,4>(coeffs)*dc);

    }
    Vector2f deriveAt(double row, double col) const{
        // bilinear variant corresponds to the simple differences!
        assert(row>=0);
        assert(col>=0);
        assert(row<1);
        assert(col<1);
        Vector<double,4> dr(1,row,row*row,row*row*row);
        Vector<double,4> dc(1,col,col*col,col*col*col);

        Vector<double,4> ddr(0,1,2*row,3*row*row);
        Vector<double,4> ddc(0,1,2*col,3*col*col);

        return Vector2f(cvl::dot(ddr,coeffs*dc),cvl::dot(dr,coeffs*ddc));
    }


};



}
