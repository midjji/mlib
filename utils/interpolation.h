#pragma once
#include <mlib/utils/cvl/matrix.h>
namespace cvl{


template<class T>
class Bilinear{
public:
    Vector<T,4> coeffs;


    void init(int row, int col,
              T& i00,
              T& i01,
              T& i10,
              T& i11){
        coeffs[0]=i00*(col + row + col*row + T(1.0)) - i01*(col + col*row) - i10*(row + col*row) + col*i11*row;
        coeffs[1]=col*i01 - col*i11 - i00*(col + T(1.0)) + i10*(col + T(1.0));
        coeffs[2]=i10*row - i11*row - i00*(row + T(1.0)) + i01*(row + T(1.0));
        coeffs[3]=i00 - i01 - i10 + i11;
    }


    template<class Image>
    Bilinear(uint row, uint col, const Image& img){
        T i00,i01,i10,i11;
        i00=T(img(row,col));
        i01=i10=i11=i00;
        if(col+1<uint(img.cols))
            i01=T(img(row,col+1));
        if(row+1<uint(img.rows))
            i10=T(img(row+1,col));
        if(row+1<uint(img.rows) && col+1<uint(img.cols))
            i11=T(img(row+1,col+1));
        init(row,col,i00,i01,i10,i11);
    }

    template<class T2>
    T2 at(const T2& row, const T2& col) const{
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
    template<class Image1f>
    Bicubic(int row, int col,
            const Image1f& img,
            const Image1f& drow,
            const Image1f& dcol,
            const Image1f& drowcol){
        if(0<=col && 0<=row && (row+1<img.rows) && (col+1<img.cols)){
            // method 3 takes 10 ms
            // should be possible to do using a convolution which would be extremely fast!

            Vector<double,16> data(img(row,col),
                                   img(row+1,col),
                                   img(row,col+1),
                                   img(row+1,col+1),
                                   drow(row,col),
                                   drow(row+1,col),
                                   drow(row,col+1),
                                   drow(row+1,col+1),
                                   dcol(row,col),
                                   dcol(row+1,col),
                                   dcol(row,col+1),
                                   dcol(row+1,col+1),
                                   drowcol(row,col),
                                   drowcol(row+1,col),
                                   drowcol(row,col+1),
                                   drowcol(row+1,col+1));
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

