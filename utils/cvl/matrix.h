#pragma once
/* ********************************* FILE ************************************/
/** \file    MatrixNxM.hpp
 *
 * \brief    This header contains the Matrix<T,Rows,Cols> small matrix class.
 *
 * \remark
 * - c++11
 * - no dependencies
 * - header only
 * - cuda enabled
 * - tested by test_pose.cpp
 *
 * \todo
 * - should implicit homogeneous transforms on multiplication be allowed? yes
 *
 *
 *
 * \author   Mikael Persson, Andreas Robinsson
 * \date     2015-04-01
 * \note BSD licence
 *
 ******************************************************************************/
// verify that the flag combinations are sane
#ifndef WITH_CUDA
#define WITH_CUDA 0 // this what would happen anyways, but now its explicit and intended
#endif

#if WITH_CUDA
#ifndef __CUDACC_VER_MAJOR__
//static_assert(false, "attempting to compile with cuda without cuda");
#endif
#endif

#ifndef mlib_host_device_
#if WITH_CUDA
#include <cuda_runtime.h>
#define mlib_host_device_ __host__ __device__
#else
#define mlib_host_device_
#endif
#endif

#include <assert.h>
#include <cmath>
//#include <mlib/utils/cvl/matrix_adapter.h>

#include <mlib/utils/informative_asserts.h>
#include <array>
#include <iostream>
#include <sstream>




/**
 * \namespace cvl
 * \brief The cvl namespace contains the cuda enabled templated algebra library developed at cvl and various converters
 *
 */
namespace cvl {

///meta template clarifiers

template<int a>  mlib_host_device_ constexpr bool isodd() {return a % 2 ==1;}
template<unsigned int a,unsigned int b>  mlib_host_device_ constexpr unsigned int maxv() {
    return a < b ? b : a;
}

// slightly improves alignment for common cases!
template<class T, int a> int constexpr good_mat_size(){return a;}
template<> constexpr int good_mat_size<float,3>(){return 4;}
template<> constexpr int good_mat_size<double,3>(){return 4;}
template<> constexpr int good_mat_size<float,9>(){return 16;}
template<> constexpr int good_mat_size<double,9>(){return 16;}
// you can stride the matrix class implicitly, but lots of work

// in c++ c17 and onwards, using align works as expected. except not really...
template<class T, unsigned int Rows, unsigned int Cols>
/**
 * @brief The Matrix class<T,Rows,Cols>
 * owns its own small non aligned memory
 * use for small matrices and vectors
 * primarily for geometry operations.
 *
 * cuda enabled
 * typdefs Vector<T,Rows>,Vector<Rows>
 */
class Matrix
{
protected:

public:
    /// the statically allocated data of the matrix.
    T _data[good_mat_size<T,Rows*Cols>()];


    ///@return Access element (i) with a static limit check. Useful for vectors and row-major iteration over matrices. () syntax is not pretty operator() < T >()
    template<unsigned int i> mlib_host_device_
    T& at()
    {
        static_assert(i<Rows*Cols,"index out of bounds");
        return _data[i];
    }

    ///@return Const access to element (i) with a static limit check. Useful for vectors and row-major iteration over matrices.
    template<unsigned int i> mlib_host_device_
    const T& at() const
    {
        static_assert(i<Rows*Cols,"index out of bounds");
        return _data[i];
    }

    //// Element access ////////////
    ///@return element (row, col)
    mlib_host_device_
    T& operator()(unsigned int row /** @param row */, unsigned int col /** @param col */)
    {




        assert(row < Rows);
        assert(col < Cols);
        return _data[row * Cols + col];

    }

    ///@return Const access to element (row, col)
    mlib_host_device_
    const T& operator()(unsigned int row /** @param row */, unsigned int col /** @param col */) const
    {
        assert(row < Rows);
        assert(col < Cols);
        return _data[row * Cols + col];
    }

    ///@return Access element (i). Useful for vectors and row-major iteration over matrices.
    mlib_host_device_
    T& operator()(unsigned int index /** @param index */)
    {
        assert(index<Rows*Cols);
        return _data[index];
    }

    ///@return Const access to element (i). Useful for vectors and row-major iteration over matrices.
    mlib_host_device_
    const T& operator()(unsigned int index /** @param index */) const
    {
        assert(index<Rows*Cols);
        return _data[index];
    }


    ///@return Access element (i). Useful for vectors and row-major iteration over matrices.
    mlib_host_device_
    T& operator[](unsigned int index /** @param index */)
    {
        assert(index<Rows*Cols);

        return _data[index];
    }

    ///@return Const access to element (i). Useful for vectors and row-major iteration over matrices.
    mlib_host_device_
    const T& operator[](unsigned int index /** @param index */) const
    {
        assert(index<Rows*Cols);
        return _data[index];
    }



    mlib_host_device_
    T& centerAccess(int crow /** @param crow */, int ccol /** @param ccol */)
    {
        static_assert(isodd<Rows>(),"Rows must me odd for center access");
        static_assert(isodd<Cols>(),"Cols must me odd for center access");
        crow+=Rows/2;
        ccol+=Cols/2;
        assert(crow >= 0);
        assert(ccol >= 0);
        assert(crow < int(Rows));
        assert(ccol < int(Cols));
        return _data[crow * Cols + ccol];
    }

    mlib_host_device_
    const T& centerAccess(int crow /** @param crow */, int ccol /** @param ccol */) const
    {
        static_assert(isodd<Rows>(),"Rows must me odd for center access");
        static_assert(isodd<Cols>(),"Cols must me odd for center access");
        crow+=Rows/2;
        ccol+=Cols/2;
        assert(crow >= 0);
        assert(ccol >= 0);
        assert(crow < int(Rows));
        assert(ccol < int(Cols));
        return _data[crow * Cols + ccol];
    }





    ///@return Get a pointer to the matrix or vector elements. The elements are stored in row-major order.
    T* data()    {        return _data;    }

    ///@return Get a const pointer to the matrix or vector elements. The elements are stored in row-major order.
    mlib_host_device_
    const T* data() const   {
        return _data;
    }
    /**
     * @brief getRowPointer
     * @param row
     * @return a pointer to the rowth row of the matrix
     */
    mlib_host_device_
    T* getRowPointer(unsigned int row) {
        assert(row<Rows);
        return &((*this)(row,0));
    }
    /**
     * @brief getRowPointer
     * @param row
     * @return a pointer to the rowth row of the matrix
     */
    mlib_host_device_
    const T* getRowPointer(unsigned int row) const{
        assert(row<Rows);
        return &((*this)(row,0));
    }
    /**
     * @brief Row
     * @param row
     * @return a copy of the rowth row of the matrix
     */
    mlib_host_device_
    Matrix<T,1,Cols> Row(unsigned int row) const {
        assert(row<Rows);
        return Matrix<T,1,Cols>( getRowPointer(row),true);
    }


    /**
     * @brief RowAsColumnVector
     * @param row
     * @return a copy of the rowth row of the matrix as a column vector.
     */
    mlib_host_device_
    Matrix<T,Cols,1> RowAsColumnVector(unsigned int row) {
        assert(row<Rows);
        return Matrix<T,Cols,1>( getRowPointer(row),true);
    }
    mlib_host_device_
    /**
     * @brief Col
     * @param column
     * @return a copy of the column:th column of the matrix
     */
    inline Matrix<T,Rows,1> Col(unsigned int column)   const {
        assert(column<Cols);
        Matrix<T,Rows,1> col;
        for(unsigned int r=0;r<Rows;++r)
            col[r]=(*this)(r,column);
        return col;
    }
    /**
     * @brief setRow set the values of a row to the values in data
     * @param data
     * @param row
     */
    mlib_host_device_
    void setRow(const T* data,unsigned int row){
        T* rowptr=getRowPointer(row);
        for(unsigned int i=0;i<Cols;++i)rowptr[i]=data[i];
    }
    mlib_host_device_
    /**
     * @brief setRow set row to row
     * @param rowvec
     * @param row
     */
    void setRow(const Matrix<T,Rows,1>& rowvec, unsigned int row){
        setRow(rowvec.begin(),row);
    }
    ///@return a pointer to the first element
    mlib_host_device_
    const T* begin() const{return &_data[0];}
    ///@return a pointer to the last element
    mlib_host_device_
    const T* end() const{
        return begin()+(Cols*Rows);
    }
    T& back(){ return _data[Rows*Cols-1];}
    const T& back() const{ return _data[Rows*Cols-1];}

    T* begin() {return &_data[0];}
    ///@return a pointer to the last element +1
    mlib_host_device_
    T* end() {
        return begin()+(Cols*Rows);
    }


    ///@return  the number of elements
    mlib_host_device_
    constexpr unsigned int size() const{return Cols*Rows;}
    ///@return the number of columns
    mlib_host_device_
    constexpr unsigned int cols() const {return Cols;}
    ///@return the number of rows
    mlib_host_device_
    constexpr unsigned int rows() const{return Rows;}

    /// Default constructor
    mlib_host_device_
    Matrix()=default;

    /**
         * @brief Matrix
         * @param first_val
         * @param remaining_vals
         *
         * n-coefficient constructor, e.g Matrix2f m(1,2,3,4);
         * optimizer removes all superfluous alloc and assign
         *
         *
         * M<3,3> m(1,2,3,1,2,3,1,2,3)
         * m= {1,2,3,1,2,3,1,2,3}
         */
    template<class... S>
    mlib_host_device_
    Matrix(T first_val, S... remaining_vals):
        _data{first_val,T(remaining_vals)...}   {

        static_assert(Cols*Rows>0,"empty matrix?");
        static_assert(sizeof...(S) == Cols * Rows - 1, "Incorrect number of elements given");
    }
    /**
     * @brief Matrix converting constructor
     * @param M
     */
    template<class U>
    mlib_host_device_
    Matrix(Matrix<U,Rows,Cols> M)
    {
        static_assert(Cols*Rows>0,"empty matrix?");
        Matrix& a = *this;
        for(unsigned int i=0;i<Rows*Cols;++i)
            a(i)=T(M(i));
    }



    static Matrix copy_from(const T* const t){
        // or unpack?
        Matrix m; // use a specialized type to fix this!
        for (unsigned int i = 0; i < Rows * Cols; i++) m.data()[i] = t[i];
        return m;
    }







    //// Elementwise arithmetic operations ////////////

    ///@return Add the elements of another matrix (elementwise addition)
    mlib_host_device_
    Matrix& operator+=(const Matrix& b  /** @param b */)
    {
        Matrix& a = *this;
        for (unsigned int i = 0; i < Rows * Cols; i++) {
            a(i) += b(i);
        }
        return a;
    }

    ///@return Subtract the elements of another matrix (elementwise subtraction)
    mlib_host_device_
    Matrix& operator-=(const Matrix& b  /** @param b */)
    {
        Matrix& a = *this;
        for (unsigned int i = 0; i < Rows * Cols; i++) {
            a(i) -= b(i);
        }
        return a;
    }

    ///@return (*this) elementwise multiplied by a scalar
    mlib_host_device_
    Matrix& operator*=(const T& s /** @param s */)
    {
        Matrix& a = *this;
        for (unsigned int i = 0; i < Rows * Cols; i++) {
            a(i) *= s;
        }
        return a;
    }

    ///@return (*this) elementwise divided by a scalar
    mlib_host_device_
    Matrix& operator/=(const T& s  /** @param s */)
    {
        Matrix& a = *this;
        T si=T(1.0)/s;
        for (unsigned int i = 0; i < Rows * Cols; i++) {
            a(i) *= si;
        }
        return a;
    }
    template<class S>
    mlib_host_device_
    ///@return (*this) point multiplied with another matrix
    Matrix& pointMultiply(const Matrix<S,Rows,Cols>& b  /** @param b */)
    {
        Matrix& a = *this;
        for (unsigned int i = 0; i < Rows * Cols; ++i) {
            a(i) *= T(b(i));
        }
        return a;
    }
    template<class S>
    mlib_host_device_
    ///@return (*this) point multiplied with another matrix
    Matrix& point_multiply(const Matrix<S,Rows,Cols>& b  /** @param b */)
    {
        Matrix& a = *this;
        for (unsigned int i = 0; i < Rows * Cols; ++i) {
            a(i) *= T(b(i));
        }
        return a;
    }


    ///@return Elementwise matrix addition (*this) +b
    mlib_host_device_
    Matrix operator+(const Matrix& b /** @param b */) const
    {
        Matrix c;
        const Matrix& a = *this;
        for (unsigned int i = 0; i < Rows * Cols; i++) {
            c(i) = a(i) + b(i);
        }
        return c;
    }

    ///@return Elementwise matrix subtraction (*this) -b
    mlib_host_device_
    Matrix operator-(const Matrix& b  /** @param b */) const
    {
        Matrix c;
        const Matrix& a = *this;
        for (unsigned int i = 0; i < Rows * Cols; i++) {
            c(i) = a(i) - b(i);
        }
        return c;
    }

    ///@return Element negation
    mlib_host_device_
    Matrix operator-() const
    {
        Matrix b;
        const Matrix& a = *this;
        for (unsigned int i = 0; i < Rows * Cols; i++) {
            b(i) = -a(i);
        }
        return b;
    }

    ///@return (*this) point multiplied by a scalar
    mlib_host_device_
    Matrix operator*(const T& scalar /** @param scalar*/) const
    {
        const Matrix& a = *this;
        Matrix b=a;
        b*=scalar;
        return b;
    }

    ///@return (*this) point divided by a scalar
    mlib_host_device_
    Matrix operator/(const T& scalar /** @param scalar */) const
    {
        Matrix b;
        const Matrix& a = *this;
        T iscalar=T(1.0)/scalar;
        for (unsigned int i = 0; i < Rows * Cols; i++) {
            b(i) = a(i) *iscalar;
        }
        return b;
    }




    //// Constant initializers /////////////


    mlib_host_device_
    /**
     * @brief setAll sets all to val
     * @param val
     */
    void setAll(const T& val){
        for (unsigned int i = 0; i < Rows * Cols; i++) {
            _data[i] = val;
        }
    }
    /// Set all elements to zero.
    mlib_host_device_
    void setZero()
    {
        setAll(0);
    }

    ///@return Return a matrix or vector with all elements set to zero.
    static Matrix Zero()
    {
        Matrix m;
        m.setAll(T(0));
        return m;
    }
    ///@return Return a matrix or vector with all elements set to one.
    static Matrix Ones()
    {
        Matrix m;
        m.setAll(T(1));
        return m;
    }


    template<class... S>
    /**
     * @brief diagonal makes a matrix with 0 everywhere and the values on the diagonal
     * @param first_val
     * @param remaining_vals if empty the first val is repeated
     * @return
     */
    static Matrix diagonal(T first_val, S... remaining_vals)
    {
        static_assert(Cols*Rows>0,"empty matrix?");
        Matrix m;
        m.setAll(T(0));

        if (sizeof...(S) == 0) {
            for (unsigned int i = 0; (i < Cols && i < Rows); ++i)  m(i,i) = first_val;
            return m;
        }
        else{
            static_assert(sizeof...(S)==0||((sizeof...(S) +1== Cols ) && (sizeof...(S) +1<=Rows)) ||
                    ((sizeof...(S) +1== Rows ) && (sizeof...(S) +1<=Cols))
                    , "Incorrect number of elements given");
            T b[] = { first_val, T(remaining_vals)... };
            for (unsigned int i = 0; i < Cols && i<Rows; ++i)
                m(i,i) = b[i];
        }
        return m;
    }

    ///@return Return an identity matrix.
    static Matrix Identity()    {       return diagonal(1);    }








    //// Various matrix operations ///////////////////////

    ///@return the matrix transpose
    mlib_host_device_
    Matrix<T, Cols, Rows> transpose() const
    {
        Matrix<T, Cols, Rows> b;
        const Matrix& a = *this;
        // slower than needed...
        for (unsigned int row = 0; row < Rows; row++) {
            for (unsigned int col = 0; col < Cols; col++) {
                b(col, row) = a(row, col);
            }
        }
        return b;
    }
    ///@return the trace of the matrix
    mlib_host_device_
    T trace() const{
        T tr=T(0);
        for (unsigned int i = 0; (i < Cols && i < Rows); ++i)
            tr+=(*this)(i,i);
        return tr;
    }

    ///@return Matrix determinant
    mlib_host_device_
    T determinant() const
    {
        static_assert(Rows == Cols,"Must be square matrix");
        static_assert(Rows != 1,"?");
        static_assert(Rows < 5 ,"a bad idea! => not implemented");

        const Matrix& a = *this;
        if(!(Rows==2 || Rows==3))
            return T(0);// not needed since the function is never compiled where this happens
        if (Rows == 2 ) {
            return a(0, 0) * a(1, 1) - a(0, 1) * a(1, 0);
        }
        if (Rows == 3 ) {

            // Minors
            T M00 = a(1, 1) * a(2, 2) - a(1, 2) * a(2, 1);
            T M10 = a(1, 2) * a(2, 0) - a(1, 0) * a(2, 2);
            T M20 = a(1, 0) * a(2, 1) - a(1, 1) * a(2, 0);

            return a(0, 0) * M00 + a(0, 1) * M10 + a(0, 2) * M20;
        }
    }

    ///@return Matrix inverse
    mlib_host_device_
    Matrix inverse() const
    {
        static_assert(Rows == Cols,"Must be square matrix");
        static_assert(Rows <5,"Why are you trying to invert a big matrix");

        const Matrix& a = *this;
        // switch is used since its pretty much the only way to avoid overzealous warnings
        switch(Rows){
        case 1:
        {
            Matrix b;b(0)=T(1)/a(0);
            return b;
        }
        case 2:
        {
            Matrix b;
            T idet = T(1) / determinant();
            b(0, 0) = a(1, 1) * idet;
            b(0, 1) = -a(0, 1) * idet;
            b(1, 0) = -a(1, 0) * idet;
            b(1, 1) = a(0, 0) * idet;
            return b;
        }
        case 3:
        {
            Matrix M; // Minors
            T idet; // Determinant

            M(0, 0) = a(1, 1) * a(2, 2) - a(1, 2) * a(2, 1);
            M(0, 1) = a(0, 2) * a(2, 1) - a(0, 1) * a(2, 2);
            M(0, 2) = a(0, 1) * a(1, 2) - a(0, 2) * a(1, 1);

            M(1, 0) = a(1, 2) * a(2, 0) - a(1, 0) * a(2, 2);
            M(1, 1) = a(0, 0) * a(2, 2) - a(0, 2) * a(2, 0);
            M(1, 2) = a(0, 2) * a(1, 0) - a(0, 0) * a(1, 2);

            M(2, 0) = a(1, 0) * a(2, 1) - a(1, 1) * a(2, 0);
            M(2, 1) = a(0, 1) * a(2, 0) - a(0, 0) * a(2, 1);
            M(2, 2) = a(0, 0) * a(1, 1) - a(0, 1) * a(1, 0);

            idet =T(1.0)/( a(0, 0) * M(0, 0) + a(0, 1) * M(1, 0) + a(0, 2) * M(2, 0));

            return (M * idet);
        }
        case 4:
        {
            T mat[16];
            T dst[16];
            T tmp[16];

            // temp array for pairs
            T src[16];

            //Copy all of the elements into the linear array
            //int k=0;
            //for(int i=0; i<4; i++)
            //	for(int j=0; j<4; j++)
            //		mat[k++] = matrix[i][j];

            for(int i=0; i<16; i++)
                mat[i] = _data[i];




            // array of transpose source rix
            T det;

            /*determinant*/
            /*transposematrix*/
            for(int i=0;i<4;i++) //>
            {
                src[i]=mat[i*4];
                src[i+4]=mat[i*4+1];
                src[i+8]=mat[i*4+2];
                src[i+12]=mat[i*4+3];
            }

            // calculate pairs for first 8 elements (cofactors)
            tmp[0]=src[10]*src[15];
            tmp[1]=src[11]*src[14];
            tmp[2]=src[9]*src[15];
            tmp[3]=src[11]*src[13];
            tmp[4]=src[9]*src[14];
            tmp[5]=src[10]*src[13];
            tmp[6]=src[8]*src[15];
            tmp[7]=src[11]*src[12];
            tmp[8]=src[8]*src[14];
            tmp[9]=src[10]*src[12];
            tmp[10]=src[8]*src[13];
            tmp[11]=src[9]*src[12];

            // calculate first 8 elements (cofactors)
            dst[0]=tmp[0]*src[5]+tmp[3]*src[6]+tmp[4]*src[7];
            dst[0]-=tmp[1]*src[5]+tmp[2]*src[6]+tmp[5]*src[7];
            dst[1]=tmp[1]*src[4]+tmp[6]*src[6]+tmp[9]*src[7];
            dst[1]-=tmp[0]*src[4]+tmp[7]*src[6]+tmp[8]*src[7];
            dst[2]=tmp[2]*src[4]+tmp[7]*src[5]+tmp[10]*src[7];
            dst[2]-=tmp[3]*src[4]+tmp[6]*src[5]+tmp[11]*src[7];
            dst[3]=tmp[5]*src[4]+tmp[8]*src[5]+tmp[11]*src[6];
            dst[3]-=tmp[4]*src[4]+tmp[9]*src[5]+tmp[10]*src[6];
            dst[4]=tmp[1]*src[1]+tmp[2]*src[2]+tmp[5]*src[3];
            dst[4]-=tmp[0]*src[1]+tmp[3]*src[2]+tmp[4]*src[3];
            dst[5]=tmp[0]*src[0]+tmp[7]*src[2]+tmp[8]*src[3];
            dst[5]-=tmp[1]*src[0]+tmp[6]*src[2]+tmp[9]*src[3];
            dst[6]=tmp[3]*src[0]+tmp[6]*src[1]+tmp[11]*src[3];
            dst[6]-=tmp[2]*src[0]+tmp[7]*src[1]+tmp[10]*src[3];
            dst[7]=tmp[4]*src[0]+tmp[9]*src[1]+tmp[10]*src[2];
            dst[7]-=tmp[5]*src[0]+tmp[8]*src[1]+tmp[11]*src[2];

            // calculate pairs for second 8 elements(cofactors)
            tmp[0]=src[2]*src[7];
            tmp[1]=src[3]*src[6];
            tmp[2]=src[1]*src[7];
            tmp[3]=src[3]*src[5];
            tmp[4]=src[1]*src[6];
            tmp[5]=src[2]*src[5];
            tmp[6]=src[0]*src[7];
            tmp[7]=src[3]*src[4];
            tmp[8]=src[0]*src[6];
            tmp[9]=src[2]*src[4];
            tmp[10]=src[0]*src[5];
            tmp[11]=src[1]*src[4];

            // calculate second 8 elements (cofactors)
            dst[8]=tmp[0]*src[13]+tmp[3]*src[14]+tmp[4]*src[15];
            dst[8]-=tmp[1]*src[13]+tmp[2]*src[14]+tmp[5]*src[15];
            dst[9]=tmp[1]*src[12]+tmp[6]*src[14]+tmp[9]*src[15];
            dst[9]-=tmp[0]*src[12]+tmp[7]*src[14]+tmp[8]*src[15];
            dst[10]=tmp[2]*src[12]+tmp[7]*src[13]+tmp[10]*src[15];
            dst[10]-=tmp[3]*src[12]+tmp[6]*src[13]+tmp[11]*src[15];
            dst[11]=tmp[5]*src[12]+tmp[8]*src[13]+tmp[11]*src[14];
            dst[11]-=tmp[4]*src[12]+tmp[9]*src[13]+tmp[10]*src[14];
            dst[12]=tmp[2]*src[10]+tmp[5]*src[11]+tmp[1]*src[9];
            dst[12]-=tmp[4]*src[11]+tmp[0]*src[9]+tmp[3]*src[10];
            dst[13]=tmp[8]*src[11]+tmp[0]*src[8]+tmp[7]*src[10];
            dst[13]-=tmp[6]*src[10]+tmp[9]*src[11]+tmp[1]*src[8];
            dst[14]=tmp[6]*src[9]+tmp[11]*src[11]+tmp[3]*src[8];
            dst[14]-=tmp[10]*src[11]+tmp[2]*src[8]+tmp[7]*src[9];
            dst[15]=tmp[10]*src[10]+tmp[4]*src[8]+tmp[9]*src[9];
            dst[15]-=tmp[8]*src[9]+tmp[11]*src[10]+tmp[5]*src[8];

            // calculate determinant
            det=src[0]*dst[0]+src[1]*dst[1]+src[2]*dst[2]+src[3]*dst[3];

            // calculate matrix inverse
            det=T(1)/det;

            for(int j=0;j<16;j++) //>
                dst[j]*=det;

            //Copy everything into the output
            return copy_from(dst);
        }
        default:
            return (*this);
        }
    }


    bool is_in(const Matrix lower,const Matrix higher) const{
        for(unsigned int i=0;i<Rows*Cols; ++i){
            if(_data[i]< lower._data[i]) return false;
            if(_data[i]> higher._data[i]) return false;
        }
        return true;
    }

    void cap(const Matrix lower,const  Matrix higher){
        for(unsigned int i=0;i<Rows*Cols; ++i){
            if(_data[i]< lower._data[i]) _data[i]=lower._data[i];
            if(_data[i]> higher._data[i]) _data[i]=higher._data[i];
        }
    }

    ///@return Matrix multiplication
    template<unsigned int N>
    mlib_host_device_
    Matrix<T, Rows, N> operator*(const Matrix<T, Cols, N>& b  /** @param b */) const
    {
        Matrix<T, Rows, N> c;
        const Matrix& a = *this;
        for (unsigned int row = 0; row < Rows; row++) {
            for (unsigned int col = 0; col < N; col++) {

                T sum = T(0);
                for (unsigned int i = 0; i < Cols; i++) {
                    sum += a(row, i) * b(i, col);
                }
                c(row, col) = sum;
            }
        }
        return c;
    }




    mlib_host_device_
    /**
     * @brief perElementMultiply - elementwize product matlab: .*
     * @param b
     * @return
     */
    Matrix<T, Rows, Cols> perElementMultiply(const Matrix<T, Rows, Cols>& b) const
    {

        Matrix<T, Rows, Cols> out;
        const Matrix& a = *this;
        for (unsigned int row = 0; row < Rows; row++)
            for (unsigned int col = 0; col < Cols; col++)
                out(row,col)=a(row,col)*b(row,col);
        return out;
    }



    ///@return the inner product of this and another vector @param b
    template<unsigned int Rows2, unsigned int Cols2>
    mlib_host_device_
    T dot(const Matrix<T, Rows2, Cols2>& b) const
    {
        static_assert((Cols == 1 || Rows == 1),"The dot product is only defined for vectors.");
        static_assert( (Cols2 == 1 || Rows2 == 1),"The dot product is only defined for vectors.");
        static_assert(Rows * Cols == Rows2 * Cols2,
                "The vectors in a dot product must have the same number of elements.");

        T sum = T(0);
        const Matrix& a = *this;
        for (unsigned int i = 0; i < Rows * Cols; ++i) {
            sum += a(i) * b(i);
        }
        return sum;
    }



    mlib_host_device_
    /**
     * @brief cross  Compute the cross product of this and another vector
     * @param b
     * @return
     */
    inline Matrix<T,3,1> cross(const Matrix& b) const
    {
        static_assert((Rows == 3 && Cols == 1) || (Rows == 1 && Cols == 3),
                "The cross product is only defined for vectors of length 3.");

        const Matrix& a = *this;
        Matrix<T,3,1> c(
                    a(1) * b(2) - a(2) * b(1),
                    a(2) * b(0) - a(0) * b(2),
                    a(0) * b(1) - a(1) * b(0)
                    );
        return c;
    }


    mlib_host_device_
    /**
     * @brief crossMatrix Return the cross product matrix corresponding to the vector
     * @return
     */
    Matrix<T, 3, 3> crossMatrix() const
    {
        static_assert((Rows == 3 && Cols == 1) || (Rows == 1 && Cols == 3),
                "The cross product matrix is only defined for vectors of length 3.");

        return Matrix<T, 3, 3> ( 0, -at<2>(), at<1>(),
                                 at<2>(), 0, -at<0>(),
                                 -at<1>(), at<0>(), 0);


    }

    ///@return The sum of all elements
    mlib_host_device_
    T sum() const
    {
        const Matrix& a = *this;
        T sum = T(0);
        for (unsigned int i = 0; i < Rows * Cols; i++) {
            sum += a(i);
        }
        return sum;
    }

    ///@return The sum of the squared elements
    mlib_host_device_
    inline T squaredNorm() const
    {
        const Matrix& a = *this;
        T sum = a(0) * a(0);
        for (unsigned int i = 1; i < Rows * Cols; ++i) {
            sum += a(i) * a(i); // does not do the right thing for complex values
        }
        return sum;
    }

    /// @return The L2 norm
    mlib_host_device_
    T norm() const
    {
        return std::sqrt(squaredNorm());
    }
    /// @return The L2 norm
    mlib_host_device_
    T norm2() const
    {
        return squaredNorm();
    }


    ///@return The vector L2 norm, with a different name
    mlib_host_device_
    T length() const
    {
        static_assert(Cols == 1 || Rows == 1,
                "length() is only defined for vectors. Use norm() with matrices.");
        return norm();
    }

    /// the matrix divided by its own L2-norm
    mlib_host_device_
    void normalize()
    {
        (*this) /= norm();
    }
    ///@return a L2 normalized copy
    Matrix normalized() const
    {
        return (*this) * (T(1) / norm());
    }

    ///@return Homogeneous coordiates, its always the last row
    mlib_host_device_
    Matrix<T, Rows - 1, Cols> hnormalized() const
    {
        Matrix< T, Rows - 1, Cols> b;
        const Matrix& a = *this;
        T iz;
        for (unsigned int col = 0; col < Cols; col++) {
            iz=(T(1.0)/a(Rows - 1, col));
            for (unsigned int row = 0; row < Rows - 1; row++) {
                b(row, col) = a(row, col) *iz;
            }
        }
        return b;
    }
    ///@return the hnormalized() vector
    mlib_host_device_
    Matrix<T, Rows - 1, Cols> dehom() const
    {
        return hnormalized();
    }
    mlib_host_device_
    Matrix<T, Rows - 1, Cols> drop_first() const noexcept
    {
        Matrix< T, Rows - 1, Cols> b;
        const Matrix& a = *this;
        for (unsigned int col = 0; col < Cols; ++col) {
            for (unsigned int row = 1; row < Rows ; ++row) {
                b(row-1, col) = a(row, col);
            }
        }
        return b;
    }

    mlib_host_device_
    Matrix<T, Rows - 1, Cols> drop_last() const noexcept
    {
        Matrix< T, Rows - 1, Cols> b;
        const Matrix& a = *this;
        for (unsigned int col = 0; col < Cols; ++col) {
            for (unsigned int row = 0; row < Rows - 1; ++row) {
                b(row, col) = a(row, col);
            }
        }
        return b;
    }

    Matrix<T,Rows,Cols> reverse(){
        Matrix<T,Rows,Cols> m;
        for(unsigned int i=0;i<Rows*Cols;++i)
            m(Rows*Cols-i-1)=(*this)(i);
        return m;
    }



    ///@return the matrix with a final row of ones
    mlib_host_device_
    Matrix< T, Rows + 1, Cols> homogeneous() const
    {
        Matrix< T, Rows + 1, Cols> b;
        const Matrix& a = *this;
        for (unsigned int col = 0; col < Cols; col++) {
            for (unsigned int row = 0; row < Rows; row++) {
                b(row, col) = a(row, col);
            }
            b(Rows, col) = T(1.0);
        }
        return b;
    }
    mlib_host_device_
    Matrix< T, Rows+1, 1> push_first_zero() const{
        static_assert(Cols==1,"");

        Matrix< T, Rows+1, 1> m=Matrix< T, Rows+1, 1>::zero();
        const Matrix& a = *this;
        for(int i=0;i<Rows;++i)
            m[i+1]=a[i];
        return m;
    }
    ///inplace line normalization:
    mlib_host_device_
    void  lineNormalize(){
        static_assert((Rows==3 && Cols == 1)||(Rows==1 && Cols == 3),"Line norm only defined for 3 vectors here");
        if((Rows==3 && Cols == 1)||(Rows==1 && Cols == 3)){
            if(_data[2]<0)
                (*this)*=T(1.0)/std::sqrt(_data[0]*_data[0] + _data[1]*_data[1]);
            else
                (*this)*=-T(1.0)/std::sqrt(_data[0]*_data[0] + _data[1]*_data[1]);
        }
    }
    ///@return whether the matrix or vector has at least one NaN element
    bool isnan() const
    {
        const Matrix& a = *this;
        for (unsigned int i = 0; i < Rows * Cols; ++i) {
            if(std::isnan(a(i))) return true;
        }
        return false;
    }
    ///@return true if any value is inf
    bool isinf() const
    {
        const Matrix& a = *this;
        for (unsigned int i = 0; i < Rows * Cols; ++i) {
            if(std::isinf(a(i))) return true;
        }
        return false;
    }
    ///@return true if all values are not nan and not inf
    inline bool isnormal()const{// std::isnormal fails for 0!
        return !(isnan()||isinf());
    }

    ///@return true if is a identical to b
    mlib_host_device_
    bool operator==(const Matrix& b /** @param b */) const
    {
        const Matrix& a = *this;

        for (unsigned int i = 0; i < Rows * Cols; i++) {
            if (a(i) != b(i)) return false;
        }
        return true;
    }
    // utility
    ///@return the maximum value in the matrix
    mlib_host_device_
    T max(){
        const Matrix& A = *this;
        T maxv=A(0);
        for(int i=1;i<Rows*Cols;++i)
            if(maxv<A(i))maxv=A(i);
        return maxv;
    }
    ///@return the minimum value in the matrix
    mlib_host_device_
    T min(){
        const Matrix& A = *this;
        T minv=A(0);
        for(unsigned int i=1;i<Rows*Cols;++i)
            if(minv>A(i))minv=A(i);
        return minv;
    }

    mlib_host_device_
    /**
     * @brief minmax sets the minv and maxv to the min and max values respectively
     * @param minv
     * @param maxv
     */
    void minmax(T& minv,T& maxv){
        const Matrix& A = *this;
        minv=maxv=A(0);
        for(unsigned int i=1;i<Rows*Cols;++i){
            if(maxv<A(i))maxv=A(i);
            if(minv>A(i))minv=A(i);
        }
    }
    ///@return max of elementwise absolute value, only for reals
    mlib_host_device_
    T absMax(){
        const Matrix& A = *this;
        T maxv=A(0);
        T tmp;
        for(unsigned int i=1;i<Rows*Cols;++i){
            tmp=A(i);
            if(tmp<0)       tmp=-tmp;
            if(maxv<tmp)    maxv=tmp;
        }
        return maxv;
    }
    mlib_host_device_
    T absSum(){
        const Matrix& A = *this;
        T absv=std::abs(A(0));

        for(unsigned int i=1;i<Rows*Cols;++i)
            absv+=std::abs(A(i));
        return absv;
    }
    ///@return Elementwise absolute
    mlib_host_device_
    Matrix abs(){
        Matrix A=(*this);
        T val;
        for(unsigned int i=0;i<Rows*Cols;++i){
            val=(*this)(i);
            if(val<0)
                val=-val;
            A(i)=val;
        }
        return A;
    }

    bool is_symmetric() const{
        if(Rows!=Cols) return false;
        return *this==this->transpose();

    }



#if 0

    mlib_host_device_
    /**
     * @brief getSubMatrix - submatrices do not take ownership of data!
     * @param rowstart
     * @param colstart
     * @param height
     * @param width
     * @return a submatrix which maps to the data in this matrix
     */
    MatrixAdapter<T> getSubMatrix(uint rowstart, uint colstart, uint height, uint width){
        assert(rowstart>0);
        assert(colstart>0);
        assert(rowstart<=Rows);
        assert(colstart<=Cols);
        assert(rowstart+height<=Rows);
        assert(colstart+width<=Cols);
        unsigned int stride=Cols;
        MatrixAdapter<T> m(&(*this)(rowstart,colstart),height,width,stride);
        return m;
    }
#endif




    ///@return The specified block. Note blocks are copies! of submatrixes use for small stuff!
    template<unsigned int rowstart,unsigned int colstart, unsigned int Height,unsigned int Width>
    mlib_host_device_
    Matrix<T, Height, Width> getBlock() const
    {
        static_assert(rowstart + Height <= Rows,"Source matrix is too small");
        static_assert(colstart + Width  <= Cols,"Source matrix is too small");
        Matrix<T,Height, Width> out;
        for(unsigned int row=0;row<Height;++row)
            for(unsigned int col=0;col<Width;++col)
                out(row,col)=(*this)(row+rowstart,col+colstart);
        return out;
    }
    ///@return The specified block. Note blocks are copies! of submatrixes use for small stuff!
    template<unsigned int rowstart,unsigned int colstart, unsigned int Height,unsigned int Width>
    mlib_host_device_
    Matrix<T, Height*Width,1> vectorize_block() const
    {
        static_assert(rowstart + Height <= Rows,"Source matrix is too small");
        static_assert(colstart + Width  <= Cols,"Source matrix is too small");
        Matrix<T,Height*Width,1 > out;
        for(unsigned int row=0;row<Height;++row)
            for(unsigned int col=0;col<Width;++col)
                out(row*Rows + col)=(*this)(row+rowstart,col+colstart);
        return out;
    }
    Matrix<T, Rows*Cols,1> vectorize(){
        Matrix<T,Rows*Cols,1 > out;
        for(unsigned int row=0;row<Rows;++row)
            for(unsigned int col=0;col<Cols;++col)
                out(row*Rows + col)=(*this)(row,col);
        return out;
    }
    ///@return the the top left 3x3 part
    mlib_host_device_
    Matrix<T,3,3> getRotationPart() const{
        return getBlock<0,0,3,3>();
    }
    ///@return the 4th column
    mlib_host_device_
    Matrix<T,3,1> getTranslationPart() const{
        return getBlock<0,3,3,1>();
    }

    bool in(Matrix lower, Matrix upper /*not inluding*/) const{
        const Matrix& a = *this;
        for(unsigned int index=0;index<Rows*Cols;++index)
            if(a(index)< lower(index)) return false;
        for(unsigned int index=0;index<Rows*Cols;++index)
            if(a(index)>= upper(index)) return false;
        return true;
    }
    std::array<T,Rows*Cols> std_array(){
        std::array<T,Rows*Cols> arr;
        for(uint i=0;i<Rows*Cols;++i)
            arr[i]=_data[i];
        return arr;
    }
    template<unsigned int RowsB>
    Matrix<T,Rows+RowsB,1> concat(Matrix<T,RowsB,1> b){
        static_assert(Cols==1 ,"Vectors only");

        Matrix<T,Rows+RowsB,1> ab;
        uint j=0;
        for(uint i=0;i<Rows;++i)
            ab[j++] = _data[i];
        for(uint i=0;i<RowsB;++i)
            ab[j++] = b._data[i];
        return ab;
    }
};



template<uint Rows, uint Cols, class T> Matrix<T, Rows, Cols>
point_multiply(const Matrix<T, Rows, Cols>& a, const Matrix<T, Rows, Cols>& b){
    Matrix<T, Rows, Cols> c;
    for(uint i=0;i<Rows*Cols;++i)
        c[i] = a[i]*b[i];
    return c;
}











// for performance and convenience ...

template<class T>
/**
 * @brief operator * (Matrix3x3 * Vector2 is implicitly Matrix3x3 * (Vector2,1))
 * @param M
 * @param v
 * @return
 */
Matrix<T,2,1> operator*(const Matrix<T,3,3>& M, const Matrix<T,2,1>& v){
    return (M*v.homogeneous()).dehom();
}
/**
 * @brief operator * (Matrix4x4 * Vector3 is implicitly Matrix4x4 * (Vector3,1))
 * @param M
 * @param v
 * @return
 */
template<class T>
Matrix<T,3,1> operator*(const Matrix<T,4,4>& M, const Matrix<T,3,1>& v){
    return (M*v.homogeneous()).dehom();
}


template<typename T, unsigned int Rows, unsigned int Cols>
mlib_host_device_
/**
 * @brief operator * Free scalar-matrix multiplication s * matrix
 * @param s
 * @param a
 * @return s*a
 */
Matrix<T, Rows, Cols> operator*(const T& s, const Matrix<T, Rows, Cols>& a)
{
    Matrix<T, Rows, Cols> b;
    for (unsigned int i = 0; i < Rows * Cols; ++i) {
        b(i) = a(i) * s;
    }
    return b;
}

template<class T, unsigned int Rows, unsigned int Cols,unsigned int Rows2, unsigned int Cols2>
mlib_host_device_
/**
 * @brief dot Compute the inner product of this and another vector
 * @param a
 * @param b
 * @return
 */
inline T dot(const Matrix<T, Rows, Cols>& a, const Matrix<T, Rows2, Cols2>& b)
{
    static_assert((Cols == 1 || Rows == 1),"The dot product is only defined for vectors.");
    static_assert( (Cols2 == 1 || Rows2 == 1),"The dot product is only defined for vectors.");
    static_assert(Rows * Cols == Rows2 * Cols2,
            "The vectors in a dot product must have the same number of elements.");

    T sum = T(0);
    for (unsigned int i = 0; i < Rows * Cols; ++i) {
        sum += a(i) * b(i);
    }
    return sum;
}

///@return Elementwise absolute value of @param a
template<typename T, unsigned int Rows, unsigned int Cols>
mlib_host_device_
Matrix<T, Rows, Cols> abs(const Matrix<T, Rows, Cols>& a)
{
    Matrix<T, Rows, Cols> b;
    for (unsigned int i = 0; i < Rows * Cols; i++) {
        if(a(i)<0)
            b(i) = -a(i);
        else
            b(i)=a(i);
    }
    return b;
}

///@return Elementwise square root
template<typename T, unsigned int Rows, unsigned int Cols>
mlib_host_device_
Matrix<T, Rows, Cols> sqrt(const Matrix<T, Rows, Cols>& a  /** @param a */)
{
    Matrix<T, Rows, Cols> b;
    for (unsigned int i = 0; i < Rows * Cols; i++) {
        b(i) = sqrt(a(i));
    }
    return b;
}

template<class T>
Matrix<T,3,3> intrinsics_inverse(Matrix<T,3,3> K){
    assert(K(0,1)==0);
    assert(K(1,0)==0);
    assert(K(2,0)==0);
    assert(K(2,1)==0);
    assert(K(2,2)==1);
    return {1.0/K(0,0),0,-K(0,2)/K(0,0),
                0,1.0/K(1,1),-K(1,2)/K(1,1),
                0,0,1};
}









/// Vector<Row,T> alias
template<class T,unsigned int R> using Vector=Matrix<T,R,1>;

template<class T, unsigned int A, unsigned int B>
Vector<T,A+B> concat(Vector<T,A> a, Vector<T,B> b){
    Vector<T,A+B> ab;
    int n=0;
    for(int i=0;i<A;++i) ab(n++)=a(i);
    for(int i=0;i<B;++i) ab(n++)=b(i);
    return ab;
}

/// Vector<2,T> alias
template <typename T> using Vector2 = Matrix<T, 2,1>;
/// Vector<3,T> alias
template <typename T> using Vector3 = Matrix<T, 3,1>;
/// Vector<4,T> alias
template <typename T> using Vector4 = Matrix<T, 4,1>;
/// Vector<5,T> alias
template <typename T> using Vector5 = Matrix<T, 5,1>;
/// Vector<6,T> alias
template <typename T> using Vector6 = Matrix<T, 6,1>;
/// Matrix<2,2,T> alias
template <typename T> using Matrix2 = Matrix<T, 2,2>;
/// Matrix<3,3,T> alias
template <typename T> using Matrix3 = Matrix<T, 3,3>;
/// Matrix<3,4,T> alias
template <typename T> using Matrix34 = Matrix<T, 3,4>;
/// Matrix<4,4,T> alias
template <typename T> using Matrix4 = Matrix<T, 4,4>;


///Size specific functions
/**
* @brief cross cross product
* @param a
* @param b
* @return a x b
*/
template<class T>
mlib_host_device_
Vector3<T> cross(const Vector3<T>& a, const Vector3<T>& b){
    return a.cross(b);
}
template<class T>
mlib_host_device_
/**
 * @brief getLineFrom2Points
 * @param a
 * @param b
 * @return the line normalized 2d line in homogeneous coordinates
 */
Vector3<T> getLineFrom2Points(const Vector2<T>& a,
                              const Vector2<T>& b){
    Vector3<T> line=a.homogeneous().cross(b.homogeneous());
    line.lineNormalize();
    return line;
}
template<class T>
mlib_host_device_
/**
 * @brief getLineFrom2HomogeneousPoints
 * @param a
 * @param b
 * @return the line normalized 2d line in homogeneous coordinates
 */
Vector3<T> getLineFrom2Points(const Vector3<T>& a,
                              const Vector3<T>& b){
    Vector3<T> line=a.cross(b);
    line.lineNormalize();
    return line;
}





template<class T>
mlib_host_device_
/**
 * @brief get4x4 - returns a 4x4 matrix with the last column set to (t;1)
 * @param t
 * @return
 */
Matrix4<T> get4x4(const Vector3<T>& t){
    return Matrix4<T>(0,0,0,t(0),
                      0,0,0,t(1),
                      0,0,0,t(2),
                      0,0,0,T(1));
}

template<class T>
mlib_host_device_
/**
 * @brief get4x4 sets the top left block to the R matrix
 * @param R
 * @return
 */
Matrix4<T> get4x4(const Matrix3<T>& R){
    return Matrix4<T>(R(0,0),R(0,1),R(0,2),0,
                      R(1,0),R(1,1),R(1,2),0,
                      R(2,0),R(2,1),R(2,2),0,
                      0     ,     0,     0,1);
}
template<class T>
mlib_host_device_
/**
 * @brief get4x4 sets the top left block to the R matrix and the last column to (t,1)
 * @param R
 * @param t
 * @return
 */
Matrix4<T> get4x4(const Matrix3<T>& R,const Vector3<T>& t){
    return Matrix4<T>(R(0,0),R(0,1),R(0,2),t(0),
                      R(1,0),R(1,1),R(1,2),t(1),
                      R(2,0),R(2,1),R(2,2),t(2),
                      T(0)     ,     T(0),     T(0), T(1)   );
}
template<class T>
mlib_host_device_
/**
 * @brief get3x4 create a projection matrix from the rigid transform R,t
 * @param R
 * @param t
 * @return
 */
Matrix3<T> get3x4(const Matrix3<T>& R,const Vector3<T>& t){
    return Matrix3<T>(R(0,0),R(0,1),R(0,2),t(0),
                      R(1,0),R(1,1),R(1,2),t(1),
                      R(2,0),R(2,1),R(2,2),t(2));
}



template<class T, int R>
bool operator < (const Vector<T,R>& a, const Vector<T,R>& b){
    for(int r=0;r<R;++r){
        if(a[r]<b[r]) return true;
        if(a[r]>b[r]) return false;
    }
    return false;
}



template<class T1,class T2,class T3>
T1 cap(const T1&  val, const T2&  min, const T3&  max){
    assert(min<max);
    if(val<min)
        return min;
    if(val>max)
        return max;
    return val;
}


template<class T1,class T2, class T3>
bool is_in(const T1&  val, const T2&  min, const T3&  max){
    return (val<max && val>min);
}





/// convenient alias
using Vector2f = Matrix<float, 2, 1>;
/// convenient alias
using Vector3f = Matrix<float, 3, 1>;
/// convenient alias
using Vector4f = Matrix<float, 4, 1>;
/// convenient alias
using Vector5f = Matrix<float, 5, 1>;
/// convenient alias
using Vector6f = Matrix<float, 6, 1>;


//typedef double real;
/// convenient alias
using  Vector2d = Matrix<double, 2, 1>;
/// convenient alias
using  Vector3d = Matrix<double, 3, 1>;
/// convenient alias
using  Vector4d = Matrix<double, 4, 1>;
/// convenient alias
using Vector5d = Matrix<double, 5, 1>;
/// convenient alias
using Vector6d = Matrix<double, 6, 1>;
/// convenient alias
using Vector7d = Matrix<double, 7, 1>;
/// convenient alias
using Matrix2f = Matrix<float, 2, 2>;
/// convenient alias
using Matrix3f = Matrix<float, 3, 3>;

using Matrix34f = Matrix<float, 3, 4>;

using Matrix4f = Matrix<float, 4, 4>;

using  Matrix2d = Matrix<double, 2, 2>;

using  Matrix3d = Matrix<double, 3, 3>;

using  Matrix34d = Matrix<double, 3, 4>;

using  Matrix4d = Matrix<double, 4, 4>;


using  Vector2i = Matrix<double, 2, 1>;
using  Vector3i = Matrix<double, 3, 1>;

// this class should be trivial for doubles, since this significantly improves speed and ease of optimization
// very important
static_assert(std::is_trivially_destructible<Matrix<double,3,3>>(),"speed");
static_assert(std::is_trivially_copyable<Matrix<double,3,3>>(),"speed");
// the following constraints are good, but not critical
static_assert(std::is_trivially_default_constructible<Matrix<double,3,3>>(),"speed");
static_assert(std::is_trivially_copy_constructible<Matrix<double,3,3>>(),"speed");
static_assert(std::is_trivially_constructible<Matrix<double,3,3>>(),"speed");
static_assert(std::is_trivially_assignable<Matrix<double,3,3>,Matrix<double,3,3>>(),"speed");
static_assert(std::is_trivial<Matrix<double,3,3>>(),"speed");



// ideally read and write are identical, but they should also be informative:
// large matrices should be summarized but the Matrix is for small ones
// this is for visualization purposes only, different in binary ones
/**
     * @brief operator << human friendly matrix output
     * @param os
     * @param v
     * @return
     * todo
     * - binary stream specialisation
     */
    template<class T, unsigned int Rows, unsigned int Cols>
    std::ostream &operator <<(std::ostream &os, Matrix<T,Rows,Cols> v){

        if(Rows>1 && Cols==1){
            return os<<v.transpose()<<"^T";
        }
        os<<"[";
        for (unsigned int row = 0; row < Rows ; ++row){
            for (unsigned int col = 0; col < Cols; ++col){
                os<<v(row,col);
                if(col+1<Cols)
                    os<<", ";
            }
            if(row+1<Rows)        os<<"\n ";
        }
        os << "]";
        return os;
    }


    /**
     * @brief operator >> parses a buffer containing human readable matrixes
     * @param is
     * @param v
     * @return
     *
     * only for debugging purposes, use binary stream specialisation otherwize
     */
    template<class T, unsigned int Rows, unsigned int Cols>
    std::istream &operator >>(std::istream &is, Matrix<T,Rows,Cols>& v){
        char toss;
        if(!(is>>toss)) return is;
        for(unsigned int row = 0; row < Rows ; ++row){
            for (unsigned int col = 0; col < Cols; ++col){
                if(!(is>>v(row,col))) return is;
                if(!(is>>toss)) return is;
            }

        }
        is>>toss;
        return is;
    }


} // namespace cvl
