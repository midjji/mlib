#pragma once
/* ********************************* FILE ************************************/
/** \file    MatrixAdapter.hpp
 *
 * \brief    This header contains the strided MatrixAdapter<T> and VectorAdapter non managed matrix,vector classes.
 *
 * See:
 * - MemManager, DevMemManager or use a existing const buffer like a cv::Mat or std::vector.
 *
 *
 * \remark
 * - c++11
 * - no dependencies
 * - header only
 * - cuda enabled
 *
 * \todo
 * - how to deal with the mlib_host_device_ defines properly
 *
 *
 *
 * \author   Mikael Persson
 * \date     2015-04-01
 * \note MIT licence
 *
 ******************************************************************************/
#include <assert.h>
#include <mlib/utils/informative_asserts.h>
#include <cstdint>
#include <memory>
#include <sstream>
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

/// the standard uint typedef
typedef unsigned int uint;
#include <mlib/utils/cvl/matrix.h>

namespace cvl{

/**
 * @brief The MatrixAdapter class
 * A Rows*Cols non managed data Matrix
 */
template<class T>
class  MatrixAdapter{
public:

    mlib_host_device_
    MatrixAdapter(){
        data=nullptr;
        rows=0;
        cols=0;
        stride=0;
    }
    mlib_host_device_
    /**
     * @brief MatrixAdapter
     * @param data pointer to data, Adapter does not take ownership!
     * @param rows
     * @param cols == stride
     */
    explicit MatrixAdapter(T* data, uint  rows, uint  cols){
        assert(data!=nullptr);
        this->data=(std::uint8_t*)data;
        this->cols=cols;
        this->rows=rows;
        this->stride=cols*sizeof(T);
        assert(stride>=cols);
    }
    mlib_host_device_
    /**
     * @brief MatrixAdapter
     * @param data pointer to data, Adapter does not take ownership!
     * @param rows
     * @param cols
     * @param stride
     */
    explicit MatrixAdapter(T* data, uint  rows, uint  cols, uint stride){
        assert(data!=nullptr);
        this->data=(std::uint8_t* )data;
        this->cols=cols;
        this->rows=rows;
        this->stride=stride;
        //informative_assert(stride>=cols*sizeof(T));
        assert(stride>=cols*sizeof(T));
    }


    mlib_host_device_
    /**
     * @brief MatrixAdapter construct submatrix from matrix m
     * @param m
     * @param row   start row
     * @param col   start col
     * @param rows  total rows
     * @param cols  total cols
     */
    MatrixAdapter(MatrixAdapter<T>& m, uint row, uint col, uint rows, uint cols){
        assert(row+rows<=this->rows);
        assert(col+cols<=this->cols);
        data=atref(row,col);
        this->cols=cols;
        this->rows=rows;
        stride=m.stride;
    }
    /// does not delete its datapointer
    mlib_host_device_ ~MatrixAdapter(){}

    // accessors

    mlib_host_device_
    /**
     * @brief operator ()
     * @param row
     * @param col
     * @return the element at (row,col)
     */
    T& operator()( uint row, uint col){
        assert(col<cols);
        assert(row<rows);
        T*  addr=(T*)(&data[row*stride +col*sizeof(T) ]);
        return addr[0];
    }

    /**
     * @brief operator ()
     * @param row
     * @param col
     * @return the element at (row,col)
     */
    mlib_host_device_
    const T& operator()( uint row, uint col ) const    {
        assert(col<cols);
        assert(row<rows);
        T*  addr=(T*)(&data[row*stride +col*sizeof(T) ]);
        return addr[0];

    }

    mlib_host_device_
    /**
     * @brief operator ()
     * @param row
     * @param col
     * @return the indexeth element, i.e. compensating for stride!
     */
    T& operator()( const uint& index){
        assert(index<cols*rows);
        // compute row
        int row=index/cols;
        int col= index-row*cols;
        T*  addr=(T*)(&data[row*stride +col*sizeof(T) ]);
        return addr[0];
    }







    /**
     * @brief at
     * @param row
     * @param col
     * @return the element at (row,col)
     */
    mlib_host_device_
    T& at(uint row, uint col){
        return this->operator ()(row,col);
    }
    /**
     * @brief atref
     * @param row
     * @param col
     * @return pointer to specified element
     */
    mlib_host_device_
    T* atref(uint row, uint col) const{
        assert(col<cols);
        assert(row<rows);
        return (T*)(&(this->operator ()(row,col)));
    }

    mlib_host_device_
    /**
     * @brief begin
     * @return  pointer to the beginning of the Adapter, note only used begin, end if the matrix is continious
     */
    T* begin(){
        assert(cols*sizeof(T)==stride);
        return &data[0];
    }
    mlib_host_device_
    /**
     * @brief end
     * @return ptr to end of matrix, see begin
     */
    T* end(){
        assert(cols*sizeof(T)==stride);
        return &data[rows*stride*sizeof(T)];
    }
    ///@return get memory which spans the matrix ie rows*stride !
    mlib_host_device_
    uint size(){return rows*stride;}


    /**
     * @brief getSubMatrix
     * @param row
     * @param col
     * @return
     */
    mlib_host_device_
    MatrixAdapter<T> getSubMatrix(uint row, uint col) const{
        assert(col<cols);        assert(row<rows);
        MatrixAdapter<T> m(atref(row,col),rows-row,cols-col,stride);
        return m;
    }
    /**
     * @brief getSubMatrix returns a submatrix pointing to elements in this matrix
     * @param row
     * @param col
     * @param rows
     * @param cols
     * @return
     */
    mlib_host_device_
    MatrixAdapter<T> getSubMatrix(uint row, uint col, uint rows_, uint cols_) const{

        assert(row+rows_<=rows);
        assert(col+cols_<=cols);
        MatrixAdapter<T> m(atref(row,col),rows_,cols_,stride);

        return m;
    }


    /**
     * @brief row returns a submatrix spanning one row of the matrix
     * @param row
     * @return
     */
    mlib_host_device_
    MatrixAdapter<T> row(uint row) {
        assert(row<rows);
        MatrixAdapter<T> m(atref(row,0),1,cols,stride);
        return m;
    }
    /**
     * @brief col returns a submatrix spanning one col of the matrix
     * @param col
     * @return
     */

    mlib_host_device_
    MatrixAdapter<T> col(uint col) {
        assert(col<cols);
        MatrixAdapter<T> m(atref(0,col),rows,1,stride);
        return m;
    }
    /**
     * @brief clone, copy the data in the matrix, does still not take ownership, use with caution!
     * @return
     * \todo replace with memcopy
     */
    MatrixAdapter<T> clone(){
        MatrixAdapter<T> m=MatrixAdapter<T>::allocate(rows,cols,256);
        // replace with mem copy
        for(uint row=0;row<rows;++row)
            for(uint col=0;col<cols;++col)
                m(row,col)=at(row,col);
        return m;
    }
    Vector2i dimensions(){
        return Vector2i(rows,cols);
    }


    /**
     * @brief convert create a copy of the matrix with a different type, does not take ownership of the data!
     * @return
     */
    template<class T1>
    MatrixAdapter<T1> convert(){
        MatrixAdapter<T1> m=MatrixAdapter<T1>::allocate(rows,cols,256);
        for(uint row=0;row<rows;++row)
            for(uint col=0;col<cols;++col)
                m(row,col)=(T1)at(row,col);
        return m;
    }
    /**
     * @brief convert create a copy of the matrix with a different type, does not take ownership of the data!
     * @param factor
     * @return the copy scaled with the factor supplied
     */
    template<class T1> MatrixAdapter<T1> convert(T1 factor){

        MatrixAdapter<T1> m=MatrixAdapter<T1>::allocate(rows,cols,256);

        for(uint row=0;row<rows;++row)
            for(uint col=0;col<cols;++col)
                m(row,col)=factor*((T1)at(row,col));
        return m;
    }

    mlib_host_device_
    ///@return if the matrix is continious
    bool isContinuous() const {return cols*sizeof(T)==stride;}
    // in bytes
    uint dataSize(){return rows*stride;}



    /// set all values in the adapter to @param val
    void setAll(const T& val){
        for(uint row=0;row<rows;++row)
            for(uint col=0;col<cols;++col)
                this->operator ()(row,col)=val;
    }
    /// the number of rows in the matrixAdapter
    uint rows;

    /// the number of columns in the maxtrixAdapter
    uint cols;
    /// the memory stride in bytes!
    uint stride;
    /// pointer to data not owned by the Adapter!
    //T* data=nullptr;
    // be very careful with this one to not mess up striding!
    T* getData(){return (T*)data;}
    void release(){delete data; data=nullptr;}



    static uint getStride( uint cols, uint stridestep=256){
        return (stridestep*((cols*sizeof(T)+stridestep-1)/stridestep));
    }
    /**
     * @brief allocate
     * @param rows
     * @param cols
     * @return a automatically strided matrix!
     * stride size defaults to 256 which might be a bit much on cpu
     * should be aligned too, but not sure what that will do
     * but you really shouldnt use this matrix class for small matrixes anyways
     * make sure this is the only one used anywhere!
     * \todo add alignas to the data array!
     */
    static MatrixAdapter<T> allocate(uint rows, uint cols, uint stridestep=256){
        uint stride=MatrixAdapter<T>::getStride(cols,stridestep);
        // what is the alignment of the first element here?
        // guaranteed atleast 4, but only matters on cuda, and there it is 256

        char* data=new char[stride*rows];
        return MatrixAdapter<T>((T*)data,rows,cols,stride);
    }
    // generic copying converter, requires mat.rows, mat.cols, (row,col)
    template<class Mat> static MatrixAdapter<T> allocate(const Mat& mat){
        uint rows=mat.rows;
        uint cols=mat.cols;
        MatrixAdapter<T> copy=MatrixAdapter<T>::allocate(rows,cols);
        for(uint row=0;row<rows;++row)
            for(uint col=0;col<cols;++col)
                copy(row,col)=mat(row,col);
        return copy;
    }
    bool is_in(uint row, uint col){
        return row<rows && col<cols;
    }
private:
    std::uint8_t* data=nullptr;
};

/**
 * @brief equal is every element equal and are the rows and cols equal
 * @param a
 * @param b
 * @return
 */
template<class T> bool equal(const MatrixAdapter<T>& a, const MatrixAdapter<T>& b ){
    if(a.rows!=b.rows) return false;
    if(a.cols!=b.cols) return false;
    for(uint r=0;r<a.rows;++r)
        for(uint c=0;c<a.cols;++c)
            if(!(a(r,c)==b(r,c))) return false;
    return true;
}

template<class T>
class MAWrapper
{
public:
    MatrixAdapter<T> ma;
    MAWrapper(int rows, int cols){

        ma=MatrixAdapter<T>(new T[rows*cols],rows,cols);
    }
    ~MAWrapper(){
        ma.release();
    }
};

template<class T> std::shared_ptr<MAWrapper<T>>
create_matrix(uint rows,uint cols){
    return std::make_shared<MAWrapper<T>>(rows, cols);
}




/**
 * @brief MatrixAdapter2String
 * @param img
 * @return a string containing a human readable representation of the matrix
 */
template<class T>
std::string MatrixAdapter2String(const MatrixAdapter<T>& img){
    std::stringstream ss;

    for(uint r = 0;r < img.rows;++r){
        ss<<"row: "<<r<<" - ";
        for(uint c = 0;c < img.cols;++c)
            ss<<img(r,c)<<", ";
        ss<<"\n";
    }
    ss<<img.rows<< " "<< img.cols<<" "<<img.stride<<"\n";
    return ss.str();
}
template<class T>
/**
 * @brief operator << a human readable matrix description
 * @param os
 * @param im
 * @return
 */
std::ostream& operator<<(std::ostream& os, const MatrixAdapter<T>& im){
    os<<MatrixAdapter2String(im);
    return os;
}


template<class T>
/**
 * @brief print  exists to mirror a cuda in kernel function
 * @param img
 */
void print(MatrixAdapter<T>& img){
    std::cout<<MatrixAdapter2String(img)<<std::endl;
}







}// end namespace cvl
