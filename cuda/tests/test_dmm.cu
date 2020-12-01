/**
 * This file tests the DevMemManager
 */



#include <mlib/cuda/devmemmanager.h>
#include <mlib/cuda/common.cuh>
 
#include <mlib/utils/random.h>
#include <mlib/utils/memmanager.h>

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

using namespace mlib;
using namespace cvl;
using std::cout;using std::endl;





typedef unsigned char uchar;


template<class T> T getnum(uint i){ return T(i);}
template<> Vector3f getnum<Vector3f>(uint i){return Vector3f(i,i,i);}
template<> Vector3<uchar> getnum<Vector3<uchar>>(uint i){return Vector3<uchar>(i,i,i);}





template<class T>
bool upload_download_autostride(){

    DevMemManager dmm;

    // create a bunch of matrixes, up and download them and check.

    // fixed known stride
    // create a maxrix rows cols stride
    MemManager mm;
    uint rows,cols;
    rows=randui<int>(1,1000);
    cols=randui<int>(1,1000);



    MatrixAdapter<T> m;
    m= MatrixAdapter<T>::allocate(rows,cols);
    mm.manage(m);



    for(int r=0;r<rows;++r)
        for(int c=0;c<cols;++c)
            m(r,c)=getnum<T>(r*m.cols +c);


    MatrixAdapter<T> devm=dmm.upload(m);
    dmm.synchronize();
    MatrixAdapter<T> back=MatrixAdapter<T>::allocate(rows,cols);
    mm.manage(back);
    dmm.download(devm,back);
    dmm.synchronize();
    // did it succeed?

    //cout<<m<<endl;
    //printdev(devm);
    //cout<<back<<endl;
    return equal(m,back);


}




TEST_CASE("DevMemManager,UPLOAD_DOWNLOAD_AUTO_STRIDE"){
    CHECK(upload_download_autostride<int>());
    CHECK(upload_download_autostride<float>());
    CHECK(upload_download_autostride<double>());
    CHECK(upload_download_autostride<char>());
    CHECK(upload_download_autostride<unsigned char>());
    CHECK(upload_download_autostride<Vector3<unsigned char>>());
    CHECK(upload_download_autostride<Vector3<float>>());
}

template<class T> void checkwierd(){
    cvl::DevMemManager dmm;
    MatrixAdapter<float> gc=dmm.allocate<float>(1024,1024);



    MatrixAdapter<float> cgpu=dmm.download(gc);
    std::cout<<cgpu(0,0)<<std::endl;


}

TEST_CASE("CHECKWIERD_ONE,HUH"){
    checkwierd<float>();
    checkwierd<double>();
    checkwierd<uchar>();
}






template<class T>
bool setall(){

    DevMemManager dmm;

    // create a bunch of matrixes, up and download them and check.

    // fixed known stride
    // create a maxrix rows cols stride
    MemManager mm;
    uint rows,cols;
    rows=randui<int>(1,10);
    cols=randui<int>(1,1000);

    T val=randui<int>(-1000,1000);


    MatrixAdapter<T> m= MatrixAdapter<T>::allocate(rows,cols);
    mm.manage(m);
    for(int r=0;r<rows;++r)
        for(int c=0;c<cols;++c)
            m(r,c)=val;


    MatrixAdapter<T> devm=dmm.allocate<T>(rows,cols);
    dmm.synchronize();
    setAllDev(devm,dmm.pool.stream(0),val);
    MatrixAdapter<T> back=MatrixAdapter<T>::allocate(rows,cols);
    mm.manage(back);
    dmm.download(devm,back);
    dmm.synchronize();
    // did it succeed?

    //cout<<m<<endl;
    //printdev(devm);
    //cout<<back<<endl;
    return equal(m,back);
}

TEST_CASE("COMMON_KERNELS,SETALLDEV"){
    //This test ensures that the setalldev works which if the preceeding tests of up and down do is almost guaranteed
    CHECK(setall<int>());
    CHECK(setall<float>());
    CHECK(setall<double>());
    CHECK(setall<short>());

}














