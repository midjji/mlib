
#if 0
#include <mlib/utils/cvl/tensor.h>

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>



using namespace cvl;



template<class T, unsigned int Rows, unsigned int Cols>
using Matrix=Tensor<std::array<T,Rows*Cols>,Rows,Cols>;

template<class T, unsigned int Size>
using Vector=Tensor<std::array<T,Size>,Size>;

TEST_CASE("Constructors and indexing"){
    Matrix<int, 2,3> m(0,1,2,3,4,5);
    Matrix<int, 2,3> m2({0,1,2,3,4,5});

    for(int i=0;i<6;++i)
        CHECK(m._data[i]==i);
    for(int i=0;i<6;++i)
        CHECK(m2._data[i]==i);
}

TEST_CASE("Indexing"){
    Matrix<int, 2,3> m(0,1,2,3,4,5);
    for(int i=0;i<6;++i)
        CHECK(m._data[i]==i);
    for(int i=0;i<6;++i)
        CHECK(m(i)==i);
    for(int i=0;i<6;++i)
        CHECK(m[i]==i);
    for(int r=0;r<2;++r)
        for(int c=0;c<3;++c)
            CHECK(m(r,c)==r*3+c);
}

TEST_CASE("abs"){
    Matrix<int, 2,3> m(0,-1,2,3,4,-5);
    Matrix<int, 2,3> a=m.abs();
    for(int i=0;i<6;++i)
        CHECK(a(i)==std::abs(i));
}

TEST_CASE("sum"){
    Matrix<int, 2,3> m(0,-1,2,3,4,-5);
    CHECK(m.sum()==0+-1+2+3+4+-5);
}

TEST_CASE("norm"){
    Matrix<double, 2,3> m(0,-1,2,3,4,-5);
    double v=0;for(int i=0;i<6;++i) v+=m(i)*m(i);
    CHECK(m.norm()== doctest::Approx(std::sqrt(v)).epsilon(1e-15));
}
TEST_CASE("abs_sum"){
    Matrix<double, 2,3> m(1,-1,2,6,4,-7);
    CHECK(m.abs().sum()==m.absSum());
}
TEST_CASE("abs_max"){
    Matrix<double, 2,3> m1(1,-1,9,6,4,-7);
    CHECK(m1.absMax()==9);
    Matrix<double, 2,3> m2(1,-1,4,-6,4,-5);
    CHECK(m2.absMax()==6);
}
#endif
