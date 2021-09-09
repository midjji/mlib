#if 0
#pragma once
#ifndef mlib_host_device_
#ifdef __CUDACC_VER_MAJOR__
#define mlib_host_device_ __host__ __device__
#else
#define mlib_host_device_
#endif
#endif

#include <memory>
#include <type_traits>
#include <assert.h>

#include <limits>
#include <cstdint>
#include <type_traits>
#include <assert.h>

#include <limits>
#include <cstdint>
#include <array>
#include <mlib/utils/cvl/matrix.h>
namespace cvl {





// so this works for both fixed stack arrays and pointers smart ptrs etc
// clone function? no problem, if storage has clone, it has clone
// view, semantic is clear, but it changes the underlying storage type
// stride is tricky. It requires something more
// slice is interesting, its a kind of view with different indexing


// Basically we consider four tensors
// fully variable:          minimum speed, needs storage for dim count then
// fixed dimensions:        probably best!
// fixed dims and stride    lower speed, fixed view, but compiletime dimensionchecking
// fixed dims ==stride      max speedst
// but the middle one can probably be ignored almost always, its for sub
// This is not a replacement for matrix today, it is a replacement for the old MatrixAdapter, designed to be used in the mlib cuda stuff
// having both stride and



// Storage is interesting,
// what do we do about alignment?
// Optimal Alignment can depend on the use i.e. the dimensions,
// answer, leave it to the user!

// its possible to avoid anyways, so no worries.
// access byte array cast to type ub? yes!
// Note, casting the pointer of any pod to any other pod and accessing it is UB in general!
// so using byte pointers is no good. Though it sure does seem to work correctly alot...
// thats the problem of the previous MatrixAdapter!
// so the real question becomes how do we handle alignment?
// the short answer is in storage, and the user must know what they are doing.

// Storage is contigous and supports indexing
// is it clearer to use non-virtual inheritance or separate classes?
// its probably clearer to use separate classes
// Storage is either a non-owning, i.e. a view,
// co-owning i.e a shared ptr, or owning i.e a unique ptr
// for cpu this corresponds to
//std::shared_ptr<int[size]> a(); // has element_type, and req for correct destructor
//std::unique_ptr<int[size]> a();

template<class element_type_,
         class index_type=std::uint64_t>
struct View {
    using element_type=element_type_;
    element_type* ptr=nullptr;
    index_type size=0;
    View()=default;
    explicit View(element_type* ptr,
                  index_type size):ptr(ptr),size(size){}
    // does index type matter here? possibly
    inline const element_type& operator[](std::uint32_t index) const {
        assert(index<size && "out of bounds");
        return ptr[index];}
    inline const element_type& operator[](std::uint64_t index) const {
        assert(index<size && "out of bounds");
        return ptr[index];}
    inline element_type& operator[](std::uint32_t index) {
        assert(index<size && "out of bounds");
        return ptr[index];}
    inline element_type& operator[](std::uint64_t index) {
        assert(index<size && "out of bounds");
        return ptr[index];}

    element_type* begin(){return ptr;}
    const element_type* begin() const{return ptr;}
    const element_type* cbegin() const{return ptr;}
    const element_type* end() const{return &ptr[size];}
    // add the rest here when I need them, is there a cost to std::bidirectional iterator? is it supported on cuda?
};
#if 0


template<class Storage,
 unsigned int... dimensions> // always atleast 1...
struct FixedTensor
{
    // ensures uint32 indexes computation is used if the number of elements is low enough to support it and uint64 otherwise
    using index_type=std::conditional_t<((std::uint64_t{1l} * ... * dimensions )<std::numeric_limits<std::uint32_t>::max()), std::uint32_t, std::uint64_t>;
    //might be possible to do this with decltype instead
    using element_type = typename Storage::element_type;

    Storage data;
    FixedTensor()=default;
    explicit FixedTensor(Storage data): data(data)
    {
        // check that data!=nullptr
        assert(data!=nullptr); // sometimes relevant, mostly not,
        // assert all dimensions are >0? dim count is selected automatically by some operations...
        //static_assert(sanity_check_dimensions()," no dimension may be zero");
        static_assert(elements()!=0,"");
    }
    static constexpr index_type elements(){
        return (dimensions * ...); // correct for 1 element too,
    }

    inline element_type& operator()(index_type index){return data[index];}
    inline const element_type& operator()(index_type index) const {return data[index];}

    template<class... Indexes>
    auto& operator()(Indexes... indexes)
    {
        // really cool to make this refer to subtensors when index count is too low
        // implicitly either refers or copies depending on storage?
        // that makes for a very strange kind of
        static_assert(sizeof...(dimensions)==sizeof...(Indexes),"not the right index count");


        index_type index=0;
        {
            // compiler will optimize this away completely!
            index_type ds[sizeof...(Indexes)]={dimensions...};
            index_type is[sizeof...(Indexes)]={index_type(indexes)...};
            for(uint i=0;i<sizeof...(dimensions);++i){            assert(is[i]<ds[i]);        }

            //std::cout<<0<<" "<<index<<std::endl;
            for(uint i=0;i<sizeof...(Indexes)-1;++i){
                uint d=ds[i+1];
                index=(is[i] +index)*d;
                //std::cout<<i<<" "<<d <<" "<<index<<std::endl;
            }
            index+=is[sizeof...(Indexes)-1];
        }
        //unsigned int index=0; // will always be case to unsigned int prior to application!
        //Indexer<Dimensions...>::comp(index,indexes...);

        return data[index];
    }
    auto clone(){
        return Tensor(data.clone());
    }
};

//static_assert(std::is_trivial<Tensor<View<int>,2,3,4>>(),"good c++ is trivial");

inline int test0(int* r){
    FixedTensor<View<int>,2,3,4> ten(View(r,2*3*4));
    return ten(1,2,3);
}

inline int test1(int a, int b, int c, int* r){
    FixedTensor<View<int>,2,3,4> ten(View(r,2*3*4));
    return ten(a,b,c);
}
inline int test2(int* r){
     FixedTensor<View<int>,2,3,4> ten(View(r,2*3*4));
    return ten.elements();
 }

inline void test_ten(){
    //
    int rr[30];for(int i=0;i<30;++i)rr[i]=i;
    Tensor<View<int>,2,3,4> ten(View(&rr[0],30));
    int index=0;
    for(int i=0;i<2;++i)
        for(int j=0;j<3;++j)
            for(int k=0;k<4;++k){

                assert(ten(i,j,k)==index);
                index++;
            }
}
/*
int main(){

    test_ten();
    return 0;
}
*/
#endif

// Stride is a must, but dimensions is very convenient,
// So a high perf version might need to be dims free, keeping only stride

template<class Storage,
         int Dimensions>
struct Tensor
{
    public:
    using index_type=std::uint32_t;
    Storage data;
    Vector<int, Dimensions> dimensions, strides;
    Tensor()=default;
    Tensor(Storage data,
           Vector<int, Dimensions> dimensions,
           Vector<int, Dimensions> strides):
        data(data),
      dimensions(dimensions), strides(strides)
    {
        static_assert(dimensions.size()==Dimensions);
    }

    template<class... Indexes>
    auto& operator()(Indexes... indexes)
    {
        // really cool to make this refer to subtensors when index count is too low
        // implicitly either refers or copies depending on storage?
        // that makes for a very strange kind of
        static_assert(Dimensions==sizeof...(Indexes),"not the right index count");
        index_type index=0;
        {
            index_type is[sizeof...(Indexes)]={index_type(indexes)...};            
            for(int i=0;i<int(Dimensions);++i){            assert(is[i]<uint(dimensions[i]));        }

            //std::cout<<0<<" "<<index<<std::endl;
            for(int i=0;i<int(sizeof...(Indexes)-1);++i){
                int d=dimensions[i+1];
                index=(is[i] +index)*d;
                //std::cout<<i<<" "<<d <<" "<<index<<std::endl;
            }
            index+=is[sizeof...(Indexes)-1];
        }
        //unsigned int index=0; // will always be case to unsigned int prior to application!
        //Indexer<Dimensions...>::comp(index,indexes...);

        return data[index];
    }
    auto clone(){
        return Tensor(data.clone(), dimensions, strides);
    }
};










template<class T, unsigned int dims>
class TensorAdapter{
public:
    T* data=nullptr;
    unsigned int dimensions[dims]; // not std array means cuda compat...

    template<class... S>
    mlib_host_device_
    explicit TensorAdapter(T* data, unsigned int first, S... args): data(data), dimensions {first, uint(args)...}{
        static_assert(sizeof...(S)+1==dims,"Must match dimension parameters with the number of dimensions");
    }

    /// does not delete its datapointer
    mlib_host_device_ ~TensorAdapter(){}

    template<class... S>
    mlib_host_device_
    bool in_tensor(unsigned int first, S... args){


        // wierd size here is required because static_if is missing...
        static_assert(sizeof...(S)+1==dims||sizeof...(S)==0,"Must match dimension parameters with the number of dimensions");
        unsigned int argvs[dims]={args...};
        for(unsigned int arg=0;arg<dims;++arg){
            if(dimensions[arg]<=argvs[arg]) return false;
        }
        return true;
    }

    // accessors

    /**
     * @brief operator ()
     * @param
     * @return the element at (x,y,z,...)
     */
    template<class... S>
    mlib_host_device_
    T& operator()(unsigned int first, S... args){

        static_assert(sizeof...(S)+1==dims || sizeof...(S)==0,"Must match dimension parameters with the number of dimensions");
        if(sizeof...(S)==0)
        { // known compile time, so the if will be opt away always...
            unsigned int index=first;
            assert(index<elements() && "out of bounds!");
            return data[index];
        }
        else
        {
            assert(in_tensor(first, args...) && "out of bounds!");
            uint argvs[dims]={first, uint(args)...};
            uint index=argvs[0];
            for(unsigned int i=1;i<dims;++i){
                index*=dimensions[i];
                index+=argvs[i];
            }
            return data[index];
        }
    }

    mlib_host_device_
    unsigned int elements(){
        unsigned int elems=dimensions[0];
        for(unsigned int arg=1;arg<dims;++arg){
            elems*=dimensions[arg];
        }
        return elems;
    }

    ///@return a pointer to the first element
    mlib_host_device_
    const T* begin() const{return &data[0];}
    ///@return a pointer to the last element +1
    mlib_host_device_
    const T* end() const{return &data[elements()];}

    T* begin() {return &data[0];}
    ///@return a pointer to the last element 01
    mlib_host_device_
    T* end() {return &data[elements()];}


    void set_all(T val){
        for(T& v :*this)
            v=val;

    }
    bool equals(TensorAdapter<T,dims> t){
        for(unsigned int i=0;i<elements();++i){
            if(t(i)!=data[i]) return false;
        }
        return true;
    }
};



template<class T, int dims>
class TensorWrapper
{

public:

    TensorAdapter<T,dims> tensor;
    std::weak_ptr<TensorWrapper<T,dims>> self;
    TensorWrapper(TensorAdapter<T,dims> tensor):tensor(tensor){}
    ~TensorWrapper()
    {

        if(devicePointer(tensor.data)){
            cudaFree(tensor.data);
            tensor.data=nullptr;
        }
        else
            delete tensor.data;
    }
    /*
    // to cpu
    std::shared_ptr<TensorWrapper<T, dims>> cpu(){
        if(devicePointer(tensor.data)){
            // will copy from cpu to device,
            T* data=nullptr;
            copy<T>(tensor.data, data, tensor.elements());
            return std::make_shared<TensorWrapper<T,dims>>(TensorAdapter<T,dims>(data, tensor.dimensions));
        }
        else
            return self.lock();
    }
    // to gpu
    std::shared_ptr<TensorWrapper<T, dims>> device(){
        if(!devicePointer(tensor.data)){
            // will copy from cpu to device,
            T* data=nullptr;

            copy<T>(tensor.data, data, tensor.elements());
            return  std::make_shared<TensorWrapper<T,dims>>(TensorAdapter<T,dims>(data, tensor.dimensions));
        }
        else
            return self.lock();
    }
    template<class... S>
    static std::shared_ptr<TensorWrapper<T,dims>> device_allocate(S... args){
        TensorAdapter<T,dims> ten(nullptr, args...);
        T* data=cudaNew<T>(ten.elements());
        auto tmp=std::make_shared<TensorWrapper<T,dims>>(TensorAdapter<T,dims>(data, args...));
        tmp->self=tmp;
        return tmp;
    }
*/


    template<class... S>
    static std::shared_ptr<TensorWrapper<T,dims>> allocate(S... args){
        TensorAdapter<T,dims> ten(nullptr, args...);
        T* data = new T[ten.elements()];
        std::cout<<"ten.elements()"<<ten.elements()<<std::endl;
        auto tmp=std::make_shared<TensorWrapper<T,dims>>(TensorAdapter<T,dims>(data, args...));
        tmp->self=tmp;
        return tmp;
    }




};

}// end namespace cvl

#endif
