#pragma once

#include <cstddef> // has std::nullptr_t


namespace cvl{



template<class T>
/**
 * @brief The common_ptr struct
 * a not thread safe smart pointer, this means it is way faster in non thread safe contexts.
 *
 *
 * There are lots of features, but I need very few...
 */
class common_ptr
{
    struct Data{
    T t;
    mutable int rc;
    template<class... Args> Data(Args... args):t(T(args...)),rc(0){}
    };
    Data* data;

    
public:

    common_ptr() noexcept: data(nullptr) {}
    // I need to have this to match shared_ptr, but then again, I never use it
    // when should it be used at all? Perhaps for custom allocators of T? No, that would be a template param to common ptr, besides I dont use them much
    // hmm, I think I can skip it and simplify things
    //explicit common_ptr(T* in) noexcept    {        t=in;        if(in!=nullptr){            rc= new int;            (*rc)=1;        }    }
    common_ptr(const common_ptr& cp ) noexcept
    {        data.t=cp.t;        data.rc=cp.rc;        (*data.rc)++;    }
    ~common_ptr() noexcept
    {        if(data==nullptr) return;        (*data.rc)--;        if((*data.rc)<1)
    {            delete data;        } 
       }

    common_ptr&  operator= (const common_ptr& a) noexcept
    {
        // account for self assign
        if(data==a.data) return *this;        

        data->rc--;
        if(data->rc==0)         {            delete data;        }
        data=a.data;
        data->rc++;
        return *this;
    }
    //const common_ptr &  operator= ( const common_ptr & a) noexcept{ (*rc)++;	rc=a.rc; t=a.t;}

    // hmm investigate const variants, 
    // replace by auto makes sense, but is less clear and messes with ides, 
    // test first
    T* get()      noexcept{return &data->t;}
    T* get()const noexcept{return &data->t;}

    T& operator*()       noexcept {return data->t;}
    T& operator*() const noexcept {return data->t;}

    T* operator->()       noexcept {return &data->t;}
    T* operator->() const noexcept {return &data->t;}
    explicit operator bool() const { return data!=nullptr; }

    
};
template<class T>
bool operator==(const common_ptr<T>& c, std::nullptr_t null){
    return c.data==null;
}
template<class T>
bool operator==(std::nullptr_t null, const common_ptr<T>& c){
    return c.data==null;
}
template<class T, class... Args>
common_ptr<T> make_common(Args... args){
    return common_ptr<T> (args...);
}
}
