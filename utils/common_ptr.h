#pragma once
#include <cstddef>


namespace cvl{



template<class T>
/**
 * @brief The common_ptr struct
 * a not thread safe smart pointer, this means it is way faster in non thread safe contexts.
 *
 *
 * There are lots of features, but I need very few...
 */
class common_ptr{

    mutable T* t;
    mutable int* rc;
public:

    common_ptr() noexcept {        rc=nullptr;        t=nullptr;    }
    explicit common_ptr(T* in) noexcept
    {        t=in;        if(in!=nullptr){            rc= new int;            (*rc)=1;        }    }
    common_ptr(const common_ptr& cp ) noexcept
    {        t=cp.t;        rc=cp.rc;        (*rc)++;    }
    ~common_ptr() noexcept
    {        if(t==nullptr) return;        (*rc)--;        if((*rc)<1){            delete t;            delete rc;        }    }

    common_ptr &  operator= (const common_ptr& a) noexcept{
        if(t==a.t) return *this;
        rc--;
        if(rc==0) {
            delete rc;
            delete t;
        }
        rc=a.rc;
        (*rc)++;
        t=a.t;
        return *this;
    }
    //const common_ptr &  operator= ( const common_ptr & a) noexcept{ (*rc)++;	rc=a.rc; t=a.t;}

    T* get()      noexcept{return t;}
    const T* get()const noexcept{return t;}

    T& operator*()       noexcept {return *t;}
    T& operator*() const noexcept {return *t;}

    T* operator->()       noexcept {return t;}
    T* operator->() const noexcept {return t;}
    explicit operator bool() const { return t!=nullptr; }
};
template<class T>
bool operator==(const common_ptr<T>& c, std::nullptr_t null){
    return c.t==null;
}
template<class T>
bool operator==(std::nullptr_t null, const common_ptr<T>& c){
    return c.t==null;
}
template<class T, class... Args>
common_ptr<T> make_common(Args... args){
    return common_ptr<T> (new T(args...));
}
}
