#pragma once


#include <map>
#include <mutex>
#include <thread>
#include <iostream>
#include <condition_variable>
#include <exception>
#include "assert.h"

#include <iostream>
#include <atomic>
#include <limits>

typedef unsigned int uint;
// sync que
namespace cvl{




template <class Key, class Value,class Less>
class SyncMap{

private:
    std::map<Key, Value, Less> map;
    std::mutex mtx;
    std::condition_variable cv;
public:

    SyncMap& operator = (SyncMap&) = delete;
    std::shared_ptr<SyncMap<Key, Value, Less>> create(){
        return std::make_shared<SyncMap<Key, Value, Less>>();
    }

    void insert(const Key& key, const Value& value){
        std::unique_lock ul(mtx);
        map[key]=value;
        ul.unlock();
        cv.notify_one();
    }
    auto find(Key key){
        std::unique_lock ul(mtx);
        return map.find(key);

    }
    uint size(){
        std::unique_lock ul(mtx); // locks
        return uint(map.size());
    }

    void clear(){
        std::unique_lock ul(mtx); // locks
        map.clear();
        ul.unlock();
        cv.notify_all();
    }
};




}// end namespace cvl

