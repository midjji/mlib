#include <iostream>
#include <map>
#include "mlib/utils/workerpool.h"


using std::cout;using std::endl;
namespace cvl {
Job::~Job(){}

std::shared_ptr<WorkerPool> WorkerPool::create(uint num_threads)
{
    auto wp=std::make_shared<WorkerPool>();
    wp->init(num_threads);
    return wp;
}

void WorkerPool::init(uint num_threads){
    // keeping this separate because it seems make shared is fond of copying.
    job_queue=std::make_shared<cvl::SyncQue<std::shared_ptr<Job>>>();
    stopped=false;
    if (num_threads == 0)
        num_threads = std::thread::hardware_concurrency();

    threads.reserve(num_threads);
    for (uint i=0; i<num_threads; ++i)
    {
        threads.push_back(std::thread([this] { this->work(); }));
    }
}

void WorkerPool::add_job(std::shared_ptr<Job> job)
{
    job_queue->push(job);
}

WorkerPool::~WorkerPool()
{
    stopped=true;
    job_queue->stop();
    for(std::thread& thr:threads)
        if(thr.joinable())
            thr.join();
}

void WorkerPool::work()
{
    while(!stopped)
    {
        // get the next image with with to do something...
        try {
            std::shared_ptr<Job> job=nullptr;
            if(!job_queue->blocking_try_pop(job))
                break;
            if(job==nullptr)
                break;
            job->work();
        }
        catch(...){
            std::cerr<<"Other error in workpool"<<std::endl;
            stopped=true;
        }
    }
    cout<<"thread pool thread shutdown"<<endl;
}
namespace workerpool_internal {
static std::mutex mtx;
static std::shared_ptr<std::map<std::string, std::shared_ptr<WorkerPool>>> pools=nullptr;
}

std::shared_ptr<WorkerPool> get_named_workpool(std::string name, uint thr_count){
    std::unique_lock<std::mutex> ul(workerpool_internal::mtx);
    if(workerpool_internal::pools==nullptr)
        workerpool_internal::pools = std::make_shared<std::map<std::string, std::shared_ptr<WorkerPool>>>();
    auto search = workerpool_internal::pools->find(name);
    if (search ==workerpool_internal::pools->end())
        (*workerpool_internal::pools)[name]=WorkerPool::create(thr_count);
    return (*workerpool_internal::pools)[name];
}

} // end namespace cvl
