#pragma once
/* ********************************* FILE ************************************/
/** \file    workerpool.h
 *
 * \brief    This header contains a simple workpool
 *
 *
 * \remark
 * - c++11
 * - self contained(just .h, cpp)
 * - no dependencies
 * - os independent works in linux, windows etc are untested for some time
 * - std::async should generally be used instead, unless you have some specific reason to limit a pool
 *
 *
 * \author   Mikael Persson
 * \date     2017-01-01
 *
 *
 ******************************************************************************/


#include <thread>
#include <vector>
#include <atomic>
#include <mlib/utils/cvl/syncque.h>

namespace cvl {



class Job{
public:
    virtual void work() const = 0;
    virtual ~Job();

};


/**
 * @brief The WorkerPool class
 *
 * Beware of jobs that block a thread infinitely... the threads do a job untill its done...
 */
class WorkerPool{
public:
    static std::shared_ptr<WorkerPool> create(uint num_threads =  std::thread::hardware_concurrency());
    void add_job(std::shared_ptr<Job> job);
    WorkerPool(uint num_threads =  std::thread::hardware_concurrency());
    ~WorkerPool();
private:

    void work();

    cvl::SyncQue<std::shared_ptr<Job>> job_queue;
    std::vector<std::thread> threads;
    // must be atomic to prevent caching optimization mutithread errors.
    std::atomic<bool> stopped{false};
};

/**
 * @brief get_named_pool
 * @param thread_count
 * @return
 *
 * for those cases where a pregen pool is usefull,
 * name them "blocking_xxx" if the jobs can block, so you dont fuck up
 *
 * Might be good for a single unified pool for non blocking jobs. but if pregen they are fast anyways...
 */
std::shared_ptr<WorkerPool> get_named_workpool(std::string, uint thread_count=4);
} // end namespace cvl


