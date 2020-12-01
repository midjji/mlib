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



class WorkerPool{
public:
    WorkerPool(){}

    // num_threads 0 means as many as are comfortably available to the cpu
    /**
     * @brief create
     * @param num_threads 0 means as many as are comfortably available to the cpu, probably use less than this if you use multiple pools
     * @param queue_size 0 means infinite size,
     * @return
     *
     * Jobs can block a thread infinitely... So make sure they are short, or that you count on this...
     * the workerpool  will drop jobs if the queue grows too large.
     */
    static std::shared_ptr<WorkerPool> create(uint num_threads = 0);
    void init(uint num_threads);

    void add_job(std::shared_ptr<Job> job);

    ~WorkerPool();


private:

    void work();

    std::shared_ptr<cvl::SyncQue<std::shared_ptr<Job>>> job_queue;
    std::vector<std::thread> threads;
    // must be atomic to prevent caching optimization mutithread errors.
    std::atomic<bool> stopped;
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


