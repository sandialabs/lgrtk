#pragma once

#include <thread>
#include <functional>
#include <vector>
#include <mutex>
#include <queue>

namespace lgr {

class thread_pool {
 public:
  explicit thread_pool(int const n_threads)
  {
    assert(n_threads > 0);
    while (int(threads_.size()) < n_threads) {
      std::thread thread;
      try {
        thread = std::thread(&thread_pool::worker, this);
      } catch (...) {
        shutdown();
        throw;
      }
      try {
        threads_.push_back(std::move(thread));
      } catch (...) {
        shutdown();
        thread.join();
        throw;
      }
    }
  }
  // delete copy/move semantics
  thread_pool(const thread_pool&) = delete;
  thread_pool& operator=(const thread_pool&) = delete;
  thread_pool(thread_pool&&) = delete;
  thread_pool& operator=(thread_pool&&) = delete;
  ~thread_pool() {
    shutdown();
  }
  void push(const std::function<void()>& functor) {
    {
      std::lock_guard<std::mutex> lock{mutex_};
      functors_.push(functor);
    }
    cond_var_.notify_one();
  }
 private:
  void worker() {
    for (;;) {
      std::function<void()> functor;
      {
        std::unique_lock<std::mutex> lock{mutex_};
        cond_var_.wait(lock, [this]{
            return done_ || !functors_.empty();
            });
        if (done_ && functors_.empty()) {
          break;
        }
        functor = functors_.front();
        functors_.pop();
      }
      functor();
    }
  }
  void shutdown() {
    {
      std::lock_guard<std::mutex> lock{mutex_};
      done_ = true;
    }
    cond_var_.notify_all();
    for (std::thread& thread : threads_) {
      thread.join();
    }
    threads_.clear();
  }
  bool done_ = false;
  std::vector<std::thread> threads_;
  std::queue<std::function<void()>> functors_;
  std::condition_variable cond_var_;
  std::mutex mutex_;
};

}
