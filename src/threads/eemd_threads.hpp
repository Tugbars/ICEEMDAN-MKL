/**
 * Threads for the EEMD/ICEEMDAN ensembles: a small fork-join pool built on
 * the C++ standard library. No OpenMP, no TBB, nothing to install.
 *
 *   - Persistent workers: started the first time a decomposition needs them,
 *     kept for the life of the process (or until shutdown_threads()).
 *   - The calling thread is slot 0 and runs its share inline; workers are
 *     slots 1, 2, ...
 *   - A fork wakes the workers of its team and no others, and waits for
 *     those alone.
 *   - A worker sleeps as soon as its share of a fork is done (block time 0,
 *     the default): it uses no processor between forks, and the system
 *     places it on a free core when it wakes. With a block time it spins
 *     that long first, which answers the next fork sooner but keeps a
 *     processor busy; on a machine whose every hyperthread is in use that
 *     was measured slower, not faster.
 *   - One fork-join at a time: two application threads that decompose at
 *     once take turns on the workers.
 *   - A body that forks again runs its slots one after another on its own
 *     thread (no nested teams).
 *
 * The thread count is a process setting: set_num_threads(), or the
 * environment variable EEMD_NUM_THREADS, read once; by default every logical
 * processor. The decompositions do not depend on it (see the README,
 * "Reproducibility").
 *
 * License: MIT
 */

#ifndef EEMD_THREADS_HPP
#define EEMD_THREADS_HPP

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <memory>
#include <mutex>
#include <system_error>
#include <thread>
#include <type_traits>
#include <vector>

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#include <immintrin.h>
#define EEMD_CPU_PAUSE() _mm_pause()
#else
#define EEMD_CPU_PAUSE() std::this_thread::yield()
#endif

#if (defined(__unix__) || defined(__APPLE__)) && !defined(_WIN32)
#include <pthread.h> // pthread_atfork
#define EEMD_THREADS_HAVE_ATFORK 1
#endif

namespace eemd
{
    namespace threads
    {
        namespace detail
        {
            class Pool
            {
            public:
                using Fn = void (*)(void *ctx, int slot);

                /**
                 * The process's pool. Never destroyed: its workers run until
                 * the process ends, after the static destructors.
                 */
                static Pool &instance() { return *current(); }

                /** True on a thread that is running a slot of a fork. */
                static bool &inside_team()
                {
                    thread_local bool inside = false;
                    return inside;
                }

                int setting() const { return setting_.load(std::memory_order_relaxed); }
                void set_setting(int n)
                {
                    setting_.store(n >= 1 ? n : hardware_threads(), std::memory_order_relaxed);
                }

                double block_ms() const { return block_ms_.load(std::memory_order_relaxed); }
                void set_block_ms(double ms) { block_ms_.store(ms, std::memory_order_relaxed); }

                /**
                 * Calls fn(ctx, slot) for every slot in [0, team): slot 0 on
                 * the calling thread, the others on workers. Returns when all
                 * have returned; rethrows the first exception a slot threw.
                 */
                void run(int team, Fn fn, void *ctx)
                {
                    if (team <= 1)
                    {
                        fn(ctx, 0);
                        return;
                    }
                    if (inside_team() || team > max_team)
                    {
                        run_serially(team, fn, ctx);
                        return;
                    }

                    std::lock_guard<std::mutex> turn(turn_);
                    grow(team - 1);
                    if (static_cast<int>(workers_.size()) < team - 1)
                    {
                        run_serially(team, fn, ctx); // the system refused a thread
                        return;
                    }

                    {
                        std::lock_guard<std::mutex> lock(m_);
                        fn_ = fn;
                        ctx_ = ctx;
                        pending_.store(team - 1, std::memory_order_relaxed);
                        signal_.store(next_signal(team), std::memory_order_release);
                        for (int i = 0; i < team - 1; ++i)
                        {
                            if (workers_[i]->sleeping)
                                workers_[i]->wake.notify_one();
                        }
                    }

                    inside_team() = true;
                    try
                    {
                        fn(ctx, 0);
                    }
                    catch (...)
                    {
                        record_error();
                    }
                    inside_team() = false;

                    // The team's workers hold references into the caller's
                    // frame: wait for every one of them, whatever slot 0 did.
                    for (int i = 0; i < 4096 && pending_.load(std::memory_order_acquire) != 0; ++i)
                        EEMD_CPU_PAUSE();
                    std::exception_ptr error;
                    {
                        std::unique_lock<std::mutex> lock(m_);
                        done_.wait(lock, [this]
                                   { return pending_.load(std::memory_order_acquire) == 0; });
                        error = error_;
                        error_ = nullptr;
                    }
                    if (error)
                        std::rethrow_exception(error);
                }

                /**
                 * Stops and joins the workers. The next fork starts new ones.
                 * For a host that unloads this library's code (a DLL) while
                 * the process goes on; not needed at process exit.
                 */
                void shutdown()
                {
                    std::lock_guard<std::mutex> turn(turn_);
                    {
                        std::lock_guard<std::mutex> lock(m_);
                        stop_.store(true, std::memory_order_release);
                        signal_.store(next_signal(0), std::memory_order_release);
                        for (const std::unique_ptr<Worker> &worker : workers_)
                            worker->wake.notify_one();
                    }
                    for (const std::unique_ptr<Worker> &worker : workers_)
                        worker->thread.join();
                    workers_.clear();
                    stop_.store(false, std::memory_order_release);
                }

            private:
                using Clock = std::chrono::steady_clock;

                struct Worker
                {
                    std::thread thread;
                    std::condition_variable wake;
                    bool sleeping = false; // under m_
                };

                /**
                 * The fork signal: a counter in the high bits, the team size
                 * in the low 16. A worker reads the fork's fields only when
                 * its slot is in the team, and the fork then waits for it.
                 */
                static constexpr int team_bits = 16;
                static constexpr int max_team = (1 << team_bits) - 1;

                Pool()
                {
                    int n = 0;
                    if (const char *env = std::getenv("EEMD_NUM_THREADS"))
                        n = std::atoi(env);
                    set_setting(n);
                }

                static int hardware_threads()
                {
                    const unsigned n = std::thread::hardware_concurrency();
                    return n > 0 ? static_cast<int>(n) : 1;
                }

                static Pool *&current()
                {
                    static Pool *pool = create();
                    return pool;
                }

                static Pool *create()
                {
#ifdef EEMD_THREADS_HAVE_ATFORK
                    pthread_atfork(nullptr, nullptr, &Pool::after_fork_in_child);
#endif
                    return new Pool();
                }

#ifdef EEMD_THREADS_HAVE_ATFORK
                /**
                 * A forked child has this thread only: the workers, and
                 * whatever they held, stayed in the parent. Give the child a
                 * new pool with the parent's settings; the old one is left
                 * as it is.
                 */
                static void after_fork_in_child()
                {
                    Pool *old = current();
                    Pool *fresh = new Pool();
                    fresh->setting_.store(old->setting(), std::memory_order_relaxed);
                    fresh->block_ms_.store(old->block_ms(), std::memory_order_relaxed);
                    current() = fresh;
                    inside_team() = false;
                }
#endif

                static void run_serially(int team, Fn fn, void *ctx)
                {
                    for (int slot = 0; slot < team; ++slot)
                        fn(ctx, slot);
                }

                static int team_of(uint64_t signal) { return static_cast<int>(signal & max_team); }

                uint64_t next_signal(int team) const
                {
                    const uint64_t count = signal_.load(std::memory_order_relaxed) >> team_bits;
                    return ((count + 1) << team_bits) | static_cast<uint64_t>(team);
                }

                void grow(int n_workers)
                {
                    while (static_cast<int>(workers_.size()) < n_workers)
                    {
                        const int slot = static_cast<int>(workers_.size()) + 1;
                        const uint64_t seen = signal_.load(std::memory_order_relaxed);
                        std::unique_ptr<Worker> worker(new Worker());
                        Worker *self = worker.get();
                        try
                        {
                            worker->thread = std::thread([this, self, slot, seen]
                                                         { worker_main(*self, slot, seen); });
                        }
                        catch (const std::system_error &)
                        {
                            break;
                        }
                        workers_.push_back(std::move(worker));
                    }
                }

                void record_error()
                {
                    std::lock_guard<std::mutex> lock(m_);
                    if (!error_)
                        error_ = std::current_exception();
                }

                /**
                 * Blocks until the signal moves on from `seen`: spins for the
                 * block time when `spin` is set, then sleeps. A sleeping
                 * worker is woken by the forks that have it in their team;
                 * the signal it then reads may be several forks on.
                 */
                uint64_t wait_signal(Worker &self, uint64_t seen, bool spin)
                {
                    const double block = block_ms(); // < 0: never sleep
                    if (spin && block != 0.0)
                    {
                        const Clock::time_point t0 = Clock::now();
                        for (unsigned i = 1;; ++i)
                        {
                            const uint64_t s = signal_.load(std::memory_order_acquire);
                            if (s != seen)
                                return s;
                            EEMD_CPU_PAUSE();
                            if ((i & 1023u) == 0 && block > 0.0 &&
                                std::chrono::duration<double, std::milli>(Clock::now() - t0).count() >= block)
                                break;
                        }
                    }
                    std::unique_lock<std::mutex> lock(m_);
                    self.sleeping = true;
                    self.wake.wait(lock, [this, seen]
                                   { return signal_.load(std::memory_order_acquire) != seen; });
                    self.sleeping = false;
                    return signal_.load(std::memory_order_acquire);
                }

                void worker_main(Worker &self, int slot, uint64_t seen)
                {
                    inside_team() = true; // a body that forks again runs its slots serially
                    bool in_team = true;  // false: the last fork did not use this worker
                    for (;;)
                    {
                        seen = wait_signal(self, seen, in_team);
                        if (stop_.load(std::memory_order_acquire))
                            return;
                        in_team = slot < team_of(seen);
                        if (!in_team)
                            continue;
                        // fn_ and ctx_ were written before the signal moved, and
                        // stay as they are until pending_ reaches 0
                        try
                        {
                            fn_(ctx_, slot);
                        }
                        catch (...)
                        {
                            record_error();
                        }
                        if (pending_.fetch_sub(1, std::memory_order_acq_rel) == 1)
                        {
                            std::lock_guard<std::mutex> lock(m_);
                            done_.notify_one();
                        }
                    }
                }

                std::mutex turn_; // one fork-join at a time; also guards workers_
                std::vector<std::unique_ptr<Worker>> workers_;

                std::mutex m_; // the fork's fields, the workers' sleeping flags, error_
                std::condition_variable done_;
                Fn fn_ = nullptr;
                void *ctx_ = nullptr;
                std::exception_ptr error_;

                std::atomic<uint64_t> signal_{0}; // moves once per fork
                std::atomic<int> pending_{0};     // the team's workers that have not finished the fork
                std::atomic<bool> stop_{false};
                std::atomic<int> setting_{1};
                std::atomic<double> block_ms_{0.0};
            };
        } // namespace detail

        /**
         * The number of slots a fork from this thread uses: the thread
         * setting, or 1 inside a fork.
         */
        inline int team_size()
        {
            return detail::Pool::inside_team() ? 1 : detail::Pool::instance().setting();
        }

        /** Calls body(slot) for every slot in [0, team), slot 0 on the calling thread. */
        template <class Body>
        void run(int team, Body &&body)
        {
            using B = typename std::remove_reference<Body>::type;
            detail::Pool::instance().run(
                team,
                [](void *ctx, int slot)
                { (*static_cast<B *>(ctx))(slot); },
                const_cast<void *>(static_cast<const void *>(&body)));
        }

        /**
         * Splits [0, count) into `team` consecutive ranges, sizes differing
         * by one at most and the longer ones first, and calls
         * body(begin, end, slot) once on every slot, also when its range is
         * empty. The split depends on count and team only.
         */
        template <class Body>
        void for_static(int team, int32_t count, Body &&body)
        {
            if (team < 1)
                team = 1;
            const int32_t q = count / team;
            const int32_t r = count % team;
            run(team, [&](int slot)
                {
                    const int32_t begin = slot * q + (slot < r ? slot : r);
                    const int32_t end = begin + q + (slot < r ? 1 : 0);
                    body(begin, end, slot); });
        }

        /**
         * Calls body(index, slot) for every index in [0, count), each index
         * once, handing the next index to whichever slot is free.
         */
        template <class Body>
        void for_dynamic(int team, int32_t count, Body &&body)
        {
            std::atomic<int32_t> next{0};
            run(team, [&](int slot)
                {
                    for (;;)
                    {
                        const int32_t index = next.fetch_add(1, std::memory_order_relaxed);
                        if (index >= count)
                            break;
                        body(index, slot);
                    } });
        }
    } // namespace threads

    /** The number of threads a decomposition uses (the calling thread included). */
    inline int get_num_threads()
    {
        return threads::detail::Pool::instance().setting();
    }

    /** Sets it, for the whole process. n < 1: every logical processor. */
    inline void set_num_threads(int n)
    {
        threads::detail::Pool::instance().set_setting(n);
    }

    /** How long a worker spins after a fork before it sleeps, in milliseconds. */
    inline double get_thread_block_ms()
    {
        return threads::detail::Pool::instance().block_ms();
    }

    /**
     * Sets it. 0 (the default): sleep at once. Negative: the workers a
     * decomposition uses never sleep (one busy processor each).
     */
    inline void set_thread_block_ms(double ms)
    {
        threads::detail::Pool::instance().set_block_ms(ms);
    }

    /**
     * Stops the worker threads; the next decomposition starts them again.
     * Call it before unloading a DLL that contains this library.
     */
    inline void shutdown_threads()
    {
        threads::detail::Pool::instance().shutdown();
    }

} // namespace eemd

#endif // EEMD_THREADS_HPP
