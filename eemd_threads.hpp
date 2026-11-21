/**
 * Threads for the EEMD/ICEEMDAN ensembles: a small fork-join pool built on
 * the C++ standard library. No OpenMP, no TBB, nothing to install.
 *
 *   - Persistent workers: started the first time a decomposition needs them,
 *     kept for the life of the process (or until shutdown_threads()).
 *   - The calling thread is slot 0 and runs its share inline; workers are
 *     slots 1, 2, ...
 *   - A worker that took part in the last fork spins for the block time
 *     (1 ms by default), so the forks that follow one another inside a
 *     decomposition find it awake, then sleeps until the next fork. A worker
 *     the last fork did not use sleeps at once, and no fork waits for it.
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

                    bool wake = false;
                    {
                        std::lock_guard<std::mutex> lock(m_);
                        fn_ = fn;
                        ctx_ = ctx;
                        pending_.store(team - 1, std::memory_order_relaxed);
                        signal_.store(next_signal(team), std::memory_order_release);
                        wake = sleepers_ > 0;
                    }
                    if (wake)
                        cv_work_.notify_all();

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
                        cv_done_.wait(lock, [this]
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
                    }
                    cv_work_.notify_all();
                    for (std::thread &worker : workers_)
                        worker.join();
                    workers_.clear();
                    stop_.store(false, std::memory_order_release);
                }

            private:
                using Clock = std::chrono::steady_clock;

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
                        try
                        {
                            workers_.emplace_back([this, slot, seen]
                                                  { worker_main(slot, seen); });
                        }
                        catch (const std::system_error &)
                        {
                            break;
                        }
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
                 * block time when `spin` is set, then sleeps.
                 */
                uint64_t wait_signal(uint64_t seen, bool spin)
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
                    ++sleepers_;
                    cv_work_.wait(lock, [this, seen]
                                  { return signal_.load(std::memory_order_acquire) != seen; });
                    --sleepers_;
                    return signal_.load(std::memory_order_acquire);
                }

                void worker_main(int slot, uint64_t seen)
                {
                    inside_team() = true; // a body that forks again runs its slots serially
                    bool in_team = true;  // false: the last fork did not use this worker
                    for (;;)
                    {
                        seen = wait_signal(seen, in_team);
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
                            cv_done_.notify_one();
                        }
                    }
                }

                std::mutex turn_; // one fork-join at a time; also guards workers_
                std::vector<std::thread> workers_;

                std::mutex m_; // the fork's fields, sleepers_, error_
                std::condition_variable cv_work_;
                std::condition_variable cv_done_;
                Fn fn_ = nullptr;
                void *ctx_ = nullptr;
                int sleepers_ = 0;
                std::exception_ptr error_;

                std::atomic<uint64_t> signal_{0}; // moves once per fork
                std::atomic<int> pending_{0};     // the team's workers that have not finished the fork
                std::atomic<bool> stop_{false};
                std::atomic<int> setting_{1};
                std::atomic<double> block_ms_{1.0};
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
     * Sets it. 0: sleep at once. Negative: the workers a decomposition uses
     * never sleep (lowest latency, one busy core each).
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
