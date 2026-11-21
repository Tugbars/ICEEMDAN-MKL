/**
 * Thread pool tests (eemd_threads.hpp): the splits cover every index once,
 * many short forks with teams of changing size, sleeping and spinning
 * workers, a fork inside a fork, an exception in a worker, two application
 * threads forking at once, shutdown and restart, the thread setting and, on
 * POSIX systems, a fork() of the process.
 *
 * Exits non-zero on any failure.
 */

#include "eemd_threads.hpp"

#include <atomic>
#include <cstdio>
#include <numeric>
#include <stdexcept>
#include <thread>
#include <vector>

#if defined(__unix__) && !defined(_WIN32)
#include <sys/wait.h>
#include <unistd.h>
#define HAVE_FORK 1
#endif

static int g_failures = 0;

static void check(bool ok, const char *what)
{
    std::printf("  %-66s %s\n", what, ok ? "PASS" : "FAIL");
    if (!ok)
        ++g_failures;
}

// for_static: one consecutive range per slot, in slot order, the longer ones
// first, every index once. for_dynamic: every index once.
static bool covers(int team, int32_t count)
{
    std::vector<int> hits(count, 0);
    std::vector<int> calls(team, 0);
    std::vector<int32_t> begins(team), ends(team);
    eemd::threads::for_static(team, count, [&](int32_t b, int32_t e, int slot)
                              {
                                  ++calls[slot];
                                  begins[slot] = b;
                                  ends[slot] = e;
                                  for (int32_t i = b; i < e; ++i)
                                      ++hits[i]; });
    bool ok = ends[team - 1] == count;
    for (int s = 0; s < team; ++s)
    {
        ok = ok && calls[s] == 1 && begins[s] == (s == 0 ? 0 : ends[s - 1]);
        if (s > 0)
            ok = ok && (ends[s] - begins[s]) <= (ends[s - 1] - begins[s - 1]);
    }
    for (int h : hits)
        ok = ok && h == 1;

    std::vector<std::atomic<int>> dynamic_hits(count);
    for (auto &h : dynamic_hits)
        h.store(0);
    eemd::threads::for_dynamic(team, count, [&](int32_t i, int)
                               { dynamic_hits[i].fetch_add(1); });
    for (auto &h : dynamic_hits)
        ok = ok && h.load() == 1;
    return ok;
}

int main()
{
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    const int default_threads = eemd::get_num_threads();
    std::printf("Thread pool (%d threads by default)\n", default_threads);

    {
        bool ok = true;
        for (int team : {1, 2, 3, 4, 7, 8, 16, 33})
            for (int32_t count : {0, 1, 2, 5, 31, 100, 1000})
                ok = ok && covers(team, count);
        check(ok, "for_static / for_dynamic: every index once, ranges in slot order");
    }

    for (double block_ms : {0.0, 1.0, -1.0})
    {
        eemd::set_thread_block_ms(block_ms);
        bool ok = true;
        for (int rep = 0; rep < 5000; ++rep)
        {
            const int team = 1 + rep % 9;
            std::vector<long long> part(team, 0);
            eemd::threads::for_static(team, 257, [&](int32_t b, int32_t e, int slot)
                                      {
                                          for (int32_t i = b; i < e; ++i)
                                              part[slot] += i; });
            ok = ok && std::accumulate(part.begin(), part.end(), 0LL) == 257LL * 256 / 2;
        }
        char label[96];
        std::snprintf(label, sizeof(label), "5000 short forks, teams of 1 to 9, block time %g ms", block_ms);
        check(ok, label);
    }
    eemd::set_thread_block_ms(0.0);

    {
        std::atomic<int> inner{0};
        std::atomic<int> nested_teams{0};
        eemd::threads::run(4, [&](int)
                           {
                               if (eemd::threads::team_size() != 1)
                                   nested_teams.fetch_add(1);
                               eemd::threads::for_static(3, 30, [&](int32_t b, int32_t e, int)
                                                         { inner.fetch_add(e - b); }); });
        check(inner.load() == 4 * 30 && nested_teams.load() == 0,
              "a fork inside a fork runs on its own thread; team_size() is 1 there");
    }

    {
        bool caught = false;
        try
        {
            eemd::threads::run(6, [&](int slot)
                               {
                                   if (slot == 3)
                                       throw std::runtime_error("slot 3"); });
        }
        catch (const std::runtime_error &)
        {
            caught = true;
        }
        std::atomic<int> ran{0};
        eemd::threads::run(6, [&](int)
                           { ran.fetch_add(1); });
        check(caught && ran.load() == 6, "an exception in a worker reaches the caller; the pool goes on");
    }

    {
        std::atomic<long long> a{0}, b{0};
        std::thread other([&]
                          {
                              for (int rep = 0; rep < 1000; ++rep)
                                  eemd::threads::for_dynamic(5, 40, [&](int32_t i, int)
                                                             { b.fetch_add(i); }); });
        for (int rep = 0; rep < 1000; ++rep)
            eemd::threads::for_dynamic(7, 40, [&](int32_t i, int)
                                       { a.fetch_add(i); });
        other.join();
        check(a.load() == 1000LL * 780 && b.load() == 1000LL * 780, "two application threads forking at once");
    }

    {
        std::atomic<int> ran{0};
        eemd::shutdown_threads();
        eemd::threads::run(5, [&](int)
                           { ran.fetch_add(1); });
        eemd::shutdown_threads();
        eemd::shutdown_threads();
        eemd::threads::run(3, [&](int)
                           { ran.fetch_add(1); });
        check(ran.load() == 8, "shutdown_threads(), then a fork starts new workers");
    }

    {
        eemd::set_num_threads(3);
        const bool three = eemd::get_num_threads() == 3 && eemd::threads::team_size() == 3;
        eemd::set_num_threads(0);
        check(three && eemd::get_num_threads() == default_threads,
              "set_num_threads(3), then 0 for every logical processor");
    }

#ifdef HAVE_FORK
    {
        eemd::threads::run(8, [](int) {});
        const pid_t child = fork();
        if (child == 0)
        {
            std::atomic<int> ran{0};
            eemd::threads::for_dynamic(8, 1000, [&](int32_t, int)
                                       { ran.fetch_add(1); });
            _exit(ran.load() == 1000 ? 0 : 1);
        }
        int status = -1;
        waitpid(child, &status, 0);
        std::atomic<int> ran{0};
        eemd::threads::for_dynamic(8, 1000, [&](int32_t, int)
                                   { ran.fetch_add(1); });
        check(WIFEXITED(status) && WEXITSTATUS(status) == 0 && ran.load() == 1000,
              "fork(): the child and the parent both go on forking");
    }
#endif

    std::printf("\n%s (%d failure%s)\n", g_failures ? "FAILED" : "ALL PASSED", g_failures,
                g_failures == 1 ? "" : "s");
    return g_failures ? 1 : 0;
}
