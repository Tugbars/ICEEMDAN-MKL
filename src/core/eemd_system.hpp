/**
 * Two facts about the machine: its installed memory and its logical
 * processors. The defaults of the public interfaces come from them (the
 * noise a decomposer may keep, the threads it runs on).
 *
 * License: MIT
 */

#ifndef EEMD_SYSTEM_HPP
#define EEMD_SYSTEM_HPP

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <thread>

#if defined(_WIN32)
// GlobalMemoryStatusEx as <windows.h> declares it, declared here so that this
// header-only library does not bring <windows.h> and its macros into every
// program that includes it. A program that includes <windows.h> as well, before
// or after, sees the same declaration twice.
extern "C"
{
    struct _MEMORYSTATUSEX;
    __declspec(dllimport) int __stdcall GlobalMemoryStatusEx(struct _MEMORYSTATUSEX *);
}
#else
#include <unistd.h>
#endif

namespace eemd
{
    namespace machine
    {
        /** The memory installed in this machine, in bytes (4 GiB when the system does not say). */
        inline size_t installed_memory()
        {
#if defined(_WIN32)
            // MEMORYSTATUSEX: dwLength (4 bytes), dwMemoryLoad (4), ullTotalPhys (8), six more
            // 8-byte counters: 64 bytes in all
            alignas(8) unsigned char status[64] = {};
            const uint32_t length = sizeof(status);
            std::memcpy(status, &length, sizeof(length));
            if (GlobalMemoryStatusEx(reinterpret_cast<struct _MEMORYSTATUSEX *>(status)))
            {
                uint64_t total = 0;
                std::memcpy(&total, status + 8, sizeof(total));
                if (total > 0)
                    return static_cast<size_t>(total);
            }
#elif defined(_SC_PHYS_PAGES) && defined(_SC_PAGE_SIZE)
            const long pages = sysconf(_SC_PHYS_PAGES);
            const long page = sysconf(_SC_PAGE_SIZE);
            if (pages > 0 && page > 0)
                return static_cast<size_t>(pages) * static_cast<size_t>(page);
#endif
            return static_cast<size_t>(4) << 30;
        }

        /** This machine's logical processors (1 when the system does not say). */
        inline int logical_processors()
        {
            const int n = static_cast<int>(std::thread::hardware_concurrency());
            return n >= 1 ? n : 1;
        }
    } // namespace machine
} // namespace eemd

#endif // EEMD_SYSTEM_HPP
