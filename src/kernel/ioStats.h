#ifndef EMULATOR_SRC_KERNEL_IOSTATS_H_
#define EMULATOR_SRC_KERNEL_IOSTATS_H_

#include <chrono>
#include <cstddef>
#include <cstdint>

// Guest file I/O counters (measurement, KYTY_PERF_STATS=1, default off: nothing is timed), reported as
// one "PERF io" line with the "PERF stats" lines: per API calls, bytes, wall time, longest call and the
// thread that spent the most time in it; opens of a path already opened before; file mutex waits;
// APR queue waits (submission to its first command, unless that command is a guest wait) and a read size
// histogram. With KYTY_COMPILE_LOG, one "I time api bytes ns" line per call (same clock as its other lines).
namespace Libs::LibKernel::IoStats {

// AprCopy: the memcpy from the APR bounce buffer into guest memory (part of Apr). AprPreopen: KYTY_APR_PREOPEN
// opens off the engine thread (not part of AprOpen). AprResolve: guest path resolution (stat of a new path).
enum class Api : uint32_t { Read, Pread, Preadv, Aio, Apr, AprCopy, Open, AprOpen, Close, AprPreopen, AprResolve, Count };

[[nodiscard]] bool Enabled() noexcept;
void               Note(Api api, uint64_t bytes, uint64_t ns) noexcept;
// Counts a successful open of `path_hash` (std::filesystem::hash_value); a hash seen before is a reopen.
void NoteOpenedPath(Api api, size_t path_hash) noexcept;
void NoteLockWait(uint64_t ns) noexcept;
void NoteAprQueueWait(uint64_t ns) noexcept;
// Prints the "PERF io" line and resets the interval counters (PerfStats::MaybeReport).
void Report(double seconds) noexcept;

// Times one call; Done() (or the destructor, with 0 bytes) records it. Locked() records the time spent
// so far as a file mutex wait.
class Timer {
public:
	explicit Timer(Api api) noexcept;
	~Timer() { Done(0); }
	void Locked() noexcept;
	void Done(uint64_t bytes) noexcept;
	Timer(const Timer&)            = delete;
	Timer& operator=(const Timer&) = delete;

private:
	Api                                   m_api;
	std::chrono::steady_clock::time_point m_start {};
	bool                                  m_on = false;
};

// While alive, reads of the calling thread are counted under `api` (Aio submissions run KernelPread).
class ApiScope {
public:
	explicit ApiScope(Api api) noexcept;
	~ApiScope();
	ApiScope(const ApiScope&)            = delete;
	ApiScope& operator=(const ApiScope&) = delete;

private:
	Api m_previous;
};

} // namespace Libs::LibKernel::IoStats

#endif // EMULATOR_SRC_KERNEL_IOSTATS_H_
