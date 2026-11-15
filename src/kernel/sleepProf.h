#ifndef KERNEL_SLEEP_PROF_H_
#define KERNEL_SLEEP_PROF_H_

#include <algorithm>
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>

// KYTY_SLEEP_PROF=1 (measurement, default 0; needs KYTY_PERF_STATS=1): guest sleep requests
// (sceKernelUsleep, usleep, sceKernelNanosleep, nanosleep, sceKernelSleep) per guest thread,
// bucketed by requested duration, with the summed oversleep (actual - requested). Reported as one
// "PERF sleep" line per PerfStats window. Off: one cached flag test per call.
namespace Libs::LibKernel::SleepProf {

constexpr int BucketCount = 4; // <= 100 us, <= 500 us, <= 1 ms, > 1 ms

[[nodiscard]] constexpr int Bucket(uint64_t requested_ns) noexcept {
	return requested_ns <= 100000 ? 0 : requested_ns <= 500000 ? 1 : requested_ns <= 1000000 ? 2 : 3;
}
static_assert(Bucket(1) == 0 && Bucket(100000) == 0 && Bucket(100001) == 1 && Bucket(500000) == 1 &&
              Bucket(500001) == 2 && Bucket(1000000) == 2 && Bucket(1000001) == 3);

// One guest thread. Written by its thread, taken (read and reset) by the report.
struct Slot {
	char                                          name[32] {};
	uint32_t                                      tid = 0;
	std::atomic<bool>                             ready {false};
	std::atomic<uint64_t>                         zero {0}; // zero-length requests: no sleep, a signal poll only
	std::array<std::atomic<uint64_t>, BucketCount> count {};
	std::array<std::atomic<uint64_t>, BucketCount> over_ns {};

	void Add(uint64_t requested_ns, uint64_t actual_ns) noexcept {
		if (requested_ns == 0) {
			zero.fetch_add(1, std::memory_order_relaxed);
			return;
		}
		const int b = Bucket(requested_ns);
		count[b].fetch_add(1, std::memory_order_relaxed);
		over_ns[b].fetch_add(actual_ns > requested_ns ? actual_ns - requested_ns : 0, std::memory_order_relaxed);
	}
};

struct Window {
	const Slot* slot = nullptr;
	uint64_t    zero = 0;
	uint64_t    count[BucketCount] {};
	uint64_t    over_ns[BucketCount] {};
	uint64_t    total = 0; // zero + all buckets
};

// Moves each ready slot's counters into `out` (one window per slot that slept, the slot restarts at
// zero), busiest first. `out` holds `n` windows. Returns the windows written.
inline size_t Take(Slot* slots, size_t n, Window* out) noexcept {
	size_t used = 0;
	for (size_t i = 0; i < n; i++) {
		Slot& s = slots[i];
		if (!s.ready.load(std::memory_order_acquire)) {
			continue;
		}
		Window w;
		w.slot  = &s;
		w.zero  = s.zero.exchange(0, std::memory_order_relaxed);
		w.total = w.zero;
		for (int b = 0; b < BucketCount; b++) {
			w.count[b]   = s.count[b].exchange(0, std::memory_order_relaxed);
			w.over_ns[b] = s.over_ns[b].exchange(0, std::memory_order_relaxed);
			w.total += w.count[b];
		}
		if (w.total != 0) {
			out[used++] = w;
		}
	}
	std::stable_sort(out, out + used, [](const Window& a, const Window& b) { return a.total > b.total; });
	return used;
}

[[nodiscard]] bool Enabled() noexcept;
// Records one finished sleep of the calling guest thread; `name` names its slot on the first call.
void Record(uint64_t requested_ns, uint64_t actual_ns, const char* name) noexcept;
// One "PERF sleep" line for the window (PerfStats report).
void Report(double seconds, uint64_t frames) noexcept;

} // namespace Libs::LibKernel::SleepProf

#endif
