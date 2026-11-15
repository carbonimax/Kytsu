#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_WHATIF_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_WHATIF_H_

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

// What-if experiments, off by default (measurement only): busy-wait N microseconds per frame at one
// site, spread over that site's calls, to measure frame time per ms of added CPU time.
//   KYTY_WHATIF_DEFUPLOAD_US=N  in BufferCache::FlushDeferredUploads, front thread, flushes with copies;
//   KYTY_WHATIF_BACK_US=N       on the CP back thread, per queued draw record it runs;
//   KYTY_WHATIF_FRONT_US=N      on the GPU thread (front), per queued draw, after its lookup and publish.
// Each call spins budget/expected-calls; the budget gets N at each frame and carries over (cap 2N),
// so overshoot and changing call counts even out. "PERF whatif" reports the dose per frame.
namespace Libs::Graphics::WhatIf {

enum class Site : int { DefUpload = 0, Back = 1, Front = 2, Count = 3 };

struct Dose {
	explicit Dose(const char* env) {
		const char* v = std::getenv(env);
		per_frame_ns  = v != nullptr ? std::strtoull(v, nullptr, 10) * 1000ull : 0ull;
	}
	uint64_t per_frame_ns;
	// Owner state: each site runs on one thread (or under its own lock).
	uint64_t              frame      = 0;
	uint64_t              budget     = 0;
	uint64_t              calls      = 0;
	uint64_t              prev_calls = 0;
	std::atomic<uint64_t> spun_ns {0};
	std::atomic<uint64_t> spins {0};
};

inline Dose g_dose[] = {Dose("KYTY_WHATIF_DEFUPLOAD_US"), Dose("KYTY_WHATIF_BACK_US"), Dose("KYTY_WHATIF_FRONT_US")};
inline const bool            g_any = g_dose[0].per_frame_ns != 0 || g_dose[1].per_frame_ns != 0 || g_dose[2].per_frame_ns != 0;
inline std::atomic<uint64_t> g_frame {0};

[[nodiscard]] inline bool On(Site site) noexcept {
	return g_dose[static_cast<int>(site)].per_frame_ns != 0;
}

// At the frame boundary (GPU thread).
inline void NoteFrame() noexcept {
	if (g_any) g_frame.fetch_add(1, std::memory_order_relaxed);
}

inline void Spin(Site site) noexcept {
	auto& d = g_dose[static_cast<int>(site)];
	if (d.per_frame_ns == 0) return;
	const uint64_t f = g_frame.load(std::memory_order_relaxed);
	if (f != d.frame) {
		d.budget     = std::min(d.budget + d.per_frame_ns * (f - d.frame), 2 * d.per_frame_ns);
		d.prev_calls = d.calls;
		d.calls      = 0;
		d.frame      = f;
	}
	d.calls++;
	const uint64_t quota = std::min(d.budget, d.prev_calls != 0 ? d.per_frame_ns / d.prev_calls : d.per_frame_ns);
	if (quota == 0) return;
	const auto start = std::chrono::steady_clock::now();
	const auto end   = start + std::chrono::nanoseconds(quota);
	auto       now   = start;
	while (now < end) {
		now = std::chrono::steady_clock::now();
	}
	const auto spun = static_cast<uint64_t>(std::chrono::nanoseconds(now - start).count());
	d.budget -= std::min(spun, d.budget);
	d.spun_ns.fetch_add(spun, std::memory_order_relaxed);
	d.spins.fetch_add(1, std::memory_order_relaxed);
}

// One line per PerfStats window; frames = frames of that window.
inline void Report(double seconds, uint64_t frames) noexcept {
	if (!g_any) return;
	const double f = frames != 0 ? static_cast<double>(frames) : 1.0;
	double       ms[3];
	double       per[3];
	for (int i = 0; i < 3; i++) {
		ms[i]  = static_cast<double>(g_dose[i].spun_ns.exchange(0, std::memory_order_relaxed)) / 1e6 / f;
		per[i] = static_cast<double>(g_dose[i].spins.exchange(0, std::memory_order_relaxed)) / f;
	}
	std::printf("PERF whatif %.1fs: frames=%" PRIu64 " defupload_us=%" PRIu64 " %.3fms/f(%.0f/f) back_us=%" PRIu64
	            " %.3fms/f(%.0f/f) front_us=%" PRIu64 " %.3fms/f(%.0f/f)\n",
	            seconds, frames, g_dose[0].per_frame_ns / 1000, ms[0], per[0], g_dose[1].per_frame_ns / 1000, ms[1],
	            per[1], g_dose[2].per_frame_ns / 1000, ms[2], per[2]);
}

} // namespace Libs::Graphics::WhatIf

#endif /* EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_WHATIF_H_ */
