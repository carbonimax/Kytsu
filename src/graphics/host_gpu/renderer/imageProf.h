#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_IMAGEPROF_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_IMAGEPROF_H_

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <utility>

// KYTY_BIND_PROF (non-zero, with KYTY_PERF_STATS=1): image-path counters per guest frame, one
// "PERF image" line per stats window with the window sums and the most expensive frame (image CPU
// time = creation + view creation + upload, and its split). Off, each site costs one cached bool test.
namespace Libs::Graphics::ImageProf {

// Where an image (un)registration, i.e. a bump of the texture cache index epoch, came from.
enum Cause : uint8_t { Other, Lookup, Overlap, Expand, Replace, BcAlias, Stencil, Gc, Unmap, Sibling, CauseCount };

enum Counter : uint32_t {
	Created, CreateNs, Views, ViewNs,
	Uploads, UploadBytes, UploadNs, UpWaitNs, UpCopyNs, UpDetileNs, UpRecordNs,
	FreedBytes, Epoch, Evicted, EvictedBytes,
	Lookups, MemoHits, MemoStale, MemoStaleSame,
	Inserted,                     // + Cause
	Freed = Inserted + CauseCount, // + Cause
	Count = Freed + CauseCount
};

using Frame = std::array<uint64_t, Count>;

[[nodiscard]] bool ReadEnabled() noexcept;
[[nodiscard]] inline bool Enabled() noexcept {
	static const bool enabled = ReadEnabled();
	return enabled;
}

// Adds to the current frame (any thread, relaxed atomic); no-op when off.
void Add(Counter counter, uint64_t value) noexcept;
// Registration (insert) or unregistration of an image, under the calling thread's current cause.
void NoteRegister(bool insert) noexcept;
// Texture cache GC, once per frame: device memory in use, budget and GC trigger.
void NoteMemory(uint64_t used, uint64_t budget, uint64_t trigger) noexcept;
// GPU thread, at each guest frame boundary (suspend point; the CP back thread is drained there).
void EndFrame() noexcept;
// GPU thread, from PerfStats::MaybeReport.
void Report(double seconds) noexcept;

[[nodiscard]] inline uint64_t Now() noexcept {
	return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
	                                 std::chrono::steady_clock::now().time_since_epoch())
	                                 .count());
}

inline thread_local uint8_t t_cause     = Other;
inline thread_local bool    t_in_upload = false;

// Image CPU time of one frame: what the "most expensive frame" is chosen by.
[[nodiscard]] inline uint64_t CpuNs(const Frame& f) noexcept {
	return f[CreateNs] + f[ViewNs] + f[UploadNs];
}

// Window of closed frames (GPU thread only).
struct Window {
	uint64_t frames    = 0;
	uint64_t max_bytes = 0; // largest upload bytes of one frame
	Frame    sum {};
	Frame    at_max {}; // the frame with the largest CpuNs
	void     Close(const Frame& f) noexcept {
		frames++;
		for (uint32_t i = 0; i < Count; i++) {
			sum[i] += f[i];
		}
		if (CpuNs(f) > CpuNs(at_max)) {
			at_max = f;
		}
		max_bytes = std::max(max_bytes, f[UploadBytes]);
	}
};

// Times its scope into `ns` and counts one `count` when on. An upload step (`step`) only counts
// inside an UploadScope, so shared paths (stream buffer waits) are timed only for image uploads.
class ScopedTimer {
public:
	explicit ScopedTimer(Counter ns, Counter count = Count, bool step = false) noexcept
	    : m_ns(ns), m_count(count), m_start(Enabled() && (!step || t_in_upload) ? Now() : 0) {}
	~ScopedTimer() {
		if (m_start != 0) {
			Add(m_ns, Now() - m_start);
			if (m_count != Count) {
				Add(m_count, 1);
			}
		}
	}
	ScopedTimer(const ScopedTimer&)            = delete;
	ScopedTimer& operator=(const ScopedTimer&) = delete;

private:
	Counter  m_ns;
	Counter  m_count;
	uint64_t m_start;
};

// One image upload (obtain the source, detile, record the copy); enables the step timers below it.
class UploadScope {
public:
	explicit UploadScope(uint64_t bytes) noexcept: m_bytes(bytes), m_start(Enabled() ? Now() : 0) {
		if (m_start != 0) {
			m_saved = std::exchange(t_in_upload, true);
		}
	}
	~UploadScope() {
		if (m_start != 0) {
			t_in_upload = m_saved;
			Add(UploadNs, Now() - m_start);
			Add(Uploads, 1);
			Add(UploadBytes, m_bytes);
		}
	}
	UploadScope(const UploadScope&)            = delete;
	UploadScope& operator=(const UploadScope&) = delete;

private:
	uint64_t m_bytes;
	uint64_t m_start;
	bool     m_saved = false;
};

// Cause of the (un)registrations below it on this thread; the innermost one wins.
class ScopedCause {
public:
	explicit ScopedCause(Cause cause) noexcept: m_on(Enabled()) {
		if (m_on) {
			m_saved = std::exchange(t_cause, cause);
		}
	}
	~ScopedCause() {
		if (m_on) {
			t_cause = m_saved;
		}
	}
	ScopedCause(const ScopedCause&)            = delete;
	ScopedCause& operator=(const ScopedCause&) = delete;

private:
	bool    m_on;
	uint8_t m_saved = Other;
};

} // namespace Libs::Graphics::ImageProf

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_IMAGEPROF_H_
