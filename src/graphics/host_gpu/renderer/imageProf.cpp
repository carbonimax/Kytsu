#include "graphics/host_gpu/renderer/imageProf.h"

#include "graphics/host_gpu/renderer/perfStats.h"

#include <algorithm>
#include <atomic>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>

namespace Libs::Graphics::ImageProf {

namespace {

std::array<std::atomic<uint64_t>, Count> g_frame {};
std::atomic<uint64_t>                    g_used {0}, g_used_max {0}, g_budget {0}, g_trigger {0};
Window                                   g_window; // GPU thread only (EndFrame, Report)

constexpr const char* CauseNames[CauseCount] = {"other", "lookup",  "overlap", "expand", "replace",
                                                "bc",    "stencil", "gc",      "unmap",  "sibling"};

double Ms(uint64_t ns) noexcept {
	return static_cast<double>(ns) / 1e6;
}

double Mb(uint64_t bytes) noexcept {
	return static_cast<double>(bytes) / 1048576.0;
}

void PrintCauses(const Frame& f, uint32_t base) noexcept {
	const char* sep = "";
	for (uint32_t c = 0; c < CauseCount; c++) {
		std::printf("%s%s=%" PRIu64, sep, CauseNames[c], f[base + c]);
		sep = ",";
	}
}

} // namespace

bool ReadEnabled() noexcept {
	const char* value = std::getenv("KYTY_BIND_PROF");
	return PerfStats::Enabled() && value != nullptr && std::atoi(value) != 0;
}

void Add(Counter counter, uint64_t value) noexcept {
	if (Enabled()) {
		g_frame[counter].fetch_add(value, std::memory_order_relaxed);
	}
}

void NoteRegister(bool insert) noexcept {
	if (Enabled()) {
		Add(static_cast<Counter>((insert ? Inserted : Freed) + t_cause), 1);
		Add(Epoch, 1);
	}
}

void NoteMemory(uint64_t used, uint64_t budget, uint64_t trigger) noexcept {
	g_used.store(used, std::memory_order_relaxed);
	g_used_max.store(std::max(used, g_used_max.load(std::memory_order_relaxed)), std::memory_order_relaxed);
	g_budget.store(budget, std::memory_order_relaxed);
	g_trigger.store(trigger, std::memory_order_relaxed);
}

void EndFrame() noexcept {
	if (!Enabled()) {
		return;
	}
	Frame f {};
	for (uint32_t i = 0; i < Count; i++) {
		f[i] = g_frame[i].exchange(0, std::memory_order_relaxed);
	}
	g_window.Close(f);
}

void Report(double seconds) noexcept {
	if (!Enabled()) {
		return;
	}
	const auto& w = g_window;
	const auto& s = w.sum;
	const auto& m = w.at_max;
	uint64_t    inserted = 0;
	uint64_t    freed    = 0;
	for (uint32_t c = 0; c < CauseCount; c++) {
		inserted += s[Inserted + c];
		freed += s[Freed + c];
	}
	std::printf("PERF image %.1fs: frames=%" PRIu64
	            " cpu_max=%.2fms(create=%.2f,views=%.2f,upload=%.2f,wait=%.2f,copy=%.2f,detile=%.2f,record=%.2f,"
	            "uploads=%" PRIu64 ",%.1fMB) bytes_max=%.1fMB created=%" PRIu64 "(%.1fms) views=%" PRIu64
	            "(%.1fms) uploads=%" PRIu64 "(%.1fms,%.1fMB,wait=%.1f,copy=%.1f,detile=%.1f,record=%.1f) ins=%" PRIu64 "(",
	            seconds, w.frames, Ms(CpuNs(m)), Ms(m[CreateNs]), Ms(m[ViewNs]), Ms(m[UploadNs]), Ms(m[UpWaitNs]),
	            Ms(m[UpCopyNs]), Ms(m[UpDetileNs]), Ms(m[UpRecordNs]), m[Uploads], Mb(m[UploadBytes]), Mb(w.max_bytes),
	            s[Created], Ms(s[CreateNs]), s[Views], Ms(s[ViewNs]), s[Uploads], Ms(s[UploadNs]), Mb(s[UploadBytes]),
	            Ms(s[UpWaitNs]), Ms(s[UpCopyNs]), Ms(s[UpDetileNs]), Ms(s[UpRecordNs]), inserted);
	PrintCauses(s, Inserted);
	std::printf(") free=%" PRIu64 "(%.1fMB,", freed, Mb(s[FreedBytes]));
	PrintCauses(s, Freed);
	std::printf(") epoch=%" PRIu64 " memo=%" PRIu64 "/%" PRIu64 "(stale=%" PRIu64 ",stale_same=%" PRIu64
	            ") evicted=%" PRIu64 "(%.1fMB) vma=%.0f/%.0fMB(max=%.0f,gc_at=%.0f)\n",
	            s[Epoch], s[MemoHits], s[Lookups], s[MemoStale], s[MemoStaleSame], s[Evicted], Mb(s[EvictedBytes]),
	            Mb(g_used.load(std::memory_order_relaxed)), Mb(g_budget.load(std::memory_order_relaxed)),
	            Mb(g_used_max.exchange(0, std::memory_order_relaxed)), Mb(g_trigger.load(std::memory_order_relaxed)));
	g_window = {};
}

} // namespace Libs::Graphics::ImageProf
