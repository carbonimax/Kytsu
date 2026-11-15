#include "kernel/sleepProf.h"

#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#if defined(_WIN32)
#include <windows.h>
#else
#include "common/threads.h"
#endif

namespace Libs::LibKernel::SleepProf {

namespace {

constexpr size_t MaxSlots = 512;
constexpr size_t TopRows  = 6;

std::array<Slot, MaxSlots> g_slots;
std::atomic<size_t>        g_used {0};
std::atomic<uint64_t>      g_dropped {0}; // threads that found no free slot (since launch)
thread_local Slot*         t_slot = nullptr;

bool EnvOn(const char* name) noexcept {
	const char* value = std::getenv(name);
	return value != nullptr && std::atoi(value) != 0;
}

uint32_t OsThreadId() noexcept {
#if defined(_WIN32)
	return static_cast<uint32_t>(GetCurrentThreadId()); // same ids as the psamp captures
#else
	return static_cast<uint32_t>(Common::Thread::GetThreadIdUnique());
#endif
}

void PrintRow(const Window& w) noexcept {
	std::printf(" [%s#%" PRIu32 " zero=%" PRIu64, w.slot->name, w.slot->tid, w.zero);
	static constexpr const char* Names[BucketCount] = {"le100", "le500", "le1ms", "gt1ms"};
	for (int b = 0; b < BucketCount; b++) {
		const double over_us = w.count[b] != 0 ? static_cast<double>(w.over_ns[b]) / 1e3 / static_cast<double>(w.count[b]) : 0.0;
		std::printf(" %s=%" PRIu64 "/%.1f", Names[b], w.count[b], over_us);
	}
	std::printf("]");
}

} // namespace

bool Enabled() noexcept {
	static const bool enabled = EnvOn("KYTY_PERF_STATS") && EnvOn("KYTY_SLEEP_PROF");
	return enabled;
}

void Record(uint64_t requested_ns, uint64_t actual_ns, const char* name) noexcept {
	if (t_slot == nullptr) {
		const size_t index = g_used.fetch_add(1, std::memory_order_relaxed);
		if (index >= MaxSlots) {
			g_used.store(MaxSlots, std::memory_order_relaxed);
			g_dropped.fetch_add(1, std::memory_order_relaxed);
			return; // shortcut: retried on every sleep of this thread, only past 512 sleeping threads
		}
		Slot& slot = g_slots[index];
		std::snprintf(slot.name, sizeof(slot.name), "%s", name != nullptr && name[0] != '\0' ? name : "guest");
		slot.tid = OsThreadId();
		slot.ready.store(true, std::memory_order_release);
		t_slot = &slot;
	}
	t_slot->Add(requested_ns, actual_ns);
}

void Report(double seconds, uint64_t frames) noexcept {
	if (!Enabled()) {
		return;
	}
	static std::array<Window, MaxSlots> windows; // report thread only
	const size_t used = Take(g_slots.data(), (std::min)(g_used.load(std::memory_order_acquire), MaxSlots), windows.data());
	// counts per window; after '/', mean oversleep (actual - requested) in us per request
	std::printf("PERF sleep %.1fs frames=%" PRIu64 " threads=%zu dropped=%" PRIu64 ":", seconds, frames, used,
	            g_dropped.load(std::memory_order_relaxed));
	for (size_t i = 0; i < used; i++) {
		// The main guest thread is always printed: no row then means it did not sleep.
		if (i < TopRows || std::strcmp(windows[i].slot->name, "MainThread") == 0) {
			PrintRow(windows[i]);
		}
	}
	std::printf("\n");
}

} // namespace Libs::LibKernel::SleepProf
