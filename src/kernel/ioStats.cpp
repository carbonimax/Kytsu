#include "kernel/ioStats.h"

#include "common/threads.h"
#include "graphics/host_gpu/renderer/perfStats.h"
#include "kernel/pthread.h"

#include <algorithm>
#include <array>
#include <cinttypes>
#include <cstdio>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>

namespace Libs::LibKernel::IoStats {

namespace {

constexpr auto ApiCount = static_cast<size_t>(Api::Count);
constexpr std::array<const char*, ApiCount> ApiNames = {"read", "pread", "preadv", "aio", "apr", "open", "apr_open", "close"};

struct Slot {
	uint64_t                                  calls  = 0;
	uint64_t                                  bytes  = 0;
	uint64_t                                  ns     = 0;
	uint64_t                                  max_ns = 0;
	uint64_t                                  reopens = 0;
	std::unordered_map<std::string, uint64_t> thread_ns;
};

struct State {
	std::mutex                 mutex;
	std::array<Slot, ApiCount> slots;
	std::array<uint64_t, 4>    sizes {}; // reads < 4 KiB, < 64 KiB, < 1 MiB, >= 1 MiB
	uint64_t                   lock_ns       = 0;
	uint64_t                   lock_max_ns   = 0;
	uint64_t                   queue_ns      = 0;
	uint64_t                   queue_max_ns  = 0;
	uint64_t                   queue_count   = 0;
	std::unordered_set<size_t> opened; // whole session
};

State& Get() {
	static State state;
	return state;
}

thread_local Api t_override = Api::Count;

const std::string& ThreadName() {
	thread_local const std::string name = [] {
		if (auto* self = PthreadSelfOrNull(); self != nullptr) {
			char buf[32] {};
			if (PthreadGetname(self, buf) == 0 && buf[0] != '\0') {
				return std::string(buf);
			}
		}
		return "host" + std::to_string(Common::Thread::GetThreadIdUnique());
	}();
	return name;
}

bool IsRead(Api api) {
	return api == Api::Read || api == Api::Pread || api == Api::Preadv || api == Api::Aio || api == Api::Apr;
}

} // namespace

bool Enabled() noexcept {
	return Graphics::PerfStats::Enabled();
}

void Note(Api api, uint64_t bytes, uint64_t ns) noexcept {
	if (!Enabled()) {
		return;
	}
	if (t_override != Api::Count && IsRead(api)) {
		api = t_override;
	}
	const auto& name  = ThreadName();
	auto&       state = Get();
	std::scoped_lock lock(state.mutex);
	auto& slot = state.slots[static_cast<size_t>(api)];
	slot.calls++;
	slot.bytes += bytes;
	slot.ns += ns;
	slot.max_ns = std::max(slot.max_ns, ns);
	slot.thread_ns[name] += ns;
	if (IsRead(api)) {
		state.sizes[bytes < 4096 ? 0 : bytes < 65536 ? 1 : bytes < 1048576 ? 2 : 3]++;
	}
}

void NoteOpenedPath(Api api, size_t path_hash) noexcept {
	if (!Enabled()) {
		return;
	}
	auto&            state = Get();
	std::scoped_lock lock(state.mutex);
	if (!state.opened.insert(path_hash).second) {
		state.slots[static_cast<size_t>(api)].reopens++;
	}
}

void NoteLockWait(uint64_t ns) noexcept {
	auto&            state = Get();
	std::scoped_lock lock(state.mutex);
	state.lock_ns += ns;
	state.lock_max_ns = std::max(state.lock_max_ns, ns);
}

void NoteAprQueueWait(uint64_t ns) noexcept {
	if (!Enabled()) {
		return;
	}
	auto&            state = Get();
	std::scoped_lock lock(state.mutex);
	state.queue_ns += ns;
	state.queue_max_ns = std::max(state.queue_max_ns, ns);
	state.queue_count++;
}

void Report(double seconds) noexcept {
	if (!Enabled()) {
		return;
	}
	auto&                      state = Get();
	std::array<Slot, ApiCount> slots;
	std::array<uint64_t, 4>    sizes {};
	uint64_t                   lock_ns = 0, lock_max = 0, queue_ns = 0, queue_max = 0, queue_n = 0;
	{
		std::scoped_lock lock(state.mutex);
		slots.swap(state.slots);
		sizes.swap(state.sizes);
		std::swap(lock_ns, state.lock_ns);
		std::swap(lock_max, state.lock_max_ns);
		std::swap(queue_ns, state.queue_ns);
		std::swap(queue_max, state.queue_max_ns);
		std::swap(queue_n, state.queue_count);
	}
	uint64_t total_calls = 0;
	for (const auto& slot: slots) {
		total_calls += slot.calls;
	}
	if (total_calls == 0 && queue_n == 0) {
		return;
	}
	std::string line;
	char        buf[256];
	for (size_t i = 0; i < ApiCount; i++) {
		const auto& slot = slots[i];
		if (slot.calls == 0) {
			continue;
		}
		const std::pair<const std::string, uint64_t>* top = nullptr;
		for (const auto& entry: slot.thread_ns) {
			if (top == nullptr || entry.second > top->second) {
				top = &entry;
			}
		}
		std::snprintf(buf, sizeof(buf), " %s=%" PRIu64 "/%.1fMB/%.0fms(max=%.1fms,reopen=%" PRIu64 ",threads=%zu,top=%s:%.0fms)",
		              ApiNames[i], slot.calls, static_cast<double>(slot.bytes) / 1048576.0,
		              static_cast<double>(slot.ns) / 1e6, static_cast<double>(slot.max_ns) / 1e6, slot.reopens,
		              slot.thread_ns.size(), top->first.c_str(), static_cast<double>(top->second) / 1e6);
		line += buf;
	}
	std::printf("PERF io %.1fs:%s lock=%.0fms(max=%.1fms) apr_queue=%.0fms(%" PRIu64 ",max=%.1fms) sizes=%" PRIu64
	            "/%" PRIu64 "/%" PRIu64 "/%" PRIu64 "\n",
	            seconds, line.c_str(), static_cast<double>(lock_ns) / 1e6, static_cast<double>(lock_max) / 1e6,
	            static_cast<double>(queue_ns) / 1e6, queue_n, static_cast<double>(queue_max) / 1e6, sizes[0], sizes[1],
	            sizes[2], sizes[3]);
}

Timer::Timer(Api api) noexcept: m_api(api) {
	if (Enabled()) {
		m_start = std::chrono::steady_clock::now();
		m_on    = true;
	}
}

void Timer::Locked() noexcept {
	if (m_on) {
		NoteLockWait(static_cast<uint64_t>(
		    std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - m_start).count()));
	}
}

void Timer::Done(uint64_t bytes) noexcept {
	if (!m_on) {
		return;
	}
	m_on = false;
	Note(m_api, bytes,
	     static_cast<uint64_t>(
	         std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - m_start).count()));
}

ApiScope::ApiScope(Api api) noexcept: m_previous(t_override) {
	t_override = api;
}

ApiScope::~ApiScope() {
	t_override = m_previous;
}

} // namespace Libs::LibKernel::IoStats
