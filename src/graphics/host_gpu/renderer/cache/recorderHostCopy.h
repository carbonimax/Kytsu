#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_CACHE_RECORDERHOSTCOPY_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_CACHE_RECORDERHOSTCOPY_H_

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

// KYTY_RECORDER_HOST_COPY=1 (default; 0 = off): the stream copy of a small CPU-dirty read binding
// (BufferCache::ObtainBuffer) is reserved in the stream ring by the command processor, but the
// recorder thread (KYTY_CP_RECORD_THREAD=1) copies the bytes, as a CmdRecord::Call queued in ring
// order, so before the submission of the commands that read them (a direct submission drains the
// ring first). The guest bytes are read later than the draw was processed, but still before the
// host GPU runs it, as the PS5's GPU reads them.
// Between the reservation and the recorder's copy, the guest could change the source bytes in
// three ways, each handled:
// - a command-processor write that overlaps a pending copy (WRITE_DATA, DMA_DATA, reference
//   clock, semaphores, LOD stats, synthetic occlusion results) drains the recorder first, so a
//   draw never sees a later CP write (NoteCpWrite);
// - a completion signal the CP writes without waiting for the host GPU (RELEASE_MEM labels with
//   KYTY_EOP_AT_GPU=0) drains the recorder whenever a copy is pending, so the guest cannot see
//   "done" and rewrite the source before the copy (DrainPending); commands from other threads
//   (memory unmaps) drain the same way;
// - a guest CPU write after the reservation: CPU-dirty pages are not protected, so it is not seen.
//   The guest has not been told the GPU read the bytes (every signal drains first), so such a
//   write races with the GPU on hardware too.
// Writes that wait for a host GPU tick (downloads, GDS reads, occlusion dumps) run after the
// submission, which follows the copies in ring order. Ring reuse: a reserved range is handed out
// again only once its tick completed on the GPU, after the copy.
// Stream reuse (KYTY_STREAM_REUSE): a reused range may still wait for its copy; the draw that reuses
// it is recorded later, so it reads the bytes after the copy. Reuse checks (=2) read the ring on
// the CPU, so they drain first.
// Off (direct copy) without the recorder thread or while it does not take the command buffer
// (CommandScheduler::Capturing: fast frames, tests), off the GPU thread, with a non-coherent stream
// ring, with KYTY_EOP_AT_GPU (a late overlay would see labels released after the draw) and with
// the late-read detector. KYTY_AB_FLAG=recorder_host_copy switches it per frame: the pending
// copies stay tracked whatever the arm.
namespace Libs::Graphics::RecorderHostCopy {

// The guest ranges of the copies queued for the recorder, in queue order (sequence numbers from
// 0). Producer only: the GPU thread, front or back of KYTY_CP_PIPELINE, never both at once.
class PendingRanges {
public:
	void Add(uint64_t begin, uint64_t end, uint64_t seq) {
		m_ranges.push_back({begin, end, seq});
		m_lo = std::min(m_lo, begin);
		m_hi = std::max(m_hi, end);
	}
	// Forgets the copies the recorder made: those with seq < done.
	void Prune(uint64_t done) {
		while (m_head < m_ranges.size() && m_ranges[m_head].seq < done) {
			m_head++;
		}
		if (m_head == m_ranges.size()) {
			Clear();
		} else if (m_head >= 4096 && m_head * 2 >= m_ranges.size()) {
			m_ranges.erase(m_ranges.begin(), m_ranges.begin() + static_cast<std::ptrdiff_t>(m_head));
			m_head = 0;
		}
	}
	[[nodiscard]] bool Overlaps(uint64_t address, uint64_t size) const noexcept {
		const auto end = address + size;
		if (size == 0 || end <= m_lo || address >= m_hi) {
			return false;
		}
		return std::any_of(m_ranges.begin() + static_cast<std::ptrdiff_t>(m_head), m_ranges.end(),
		                   [&](const Range& r) { return address < r.end && r.begin < end; });
	}
	// A CP write of [address, address + size) with `done` copies made: true when it must drain.
	[[nodiscard]] bool NeedsDrain(uint64_t address, uint64_t size, uint64_t done) {
		if (Empty()) {
			return false;
		}
		Prune(done);
		return Overlaps(address, size);
	}
	[[nodiscard]] bool Empty() const noexcept { return m_head == m_ranges.size(); }
	void               Clear() noexcept {
		m_ranges.clear();
		m_head = 0;
		m_lo   = UINT64_MAX;
		m_hi   = 0;
	}

private:
	struct Range {
		uint64_t begin;
		uint64_t end;
		uint64_t seq;
	};
	std::vector<Range> m_ranges;
	size_t             m_head = 0;
	uint64_t           m_lo   = UINT64_MAX;
	uint64_t           m_hi   = 0;
};

// GPU thread: a CP write of guest memory [address, address + size) is about to happen; drains the
// recorder when it overlaps a pending copy.
void NoteCpWrite(uint64_t address, uint64_t size);
// GPU thread: a completion signal (or a command from another thread) is about to reach the guest;
// drains the recorder when any copy is pending.
void DrainPending();
// PERF hostcopy line (KYTY_PERF_STATS=1).
void Report(double seconds, uint64_t frames);

} // namespace Libs::Graphics::RecorderHostCopy

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_CACHE_RECORDERHOSTCOPY_H_
