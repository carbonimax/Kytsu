#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_CACHE_EAGERPAGES_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_CACHE_EAGERPAGES_H_

#include "graphics/host_gpu/rangeSet.h"

#include <cstdint>
#include <vector>

namespace Libs::Graphics {

// KYTY_EAGER_READBACK: a write binding over read-back pages (BufferCache::EagerNoteWrite).
struct EagerWrite {
	uint64_t begin = 0, end = 0, work = 0; // work: draws and dispatches started before the binding
};

// The pages a submission downloads (BufferCache::EagerRecord), `work` draws and dispatches started so far.
// A noted write counts once a later draw started; it then leaves `writes`. Left out: a page a binding still
// waiting for its draw notes (that draw writes it next, it stays noted), and a page of a readback in flight
// (`in_flight`: its bytes left the dirty set unpublished, and that readback publishes them).
inline RangeSet EagerReadyPages(std::vector<EagerWrite>& writes, uint64_t work, const RangeSet& in_flight) {
	RangeSet ready;
	std::erase_if(writes, [&](const EagerWrite& write) {
		if (write.work >= work) {
			return false;
		}
		ready.Add(write.begin, write.end - write.begin);
		return true;
	});
	for (const auto& write: writes) {
		ready.Subtract(write.begin, write.end - write.begin);
	}
	in_flight.ForEach([&](uint64_t begin, uint64_t end) { ready.Subtract(begin, end - begin); });
	return ready;
}

// Who waits for an eager download in flight (BufferCache::WaitEager). `bytes`: what it downloaded; `keep`: its
// pages bound for writing since. A reader of [vaddr, vaddr + size) waits when the download carries those bytes
// (Read). A window copied now waits first when the download's pages were bound for writing since (BeforeCopy):
// the copy takes the newer bytes, and the download, published after it, would put its older ones back.
enum class EagerWait { Read, BeforeCopy };
inline bool EagerMustWait(EagerWait kind, const RangeSet& bytes, const RangeSet& keep, uint64_t vaddr,
                          uint64_t size) {
	return (kind == EagerWait::Read ? bytes : keep).Intersects(vaddr, size);
}

} // namespace Libs::Graphics

#endif /* EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_CACHE_EAGERPAGES_H_ */
