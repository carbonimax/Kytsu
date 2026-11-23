#include "graphics/host_gpu/renderer/cache/eagerPages.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

using Libs::Graphics::EagerReadyPages;
using Libs::Graphics::EagerWrite;
using Libs::Graphics::RangeSet;

namespace {

constexpr uint64_t P = 0x1000; // tracker page

void Check(bool value, const char* message) {
	if (!value) {
		std::fprintf(stderr, "EagerPagesTests: failed: %s\n", message);
		std::abort();
	}
}

bool Is(const RangeSet& set, uint64_t begin, uint64_t end) {
	uint64_t n = 0, b = 0, e = 0;
	set.ForEach([&](uint64_t x, uint64_t y) {
		n++;
		b = x;
		e = y;
	});
	return n == 1 && b == begin && e == end;
}

void TestWriteCountsAfterItsDraw() {
	std::vector<EagerWrite> writes {{10 * P, 11 * P, 4}};
	Check(EagerReadyPages(writes, 4, {}).Empty() && writes.size() == 1,
	      "a write whose draw has not started was taken (mid-draw submission)");
	Check(Is(EagerReadyPages(writes, 5, {}), 10 * P, 11 * P) && writes.empty(),
	      "a write followed by its draw was not taken");
}

// Review case (a): a page an earlier draw wrote and the draw being prepared binds again.
void TestPageBoundByPendingDrawWaits() {
	std::vector<EagerWrite> writes {{10 * P, 12 * P, 3}, {11 * P, 12 * P, 5}};
	Check(Is(EagerReadyPages(writes, 5, {}), 10 * P, 11 * P),
	      "a page the draw being prepared binds was downloaded before that draw wrote it");
	Check(writes.size() == 1 && writes[0].begin == 11 * P, "the pending binding was not kept noted");
	Check(Is(EagerReadyPages(writes, 6, {}), 11 * P, 12 * P), "the page was not downloaded after its draw");
}

// Review case (b): a page whose bytes a readback in flight took out of the dirty set.
void TestPageOfReadbackInFlightIsLeft() {
	std::vector<EagerWrite> writes {{10 * P, 13 * P, 1}};
	RangeSet               in_flight;
	in_flight.Add(11 * P, P);
	const auto ready = EagerReadyPages(writes, 2, in_flight);
	Check(!ready.Intersects(11 * P, P), "a page with a readback in flight was downloaded");
	Check(ready.Contains(10 * P, P) && ready.Contains(12 * P, P), "the other pages were not downloaded");
	Check(writes.empty(), "the readback's page stayed noted (that readback publishes it)");
}

// Model of a window read on the readback queue (BufferCache::ReadWindowOnSideQueue, =1): the eager downloads in
// flight that `must_wait` selects publish first, the copy then publishes the window's dirty bytes, and the others
// publish afterwards (priority-operation thread).
struct Download {
	RangeSet bytes, keep;
	uint8_t  value = 0;
};
using Selector = bool (*)(Libs::Graphics::EagerWait, const RangeSet&, const RangeSet&, uint64_t, uint64_t);
std::vector<uint8_t> ReadWindow(std::vector<uint8_t> backing, const std::vector<uint8_t>& gpu, const RangeSet& dirty,
                                const std::vector<Download>& in_flight, uint64_t begin, uint64_t size,
                                Selector must_wait) {
	const auto publish = [&](const Download& d) {
		d.bytes.ForEach([&](uint64_t b, uint64_t e) {
			for (auto x = b; x < e; x++) {
				backing[x] = d.value;
			}
		});
	};
	std::vector<const Download*> later;
	for (const auto& d: in_flight) {
		if (must_wait(Libs::Graphics::EagerWait::BeforeCopy, d.bytes, d.keep, begin, size)) {
			publish(d);
		} else {
			later.push_back(&d);
		}
	}
	dirty.ForEachInRange(begin, size, [&](uint64_t b, uint64_t e) {
		for (auto x = b; x < e; x++) {
			backing[x] = gpu[x];
		}
	});
	for (const auto* d: later) {
		publish(*d);
	}
	return backing;
}

bool NeverWait(Libs::Graphics::EagerWait, const RangeSet&, const RangeSet&, uint64_t, uint64_t) {
	return false;
}

// A page rewritten after its eager download was recorded: the download carries the older bytes, the window copy the
// newer ones. The older ones must not be published over the newer.
void TestRewrittenPageLandsBeforeCopy() {
	std::vector<uint8_t> backing(3 * P, 0), gpu(3 * P, 0);
	Download             rewritten;
	rewritten.bytes.Add(P + 16, 16);
	rewritten.value = 1;          // the GPU wrote 1, downloaded early
	rewritten.keep.Add(P, P);     // then a write binding over that page
	for (uint64_t x = P + 16; x < P + 32; x++) {
		gpu[x] = 2;               // which wrote 2
	}
	RangeSet dirty;
	dirty.Add(P + 16, 16);
	Download other; // a download of other bytes, not rewritten
	other.bytes.Add(2 * P, 16);
	other.value = 7;
	const std::vector<Download> in_flight {rewritten, other};

	Check(!Libs::Graphics::EagerMustWait(Libs::Graphics::EagerWait::BeforeCopy, other.bytes, other.keep, P, 2 * P),
	      "a window copy waited for a download whose pages were not rewritten");
	const auto good = ReadWindow(backing, gpu, dirty, in_flight, P, 2 * P, Libs::Graphics::EagerMustWait);
	Check(good[P + 16] == 2 && good[P + 31] == 2, "an eager download published its older bytes over the window copy");
	Check(good[2 * P] == 7, "the other download's bytes were lost");
	// The model catches the broken order: without that wait the older bytes win.
	const auto bad = ReadWindow(backing, gpu, dirty, in_flight, P, 2 * P, NeverWait);
	Check(bad[P + 16] == 1, "the model does not detect a download published after the copy");
	// A reader waits for a download of its own bytes only.
	Check(Libs::Graphics::EagerMustWait(Libs::Graphics::EagerWait::Read, rewritten.bytes, rewritten.keep, P + 20, 4) &&
	          !Libs::Graphics::EagerMustWait(Libs::Graphics::EagerWait::Read, rewritten.bytes, rewritten.keep, P + 40, 4),
	      "a reader waited for a download of other bytes of its page, or not for its own");
}

} // namespace

int main() {
	TestWriteCountsAfterItsDraw();
	TestPageBoundByPendingDrawWaits();
	TestPageOfReadbackInFlightIsLeft();
	TestRewrittenPageLandsBeforeCopy();
	std::printf("EagerPagesTests: ok\n");
	return 0;
}
