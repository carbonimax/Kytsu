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

} // namespace

int main() {
	TestWriteCountsAfterItsDraw();
	TestPageBoundByPendingDrawWaits();
	TestPageOfReadbackInFlightIsLeft();
	std::printf("EagerPagesTests: ok\n");
	return 0;
}
