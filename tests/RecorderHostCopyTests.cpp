#include "graphics/host_gpu/renderer/cache/recorderHostCopy.h"

#include <cstdio>
#include <cstdlib>

namespace {

using Libs::Graphics::RecorderHostCopy::PendingRanges;

void Check(bool value, const char* message) {
	if (!value) {
		std::fprintf(stderr, "RecorderHostCopyTests: failed: %s\n", message);
		std::abort();
	}
}

} // namespace

int main() {
	PendingRanges pending;
	Check(!pending.NeedsDrain(0x1000, 0x100, 0), "no pending copy: no drain");

	// Copies 0 and 1 queued, none made yet.
	pending.Add(0x1000, 0x1100, 0);
	pending.Add(0x8000, 0x8040, 1);
	Check(pending.NeedsDrain(0x10f0, 8, 0), "a CP write overlapping a pending copy drains");
	Check(pending.NeedsDrain(0x7ff8, 16, 0), "an overlap at the start of a copy drains");
	Check(!pending.NeedsDrain(0x1100, 8, 0), "a write just after a copy does not drain");
	Check(!pending.NeedsDrain(0x0ff8, 8, 0), "a write just before a copy does not drain");
	Check(!pending.NeedsDrain(0x4000, 0x100, 0), "a write between the copies does not drain");
	Check(!pending.NeedsDrain(0x1000, 0, 0), "an empty write does not drain");

	// The recorder made copy 0: its range no longer drains, copy 1's still does.
	Check(!pending.NeedsDrain(0x1000, 8, 1), "a made copy does not drain");
	Check(pending.NeedsDrain(0x8020, 4, 1), "the copy still pending drains");
	// Both made: nothing pending.
	Check(!pending.NeedsDrain(0x8020, 4, 2), "all copies made: no drain");
	Check(pending.Empty(), "all copies made: empty");

	// Long runs compact without losing the pending tail.
	for (uint64_t i = 0; i < 10000; i++) {
		pending.Add(0x100000 + i * 0x100, 0x100000 + i * 0x100 + 0x40, 2 + i);
	}
	pending.Prune(2 + 9000);
	Check(!pending.NeedsDrain(0x100000, 0x40, 2 + 9000), "made copies dropped after compaction");
	Check(pending.NeedsDrain(0x100000 + 9500 * 0x100, 4, 2 + 9000), "pending copies kept after compaction");

	std::printf("RecorderHostCopyTests: ok\n");
	return 0;
}
