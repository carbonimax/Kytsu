#include "graphics/host_gpu/renderer/pipeline/asyncPipelinePolicy.h"

#include <cstdio>
#include <cstdlib>
#include <deque>
#include <vector>

namespace {

using Libs::Graphics::ASYNC_PIPELINE_NOT_SKIPPED;
using Libs::Graphics::ASYNC_PIPELINE_SKIP_FRAMES;
using Libs::Graphics::AsyncPipelineMaySkip;
using Libs::Graphics::AsyncPipelinePop;

// Jobs from 100 up stand for uncapped (tessellation) keys.
constexpr bool Uncapped(int job) {
	return job >= 100;
}

// Evaluated by the compiler too (clang-cl /Zs): capped skipped keys before uncapped ones, each in order.
constexpr bool CappedFirst() {
	std::vector<int> skipped {100, 1, 101, 2};
	std::vector<int> predicted {10};
	const int        order[] = {1, 2, 100, 101, 10};
	for (const int expected: order) {
		if (AsyncPipelinePop(skipped, predicted, Uncapped) != expected) return false;
	}
	return skipped.empty() && predicted.empty();
}
static_assert(CappedFirst(), "capped skipped keys must go before uncapped ones");

// A target may be skipped from the frame after its first real draw, while it is drawn again within
// ASYNC_TARGET_STALE_FRAMES; after a longer gap it is new again.
constexpr bool TargetDrawnBefore() {
	using Libs::Graphics::ASYNC_TARGET_STALE_FRAMES;
	using Libs::Graphics::AsyncTargetDrawnBefore;
	using Libs::Graphics::AsyncTargetHistory;
	using Libs::Graphics::AsyncTargetNoteDrawn;
	AsyncTargetHistory target {};
	if (AsyncTargetDrawnBefore(target, 5)) return false;
	AsyncTargetNoteDrawn(target, 5);
	if (AsyncTargetDrawnBefore(target, 5) || !AsyncTargetDrawnBefore(target, 6)) return false;
	AsyncTargetNoteDrawn(target, 9);
	if (!AsyncTargetDrawnBefore(target, 9 + ASYNC_TARGET_STALE_FRAMES)) return false;
	const uint64_t later = 10 + ASYNC_TARGET_STALE_FRAMES;
	if (AsyncTargetDrawnBefore(target, later)) return false;
	AsyncTargetNoteDrawn(target, later);
	return !AsyncTargetDrawnBefore(target, later) && AsyncTargetDrawnBefore(target, later + 1);
}
static_assert(TargetDrawnBefore(), "only targets drawn in an earlier frame may be skipped");

void Check(bool value, const char* message) {
	if (!value) {
		std::fprintf(stderr, "AsyncPipelinePolicyTests: failed: %s\n", message);
		std::abort();
	}
}

} // namespace

int main() {
	uint64_t first = ASYNC_PIPELINE_NOT_SKIPPED;
	Check(AsyncPipelineMaySkip(first, 100) && first == 100, "the first skip did not start the cap");
	Check(AsyncPipelineMaySkip(first, 100), "a second draw in the same frame was refused");
	Check(AsyncPipelineMaySkip(first, 100 + ASYNC_PIPELINE_SKIP_FRAMES - 1),
	      "the last frame of the cap was refused");
	Check(!AsyncPipelineMaySkip(first, 100 + ASYNC_PIPELINE_SKIP_FRAMES), "the cap was not applied");
	Check(!AsyncPipelineMaySkip(first, 100 + 10 * ASYNC_PIPELINE_SKIP_FRAMES),
	      "a draw long after the cap was skipped");
	Check(first == 100, "a later skip moved the cap start");

	// A target baked once at a load (then only sampled) waits; memory reused long after by a new target
	// of the same format is new again.
	using Libs::Graphics::ASYNC_TARGET_STALE_FRAMES;
	using Libs::Graphics::AsyncTargetDrawnBefore;
	using Libs::Graphics::AsyncTargetNoteDrawn;
	Libs::Graphics::AsyncTargetHistory target {};
	Check(!AsyncTargetDrawnBefore(target, 900), "a never drawn target may be skipped");
	AsyncTargetNoteDrawn(target, 900);
	AsyncTargetNoteDrawn(target, 900);
	Check(!AsyncTargetDrawnBefore(target, 900), "a later draw of the target's first frame may be skipped");
	Check(AsyncTargetDrawnBefore(target, 901), "a target drawn in the previous frame waits");
	Check(AsyncTargetDrawnBefore(target, 900 + ASYNC_TARGET_STALE_FRAMES), "a recently drawn target waits");
	Check(!AsyncTargetDrawnBefore(target, 901 + ASYNC_TARGET_STALE_FRAMES), "a stale target may be skipped");
	AsyncTargetNoteDrawn(target, 3900);
	Check(!AsyncTargetDrawnBefore(target, 3900), "a reused target may be skipped in its first frame");
	Check(AsyncTargetDrawnBefore(target, 3901), "a reused target waits after its first frame");

	uint64_t from_zero = ASYNC_PIPELINE_NOT_SKIPPED;
	Check(AsyncPipelineMaySkip(from_zero, 0) && from_zero == 0, "frame 0 was not a valid start");
	Check(!AsyncPipelineMaySkip(from_zero, ASYNC_PIPELINE_SKIP_FRAMES), "the cap from frame 0 was not applied");

	uint64_t tess = ASYNC_PIPELINE_NOT_SKIPPED;
	Check(AsyncPipelineMaySkip(tess, 7, true) && tess == 7, "an uncapped skip did not start the clock");
	Check(AsyncPipelineMaySkip(tess, 7 + 100 * ASYNC_PIPELINE_SKIP_FRAMES, true),
	      "an uncapped draw was refused past the cap");
	Check(!AsyncPipelineMaySkip(tess, 7 + ASYNC_PIPELINE_SKIP_FRAMES), "the cap was lost after uncapped skips");

	std::deque<int> skipped {1, 2};
	std::deque<int> predicted {10, 11};
	Check(AsyncPipelinePop(skipped, predicted, Uncapped) == 1, "the oldest skipped key was not first");
	skipped.push_back(3);
	Check(AsyncPipelinePop(skipped, predicted, Uncapped) == 2, "a newer skipped key went ahead of an older one");
	Check(AsyncPipelinePop(skipped, predicted, Uncapped) == 3, "a skipped key went after a predicted one");
	Check(AsyncPipelinePop(skipped, predicted, Uncapped) == 10 && AsyncPipelinePop(skipped, predicted, Uncapped) == 11,
	      "the predicted keys were not taken in order");
	Check(skipped.empty() && predicted.empty(), "a pop left a job behind");

	// KYTY_ASYNC_TESS: an uncapped key waits behind capped ones queued after it, not behind predictions.
	skipped   = {100, 1};
	predicted = {10};
	Check(AsyncPipelinePop(skipped, predicted, Uncapped) == 1, "an uncapped key went ahead of a capped one");
	skipped.push_back(2);
	Check(AsyncPipelinePop(skipped, predicted, Uncapped) == 2, "a newer capped key waited behind an uncapped one");
	Check(AsyncPipelinePop(skipped, predicted, Uncapped) == 100, "an uncapped key went after a predicted one");
	Check(AsyncPipelinePop(skipped, predicted, [](int) { return false; }) == 10, "the predicted key was lost");

	std::puts("AsyncPipelinePolicyTests: all cases passed");
	return 0;
}
