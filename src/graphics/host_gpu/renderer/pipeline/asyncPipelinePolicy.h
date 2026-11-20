#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_ASYNCPIPELINEPOLICY_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_ASYNCPIPELINEPOLICY_H_

#include <algorithm>
#include <cstdint>
#include <utility>

namespace Libs::Graphics {

// KYTY_ASYNC_PIPELINE: guest frames a draw may be skipped while its pipeline is created; then it
// waits like any miss. Bounds a hole (missing geometry) to ~1-2 s even when a creation backlog
// would keep a pipeline queued for much longer.
inline constexpr uint64_t ASYNC_PIPELINE_SKIP_FRAMES = 30;
inline constexpr uint64_t ASYNC_PIPELINE_NOT_SKIPPED = UINT64_MAX;

// `first_skip`: guest frame of the key's first skip (ASYNC_PIPELINE_NOT_SKIPPED: none yet, set to
// `frame`). True while a draw of the key may still be skipped at `frame`. `uncapped`
// (KYTY_ASYNC_TESS, default on, tessellation draws): skipped until created, as a creation of seconds made
// inline at the cap froze the draw thread for all of it.
[[nodiscard]] constexpr bool AsyncPipelineMaySkip(uint64_t& first_skip, uint64_t frame,
                                                  bool uncapped = false) {
	if (first_skip == ASYNC_PIPELINE_NOT_SKIPPED) {
		first_skip = frame;
	}
	return uncapped || frame - first_skip < ASYNC_PIPELINE_SKIP_FRAMES;
}

// The next job for a pipeline worker: the skipped keys first, in first-skip order (the nearest cap
// first), then the predicted ones. LIFO among skipped keys left the oldest queued until their cap,
// and the draw then created it inline. `uncapped(job)` (KYTY_ASYNC_TESS, tessellation): such skipped
// keys go after the capped ones, as they have no cap to miss; ahead of them, their creations of
// seconds held every worker while capped keys reached their cap. One of the queues is not empty.
template <class Queue, class Uncapped>
[[nodiscard]] constexpr typename Queue::value_type AsyncPipelinePop(Queue& skipped, Queue& predicted,
                                                                    Uncapped uncapped) {
	const auto capped = std::find_if_not(skipped.begin(), skipped.end(), uncapped);
	auto&      from   = skipped.empty() ? predicted : skipped;
	const auto next   = capped != skipped.end() ? capped : from.begin();
	auto       job    = std::move(*next);
	from.erase(next);
	return job;
}

} // namespace Libs::Graphics

#endif /* EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_ASYNCPIPELINEPOLICY_H_ */
