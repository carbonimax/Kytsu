#ifndef EMULATOR_SRC_GRAPHICS_SHADER_TESSPRETRANSLATEPOLICY_H_
#define EMULATOR_SRC_GRAPHICS_SHADER_TESSPRETRANSLATEPOLICY_H_

namespace Libs::Graphics {

// KYTY_TESS_PRETRANSLATE (ShaderPretranslate::TakeTess): what a LS, HS or DS program miss does.
//   Ready:   take the worker's translation; the draw thread only emits it.
//   Pending: a worker has the key queued or running; the draw is skipped.
//   Queue:   queue the key for a worker; the draw is skipped.
//   Inline:  translate on the draw thread, as without the option.
enum class TessPretranslateTake { Ready, Pending, Queue, Inline };

// `held`: the pool holds a job of this key; `ready`: it is translated; `same_code`: its code words
// are the draw's; `seen`: the key was queued before (taken, failed, evicted or its worker retired);
// `stopped`: pretranslation stopped. A key is queued once: never again after a failure or eviction,
// so a draw is never skipped for good.
[[nodiscard]] constexpr TessPretranslateTake TessPretranslateDecide(bool held, bool ready, bool same_code,
                                                                    bool seen, bool stopped) {
	if (held) {
		return !ready ? TessPretranslateTake::Pending
		       : same_code ? TessPretranslateTake::Ready
		                   : TessPretranslateTake::Inline;
	}
	return seen || stopped ? TessPretranslateTake::Inline : TessPretranslateTake::Queue;
}

} // namespace Libs::Graphics

#endif /* EMULATOR_SRC_GRAPHICS_SHADER_TESSPRETRANSLATEPOLICY_H_ */
