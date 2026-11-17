#include "graphics/shader/tessPretranslatePolicy.h"

#include <cstdio>

namespace {

using Libs::Graphics::TessPretranslateDecide;
using Libs::Graphics::TessPretranslateTake;
using T = TessPretranslateTake;

// Evaluated by the compiler (clang-cl /Zs too). Arguments: held, ready, same_code, seen, stopped.
// A new key is queued, and its draw skipped while a worker has it.
static_assert(TessPretranslateDecide(false, false, false, false, false) == T::Queue);
static_assert(TessPretranslateDecide(true, false, true, true, false) == T::Pending);
static_assert(TessPretranslateDecide(true, true, true, true, false) == T::Ready);
// The code changed since the job was queued: translated inline.
static_assert(TessPretranslateDecide(true, true, false, true, false) == T::Inline);
// A key out of the pool after it was queued (taken, failed, evicted, retired worker) is never
// queued again: a draw cannot stay skipped.
static_assert(TessPretranslateDecide(false, false, false, true, false) == T::Inline);
// Pretranslation stopped (too many retired workers): everything inline.
static_assert(TessPretranslateDecide(false, false, false, false, true) == T::Inline);
// A job still held keeps its draw pending even once stopped (its worker finishes it).
static_assert(TessPretranslateDecide(true, false, true, true, true) == T::Pending);
// After a deferred stage of the same draw: a new key is still queued, nothing is taken or inline.
using Libs::Graphics::TessPretranslatePeek;
static_assert(TessPretranslatePeek(T::Queue) == T::Queue);
static_assert(TessPretranslatePeek(T::Ready) == T::Pending);
static_assert(TessPretranslatePeek(T::Pending) == T::Pending);
static_assert(TessPretranslatePeek(T::Inline) == T::Pending);

} // namespace

int main() {
	std::printf("TessPretranslatePolicyTests: ok\n");
	return 0;
}
