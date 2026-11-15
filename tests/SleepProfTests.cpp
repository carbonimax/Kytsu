#include "kernel/sleepProf.h"

#include <cstdio>
#include <cstdlib>

namespace {

using namespace Libs::LibKernel::SleepProf;

void Check(bool value, const char* message) {
	if (!value) {
		std::fprintf(stderr, "SleepProfTests: failed: %s\n", message);
		std::abort();
	}
}

} // namespace

int main() {
	static Slot slots[3];
	for (auto& s: slots) {
		s.ready = true;
	}
	slots[2].ready = false; // claimed, not yet named: skipped

	// Thread 0: two short requests, one at the 100 us edge, one long one.
	slots[0].Add(10000, 500000);   // 10 us -> 490 us over
	slots[0].Add(100000, 500000);  // 100 us -> 400 us over
	slots[0].Add(100001, 600000);  // 100.001 us -> <= 500 us bucket
	slots[0].Add(2000000, 1900000); // woke early: over clamps at 0
	slots[0].Add(0, 7000);         // zero-length: counted apart, not in a bucket
	// Thread 1: more requests, all in the 1 ms bucket.
	for (int i = 0; i < 6; i++) {
		slots[1].Add(1000000, 1500000);
	}
	slots[2].Add(10000, 20000);

	Window out[3];
	size_t used = Take(slots, 3, out);
	Check(used == 2, "two slots slept and are ready");
	Check(out[0].slot == &slots[1] && out[0].total == 6, "busiest first");
	Check(out[0].count[2] == 6 && out[0].over_ns[2] == 6 * 500000ull, "1 ms bucket");
	const Window& w = out[1];
	Check(w.zero == 1 && w.total == 5, "zero-length counted apart");
	Check(w.count[0] == 2 && w.over_ns[0] == 890000ull, "<= 100 us bucket and summed oversleep");
	Check(w.count[1] == 1 && w.over_ns[1] == 499999ull, "<= 500 us bucket");
	Check(w.count[3] == 1 && w.over_ns[3] == 0, "early wake clamps at 0");

	// The window restarts at zero; a slot that did not sleep since is not reported.
	slots[0].Add(50000, 550000);
	used = Take(slots, 3, out);
	Check(used == 1 && out[0].slot == &slots[0], "per-window reset");
	Check(out[0].total == 1 && out[0].count[0] == 1 && out[0].over_ns[0] == 500000ull, "new window only");

	std::printf("SleepProfTests: ok\n");
	return 0;
}
