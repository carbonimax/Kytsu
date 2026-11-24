#include "graphics/host_gpu/renderer/cache/streamReusePolicy.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>

namespace {

using namespace Libs::Graphics::StreamReuse;

void Check(bool value, const char* message) {
	if (!value) {
		std::fprintf(stderr, "StreamReusePolicyTests: failed: %s\n", message);
		std::abort();
	}
}

} // namespace

int main() {
	auto        table = std::make_unique<Table>();
	int         ring_a = 0;
	int         ring_b = 0;
	const Key   key {0x10000, 256, &ring_a};
	const Stamp stamp {5, 7, 100, 2};
	uint64_t    offset = 0;

	Check(table->Find(key, stamp, offset) == Result::Empty, "empty table misses");
	table->Store(key, stamp, 0x400);
	Check(table->Find(key, stamp, offset) == Result::Hit && offset == 0x400, "same range and stamp hits");

	Check(table->Find({0x10000, 128, &ring_a}, stamp, offset) == Result::Empty, "other size misses");
	Check(table->Find({0x10100, 256, &ring_a}, stamp, offset) == Result::Empty, "other address misses");
	Check(table->Find({0x10000, 256, &ring_b}, stamp, offset) == Result::Empty, "other ring misses");

	Check(table->Find(key, {6, 7, 100, 2}, offset) == Result::PacketEpoch, "packet epoch change misses");
	Check(table->Find(key, {5, 8, 100, 2}, offset) == Result::StoreEpoch, "store epoch change misses");
	Check(table->Find(key, {5, 7, 101, 2}, offset) == Result::Tick, "other command buffer misses");
	Check(table->Find(key, {5, 7, 100, 3}, offset) == Result::Lap, "ring wrap misses");

	// A new copy replaces the old one; Forget drops it.
	table->Store(key, {6, 7, 100, 2}, 0x800);
	Check(table->Find(key, {6, 7, 100, 2}, offset) == Result::Hit && offset == 0x800, "new copy hits");
	Check(table->Find(key, stamp, offset) == Result::PacketEpoch, "old stamp no longer hits");
	table->Forget(key);
	Check(table->Find(key, {6, 7, 100, 2}, offset) == Result::Empty, "forgotten copy misses");

	// A range that shares the slot evicts the other one, never answers for it.
	for (uint64_t i = 1; i < 4 * Table::Slots; i++) {
		const Key other {0x10000 + i * 0x40, 256, &ring_a};
		table->Store(key, stamp, 0x400);
		table->Store(other, stamp, 0xc00);
		const auto result = table->Find(key, stamp, offset);
		Check(result == Result::Empty || (result == Result::Hit && offset == 0x400), "collision never aliases");
	}

	std::printf("StreamReusePolicyTests: ok\n");
	return 0;
}
