#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_CACHE_STREAMREUSEPOLICY_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_CACHE_STREAMREUSEPOLICY_H_

#include <array>
#include <cstdint>

// KYTY_STREAM_REUSE: stream copies of small CPU-dirty ranges, reused while the guest cannot have
// changed the bytes. Between two guest sync points the guest has no sign that the GPU read a range,
// so a copy taken earlier in that span is as valid as a new one. A copy is reused for the same
// (address, size, ring) while all of these still match:
// - the packet epoch: PM4 packets that are not register writes or direct draws, submission starts
//   and guest sync points (carried by the record of a queued draw);
// - the store epoch: emulator stores into guest memory (backing writes, EOP labels, query results);
// - the command buffer (scheduler tick): the ring keeps the range only for the buffer that used it;
// - the ring lap: a wrap may reuse the range for newer data.
namespace Libs::Graphics::StreamReuse {

struct Key {
	uint64_t    vaddr = 0;
	uint64_t    size  = 0;
	const void* ring  = nullptr;
};

struct Stamp {
	uint64_t packet_epoch = 0;
	uint64_t store_epoch  = 0;
	uint64_t tick         = 0;
	uint64_t lap          = 0;
};

enum class Result : uint32_t { Hit, Empty, PacketEpoch, StoreEpoch, Tick, Lap, Count };

class Table {
public:
	static constexpr uint32_t Slots = 8192;

	// Hit: offset is the ring offset of the earlier copy.
	[[nodiscard]] Result Find(const Key& key, const Stamp& now, uint64_t& offset) const noexcept {
		const auto& e = m_entries[Slot(key)];
		if (!e.valid || e.key.vaddr != key.vaddr || e.key.size != key.size || e.key.ring != key.ring) {
			return Result::Empty;
		}
		if (e.stamp.packet_epoch != now.packet_epoch) return Result::PacketEpoch;
		if (e.stamp.store_epoch != now.store_epoch) return Result::StoreEpoch;
		if (e.stamp.tick != now.tick) return Result::Tick;
		if (e.stamp.lap != now.lap) return Result::Lap;
		offset = e.offset;
		return Result::Hit;
	}
	void Store(const Key& key, const Stamp& stamp, uint64_t offset) noexcept {
		m_entries[Slot(key)] = {key, stamp, offset, true};
	}
	void Forget(const Key& key) noexcept { m_entries[Slot(key)].valid = false; }

private:
	struct Entry {
		Key      key;
		Stamp    stamp;
		uint64_t offset = 0;
		bool     valid  = false;
	};
	[[nodiscard]] static uint32_t Slot(const Key& key) noexcept {
		const uint64_t h = (key.vaddr >> 4u) ^ (key.size * 0x9e3779b97f4a7c15ull);
		return static_cast<uint32_t>((h ^ (h >> 29u)) & (Slots - 1));
	}
	std::array<Entry, Slots> m_entries {};
};

} // namespace Libs::Graphics::StreamReuse

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_CACHE_STREAMREUSEPOLICY_H_
