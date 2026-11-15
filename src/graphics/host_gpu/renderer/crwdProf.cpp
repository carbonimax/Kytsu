#include "graphics/host_gpu/renderer/crwdProf.h"

#include "graphics/host_gpu/renderer/perfStats.h"

#include <xxhash.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <vector>

namespace Libs::Graphics::CrwdProf {

namespace {

// KYTY_CRWD_PROF=3 log, little-endian, records back to back. `ns` = steady_clock (QueryPerformanceCounter
// on Windows, the clock of Python's perf_counter).
// Frame: FrameHeader, then `keys` KeyRecords (every (VS, PS, indirect) drawn in that frame).
// Present: PresentRecord.
struct FrameHeader {
	uint32_t magic = 0x33465243; // "CRF3"
	uint32_t keys  = 0;
	uint64_t frame = 0; // index, from 0
	uint64_t ns    = 0;
	uint64_t write_ns = 0; // time the previous frame's record took to write (log overhead)
};
struct KeyRecord {
	uint64_t vs = 0, ps = 0;
	uint64_t instances = 0; // direct: instance sum; indirect: record count sum
	uint64_t params    = 0; // order-independent sum of mixed per-draw params
	uint32_t draws = 0, indirect = 0;
};
struct PresentRecord {
	uint32_t magic = 0x33505243; // "CRP3"
	uint32_t pad   = 0;
	uint64_t presented = 0, frame = 0, ns = 0;
};
static_assert(sizeof(FrameHeader) == 32 && sizeof(KeyRecord) == 40 && sizeof(PresentRecord) == 32);

uint64_t NowNs() {
	return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
	                                 std::chrono::steady_clock::now().time_since_epoch())
	                                 .count());
}

struct Tables {
	std::mutex  mutex;
	ToggleTable draw {true};  // direct draws by (VS, PS): draws and instance sum
	ToggleTable drwi {true};  // indirect draws by (VS, PS): draws (instances are in GPU records)
	// Indirect packets by their order in the frame (their arguments sit in rings: new addresses each frame).
	ToggleTable icpu {true}; // read by the CPU: count and records
	ToggleTable igpu {true}; // read by the GPU: count only (records not observed)
	uint64_t    icpu_next = 0, igpu_next = 0;
	ToggleTable side {false}; // (address, size) of GPU-written bytes read back: content
	uint64_t    frames = 0;
	uint64_t    side_bytes = 0;
	// KYTY_CRWD_PROF=3
	std::unordered_map<uint64_t, KeyRecord> log;
	std::vector<KeyRecord>                  log_out;
	std::FILE*                              file     = nullptr;
	uint64_t                                write_ns = 0;
	std::atomic<uint64_t>                   frame_index {0};
};

Tables& Get() {
	static Tables tables;
	return tables;
}

uint64_t Mix(uint64_t a, uint64_t b) {
	return (a ^ (b + 0x9E3779B97F4A7C15ull + (a << 6u) + (a >> 2u))) * 0xBF58476D1CE4E5B9ull;
}

} // namespace

bool Enabled() noexcept {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_CRWD_PROF");
		return PerfStats::Enabled() && value != nullptr && std::atoi(value) >= 2;
	}();
	return enabled;
}

bool LogEnabled() noexcept {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_CRWD_PROF");
		return Enabled() && std::atoi(value) >= 3;
	}();
	return enabled;
}

// Called with the mutex held.
static std::FILE* LogFile(Tables& t) {
	if (t.file == nullptr) {
		const char* path = std::getenv("KYTY_CRWD_LOG");
		t.file           = std::fopen(path != nullptr && *path != 0 ? path : "crwd3.bin", "wb");
		if (t.file != nullptr) std::setvbuf(t.file, nullptr, _IOFBF, size_t {1} << 20u);
	}
	return t.file;
}

void NoteDraw(uint64_t vs_hash, uint64_t ps_hash, uint32_t instances, bool indirect, uint64_t params) noexcept {
	auto&           t = Get();
	std::lock_guard lock(t.mutex);
	const auto      key = Mix(vs_hash, ps_hash);
	// Draw count in the high bits, instance sum in the low 40.
	(indirect ? t.drwi : t.draw)
	    .AddSum(key, vs_hash, ps_hash, (uint64_t {1} << 40u) + (indirect ? 0 : instances));
	if (LogEnabled()) {
		auto& r = t.log[key ^ (indirect ? 1u : 0u)];
		r.vs    = vs_hash;
		r.ps    = ps_hash;
		r.draws++;
		r.indirect = indirect ? 1 : 0;
		r.instances += instances;
		// A sum: the same draws submitted in another order give the same value.
		r.params += Mix(params, instances);
	}
}

uint64_t CurrentFrame() noexcept {
	return Get().frame_index.load(std::memory_order_relaxed);
}

void NotePresent(uint64_t presented, uint64_t frame) noexcept {
	auto&           t = Get();
	std::lock_guard lock(t.mutex);
	if (auto* f = LogFile(t)) {
		const PresentRecord r {.presented = presented, .frame = frame, .ns = NowNs()};
		std::fwrite(&r, sizeof(r), 1, f);
	}
}

void NoteIndirect(uint64_t args_addr, uint32_t count, bool cpu, uint64_t records_hash) noexcept {
	auto&           t = Get();
	std::lock_guard lock(t.mutex);
	if (cpu) {
		t.icpu.AddSeq(t.icpu_next++, args_addr, count, Mix(count, records_hash));
	} else {
		t.igpu.AddSeq(t.igpu_next++, args_addr, count, count);
	}
}

void NoteSideRead(uint64_t address, uint64_t size, const void* data) noexcept {
	const auto      hash = XXH3_64bits(data, static_cast<size_t>(size));
	auto&           t    = Get();
	std::lock_guard lock(t.mutex);
	t.side.AddSeq(Mix(address, size), address, size, hash);
	t.side_bytes += size;
}

void EndFrame() noexcept {
	auto&           t = Get();
	std::lock_guard lock(t.mutex);
	t.draw.EndFrame();
	t.drwi.EndFrame();
	t.icpu.EndFrame();
	t.igpu.EndFrame();
	t.side.EndFrame();
	t.icpu_next = t.igpu_next = 0;
	t.frames++;
	if (LogEnabled()) {
		const auto start = NowNs();
		const auto frame = t.frame_index.load(std::memory_order_relaxed);
		if (auto* f = LogFile(t)) {
			t.log_out.clear();
			for (const auto& [key, r]: t.log) t.log_out.push_back(r);
			const FrameHeader h {.keys     = static_cast<uint32_t>(t.log_out.size()),
			                     .frame    = frame,
			                     .ns       = start,
			                     .write_ns = t.write_ns};
			std::fwrite(&h, sizeof(h), 1, f);
			std::fwrite(t.log_out.data(), sizeof(KeyRecord), t.log_out.size(), f);
			// shortcut: no flush at exit, up to 32 frames (~1.5 s) lost at the end of a session.
			if ((frame & 31u) == 31u) std::fflush(f);
		}
		t.log.clear();
		t.write_ns = NowNs() - start;
	}
	t.frame_index.store(t.frame_index.load(std::memory_order_relaxed) + 1, std::memory_order_relaxed);
}

void Report(std::string& out, std::string& hot) noexcept {
	struct Top {
		char     tag;
		uint64_t a, b;
		uint32_t evaluated, toggles;
	};
	auto&            t = Get();
	std::lock_guard  lock(t.mutex);
	std::vector<Top> top;
	char             text[256];
	std::snprintf(text, sizeof(text), " frames=%" PRIu64 " side_bytes=%" PRIu64, t.frames, t.side_bytes);
	out += text;
	// A per-frame value: draws and instances for the draw tables, else the low bits of the hash; - = absent.
	const auto value = [&](bool present, uint64_t v, bool sum) {
		if (!present) return std::string("-");
		if (sum) {
			std::snprintf(text, sizeof(text), "d%" PRIu64 "i%" PRIu64, v >> 40u, v & ((uint64_t {1} << 40u) - 1));
		} else {
			std::snprintf(text, sizeof(text), "%08" PRIx64, v & 0xFFFFFFFFu);
		}
		return std::string(text);
	};
	const auto table = [&](const char* name, char tag, ToggleTable& table) {
		uint64_t evaluated = 0, changes = 0, toggles = 0, hot_keys = 0, max_pct = 0;
		std::vector<const ToggleTable::Entry*> hot_list;
		for (const auto& [key, e]: table.Entries()) {
			evaluated += e.evaluated;
			changes += e.changes;
			toggles += e.toggles;
			// Gate keys: >= 20 % of at least 20 frame pairs.
			if (e.evaluated >= 20) {
				if (e.toggles * 5 >= e.evaluated) {
					hot_keys++;
					hot_list.push_back(&e);
				}
				max_pct = std::max<uint64_t>(max_pct, uint64_t {e.toggles} * 100 / e.evaluated);
			}
			if (e.toggles != 0) {
				top.push_back({tag, e.a, e.b, e.evaluated, e.toggles});
			}
		}
		std::snprintf(text, sizeof(text),
		              " %s=keys:%zu,eval:%" PRIu64 ",chg:%" PRIu64 ",tog:%" PRIu64 ",hot20:%" PRIu64 ",max:%" PRIu64 "%%",
		              name, table.Entries().size(), evaluated, changes, toggles, hot_keys, max_pct);
		out += text;
		if (!hot_list.empty()) {
			// Every gate key (at most 100), with the two values it alternates between.
			std::sort(hot_list.begin(), hot_list.end(), [](const auto* l, const auto* r) { return l->toggles > r->toggles; });
			hot += "PERF crwd2 hot ";
			hot += name;
			hot += ":";
			const bool sum = tag == 'D' || tag == 'd';
			for (size_t i = 0; i < std::min<size_t>(hot_list.size(), 100); i++) {
				const auto& e  = *hot_list[i];
				const auto  v0 = value((e.tog_have & 1u) != 0, e.tog_v0, sum);
				const auto  v1 = value((e.tog_have & 2u) != 0, e.tog_v1, sum);
				std::snprintf(text, sizeof(text), " %016" PRIx64 "/%" PRIx64 ":%u/%u=%u%%[%s|%s]", e.a, e.b, e.toggles,
				              e.evaluated, e.toggles * 100 / e.evaluated, v0.c_str(), v1.c_str());
				hot += text;
			}
			hot += "\n";
		}
		table.Clear();
	};
	table("draw", 'D', t.draw);
	table("drwi", 'd', t.drwi);
	table("icpu", 'I', t.icpu);
	table("igpu", 'G', t.igpu);
	table("side", 'S', t.side);
	const auto count = std::min<size_t>(top.size(), 5);
	std::partial_sort(top.begin(), top.begin() + static_cast<std::ptrdiff_t>(count), top.end(),
	                  [](const Top& l, const Top& r) {
		                  return l.toggles != r.toggles ? l.toggles > r.toggles : l.evaluated < r.evaluated;
	                  });
	out += " top:";
	for (size_t i = 0; i < count; i++) {
		const auto& e = top[i];
		// D, d: vs/ps (d = indirect draws); I, G: last packet address/count at that rank; S: address/size.
		std::snprintf(text, sizeof(text), " %c:%016" PRIx64 "/%" PRIx64 ":%u/%u=%u%%", e.tag, e.a, e.b, e.toggles,
		              e.evaluated, e.toggles * 100 / std::max(e.evaluated, 1u));
		out += text;
	}
	t.frames     = 0;
	t.side_bytes = 0;
}

} // namespace Libs::Graphics::CrwdProf
