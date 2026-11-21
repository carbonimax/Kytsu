#include "gpu_blit_shaders/gpu_mesh_indirect_spv.h"
#include "graphics/host_gpu/renderer/image/imageInfo.h"

#include <spirv/unified1/spirv.hpp>

#include <cstdint>
#include <cstdio>
#include <iterator>

namespace {

// sRGB, linear, PQ and linear-nit PQ formats in R and B order; drawn sources go through the
// overlay shader, the others are blitted.
bool VideoOutSourceFlagsAndRouting() {
	struct Case {
		uint64_t format;
		uint32_t flags;
		bool     draw_on_sdr;
		bool     draw_on_hdr;
	};
	bool ok = true;
	for (const auto& c: {Case {0x8000000022000000ull, 0, false, false},
	                     Case {0x8000000000000000ull, 0, false, false},
	                     Case {0x8100000022000000ull, 0, false, false},
	                     Case {0x8100000000000000ull, 0, false, false},
	                     Case {0x8100000622000000ull, 8, true, true},
	                     Case {0x8100000600000000ull, 8, true, true},
	                     Case {0xc001000622000000ull, 8, true, true},
	                     Case {0xc001000600000000ull, 8, true, true},
	                     Case {0x8100070422000000ull, 4, true, false},
	                     Case {0x8100070400000000ull, 4, true, false},
	                     Case {0xc001070722000000ull, 12, true, true},
	                     Case {0xc001070700000000ull, 12, true, true}}) {
		const uint32_t flags = Libs::Graphics::VideoOutSourceFlags(c.format);
		const bool     sdr   = Libs::Graphics::VideoOutSourceNeedsDraw(flags, false);
		const bool     hdr   = Libs::Graphics::VideoOutSourceNeedsDraw(flags, true);
		if (flags != c.flags || sdr != c.draw_on_sdr || hdr != c.draw_on_hdr) {
			std::printf("video out format %016llx: flags=%u draw_on_sdr=%d draw_on_hdr=%d\n",
			            static_cast<unsigned long long>(c.format), flags, sdr, hdr);
			ok = false;
		}
	}
	return ok;
}

// At SPIR-V 1.6 glslang declares a task shader's workgroup size with LocalSizeId, which needs
// maintenance4: the build targets SPIR-V 1.5, which keeps LocalSize.
bool MeshIndirectTaskDeclaresLocalSize() {
	bool local_size    = false;
	bool local_size_id = false;
	for (size_t i = 5; i < std::size(GPU_MESH_INDIRECT_SPV);) {
		const uint32_t word  = GPU_MESH_INDIRECT_SPV[i];
		const uint32_t count = word >> spv::WordCountShift;
		if (count == 0 || i + count > std::size(GPU_MESH_INDIRECT_SPV)) {
			std::printf("gpu_mesh_indirect: malformed instruction at word %zu\n", i);
			return false;
		}
		const auto op = static_cast<spv::Op>(word & spv::OpCodeMask);
		if (count > 2 && (op == spv::OpExecutionMode || op == spv::OpExecutionModeId)) {
			const auto mode = static_cast<spv::ExecutionMode>(GPU_MESH_INDIRECT_SPV[i + 2]);
			local_size |= mode == spv::ExecutionModeLocalSize;
			local_size_id |= mode == spv::ExecutionModeLocalSizeId;
		}
		i += count;
	}
	if (!local_size || local_size_id) {
		std::printf("gpu_mesh_indirect: LocalSize=%d LocalSizeId=%d\n", local_size, local_size_id);
		return false;
	}
	return true;
}

} // namespace

int main() {
	if (!VideoOutSourceFlagsAndRouting()) {
		std::printf("FAIL VideoOutFormat.SourceFlagsAndRouting\n");
		return 1;
	}
	std::printf("PASS VideoOutFormat.SourceFlagsAndRouting\n");
	if (!MeshIndirectTaskDeclaresLocalSize()) {
		std::printf("FAIL GpuBlitShaders.MeshIndirectTaskDeclaresLocalSize\n");
		return 1;
	}
	std::printf("PASS GpuBlitShaders.MeshIndirectTaskDeclaresLocalSize\n");
	return 0;
}
