#include "graphics/host_gpu/renderer/pipeline/descriptors.h"

#include "common/alignment.h"
#include "graphics/host_gpu/renderer/mipStats.h"
#include "common/assert.h"
#include "common/common.h"
#include "common/emulatorConfig.h"
#include "common/file.h"
#include "common/logging/log.h"
#include "common/onceValue.h"
#include "common/profiler.h"
#include "common/stringUtils.h"
#include "common/threads.h"
#include "graphics/guest_gpu/gpu_defs.h"
#include "graphics/guest_gpu/gpu_format.h"
#include "graphics/guest_gpu/graphicsRun.h"
#include "graphics/guest_gpu/hardwareContext.h"
#include "graphics/guest_gpu/tile.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/colorRenderTarget.h"
#include "graphics/host_gpu/renderer/debug.h"
#include "graphics/host_gpu/renderer/depthRenderTarget.h"
#include "graphics/host_gpu/renderer/perfStats.h"
#include "graphics/host_gpu/renderer/image/imageView.h"
#include "graphics/host_gpu/renderer/image/textureCommon.h"
#include "graphics/host_gpu/renderer/pipeline/shaderResourceBarrier.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "graphics/host_gpu/vulkanCommon.h"
#include "graphics/shader/recompiler/backend/spirv/SpirvEmitter.h"
#include "graphics/shader/recompiler/ir/ShaderIR.h"
#include "graphics/shader/recompiler/ir/passes/ResourceMaterialization.h"
#include "graphics/shader/shader.h"
#include "graphics/guest_gpu/gpuCapture.h"
#include "kernel/memory.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fmt/format.h>
#include <limits>
#include <memory>
#include <mutex>
#include <set>
#include <span>
#include <vector>

namespace Libs::Graphics {

namespace {

using BindingKind = ShaderRecompiler::IR::DescriptorBindingKind;

// The shader whose images PrepareBindings resolves (0 elsewhere), for the S4 counter.
thread_local uint64_t t_binding_shader_hash = 0;

// shortcut: measurement only (KYTY_PERF_STATS=1), no behaviour change; remove once the
// depth-compare rule for color formats is decided. Every depth-compare bind counts; one that ends
// on a color image with a guest format outside DEPTH_FORMAT_POLICIES keeps a color view with a
// comparison sampler, which Vulkan leaves undefined. Logs each new (shader, guest format) pair
// of those, and the totals at each power of two of all depth-compare binds.
void NoteDepthCompareBind(Prospero::BufferFormat guest_format, vk::Format image_format,
                          bool color_image) {
	if (!PerfStats::Enabled()) {
		return;
	}
	static std::mutex                              mutex;
	static std::set<std::pair<uint64_t, uint32_t>> seen;
	static uint64_t                                binds = 0, outside = 0;
	const auto format = static_cast<uint32_t>(guest_format);
	const auto* policy = FindGuestDepthFormatPolicy(guest_format);
	const bool  hit    = color_image &&
	                 (policy == nullptr || policy->depth_attachment_format == vk::Format::eUndefined);
	std::lock_guard lock(mutex);
	++binds;
	if (hit) {
		++outside;
		if (seen.emplace(t_binding_shader_hash, format).second) {
			std::printf("PERF arifix-s4 shader=0x%016" PRIx64 " guest_format=%u image_format=%d\n",
			            t_binding_shader_hash, format, static_cast<int>(image_format));
		}
	}
	if (std::has_single_bit(binds)) {
		std::printf("PERF arifix-s4-seen compare_binds=%" PRIu64 " outside_table=%" PRIu64
		            " pairs=%zu\n",
		            binds, outside, seen.size());
	}
}

} // namespace

vk::DescriptorType NativeDescriptorType(BindingKind kind) {
	const auto image_class = ShaderRecompiler::IR::ImageBindingResourceClass(kind);
	if (image_class == ShaderRecompiler::IR::ImageResourceClass::Sampled) {
		return vk::DescriptorType::eSampledImage;
	}
	if (image_class == ShaderRecompiler::IR::ImageResourceClass::Storage) {
		return vk::DescriptorType::eStorageImage;
	}
	switch (kind) {
		case BindingKind::Samplers: return vk::DescriptorType::eSampler;
		case BindingKind::Buffers:
		case BindingKind::Gds:
		case BindingKind::BdaPagetable:
		case BindingKind::FaultBuffer:
		case BindingKind::FlattenedSrt:
		case BindingKind::ShaderData:
		case BindingKind::SharedMemory: return vk::DescriptorType::eStorageBuffer;
		case BindingKind::Count: EXIT("invalid native descriptor binding kind");
	}
	EXIT("invalid native descriptor binding kind");
}

vk::DescriptorImageInfo MakeImageInfo(const TextureBinding& texture, uint32_t element) {
	if (!texture.image_id) {
		EXIT_IF(texture.image_view != nullptr || !texture.mip_views.empty());
		return {nullptr, nullptr, vk::ImageLayout::eGeneral};
	}
	vk::ImageView view = nullptr;
	if (texture.mip_views.empty()) {
		if (element == 0u) {
			view = texture.image_view;
		}
	} else if (element < texture.mip_views.size()) {
		view = texture.mip_views[element];
	}
	EXIT_IF(view == nullptr || texture.layout == vk::ImageLayout::eUndefined);
	return {nullptr, view, texture.layout};
}

static Prospero::ImageType TextureType(const ShaderTextureResource& descriptor) {
	const auto type = descriptor.Type();
	return type == Prospero::ImageType::kCube ? Prospero::ImageType::kColor2DArray : type;
}

// 1D textures are 2D images one texel high, so they share 2D surfaces at their address.
static Prospero::ImageType TextureBaseType(Prospero::ImageType type) {
	switch (type) {
		case Prospero::ImageType::kColor1D:
		case Prospero::ImageType::kColor1DArray:
		case Prospero::ImageType::kColor2DArray:
		case Prospero::ImageType::kColor2DMsaa:
		case Prospero::ImageType::kColor2DMsaaArray: return Prospero::ImageType::kColor2D;
		default: return type;
	}
}

static bool IsMultisampledTexture(Prospero::ImageType type) {
	return type == Prospero::ImageType::kColor2DMsaa ||
	       type == Prospero::ImageType::kColor2DMsaaArray;
}

static vk::DescriptorBufferInfo
NativeStorageBuffer(RenderContext& context, const PreparedBindings::BufferSource& source,
                    const ShaderRecompiler::IR::BufferResource& resource, uint32_t& buffer_offset,
                    const Buffer** obtained = nullptr) {
	buffer_offset = 0;

	const auto& [address, size, id] = source;
	const auto& graphics = context.GetGraphics();
	const auto  profile  = ShaderRecompiler::Spirv::Profile {
	     .storage_bounds = static_cast<ShaderRecompiler::Spirv::StorageBounds>(graphics.storage_bounds),
	     .storage_bounds_alignment = graphics.storage_bounds_alignment};
	const bool device_bounds = profile.storage_bounds == ShaderRecompiler::Spirv::StorageBounds::Device;
	// A base in guest page 0 is a null or garbage descriptor.
	if (address < BufferCache::CACHING_PAGESIZE || size == 0) {
		// Without shader checks only a null descriptor reads zero and drops writes.
		if (device_bounds) return {nullptr, 0, vk::WholeSize};
		return {context.GetBufferCache().GetBuffer(NULL_BUFFER_ID).Handle(), 0, 16};
	}
	const auto  alignment = graphics.StorageMinAlignment();
	if (size > graphics.GetPhysicalDeviceProperties().limits.maxStorageBufferRange) {
		EXIT("storage buffer range is unsupported\n");
	}
	const LateRead::Scope late_read(LateRead::Site::Descriptor);
	auto [buffer, offset] = context.GetBufferCache().ObtainBuffer(
	    address, size, resource.written,
	    resource.formatted || (resource.read && Config::SyncRawImageBuffersEnabled()), id);
	if (obtained != nullptr) {
		*obtained = buffer;
	}
	const auto aligned_offset = Common::AlignDown(offset, alignment);
	const auto adjustment     = offset - aligned_offset;
	const auto max_range      = graphics.GetPhysicalDeviceProperties().limits.maxStorageBufferRange;
	if (adjustment >= 256 || size > max_range - adjustment) {
		EXIT("storage buffer offset adjustment is unsupported\n");
	}
	buffer_offset = static_cast<uint32_t>(adjustment);
	// The shader still compares against size + adjustment; see Spirv::StorageDeviceRange.
	const auto range = ShaderRecompiler::Spirv::StorageDeviceRange(profile, size + adjustment, resource.subword);
	const vk::DescriptorBufferInfo result = range == 0
	    ? vk::DescriptorBufferInfo {nullptr, 0, vk::WholeSize}
	    : vk::DescriptorBufferInfo {buffer->Handle(), aligned_offset, range};
	if (resource.written) {
		context.GetTextureCache().InvalidateMemoryFromGPU(address, size);
	}
	return result;
}

// Reserved and unsupported T# fields must be zero; perf_mod only scales sampler state and the
// mip-stats counter only counts sampling.
static bool IsCanonicalTextureEncoding(const ShaderTextureResource& descriptor, bool r128) {
	constexpr uint32_t field1_reserved_mask = 1u << 29u;
	constexpr uint32_t field2_reserved_mask = 0xf0003000u;
	const uint32_t     field3_expected = descriptor.DstSelXYZW() |
	                                     (static_cast<uint32_t>(descriptor.BaseLevel()) << 12u) |
	                                     (static_cast<uint32_t>(descriptor.LastLevel()) << 16u) |
	                                     (static_cast<uint32_t>(descriptor.TileMode()) << 20u) |
	                                     (static_cast<uint32_t>(descriptor.Type()) << 28u);
	const uint32_t     field4_expected = descriptor.Depth() | (descriptor.BaseArray5() << 16u);
	const uint32_t     field5_expected = (static_cast<uint32_t>(descriptor.PerfMod5()) << 20u) |
	                                     (uint32_t {descriptor.MipStatsCntEn()} << 25u) |
	                                     (static_cast<uint32_t>(descriptor.MaxMip()) << 4u);
	if ((descriptor.fields[1] & field1_reserved_mask) != 0 ||
	    (descriptor.fields[2] & field2_reserved_mask) != 0 || descriptor.fields[3] != field3_expected) {
		return false;
	}
	if (r128) {
		return descriptor.fields[4] == 0 && descriptor.fields[5] == 0 && descriptor.fields[6] == 0 &&
		       descriptor.fields[7] == 0;
	}
	return descriptor.fields[4] == field4_expected && descriptor.fields[5] == field5_expected;
}

bool IsSupportedDepthTextureEncoding(const ShaderTextureResource& descriptor, bool r128) {
	if (!IsCanonicalTextureEncoding(descriptor, r128)) {
		return false;
	}
	if (r128) {
		return true;
	}
	if (descriptor.MsaaDepth() && !IsMultisampledTexture(descriptor.Type())) {
		return false;
	}
	// Bits 0-7 name the mip-stats counter. A clear COMPRESSION_EN disables the metadata and its
	// settings. MaxUncompBlkSize/MaxCompBlkSize (bits 15-18) size DCC blocks; HTILE depth ignores them.
	constexpr uint32_t dcc_block_size_mask = 0x00078000u;
	const auto         metadata_control    = descriptor.fields[6] & 0x00ffff00u & ~dcc_block_size_mask;
	constexpr uint32_t pipe_aligned     = 1u << 19u;
	if (!descriptor.MetaCompress()) {
		return true;
	}
	constexpr uint32_t htile_control = pipe_aligned | 1u << 21u;
	const uint32_t expected_control  = htile_control | (descriptor.MsaaDepth() ? (1u << 10u) : 0u);
	const auto     metadata_addr     = descriptor.MetaAddr() << 8u;
	return metadata_control == expected_control && GuestRange {metadata_addr, 1}.Valid() &&
	       (metadata_addr & 0x7fffu) == 0 &&
	       descriptor.TileMode() == Prospero::TileMode::kDepth;
}

static void ValidateSampledDepthBinding(const ShaderRecompiler::IR::ImageResource& resource,
                                        const ShaderTextureResource& descriptor, const Image& image,
                                        vk::Format view_format, uint64_t size) {
	const bool resource_ok = IsSupportedSampledDepthResource(resource);
	// Comparison sampling changes the host format, not the guest metadata encoding.
	const bool encoding_ok = descriptor.TileMode() != Prospero::TileMode::kDepth ||
	                         IsSupportedDepthTextureEncoding(descriptor, resource.r128);
	const bool view_ok =
	    IsSupportedSampledDepthView(image.info.pixel_format, view_format, descriptor.DstSelXYZW());
	if (resource_ok && encoding_ok && view_ok) {
		return;
	}
	const auto descriptor_pitch =
	    TileGetTexturePitch(descriptor.Format(), static_cast<uint32_t>(descriptor.Width5()) + 1u,
	                        descriptor.TileMode());
	EXIT("unsupported sampled depth image: resource=%d encoding=%d view=%d "
	     "class=%u numeric=%u dimension=%u mip_mode=%u read=%d written=%d atomic=%d compare=%d "
	     "guest_format=%u swizzle=0x%03x image_format=%d view_format=%d image_layers=%u "
	     "descriptor_type=%u base_array=%u depth=%u descriptor_pitch=%u target_pitch=%u "
	     "addr=0x%016" PRIx64 " size=0x%016" PRIx64
	     " dwords=%08x,%08x,%08x,%08x,%08x,%08x,%08x,%08x\n",
	     resource_ok, encoding_ok, view_ok,
	     static_cast<uint32_t>(resource.resource_class),
	     static_cast<uint32_t>(resource.numeric_class), static_cast<uint32_t>(resource.dimension),
	     static_cast<uint32_t>(resource.mip_mode), resource.read, resource.written, resource.atomic,
	     resource.depth_compare, static_cast<uint32_t>(descriptor.Format()),
	     descriptor.DstSelXYZW(), static_cast<int>(image.info.pixel_format),
	     static_cast<int>(view_format), image.info.resources.layers,
	     static_cast<uint32_t>(descriptor.Type()), descriptor.BaseArray5(), descriptor.Depth(),
	     descriptor_pitch, image.info.pitch, descriptor.Base40(), size, descriptor.fields[0],
	     descriptor.fields[1], descriptor.fields[2], descriptor.fields[3], descriptor.fields[4],
	     descriptor.fields[5], descriptor.fields[6], descriptor.fields[7]);
}

static bool IsSupportedStorageTextureDescriptor(const ShaderRecompiler::IR::ImageResource& resource,
                                                const ShaderTextureResource& descriptor) {
	const auto tile              = descriptor.TileMode();
	const bool is_color_1d       = descriptor.Type() == Prospero::ImageType::kColor1D;
	const bool is_color_1d_array = descriptor.Type() == Prospero::ImageType::kColor1DArray;
	const bool valid_1d_slice =
	    (is_color_1d && descriptor.Depth() == 0 && descriptor.BaseArray5() == 0) ||
	    (is_color_1d_array && descriptor.BaseArray5() <= descriptor.Depth());
	const bool is_1d = resource.dimension == ShaderRecompiler::Decoder::ImageDimension::Dim1D &&
	                   descriptor.Height5() == 0 && valid_1d_slice;
	const bool is_1d_array =
	    resource.dimension == ShaderRecompiler::Decoder::ImageDimension::Dim1DArray &&
	    is_color_1d_array && descriptor.Height5() == 0 &&
	    descriptor.BaseArray5() <= descriptor.Depth();
	const bool is_color_2d       = descriptor.Type() == Prospero::ImageType::kColor2D;
	const bool is_color_2d_array = descriptor.Type() == Prospero::ImageType::kColor2DArray;
	const bool valid_2d_slice =
	    (is_color_2d && descriptor.Depth() == 0 && descriptor.BaseArray5() == 0) ||
	    (is_color_2d_array && descriptor.BaseArray5() <= descriptor.Depth());
	const bool is_2d =
	    resource.dimension == ShaderRecompiler::Decoder::ImageDimension::Dim2D && valid_2d_slice;
	// Storage cube coordinates address individual faces, including partial cube views.
	const bool is_cube = resource.cube && descriptor.Type() == Prospero::ImageType::kCube &&
	                     descriptor.Width5() == descriptor.Height5() &&
	                     descriptor.BaseArray5() <= descriptor.Depth();
	const bool is_2d_array =
	    resource.dimension == ShaderRecompiler::Decoder::ImageDimension::Dim2DArray &&
	    ((!resource.cube && is_color_2d_array && descriptor.BaseArray5() <= descriptor.Depth()) ||
	     is_cube);
	const bool is_3d = resource.dimension == ShaderRecompiler::Decoder::ImageDimension::Dim3D &&
	                   descriptor.Type() == Prospero::ImageType::kColor3D &&
	                   descriptor.BaseArray5() == 0;
	const bool is_2d_msaa =
	    resource.dimension == ShaderRecompiler::Decoder::ImageDimension::Dim2DMsaa &&
	    descriptor.Type() == Prospero::ImageType::kColor2DMsaa && descriptor.Depth() == 0 &&
	    descriptor.BaseArray5() == 0;
	TileTextureBlockLayout tile_layout {};
	bool                   supported_tile = false;
	switch (tile) {
		case Prospero::TileMode::kLinear: supported_tile = true; break;
		case Prospero::TileMode::kDepth:
			supported_tile =
			    !resource.read && !Prospero::IsFmaskTextureFormat(descriptor.Format()) &&
			    (is_2d || is_2d_array) &&
			    TileGetTextureBlockLayout(descriptor.Format(), tile, false, tile_layout);
			break;
		case Prospero::TileMode::kStandard256B:
			supported_tile =
			    (is_2d || is_2d_array) &&
			    TileGetTextureBlockLayout(descriptor.Format(), tile, false, tile_layout);
			break;
		case Prospero::TileMode::kStandard4KB:
		case Prospero::TileMode::kStandard64KB:
			supported_tile =
			    TileGetTextureBlockLayout(descriptor.Format(), tile, is_3d, tile_layout);
			break;
		case Prospero::TileMode::kRenderTarget:
			supported_tile =
			    TileGetTextureBlockLayout(descriptor.Format(), tile, false, tile_layout);
			break;
		default: break;
	}
	const auto swizzle = descriptor.DstSelXYZW();
	const bool supported_swizzle =
	    IsValidImageSwizzle(swizzle) &&
	    (swizzle == DstSel(4, 5, 6, 7) || !resource.read || resource.atomic);
	// A 4.8 fixed-point MIN_LOD at or below BASE_LEVEL clamps no level of the view.
	return (is_1d || is_1d_array || is_2d || is_2d_array || is_3d || is_2d_msaa) &&
	       supported_tile && descriptor.BaseLevel() <= descriptor.LastLevel() &&
	       descriptor.MinLod() <= descriptor.BaseLevel() * 256u && supported_swizzle &&
	       descriptor.BCSwizzle() == 0 && !descriptor.MsaaDepth();
}

void ValidateStorageTexture(const ShaderRecompiler::IR::ImageResource& resource,
                            const ShaderTextureResource& descriptor, uint64_t size) {
	const auto format        = descriptor.Format();
	const bool resource_ok   = IsSupportedStorageImageResource(resource);
	const bool descriptor_ok = IsSupportedStorageTextureDescriptor(resource, descriptor);
	const bool encoding_ok   = IsCanonicalTextureEncoding(descriptor, resource.r128);
	const bool uint_resource    = resource.numeric_class == Prospero::TextureNumericClass::Uint;
	const bool raw_sint_storage = uint_resource &&
	    ShaderRecompiler::IR::RawSignedStorageFormat(resource, format) != Prospero::BufferFormat::kInvalid;
	const auto numeric_class =
	    Prospero::SampledTextureNumericClass(Prospero::RemapTextureFormat(format));
	const bool raw_float_atomic = format == Prospero::BufferFormat::k32Float && uint_resource &&
	                              resource.atomic && !resource.atomic64;
	const bool format_ok =
	    raw_sint_storage || raw_float_atomic ||
	    (numeric_class != Prospero::TextureNumericClass::Unsupported &&
	     numeric_class != Prospero::TextureNumericClass::Sint &&
	     uint_resource == (numeric_class == Prospero::TextureNumericClass::Uint) &&
	     (!resource.atomic || format == (resource.atomic64 ? Prospero::BufferFormat::k32_32UInt
	                                                       : Prospero::BufferFormat::k32UInt)));
	if (resource_ok && descriptor_ok && encoding_ok && format_ok && size != 0) {
		return;
	}
	EXIT("unsupported storage texture: resource=%d descriptor=%d encoding=%d format=%d "
	     "class=%u numeric=%u dimension=%u mip_mode=%u atomic=%d compare=%d "
	     "base_level=%u last_level=%u max_mip=%u min_lod=%u base_array=%u bc=%u msaa=%d "
	     "depth_tile_bpe=%u swizzle_ok=%d "
	     "addr=0x%016" PRIx64 " size=0x%016" PRIx64
	     " extent=%ux%ux%u type=%u format=%u tile=%u swizzle=0x%03x read=%d written=%d "
	     "dwords=%08x,%08x,%08x,%08x,%08x,%08x,%08x,%08x\n",
	     resource_ok, descriptor_ok, encoding_ok, format_ok,
	     static_cast<uint32_t>(resource.resource_class),
	     static_cast<uint32_t>(resource.numeric_class), static_cast<uint32_t>(resource.dimension),
	     static_cast<uint32_t>(resource.mip_mode), resource.atomic, resource.depth_compare,
	     descriptor.BaseLevel(), descriptor.LastLevel(), descriptor.MaxMip(), descriptor.MinLod(),
	     descriptor.BaseArray5(), descriptor.BCSwizzle(), descriptor.MsaaDepth(),
	     Prospero::RenderTargetBytesPerElement(format),
	     IsValidImageSwizzle(descriptor.DstSelXYZW()), descriptor.Base40(), size,
	     static_cast<uint32_t>(descriptor.Width5()) + 1u,
	     static_cast<uint32_t>(descriptor.Height5()) + 1u,
	     static_cast<uint32_t>(descriptor.Depth()) + 1u, static_cast<uint32_t>(descriptor.Type()),
	     static_cast<uint32_t>(format), static_cast<uint32_t>(descriptor.TileMode()),
	     descriptor.DstSelXYZW(), resource.read, resource.written, descriptor.fields[0],
	     descriptor.fields[1], descriptor.fields[2], descriptor.fields[3], descriptor.fields[4],
	     descriptor.fields[5], descriptor.fields[6], descriptor.fields[7]);
}

// sRGB views decode before filtering; a sampler that skips degamma reads through a UNorm view.
static vk::Format SampledViewFormat(const GraphicContext& graphics, Prospero::BufferFormat format,
                                    vk::Format pixel_format, bool skip_degamma) {
	if (skip_degamma) return SrgbUnormViewFormat(pixel_format);
	switch (format) {
		case Prospero::BufferFormat::k8Srgb:
			return graphics.r8_srgb_views ? vk::Format::eR8Srgb : pixel_format;
		case Prospero::BufferFormat::k8_8Srgb:
			return graphics.r8g8_srgb_views ? vk::Format::eR8G8Srgb : pixel_format;
		default: return pixel_format;
	}
}

static ImageViewInfo TextureViewInfo(const ShaderRecompiler::IR::ImageResource& resource,
                                     const ShaderTextureResource& descriptor, vk::Format format,
                                     const SurfaceFormatInfo& surface_format, bool storage,
                                     uint32_t view_levels, uint32_t image_layers) {
	ImageViewInfo view {};
	view.format      = format;
	view.aspect      = vk::ImageAspectFlagBits::eColor;
	view.base_level  = descriptor.BaseLevel();
	view.level_count = view_levels;
	if (descriptor.MinLod() > descriptor.LastLevel() * 256u) {
		EXIT("texture minimum LOD exceeds last mip level: min_lod=%u last_level=%u\n",
		     descriptor.MinLod(), descriptor.LastLevel());
	}
	const auto base_lod = view.base_level * 256u;
	if (descriptor.MinLod() > base_lod) {
		view.min_lod = descriptor.MinLod() - base_lod;
	}
	if (resource.sample_min_lod != 0u) {
		// The shader clamps the LOD, so the view must not need VK_EXT_image_view_min_lod.
		EXIT_IF(storage || resource.sample_min_lod != view.min_lod);
		view.min_lod = 0;
	}
	if (resource.view_min_level != 0u) {
		// Unmipmapped samplers read the level the minimum LOD selects: start the view there.
		EXIT_IF(storage || resource.view_min_level >= view.level_count ||
		        resource.view_min_level != view.min_lod / 256u);
		view.base_level += resource.view_min_level;
		view.level_count -= resource.view_min_level;
		view.min_lod = 0;
	}
	view.usage = storage ? vk::ImageUsageFlagBits::eStorage : vk::ImageUsageFlagBits::eSampled;
	view.mapping =
	    storage || resource.conversion_format != Prospero::BufferFormat::kInvalid
	        ? vk::ComponentMapping {}
	        : TextureGetComponentMapping(descriptor.DstSelXYZW(), surface_format.host_to_storage);
	const auto base_layer = descriptor.Type() == Prospero::ImageType::kColor1D ||
	                                descriptor.Type() == Prospero::ImageType::kColor2D
	                            ? 0u
	                            : descriptor.BaseArray5();
	switch (resource.dimension) {
		case ShaderRecompiler::Decoder::ImageDimension::Dim3D:
			view.type        = vk::ImageViewType::e3D;
			view.base_layer  = 0;
			view.layer_count = 1;
			break;
		case ShaderRecompiler::Decoder::ImageDimension::Dim1DArray:
		case ShaderRecompiler::Decoder::ImageDimension::Dim2DArray:
		case ShaderRecompiler::Decoder::ImageDimension::Dim2DMsaaArray:
			view.type       = vk::ImageViewType::e2DArray;
			view.base_layer = descriptor.BaseArray5();
			if (view.base_layer >= image_layers) {
				EXIT("texture array base layer is out of bounds\n");
			}
			view.layer_count = image_layers - view.base_layer;
			break;
		case ShaderRecompiler::Decoder::ImageDimension::Dim1D:
		case ShaderRecompiler::Decoder::ImageDimension::Dim2D:
		case ShaderRecompiler::Decoder::ImageDimension::Dim2DMsaa:
			view.type       = vk::ImageViewType::e2D;
			view.base_layer = base_layer;
			if (view.base_layer >= image_layers) {
				EXIT("texture base layer is out of bounds\n");
			}
			view.layer_count = 1;
			break;
		default: EXIT("unsupported texture view dimension\n");
	}
	return view;
}

static bool ResolveTextureMipViewIn(ImageInfo& info, uint32_t physical_levels, bool metadata,
                                    uint32_t view_levels, uint32_t& base_level,
                                    bool rebase_past_max) {
	if (!info.IsTiled()) {
		return false;
	}
	ImageInfo physical = info;
	// A view entirely past MAX_MIP addresses its own chain starting at its base mip (PPSA02110
	// compresses mip 1 through a T# that declares one level).
	if (rebase_past_max && base_level >= physical_levels) {
		const auto mip  = info.MipExtent(base_level);
		physical.extent = {mip.width, mip.height, info.IsVolume() ? mip.depth : 1u};
		physical_levels = view_levels;
	}
	physical.resources.levels = physical_levels;
	physical.UpdateSize();
	info.UpdateSize();
	const auto same_mip = [&](uint32_t target, uint32_t source) {
		return physical.mip_layout[target] == info.mip_layout[source] &&
		       physical.MipExtent(target) == info.MipExtent(source);
	};
	if (physical.first_tail_level == info.first_tail_level &&
	    physical.tiled_slice_stride == info.tiled_slice_stride &&
	    physical.data.size == info.data.size) {
		bool matches = true;
		for (uint32_t level = 0; level < physical_levels; ++level) {
			matches &= same_mip(level, level);
		}
		if (matches) {
			return true;
		}
	}
	if (metadata ||
	    (info.TransferLayers() > 1 && physical.tiled_slice_stride != info.tiled_slice_stride)) {
		return false;
	}
	// T# addresses the last mip. A view can select the same stored subresources
	// with different mip indices; inaccessible mips need no host representation.
	for (uint32_t base = 0; base + view_levels <= physical_levels; ++base) {
		bool matches = true;
		for (uint32_t i = 0; i < view_levels; ++i) {
			const auto source = base_level + i;
			const auto target = base + i;
			if (!same_mip(target, source) ||
			    (target >= physical.first_tail_level) != (source >= info.first_tail_level)) {
				matches = false;
				break;
			}
		}
		if (matches) {
			info       = std::move(physical);
			base_level = base;
			return true;
		}
	}
	return false;
}

// Fork: a view past MAX_MIP first keeps the pre-rebase resolution (a mip-tail view, e.g. base 6 of
// a 6-level 1920x1080 chain, matches a stored mip of the declared chain); only a view that matches
// nothing there is rebased onto its own chain.
static bool ResolveTextureMipView(ImageInfo& info, uint32_t physical_levels, bool metadata,
                                  uint32_t view_levels, uint32_t& base_level) {
	if (base_level >= physical_levels) {
		ImageInfo kept      = info;
		uint32_t  kept_base = base_level;
		if (ResolveTextureMipViewIn(kept, physical_levels, metadata, view_levels, kept_base,
		                            false)) {
			info       = std::move(kept);
			base_level = kept_base;
			return true;
		}
	}
	return ResolveTextureMipViewIn(info, physical_levels, metadata, view_levels, base_level, true);
}

namespace {

// What ResolveTexture derives from a T# and the shader's image resource alone, before the
// texture cache lookup: the image and view descriptions and the formats the checks after the
// lookup compare.
struct ResolvedTextureDesc {
	TextureCache::ImageDesc desc;
	vk::Format              pixel_format      = vk::Format::eUndefined;
	vk::Format              view_format       = vk::Format::eUndefined;
};

// KYTY_TEXTURE_DESC_MEMO (default 1; 0 = off; 2 = verify): ResolveTexture keeps the
// description it derived for a (resource, T#) pair and reuses it the next time any draw binds
// the same T# through an equal resource, skipping the descriptor decode, the tiled mip layout
// (ImageInfo::UpdateSize) and the view setup. The description is a pure function of those
// inputs; the texture cache lookup and every check against the image found still run per
// bind. Image heap resources (which also depend on the guest mapping) are not kept. 2 = derive
// anyway and exit on any difference with the kept description.
uint32_t TextureDescMemoMode() noexcept {
	static std::atomic<int64_t> mode_once {Common::OnceUnset};
	const auto mode = static_cast<uint32_t>(Common::Once(mode_once, [] {
		const char* value = std::getenv("KYTY_TEXTURE_DESC_MEMO");
		return value == nullptr ? 1u : static_cast<uint32_t>(std::strtoul(value, nullptr, 10));
	}));
	return mode;
}

struct TextureDescMemoEntry {
	bool                                valid = false;
	ShaderRecompiler::IR::DescriptorValue value;
	ShaderRecompiler::IR::ImageResource resource;
	ResolvedTextureDesc                 resolved;
	// The image FindImage gave for resolved.desc (KYTY_FIND_IMAGE_MEMO); reset with the entry.
	TextureCache::FindImageMemo find;
	// KYTY_RESOLVE_INPLACE: the program-owned resource equal to `resource` (PrepareBindings only); a
	// published program's info never changes or moves, so its address stands for its content.
	const ShaderRecompiler::IR::ImageResource* stable = nullptr;
};

// Direct-mapped, one per host thread that resolves textures (the GPU thread, and the
// KYTY_CP_PIPELINE back thread): no lock, no shared state.
constexpr size_t TextureDescMemoSlots = 4096;
thread_local std::unique_ptr<std::array<TextureDescMemoEntry, TextureDescMemoSlots>>
    t_texture_desc_memo;
// Bumped when a RenderExecutor (so a render context) is created or destroyed: the programs and images
// an entry names may be gone, and the next context may reuse their addresses and image ids.
std::atomic<uint64_t> g_texture_desc_memo_generation {0};
thread_local uint64_t t_texture_desc_memo_generation = 0;

// The calling thread's memo (null when it has none and !create), emptied if a context came or went.
std::array<TextureDescMemoEntry, TextureDescMemoSlots>* TextureDescMemo(bool create) {
	const auto generation = g_texture_desc_memo_generation.load(std::memory_order_relaxed);
	if (t_texture_desc_memo_generation != generation) {
		t_texture_desc_memo.reset();
		t_texture_desc_memo_generation = generation;
	}
	if (create && !t_texture_desc_memo) {
		t_texture_desc_memo = std::make_unique<std::array<TextureDescMemoEntry, TextureDescMemoSlots>>();
	}
	return t_texture_desc_memo.get();
}

size_t TextureDescMemoSlot(const ShaderRecompiler::IR::ImageResource& resource,
                           const ShaderRecompiler::IR::DescriptorValue& value) noexcept {
	uint64_t hash = 0x9e3779b97f4a7c15ull ^ value.dword_count;
	for (const auto dword: value.dwords) {
		hash = (hash ^ dword) * 0x100000001b3ull;
		hash ^= hash >> 29u;
	}
	hash = (hash ^ reinterpret_cast<uintptr_t>(&resource)) * 0x9e3779b97f4a7c15ull;
	return static_cast<size_t>(hash >> 40u) & (TextureDescMemoSlots - 1u);
}

bool SameResolvedTextureDesc(const ResolvedTextureDesc& a, const ResolvedTextureDesc& b) {
	const auto& x = a.desc.info;
	const auto& y = b.desc.info;
	return a.pixel_format == b.pixel_format && a.view_format == b.view_format &&
	       a.desc.type == b.desc.type &&
	       a.desc.heap == b.desc.heap && a.desc.view_info == b.desc.view_info &&
	       x.data == y.data && x.stencil == y.stencil && x.metadata.range == y.metadata.range &&
	       x.metadata.kind == y.metadata.kind && x.metadata.dcc_alpha_msb == y.metadata.dcc_alpha_msb &&
	       x.pixel_format == y.pixel_format && x.guest_format == y.guest_format &&
	       x.type == y.type && x.extent == y.extent && x.resources == y.resources &&
	       x.pitch == y.pitch && x.bytes_per_block == y.bytes_per_block &&
	       x.samples == y.samples && x.tile_mode == y.tile_mode &&
	       // UpdateSize no longer re-zeroes records past resources.levels (equal on both sides here).
	       std::equal(x.mip_layout.begin(),
	                  x.mip_layout.begin() + std::min<size_t>(x.resources.levels, x.mip_layout.size()),
	                  y.mip_layout.begin()) &&
	       x.tiled_slice_stride == y.tiled_slice_stride && x.first_tail_level == y.first_tail_level &&
	       std::memcmp(&x.tiling, &y.tiling, sizeof(x.tiling)) == 0;
}

} // namespace

// The part of ResolveTexture before the texture cache lookup: no texture cache, no guest memory
// except, for image heap resources, whether the image is mapped (false: bind nothing).
static bool ResolveTextureDesc(const RenderContext& context,
                               const ShaderRecompiler::IR::ImageResource& resource,
                               const ShaderTextureResource& descriptor, ResolvedTextureDesc& out);

TextureBinding RenderExecutor::ResolveTexture(const ShaderRecompiler::IR::ImageResource&   resource,
                                              const ShaderRecompiler::IR::DescriptorValue& value) {
	if (resource.atomic64 && !m_context.GetGraphics().shader_image_int64_atomics_enabled) {
		EXIT("64-bit image atomics require shaderImageInt64Atomics\n");
	}
	auto descriptor = DecodeNativeDescriptor<ShaderTextureResource>(value);
	const bool storage = resource.written;
	if (storage) {
		ValidateStorageImageResource(resource);
	}

	auto& texture_cache = m_context.GetTextureCache();
	if (descriptor.IsNull()) {
		return {.layout = vk::ImageLayout::eGeneral};
	}
	// Image heap resolves do not follow the draws that sample them and storage access is not
	// sampling: neither reports samples.
	if (m_lod_stats && descriptor.MipStatsCntEn() && !resource.heap && !storage) {
		auto&         counter = m_mip_stats[descriptor.MipStatsCntId()];
		const uint8_t base    = descriptor.BaseLevel();
		counter.finest_level  = std::min(counter.finest_level, base);
		if (descriptor.MinLod() > base << 8u) {
			counter.clamped = true;
		} else {
			counter.unclamped = true;
		}
	}

	const auto          memo_mode = resource.heap ? 0u : TextureDescMemoMode();
	TextureDescMemoEntry* memo     = nullptr;
	if (memo_mode != 0) {
		memo = &(*TextureDescMemo(true))[TextureDescMemoSlot(resource, value)];
		if (!(memo->valid && memo->value == value && memo->resource == resource)) {
			memo = nullptr;
		} else if (memo_mode == 2) {
			ResolvedTextureDesc check;
			if (!ResolveTextureDesc(m_context, resource, descriptor, check) ||
			    !SameResolvedTextureDesc(check, memo->resolved)) {
				EXIT("KYTY_TEXTURE_DESC_MEMO=2: kept texture description differs: "
				     "dwords=%08x,%08x,%08x,%08x,%08x,%08x,%08x,%08x\n",
				     descriptor.fields[0], descriptor.fields[1], descriptor.fields[2],
				     descriptor.fields[3], descriptor.fields[4], descriptor.fields[5],
				     descriptor.fields[6], descriptor.fields[7]);
			}
		}
	}
	ResolvedTextureDesc          resolved_storage;
	const ResolvedTextureDesc*   resolved  = &resolved_storage;
	TextureCache::FindImageMemo* find_memo = nullptr;
	if (memo != nullptr) {
		resolved  = &memo->resolved;
		find_memo = &memo->find;
	} else {
		if (!ResolveTextureDesc(m_context, resource, descriptor, resolved_storage)) {
			return {.layout = vk::ImageLayout::eGeneral};
		}
		if (memo_mode != 0) {
			auto& entry    = (*t_texture_desc_memo)[TextureDescMemoSlot(resource, value)];
			entry.valid    = true;
			entry.value    = value;
			entry.resource = resource;
			entry.resolved = resolved_storage;
			entry.find     = {};
			entry.stable   = nullptr;
			find_memo      = &entry.find;
		}
	}
	const auto pixel_format = resolved->pixel_format;
	const auto view_format  = resolved->view_format;
	auto       desc         = resolved->desc;

	auto id = [&] {
		PerfStats::ScopedTimer find_timer(PerfStats::Counter::BindFindImageNs, PerfStats::SrtProfEnabled());
		return texture_cache.FindImage(desc, false, find_memo);
	}();
	if (!id) {
		// Another extent at the address holds it; the heap retries once an image is created.
		return {.desc = std::move(desc), .layout = vk::ImageLayout::eGeneral};
	}
	auto*      image               = &texture_cache.GetImage(id);
	const bool stencil_association = static_cast<bool>(image->depth_id);
	if (resource.depth_compare) {
		NoteDepthCompareBind(desc.info.guest_format, image->info.pixel_format,
		                     !stencil_association && !image->info.IsDepth());
	}
	if (stencil_association) {
		id    = image->depth_id;
		image = &texture_cache.GetImage(id);
	} else if (image->info.IsDepth()) {
		if (storage) {
			EXIT("depth target cannot be bound as a storage image\n");
		}
		ValidateSampledDepthBinding(resource, descriptor, *image, pixel_format,
		                            desc.info.data.size);
	} else if (storage) {
		ValidateStorageColorView(image->info.pixel_format, view_format, descriptor.DstSelXYZW());
	} else {
		// Set after FindImage, whose metadata clears need the attachment-capable format.
		desc.view_info.format = SampledViewFormat(m_context.GetGraphics(), desc.info.guest_format,
		                                          pixel_format, resource.skip_degamma);
		(void)SelectSampledColorView(image->info.pixel_format, desc.view_info.format,
		                             descriptor.DstSelXYZW());
	}
	TextureBinding binding {id, nullptr, std::move(desc)};
	return binding;
}

namespace {

// KYTY_RESOLVE_INPLACE (default 1 = on; 0 = off, every bind through ResolveTexture as before; 2 =
// verify), with KYTY_TEXTURE_DESC_MEMO=1 only: when PrepareBindings binds a program texture whose kept
// description is in the memo, it writes the binding straight into its slot (one copy of the
// description instead of three, no scratch description) and knows the resource by its address
// instead of comparing it field by field. An address holds for one render context: the memos are
// dropped when a RenderExecutor is created or destroyed. Any other case runs ResolveTexture. 2 = also
// resolve each such binding through ResolveTexture and compare ("PERF resolveinplace"). Switchable
// between frames: KYTY_AB_FLAG=resolve_inplace.
uint32_t ResolveInplaceMode() noexcept {
	static const uint32_t mode = [] {
		const char* value = std::getenv("KYTY_RESOLVE_INPLACE");
		const auto  m     = value == nullptr ? 1u : static_cast<uint32_t>(std::strtoul(value, nullptr, 10));
		if (value != nullptr) {
			std::printf("KYTY_RESOLVE_INPLACE=%u\n", m);
		}
		return m;
	}();
	return mode;
}

// After ResolveTexture: marks the memo entry it kept or stored for `resource` (owned by a program).
void MarkStableTexture(const ShaderRecompiler::IR::ImageResource& resource,
                       const ShaderRecompiler::IR::DescriptorValue& value) {
	auto* const table = TextureDescMemo(false);
	if (table == nullptr) {
		return;
	}
	auto& entry = (*table)[TextureDescMemoSlot(resource, value)];
	if (entry.valid && entry.value == value && entry.resource == resource) {
		entry.stable = &resource;
	}
}

// KYTY_RESOLVE_INPLACE=2: every field (ImageInfo has no operator==).
bool SameImageDescExact(const TextureCache::ImageDesc& a, const TextureCache::ImageDesc& b) {
	const auto& x = a.info;
	const auto& y = b.info;
	const auto& m = x.metadata;
	const auto& n = y.metadata;
	return a.type == b.type && a.heap == b.heap && a.view_info == b.view_info && x.data == y.data &&
	       x.stencil == y.stencil && m.range == n.range && m.kind == n.kind && m.clear_word == n.clear_word &&
	       m.compression == n.compression && m.stencil_compressed == n.stencil_compressed &&
	       m.clear_register_valid == n.clear_register_valid && m.dcc_alpha_msb == n.dcc_alpha_msb &&
	       x.htile_initial_clear == y.htile_initial_clear && x.pixel_format == y.pixel_format &&
	       x.guest_format == y.guest_format && x.type == y.type && x.extent == y.extent &&
	       x.resources == y.resources && x.pitch == y.pitch && x.bytes_per_block == y.bytes_per_block &&
	       x.samples == y.samples && x.tile_mode == y.tile_mode && x.bgra16 == y.bgra16 &&
	       x.mip_layout == y.mip_layout && std::memcmp(&x.tiling, &y.tiling, sizeof(x.tiling)) == 0 &&
	       x.tiled_slice_stride == y.tiled_slice_stride && x.first_tail_level == y.first_tail_level;
}

} // namespace

void RenderExecutor::DropTextureDescMemos() noexcept {
	g_texture_desc_memo_generation.fetch_add(1, std::memory_order_relaxed);
}

// ResolveTexture's path for a description kept in the memo, written into `out`; false, with nothing
// done, when the memo does not hold this program resource and T#.
bool RenderExecutor::ResolveKeptTexture(const ShaderRecompiler::IR::ImageResource&   resource,
                                        const ShaderRecompiler::IR::DescriptorValue& value,
                                        TextureBinding& out, bool verify) {
	auto* const table = resource.heap ? nullptr : TextureDescMemo(false);
	if (table == nullptr) {
		return false;
	}
	auto& memo = (*table)[TextureDescMemoSlot(resource, value)];
	if (memo.stable != &resource || !memo.valid || !(memo.value == value)) {
		return false;
	}
	if (verify && !(memo.resource == resource)) {
		PerfStats::Add(PerfStats::Counter::RiResourceDiffs, 1);
	}
	if (resource.atomic64 && !m_context.GetGraphics().shader_image_int64_atomics_enabled) {
		EXIT("64-bit image atomics require shaderImageInt64Atomics\n");
	}
	// Not null: the memo keeps no null T#.
	const auto descriptor = DecodeNativeDescriptor<ShaderTextureResource>(value);
	const bool storage    = resource.written;
	if (storage) {
		ValidateStorageImageResource(resource);
	}
	if (m_lod_stats && descriptor.MipStatsCntEn() && !storage) {
		auto&         counter = m_mip_stats[descriptor.MipStatsCntId()];
		const uint8_t base    = descriptor.BaseLevel();
		counter.finest_level  = std::min(counter.finest_level, base);
		if (descriptor.MinLod() > base << 8u) {
			counter.clamped = true;
		} else {
			counter.unclamped = true;
		}
	}
	auto&      texture_cache     = m_context.GetTextureCache();
	const auto pixel_format      = memo.resolved.pixel_format;
	const auto view_format       = memo.resolved.view_format;
	out.desc       = memo.resolved.desc;
	out.image_view = nullptr;
	out.layout     = vk::ImageLayout::eUndefined;
	out.mip_views.clear();
	auto id = [&] {
		PerfStats::ScopedTimer find_timer(PerfStats::Counter::BindFindImageNs, PerfStats::SrtProfEnabled());
		return texture_cache.FindImage(out.desc, false, &memo.find);
	}();
	out.image_id = id;
	if (!id) {
		out.layout = vk::ImageLayout::eGeneral;
		return true;
	}
	auto* image = &texture_cache.GetImage(id);
	if (image->depth_id) {
		id    = image->depth_id;
		image = &texture_cache.GetImage(id);
	} else if (image->info.IsDepth()) {
		if (storage) {
			EXIT("depth target cannot be bound as a storage image\n");
		}
		ValidateSampledDepthBinding(resource, descriptor, *image, pixel_format, out.desc.info.data.size);
	} else if (storage) {
		ValidateStorageColorView(image->info.pixel_format, view_format, descriptor.DstSelXYZW());
	} else {
		out.desc.view_info.format = SampledViewFormat(m_context.GetGraphics(), out.desc.info.guest_format,
		                                              pixel_format, resource.skip_degamma);
		(void)SelectSampledColorView(image->info.pixel_format, out.desc.view_info.format,
		                             descriptor.DstSelXYZW());
	}
	out.image_id = id;
	return true;
}

static bool ResolveTextureDesc(const RenderContext& context,
                               const ShaderRecompiler::IR::ImageResource& resource,
                               const ShaderTextureResource& descriptor, ResolvedTextureDesc& out) {
	const bool storage         = resource.written;
	const auto address         = descriptor.Base40();
	const auto width           = static_cast<uint32_t>(descriptor.Width5()) + 1u;
	const auto height          = static_cast<uint32_t>(descriptor.Height5()) + 1u;
	const auto base_level      = descriptor.BaseLevel();
	const auto last_level      = descriptor.LastLevel();
	const auto type            = TextureType(descriptor);
	const bool multisampled    = IsMultisampledTexture(type);
	const auto max_mip         = resource.r128 ? last_level : descriptor.MaxMip();
	const auto physical_levels = multisampled ? 1u : static_cast<uint32_t>(max_mip) + 1u;
	// IMAGE_STORE addresses BASE_LEVEL; only IMAGE_STORE_MIP selects other view mips.
	const bool single_storage_mip =
	    storage && resource.mip_mode != ShaderRecompiler::IR::ImageMipMode::Dynamic;
	auto view_levels = multisampled || single_storage_mip
	                       ? 1u
	                       : static_cast<uint32_t>(last_level - base_level) + 1u;
	auto levels =
	    multisampled ? 1u : std::max(physical_levels, base_level + view_levels);
	const auto tile       = descriptor.TileMode();
	const bool depth_tile = tile == Prospero::TileMode::kDepth;
	const bool msaa_tile  = depth_tile || tile == Prospero::TileMode::kRenderTarget;
	const bool msaa_array = type == Prospero::ImageType::kColor2DMsaaArray;
	if ((!multisampled && base_level > last_level) ||
	    (multisampled &&
	     (base_level != 0 || last_level == 0 || last_level > 3 || max_mip != last_level ||
	      !msaa_tile || (descriptor.MsaaDepth() && !depth_tile) ||
	      (!msaa_array && (descriptor.Depth() != 0 || descriptor.BaseArray5() != 0))))) {
		EXIT("unsupported texture mip view: base=%u last=%u levels=%u max=%u type=%u tile=%u "
		     "class=%u numeric=%u dimension=%u mip_mode=%u read=%d written=%d "
		     "dwords=%08x,%08x,%08x,%08x,%08x,%08x,%08x,%08x\n",
		     base_level, last_level, levels, descriptor.MaxMip(),
		     static_cast<uint32_t>(descriptor.Type()), static_cast<uint32_t>(tile),
		     static_cast<uint32_t>(resource.resource_class),
		     static_cast<uint32_t>(resource.numeric_class),
		     static_cast<uint32_t>(resource.dimension), static_cast<uint32_t>(resource.mip_mode),
		     resource.read, resource.written, descriptor.fields[0], descriptor.fields[1],
		     descriptor.fields[2], descriptor.fields[3], descriptor.fields[4], descriptor.fields[5],
		     descriptor.fields[6], descriptor.fields[7]);
	}
	const auto samples = multisampled ? 1u << last_level : 1u;
	const auto depth          = static_cast<uint32_t>(descriptor.Depth()) + 1u;
	const auto format         = ShaderRecompiler::IR::ReadsDepthBits(resource)
	                                ? Prospero::BufferFormat::k16UNorm
	                                : ShaderRecompiler::IR::CompareFormat(resource, descriptor.Format());
	const auto surface_format = TextureGetSurfaceFormatInfo(format);
	const bool sampled_numeric_class =
	    storage || resource.numeric_class ==
	                   Prospero::SampledTextureNumericClass(Prospero::RemapTextureFormat(format));
	if (!storage && resource.resource_class == ShaderRecompiler::IR::ImageResourceClass::Sampled &&
	    !sampled_numeric_class) {
		EXIT("sampled image numeric class mismatch: numeric=%u format=%u addr=0x%016" PRIx64 "\n",
		     static_cast<uint32_t>(resource.numeric_class), static_cast<uint32_t>(format), address);
	}

	const bool    volume       = type == Prospero::ImageType::kColor3D;
	const bool    layered      = type == Prospero::ImageType::kColor1DArray ||
	                             type == Prospero::ImageType::kColor2DArray ||
	                             type == Prospero::ImageType::kColor2DMsaaArray;
	auto          image_layers = layered ? depth : 1u;
	// A cube holds six faces even when DEPTH, its last face slice, ends before them (Teardown's
	// sky cube has DEPTH 0).
	if (descriptor.Type() == Prospero::ImageType::kCube) {
		image_layers = std::max(image_layers, 6u);
	}
	auto          view_base    = static_cast<uint32_t>(base_level);
	auto pixel_format = surface_format.vk_format;
	if (resource.depth_compare) {
		if (const auto* depth_format = FindGuestDepthFormatPolicy(format);
		    depth_format && depth_format->depth_attachment_format != vk::Format::eUndefined) {
			pixel_format = depth_format->depth_attachment_format;
		}
	}
	const auto raw_sint_format = ShaderRecompiler::IR::RawSignedStorageFormat(resource, format);
	const auto storage_view_format = resource.atomic64 ? vk::Format::eR64Uint
	                                 : storage && resource.atomic ? vk::Format::eR32Uint
	                                 : raw_sint_format != Prospero::BufferFormat::kInvalid
	                                     ? VulkanFormat(raw_sint_format)
	                                     : SrgbUnormViewFormat(pixel_format);
	const auto view_format         = storage && storage_view_format != vk::Format::eUndefined
	                                     ? storage_view_format
	                                     : pixel_format;
	const auto block_bytes         = Prospero::BlockCompressedBytesPerBlock(format);
	const bool meta_compress       = block_bytes == 0 && !resource.r128 && descriptor.MetaCompress();
	auto& desc = out.desc;
	desc       = {};
	desc.info.data         = {address, 0};
	desc.info.pixel_format = pixel_format;
	desc.info.guest_format = format;
	desc.info.type         = TextureBaseType(type);
	desc.info.extent       = {width, height, volume ? depth : 1u};
	desc.info.resources    = {levels, image_layers};
	desc.info.bytes_per_block =
	    block_bytes != 0 ? block_bytes : Prospero::NumBytesPerElement(format);
	desc.info.samples   = samples;
	desc.info.tile_mode = tile;
	if (meta_compress && tile != Prospero::TileMode::kDepth) {
		desc.info.metadata.kind = ImageMetadataKind::Dcc;
	}
	if (levels > physical_levels) {
		if (!ResolveTextureMipView(desc.info, physical_levels,
		                           meta_compress, view_levels,
		                           view_base)) {
			// The layout stores no mip past MAX_MIP here, so a view starting in the chain ends at
			// MAX_MIP; dynamic-mip shaders still count mips up to LAST_LEVEL.
			if (base_level >= physical_levels ||
			    resource.mip_mode == ShaderRecompiler::IR::ImageMipMode::Dynamic) {
				EXIT("unsupported texture mip view changes physical layout: base=%u last=%u "
				     "max=%u extent=%ux%ux%u tile=%u\n",
				     base_level, last_level, max_mip, width, height, depth,
				     static_cast<uint32_t>(tile));
			}
			view_levels                = physical_levels - base_level;
			desc.info.resources.levels = physical_levels;
			desc.info.data.size        = 0;
			desc.info.UpdateSize();
		}
	} else {
		desc.info.UpdateSize();
	}
	// Image heaps keep the descriptors of unmapped textures, which PS5 never samples, whatever
	// their layout.
	if (resource.heap && !context.IsMapped(address, desc.info.data.size)) {
		return false;
	}
	const auto alignment = desc.info.DataAlignment();
	EXIT_NOT_IMPLEMENTED(desc.info.data.size == 0 || alignment == 0 ||
	                     (address & (static_cast<uint64_t>(alignment) - 1u)) != 0);
	if (storage) {
		ValidateStorageTexture(resource, descriptor, desc.info.data.size);
	}
	if (meta_compress && tile != Prospero::TileMode::kDepth) {
		TileSizeAlign metadata_size {};
		(void)TileGetDccSize(width, height, volume ? depth : image_layers,
		                     desc.info.bytes_per_block, physical_levels, tile, metadata_size,
		                     std::countr_zero(samples));
		desc.info.metadata.range         = {descriptor.MetaAddr() << 8u, metadata_size.size};
		desc.info.metadata.dcc_alpha_msb = descriptor.DccAlphaPos();
	}
	desc.view_info = TextureViewInfo(resource, descriptor, view_format, surface_format, storage,
	                                 view_levels, desc.info.resources.layers);
	desc.view_info.base_level = view_base + resource.view_min_level;
	desc.type = storage ? TextureCache::BindingType::Storage : TextureCache::BindingType::Texture;
	desc.heap = resource.heap;
	out.pixel_format      = pixel_format;
	out.view_format       = view_format;
	return true;
}

static vk::Sampler NativeSampler(RenderContext& context, const ShaderRecompiler::IR::CompiledShaderInfo& program, uint32_t index, const ShaderRecompiler::IR::DescriptorValue& value) {
	auto        descriptor = DecodeNativeDescriptor<ShaderSamplerResource>(value);
	const auto& sampler = program.info.samplers[index];
	if (!sampler.depth_compare) {
		descriptor.fields[0] &= ~(0x7u << 12u);
	}
	if (sampler.force_point_filtering) {
		descriptor.SetPointFiltering();
	}
	return context.GetSamplerCache().GetSampler(descriptor, sampler.integer_border);
}

static vk::DescriptorBufferInfo NativeUpload(RenderContext&            context,
                                             std::span<const uint32_t> data) {
	EXIT_IF(data.empty());
	EXIT_IF(context.GetCommandScheduler().Current().IsInvalid());
	auto&      buffer = context.GetBufferCache().GetUtilityBuffer(MemoryUsage::Stream);
	const auto offset = buffer.Copy(data.data(), data.size_bytes(), 256);
	return {buffer.Handle(), offset, data.size_bytes()};
}

void RenderExecutor::BindImage(ImageId id, bool storage) {
	if (!id) {
		return;
	}
	auto& image = m_context.GetTextureCache().GetImage(id);
	if (image.binding.is_bound) {
		image.binding.force_general |= image.binding.shader_write != storage;
	}
	image.binding.is_bound = true;
	image.binding.shader_write |= storage;
	m_bound_images.push_back(id);
}

void RenderExecutor::BindRenderTarget(ImageId id) {
	auto& image             = m_context.GetTextureCache().GetImage(id);
	image.binding.is_target = true;
	m_bound_images.push_back(id);
}

void RenderExecutor::ResetBindings() {
	for (const auto id: m_bound_images) {
		m_context.GetTextureCache().m_slot_images[id].binding = {};
	}
	m_bound_images.clear();
}

void RenderExecutor::PrepareBindings(const ShaderStageRuntime& runtime,
                                     PreparedBindings& prepared) {
	KYTY_PROFILER_FUNCTION();
	PerfStats::ScopedTimer bind_timer(PerfStats::Counter::BindPrepareNs, PerfStats::BindProfEnabled());
	EXIT_IF(!runtime);
	const auto& program  = *runtime.program;
	const auto& snapshot = *runtime.resources;
	prepared.runtime = &runtime;
	prepared.gds = {nullptr, 0, VK_WHOLE_SIZE};
	prepared.flattened_srt = {};
	prepared.shader_data_buffer = {};
	prepared.buffer_sources.clear();
	prepared.buffers.clear();
	prepared.shared_memory = {};
	prepared.images.resize(program.info.images.size());
	prepared.samplers.clear();
	prepared.shader_data.clear();
	const uint32_t inplace = ResolveInplaceMode() == 0 || TextureDescMemoMode() != 1 ||
	                                 (ResolveInplaceMode() == 1 && PerfStats::AbOff("resolve_inplace"))
	                             ? 0u
	                             : ResolveInplaceMode();
	uint32_t kept = 0;
	for (uint32_t i = 0; i < program.info.images.size(); i++) {
		if (program.info.images[i].heap) {
			prepared.images[i] = {};
			continue;
		}
		// A sampled texture whose descriptor enables mip statistics counts in its counter, which
		// texture streamers read back (MipStats::Report).
		if (MipStats::ReliefEnabled() && !program.info.images[i].written &&
		    snapshot.images[i].dword_count >= 8) {
			const auto descriptor = DecodeNativeDescriptor<ShaderTextureResource>(snapshot.images[i]);
			if (descriptor.MipStatsCntEn()) {
				MipStats::Mark(descriptor.MipStatsCntId());
			}
		}
		if (inplace != 0) {
			auto&          out   = prepared.images[i];
			const uint64_t epoch = inplace == 2 ? m_context.GetTextureCache().ImageIndexEpoch() : 0;
			if (ResolveKeptTexture(program.info.images[i], snapshot.images[i], out, inplace == 2)) {
				kept++;
				if (inplace == 2) {
					const auto check = ResolveTexture(program.info.images[i], snapshot.images[i]);
					const bool same_rest = check.image_view == out.image_view && check.layout == out.layout &&
					                       check.mip_views.empty() && SameImageDescExact(check.desc, out.desc);
					PerfStats::Add(PerfStats::Counter::RiChecked, 1);
					if (!same_rest || check.image_id != out.image_id) {
						const bool raced = same_rest &&
						                   m_context.GetTextureCache().ImageIndexEpoch() != epoch;
						PerfStats::Add(raced ? PerfStats::Counter::RiRaced : PerfStats::Counter::RiDiffs, 1);
						static std::atomic<uint32_t> logged {0};
						if (logged.fetch_add(1, std::memory_order_relaxed) < 32) {
							std::printf("KYTY_RESOLVE_INPLACE=2: %s addr=0x%016" PRIx64 " image %u/%u desc %s\n",
							            raced ? "raced" : "diff", out.desc.info.data.address,
							            out.image_id.index, check.image_id.index, same_rest ? "same" : "differs");
						}
					}
				}
				BindImage(out.image_id, out.desc.type == TextureCache::BindingType::Storage);
				continue;
			}
		}
		t_binding_shader_hash = program.shader_hash;
		auto binding = ResolveTexture(program.info.images[i], snapshot.images[i]);
		t_binding_shader_hash = 0;
		BindImage(binding.image_id, binding.desc.type == TextureCache::BindingType::Storage);
		binding.mip_views.swap(prepared.images[i].mip_views);
		binding.mip_views.clear();
		prepared.images[i] = std::move(binding);
		if (inplace != 0) {
			MarkStableTexture(program.info.images[i], snapshot.images[i]);
		}
	}
	if (kept != 0) {
		// Batched: no atomic per binding in the arm being timed.
		thread_local uint64_t pending = 0;
		if ((pending += kept) >= 4096) {
			PerfStats::Add(PerfStats::Counter::RiKept, pending);
			pending = 0;
		}
	}
	prepared.samplers.reserve(program.info.samplers.size());
	PerfStats::ScopedTimer sampler_timer(PerfStats::Counter::BindSamplerNs,
	                                     PerfStats::SrtProfEnabled() && !program.info.samplers.empty());
	for (uint32_t i = 0; i < program.info.samplers.size(); i++) {
		prepared.samplers.push_back(NativeSampler(
		    m_context, program, i, snapshot.samplers[program.info.samplers[i].snapshot_index]));
	}
	prepared.shader_data.reserve(program.bindings.ShaderDataDwords());
	for (const auto reg: program.info.user_data_registers) {
		prepared.shader_data.push_back(snapshot.user_data[reg - program.user_data_base]);
	}
	prepared.shader_data.resize(program.bindings.ShaderDataDwords());
	if (program.bindings.descriptor_counts[static_cast<size_t>(BindingKind::Gds)] != 0) {
		prepared.gds.buffer = m_context.GetBufferCache().GetGdsBuffer()->Handle();
	}
}

namespace {

using StatsClock = std::chrono::steady_clock;

uint64_t StatsNs(StatsClock::time_point a, StatsClock::time_point b) {
	return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(b - a).count());
}

// KYTY_BIND_MEMO_STATS: one FindBuffers slot, timed from construction; at destruction it counts
// whether the slot found the same source for the same program and V# as its previous draw.
struct FindSlotStats {
	FindSlotStats(PreparedBindings::BufferSlotStats* slot, const PreparedBindings::BufferSource& source,
	              const void* program, const ShaderRecompiler::IR::DescriptorValue& value, bool written,
	              const BufferCache& cache)
	    : slot(slot), source(source), program(program), value(value), written(written), cache(cache) {
		if (slot != nullptr) {
			const auto c0 = StatsClock::now();
			start         = StatsClock::now();
			PerfStats::Add(PerfStats::Counter::FbClockNs, StatsNs(c0, start));
			epoch = cache.BufferSetEpoch();
		}
	}
	~FindSlotStats() {
		if (slot == nullptr) return;
		const auto ns = StatsNs(start, StatsClock::now());
		PerfStats::Add(PerfStats::Counter::FbSlots, 1);
		PerfStats::Add(PerfStats::Counter::FbNs, ns);
		// Sentinel write ranges also depend on the other stages (not kept, as in the fork).
		const bool sentinel = written && DecodeNativeDescriptor<ShaderBufferResource>(value).NumRecords() == UINT32_MAX;
		if (!sentinel && slot->program == program && slot->value == value && slot->find_epoch == epoch &&
		    epoch == cache.BufferSetEpoch() && slot->source.address == source.address &&
		    slot->source.size == source.size && slot->source.id == source.id) {
			PerfStats::Add(PerfStats::Counter::FbHit, 1);
			PerfStats::Add(PerfStats::Counter::FbHitNs, ns);
		}
		slot->program    = sentinel ? nullptr : program;
		slot->value      = value;
		slot->source     = source;
		slot->find_epoch = cache.BufferSetEpoch();
	}
	PreparedBindings::BufferSlotStats*           slot;
	const PreparedBindings::BufferSource&        source;
	const void*                                  program;
	const ShaderRecompiler::IR::DescriptorValue& value;
	bool                                         written;
	const BufferCache&                           cache;
	StatsClock::time_point                       start {};
	uint64_t                                     epoch = 0;
};

// KYTY_BIND_MEMO_STATS: classifies one NativeStorageBuffer call of RebindBuffers (CP back thread).
void NoteBindStats(BufferCache& cache, PreparedBindings::BufferSlotStats& slot,
                   const PreparedBindings::BufferSource& source, const ShaderRecompiler::IR::BufferResource& resource,
                   const vk::DescriptorBufferInfo& info, uint32_t offset, uint64_t epoch0, uint64_t uploads0,
                   uint64_t other0, uint64_t clock_ns, uint64_t ns) {
	using PerfStats::Add;
	using C = PerfStats::Counter;
	Add(C::BmCalls, 1);
	Add(C::BmNs, ns);
	Add(C::BmClockNs, clock_ns);
	const bool texel = resource.formatted || (resource.read && Config::SyncRawImageBuffersEnabled());
	const auto kind  = static_cast<uint32_t>(resource.written ? 1 : texel ? 3 : 2);
	const bool upload = cache.SyncUploads() != uploads0;
	const bool other  = cache.SyncOther() != other0;
	const bool stream = info.buffer && info.buffer == cache.GetUtilityBuffer(MemoryUsage::Stream).Handle();
	bool       kept   = false;
	if (source.address == 0 || source.size == 0) {
		Add(C::BmNull, 1);
		Add(C::BmNullNs, ns);
	} else if (resource.written) {
		Add(C::BmWritten, 1);
		Add(C::BmWrittenNs, ns);
	} else if (stream) {
		Add(C::BmStream, 1);
		Add(C::BmStreamNs, ns);
	} else if (upload || other) {
		Add(C::BmWork, 1);
		Add(C::BmWorkNs, ns);
		kept = !other; // after an upload the range is clean: a memo may store it
	} else {
		kept = true;
		Add(C::BmClean, 1);
		Add(C::BmCleanNs, ns);
		if (kind == 3) {
			Add(C::BmTexel, 1);
			Add(C::BmTexelNs, ns);
		}
		const bool same_key = slot.kept && slot.address == source.address && slot.size == source.size &&
		                      slot.kind == kind && slot.bind_epoch == epoch0 && epoch0 == cache.BufferSetEpoch();
		if (same_key && slot.info == info && slot.offset == offset) {
			Add(C::BmSlotHit, 1);
			Add(C::BmSlotHitNs, ns);
			if (kind == 3) {
				Add(C::BmSlotTexel, 1);
				Add(C::BmSlotTexelNs, ns);
			}
		} else if (same_key) {
			Add(C::BmSlotDiff, 1);
		}
		struct RangeEntry {
			uint64_t                 address = 0, size = 0, epoch = UINT64_MAX;
			vk::DescriptorBufferInfo info;
			uint32_t                 offset = 0, kind = 0;
		};
		static std::array<RangeEntry, 4096> table {}; // back thread only
		auto& e = table[((source.address ^ (source.size << 40u) ^ kind) * 0x9e3779b97f4a7c15ull) >> 52u];
		if (e.address == source.address && e.size == source.size && e.kind == kind && e.epoch == epoch0 &&
		    epoch0 == cache.BufferSetEpoch() && e.info == info && e.offset == offset) {
			Add(C::BmRangeHit, 1);
			Add(C::BmRangeHitNs, ns);
		}
		e = {source.address, source.size, cache.BufferSetEpoch(), info, offset, kind};
	}
	slot = {slot.program, slot.value, slot.source, slot.find_epoch, source.address, source.size,
	        cache.BufferSetEpoch(), info, offset, kind, kept};
}

// KYTY_BIND_MEMO (default 1): buffer slot memo, after pengui07/KytyPS5 (Petr, GPL-2.0).
//   0: off: every slot looks its buffer up (FindBuffers) and binds it (RebindBuffers) at each draw;
//   1 (default): a slot keeps its source while its V# and slot metadata, the buffer set and the guest virtual
//      ranges are unchanged, and keeps its plain-read binding while its range, the buffer set and the
//      CPU-dirty epochs of the 64 KiB blocks it covers are unchanged; in-session A/B:
//      KYTY_AB_FLAG=bind_memo (arm 0 = off). Entries left during an off arm stay exact: every input
//      they depend on is in their key;
//   2: verify: each slot a memo would keep is looked up and bound anyway and compared ("PERF bindverify").
// Off while imported memory, the late-read detector or a GPU capture is active (their bookkeeping
// runs inside the lookups).
int BindMemoMode(const BufferCache& cache) {
	static const int mode = [] {
		const char* value = std::getenv("KYTY_BIND_MEMO");
		const int   m     = value == nullptr ? 1 : std::atoi(value);
		if (value != nullptr) {
			std::printf("KYTY_BIND_MEMO=%d\n", m);
		}
		return m;
	}();
	if (mode == 0 || cache.ImportActive() || LateRead::Enabled() || GpuCapture::Enabled()) [[unlikely]] {
		return 0;
	}
	return mode == 1 && PerfStats::AbOff("bind_memo") ? 0 : mode;
}

// KYTY_BIND_MEMO=2: the first differences, in the log (counts: "PERF bindverify", KYTY_PERF_STATS=1).
void NoteVerifyDiff(const char* what, uint64_t address, uint64_t size, const char* why) {
	static std::atomic<uint32_t> logged {0};
	if (logged.fetch_add(1, std::memory_order_relaxed) < 32) {
		std::printf("KYTY_BIND_MEMO=2: %s diff addr=0x%016" PRIx64 " size=0x%" PRIx64 ": %s\n", what, address, size, why);
	}
}

// KYTY_BIND_MEMO: one FindBuffers slot. Reads the generations before the lookup (an edit during it
// makes the next draw miss) and stores the found source at destruction; in verify mode compares it
// with the kept one.
struct FindSlotMemo {
	FindSlotMemo(PreparedBindings::BufferSlotMemo* memo, const PreparedBindings::BufferSource& source,
	             const ShaderRecompiler::IR::DescriptorValue& value, const ShaderRecompiler::IR::BufferResource& metadata,
	             const BufferCache& cache, bool check)
	    : memo(memo), source(source), value(value), metadata(metadata), check(check) {
		if (memo != nullptr) {
			buffers = cache.BufferSetEpoch();
			ranges  = LibKernel::Memory::VirtualRangesGeneration();
			kept    = memo->source;
		}
	}
	~FindSlotMemo() {
		if (memo == nullptr) return;
		if (check) {
			PerfStats::Add(PerfStats::Counter::BvFindChecked, 1);
			if (source.address != kept.address || source.size != kept.size || source.id != kept.id) {
				PerfStats::Add(PerfStats::Counter::BvFindDiffs, 1);
				NoteVerifyDiff("find", source.address, source.size, "other source");
			}
		}
		// Sentinel write ranges also depend on the other stages' reads: not kept.
		const bool sentinel =
		    metadata.written && DecodeNativeDescriptor<ShaderBufferResource>(value).NumRecords() == UINT32_MAX;
		memo->value           = value;
		memo->max_byte_extent = metadata.max_byte_extent;
		memo->written         = metadata.written;
		memo->scalar          = metadata.scalar;
		memo->find_kept       = !sentinel;
		memo->find_buffers    = buffers;
		memo->find_ranges     = ranges;
		memo->source          = source;
	}
	PreparedBindings::BufferSlotMemo*            memo;
	const PreparedBindings::BufferSource&        source;
	const ShaderRecompiler::IR::DescriptorValue& value;
	const ShaderRecompiler::IR::BufferResource&  metadata;
	bool                                         check;
	PreparedBindings::BufferSource               kept;
	uint64_t                                     buffers = 0, ranges = 0;
};

bool FindKept(const PreparedBindings::BufferSlotMemo& memo, const ShaderRecompiler::IR::DescriptorValue& value,
              const ShaderRecompiler::IR::BufferResource& metadata, const BufferCache& cache) {
	return memo.find_kept && memo.value == value && memo.max_byte_extent == metadata.max_byte_extent &&
	       memo.written == metadata.written && memo.scalar == metadata.scalar &&
	       memo.find_buffers == cache.BufferSetEpoch() &&
	       memo.find_ranges == LibKernel::Memory::VirtualRangesGeneration();
}

} // namespace

void RenderExecutor::FindBuffers(std::span<PreparedBindings* const> stages) {
	KYTY_PROFILER_FUNCTION();
	PerfStats::ScopedTimer bind_timer(PerfStats::Counter::BindFindBuffersNs, PerfStats::BindProfEnabled());
	auto& cache = m_context.GetBufferCache();
	const bool memo_stats = PerfStats::BindMemoStatsEnabled();
	const int  memo_mode  = BindMemoMode(cache);
	BufferCache::UploadBatch upload_batch(cache);
	for (auto* stage: stages) {
		auto& prepared = *stage;
		const auto& program  = *prepared.runtime->program;
		const auto& snapshot = *prepared.runtime->resources;
		const auto& layout   = program.bindings;
		prepared.buffer_sources.resize(layout.descriptor_counts[static_cast<size_t>(BindingKind::Buffers)]);
		if (layout.descriptor_counts[static_cast<size_t>(BindingKind::Buffers)] == 0) {
			continue;
		}
		if (memo_stats) {
			prepared.buffer_stats.resize(prepared.buffer_sources.size());
		}
		if (memo_mode != 0) {
			prepared.buffer_memo.resize(prepared.buffer_sources.size());
		}
		for (uint32_t resource = 0; resource < program.info.buffers.size(); ++resource) {
			const auto& metadata = program.info.buffers[resource];
			if (metadata.descriptor_index == UINT32_MAX) continue;
			auto& source = prepared.buffer_sources[metadata.descriptor_index];
			PreparedBindings::BufferSlotMemo* memo  = nullptr;
			bool                              check = false;
			if (memo_mode != 0) {
				memo = &prepared.buffer_memo[metadata.descriptor_index];
				if (FindKept(*memo, snapshot.buffers[resource], metadata, cache)) {
					if (memo_mode == 1) {
						source = memo->source;
						continue;
					}
					check = true;
				}
			}
			const FindSlotMemo slot_memo(memo, source, snapshot.buffers[resource], metadata, cache, check);
			const FindSlotStats slot_stats(memo_stats ? &prepared.buffer_stats[metadata.descriptor_index] : nullptr,
			                               source, &program, snapshot.buffers[resource], metadata.written, cache);
			const auto descriptor = DecodeNativeDescriptor<ShaderBufferResource>(snapshot.buffers[resource]);
			const auto address = descriptor.Base48();
			auto size = descriptor.GetSize();
			if (size != 0 && descriptor.Stride() == 0 && descriptor.OutOfBounds() == 2 &&
			    !metadata.scalar) {
				// OOB_SELECT=2 only tests NumRecords != 0; a zero-stride fetch
				// still needs its full payload when NumRecords is smaller than it.
				size = std::max(size, uint64_t {metadata.max_byte_extent});
			}
			if (address < BufferCache::CACHING_PAGESIZE || size == 0) {
				source = {};
				continue;
			}
			if (descriptor.NumRecords() == UINT32_MAX) {
				if (program.info.buffers[resource].written) {
					// Sentinel write ranges end before tables captured by any bound stage. A
					// compute dispatch's own scalar reads may overlap its writes (the scalar cache
					// is not coherent with the dispatch's vector writes, so the values captured at
					// dispatch start are a valid result), so a table at the start of its own range
					// is allowed; later tables still end the range so it never grows past the
					// existing limits.
					for (const auto* reader: stages) {
						const bool own_compute_reads =
						    reader == stage && program.stage == ShaderType::Compute;
						for (const auto [read_address, read_size]: reader->runtime->resources->specialization_reads) {
							if (read_size == 0) continue;
							if (read_address > address) {
								size = std::min(size, read_address - address);
							} else if (address - read_address < read_size && !own_compute_reads) {
								EXIT("scalar resource reads overlap a shader buffer write\n");
							}
						}
					}
				}
				//  PPSA15246
				const auto& graphics = m_context.GetGraphics();
				const auto  limit =
				    uint64_t {graphics.GetPhysicalDeviceProperties().limits.maxStorageBufferRange} -
				    (graphics.StorageMinAlignment() - 1);
				size = std::min({size, uint64_t {256} * 1024 * 1024, limit});
			}
			// Finite GPU descriptors can span sparse PRT holes (reserved ranges are known); clamp
			// only oversized/sentinel ranges. Guest shaders often leave descriptors that a runtime
			// branch never uses filled with stale data; hardware only faults on access, so a range
			// outside every guest mapping binds nothing (reads return zero, writes are dropped)
			// instead of aborting.
			bool unmapped = size != 0 && !Libs::LibKernel::Memory::IsVirtualAddressKnown(address);
			if (!unmapped && (size >= (uint64_t {1} << 30u) || descriptor.NumRecords() == UINT32_MAX)) {
				// Oversized finite pool windows are mapped in pieces, so they end at their last
				// committed range; sentinel ranges stop at the first hole.
				const auto mapped_size = Libs::LibKernel::Memory::TryClampRangeSize(
				    address, size, descriptor.NumRecords() != UINT32_MAX);
				unmapped               = mapped_size == 0;
				size                   = mapped_size;
				// A window over 4 GiB or more of adjacent mappings still exceeds one storage view.
				// shortcut: accesses past the device limit read zero and drop writes; route such
				// buffers through BDA addressing (upstream KytyPS5 #978) if a title needs them.
				const auto& graphics = m_context.GetGraphics();
				size = std::min(size,
				                uint64_t {graphics.GetPhysicalDeviceProperties().limits.maxStorageBufferRange} -
				                    (graphics.StorageMinAlignment() - 1));
			}
			if (unmapped) {
				static std::atomic<uint32_t> logged {0};
				if (logged.fetch_add(1, std::memory_order_relaxed) < 32) {
					LOGF("FindBuffers: unmapped buffer descriptor ignored: shader=0x%016" PRIx64
					     " buffer=%u addr=0x%016" PRIx64 " size=0x%016" PRIx64 "\n",
					     program.shader_hash, resource, address, size);
				}
				source = {};
				continue;
			}
			source = {address, size, cache.FindBuffer(address, size)};
		}
	}
}

void RenderExecutor::RebindBuffers(PreparedBindings& prepared) {
	KYTY_PROFILER_FUNCTION();
	PerfStats::ScopedTimer bind_timer(PerfStats::Counter::BindBuffersNs, PerfStats::BindProfEnabled());
	const auto& program   = *prepared.runtime->program;
	const auto& snapshot  = *prepared.runtime->resources;
	const auto& layout    = program.bindings;

	prepared.buffers.resize(layout.descriptor_counts[static_cast<size_t>(BindingKind::Buffers)]);
	std::fill(prepared.shader_data.begin() + layout.memory_offset_dword,
	          prepared.shader_data.end(), 0);
	auto pack_memory_offset = [&](uint32_t index, uint32_t offset) {
		const auto dword = layout.memory_offset_dword + index / 4u;
		const auto shift = (index % 4u) * 8u;
		prepared.shader_data[dword] |= offset << shift;
	};
	const bool memo_stats = PerfStats::BindMemoStatsEnabled();
	if (memo_stats) {
		prepared.buffer_stats.resize(prepared.buffers.size());
	}
	auto&     cache     = m_context.GetBufferCache();
	const int memo_mode = BindMemoMode(cache);
	if (memo_mode != 0) {
		prepared.buffer_memo.resize(prepared.buffers.size());
	}
	for (const auto& resource: program.info.buffers) {
		const auto index = resource.descriptor_index;
		if (index == UINT32_MAX) continue;
		uint32_t buffer_offset = 0;
		PerfStats::ScopedTimer obtain_timer(PerfStats::Counter::BindObtainNs, PerfStats::SrtProfEnabled());
		const auto& source = prepared.buffer_sources[index];
		// Plain reads only: written bindings record GPU ownership and texel reads may copy an image.
		const bool plain_read = !resource.written && !resource.formatted &&
		                        !(resource.read && Config::SyncRawImageBuffersEnabled());
		if (memo_mode != 0 && plain_read && source.address != 0 && source.size != 0) {
			auto&      memo      = prepared.buffer_memo[index];
			const auto cpu_epoch = cache.CpuDirtyEpoch(source.address, source.size); // before the dirty check
			const bool kept = memo.bind_kept && memo.address == source.address && memo.size == source.size &&
			                  memo.bind_buffers == cache.BufferSetEpoch() && memo.cpu_epoch == cpu_epoch;
			if (kept && memo_mode == 1) {
				// The range is as clean as when it was bound: the lookup would find the same buffer
				// and have nothing to upload.
				GpuCapture::Touch(source.address, source.size);
				cache.TouchBinding(*memo.buffer);
				prepared.buffers[index] = memo.info;
				buffer_offset           = memo.offset;
			} else {
				const auto    uploads0 = cache.SyncUploads();
				const auto    other0   = cache.SyncOther();
				const Buffer* buffer   = nullptr;
				prepared.buffers[index] = NativeStorageBuffer(m_context, source, resource, buffer_offset, &buffer);
				const auto& info   = prepared.buffers[index];
				const bool  stream = info.buffer == cache.GetUtilityBuffer(MemoryUsage::Stream).Handle();
				const bool  other  = cache.SyncOther() != other0;
				if (kept) {
					PerfStats::Add(PerfStats::Counter::BvBindChecked, 1);
					const char* why = stream                            ? "stream copy"
					                  : other                           ? "image or imported sync"
					                  : cache.SyncUploads() != uploads0 ? "upload"
					                  : info != memo.info               ? "other descriptor"
					                  : buffer_offset != memo.offset    ? "other offset"
					                                                    : nullptr;
					if (why != nullptr && cache.CpuDirtyEpoch(source.address, source.size) != cpu_epoch) {
						// A guest write that faulted during this lookup: had it landed just after,
						// the full path would miss it too until the next draw.
						PerfStats::Add(PerfStats::Counter::BvBindRaced, 1);
					} else if (why != nullptr) {
						PerfStats::Add(PerfStats::Counter::BvBindDiffs, 1);
						NoteVerifyDiff("bind", source.address, source.size, why);
					}
				} else if (memo_mode == 2) {
					PerfStats::Add(PerfStats::Counter::BvBindMisses, 1);
				}
				// A stream copy is valid for this draw only; image copies are redone at each bind.
				memo.bind_kept    = !stream && !other && buffer != nullptr;
				memo.address      = source.address;
				memo.size         = source.size;
				memo.bind_buffers = cache.BufferSetEpoch();
				memo.cpu_epoch    = cpu_epoch;
				memo.buffer       = buffer;
				memo.info         = info;
				memo.offset       = buffer_offset;
			}
		} else if (memo_stats) [[unlikely]] {
			const auto c0      = StatsClock::now();
			const auto c1      = StatsClock::now();
			const auto epoch0  = cache.BufferSetEpoch();
			const auto upload0 = cache.SyncUploads();
			const auto other0  = cache.SyncOther();
			prepared.buffers[index] = NativeStorageBuffer(m_context, prepared.buffer_sources[index],
			                                              resource, buffer_offset);
			NoteBindStats(cache, prepared.buffer_stats[index], prepared.buffer_sources[index], resource,
			              prepared.buffers[index], buffer_offset, epoch0, upload0, other0, StatsNs(c0, c1),
			              StatsNs(c1, StatsClock::now()));
		} else {
			prepared.buffers[index] = NativeStorageBuffer(m_context, prepared.buffer_sources[index],
			                                              resource, buffer_offset);
		}
		pack_memory_offset(index, buffer_offset);
		// Null bindings allow 0 bytes, not the shared null buffer's 16, so no store reaches it.
		prepared.shader_data[layout.MemorySizeDword() + index] =
		    static_cast<uint32_t>(prepared.buffer_sources[index].size + buffer_offset);
	}
	PerfStats::ScopedTimer upload_timer(PerfStats::Counter::BindUploadNs, PerfStats::SrtProfEnabled());
	if (layout.descriptor_counts[static_cast<size_t>(BindingKind::FlattenedSrt)] != 0) {
		prepared.flattened_srt = NativeUpload(m_context, snapshot.flattened_srt);
	}
	if (layout.descriptor_counts[static_cast<size_t>(BindingKind::ShaderData)] != 0) {
		prepared.shader_data_buffer = NativeUpload(m_context, prepared.shader_data);
	}
}

void RenderExecutor::RebindImages(PreparedBindings& prepared) {
	KYTY_PROFILER_FUNCTION();
	PerfStats::ScopedTimer bind_timer(PerfStats::Counter::BindImagesNs, PerfStats::BindProfEnabled());
	const auto& program  = *prepared.runtime->program;
	const auto& snapshot = *prepared.runtime->resources;
	auto&       images   = prepared.images;
	auto& texture_cache = m_context.GetTextureCache();
	// Acquire each view before a later overlapping descriptor can replace its image.
	for (uint32_t i = 0; i < program.info.images.size(); i++) {
		if (!images[i].image_id) continue;
		auto& old_image = texture_cache.m_slot_images[images[i].image_id];
		if (!old_image.registered || old_image.binding.needs_rebind) {
			old_image.binding = {};
			images[i] = ResolveTexture(program.info.images[i], snapshot.images[i]);
			BindImage(images[i].image_id,
			          images[i].desc.type == TextureCache::BindingType::Storage);
		}
		auto& binding = images[i];
		binding.mip_views.clear();
		const auto& resource = program.info.images[i];
		if (resource.mip_mode == ShaderRecompiler::IR::ImageMipMode::Dynamic) {
			EXIT_IF(resource.mip_count == 0u ||
			        resource.mip_count != binding.desc.view_info.level_count);
			binding.mip_views.reserve(resource.mip_count);
			for (uint32_t mip = 0; mip < resource.mip_count; mip++) {
				auto desc = binding.desc;
				desc.view_info.base_level += mip;
				desc.view_info.level_count = 1;
				// The shader selects the mip after applying the guest minimum LOD.
				desc.view_info.min_lod = 0;
				binding.mip_views.push_back(texture_cache.FindTexture(binding.image_id, desc));
			}
			binding.image_view = binding.mip_views.front();
		} else {
			PerfStats::ScopedTimer view_timer(PerfStats::Counter::BindViewNs, PerfStats::SrtProfEnabled());
			binding.image_view = texture_cache.FindTexture(binding.image_id, binding.desc);
		}
	}
}

void RenderExecutor::PrepareGraphicsBindings(std::span<PreparedBindings* const> stages,
                                             std::span<RenderColorInfo> colors) {
	bool uses_dma        = false;
	bool uses_image_heap = false;
	FindBuffers(stages);
	for (auto* stage: stages) {
		uses_dma |= stage->runtime->program->info.uses_dma;
		uses_image_heap |= stage->runtime->program->info.uses_image_heap;
	}
	if (uses_dma) {
		m_context.PrepareBda();
	}
	if (uses_image_heap) {
		PerfStats::ScopedTimer heap_timer(PerfStats::Counter::BindHeapNs, PerfStats::BindProfEnabled());
		m_context.GetImageHeap().Rebind(*this);
	}
	for (auto* stage: stages) {
		RebindImages(*stage);
	}
	auto& cache = m_context.GetTextureCache();
	for (auto& target: colors) {
		EXIT_IF(!target.image_id);
		auto& old_image = cache.m_slot_images[target.image_id];
		if (!old_image.registered || old_image.binding.needs_rebind) {
			old_image.binding = {};
			target.desc.view_info.base_level = target.guest_mip_level;
			target.desc.view_info.base_layer = target.guest_array_layer;
			target.image_id = cache.FindImage(target.desc);
			BindRenderTarget(target.image_id);
		}
	}
	// Discovery can read back PS5 metadata and submit the scheduler. Reserve draw buffers only
	// after image identities are final; attachment layout transitions follow buffer alias copies.
	BufferCache::UploadBatch upload_batch(m_context.GetBufferCache(), true);
	for (auto* stage: stages) {
		RebindBuffers(*stage);
	}
}

void RenderExecutor::CommitBindings(CommandBuffer&                     buffer,
                                    vk::PipelineBindPoint              pipeline_bind_point,
                                    const PipelineCache::Pipeline&     pipeline,
                                    std::span<PreparedBindings* const> prepared_bindings) {
	KYTY_PROFILER_FUNCTION();
	const bool             bind_prof = PerfStats::BindProfEnabled();
	PerfStats::ScopedTimer bind_timer(PerfStats::Counter::BindCommitNs, bind_prof);
	using BindClock      = std::chrono::steady_clock;
	const auto bind_ns   = [](BindClock::time_point start) {
        return static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(BindClock::now() - start).count());
	};
	const auto check_start = bind_prof ? BindClock::now() : BindClock::time_point {};
	auto   vk_buffer        = buffer.Handle();
	size_t image_descriptor_count = 0;
	ShaderRecompiler::IR::PushData push_data;
	bool                           has_push_data = false;
	for (const auto* prepared: prepared_bindings) {
		const auto& program = *prepared->runtime->program;
		for (auto mask = program.bindings.descriptor_mask; mask != 0; mask &= mask - 1) {
			const auto index = std::countr_zero(mask);
			const auto count = program.bindings.descriptor_counts[index];
			if (index >= ShaderRecompiler::IR::FirstImageBinding &&
			    index <= static_cast<size_t>(BindingKind::Samplers)) {
				image_descriptor_count += count;
			}
		}
		EXIT_IF(!(pipeline.push_stages & NativeShaderStage(program.stage)));
	}
	if (pipeline.uses_image_heap) {
		m_context.GetImageHeap().Commit(buffer, pipeline_bind_point, pipeline.pipeline_layout);
	}
	if (bind_prof) {
		PerfStats::Add(PerfStats::Counter::BindCheckNs, bind_ns(check_start));
		PerfStats::Add(PerfStats::Counter::BindCommitCount, 1);
	}
	m_descriptor_images.resize(image_descriptor_count);
	m_descriptor_writes.clear();
	size_t image_cursor = 0;
	std::array<vk::DescriptorBufferInfo, 2> address_buffers;

	for (auto* prepared: prepared_bindings) {
		const auto& program       = *prepared->runtime->program;
		auto&       descriptors   = *prepared;
		if (descriptors.gds.buffer != nullptr) {
			buffer.EndRendering();
			const auto barrier       = MakeGdsDependency(descriptors.gds.buffer);
			const auto shader_stages = ShaderPipelineStages(NativeShaderStage(program.stage));
			vk_buffer.pipelineBarrier(
			    vk::PipelineStageFlagBits::eHost | vk::PipelineStageFlagBits::eTransfer |
			        vk::PipelineStageFlagBits::eAllGraphics |
			        vk::PipelineStageFlagBits::eComputeShader,
			    shader_stages, vk::DependencyFlags {}, 0, nullptr, 1, &barrier, 0, nullptr);
		}

		const auto transit_start = bind_prof ? BindClock::now() : BindClock::time_point {};
		for (uint32_t i = 0; i < program.info.images.size(); i++) {
			auto& binding = descriptors.images[i];
			if (!binding.image_id) continue;
			auto& image = m_context.GetTextureCache().GetImage(binding.image_id);
			const auto&                 view = binding.desc.view_info;
			const ImageSubresourceRange range {view.base_level, view.level_count, view.base_layer,
			                                   view.layer_count};
			const bool storage = binding.desc.type == TextureCache::BindingType::Storage;
			if (image.binding.is_target) {
				const auto layout = image.binding.attachment_layout;
				EXIT_IF(layout == vk::ImageLayout::eUndefined);
				if (image.info.IsDepth()) {
					const auto host_view =
					    std::ranges::find(image.views, binding.image_view, &CachedImageView::view);
					EXIT_IF(storage || host_view == image.views.end());
					const auto aspect = host_view->info.aspect;
					if (aspect & ~DepthReadableAspects(layout)) {
						EXIT("sampling a writable depth/stencil attachment aspect\n");
					}
				}
				image.Transit(layout,
				              image.binding.attachment_access | vk::AccessFlagBits2::eShaderRead |
				                  (image.binding.shader_write ? vk::AccessFlagBits2::eShaderWrite
				                                              : vk::AccessFlags2 {}),
				              {}, vk_buffer);
			} else if (image.binding.force_general && !image.info.IsDepth()) {
				const vk::AccessFlags2 storage_access = image.binding.shader_write
				                                            ? vk::AccessFlagBits2::eShaderWrite
				                                            : vk::AccessFlags2 {};
				image.Transit(vk::ImageLayout::eGeneral,
				              vk::AccessFlagBits2::eShaderRead | storage_access, {}, vk_buffer);
			} else if (storage) {
				image.Transit(vk::ImageLayout::eGeneral,
				              vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite,
				              range, vk_buffer);
			} else {
				image.Transit(image.ReadLayout(), vk::AccessFlagBits2::eShaderRead, range, vk_buffer);
			}
			binding.layout = image.backing.state.layout;
		}
		if (bind_prof) {
			PerfStats::Add(PerfStats::Counter::BindTransitNs, bind_ns(transit_start));
		}

		std::array<size_t, ShaderRecompiler::IR::ImageBindingCount> image_starts {};
		for (auto mask = program.bindings.descriptor_mask; mask != 0; mask &= mask - 1) {
			const auto index = std::countr_zero(mask);
			const auto kind = static_cast<BindingKind>(index);
			vk::WriteDescriptorSet write {};
			write.dstBinding = ShaderRecompiler::IR::NativeBinding(program.stage, kind);
			write.descriptorType = NativeDescriptorType(kind);
			write.descriptorCount = program.bindings.descriptor_counts[index];
			const auto image_start = image_cursor;
			if (ShaderRecompiler::IR::ImageBindingResourceClass(kind) !=
			    ShaderRecompiler::IR::ImageResourceClass::None) {
				image_starts[ShaderRecompiler::IR::ImageBindingIndex(kind)] = image_start;
				image_cursor += write.descriptorCount;
			} else {
				switch (kind) {
					case BindingKind::Buffers:
						write.pBufferInfo = descriptors.buffers.data();
						break;
					case BindingKind::BdaPagetable:
					case BindingKind::FaultBuffer: {
						auto& cache = m_context.GetBufferCache();
						const auto* bda_buffer = kind == BindingKind::BdaPagetable
						                             ? cache.GetBdaPageTableBuffer()
						                             : cache.GetFaultBuffer();
						auto& view = address_buffers[kind == BindingKind::FaultBuffer];
						view = {bda_buffer->Handle(), 0, bda_buffer->Size()};
						write.pBufferInfo = &view;
						break;
					}
					case BindingKind::FlattenedSrt:
					case BindingKind::ShaderData:
					case BindingKind::SharedMemory:
					case BindingKind::Gds: {
						const vk::DescriptorBufferInfo* view = &descriptors.gds;
						if (kind == BindingKind::FlattenedSrt) {
							view = &descriptors.flattened_srt;
						} else if (kind == BindingKind::ShaderData) {
							view = &descriptors.shader_data_buffer;
						} else if (kind == BindingKind::SharedMemory) {
							view = &descriptors.shared_memory;
						}
						EXIT_IF(view->buffer == nullptr);
						write.pBufferInfo = view;
						break;
					}
					case BindingKind::Samplers:
						for (const auto sampler: descriptors.samplers) {
							m_descriptor_images[image_cursor++] = {sampler, nullptr, vk::ImageLayout::eUndefined};
						}
						break;
					case BindingKind::Count: EXIT("invalid descriptor binding kind");
				}
			}
			if (image_cursor != image_start) {
				write.pImageInfo = m_descriptor_images.data() + image_start;
			}
			m_descriptor_writes.push_back(write);
		}
		for (uint32_t i = 0; i < program.info.images.size(); ++i) {
			const auto& resource = program.info.images[i];
			if (resource.null_fallback) {
				EXIT_IF(descriptors.images[i].image_id);
				continue;
			}
			const auto& texture = descriptors.images[i];
			if (resource.heap) continue;
			const auto kind = *ShaderRecompiler::IR::DescriptorBindingForImage(resource);
			const auto start = image_starts[ShaderRecompiler::IR::ImageBindingIndex(kind)] +
			                   resource.descriptor_index;
			for (uint32_t mip = 0; mip < resource.mip_count; ++mip) {
				EXIT_IF(!m_context.GetGraphics().null_descriptor_enabled && !texture.image_id);
				m_descriptor_images[start + mip] = MakeImageInfo(texture, mip);
			}
		}

		if (program.bindings.UsesPushData()) {
			std::ranges::copy(prepared->shader_data,
			                  push_data.dwords.begin() + program.bindings.push_data_start_dword);
			has_push_data = true;
		}
	}

	if (has_push_data) {
		vk_buffer.pushConstants(pipeline.pipeline_layout, pipeline.push_stages, 0,
		                        sizeof(push_data), push_data.dwords.data());
	}

	const auto push_start = bind_prof ? BindClock::now() : BindClock::time_point {};
	if (bind_prof) {
		uint64_t descriptors = 0;
		for (const auto& write: m_descriptor_writes) {
			descriptors += write.descriptorCount;
		}
		PerfStats::Add(PerfStats::Counter::BindWriteCount, m_descriptor_writes.size());
		PerfStats::Add(PerfStats::Counter::BindDescriptorCount, descriptors);
	}
	if (!m_descriptor_writes.empty()) {
		if (pipeline.uses_push_descriptors) {
			vk_buffer.pushDescriptorSetKHR(pipeline_bind_point, pipeline.pipeline_layout, 0,
			                               static_cast<uint32_t>(m_descriptor_writes.size()),
			                               m_descriptor_writes.data());
		} else {
			const auto set = m_context.GetDescriptorHeap().Commit(pipeline.descriptor_set_layout);
			for (auto& write: m_descriptor_writes) {
				write.dstSet = set;
			}
			m_context.GetGraphics().device.updateDescriptorSets(
			    static_cast<uint32_t>(m_descriptor_writes.size()), m_descriptor_writes.data(), 0,
			    nullptr);
			vk_buffer.bindDescriptorSets(pipeline_bind_point, pipeline.pipeline_layout, 0, 1, &set,
			                             0, nullptr);
		}
	}
	if (bind_prof) {
		PerfStats::Add(PerfStats::Counter::BindPushNs, bind_ns(push_start));
	}
}

} // namespace Libs::Graphics
