#ifndef EMULATOR_INCLUDE_EMULATOR_GRAPHICS_HOST_GPU_RENDERER_IMAGE_TEXTURECOMMON_H_
#define EMULATOR_INCLUDE_EMULATOR_GRAPHICS_HOST_GPU_RENDERER_IMAGE_TEXTURECOMMON_H_

#include "graphics/guest_gpu/gpu_defs.h"
#include "graphics/host_gpu/vulkanCommon.h"

namespace Libs::Graphics {

struct GraphicContext;

struct RenderTargetFormatInfo {
	vk::Format                      format            = vk::Format::eUndefined;
	uint32_t                        bytes_per_element = 0;
	Prospero::ColorComponentMapping export_mapping;
	Prospero::BufferFormat          guest_format = Prospero::BufferFormat::kInvalid;
};

struct SurfaceFormatInfo {
	vk::Format                      vk_format;
	Prospero::BufferFormat          conversion_format;
	Prospero::ColorComponentMapping host_to_storage;
};

vk::ComponentMapping   TextureGetComponentMapping(uint32_t                        swizzle,
                                                  Prospero::ColorComponentMapping host_to_storage,
                                                  uint32_t component_count = 4);
// KYTY_TEXTURE_CHANNEL_REPLICATE=1: sampled 1- and 2-channel views repeat their stored channels
// before DST_SEL (component_count above); off by default, tests switch it with the setter.
[[nodiscard]] bool     TextureChannelReplication() noexcept;
void                   SetTextureChannelReplication(bool enabled) noexcept;
SurfaceFormatInfo      TextureGetSurfaceFormatInfo(Prospero::BufferFormat format);
RenderTargetFormatInfo TextureGetRenderTargetFormat(const GraphicContext&   graphics,
                                                    Prospero::ChannelLayout layout,
                                                    Prospero::ChannelType   type,
                                                    Prospero::ChannelOrder  order);

} // namespace Libs::Graphics

#endif /* EMULATOR_INCLUDE_EMULATOR_GRAPHICS_HOST_GPU_RENDERER_IMAGE_TEXTURECOMMON_H_ */
