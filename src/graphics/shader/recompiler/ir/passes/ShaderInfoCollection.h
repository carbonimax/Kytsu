#ifndef EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_SHADERINFOCOLLECTION_H_
#define EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_SHADERINFOCOLLECTION_H_

#include "graphics/shader/recompiler/ir/ShaderIR.h"

namespace Libs::Graphics::ShaderRecompiler::IR {

// Collects the shader interface, resource usage and instruction features after resource tracking.
// buffer_int64_atomics: the host has 64-bit buffer atomics (a coherent DWORDX2 store uses them).
void CollectShaderInfo(Program& program, ShaderStageInputInfo input_info,
                       bool buffer_int64_atomics = true);

} // namespace Libs::Graphics::ShaderRecompiler::IR

#endif /* EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_SHADERINFOCOLLECTION_H_ */
