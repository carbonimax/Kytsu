#pragma once

#include <cstdint>

namespace Libs::Graphics::ShaderRecompiler::IR {

enum class ScalarReg : uint16_t {};
enum class VectorReg : uint16_t {};

// Scalar register slots are the RDNA scalar operand codes: SGPRs 0-105, then trap temporaries
// TTMP0-15 at their codes 108-123. Codes 106/107 are VCC, kept as separate IR values, so the
// slots stay unused and every scalar code consumer (CFG, calls, tessellation) sees one namespace.
constexpr uint32_t NumSgprs      = 106;
constexpr uint32_t FirstTrapReg  = 108;
constexpr uint32_t NumTrapRegs   = 16;
constexpr uint32_t NumScalarRegs = FirstTrapReg + NumTrapRegs;
constexpr uint32_t NumVectorRegs = 256;

constexpr uint32_t RegIndex(ScalarReg reg) {
	return static_cast<uint32_t>(reg);
}

constexpr uint32_t RegIndex(VectorReg reg) {
	return static_cast<uint32_t>(reg);
}

} // namespace Libs::Graphics::ShaderRecompiler::IR
