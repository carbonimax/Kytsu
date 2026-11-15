#ifndef KYTY_COMMON_FP_ENV_H_
#define KYTY_COMMON_FP_ENV_H_

#include "common/common.h"

namespace Common {

struct FpEnv {
	uint32_t mxcsr;
	uint16_t control;
};

// x87 control word 0x037F (64-bit precision, all exceptions masked, round to nearest). MXCSR 0x9FC0:
// all exceptions masked, round to nearest, flush-to-zero and denormals-are-zero, as on the console.
inline constexpr FpEnv GuestFpEnv {0x9fc0, 0x037f};

// Guest code runs only on x86-64 hosts (macOS through Rosetta 2), so other hosts have no guest FP
// state to keep: reads return zeros and writes do nothing.
inline FpEnv ReadFpEnv() {
	FpEnv env {};
#if defined(__x86_64__) || defined(_M_X64)
	asm volatile("stmxcsr %0\n\tfnstcw %1" : "=m"(env.mxcsr), "=m"(env.control) : : "memory");
#endif
	return env;
}

inline void WriteFpEnv(FpEnv env) {
#if defined(__x86_64__) || defined(_M_X64)
	asm volatile("ldmxcsr %0\n\tfldcw %1" : : "m"(env.mxcsr), "m"(env.control) : "memory");
#else
	(void)env;
#endif
}

} // namespace Common

#endif /* KYTY_COMMON_FP_ENV_H_ */
