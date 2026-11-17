// FO-11: hardware-measured instruction tables, one selector per table (--hw-table-only <name>).
// The table data comes from AnyPS5 (github.com/boykopovar/AnyPS5 at 342b06eb3a, GPL-2.0):
// core/libs/prx/libSceAgcDriver/tests/execution/RcpHardware.cpp and SinCosReduction.cpp. The drivers
// and the checks below are ours. The recompiler does not reproduce RDNA 2 bit for bit (V_RCP_F32 is an
// OpFDiv, V_SIN_F32/V_COS_F32 fold turns exactly, then use GLSL Sin), so each table checks a bound, prints
// how many results hold the hardware bits, and KYTY_HW_TABLE_EXACT=1 requires every bit.

namespace HardwareTables {

// RDNA 2 v_rcp_f32 of 1.m: {m, result}. Other exponents e scale the result exponent by 127 - e.
constexpr std::array<std::array<u32, 2>, 256> kRcpMantissas{{
    {0x000000u, 0x3f800000u}, {0x002922u, 0x3f7fadd6u}, {0x00a1a9u, 0x3f7ebe44u}, {0x010c47u, 0x3f7debceu},
    {0x013df5u, 0x3f7d8a32u}, {0x01a87fu, 0x3f7cb9ddu}, {0x024f43u, 0x3f7b766fu}, {0x02d76du, 0x3f7a70beu},
    {0x031690u, 0x3f79f820u}, {0x033035u, 0x3f79c742u}, {0x03adabu, 0x3f78d946u}, {0x04435bu, 0x3f77bfa4u},
    {0x04f3a1u, 0x3f76772au}, {0x057a96u, 0x3f757df8u}, {0x05c6afu, 0x3f74f253u}, {0x06342au, 0x3f742a80u},
    {0x06bf6fu, 0x3f732e24u}, {0x07709au, 0x3f71f00au}, {0x07eb02u, 0x3f711627u}, {0x086273u, 0x3f704304u},
    {0x09362bu, 0x3f6ed04au}, {0x099950u, 0x3f6e2438u}, {0x09f75cu, 0x3f6d81e2u}, {0x0a097cu, 0x3f6d62b3u},
    {0x0ab7e3u, 0x3f6c3840u}, {0x0b1a69u, 0x3f6b90f0u}, {0x0b93a5u, 0x3f6ac454u}, {0x0becd7u, 0x3f6a2eadu},
    {0x0bf0b4u, 0x3f6a2836u}, {0x0c5c7fu, 0x3f697463u}, {0x0c9164u, 0x3f691c8au}, {0x0cb1e2u, 0x3f68e6b5u},
    {0x0d5f89u, 0x3f67c8a0u}, {0x0dd27au, 0x3f670cc6u}, {0x0e09e4u, 0x3f66b2a2u}, {0x0ead02u, 0x3f65aae2u},
    {0x0ed904u, 0x3f656421u}, {0x0f17a3u, 0x3f64ffbeu}, {0x0f21ddu, 0x3f64ef61u}, {0x0f2bcdu, 0x3f64df7du},
    {0x0f4205u, 0x3f64bbfeu}, {0x0f8808u, 0x3f644c6cu}, {0x0faddcu, 0x3f641050u}, {0x0fd630u, 0x3f63d05fu},
    {0x1012f0u, 0x3f637050u}, {0x108fe7u, 0x3f62abb4u}, {0x10a3d6u, 0x3f628c77u}, {0x1184b9u, 0x3f612e5au},
    {0x119a72u, 0x3f610cc2u}, {0x11e20bu, 0x3f609e4fu}, {0x123f59u, 0x3f600f00u}, {0x128b2fu, 0x3f5f9b0du},
    {0x12bd4au, 0x3f5f4eb3u}, {0x12f18du, 0x3f5eff48u}, {0x135d16u, 0x3f5e5c8du}, {0x13deefu, 0x3f5d994bu},
    {0x140ceeu, 0x3f5d5472u}, {0x14a0f9u, 0x3f5c77fdu}, {0x14f473u, 0x3f5bfc70u}, {0x15051au, 0x3f5be3dau},
    {0x153e7cu, 0x3f5b8f4fu}, {0x15b949u, 0x3f5adb3au}, {0x1600a3u, 0x3f5a7320u}, {0x1676ebu, 0x3f59c766u},
    {0x172abbu, 0x3f58c45au}, {0x1738f7u, 0x3f58aff3u}, {0x17a05au, 0x3f581c33u}, {0x17f5e8u, 0x3f57a288u},
    {0x1818e8u, 0x3f5770e9u}, {0x185f1au, 0x3f570da8u}, {0x18f135u, 0x3f564038u}, {0x194f2fu, 0x3f55bce2u},
    {0x1a61dbu, 0x3f54409cu}, {0x1a63e2u, 0x3f543dd2u}, {0x1af098u, 0x3f537d12u}, {0x1bd958u, 0x3f52413au},
    {0x1cb224u, 0x3f511e54u}, {0x1deb7au, 0x3f4f7f68u}, {0x1df9fdu, 0x3f4f6c59u}, {0x1e27a1u, 0x3f4f307du},
    {0x1e398fu, 0x3f4f1903u}, {0x1ef9d8u, 0x3f4e1e85u}, {0x1fb17cu, 0x3f4d317eu}, {0x1fec12u, 0x3f4ce652u},
    {0x202036u, 0x3f4ca39au}, {0x20b671u, 0x3f4be450u}, {0x211c70u, 0x3f4b633bu}, {0x21b455u, 0x3f4aa42eu},
    {0x2217beu, 0x3f4a27e7u}, {0x22c947u, 0x3f494b6eu}, {0x230d97u, 0x3f48f719u}, {0x23a75fu, 0x3f483a41u},
    {0x244f6fu, 0x3f476d74u}, {0x24ede6u, 0x3f46add8u}, {0x250d33u, 0x3f46882au}, {0x254b0cu, 0x3f463de2u},
    {0x263ba6u, 0x3f451ef3u}, {0x269e0du, 0x3f44aa89u}, {0x26a2c0u, 0x3f44a4fdu}, {0x26bb7du, 0x3f4487d0u},
    {0x26e875u, 0x3f4452ddu}, {0x2744e3u, 0x3f43e660u}, {0x281972u, 0x3f42eeaau}, {0x29b882u, 0x3f4111f2u},
    {0x2a3af4u, 0x3f407e00u}, {0x2a96fbu, 0x3f401628u}, {0x2afa95u, 0x3f3fa642u}, {0x2b0537u, 0x3f3f9a58u},
    {0x2baf6au, 0x3f3edc66u}, {0x2c7557u, 0x3f3e015au}, {0x2d1c9au, 0x3f3d49c5u}, {0x2d711au, 0x3f3ced8cu},
    {0x2e0531u, 0x3f3c4cc6u}, {0x2e4415u, 0x3f3c08d1u}, {0x2e8e7du, 0x3f3bb8aau}, {0x2eae05u, 0x3f3b96c8u},
    {0x2fa54eu, 0x3f3a8eaeu}, {0x301850u, 0x3f3a14d7u}, {0x30d16au, 0x3f39520au}, {0x31f5efu, 0x3f38216cu},
    {0x32c6ecu, 0x3f374a2du}, {0x33d0ceu, 0x3f363b28u}, {0x34b9b5u, 0x3f355051u}, {0x34c9ebu, 0x3f35400eu},
    {0x364b9du, 0x3f33c092u}, {0x36f675u, 0x3f3318bau}, {0x376cbfu, 0x3f32a53au}, {0x37dc76u, 0x3f3238afu},
    {0x387a34u, 0x3f31a04au}, {0x3898d1u, 0x3f3182d5u}, {0x392630u, 0x3f30fb4bu}, {0x398406u, 0x3f30a1c6u},
    {0x3aa8b5u, 0x3f2f8cd0u}, {0x3b1287u, 0x3f2f2983u}, {0x3b6186u, 0x3f2edfabu}, {0x3bbbe9u, 0x3f2e8b79u},
    {0x3bfb1eu, 0x3f2e50c8u}, {0x3cf38cu, 0x3f2d6b98u}, {0x3d9c17u, 0x3f2cd172u}, {0x3e7d1bu, 0x3f2c054du},
    {0x3e875fu, 0x3f2bfc08u}, {0x3f63afu, 0x3f2b360fu}, {0x3f98e2u, 0x3f2b0685u}, {0x3fa7d9u, 0x3f2af92au},
    {0x407b49u, 0x3f2a3d5au}, {0x4174afu, 0x3f2961e2u}, {0x427df9u, 0x3f287ad8u}, {0x43435cu, 0x3f27d089u},
    {0x437bcbu, 0x3f27a016u}, {0x44985eu, 0x3f26ad72u}, {0x451abdu, 0x3f263f34u}, {0x457d42u, 0x3f25ec44u},
    {0x472077u, 0x3f248ef6u}, {0x4724dau, 0x3f248b56u}, {0x47469au, 0x3f246f78u}, {0x482c9cu, 0x3f23b287u},
    {0x486824u, 0x3f2381e6u}, {0x48db40u, 0x3f232432u}, {0x499523u, 0x3f228dc2u}, {0x49b64au, 0x3f22730au},
    {0x49dea9u, 0x3f22528du}, {0x4a23d5u, 0x3f221b02u}, {0x4b14f9u, 0x3f215a86u}, {0x4c25fcu, 0x3f2082beu},
    {0x4cbd87u, 0x3f200befu}, {0x4cdd20u, 0x3f1ff340u}, {0x4dd422u, 0x3f1f334cu}, {0x4ef8aau, 0x3f1e5249u},
    {0x4f20f3u, 0x3f1e337eu}, {0x4f426du, 0x3f1e19f1u}, {0x5051c1u, 0x3f1d4c05u}, {0x506bf2u, 0x3f1d3841u},
    {0x507313u, 0x3f1d32e0u}, {0x519088u, 0x3f1c5cc0u}, {0x520935u, 0x3f1c02eau}, {0x52e6b4u, 0x3f1b5f10u},
    {0x536d35u, 0x3f1afc38u}, {0x54a6d8u, 0x3f1a17a2u}, {0x55c663u, 0x3f19485eu}, {0x56ed5bu, 0x3f187600u},
    {0x571242u, 0x3f185bd8u}, {0x5790f8u, 0x3f180249u}, {0x57ee05u, 0x3f17c0c8u}, {0x588a7eu, 0x3f175320u},
    {0x58d556u, 0x3f171ee4u}, {0x59a54au, 0x3f168e80u}, {0x59f999u, 0x3f165444u}, {0x5affb2u, 0x3f15a05bu},
    {0x5b855cu, 0x3f15453fu}, {0x5bd86du, 0x3f150cd9u}, {0x5c90a9u, 0x3f149059u}, {0x5cd546u, 0x3f146230u},
    {0x5d9dc9u, 0x3f13dbf0u}, {0x5e14d9u, 0x3f138caau}, {0x5e8766u, 0x3f1340b6u}, {0x5f41cau, 0x3f12c5c6u},
    {0x5f5572u, 0x3f12b8dcu}, {0x60fd9bu, 0x3f11a440u}, {0x616499u, 0x3f1161b4u}, {0x625d1cu, 0x3f10c218u},
    {0x62c33au, 0x3f1080e8u}, {0x63c0e3u, 0x3f0fdff7u}, {0x641547u, 0x3f0faabcu}, {0x64b8d7u, 0x3f0f43feu},
    {0x651327u, 0x3f0f0b83u}, {0x658cdau, 0x3f0ebfadu}, {0x65dc9fu, 0x3f0e8e23u}, {0x662b59u, 0x3f0e5d60u},
    {0x66d228u, 0x3f0df67eu}, {0x670f27u, 0x3f0dd104u}, {0x67e4c4u, 0x3f0d4e61u}, {0x6916adu, 0x3f0c94eeu},
    {0x6a50dfu, 0x3f0bd86cu}, {0x6a7a5au, 0x3f0bbfaeu}, {0x6b0a18u, 0x3f0b6a37u}, {0x6b0d54u, 0x3f0b684cu},
    {0x6b4013u, 0x3f0b4a3au}, {0x6b4cb2u, 0x3f0b42c1u}, {0x6bd361u, 0x3f0af338u}, {0x6bf46cu, 0x3f0adfc3u},
    {0x6cad4au, 0x3f0a734au}, {0x6cb93du, 0x3f0a6c4cu}, {0x6d76b0u, 0x3f09fdddu}, {0x6dc676u, 0x3f09cf92u},
    {0x6e36aau, 0x3f098ea8u}, {0x6f0367u, 0x3f0918d3u}, {0x6fa5cau, 0x3f08bbedu}, {0x70c958u, 0x3f08165eu},
    {0x71e6c2u, 0x3f0775ccu}, {0x721583u, 0x3f075ba3u}, {0x72e6ccu, 0x3f06e703u}, {0x72fdf2u, 0x3f06da29u},
    {0x73596du, 0x3f06a778u}, {0x7403e4u, 0x3f064966u}, {0x744916u, 0x3f06235cu}, {0x74c9dfu, 0x3f05dccau},
    {0x74e69au, 0x3f05cd16u}, {0x756dbau, 0x3f05836cu}, {0x7631a9u, 0x3f05192au}, {0x7731afu, 0x3f048f50u},
    {0x7774a2u, 0x3f046b72u}, {0x791a7cu, 0x3f038b32u}, {0x795e82u, 0x3f036751u}, {0x7aa2c1u, 0x3f02bd52u},
    {0x7c2684u, 0x3f01f444u}, {0x7c64ebu, 0x3f01d422u}, {0x7d2cafu, 0x3f016db2u}, {0x7d959fu, 0x3f013822u},
    {0x7e62aau, 0x3f00cffbu}, {0x7ebff2u, 0x3f00a0d0u}, {0x7eecf4u, 0x3f008a1au}, {0x7f1505u, 0x3f0075eau},
    {0x7f1b10u, 0x3f0072dfu}, {0x7f2614u, 0x3f006d53u}, {0x7ff486u, 0x3f0005beu}, {0x7fffffu, 0x3f000001u},
}};
constexpr std::array<u32, 6> kRcpExponents{1u, 60u, 127u, 128u, 200u, 252u};

// RDNA 2 v_sin_f32 and v_cos_f32 of x (in turns).
struct SinCosRow {
  u32 x, sin, cos;
};
constexpr std::array<SinCosRow, 32> kSinCos{{
    {0x80800000u, 0x81c90fd5u, 0x3f800000u},
    {0x8f800000u, 0x90c90fd5u, 0x3f800000u},
    {0xa3d42337u, 0xa5269cc5u, 0x3f800000u},
    {0xb0800000u, 0xb1c90fd5u, 0x3f800000u},
    {0xb2800000u, 0xb3c90fd5u, 0x3f800000u},
    {0xb3000000u, 0xb4490fd5u, 0x3f800000u},
    {0xb3800000u, 0xb4c90fd5u, 0x3f800000u},
    {0xb3c00000u, 0xb516cbe0u, 0x3f800000u},
    {0xb8732b55u, 0xb9befc10u, 0x3f7ffffeu},
    {0xbc23d70au, 0xbd80984du, 0x3f7f7eadu},
    {0xbe82a4d7u, 0xbf7fdd81u, 0xbd04dfbau},
    {0xbe000000u, 0xbf3504f4u, 0x3f3504f3u},
    {0xbeb33333u, 0xbf4f1bbdu, 0xbf167918u},
    {0xbef5c28fu, 0xbe005760u, 0xbf7dfb3au},
    {0x00800000u, 0x01c90fd5u, 0x3f800000u},
    {0x30800000u, 0x31c90fd5u, 0x3f800000u},
    {0x33800000u, 0x34c90fd5u, 0x3f800000u},
    {0x3c23d70au, 0x3d80984du, 0x3f7f7eadu},
    {0x3e82a4d7u, 0x3f7fdd81u, 0xbd04dfbau},
    {0x3f3d70a4u, 0xbf7f7eadu, 0xbd809845u},
    {0xbf3d70a4u, 0x3f7f7eadu, 0xbd809845u},
    {0x42c8999au, 0x3f73780du, 0xbe9e39ddu},
    {0xc2c8999au, 0xbf73780du, 0xbe9e39ddu},
    {0xc348b333u, 0x3f73780du, 0xbe9e39ddu},
    {0x80000000u, 0x80000000u, 0x3f800000u},
    {0x00000000u, 0x00000000u, 0x3f800000u},
    {0xbf000000u, 0x00000000u, 0xbf800000u},
    {0xbe800000u, 0xbf800000u, 0x00000000u},
    {0xbf800000u, 0x00000000u, 0x3f800000u},
    {0xcb000001u, 0x00000000u, 0x3f800000u},
    {0x7f7fffffu, 0x00000000u, 0x3f800000u},
    {0xff800000u, 0xffc00000u, 0xffc00000u},
}};

bool ExactRequired() {
  static const bool exact = [] {
    const char *value = std::getenv("KYTY_HW_TABLE_EXACT");
    return value != nullptr && value[0] == '1';
  }();
  return exact;
}

// results[k][i]: VOP1 opcode k of inputs[i]; 32 lanes per dispatch, one input per lane.
std::vector<std::vector<u32>> RunVop1(VulkanHarness &vulkan, const char *name,
                                      std::span<const u32> opcodes, std::span<const u32> inputs) {
  constexpr u32 lanes = 32;
  TestCase test;
  test.name = name;
  test.has_compute_info = true;
  test.compute_info.wave_size = lanes;
  test.compute_info.threads_num[0] = lanes;
  test.compute_info.threads_num[1] = test.compute_info.threads_num[2] = 1u;
  test.compute_info.thread_ids_num = 1u;
  auto &code = test.code;
  code.push_back(EncodeVop2(0x1a, 30, InlineU32(2), 0)); // v30 = lane * 4
  AppendBufferLoadDword(&code, 1, 30);
  for (u32 k = 0; k < opcodes.size(); k++) {
    code.push_back(EncodeVop1(opcodes[k], 2 + k, Vgpr(1)));
    code.insert(code.end(), {EncodeVop2(0x25, 31, 255, 30), (k + 1) * lanes * 4u}); // v31 = v30 + slot
    AppendBufferStoreDword(&code, 2 + k, 31);
  }
  AppendEnd(&code);
  test.initial.assign(lanes * (opcodes.size() + 1), 0xdeadbeefu);
  const auto compiled = CompileCase(test, vulkan.SubgroupSize());
  std::vector<std::vector<u32>> results(opcodes.size());
  for (size_t first = 0; first < inputs.size(); first += lanes) {
    auto words = test.initial;
    for (u32 lane = 0; lane < lanes; lane++) words[lane] = inputs[std::min(first + lane, inputs.size() - 1)];
    auto buffer = vulkan.CreateStorageBuffer(name, words, words.size());
    vulkan.Dispatch(test, compiled, buffer);
    const auto got = vulkan.ReadBuffer(name, buffer, words.size());
    vulkan.DestroyBuffer(&buffer);
    for (size_t k = 0; k < opcodes.size(); k++) {
      for (u32 lane = 0; lane < lanes && first + lane < inputs.size(); lane++) {
        results[k].push_back(got[(k + 1) * lanes + lane]);
      }
    }
  }
  return results;
}

// Distance in units in the last place between two finite floats of the same sign (else UINT32_MAX).
u32 UlpDistance(u32 a, u32 b) {
  if ((a ^ b) >> 31u) return a == b ? 0u : UINT32_MAX;
  return a > b ? a - b : b - a;
}

// v_rcp_f32: within 1 ulp of RDNA 2 on 3072 inputs (256 mantissas, 6 exponents, both signs).
void CheckRcp(VulkanHarness &vulkan) {
  constexpr const char *name = "HardwareTableRcpF32";
  std::vector<u32> inputs, expected;
  for (u32 sign = 0; sign < 2; sign++) {
    for (const auto exponent : kRcpExponents) {
      for (const auto &[mantissa, result] : kRcpMantissas) {
        inputs.push_back(sign << 31u | exponent << 23u | mantissa);
        expected.push_back(sign << 31u | ((result >> 23u) + 127u - exponent) << 23u | (result & 0x7fffffu));
      }
    }
  }
  const std::array<u32, 1> opcodes{0x2au};
  const auto got = RunVop1(vulkan, name, opcodes, inputs)[0];
  // Count every input before failing, so a failing run still reports how far the host is from RDNA 2.
  u32 exact = 0, worst = 0, failed = 0;
  std::string first;
  for (size_t i = 0; i < inputs.size(); i++) {
    const auto ulps = UlpDistance(got[i], expected[i]);
    exact += ulps == 0 ? 1 : 0;
    worst = std::max(worst, ulps);
    if (ulps > (ExactRequired() ? 0u : 1u) && failed++ == 0) {
      first = fmt::format("v_rcp_f32 of {:#010x} is {:#010x}, RDNA 2 gives {:#010x}", inputs[i], got[i], expected[i]);
    }
  }
  std::printf("[compute] %s: %u/%zu with the RDNA 2 bits, worst %u ulp, %u over the bound\n", name, exact,
              inputs.size(), worst, failed);
  Require(name, ExactRequired() ? "RDNA 2 bits" : "1 ulp of RDNA 2", failed == 0, first);
  std::printf("[compute] %-32s ok\n", name);
}

// v_sin_f32 / v_cos_f32: zero and infinite results exact, NaN a NaN; the others within 2^-10 (relative) of
// RDNA 2 when the host is that accurate on the small positive inputs, as the source table checks it, else
// within the Vulkan bound of Sin (2^-11 absolute), so a wrong reduction still fails on a coarse host.
void CheckSinCos(VulkanHarness &vulkan) {
  constexpr const char *name = "HardwareTableSinCosF32";
  std::vector<u32> inputs;
  for (const auto &row : kSinCos) inputs.push_back(row.x);
  const std::array<u32, 2> opcodes{0x35u, 0x36u};
  const auto got = RunVop1(vulkan, name, opcodes, inputs);
  const auto within = [](u32 actual, u32 expected) {
    const double value = std::bit_cast<float>(expected);
    return std::abs(static_cast<double>(std::bit_cast<float>(actual)) - value) <= std::abs(value) * 0x1p-10;
  };
  bool accurate = true;
  for (size_t i = 0; i < kSinCos.size(); i++) {
    const float x = std::bit_cast<float>(kSinCos[i].x);
    if (x > 0.0f && x < 0.5f) accurate = accurate && within(got[0][i], kSinCos[i].sin) && within(got[1][i], kSinCos[i].cos);
  }
  if (!accurate) {
    std::printf("[compute] %s: host sin/cos not within 2^-10 of RDNA 2 on small positive inputs, the other "
                "finite results checked against the Vulkan bound (2^-11 absolute)\n", name);
  }
  const auto within_vulkan = [](u32 actual, u32 expected) {
    return std::abs(double(std::bit_cast<float>(actual)) - double(std::bit_cast<float>(expected))) <= 0x1p-11;
  };
  u32 exact = 0, failed = 0;
  for (size_t i = 0; i < kSinCos.size(); i++) {
    for (u32 k = 0; k < 2; k++) {
      const auto want = k == 0 ? kSinCos[i].sin : kSinCos[i].cos;
      const auto have = got[k][i];
      const float value = std::bit_cast<float>(want);
      const bool bits = !std::isfinite(value) || value == 0.0f || ExactRequired();
      // Any NaN for a NaN (the payload is the host's) unless every bit is required.
      const bool nan = std::isnan(value) && !ExactRequired();
      exact += have == want ? 1 : 0;
      const bool ok = nan    ? std::isnan(std::bit_cast<float>(have))
                      : bits ? have == want
                             : accurate ? within(have, want) : within_vulkan(have, want);
      if (!ok) {
        // Every failing result, then one failure: the run shows the whole gap.
        std::printf("[compute] %s: %s of %#010x is %#010x, RDNA 2 gives %#010x (%s)\n", name,
                    k == 0 ? "v_sin_f32" : "v_cos_f32", kSinCos[i].x, have, want,
                    bits ? "bits" : accurate ? "2^-10" : "2^-11 absolute");
        failed++;
      }
    }
  }
  std::printf("[compute] %s: %u/%zu with the RDNA 2 bits, %u over the bound\n", name, exact, 2 * kSinCos.size(), failed);
  Require(name, "RDNA 2 table", failed == 0, fmt::format("{} results over the bound", failed));
  std::printf("[compute] %-32s ok\n", name);
}

} // namespace HardwareTables
