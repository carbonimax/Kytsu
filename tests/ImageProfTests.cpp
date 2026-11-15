#include "graphics/host_gpu/renderer/imageProf.h"

#include <cstdio>
#include <cstdlib>

namespace {

using namespace Libs::Graphics::ImageProf;

void Check(bool value, const char* message) {
	if (!value) {
		std::fprintf(stderr, "ImageProfTests: failed: %s\n", message);
		std::abort();
	}
}

Frame MakeFrame(uint64_t create_ns, uint64_t view_ns, uint64_t upload_ns, uint64_t bytes) {
	Frame f {};
	f[CreateNs]    = create_ns;
	f[ViewNs]      = view_ns;
	f[UploadNs]    = upload_ns;
	f[UploadBytes] = bytes;
	f[Uploads]     = bytes != 0 ? 1 : 0;
	return f;
}

} // namespace

int main() {
	Window w;
	w.Close(Frame {});
	Check(w.frames == 1 && CpuNs(w.at_max) == 0 && w.max_bytes == 0, "an empty frame changed the maximum");

	const auto heavy = MakeFrame(2'000'000, 1'000'000, 3'000'000, 10);
	const auto big   = MakeFrame(0, 500'000, 0, 100);
	w.Close(heavy);
	w.Close(big);
	Check(w.frames == 3, "frames were not counted");
	Check(CpuNs(w.at_max) == 6'000'000 && w.at_max[CreateNs] == 2'000'000 && w.at_max[UploadBytes] == 10,
	      "the most expensive frame was not kept whole (creation + views + upload)");
	Check(w.max_bytes == 100, "the largest upload frame is not the most expensive one");
	Check(w.sum[UploadBytes] == 110 && w.sum[ViewNs] == 1'500'000 && w.sum[Uploads] == 2, "window sums are wrong");

	w.Close(MakeFrame(0, 0, 7'000'000, 1));
	Check(CpuNs(w.at_max) == 7'000'000 && w.at_max[UploadBytes] == 1, "a later, more expensive frame did not win");
	w.Close(heavy);
	Check(CpuNs(w.at_max) == 7'000'000, "a cheaper frame replaced the maximum");

	// Report() starts the next window with `g_window = {}`.
	w = {};
	w.Close(big);
	Check(w.frames == 1 && CpuNs(w.at_max) == 500'000 && w.max_bytes == 100 && w.sum[UploadBytes] == 100,
	      "the previous window's maximum survived the reset");

	std::puts("ImageProfTests: all cases passed");
	return 0;
}
