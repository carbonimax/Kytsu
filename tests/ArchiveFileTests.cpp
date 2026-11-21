#include "common/file.h"
#include "common/archive.h"
#include "common/stringUtils.h"
#include "ArchiveTestFixture.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <string>
#include <thread>
#include <vector>
#include <zarchive/zarchivecommon.h>

#if !defined(_WIN32) || defined(_GLIBCXX_GCC_GTHR_POSIX_H)
#include <pthread.h>
#endif

namespace {

void Check(bool value, const char *text) {
  if (!value) {
    std::fprintf(stderr, "ArchiveFileTests: failed: %s\n", text);
    std::abort();
  }
}

class TempDirectory {
public:
  TempDirectory() {
    const auto unique =
        std::chrono::steady_clock::now().time_since_epoch().count();
    m_path = std::filesystem::temp_directory_path() /
             ("kyty_zarchive_test_!_" + std::to_string(unique));
    Check(std::filesystem::create_directories(m_path),
          "create temporary directory");
  }

  ~TempDirectory() {
    std::error_code error;
    std::filesystem::remove_all(m_path, error);
  }

  [[nodiscard]] const std::filesystem::path &Path() const { return m_path; }

  KYTY_CLASS_NO_COPY(TempDirectory);

private:
  std::filesystem::path m_path;
};

template <typename T>
void SetValue(std::vector<std::byte> &bytes, size_t offset, T value) {
  value = _ZARCHIVE::_store(value);
  Check(offset <= bytes.size() && sizeof(value) <= bytes.size() - offset,
        "patch lies inside fixture");
  std::memcpy(bytes.data() + offset, &value, sizeof(value));
}

void SetMemberNameOffset(std::vector<std::byte> &bytes,
                         const _ZARCHIVE::Footer &footer, uint32_t name_offset) {
  const auto entry_offset = footer.sectionFileTree.offset + sizeof(_ZARCHIVE::FileDirectoryEntry);
  uint32_t original = 0;
  std::memcpy(&original, bytes.data() + entry_offset, sizeof(original));
  SetValue(bytes, entry_offset, (_ZARCHIVE::_load(original) & 0x80000000u) | name_offset);
}

template <typename Edit>
std::filesystem::path MutateArchive(const std::filesystem::path &source,
                                   const char *name, Edit edit) {
  Common::File input(source, Common::File::Mode::Read);
  auto bytes = input.ReadWholeBuffer();
  _ZARCHIVE::Footer footer{};
  Check(bytes.size() >= sizeof(footer), "fixture has a footer");
  std::memcpy(&footer, bytes.data() + bytes.size() - sizeof(footer), sizeof(footer));
  _ZARCHIVE::Footer::Deserialize(&footer, &footer);
  edit(bytes, footer);
  _ZARCHIVE::Footer::Serialize(&footer, &footer);
  std::memcpy(bytes.data() + bytes.size() - sizeof(footer), &footer, sizeof(footer));
  const auto destination = source.parent_path() / name;
  Common::File output;
  Check(output.Create(destination), "create malformed archive fixture");
  uint32_t written = 0;
  output.Write(bytes.data(), static_cast<uint32_t>(bytes.size()), &written);
  Check(written == bytes.size(), "write malformed archive fixture");
  return destination;
}

void CheckMalformedArchives(const std::filesystem::path &source,
                            const std::vector<uint8_t> &payload) {
  const auto bad_name = MutateArchive(source, "bad-name.zar", [](auto &bytes, auto &footer) {
    SetMemberNameOffset(bytes, footer, static_cast<uint32_t>(footer.sectionNames.size));
  });
  Check(!Common::OpenArchive(bad_name), "reject name offset at the end of name table");

  const auto bad_length = MutateArchive(source, "bad-length.zar", [](auto &bytes, auto &footer) {
    SetMemberNameOffset(bytes, footer, static_cast<uint32_t>(footer.sectionNames.size - 1));
    bytes[footer.sectionNames.offset + footer.sectionNames.size - 1] = std::byte{0x80};
  });
  Check(!Common::OpenArchive(bad_length), "reject truncated extended name length");

  const auto bad_children = MutateArchive(source, "bad-children.zar", [](auto &bytes, auto &footer) {
    SetValue(bytes, footer.sectionFileTree.offset + sizeof(uint32_t), UINT32_MAX);
  });
  Check(!Common::OpenArchive(bad_children), "reject invalid directory children without throwing");

  const auto bad_range = MutateArchive(source, "bad-range.zar", [](auto &, auto &footer) {
    footer.sectionCompressedData.offset = std::numeric_limits<uint64_t>::max();
  });
  Check(!Common::OpenArchive(bad_range), "reject overflowing footer section range");

  const auto truncated = source.parent_path() / "truncated.zar";
  Check(std::filesystem::copy_file(source, truncated), "copy truncated archive fixture");
  std::filesystem::resize_file(truncated, std::filesystem::file_size(truncated) - 1);
  Check(!Common::OpenArchive(truncated), "reject missing footer byte");

  const auto corrupted = MutateArchive(source, "corrupt-data.zar", [](auto &bytes, auto &footer) {
    _ZARCHIVE::CompressionOffsetRecord record{};
    std::memcpy(&record, bytes.data() + footer.sectionOffsetRecords.offset, sizeof(record));
    _ZARCHIVE::CompressionOffsetRecord::Deserialize(&record, 1, &record);
    const auto second_block = footer.sectionCompressedData.offset + record.baseOffset + record.size[0] + 1;
    std::fill_n(bytes.data() + second_block, record.size[1] + 1, std::byte{});
  });
  Common::File damaged(Common::MakeArchivePath(corrupted, "assets/subdir/data.bin"),
                       Common::File::Mode::Read);
  Check(!damaged.IsInvalid(), "open archive with damaged compressed data");
  std::vector<uint8_t> bytes(payload.size());
  uint32_t read = 0;
  damaged.Read(bytes.data(), static_cast<uint32_t>(bytes.size()), &read);
  Check(read > 0 && read < 64 * 1024 && damaged.Tell() == read &&
            std::equal(bytes.begin(), bytes.begin() + read, payload.begin()),
        "report the valid prefix before a corrupt compressed block");
  Check(damaged.Seek(0), "reset damaged archive offset");
  Check(!damaged.ReadIntoGuest(bytes.data(), static_cast<uint32_t>(bytes.size()), &read) && read > 0 && read < 64 * 1024 && damaged.Tell() == read && std::equal(bytes.begin(), bytes.begin() + read, payload.begin()), "guest read reports partial archive corruption without retry");
  Check(damaged.Seek(2 * 64 * 1024), "seek past corrupt compressed block");
  damaged.Read(bytes.data(), 97, &read);
  Check(read == 97 && std::equal(bytes.begin(), bytes.begin() + read,
                                 payload.begin() + 2 * 64 * 1024),
        "recover by reading an unaffected compressed block");
}

void WriteHostFile(const std::filesystem::path &path, const void *data, size_t size) {
  std::ofstream stream(path, std::ios::binary);
  stream.write(static_cast<const char *>(data), static_cast<std::streamsize>(size));
  Check(stream.good(), "write pack source file");
}

void CheckPackArchive(const std::filesystem::path &root, const std::vector<uint8_t> &payload) {
  const auto source = root / "pack-source";
  const auto unicode = Common::PathFromUtf8(ArchiveTests::UnicodeFilename);
  std::error_code error;
  Check(std::filesystem::create_directories(source / "assets/subdir", error) &&
            std::filesystem::create_directories(source / "empty", error),
        "create pack source tree");
  WriteHostFile(source / "eboot.bin", ArchiveTests::Eboot.data(), ArchiveTests::Eboot.size());
  WriteHostFile(source / "assets/subdir/data.bin", payload.data(), payload.size());
  WriteHostFile(source / unicode, nullptr, 0);
  uint64_t expected = ArchiveTests::Eboot.size() + payload.size();
#if !defined(_WIN32)
  std::filesystem::create_directory_symlink("assets", source / "linked", error);
  Check(!error, "link a pack source directory");
  expected += payload.size();
#endif

  const auto archive = root / "packed.zar";
  uint64_t done = 0;
  uint64_t total = 0;
  Check(Common::PackArchive(source, archive,
                            [&](uint64_t packed, uint64_t size) {
                              done = packed;
                              total = size;
                              return true;
                            }),
        "pack a host directory");
  Check(done == expected && total == expected, "report every packed byte");

  const auto packed = Common::MakeArchivePath(archive);
  const auto check_payload = [&](const std::filesystem::path &member) {
    Common::File file(packed / member, Common::File::Mode::Read);
    const auto data = file.ReadWholeBuffer();
    Check(data.size() == payload.size() &&
              std::memcmp(data.data(), payload.data(), payload.size()) == 0,
          "round-trip packed file contents");
  };
  check_payload("assets/subdir/data.bin");
#if !defined(_WIN32)
  check_payload("linked/subdir/data.bin");
#endif
  const auto empty_file = Common::File::GetInfo(packed / unicode);
  Check(empty_file && empty_file->is_file && empty_file->size == 0 &&
            Common::File::IsDirectoryExisting(packed / "empty"),
        "keep empty files and directories");

  const auto failed = root / "failed.zar";
  const auto rejected = [&](const std::filesystem::path &directory, bool proceed) {
    return !Common::PackArchive(directory, failed, [=](uint64_t, uint64_t) { return proceed; }) &&
           !std::filesystem::exists(failed, error);
  };
  Check(rejected(source, false), "cancelling removes the partial archive");
  Check(rejected(root / "missing", true), "reject a missing source directory");
#if defined(__linux__)
  // Archive names compare case-insensitively, so such a folder cannot be packed faithfully.
  WriteHostFile(source / "EBOOT.BIN", nullptr, 0);
  Check(rejected(source, true), "reject names that differ only in case");
#endif
}

#if !defined(_WIN32) || defined(_GLIBCXX_GCC_GTHR_POSIX_H)
void CheckCancellation(const std::filesystem::path &path,
                       const std::vector<uint8_t> &payload) {
  // Keep the file and destination on the parent thread: cancellation must only release the
  // reader thread's request and staging buffer after the worker has finished using them.
  struct CancelledRead {
    Common::File file;
    std::vector<uint8_t> data;
    std::atomic_bool ready = false;
    std::atomic_bool start = false;
    uint32_t bytes = 0;
  } read;
  Check(read.file.Open(path, Common::File::Mode::Read), "open cancellation fixture");
  read.data.resize(payload.size());
  pthread_t thread;
  Check(pthread_create(&thread, nullptr, [](void *opaque) -> void * {
    auto &read = *static_cast<CancelledRead *>(opaque);
    read.ready = true;
    while (!read.start.load()) {}
    read.file.Read(read.data.data(), static_cast<uint32_t>(read.data.size()), &read.bytes);
    pthread_testcancel();
    return nullptr;
  }, &read) == 0, "start cancellable archive reader");
  while (!read.ready.load()) {}
  Check(pthread_cancel(thread) == 0, "queue cancellation before the archive worker wait");
  read.start = true;
  void *result = nullptr;
  Check(pthread_join(thread, &result) == 0 && result == PTHREAD_CANCELED,
        "restore cancellation after the archive read");
  Check(read.bytes == payload.size() && read.data == payload,
        "finish worker reads and copies before releasing a cancelled caller");

  int previous = PTHREAD_CANCEL_ENABLE;
  int observed = PTHREAD_CANCEL_ENABLE;
  pthread_setcancelstate(PTHREAD_CANCEL_DISABLE, &previous);
  read.file.Seek(0);
  read.file.Read(read.data.data(), 97, &read.bytes);
  pthread_setcancelstate(PTHREAD_CANCEL_DISABLE, &observed);
  pthread_setcancelstate(previous, nullptr);
  Check(observed == PTHREAD_CANCEL_DISABLE && read.bytes == 97,
        "preserve a caller's disabled cancellation state");
}
#endif

void CheckGuestReads(const std::filesystem::path &directory, const std::filesystem::path &archive_member, const std::vector<uint8_t> &payload) {
  const auto native_path = directory / "guest-read.bin";
  Common::File output;
  Check(output.Create(native_path), "create native guest-read fixture");
  uint32_t written = 0;
  output.Write(payload.data(), static_cast<uint32_t>(payload.size()), &written);
  Check(written == payload.size(), "write native guest-read fixture");
  output.Close();
  for (const auto &path : {native_path, archive_member}) {
    Common::File input(path, Common::File::Mode::Read);
    std::vector<uint8_t> data(payload.size() + 29, 0xcd);
    uint32_t got = 99;
    Check(input.ReadIntoGuest(nullptr, 0, &got) && got == 0 && input.Tell() == 0, "zero read preserves offset");
    Check(input.Seek(17) && input.ReadIntoGuest(data.data(), static_cast<uint32_t>(data.size()), &got) && got == payload.size() - 17 && input.Tell() == payload.size() && std::equal(data.begin(), data.begin() + got, payload.begin() + 17) && data[got] == 0xcd, "multi-chunk guest read preserves prefix and EOF");
    Check(input.ReadIntoGuest(data.data(), 29, &got) && got == 0, "EOF is not an I/O error");
    Check(input.Seek(31) && input.ReadIntoGuest(data.data(), 97, &got) && got == 97 && input.Tell() == 128 && std::equal(data.begin(), data.begin() + got, payload.begin() + 31), "guest read resumes at the requested offset");
  }
#if !defined(_WIN32)
  Common::File non_file(directory, Common::File::Mode::Read);
  std::array<uint8_t, 97> data{};
  uint32_t got = 99;
  Check(!non_file.IsInvalid() && !non_file.ReadIntoGuest(data.data(), data.size(), &got) && got == 0 && non_file.Tell() == 0, "permanent host error must fail without retry or offset advance");
#endif
}

// Builds a small raw exFAT volume the way a formatter lays it out: 512-byte sectors and clusters,
// FAT at sector 1, cluster heap at sector 2, root directory in cluster 2 with the volume label,
// bitmap and up-case entries first and a deleted entry before the files.
std::vector<uint8_t> MakeExfatImage(uint32_t fragment_next) {
  constexpr uint32_t Clusters = 12;
  std::vector<uint8_t> image((2 + Clusters) * 512, 0);
  const auto put = [&](size_t offset, uint64_t value, unsigned bytes) {
    for (unsigned i = 0; i < bytes; ++i) image[offset + i] = static_cast<uint8_t>(value >> (8 * i));
  };
  std::memcpy(image.data() + 3, "EXFAT   ", 8);
  put(72, 2 + Clusters, 8); // Volume length in sectors.
  put(80, 1, 4);            // FAT offset.
  put(84, 1, 4);            // FAT length.
  put(88, 2, 4);            // Cluster heap offset.
  put(92, Clusters, 4);
  put(96, 2, 4);            // Root directory cluster.
  put(104, 0x0100, 2);      // Revision 1.0.
  image[108] = 9;           // 512-byte sectors.
  image[109] = 0;           // One sector per cluster.
  image[110] = 1;           // One FAT.
  put(510, 0xaa55, 2);
  const auto cluster = [](uint32_t index) { return size_t(2 + index - 2) * 512; };
  const auto fat = [&](uint32_t index, uint32_t next) { put(512 + index * 4, next, 4); };
  fat(0, 0xfffffff8);
  fat(1, 0xffffffff);
  fat(2, 11); // The root directory spans clusters 2 and 11.
  fat(11, 0xffffffff);
  fat(6, 8);
  fat(8, fragment_next);
  fat(9, 0xffffffff);

  const auto add = [&](size_t& offset, std::u16string_view name, bool directory, uint32_t first,
                       uint64_t length, uint64_t valid, bool contiguous, uint8_t type = 0x85) {
    const size_t names = (name.size() + 14) / 15;
    image[offset] = type;
    image[offset + 1] = static_cast<uint8_t>(1 + names);
    put(offset + 4, directory ? 0x10 : 0x20, 2);
    const size_t stream = offset + 32;
    image[stream] = type == 0x85 ? 0xc0 : 0x40;
    image[stream + 1] = static_cast<uint8_t>(0x01 | (contiguous ? 0x02 : 0));
    image[stream + 3] = static_cast<uint8_t>(name.size());
    put(stream + 8, valid, 8);
    put(stream + 20, first, 4);
    put(stream + 24, length, 8);
    for (size_t part = 0; part < names; ++part) {
      const size_t entry = stream + 32 * (part + 1);
      image[entry] = type == 0x85 ? 0xc1 : 0x41;
      for (size_t k = 0; k < 15 && part * 15 + k < name.size(); ++k) {
        put(entry + 2 + k * 2, name[part * 15 + k], 2);
      }
    }
    offset += 32 * (2 + names);
  };
  size_t root = cluster(2);
  image[root] = 0x83; // Volume label.
  image[root + 32] = 0x81; // Allocation bitmap.
  put(root + 32 + 20, 12, 4);
  put(root + 32 + 24, 2, 8);
  image[root + 64] = 0x82; // Up-case table.
  root += 96;
  add(root, u"gone.bin", false, 4, 11, 11, true, 0x05);
  add(root, u"eboot.bin", false, 4, 11, 11, true);
  add(root, u"sce_sys", true, 3, 512, 512, true);
  add(root, u"fragmented-member-name.bin", false, 6, 1300, 1100, false);
  root = cluster(11); // Cluster 2 is full (512 bytes of entries).
  add(root, u"データ.bin", false, 10, 4, 4, true);
  size_t sce_sys = cluster(3);
  add(sce_sys, u"param.json", false, 5, 15, 15, true);
  std::memcpy(image.data() + cluster(4), "ELF fixture", 11);
  std::memcpy(image.data() + cluster(5), "{\"title\":\"ex\"}", 15);
  std::memset(image.data() + cluster(7), 0xee, 512); // Not in the fragmented chain.
  for (uint32_t i = 0; i < 1300; ++i) {
    const uint32_t index = i / 512;
    const uint32_t at = static_cast<uint32_t>(cluster(index == 0 ? 6 : index == 1 ? 8 : 9)) + i % 512;
    image[at] = i < 1100 ? static_cast<uint8_t>(i * 7 + 3) : 0x5a; // Past valid length: must read 0.
  }
  std::memcpy(image.data() + cluster(10), "data", 4);
  return image;
}

void CheckExfatImage(const std::filesystem::path &directory) {
  const auto write = [&](const char *name, const std::vector<uint8_t> &bytes) {
    const auto path = directory / name;
    WriteHostFile(path, bytes.data(), bytes.size());
    return path;
  };
  const auto good = MakeExfatImage(9);
  const auto image = write("image.exfat", good);
  const auto root = Common::MakeArchivePath(image);
  Check(Common::IsSupportedArchive(image) && Common::IsSupportedArchive("GAME.EXFAT"),
        "exFAT images are a supported archive format");
  Check(Common::File::IsDirectoryExisting(root) &&
            Common::File::IsFileExisting(root / "EBOOT.BIN") &&
            Common::File::IsDirectoryExisting(root / "sce_sys") &&
            Common::File::IsFileExisting(root / "sce_sys/param.json") &&
            !Common::File::IsFileExisting(root / "gone.bin") &&
            !Common::File::IsFileExisting(root / "../image.exfat"),
        "exFAT lookup is case insensitive, skips deleted entries and rejects traversal");
  const auto entries = Common::File::GetDirEntries(root);
  const auto has = [&](std::string_view name, bool is_file) {
    return std::ranges::any_of(entries, [&](const auto &entry) {
      return entry.name == name && entry.is_file == is_file;
    });
  };
  Check(entries.size() == 4 && has("eboot.bin", true) && has("sce_sys", false) &&
            has("fragmented-member-name.bin", true) &&
            has("\xe3\x83\x87\xe3\x83\xbc\xe3\x82\xbf.bin", true),
        "exFAT root lists files, folders, long and UTF-16 names only");

  Common::File fragmented(root / "fragmented-member-name.bin", Common::File::Mode::Read);
  Check(!fragmented.IsInvalid() && fragmented.Size() == 1300, "open fragmented exFAT file");
  std::vector<uint8_t> data(1400, 0xcd);
  uint32_t got = 0;
  fragmented.Read(data.data(), static_cast<uint32_t>(data.size()), &got);
  bool same = got == 1300;
  for (uint32_t i = 0; same && i < 1300; ++i) {
    same = data[i] == (i < 1100 ? static_cast<uint8_t>(i * 7 + 3) : 0);
  }
  Check(same, "exFAT read follows the FAT chain and zero-fills past the valid length");
  Check(fragmented.Seek(500), "seek in fragmented exFAT file");
  fragmented.Read(data.data(), 30, &got);
  Check(got == 30 && data[0] == static_cast<uint8_t>(500 * 7 + 3) &&
            data[29] == static_cast<uint8_t>(529 * 7 + 3),
        "exFAT read crosses a cluster boundary");
  Common::File unicode(root / Common::PathFromUtf8("\xe3\x83\x87\xe3\x83\xbc\xe3\x82\xbf.bin"),
                       Common::File::Mode::Read);
  const auto unicode_data = unicode.ReadWholeBuffer();
  Common::File param(root / "sce_sys/param.json", Common::File::Mode::Read);
  const auto param_data = param.ReadWholeBuffer();
  Check(unicode_data.size() == 4 && std::memcmp(unicode_data.data(), "data", 4) == 0 &&
            param_data.size() == 15 && std::memcmp(param_data.data(), "{\"title\":\"ex\"}", 15) == 0,
        "read exFAT files by UTF-8 name and in a subfolder");
  Common::File writable;
  Check(!writable.Open(root / "eboot.bin", Common::File::Mode::ReadWrite),
        "exFAT members are read-only");

  auto bad_boot = good;
  bad_boot[3] = 'N';
  const auto bad_boot_root = Common::MakeArchivePath(write("bad-boot.exfat", bad_boot));
  auto bad_geometry = good;
  bad_geometry[92] = 200; // More clusters than the volume holds.
  const auto bad_geometry_root = Common::MakeArchivePath(write("bad-geometry.exfat", bad_geometry));
  const auto broken_chain_root =
      Common::MakeArchivePath(write("broken-chain.exfat", MakeExfatImage(0)));
  Check(!Common::File::IsFileExisting(bad_boot_root / "eboot.bin") &&
            !Common::File::IsFileExisting(bad_geometry_root / "eboot.bin") &&
            Common::File::IsFileExisting(broken_chain_root / "eboot.bin") &&
            !Common::File::IsFileExisting(broken_chain_root / "fragmented-member-name.bin"),
        "exFAT rejects a bad boot sector, bad geometry and a broken cluster chain");

  // The UTF-16 name of the root entry in cluster 11 starts at byte 66 of that cluster.
  constexpr size_t UnicodeName = 11 * 512 + 66;
  auto lone_surrogate = good;
  lone_surrogate[UnicodeName] = 0x00;
  lone_surrogate[UnicodeName + 1] = 0xd8; // High surrogate followed by a plain unit.
  auto trailing_surrogate = good;
  trailing_surrogate[UnicodeName + 12] = 0xff;
  trailing_surrogate[UnicodeName + 13] = 0xdb; // High surrogate as the last unit.
  Check(!Common::File::IsFileExisting(
            Common::MakeArchivePath(write("lone-surrogate.exfat", lone_surrogate)) / "eboot.bin") &&
            !Common::File::IsFileExisting(
                Common::MakeArchivePath(write("trailing-surrogate.exfat", trailing_surrogate)) /
                "eboot.bin"),
        "exFAT rejects a directory with a malformed UTF-16 name");
}

} // namespace

int main() {
  TempDirectory temporary;
  const auto archive_path =
      temporary.Path() / std::filesystem::path(u8"game-テスト.zar");

  std::vector<uint8_t> payload(3 * 1024 * 1024 + 37);
  for (size_t index = 0; index < payload.size(); index++) {
    payload[index] = static_cast<uint8_t>((index * 37 + 11) & 0xff);
  }
  Check(ArchiveTests::CreateArchive(archive_path, payload), "create archive fixture");

  const auto native_file = Common::File::GetInfo(archive_path);
  const auto native_directory = Common::File::GetInfo(temporary.Path() / "");
  Check(native_file && native_file->is_file &&
            native_file->size == std::filesystem::file_size(archive_path),
        "query native file type and size together");
  Check(native_directory && !native_directory->is_file && native_directory->size == 0,
        "query native directory with trailing separator");
  Check(!Common::File::GetInfo(archive_path / "") && Common::File::Size(archive_path / "") == 0 &&
            !Common::File::GetInfo(temporary.Path() / "missing.bin"),
        "reject missing native paths and files with a trailing separator");

  const auto root = Common::MakeArchivePath(archive_path);
  auto reader_pin = Common::OpenArchive(archive_path);
  Check(reader_pin && Common::OpenArchive(root) == reader_pin,
        "mounted archive pin reuses the same reader and index");
  const auto archive_file = Common::File::GetInfo(root / "assets/subdir/data.bin");
  const auto archive_directory = Common::File::GetInfo(root / "assets/");
  Check(archive_file && archive_file->is_file && archive_file->size == payload.size(),
        "query archive member type and size together");
  const auto host_times = Common::File::GetTimes(archive_path);
  const auto member_times = Common::File::GetTimes(root / "assets/subdir/data.bin");
  Check(host_times && member_times && member_times->write_ns == host_times->write_ns,
        "archive members report the host archive time");
  Check(archive_directory && !archive_directory->is_file && archive_directory->size == 0 &&
            !Common::File::GetInfo(root / "missing.bin"),
        "distinguish archive directories from missing members");
  Check(Common::IsSupportedArchive(archive_path) &&
            Common::IsSupportedArchive("GAME.ZAR") &&
            !Common::IsSupportedArchive("game.zip"),
        "select an archive backend by supported extension");
  Check(!Common::IsArchivePath(temporary.Path() / "ordinary!" / "asset.bin"),
        "ordinary host paths with exclamation marks stay host paths");
  Check(Common::IsArchivePath(root), "recognize virtual archive root");
  Check(Common::GetArchiveHostPath(root) == archive_path,
        "recover host archive path");
  Check(Common::File::IsDirectoryExisting(root), "find archive root directory");
  Check(Common::File::IsFileExisting(root / "EBOOT.BIN"),
        "lookup is case insensitive");
  Check(!Common::File::IsFileExisting(root / "missing.bin"),
        "reject missing member");
  Check(Common::IsArchivePath(root / "../outside.bin"),
        "retain archive classification for invalid traversal");
  Check(!Common::File::IsFileExisting(root / "../outside.bin"),
        "reject leading parent traversal");
  Check(Common::File::IsFileExisting(root / "assets/../eboot.bin"),
        "normalize internal parent traversal");
  Check(Common::File::IsFileExisting(root / ArchiveTests::LongFilename),
        "decode extended-length archive names");
  Check(Common::File::IsFileExisting(root / Common::PathFromUtf8(ArchiveTests::UnicodeFilename)),
        "lookup Unicode archive member");

  const auto entries = Common::File::GetDirEntries(root);
  Check(std::ranges::any_of(entries,
                            [](const auto &entry) {
                              return entry.is_file && entry.name == "eboot.bin";
                            }),
        "enumerate root file");
  Check(std::ranges::any_of(entries,
                            [](const auto &entry) {
                              return !entry.is_file && entry.name == "assets";
                            }),
        "enumerate root directory");
  Check(std::ranges::any_of(entries, [](const auto &entry) {
          return entry.is_file && entry.name == ArchiveTests::UnicodeFilename;
        }), "enumerate Unicode member names as UTF-8");

  const auto data_path = root / "assets/subdir/data.bin";
  Check(Common::File::Size(data_path) == payload.size(), "report member size");
  Common::File file(data_path, Common::File::Mode::Read);
  Check(!file.IsInvalid() && file.GetTimes() && file.GetTimes()->write_ns == host_times->write_ns,
        "open archive member reports the host archive time");
  Check(file.Remaining() == payload.size(), "report initial remaining bytes");
  Check(file.Seek(64 * 1024 - 19), "seek before compression block boundary");
  std::vector<uint8_t> slice(97);
  uint32_t bytes_read = 0;
  file.Read(slice.data(), static_cast<uint32_t>(slice.size()), &bytes_read);
  Check(bytes_read == slice.size(), "read across compression block boundary");
  Check(
      std::equal(slice.begin(), slice.end(), payload.begin() + 64 * 1024 - 19),
      "preserve data across compression block boundary");

  Check(file.Seek(payload.size() - 9), "seek near member end");
  file.Read(slice.data(), static_cast<uint32_t>(slice.size()), &bytes_read);
  Check(bytes_read == 9, "truncate read at member end");
  Check(file.IsEOF(), "report member end of file");
  Check(file.Seek(payload.size() + 100), "allow seek past member end");
  Check(file.Remaining() == 0, "report no bytes remaining past member end");
  file.Read(slice.data(), static_cast<uint32_t>(slice.size()), &bytes_read);
  Check(bytes_read == 0, "read past member end returns no data");

  Check(file.Seek(0), "seek to member start");
  std::vector<uint8_t> full(payload.size());
  file.Read(full.data(), static_cast<uint32_t>(full.size()), &bytes_read);
  Check(bytes_read == full.size(), "read a multi-chunk member");
  Check(full == payload, "preserve a multi-chunk member");

  std::atomic_bool concurrent_reads_ok = true;
  std::vector<std::thread> readers;
  for (uint32_t worker = 0; worker < 8; worker++) {
    readers.emplace_back([&, worker] {
      Common::File concurrent_file(data_path, Common::File::Mode::Read);
      std::array<uint8_t, 4096> concurrent_data{};
      for (uint32_t iteration = 0; iteration < 16; iteration++) {
        const auto offset =
            static_cast<size_t>(worker * 7919 + iteration * 12347) %
            (payload.size() - concurrent_data.size());
        uint32_t concurrent_bytes_read = 0;
        if (concurrent_file.IsInvalid() || !concurrent_file.Seek(offset)) {
          concurrent_reads_ok = false;
          return;
        }
        concurrent_file.Read(concurrent_data.data(), concurrent_data.size(),
                             &concurrent_bytes_read);
        if (concurrent_bytes_read != concurrent_data.size() ||
            !std::equal(concurrent_data.begin(), concurrent_data.end(),
                        payload.begin() +
                            static_cast<std::ptrdiff_t>(offset))) {
          concurrent_reads_ok = false;
          return;
        }
      }
    });
  }
  for (auto &reader : readers) {
    reader.join();
  }
  Check(concurrent_reads_ok, "read concurrently through independent file handles");

#if !defined(_WIN32) || defined(_GLIBCXX_GCC_GTHR_POSIX_H)
  CheckCancellation(data_path, payload);
#endif

#if !defined(_WIN32)
  // Guest threads have much smaller stacks than std::thread's default stack.
  pthread_attr_t attributes;
  Check(pthread_attr_init(&attributes) == 0 &&
            pthread_attr_setstacksize(&attributes, 64 * 1024) == 0,
        "configure a guest-sized stack");
  struct SmallStackRead {
    const std::filesystem::path &path;
    const std::vector<uint8_t> &expected;
    bool passed = false;
  } small_stack{data_path, payload};
  pthread_t thread;
  Check(pthread_create(&thread, &attributes, [](void *opaque) -> void * {
    auto &test = *static_cast<SmallStackRead *>(opaque);
    Common::File input(test.path, Common::File::Mode::Read);
    auto data = input.ReadWholeBuffer();
    test.passed = data.size() == test.expected.size() &&
                  std::memcmp(data.data(), test.expected.data(), data.size()) == 0;
    return nullptr;
  }, &small_stack) == 0, "start guest-sized-stack reader");
  pthread_attr_destroy(&attributes);
  Check(pthread_join(thread, nullptr) == 0 && small_stack.passed,
        "decompress a multi-block file on a guest-sized stack");
#endif

  Common::File metadata(root / "sce_sys/param.json", Common::File::Mode::Read);
  const auto metadata_data = metadata.ReadWholeBuffer();
  const std::string metadata_text(
      reinterpret_cast<const char *>(metadata_data.data()),
      metadata_data.size());
  Check(metadata_text.find("Archive Test") != std::string::npos,
        "read metadata member");

  Common::File writable;
  Check(!writable.Open(data_path, Common::File::Mode::ReadWrite),
        "reject writable member open");
  Check(!Common::File::DeleteFile(data_path), "reject member deletion");
  Check(!Common::File::CreateDirectory(root / "new-dir"),
        "reject member directory creation");
  Check(!file.Truncate(0) && !file.Unlink(), "reject descriptor mutations");
  const auto host_file = temporary.Path() / "host.bin";
  Common::File host_fixture;
  Check(host_fixture.Create(host_file), "create host mutation fixture");
  host_fixture.Close();
  Check(!Common::File::RenameFile(data_path, host_file) &&
            !Common::File::RenameFile(host_file, data_path),
        "reject rename operations involving archives");

  CheckGuestReads(temporary.Path(), data_path, payload);
  CheckMalformedArchives(archive_path, payload);

  file.Close();
  metadata.Close();
  Check(Common::OpenArchive(archive_path) == reader_pin,
        "mount pin retains the archive reader after the last member closes");
  reader_pin.reset();
  std::error_code remove_error;
  Check(std::filesystem::remove(archive_path, remove_error) && !remove_error,
        "release archive handle after the final reader closes");
  const std::array<uint8_t, 3> replacement{42, 7, 19};
  Check(ArchiveTests::CreateArchive(archive_path, replacement), "replace a closed archive");
  Common::File replaced(data_path, Common::File::Mode::Read);
  const auto replaced_data = replaced.ReadWholeBuffer();
  Check(replaced_data.size() == replacement.size() &&
            std::memcmp(replaced_data.data(), replacement.data(), replacement.size()) == 0,
        "reopening after the final reader closes sees replacement contents");

  CheckPackArchive(temporary.Path(), payload);
  CheckExfatImage(temporary.Path());

  std::printf("ArchiveFileTests: all cases passed\n");
  return 0;
}
