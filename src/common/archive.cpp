#include "common/archive.h"

#include "common/stringUtils.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <condition_variable>
#include <cstring>
#include <fstream>
#include <limits>
#include <mutex>
#include <optional>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <utility>
#include <zarchive/zarchivereader.h>

// POSIX hosts and MinGW's pthread-backed libstdc++ have cancellable condition waits.
// MSVC/ClangCL and MinGW's native Win32 threading backend do not use pthread waits.
#if !defined(_WIN32) || defined(_GLIBCXX_GCC_GTHR_POSIX_H)
#include <pthread.h>
#define KYTY_ARCHIVE_PTHREAD_CANCELLATION
#endif

namespace Common {

// Backends expose immutable metadata and offset-based reads. Cursor management, path routing,
// reader lifetime and the host-stack worker are shared by every archive format.
class ArchiveReader {
public:
	struct Entry {
		uint64_t id;
		uint64_t size;
		bool     is_file;
	};

	virtual ~ArchiveReader()                                                       = default;
	virtual std::optional<Entry>        Find(std::string_view member)              = 0;
	virtual std::vector<File::DirEntry> List(std::string_view member)              = 0;
	virtual uint64_t Read(uint64_t id, uint64_t offset, uint32_t size, void* data) = 0;
};

namespace {

class ZArchiveReaderBackend final: public ArchiveReader {
public:
	explicit ZArchiveReaderBackend(std::unique_ptr<ZArchiveReader> reader)
	    : m_reader(std::move(reader)) {}

	std::optional<Entry> Find(std::string_view member) override {
		const auto node = m_reader->LookUp(member);
		if (node == ZARCHIVE_INVALID_NODE) {
			return {};
		}
		return Entry {node, m_reader->GetFileSize(node), m_reader->IsFile(node)};
	}

	std::vector<File::DirEntry> List(std::string_view member) override {
		const auto node = m_reader->LookUp(member);
		if (!m_reader->IsDirectory(node)) {
			return {};
		}
		const auto count = m_reader->GetDirEntryCount(node);
		if (count > (1u << 20u)) {
			return {};
		}
		std::vector<File::DirEntry> result;
		result.reserve(count);
		for (uint32_t index = 0; index < count; ++index) {
			ZArchiveReader::DirEntry entry {};
			if (m_reader->GetDirEntry(node, index, entry)) {
				result.push_back({std::string(entry.name), entry.isFile});
			}
		}
		return result;
	}

	uint64_t Read(uint64_t id, uint64_t offset, uint32_t size, void* data) override {
		// ZArchiveReader already serializes access to its stream and decompression cache.
		return m_reader->ReadFromFile(static_cast<ZArchiveNodeHandle>(id), offset, size, data);
	}

private:
	std::unique_ptr<ZArchiveReader> m_reader;
};

std::shared_ptr<ArchiveReader> OpenZArchive(const std::filesystem::path& path) {
	auto reader = std::unique_ptr<ZArchiveReader>(ZArchiveReader::OpenFromFile(path));
	return reader ? std::make_shared<ZArchiveReaderBackend>(std::move(reader)) : nullptr;
}

// Read-only raw exFAT volume image (not a VHD/VHDX container). The image is user input: every
// offset, length and cluster chain is checked against the volume before use.
class ExfatReaderBackend final: public ArchiveReader {
public:
	static std::shared_ptr<ArchiveReader> Open(const std::filesystem::path& path) {
		auto reader = std::make_shared<ExfatReaderBackend>();
		return reader->Init(path) ? reader : nullptr;
	}

	std::optional<Entry> Find(std::string_view member) override {
		Node node;
		if (!FindNode(member, &node)) {
			return {};
		}
		if (!node.directory && !PrepareForRead(&node)) {
			return {};
		}
		// One id per looked-up member path, so the table is bounded by the paths the guest uses.
		std::lock_guard lock(m_cache_mutex);
		auto [it, added] = m_ids.try_emplace(std::string(member), m_nodes.size());
		if (added) {
			m_nodes.push_back(node);
		}
		return Entry {it->second, node.directory ? 0 : node.data_length, !node.directory};
	}

	std::vector<File::DirEntry> List(std::string_view member) override {
		Node                        directory;
		std::vector<Node>           children;
		std::vector<File::DirEntry> result;
		if (FindNode(member, &directory) && ReadDirectory(directory, &children)) {
			result.reserve(children.size());
			for (const auto& child: children) {
				result.push_back({child.name, !child.directory});
			}
		}
		return result;
	}

	uint64_t Read(uint64_t id, uint64_t offset, uint32_t size, void* data) override {
		Node node;
		{
			std::lock_guard lock(m_cache_mutex);
			if (id >= m_nodes.size()) {
				return 0;
			}
			node = m_nodes[id];
		}
		if (node.directory || offset >= node.data_length) {
			return 0;
		}
		auto     remaining = std::min<uint64_t>(size, node.data_length - offset);
		auto*    out       = static_cast<uint8_t*>(data);
		uint64_t read      = 0;
		while (remaining != 0) {
			const auto index      = offset / m_cluster_size;
			const auto in_cluster = offset % m_cluster_size;
			const auto amount =
			    static_cast<size_t>(std::min(remaining, m_cluster_size - in_cluster));
			const auto cluster = node.contiguous ? uint64_t(node.first_cluster) + index
			                     : (node.clusters != nullptr && index < node.clusters->size())
			                         ? uint64_t((*node.clusters)[static_cast<size_t>(index)])
			                         : uint64_t(0);
			uint64_t cluster_offset = 0;
			if (cluster > UINT32_MAX ||
			    !ClusterOffset(static_cast<uint32_t>(cluster), &cluster_offset)) {
				break;
			}
			// Bytes past the valid data length read as zero, as on exFAT itself.
			const auto valid = offset < node.valid_length
			                       ? static_cast<size_t>(
			                             std::min<uint64_t>(amount, node.valid_length - offset))
			                       : 0;
			if (valid != 0 && !ReadAt(cluster_offset + in_cluster, out, valid)) {
				break;
			}
			std::memset(out + valid, 0, amount - valid);
			out += amount;
			offset += amount;
			remaining -= amount;
			read += amount;
		}
		return read;
	}

private:
	struct Node {
		std::string                            name;
		uint32_t                               first_cluster = 0;
		uint64_t                               data_length   = 0;
		uint64_t                               valid_length  = 0;
		bool                                   contiguous    = false;
		bool                                   directory     = false;
		std::shared_ptr<std::vector<uint32_t>> clusters;
	};

	static uint16_t U16(const uint8_t* p) { return uint16_t(p[0] | (p[1] << 8)); }
	static uint32_t U32(const uint8_t* p) { return uint32_t(U16(p)) | (uint32_t(U16(p + 2)) << 16); }
	static uint64_t U64(const uint8_t* p) { return uint64_t(U32(p)) | (uint64_t(U32(p + 4)) << 32); }

	static bool SameName(std::string_view a, std::string_view b) {
		return std::equal(a.begin(), a.end(), b.begin(), b.end(), [](char x, char y) {
			return std::tolower(static_cast<unsigned char>(x)) ==
			       std::tolower(static_cast<unsigned char>(y));
		});
	}

	bool ReadAt(uint64_t offset, void* data, size_t length) {
		if (offset > m_image_size || length > m_image_size - offset ||
		    offset > static_cast<uint64_t>(std::numeric_limits<std::streamoff>::max())) {
			return false;
		}
		std::lock_guard lock(m_file_mutex);
		m_file.clear();
		m_file.seekg(static_cast<std::streamoff>(offset));
		m_file.read(static_cast<char*>(data), static_cast<std::streamsize>(length));
		return m_file.good() || (length == 0 && !m_file.bad());
	}

	bool Init(const std::filesystem::path& path) {
		std::error_code ec;
		m_image_size = std::filesystem::file_size(path, ec);
		if (ec || m_image_size < 512) {
			return false;
		}
		m_file.open(path, std::ios::binary);
		std::array<uint8_t, 512> boot {};
		if (!m_file || !ReadAt(0, boot.data(), boot.size()) ||
		    std::memcmp(boot.data() + 3, "EXFAT   ", 8) != 0 || U16(boot.data() + 510) != 0xaa55) {
			return false;
		}
		const auto sector_shift  = boot[108];
		const auto cluster_shift = boot[109];
		if (sector_shift < 9 || sector_shift > 12 || cluster_shift > 25 - sector_shift ||
		    boot[110] != 1 || U16(boot.data() + 104) != 0x0100) {
			return false;
		}
		const uint64_t sector_size = uint64_t(1) << sector_shift;
		m_cluster_size             = sector_size << cluster_shift;
		m_fat_offset               = uint64_t(U32(boot.data() + 80)) * sector_size;
		const auto fat_length      = uint64_t(U32(boot.data() + 84)) * sector_size;
		m_heap_offset              = uint64_t(U32(boot.data() + 88)) * sector_size;
		m_cluster_count            = U32(boot.data() + 92);
		m_root_cluster             = U32(boot.data() + 96);
		const auto volume_sectors  = U64(boot.data() + 72);
		if (volume_sectors == 0 || volume_sectors > m_image_size / sector_size) {
			return false;
		}
		m_image_size = volume_sectors * sector_size;
		if (m_cluster_count == 0 || m_root_cluster < 2 ||
		    uint64_t(m_root_cluster) >= uint64_t(m_cluster_count) + 2 ||
		    m_fat_offset > m_image_size || fat_length > m_image_size - m_fat_offset ||
		    (uint64_t(m_cluster_count) + 2) * 4 > fat_length || m_heap_offset > m_image_size ||
		    uint64_t(m_cluster_count) > (m_image_size - m_heap_offset) / m_cluster_size) {
			return false;
		}
		const auto fat_bytes = (uint64_t(m_cluster_count) + 2) * 4;
		if (fat_bytes <= 64ull * 1024 * 1024) {
			m_fat_cache.resize(static_cast<size_t>(fat_bytes));
			if (!ReadAt(m_fat_offset, m_fat_cache.data(), m_fat_cache.size())) {
				return false;
			}
		}
		return true;
	}

	bool ClusterOffset(uint32_t cluster, uint64_t* offset) const {
		if (cluster < 2 || uint64_t(cluster) >= uint64_t(m_cluster_count) + 2) {
			return false;
		}
		*offset = m_heap_offset + uint64_t(cluster - 2) * m_cluster_size;
		return *offset <= m_image_size && m_cluster_size <= m_image_size - *offset;
	}

	bool NextCluster(uint32_t cluster, uint32_t* next) {
		if (cluster < 2 || uint64_t(cluster) >= uint64_t(m_cluster_count) + 2) {
			return false;
		}
		if (!m_fat_cache.empty()) {
			*next = U32(m_fat_cache.data() + size_t(cluster) * 4);
			return true;
		}
		uint8_t value[4] {};
		if (!ReadAt(m_fat_offset + uint64_t(cluster) * 4, value, sizeof(value))) {
			return false;
		}
		*next = U32(value);
		return true;
	}

	bool PrepareForRead(Node* node) {
		if (node->contiguous || node->clusters != nullptr || node->data_length == 0) {
			return true;
		}
		std::lock_guard lock(m_cache_mutex);
		if (const auto it = m_chain_cache.find(node->first_cluster); it != m_chain_cache.end()) {
			node->clusters = it->second;
			return true;
		}
		const uint64_t count = (node->data_length - 1) / m_cluster_size + 1;
		if (count > m_cluster_count) {
			return false;
		}
		auto chain = std::make_shared<std::vector<uint32_t>>();
		chain->reserve(static_cast<size_t>(count));
		uint32_t cluster = node->first_cluster;
		for (uint64_t i = 0; i < count; ++i) {
			uint64_t offset = 0;
			if (!ClusterOffset(cluster, &offset)) {
				return false;
			}
			chain->push_back(cluster);
			if (i + 1 < count && (!NextCluster(cluster, &cluster) || cluster >= 0xfffffff8)) {
				return false;
			}
		}
		node->clusters                     = std::move(chain);
		m_chain_cache[node->first_cluster] = node->clusters;
		return true;
	}

	bool ReadDirectory(const Node& directory, std::vector<Node>* entries) {
		entries->clear();
		if (!directory.directory) {
			return false;
		}
		if (directory.data_length == 0 && directory.first_cluster == 0) {
			return true;
		}
		{
			std::lock_guard lock(m_cache_mutex);
			if (const auto it = m_directory_cache.find(directory.first_cluster);
			    it != m_directory_cache.end()) {
				*entries = it->second;
				return true;
			}
		}
		// Directory entries are small; cap malformed images before allocating memory.
		constexpr uint64_t   max_directory_bytes = 64ull * 1024 * 1024;
		std::vector<uint8_t> bytes;
		uint32_t             cluster   = directory.first_cluster;
		uint64_t             remaining = directory.data_length;
		for (uint64_t i = 0; i < m_cluster_count && bytes.size() < max_directory_bytes; ++i) {
			uint64_t offset = 0;
			if (!ClusterOffset(cluster, &offset)) {
				return false;
			}
			const auto size = static_cast<size_t>(
			    directory.data_length == 0 ? m_cluster_size : std::min(m_cluster_size, remaining));
			if (size > max_directory_bytes - bytes.size()) {
				return false;
			}
			const auto old = bytes.size();
			bytes.resize(old + size);
			if (!ReadAt(offset, bytes.data() + old, size)) {
				return false;
			}
			if (directory.data_length != 0) {
				remaining -= size;
				if (remaining == 0) {
					break;
				}
			}
			if (directory.contiguous) {
				++cluster;
			} else if (!NextCluster(cluster, &cluster)) {
				return false;
			} else if (cluster >= 0xfffffff8) {
				if (directory.data_length != 0 && remaining != 0) {
					return false;
				}
				break;
			}
		}
		for (size_t i = 0; i + 32 <= bytes.size();) {
			const auto* primary = bytes.data() + i;
			if (primary[0] == 0) {
				break;
			}
			// Only in-use file entries (0x85); bitmap, up-case, label and deleted entries are skipped.
			if (primary[0] != 0x85) {
				i += 32;
				continue;
			}
			const auto secondaries = primary[1];
			if (secondaries < 2 || secondaries > 18 ||
			    size_t(secondaries + 1) * 32 > bytes.size() - i) {
				return false;
			}
			const auto* stream = primary + 32;
			if (stream[0] != 0xc0) {
				i += size_t(secondaries + 1) * 32;
				continue;
			}
			const auto     name_length = stream[3];
			std::u16string name;
			for (unsigned j = 2; j <= secondaries && name.size() < name_length; ++j) {
				const auto* part = primary + j * 32;
				if (part[0] != 0xc1) {
					return false;
				}
				for (unsigned k = 0; k < 15 && name.size() < name_length; ++k) {
					name.push_back(static_cast<char16_t>(U16(part + 2 + k * 2)));
				}
			}
			if (name.size() != name_length || name.empty()) {
				return false;
			}
			Node node;
			try {
				node.name = Utf16ToUtf8(name);
			} catch (...) {
				return false;
			}
			if (node.name == "." || node.name == ".." ||
			    node.name.find_first_of(std::string_view("/\\\0", 3)) != std::string::npos) {
				return false;
			}
			node.first_cluster = U32(stream + 20);
			node.data_length   = U64(stream + 24);
			node.valid_length  = U64(stream + 8);
			node.contiguous    = (stream[1] & 2) != 0;
			node.directory     = (U16(primary + 4) & 0x10) != 0;
			if (node.valid_length > node.data_length ||
			    (node.data_length != 0 &&
			     (node.first_cluster < 2 ||
			      uint64_t(node.first_cluster) >= uint64_t(m_cluster_count) + 2))) {
				return false;
			}
			entries->push_back(std::move(node));
			i += size_t(secondaries + 1) * 32;
		}
		std::lock_guard lock(m_cache_mutex);
		m_directory_cache[directory.first_cluster] = *entries;
		return true;
	}

	// member is already normalized by ParsePath: no empty, "." or ".." components.
	bool FindNode(std::string_view member, Node* node) {
		Node current;
		current.directory     = true;
		current.first_cluster = m_root_cluster;
		while (!member.empty()) {
			const auto slash = member.find('/');
			const auto name  = member.substr(0, slash);
			member.remove_prefix(slash == std::string_view::npos ? member.size() : slash + 1);
			std::vector<Node> children;
			if (!current.directory || !ReadDirectory(current, &children)) {
				return false;
			}
			const auto it = std::find_if(children.begin(), children.end(),
			                             [name](const Node& child) { return SameName(child.name, name); });
			if (it == children.end()) {
				return false;
			}
			current = *it;
		}
		*node = std::move(current);
		return true;
	}

	std::ifstream        m_file;
	std::mutex           m_file_mutex;
	uint64_t             m_image_size    = 0;
	uint64_t             m_cluster_size  = 0;
	uint64_t             m_fat_offset    = 0;
	uint64_t             m_heap_offset   = 0;
	uint32_t             m_cluster_count = 0;
	uint32_t             m_root_cluster  = 0;
	std::vector<uint8_t> m_fat_cache;
	// Guards the caches and the id table; ids stay valid for the reader's lifetime.
	std::mutex                                                           m_cache_mutex;
	std::unordered_map<uint32_t, std::shared_ptr<std::vector<uint32_t>>> m_chain_cache;
	std::unordered_map<uint32_t, std::vector<Node>>                      m_directory_cache;
	std::unordered_map<std::string, uint64_t>                            m_ids;
	std::vector<Node>                                                    m_nodes;
};

struct ArchiveFormat {
	std::string_view extension;
	std::shared_ptr<ArchiveReader> (*open)(const std::filesystem::path&);
};

constexpr ArchiveFormat Formats[] = {{".zar", OpenZArchive}, {".exfat", ExfatReaderBackend::Open}};
using NativeView                  = std::basic_string_view<std::filesystem::path::value_type>;

const ArchiveFormat* FindFormat(NativeView path) {
	for (const auto& format: Formats) {
		if (path.size() < format.extension.size()) {
			continue;
		}
		const auto extension = path.substr(path.size() - format.extension.size());
		if (std::equal(
		        extension.begin(), extension.end(), format.extension.begin(),
		        [](auto a, char b) { return (a >= 'A' && a <= 'Z' ? a + ('a' - 'A') : a) == b; })) {
			return &format;
		}
	}
	return nullptr;
}

size_t FindMarker(NativeView path) {
	// Native files use this check on every operation. The usual path has no marker and allocates
	// nothing.
	for (auto marker = path.find('!'); marker != NativeView::npos;
	     marker      = path.find('!', marker + 1)) {
		if ((marker + 1 == path.size() || path[marker + 1] == '/' || path[marker + 1] == '\\') &&
		    FindFormat(path.substr(0, marker)) != nullptr) {
			return marker;
		}
	}
	return NativeView::npos;
}

struct ParsedPath {
	std::filesystem::path archive;
	std::string           member;
};

std::optional<ParsedPath> ParsePath(const std::filesystem::path& path) {
	const auto& native = path.native();
	const auto  marker = FindMarker(native);
	if (marker == NativeView::npos) {
		return {};
	}
	ParsedPath parsed {.archive = std::filesystem::path(native.substr(0, marker))};
	auto       member = PathToGenericString(std::filesystem::path(native.substr(marker + 1)));
	std::replace(member.begin(), member.end(), '\\', '/');
	std::string_view remaining = member;
	while (!remaining.empty()) {
		const auto separator = remaining.find('/');
		const auto component = remaining.substr(0, separator);
		remaining.remove_prefix(separator == std::string_view::npos ? remaining.size() : separator + 1);
		if (component.empty() || component == ".") {
			continue;
		}
		if (component == "..") {
			if (parsed.member.empty()) {
				return {};
			}
			const auto separator = parsed.member.rfind('/');
			parsed.member.resize(separator == std::string::npos ? 0 : separator);
		} else {
			if (!parsed.member.empty()) {
				parsed.member += '/';
			}
			parsed.member += component;
		}
	}
	return parsed;
}

std::mutex                                                              g_readers_mutex;
std::unordered_map<std::filesystem::path, std::weak_ptr<ArchiveReader>> g_readers;

// Guest stacks can be only 64 KiB. Decompress on an ordinary host stack, then copy into guest
// memory on the caller so GPU-tracked writes use its fault handler. A single worker bounds host
// threads and avoids duplicating the archive's own block cache. Requests live on waiting callers.
class ArchiveIoThread {
public:
	static ArchiveIoThread& Instance() {
		// Guest exit can run static destructors while other guest threads are still reading.
		// Keep the worker alive until process termination, as with the guest threads themselves.
		static auto* instance = new ArchiveIoThread;
		return *instance;
	}

	uint64_t Read(ArchiveReader& reader, uint64_t id, uint64_t offset, uint32_t size, void* data) {
		Request          request {reader, id, offset, size, data};
		std::unique_lock lock(m_mutex);
		if (m_last != nullptr) {
			m_last->next = &request;
		} else {
			m_first = &request;
		}
		m_last = &request;
		m_wake.notify_one();
		request.ready.wait(lock, [&] { return request.done; });
		return request.result;
	}

	KYTY_CLASS_NO_COPY(ArchiveIoThread);

private:
	struct Request {
		ArchiveReader&          reader;
		uint64_t                id;
		uint64_t                offset;
		uint32_t                size;
		void*                   data;
		Request*                next   = nullptr;
		uint64_t                result = 0;
		bool                    done   = false;
		std::condition_variable ready {};
	};

	ArchiveIoThread() {
		std::thread([this] { Loop(); }).detach();
	}

	void Loop() {
		for (;;) {
			Request* request;
			{
				std::unique_lock lock(m_mutex);
				m_wake.wait(lock, [this] { return m_first != nullptr; });
				request = m_first;
				m_first = request->next;
				if (m_first == nullptr) {
					m_last = nullptr;
				}
			}
			const auto result =
			    request->reader.Read(request->id, request->offset, request->size, request->data);
			{
				std::lock_guard lock(m_mutex);
				request->result = result;
				request->done   = true;
				// Notify before releasing the mutex: the waiting caller owns the condition
				// variable.
				request->ready.notify_one();
			}
		}
	}

	std::mutex              m_mutex;
	std::condition_variable m_wake;
	Request*                m_first = nullptr;
	Request*                m_last  = nullptr;
};

} // namespace

struct ArchiveFile::Private {
	std::shared_ptr<ArchiveReader> reader;
	ArchiveReader::Entry           entry;
	uint64_t                       position = 0;
};

ArchiveFile::ArchiveFile(std::unique_ptr<Private> p): m_p(std::move(p)) {}
ArchiveFile::~ArchiveFile() = default;

uint64_t ArchiveFile::Size() const {
	return m_p->entry.size;
}
uint64_t ArchiveFile::Tell() const {
	return m_p->position;
}
bool ArchiveFile::Seek(uint64_t offset) {
	m_p->position = offset;
	return true;
}

void ArchiveFile::Read(void* data, uint32_t size, uint32_t* bytes_read) {
#if defined(KYTY_ARCHIVE_PTHREAD_CANCELLATION)
	// A pending cancellation must not destroy the waiting request or its TLS buffer while the
	// worker still uses them. Restore the caller's state only after the copy is complete.
	int cancellation_state = PTHREAD_CANCEL_ENABLE;
	pthread_setcancelstate(PTHREAD_CANCEL_DISABLE, &cancellation_state);
#endif
	constexpr uint32_t                chunk_size = 1u << 20u;
	thread_local std::vector<uint8_t> chunk;
	uint32_t                          read = 0;
	if (data != nullptr && size != 0 && m_p->position < Size()) {
		const auto requested =
		    static_cast<uint32_t>(std::min<uint64_t>(size, Size() - m_p->position));
		if (chunk.size() < std::min(chunk_size, requested)) {
			chunk.resize(std::min(chunk_size, requested));
		}
		while (read < requested) {
			const auto step = std::min(chunk_size, requested - read);
			const auto got  = ArchiveIoThread::Instance().Read(*m_p->reader, m_p->entry.id,
			                                                   m_p->position, step, chunk.data());
			if (got == 0 || got > step) {
				break;
			}
			std::memcpy(static_cast<uint8_t*>(data) + read, chunk.data(), got);
			read += static_cast<uint32_t>(got);
			m_p->position += got;
			if (got < step) {
				break;
			}
		}
	}
	if (bytes_read != nullptr) {
		*bytes_read = read;
	}
#if defined(KYTY_ARCHIVE_PTHREAD_CANCELLATION)
	pthread_setcancelstate(cancellation_state, nullptr);
#endif
}

std::filesystem::path MakeArchivePath(const std::filesystem::path& archive,
                                      const std::filesystem::path& member) {
	auto root = archive;
	root += "!";
	return member.empty() ? root : root / member.relative_path();
}

bool IsSupportedArchive(const std::filesystem::path& path) {
	return FindFormat(path.native()) != nullptr;
}

bool IsArchivePath(const std::filesystem::path& path) {
	return FindMarker(path.native()) != NativeView::npos;
}

std::filesystem::path GetArchiveHostPath(const std::filesystem::path& path) {
	const auto marker = FindMarker(path.native());
	return marker == NativeView::npos ? std::filesystem::path {}
	                                  : std::filesystem::path(path.native().substr(0, marker));
}

std::shared_ptr<ArchiveReader> OpenArchive(const std::filesystem::path& path) {
	const auto  archive = IsArchivePath(path) ? GetArchiveHostPath(path) : path;
	const auto* format  = FindFormat(archive.native());
	if (format == nullptr) {
		return {};
	}
	std::error_code error;
	auto            key = std::filesystem::absolute(archive, error);
	if (error) {
		return {};
	}
	{
		std::lock_guard lock(g_readers_mutex);
		if (auto it = g_readers.find(key); it != g_readers.end()) {
			if (auto reader = it->second.lock()) {
				return reader;
			}
		}
	}
	// Do not hold the cache mutex while opening an unrelated archive from disk.
	auto reader = format->open(archive);
	if (reader == nullptr) {
		return {};
	}
	std::lock_guard lock(g_readers_mutex);
	std::erase_if(g_readers, [](const auto& entry) { return entry.second.expired(); });
	auto& cached = g_readers[key];
	if (auto existing = cached.lock()) {
		return existing;
	}
	cached = reader;
	return reader;
}

std::optional<File::Info> GetArchiveInfo(const std::filesystem::path& path) {
	const auto parsed = ParsePath(path);
	if (parsed) {
		if (auto reader = OpenArchive(parsed->archive)) {
			if (auto entry = reader->Find(parsed->member)) {
				return File::Info {entry->is_file, entry->size};
			}
		}
	}
	return {};
}

std::vector<File::DirEntry> GetArchiveDirEntries(const std::filesystem::path& path) {
	const auto parsed = ParsePath(path);
	if (parsed) {
		if (auto reader = OpenArchive(parsed->archive)) {
			return reader->List(parsed->member);
		}
	}
	return {};
}

std::unique_ptr<ArchiveFile> OpenArchiveFile(const std::filesystem::path& path) {
	auto parsed = ParsePath(path);
	if (!parsed) {
		return {};
	}
	auto       reader = OpenArchive(parsed->archive);
	const auto entry  = reader ? reader->Find(parsed->member) : std::nullopt;
	if (!entry || !entry->is_file) {
		return {};
	}
	auto p     = std::make_unique<ArchiveFile::Private>();
	p->reader  = std::move(reader);
	p->entry   = *entry;
	return std::unique_ptr<ArchiveFile>(new ArchiveFile(std::move(p)));
}

} // namespace Common
