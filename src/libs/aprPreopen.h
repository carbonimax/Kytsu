#ifndef EMULATOR_SRC_LIBS_APRPREOPEN_H_
#define EMULATOR_SRC_LIBS_APRPREOPEN_H_

#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <functional>
#include <mutex>
#include <thread>
#include <unordered_set>
#include <vector>

// KYTY_APR_PREOPEN=1 (default off). The APR engine reads on one thread and its first open of a file is slow
// when the host cache is cold (~2-4 ms each, ~28k files per level). This opens and closes the files of resolved
// paths and queued reads on a few threads ahead of the engine, so those cold opens overlap and the engine's own
// open is warm. No handle is kept: file changes, deletes and exit need nothing special.
namespace Libs::AprShared {

class AprPreopener {
public:
	using OpenFn = std::function<bool(uint32_t id)>; // opens and closes the file of `id`, false on failure

	AprPreopener(size_t threads, size_t max_pending, OpenFn open): m_max_pending(max_pending), m_open(std::move(open)) {
		for (size_t i = 0; i < threads; i++) {
			m_workers.emplace_back([this](std::stop_token stop) { Run(stop); });
		}
	}

	~AprPreopener() {
		{
			std::scoped_lock lock(m_mutex);
			for (auto& worker: m_workers) {
				worker.request_stop();
			}
		}
		m_cv.notify_all();
	}

	AprPreopener(const AprPreopener&)            = delete;
	AprPreopener& operator=(const AprPreopener&) = delete;

	// Each id is opened once; a full queue drops it and a failed open forgets it, so a later Queue retries.
	void Queue(uint32_t id) {
		{
			std::scoped_lock lock(m_mutex);
			if (m_pending.size() >= m_max_pending || !m_seen.insert(id).second) {
				return;
			}
			m_pending.push_back(id);
		}
		m_cv.notify_one();
	}

	// Tests: waits until nothing is queued or being opened.
	void WaitIdle() {
		std::unique_lock lock(m_mutex);
		m_idle_cv.wait(lock, [this] { return m_pending.empty() && m_busy == 0; });
	}

private:
	void Run(std::stop_token stop) {
		std::unique_lock lock(m_mutex);
		while (!stop.stop_requested()) {
			if (m_pending.empty()) {
				m_cv.wait(lock);
				continue;
			}
			const uint32_t id = m_pending.front();
			m_pending.pop_front();
			m_busy++;
			lock.unlock();
			const bool ok = m_open(id);
			lock.lock();
			if (!ok) {
				m_seen.erase(id);
			}
			m_busy--;
			if (m_pending.empty() && m_busy == 0) {
				m_idle_cv.notify_all();
			}
		}
	}

	std::mutex                   m_mutex;
	std::condition_variable      m_cv;
	std::condition_variable      m_idle_cv;
	std::deque<uint32_t>         m_pending;
	std::unordered_set<uint32_t> m_seen;
	size_t                       m_busy = 0;
	size_t                       m_max_pending;
	OpenFn                       m_open;
	std::vector<std::jthread>    m_workers; // last: joined before the members above are destroyed
};

} // namespace Libs::AprShared

#endif // EMULATOR_SRC_LIBS_APRPREOPEN_H_
