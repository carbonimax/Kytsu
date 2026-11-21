#include "graphics/host_gpu/renderer/pipeline/descriptorHeap.h"

#include "common/assert.h"
#include "common/profiler.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/masterSemaphore.h"
#include "graphics/host_gpu/renderer/perfStats.h"

namespace Libs::Graphics {
namespace {

constexpr uint32_t DescriptorHeapCount = 1024;
// Sized for layouts with large texture arrays: a set must fit in a fresh pool.
constexpr std::array DescriptorPoolSizes = {
    vk::DescriptorPoolSize {vk::DescriptorType::eStorageBuffer, 32768},
    vk::DescriptorPoolSize {vk::DescriptorType::eSampledImage, 65536},
    vk::DescriptorPoolSize {vk::DescriptorType::eStorageImage, 8192},
    vk::DescriptorPoolSize {vk::DescriptorType::eSampler, 8192},
};

} // namespace

DescriptorHeap::DescriptorHeap(GraphicContext& graphics, MasterSemaphore& master_semaphore)
    : m_graphics(graphics), m_master_semaphore(master_semaphore) {
	CreateDescriptorPool();
}

DescriptorHeap::~DescriptorHeap() {
	m_graphics.device.destroyDescriptorPool(m_current_pool, nullptr);
	for (const auto& [pool, tick]: m_pending_pools) {
		m_master_semaphore.Wait(tick);
		m_graphics.device.destroyDescriptorPool(pool, nullptr);
	}
}

vk::DescriptorSet DescriptorHeap::Commit(vk::DescriptorSetLayout layout) {
	KYTY_PROFILER_FUNCTION();
	auto& batch = m_sets[layout];
	if (batch.size != 0) {
		return batch.sets[--batch.size];
	}

	std::array<vk::DescriptorSetLayout, DescriptorSetBatch> layouts;
	layouts.fill(layout);
	vk::DescriptorSetAllocateInfo allocate {};
	allocate.descriptorPool     = m_current_pool;
	allocate.descriptorSetCount = DescriptorSetBatch;
	allocate.pSetLayouts        = layouts.data();
	const auto result = m_graphics.device.allocateDescriptorSets(&allocate, batch.sets.data());
	if (result == vk::Result::eSuccess) {
		batch.size = DescriptorSetBatch;
		PerfStats::Add(PerfStats::Counter::DescSetAllocs, DescriptorSetBatch);
		return batch.sets[--batch.size];
	}
	EXIT_IF(result != vk::Result::eErrorOutOfPoolMemory &&
	        result != vk::Result::eErrorFragmentedPool);

	// A failed multi-set allocation can leave a pool unable to allocate even one set until it is
	// reset, so the rest of this pool and a fresh one are filled one set per call.
	const auto fill = [&](Batch& target) {
		while (target.size < DescriptorSetBatch &&
		       m_graphics.device.allocateDescriptorSets(&allocate, &target.sets[target.size]) ==
		           vk::Result::eSuccess) {
			target.size++;
		}
		PerfStats::Add(PerfStats::Counter::DescSetAllocs, target.size);
		return target.size != 0;
	};
	allocate.descriptorSetCount = 1;
	if (fill(batch)) {
		return batch.sets[--batch.size];
	}

	PerfStats::Add(PerfStats::Counter::DescPoolFull, 1);
	m_pending_pools.emplace_back(m_current_pool, m_master_semaphore.CurrentTick());
	m_master_semaphore.Refresh();
	if (const auto& [pool, tick] = m_pending_pools.front(); m_master_semaphore.IsFree(tick)) {
		m_current_pool = pool;
		m_pending_pools.pop_front();
		EXIT_IF(m_graphics.device.resetDescriptorPool(m_current_pool, {}) != vk::Result::eSuccess);
		PerfStats::Add(PerfStats::Counter::DescPoolResets, 1);
	} else {
		CreateDescriptorPool();
	}

	m_sets.clear();
	auto& fresh_batch       = m_sets[layout];
	allocate.descriptorPool = m_current_pool;
	EXIT_IF(!fill(fresh_batch));
	return fresh_batch.sets[--fresh_batch.size];
}

void DescriptorHeap::CreateDescriptorPool() {
	vk::DescriptorPoolCreateInfo create {};
	// Ray-query layouts carry one unused acceleration-structure binding (AddRayQueryLayoutBinding).
	std::vector<vk::DescriptorPoolSize> sizes(DescriptorPoolSizes.begin(), DescriptorPoolSizes.end());
	if (m_graphics.ray_query_enabled) {
		sizes.push_back({vk::DescriptorType::eAccelerationStructureKHR, DescriptorHeapCount});
	}
	create.maxSets       = DescriptorHeapCount;
	create.poolSizeCount = static_cast<uint32_t>(sizes.size());
	create.pPoolSizes    = sizes.data();
	EXIT_IF(m_graphics.device.createDescriptorPool(&create, nullptr, &m_current_pool) !=
	        vk::Result::eSuccess);
	PerfStats::Add(PerfStats::Counter::DescPoolCreates, 1);
}

} // namespace Libs::Graphics
