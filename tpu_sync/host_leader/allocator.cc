// Copyright 2026 Google LLC.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "tpu_sync/host_leader/allocator.h"

#include <unistd.h>

#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <stdexcept>

#include "absl/status/status.h"
#include "absl/strings/match.h"
#include "tpu_sync/core/tpu_utils.h"
#include "tpu_sync/host_leader/session.h"

namespace tpu_raiden::host_leader {
namespace {
struct Allocation {
  std::shared_ptr<Region> region;
  uint8_t* ptr = nullptr;
  size_t bytes = 0;
  xla::PjRtClient* client = nullptr;
  bool dma_mapped = false;
  ~Allocation() {
    if (dma_mapped) client->DmaUnmap(ptr).IgnoreError();
  }
};
std::mutex allocations_mu;
std::map<uintptr_t, std::weak_ptr<Allocation>> allocations;
struct NumaPolicy {
  explicit NumaPolicy(const xla::PjRtDevice* device)
      : node(GetPjRtDeviceNumaNode(device)) {
    if (node >= 0) SetThreadMempolicy(1, node);
  }
  ~NumaPolicy() {
    if (node >= 0) SetThreadMempolicy(0);
  }
  int node;
};
}  // namespace
const std::string& SocketPath() {
  static const std::string path = [] {
    const char* value = std::getenv("TPU_RAIDEN_HOST_LEADER_SOCKET");
    if (value != nullptr && *value != '\0') {
      const char* reuse = std::getenv("TPU_RAIDEN_DATA_CONN_REUSE");
      if (reuse != nullptr && *reuse != '\0' &&
          !absl::EqualsIgnoreCase(reuse, "pooled")) {
        throw std::invalid_argument(
            "host leader requires TPU_RAIDEN_DATA_CONN_REUSE=pooled");
      }
    }
    return value == nullptr ? std::string() : std::string(value);
  }();
  return path;
}
HostBufferAllocator MakeAllocator(xla::PjRtClient* client) {
  return [client](size_t bytes, const xla::PjRtDevice* device)
             -> absl::StatusOr<HostBufferAllocation> {
    if (bytes == 0) return HostBufferAllocation{};
    const long page_size = sysconf(_SC_PAGESIZE);
    if (page_size <= 0) {
      return absl::InternalError("cannot determine host page size");
    }
    const size_t padding = (page_size - bytes % page_size) % page_size;
    if (bytes > SIZE_MAX - padding) {
      return absl::InvalidArgumentError(
          "host leader allocation size overflows");
    }
    auto allocation = std::make_shared<Allocation>();
    allocation->bytes = bytes + padding;
    allocation->client = client;
    NumaPolicy policy(device);
    auto session = Session::ForProcess(SocketPath());
    if (!session.ok()) return session.status();
    auto region = (*session)->AllocateRegion(allocation->bytes);
    if (!region.ok()) return region.status();
    allocation->region = std::move(*region);
    allocation->ptr = allocation->region->data();
    void* ptr = allocation->ptr;
    std::memset(ptr, 0, allocation->bytes);
    if (client != nullptr && client->platform_name() != "cpu") {
      auto status = client->DmaMap(ptr, allocation->bytes);
      if (!status.ok()) return status;
      allocation->dma_mapped = true;
    }
    {
      std::lock_guard<std::mutex> lock(allocations_mu);
      for (auto it = allocations.begin(); it != allocations.end();) {
        if (it->second.expired())
          it = allocations.erase(it);
        else
          ++it;
      }
      allocations[reinterpret_cast<uintptr_t>(ptr)] = allocation;
    }
    return HostBufferAllocation{allocation->ptr, bytes, allocation};
  };
}
RetainedSpan Describe(const uint8_t* ptr, size_t bytes) {
  std::lock_guard<std::mutex> lock(allocations_mu);
  uintptr_t address = reinterpret_cast<uintptr_t>(ptr);
  auto it = allocations.upper_bound(address);
  if (it == allocations.begin()) {
    throw std::runtime_error(
        "host leader push does not use its transient shared allocator");
  }
  auto allocation = (--it)->second.lock();
  if (allocation == nullptr || address < it->first ||
      address - it->first > allocation->bytes ||
      bytes > allocation->bytes - (address - it->first)) {
    throw std::runtime_error(
        "host leader chunk lies outside a live shared allocation");
  }
  return {{allocation->region, address - it->first, bytes}, allocation};
}
}  // namespace tpu_raiden::host_leader
