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

#ifndef TPU_SYNC_HOST_LEADER_ALLOCATOR_H_
#define TPU_SYNC_HOST_LEADER_ALLOCATOR_H_

#include <memory>
#include <string>

#include "tpu_sync/core/host_memory_allocator.h"
#include "tpu_sync/host_leader/memory.h"

namespace tpu_raiden::host_leader {
const std::string& SocketPath();
HostBufferAllocator MakeAllocator(xla::PjRtClient* client);
// The retained owner prevents source allocation replacement while queued.
struct RetainedSpan {
  Span span;
  std::shared_ptr<void> owner;
};
RetainedSpan Describe(const uint8_t* ptr, size_t bytes);
}  // namespace tpu_raiden::host_leader
#endif
