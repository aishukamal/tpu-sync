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

#ifndef TPU_SYNC_HOST_LEADER_MEMORY_H_
#define TPU_SYNC_HOST_LEADER_MEMORY_H_
#include <cstddef>
#include <cstdint>
#include <memory>

#include "absl/status/status.h"
namespace tpu_raiden::host_leader {
class Session;
// Size-sealed memfd. Owns its Session, without device or model dependencies.
class Region {
 public:
  ~Region();
  Region(const Region&) = delete;
  Region& operator=(const Region&) = delete;
  uint8_t* data() const;
  size_t size() const;
  uint64_t id() const;
  // Reject new users and drain admitted I/O before daemon unmap/close ACK.
  // Does not cancel unrelated jobs; normal RAII release follows last Job use.
  absl::Status Retire();

 private:
  class Impl;
  explicit Region(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
  friend class Session;
};
struct Span {
  std::shared_ptr<Region> region;
  uint64_t offset = 0, bytes = 0;
};
}  // namespace tpu_raiden::host_leader
#endif
