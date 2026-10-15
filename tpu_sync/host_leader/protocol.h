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

#ifndef TPU_SYNC_HOST_LEADER_PROTOCOL_H_
#define TPU_SYNC_HOST_LEADER_PROTOCOL_H_
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>
namespace tpu_raiden::host_leader {
enum class Direction : uint64_t { kWrite = 1, kRead = 2 };
struct Route {
  std::string local_ip;
  std::string remote_endpoint;
};
struct Peer {
  std::string host_key;
  std::vector<Route> routes;
};
struct WireSpan {
  uint64_t region_id = 0, offset = 0, bytes = 0;
};
struct Stats {
  uint64_t sessions = 0, endpoints = 0, registered_regions = 0,
           mapped_bytes = 0;
  uint64_t queued = 0, prepared = 0, inflight = 0, connections = 0;
};
class Fd {
 public:
  explicit Fd(int fd = -1) : fd_(fd) {}
  ~Fd();
  Fd(Fd&& other) noexcept;
  Fd& operator=(Fd&& other) noexcept;
  Fd(const Fd&) = delete;
  Fd& operator=(const Fd&) = delete;
  int get() const { return fd_; }
  int release();
  void reset(int fd = -1);

 private:
  int fd_;
};
}  // namespace tpu_raiden::host_leader
#endif
