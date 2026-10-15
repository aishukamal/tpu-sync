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

#ifndef TPU_SYNC_HOST_LEADER_INTERNAL_H_
#define TPU_SYNC_HOST_LEADER_INTERNAL_H_
#include <chrono>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

#include "absl/status/status.h"
#include "tpu_sync/host_leader/protocol.h"
namespace tpu_raiden::host_leader::internal {
Fd AllocateMemory(size_t bytes);
enum class Type : uint64_t {
  kHello = 1,
  kRegisterRegion,
  kRetireRegion,
  kRegisterEndpoint,
  kCloseEndpoint,
  kSubmit,
  kCancel,
  kClose,
  kStats,
  kPrepare,
  kFinish,
  kReply,
  kDone
};
struct Frame {
  Type type = Type::kReply;
  uint64_t id = 0;
  absl::Status status;
  std::vector<uint64_t> words;
  std::string key, data;
  std::vector<WireSpan> spans;
  Peer peer;
  Fd fd;
};
using Deadline = std::chrono::steady_clock::time_point;
inline Deadline InfiniteDeadline() { return Deadline::max(); }
void Read(int fd, void* ptr, size_t size,
          Deadline deadline = InfiniteDeadline());
void Write(int fd, const void* ptr, size_t size,
           Deadline deadline = InfiniteDeadline());
void SendPacket(int fd, const Frame& frame);
Frame ReceivePacket(int fd);
void SendFrame(int fd, const Frame& frame, Deadline deadline);
Frame ReceiveFrame(int fd, Deadline deadline);
int ConnectUnix(const std::string& address);
int ListenUnix(const std::string& address);
int ListenTcp(const std::string& endpoint);
void Throw(const absl::Status& status);
absl::Status Error(const std::exception& error);
Frame Reply(uint64_t id, absl::Status status = absl::OkStatus());
std::vector<uint64_t> EncodeStats(const Stats& stats);
Stats DecodeStats(const std::vector<uint64_t>& values);
class Executor {
 public:
  explicit Executor(size_t threads);
  ~Executor();
  void Post(std::function<void()> fn);
  void Drain();

 private:
  std::mutex mu_;
  std::condition_variable cv_, drained_;
  std::queue<std::function<void()>> queue_;
  std::vector<std::thread> workers_;
  size_t active_ = 0;
  bool stopping_ = false;
};
}  // namespace tpu_raiden::host_leader::internal
#endif
