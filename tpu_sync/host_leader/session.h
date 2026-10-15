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

#ifndef TPU_SYNC_HOST_LEADER_SESSION_H_
#define TPU_SYNC_HOST_LEADER_SESSION_H_
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "tpu_sync/host_leader/memory.h"
#include "tpu_sync/host_leader/protocol.h"
namespace tpu_raiden::host_leader {
struct IoResult {
  absl::Status status;
  uint64_t bytes = 0;
};
struct PrepareRequest {
  uint64_t ticket = 0;
  Direction direction = Direction::kWrite;
  uint64_t bytes = 0;
  std::string metadata;
};
struct Prepared {
  uint64_t cookie = 0;
  std::vector<Span> spans;
};
class EndpointHandler {
 public:
  virtual ~EndpointHandler() = default;
  // Immediate lookup/pin only; return FailedPrecondition for NotReady.
  // Session retains returned spans through I/O, then drops them before Finish.
  // Handler retains application pin; retain Region too if Finish accesses it.
  virtual absl::StatusOr<Prepared> Prepare(const PrepareRequest& request) = 0;
  // Exactly once, after payload I/O stopped; never runs on Session reader.
  virtual absl::Status Finish(uint64_t cookie, const IoResult& result) = 0;
};
struct JobRequest {
  Direction direction = Direction::kWrite;
  Peer peer;
  std::string endpoint;
  std::string metadata;
  std::vector<Span> spans;
  // Queue/data timeout; TCP connect retains ConnectToPeer's fixed 3s bound.
  std::chrono::milliseconds timeout{120000};
};
class Job {
 public:
  ~Job();
  Job(const Job&) = delete;
  Job& operator=(const Job&) = delete;
  uint64_t id() const;
  // Queued jobs cancel immediately; active jobs drain, then report Cancelled.
  absl::Status Cancel();
  // Terminal always fences local I/O. Success includes responder Finish ACK;
  // network failure cannot prove remote application callback completion.
  IoResult Wait();

 private:
  class Impl;
  explicit Job(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
  friend class Session;
};
class Endpoint {
 public:
  ~Endpoint();
  Endpoint(const Endpoint&) = delete;
  Endpoint& operator=(const Endpoint&) = delete;
  // Synchronous drain requires an external owner; own callback returns
  // FailedPrecondition. Destruction in a callback delegates drain to a closer.
  absl::Status Close();

 private:
  class Impl;
  explicit Endpoint(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
  friend class Session;
};
class Session : public std::enable_shared_from_this<Session> {
 public:
  // '@name' is Linux abstract UDS; no per-worker listener is created.
  static absl::StatusOr<std::shared_ptr<Session>> Connect(
      const std::string& socket_name);
  static absl::StatusOr<std::shared_ptr<Session>> ForProcess(
      const std::string& socket_name);
  ~Session();
  absl::StatusOr<std::shared_ptr<Region>> AllocateRegion(size_t bytes);
  // Adopt resize-sealed memfd (F_SEAL_GROW|F_SEAL_SHRINK), map and register
  // via SCM_RIGHTS; all error paths close fd.
  absl::StatusOr<std::shared_ptr<Region>> RegisterRegion(Fd fd, size_t bytes);
  absl::StatusOr<std::unique_ptr<Endpoint>> RegisterEndpoint(
      const std::string& key, std::weak_ptr<EndpointHandler> handler);
  // Job owns spans. Session completion state never owns worker Region objects.
  absl::StatusOr<std::unique_ptr<Job>> Submit(JobRequest request);
  // External-owner drain of jobs/Finish and daemon mappings before ACK/close.
  // Synchronous Close from this Session callback returns FailedPrecondition.
  absl::Status Close();

 private:
  class Impl;
  explicit Session(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
  friend class Region;
  friend class Job;
  friend class Endpoint;
};
// Stateless request, excluded from worker Session count.
absl::StatusOr<Stats> GetStats(const std::string& socket_name);
}  // namespace tpu_raiden::host_leader
#endif
