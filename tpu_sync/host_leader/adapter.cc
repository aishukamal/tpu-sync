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

#include "tpu_sync/host_leader/adapter.h"

#include <algorithm>
#include <atomic>
#include <climits>
#include <cstdlib>
#include <limits>
#include <map>
#include <mutex>
#include <stdexcept>
#include <utility>

#include "absl/log/log.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_split.h"
#include "tpu_sync/core/numa_thread_pool.h"
#include "tpu_sync/host_leader/allocator.h"
#include "tpu_sync/host_leader/session.h"

namespace tpu_raiden::host_leader {
namespace {
void Check(const absl::Status& status) {
  if (!status.ok()) throw std::runtime_error(status.ToString());
}
std::string RankPeer(const std::string& ip, int port) {
  return ip.find(':') == std::string::npos ? absl::StrCat(ip, ":", port)
                                           : absl::StrCat("[", ip, "]:", port);
}
std::string PeerIp(const std::string& endpoint) {
  if (!endpoint.empty() && endpoint.front() == '[') {
    const auto end = endpoint.find(']');
    if (end != std::string::npos && end + 1 < endpoint.size() &&
        endpoint[end + 1] == ':')
      return endpoint.substr(1, end - 1);
  } else {
    const auto end = endpoint.rfind(':');
    if (end != std::string::npos) return endpoint.substr(0, end);
  }
  throw std::invalid_argument("host leader requires an IP:port rank endpoint");
}
bool SameRoutes(const Peer& lhs, const Peer& rhs) {
  if (lhs.routes.size() != rhs.routes.size()) return false;
  for (size_t i = 0; i < lhs.routes.size(); ++i) {
    if (lhs.routes[i].local_ip != rhs.routes[i].local_ip ||
        lhs.routes[i].remote_endpoint != rhs.routes[i].remote_endpoint)
      return false;
  }
  return true;
}
std::map<std::string, Peer> ReadPeers() {
  const char* value = std::getenv("TPU_RAIDEN_HOST_LEADER_PEERS");
  if (value == nullptr || *value == '\0')
    throw std::invalid_argument(
        "host leader requires explicit TPU_RAIDEN_HOST_LEADER_PEERS");
  std::map<std::string, Peer> peers, hosts;
  for (absl::string_view entry : absl::StrSplit(value, ';')) {
    const auto equal = entry.find('=');
    const auto at = entry.find('@', equal == entry.npos ? 0 : equal + 1);
    if (equal == entry.npos || equal == 0 || at == entry.npos ||
        at <= equal + 1)
      throw std::invalid_argument(
          "invalid host leader peer: expected "
          "rank_ip=host_key@local_ip>endpoint+...");
    const std::string alias(entry.substr(0, equal));
    Peer peer;
    peer.host_key = std::string(entry.substr(equal + 1, at - equal - 1));
    for (absl::string_view route : absl::StrSplit(entry.substr(at + 1), '+')) {
      const auto arrow = route.find('>');
      if (arrow == route.npos || arrow == 0 || arrow + 1 == route.size() ||
          route.find('>', arrow + 1) != route.npos)
        throw std::invalid_argument(
            "invalid host leader route: expected local_ip>endpoint");
      peer.routes.push_back({std::string(route.substr(0, arrow)),
                             std::string(route.substr(arrow + 1))});
    }
    auto [host, inserted] = hosts.emplace(peer.host_key, peer);
    if (!inserted && !SameRoutes(host->second, peer))
      throw std::invalid_argument(
          "host leader aliases for one host_key require identical ordered "
          "routes");
    if (!peers.emplace(alias, std::move(peer)).second)
      throw std::invalid_argument("duplicate host leader rank IP alias");
  }
  return peers;
}

// Only this bridge interprets PD metadata. The service sees an opaque blob.
struct PoolPush {
  uint64_t uuid = 0;
  uint64_t sender = 0;
  uint32_t pool = 0;
  std::vector<int> src_ids, dst_ids;
  std::vector<uint64_t> payload_bytes;
};
void Append(std::string& output, uint64_t number) {
  for (int shift = 56; shift >= 0; shift -= 8)
    output.push_back(static_cast<char>(number >> shift));
}
std::string Encode(const PoolPush& push) {
  std::string result;
  Append(result, 1);  // PD metadata version, independent of generic framing.
  Append(result, push.uuid);
  Append(result, push.sender);
  Append(result, push.pool);
  Append(result, push.src_ids.size());
  for (int id : push.src_ids) Append(result, id);
  for (int id : push.dst_ids) Append(result, id);
  Append(result, push.payload_bytes.size());
  for (uint64_t bytes : push.payload_bytes) Append(result, bytes);
  return result;
}
PoolPush Decode(absl::string_view input) {
  auto read = [&]() -> uint64_t {
    if (input.size() < 8)
      throw std::invalid_argument("truncated PD bridge metadata");
    uint64_t number = 0;
    for (int i = 0; i < 8; ++i)
      number = (number << 8) | static_cast<unsigned char>(input[i]);
    input.remove_prefix(8);
    return number;
  };
  if (read() != 1)
    throw std::invalid_argument("unknown PD bridge metadata version");
  PoolPush push;
  push.uuid = read();
  push.sender = read();
  const auto pool = read();
  if (pool > UINT32_MAX || push.sender > INT64_MAX)
    throw std::invalid_argument("PD bridge sender or pool exceeds its domain");
  push.pool = pool;
  const auto blocks = read();
  if (blocks == 0 || blocks > input.size() / 16)
    throw std::invalid_argument("invalid PD bridge block count");
  auto ids = [&](std::vector<int>& target) {
    target.reserve(blocks);
    for (uint64_t i = 0; i < blocks; ++i) {
      const auto id = read();
      if (id > INT_MAX)
        throw std::invalid_argument("invalid PD bridge block ID");
      target.push_back(id);
    }
  };
  ids(push.src_ids);
  ids(push.dst_ids);
  const auto payloads = read();
  if (payloads > input.size() / 8)
    throw std::invalid_argument("invalid PD bridge payload count");
  push.payload_bytes.reserve(payloads);
  for (uint64_t i = 0; i < payloads; ++i) push.payload_bytes.push_back(read());
  if (!input.empty())
    throw std::invalid_argument("trailing PD bridge metadata");
  return push;
}
}  // namespace

class Adapter::Impl : public EndpointHandler,
                      public std::enable_shared_from_this<Adapter::Impl> {
 public:
  explicit Impl(transport::BlockTransportDelegate* delegate)
      : delegate_(delegate), peers_(ReadPeers()) {
    auto session = Session::ForProcess(SocketPath());
    Check(session.status());
    session_ = std::move(*session);
  }
  void Register(int rank_port, absl::Span<const std::string> rank_ips) {
    try {
      for (const auto& ip : rank_ips) {
        auto endpoint = session_->RegisterEndpoint(RankPeer(ip, rank_port),
                                                   weak_from_this());
        Check(endpoint.status());
        endpoints_.push_back(std::move(*endpoint));
      }
    } catch (...) {
      Close();
      throw;
    }
  }
  void Close() {
    {
      std::lock_guard<std::mutex> lock(jobs_mu_);
      if (stopping_) return;
      stopping_ = true;
      for (auto* job : jobs_) job->Cancel().IgnoreError();
    }
    // Queued tasks observe stopping and fail without submitting; active jobs
    // cancel and Wait for both local I/O and the remote application's Finish.
    sends_.reset();
    for (auto& endpoint : endpoints_) {
      auto status = endpoint->Close();
      if (!status.ok())
        LOG(WARNING) << "PD host transport endpoint drain: " << status;
    }
    endpoints_.clear();
  }
  void Forget(uint64_t uuid) {
    std::lock_guard<std::mutex> lock(progress_mu_);
    progress_.erase(uuid);
  }
  Peer Resolve(const std::string& endpoint) const {
    const auto it = peers_.find(PeerIp(endpoint));
    if (it == peers_.end())
      throw std::invalid_argument(
          "rank IP has no explicit host leader peer mapping");
    return it->second;
  }
  void Start(std::vector<JobRequest> requests,
             std::vector<std::shared_ptr<void>> owners,
             std::vector<int> dst_ids,
             transport::lib::CompletionCallback callback) {
    sends_->Schedule([this, requests = std::move(requests),
                      owners = std::move(owners), dst_ids = std::move(dst_ids),
                      callback = std::move(callback)]() mutable {
      absl::Status status;
      for (auto& request : requests) {
        std::unique_ptr<Job> job;
        {
          std::lock_guard<std::mutex> lock(jobs_mu_);
          if (stopping_) {
            status = absl::CancelledError("PD host transport is closing");
            break;
          }
          auto submitted = session_->Submit(std::move(request));
          if (!submitted.ok()) {
            status = submitted.status();
            break;
          }
          job = std::move(*submitted);
          jobs_.push_back(job.get());
        }
        status = job->Wait().status;
        {
          std::lock_guard<std::mutex> lock(jobs_mu_);
          jobs_.erase(std::find(jobs_.begin(), jobs_.end(), job.get()));
        }
        if (!status.ok()) break;
      }
      // Callback is outside transport exception handling: never invoke twice
      // if application code throws after receiving its terminal result.
      if (callback) {
        if (status.ok())
          callback(std::move(dst_ids));
        else
          callback(status);
      }
    });
  }
  absl::StatusOr<Prepared> Prepare(const PrepareRequest& request) override {
    bool pinned = false;
    uint64_t uuid = 0;
    try {
      if (request.direction != Direction::kWrite)
        throw std::invalid_argument(
            "PD pool bridge accepts only explicit writes");
      PoolPush push = Decode(request.metadata);
      uuid = push.uuid;
      auto spec = delegate_->GetPoolPushProgressSpec(push.pool, uuid);
      Check(spec.status());
      const size_t shards = delegate_->num_shards();
      if (!spec->has_value() || (*spec)->expected_pushes == 0 ||
          (*spec)->expected_pools == 0 || shards == 0 ||
          push.dst_ids.size() > SIZE_MAX / shards ||
          push.payload_bytes.size() != shards * push.dst_ids.size())
        throw std::invalid_argument(
            "PD host transport requires a registered explicit pool push");
      Check(delegate_->BeginIncomingPush(uuid));
      pinned = true;
      auto context = std::make_unique<Incoming>();
      context->push = std::move(push);
      context->expected_pushes = (*spec)->expected_pushes;
      Prepared prepared;
      prepared.cookie = next_cookie_++;
      uint64_t total = 0;
      size_t index = 0;
      for (size_t shard = 0; shard < shards; ++shard) {
        for (size_t block = 0; block < context->push.dst_ids.size(); ++block) {
          int64_t dst = context->push.dst_ids[block];
          const uint64_t bytes = context->push.payload_bytes[index++];
          auto chunks = delegate_->GetBlockChunks(
              context->push.pool, shard, {&dst, 1}, bytes, uuid,
              context->push.sender, "", context->push.src_ids[block]);
          uint64_t planned = 0;
          for (const auto& chunk : chunks) {
            if (chunk.size > bytes - planned)
              throw std::invalid_argument(
                  "PD host transport receive size mismatch");
            auto retained = Describe(chunk.ptr, chunk.size);
            planned += chunk.size;
            prepared.spans.push_back(retained.span);
            context->regions.push_back(std::move(retained.span.region));
            context->owners.push_back(std::move(retained.owner));
          }
          if (planned != bytes || bytes == 0 || bytes > request.bytes - total)
            throw std::invalid_argument(
                "PD host transport receive size mismatch");
          total += bytes;
        }
      }
      if (total != request.bytes)
        throw std::invalid_argument(
            "PD host transport aggregate receive size mismatch");
      {
        std::lock_guard<std::mutex> lock(incoming_mu_);
        incoming_.emplace(prepared.cookie, std::move(context));
      }
      return prepared;
    } catch (const std::exception& error) {
      if (pinned) delegate_->EndIncomingPush(uuid).IgnoreError();
      return absl::InvalidArgumentError(error.what());
    }
  }
  absl::Status Finish(uint64_t cookie, const IoResult& result) override {
    std::unique_ptr<Incoming> context;
    {
      std::lock_guard<std::mutex> lock(incoming_mu_);
      const auto it = incoming_.find(cookie);
      if (it == incoming_.end())
        return absl::NotFoundError("PD host transport unknown receive cookie");
      context = std::move(it->second);
      incoming_.erase(it);
    }
    absl::Status status = result.status;
    if (status.ok()) {
      try {
        bool pool_complete;
        const auto& push = context->push;
        {
          std::lock_guard<std::mutex> lock(progress_mu_);
          auto& progress = progress_[push.uuid][push.pool];
          if (progress.expected == 0)
            progress.expected = context->expected_pushes;
          if (progress.expected != context->expected_pushes ||
              progress.landed >= progress.expected)
            throw std::invalid_argument(
                "PD host transport excess or inconsistent pool pushes");
          pool_complete = ++progress.landed == progress.expected;
        }
        if (pool_complete)
          Check(delegate_->OnPoolReceived(push.pool, push.uuid));
        Check(delegate_->OnBlocksReceived(push.dst_ids, push.uuid));
      } catch (const std::exception& error) {
        status = absl::InvalidArgumentError(error.what());
      }
    }
    auto ended = delegate_->EndIncomingPush(context->push.uuid);
    return status.ok() ? ended : status;
  }

 private:
  struct Progress {
    size_t landed = 0, expected = 0;
  };
  struct Incoming {
    PoolPush push;
    size_t expected_pushes;
    std::vector<std::shared_ptr<Region>> regions;
    std::vector<std::shared_ptr<void>> owners;
  };
  transport::BlockTransportDelegate* delegate_;
  const std::map<std::string, Peer> peers_;
  std::shared_ptr<Session> session_;
  std::vector<std::unique_ptr<Endpoint>> endpoints_;
  std::unique_ptr<NumaThreadPool> sends_ = std::make_unique<NumaThreadPool>(4);
  std::mutex jobs_mu_;
  bool stopping_ = false;
  std::vector<Job*> jobs_;
  std::mutex incoming_mu_;
  std::atomic<uint64_t> next_cookie_{1};
  std::map<uint64_t, std::unique_ptr<Incoming>> incoming_;
  std::mutex progress_mu_;
  std::map<uint64_t, std::map<uint32_t, Progress>> progress_;
};

Adapter::Adapter(transport::BlockTransportDelegate* delegate, int rank_port,
                 absl::Span<const std::string> rank_ips)
    : impl_(std::make_shared<Impl>(delegate)) {
  impl_->Register(rank_port, rank_ips);
}
Adapter::~Adapter() { impl_->Close(); }
void Adapter::Forget(uint64_t uuid) { impl_->Forget(uuid); }
absl::StatusOr<transport::lib::Status> Adapter::Poll(transport::lib::Handle) {
  return absl::UnimplementedError(
      "PD host transport uses completion callbacks");
}
absl::StatusOr<transport::lib::Handle> Adapter::Post(
    absl::Span<const std::string> peers,
    absl::Span<const transport::lib::Request> requests,
    absl::Span<const int> src_ids, absl::Span<const int> dst_ids,
    transport::lib::CompletionCallback callback) {
  auto fail = [&callback](const std::string& message)
      -> absl::StatusOr<transport::lib::Handle> {
    auto status = absl::InvalidArgumentError(message);
    if (callback) callback(status);
    return status;
  };
  if (requests.empty() || peers.size() != 1 || src_ids.empty() ||
      src_ids.size() != dst_ids.size() || requests.front().socket_opcode != 6 ||
      requests.front().major_order != 0 || requests.front().parallelism <= 0 ||
      requests.front().local_id == UINT32_MAX ||
      static_cast<size_t>(requests.front().parallelism) > src_ids.size())
    return fail(
        "PD host transport supports only layer-major explicit pool writes to "
        "one rank");
  try {
    const auto& first = requests.front();
    const Peer peer = impl_->Resolve(peers.front());
    const size_t p = first.parallelism;
    std::vector<JobRequest> jobs;
    std::vector<std::shared_ptr<void>> owners;
    size_t request_index = 0;
    for (size_t stream = 0; stream < p; ++stream) {
      const size_t blocks = src_ids.size() / p + (stream < src_ids.size() % p);
      const size_t offset =
          stream * (src_ids.size() / p) + std::min(stream, src_ids.size() % p);
      PoolPush push;
      push.uuid = first.uuid;
      push.sender = first.remote_id;
      push.pool = first.local_id;
      push.src_ids.assign(src_ids.begin() + offset,
                          src_ids.begin() + offset + blocks);
      push.dst_ids.assign(dst_ids.begin() + offset,
                          dst_ids.begin() + offset + blocks);
      JobRequest job;
      job.peer = peer;
      job.endpoint = peers.front();
      while (request_index < requests.size() &&
             requests[request_index].stream_idx == static_cast<int>(stream)) {
        const auto id = requests[request_index].request_id;
        uint64_t bytes = 0;
        do {
          const auto& chunk = requests[request_index++];
          if (chunk.uuid != first.uuid || chunk.remote_id != first.remote_id ||
              chunk.local_id != first.local_id ||
              chunk.parallelism != first.parallelism ||
              chunk.socket_opcode != 6 || chunk.major_order != 0 ||
              chunk.len == 0 || chunk.len > UINT64_MAX - bytes)
            throw std::invalid_argument(
                "PD host transport inconsistent request group");
          auto retained = Describe(chunk.laddr, chunk.len);
          bytes += chunk.len;
          job.spans.push_back(std::move(retained.span));
          owners.push_back(std::move(retained.owner));
        } while (request_index < requests.size() &&
                 requests[request_index].stream_idx ==
                     static_cast<int>(stream) &&
                 requests[request_index].request_id == id);
        push.payload_bytes.push_back(bytes);
      }
      if (push.payload_bytes.empty() || push.payload_bytes.size() % blocks != 0)
        throw std::invalid_argument(
            "PD host transport stream has incomplete block payloads");
      job.metadata = Encode(push);
      jobs.push_back(std::move(job));
    }
    if (request_index != requests.size())
      throw std::invalid_argument(
          "PD host transport requests must be ordered by stream and payload");
    for (int id : src_ids)
      if (id < 0) throw std::invalid_argument("negative PD source block ID");
    for (int id : dst_ids)
      if (id < 0)
        throw std::invalid_argument("negative PD destination block ID");
    impl_->Start(std::move(jobs), std::move(owners),
                 std::vector<int>(dst_ids.begin(), dst_ids.end()),
                 std::move(callback));
    return 0;
  } catch (const std::exception& error) {
    return fail(error.what());
  }
}
}  // namespace tpu_raiden::host_leader
