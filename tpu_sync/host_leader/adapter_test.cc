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

#include <gtest/gtest.h>
#include <poll.h>

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <future>
#include <mutex>
#include <set>
#include <vector>

#include "tpu_sync/host_leader/allocator.h"
#include "tpu_sync/host_leader/test_util.h"

namespace tpu_raiden::host_leader {
namespace {
using namespace test_util;
constexpr size_t kBytes = 16;
constexpr uint64_t kUuid = 1701;
void SelectSocket(const Daemon& local, const Daemon& remote) {
  Require(setenv("TPU_RAIDEN_HOST_LEADER_SOCKET", local.socket.c_str(), 1) == 0,
          "setenv failed");
  const std::string peers = "127.0.0.1=remote@127.0.0.1>" + remote.endpoint +
                            ";127.0.0.2=remote@127.0.0.1>" + remote.endpoint +
                            ";127.0.0.3=remote@127.0.0.1>" + remote.endpoint;
  Require(setenv("TPU_RAIDEN_HOST_LEADER_PEERS", peers.c_str(), 1) == 0,
          "setenv failed");
}
uint8_t Pattern(int receiver, int sender, int transaction, size_t byte) {
  return receiver * 64 + sender * 16 + transaction * 8 + byte;
}
class Delegate : public transport::BlockTransportDelegate {
 public:
  explicit Delegate(size_t bytes = 4 * kBytes) {
    auto allocated = MakeAllocator(nullptr)(bytes, nullptr);
    Require(allocated.ok(), allocated.status().ToString());
    buffer = std::move(*allocated);
    std::fill_n(buffer.ptr, buffer.size, 0);
  }
  uint8_t* GetHostPointer(size_t, size_t) override { return buffer.ptr; }
  size_t GetHostSize(size_t, size_t) override { return buffer.size; }
  size_t num_layers() const override { return 1; }
  size_t num_shards() const override { return 1; }
  size_t slice_byte_size() const override { return kBytes; }
  size_t shard_factor() const override { return 1; }
  int GetRemoteReadBlockId(int id, int offset) override { return id + offset; }
  absl::StatusOr<std::vector<int>> AllocateBlocks(size_t, uint64_t) override {
    return std::vector<int>{0};
  }
  absl::StatusOr<std::optional<transport::PoolPushProgressSpec>>
  GetPoolPushProgressSpec(size_t pool, uint64_t uuid) const override {
    if (pool != 0 || uuid < kUuid || uuid > kUuid + 1)
      return absl::InvalidArgumentError("unknown test plan");
    return transport::PoolPushProgressSpec{expected_pushes, 1};
  }
  std::vector<transport::BlockChunk> GetBlockChunks(
      size_t, size_t, absl::Span<const int64_t>, size_t bytes, uint64_t uuid,
      int64_t sender, absl::string_view, int64_t, int64_t) override {
    Await([&] { return !prepare_gate; }, "Prepare test gate never opened");
    Require(sender >= 0 && sender < 2 && bytes == kBytes,
            "wrong sender or payload length");
    return {{buffer.ptr + ((uuid - kUuid) * 2 + sender) * kBytes, bytes}};
  }
  absl::Status BeginIncomingPush(uint64_t) override {
    ++begun;
    return absl::OkStatus();
  }
  absl::Status EndIncomingPush(uint64_t) override {
    ++ended;
    return absl::OkStatus();
  }
  absl::Status OnPoolReceived(size_t, uint64_t uuid) override {
    finish_entered = true;
    Await([&] { return !finish_gate; }, "Finish test gate never opened");
    ++completed[uuid - kUuid];
    return absl::OkStatus();
  }
  HostBufferAllocation buffer;
  size_t expected_pushes = 2;
  std::atomic<int> begun{0}, ended{0};
  std::atomic<int> completed[2]{};
  std::atomic<bool> prepare_gate{false}, finish_gate{false},
      finish_entered{false};
};
std::future<absl::Status> Send(Adapter& adapter, uint8_t* pointer, int receiver,
                               int sender, int transaction) {
  transport::lib::Request request{};
  request.socket_opcode = 6;
  request.laddr = pointer;
  request.len = kBytes;
  request.parallelism = 1;
  request.remote_id = sender;
  request.local_id = 0;
  request.uuid = kUuid + transaction;
  auto done = std::make_shared<std::promise<absl::Status>>();
  auto result = done->get_future();
  const std::vector<std::string> peers{"127.0.0." +
                                       std::to_string(2 + receiver) + ":" +
                                       std::to_string(32000 + receiver)};
  const std::vector<int> ids{0};
  auto posted = adapter.Post(peers, {&request, 1}, ids, ids,
                             [done](absl::StatusOr<std::vector<int>> result) {
                               done->set_value(result.status());
                             });
  Require(posted.ok(), posted.status().ToString());
  return result;
}
void Receiver(const Daemon& daemon, int rank, int ready, int stop) {
  SelectSocket(daemon, daemon);
  Delegate delegate;
  {
    Adapter adapter(&delegate, 32000 + rank,
                    {"127.0.0." + std::to_string(2 + rank)});
    Require(write(ready, "R", 1) == 1, "receiver ready failed");
    char signal;
    Require(read(stop, &signal, 1) == 1, "receiver stop failed");
  }
  Require(delegate.begun == 4 && delegate.ended == 4,
          "incoming pins did not balance");
  for (int transaction = 0; transaction < 2; ++transaction) {
    Require(delegate.completed[transaction] == 1,
            "pool completion is not exactly once");
    for (int sender = 0; sender < 2; ++sender)
      for (size_t byte = 0; byte < kBytes; ++byte)
        Require(
            delegate.buffer.ptr[(transaction * 2 + sender) * kBytes + byte] ==
                Pattern(rank, sender, transaction, byte),
            "rank, sender, pool or UUID isolation lost payload bytes");
  }
}
void Sender(const Daemon& local, const Daemon& remote, int sender) {
  SelectSocket(local, remote);
  Delegate delegate;
  Adapter adapter(&delegate, 31000 + sender, {"127.0.0.1"});
  std::vector<std::future<absl::Status>> pending;
  for (int transaction = 0; transaction < 2; ++transaction) {
    for (int receiver = 0; receiver < 2; ++receiver) {
      auto allocation = MakeAllocator(nullptr)(kBytes, nullptr);
      Require(allocation.ok(), allocation.status().ToString());
      for (size_t byte = 0; byte < kBytes; ++byte)
        allocation->ptr[byte] = Pattern(receiver, sender, transaction, byte);
      pending.push_back(
          Send(adapter, allocation->ptr, receiver, sender, transaction));
      allocation->owner.reset();  // Queued Post must retain the source owner.
    }
  }
  for (auto& result : pending) {
    Require(result.wait_for(5s) == std::future_status::ready,
            "send callback did not drain");
    Require(result.get().ok(), "send callback failed");
  }
}
class PoolBridgeChannelsTest : public testing::TestWithParam<int> {};
TEST_P(PoolBridgeChannelsTest,
       CrossProcessRanksAndUuidsShareBoundedHostChannels) {
  const int streams = GetParam();
  Child test([&] {
    Daemon a(streams), b(streams);
    int ready[2], stop[2];
    Require(pipe2(ready, O_CLOEXEC) == 0 && pipe2(stop, O_CLOEXEC) == 0,
            "pipe failed");
    Fd ready_read(ready[0]), ready_write(ready[1]), stop_read(stop[0]),
        stop_write(stop[1]);
    Child receiver_a([&] { Receiver(b, 0, ready[1], stop[0]); });
    Child receiver_b([&] { Receiver(b, 1, ready[1], stop[0]); });
    for (int rank = 0; rank < 2; ++rank) {
      pollfd fd{ready[0], POLLIN, 0};
      Require(poll(&fd, 1, 5000) == 1, "receiver readiness timeout");
      char signal;
      Require(read(ready[0], &signal, 1) == 1, "receiver readiness failed");
    }
    Child sender_a([&] { Sender(a, b, 0); });
    Child sender_b([&] { Sender(a, b, 1); });
    Require(sender_a.Wait() == 0 && sender_b.Wait() == 0,
            "sender process failed");
    Require(write(stop[1], "SS", 2) == 2, "receiver stop failed");
    Require(receiver_a.Wait() == 0 && receiver_b.Wait() == 0,
            "receiver process failed");
    Require(a.stats().connections >= 1 &&
                a.stats().connections <= static_cast<uint64_t>(streams),
            "ranks multiplied the host connection budget");
    Await(
        [&] {
          return a.stats().registered_regions == 0 &&
                 b.stats().registered_regions == 0;
        },
        "worker exit retained memfd registrations");
  });
  EXPECT_EQ(test.Wait(), 0);
}
INSTANTIATE_TEST_SUITE_P(StreamBudgets, PoolBridgeChannelsTest,
                         testing::Values(1, 2));

class UnevenDelegate : public Delegate {
 public:
  UnevenDelegate() : Delegate(5 * kBytes) { expected_pushes = 3; }
  std::vector<transport::BlockChunk> GetBlockChunks(
      size_t, size_t, absl::Span<const int64_t> ids, size_t bytes, uint64_t,
      int64_t, absl::string_view, int64_t source, int64_t) override {
    Require(ids.size() == 1 && ids[0] >= 20 && ids[0] < 25 &&
                source == ids[0] - 10 && bytes == kBytes,
            "uneven streams lost source or destination IDs");
    std::lock_guard<std::mutex> lock(mu);
    Require(seen.insert(ids[0]).second, "uneven stream duplicated a block");
    auto* pointer = buffer.ptr + (ids[0] - 20) * kBytes;
    return {{pointer, 5}, {pointer + 5, kBytes - 5}};
  }
  std::mutex mu;
  std::set<int64_t> seen;
};
TEST(PoolBridgeTest, UnevenStreamsRetainEveryBlockAndChunk) {
  Child test([] {
    Daemon daemon(2);
    SelectSocket(daemon, daemon);
    UnevenDelegate receiver;
    Adapter receive(&receiver, 32000, {"127.0.0.2"});
    Delegate sender;
    Adapter send(&sender, 31000, {"127.0.0.1"});
    auto allocation = MakeAllocator(nullptr)(5 * kBytes, nullptr);
    Require(allocation.ok(), allocation.status().ToString());
    for (size_t i = 0; i < 5 * kBytes; ++i) allocation->ptr[i] = i;
    std::vector<transport::lib::Request> requests;
    std::vector<int> src, dst;
    for (int block = 0; block < 5; ++block) {
      src.push_back(10 + block);
      dst.push_back(20 + block);
      for (int chunk = 0; chunk < 2; ++chunk) {
        transport::lib::Request request{};
        request.socket_opcode = 6;
        request.laddr = allocation->ptr + block * kBytes + (chunk == 0 ? 0 : 7);
        request.len = chunk == 0 ? 7 : kBytes - 7;
        request.parallelism = 3;
        request.uuid = kUuid;
        request.request_id = block;
        request.stream_idx = block / 2;
        requests.push_back(request);
      }
    }
    std::promise<absl::Status> done;
    auto result = done.get_future();
    auto posted =
        send.Post(std::vector<std::string>{"127.0.0.2:32000"}, requests, src,
                  dst, [&done](absl::StatusOr<std::vector<int>> result) {
                    done.set_value(result.status());
                  });
    Require(posted.ok(), posted.status().ToString());
    allocation->owner.reset();
    Require(
        result.wait_for(5s) == std::future_status::ready && result.get().ok(),
        "uneven transfer failed");
    Require(receiver.seen.size() == 5 && receiver.completed[0] == 1 &&
                receiver.begun == 3 && receiver.ended == 3,
            "2/2/1 logical streams lost blocks or pool completion");
    for (size_t i = 0; i < 5 * kBytes; ++i)
      Require(receiver.buffer.ptr[i] == i, "uneven payload bytes changed");
  });
  EXPECT_EQ(test.Wait(), 0);
}
TEST(PoolBridgeTest, SuccessfulFinishDrainsBeforeBridgeDestruction) {
  Child test([] {
    Daemon daemon;
    SelectSocket(daemon, daemon);
    Delegate receiver, sender;
    receiver.expected_pushes = 1;
    receiver.finish_gate = true;
    auto receive = std::make_unique<Adapter>(
        &receiver, 32000, std::vector<std::string>{"127.0.0.2"});
    Adapter send(&sender, 31000, {"127.0.0.1"});
    auto done = Send(send, sender.buffer.ptr, 0, 0, 0);
    Await([&] { return receiver.finish_entered.load(); },
          "incoming Finish never ran");
    auto drained = std::async(std::launch::async, [&] { receive.reset(); });
    Require(
        drained.wait_for(20ms) == std::future_status::timeout &&
            receiver.ended == 0,
        "bridge destructor returned before incoming application pin drained");
    receiver.finish_gate = false;
    Require(drained.wait_for(5s) == std::future_status::ready,
            "bridge destructor failed to drain Finish");
    drained.get();
    Require(done.wait_for(5s) == std::future_status::ready && done.get().ok(),
            "send completion failed");
    Require(receiver.begun == 1 && receiver.ended == 1 &&
                receiver.completed[0] == 1,
            "successful receive completion or pin not exactly once");
  });
  EXPECT_EQ(test.Wait(), 0);
}
TEST(PoolBridgeTest, DaemonSigkillUnpinsPreparedReceiveWithoutCompletingPool) {
  Child test([] {
    Daemon daemon;
    SelectSocket(daemon, daemon);
    Delegate receiver, sender;
    receiver.expected_pushes = 1;
    receiver.prepare_gate = true;
    Adapter receive(&receiver, 32000, {"127.0.0.2"});
    Adapter send(&sender, 31000, {"127.0.0.1"});
    auto done = Send(send, sender.buffer.ptr, 0, 0, 0);
    Await([&] { return receiver.begun == 1; },
          "receiver never acquired incoming pin");
    daemon.Kill();
    receiver.prepare_gate = false;
    Await([&] { return receiver.ended == 1; },
          "daemon EOF failed to release prepared incoming pin");
    Require(done.wait_for(5s) == std::future_status::ready && !done.get().ok(),
            "daemon failure reported successful send");
    Require(receiver.completed[0] == 0 && receiver.begun == receiver.ended,
            "failed incoming payload completed pool or unbalanced pin");
  });
  EXPECT_EQ(test.Wait(), 0);
}
TEST(PoolBridgeTest, InvalidOrConflictingPeerConfigurationRejectsAtEntry) {
  for (const auto* peers :
       {"127.0.0.1=@127.0.0.1>127.0.0.1:1",
        "127.0.0.1=h@127.0.0.1>127.0.0.1:1;127.0.0.1=h@127.0.0.1>127.0.0.1:1",
        "127.0.0.1=h@127.0.0.1>127.0.0.1:1;127.0.0.2=h@127.0.0.1>127.0.0.2:"
        "1"}) {
    Child child([&] {
      setenv("TPU_RAIDEN_HOST_LEADER_PEERS", peers, 1);
      try {
        Adapter adapter(nullptr, 31000, {"127.0.0.1"});
      } catch (const std::invalid_argument&) {
        return;
      }
      throw std::runtime_error("invalid explicit peers accepted");
    });
    EXPECT_EQ(child.Wait(), 0);
  }
}
}  // namespace
}  // namespace tpu_raiden::host_leader
