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

#include <gtest/gtest.h>
#include <poll.h>
#include <sys/socket.h>

#include <algorithm>
#include <atomic>
#include <future>
#include <map>
#include <mutex>
#include <vector>

#include "absl/flags/flag.h"
#include "tpu_sync/host_leader/service.h"
#include "tpu_sync/host_leader/session.h"
#include "tpu_sync/host_leader/test_util.h"
#include "tpu_sync/transport/lib/socket/tcp_psp_helper.h"
#include "tpu_sync/transport/lib/socket/util.h"

namespace tpu_raiden::host_leader {
namespace {
using namespace test_util;
constexpr size_t kBytes = 16;

std::shared_ptr<Session> Connect(const Daemon& daemon) {
  auto session = Session::Connect(daemon.socket);
  Require(session.ok(), session.status().ToString());
  return std::move(*session);
}
std::shared_ptr<Region> Allocate(const std::shared_ptr<Session>& session,
                                 size_t bytes = 8 * kBytes) {
  auto region = session->AllocateRegion(bytes);
  Require(region.ok(), region.status().ToString());
  return std::move(*region);
}
class BufferHandler : public EndpointHandler {
 public:
  explicit BufferHandler(std::shared_ptr<Region> region)
      : region(std::move(region)) {}
  absl::StatusOr<Prepared> Prepare(const PrepareRequest& request) override {
    size_t offset = std::stoull(request.metadata);
    if (offset > region->size() || request.bytes > region->size() - offset)
      return absl::InvalidArgumentError("test request outside region");
    std::lock_guard<std::mutex> lock(mu);
    retained.emplace(request.ticket, region);
    return Prepared{request.ticket, {{region, offset, request.bytes}}};
  }
  absl::Status Finish(uint64_t cookie, const IoResult& result) override {
    ++finishes;
    Await([&] { return !gate; }, "test Finish gate never opened");
    std::lock_guard<std::mutex> lock(mu);
    Require(retained.erase(cookie) == 1,
            "Finish called twice or without Prepare");
    return result.status;
  }
  std::shared_ptr<Region> region;
  std::atomic<bool> gate{false};
  std::atomic<int> finishes{0};
  std::mutex mu;
  std::map<uint64_t, std::shared_ptr<Region>> retained;
};
std::unique_ptr<Endpoint> EndpointFor(
    const std::shared_ptr<Session>& session, const std::string& key,
    const std::shared_ptr<EndpointHandler>& handler) {
  auto endpoint = session->RegisterEndpoint(key, handler);
  Require(endpoint.ok(), endpoint.status().ToString());
  return std::move(*endpoint);
}
std::unique_ptr<Job> Submit(const std::shared_ptr<Session>& session,
                            const Peer& peer, const std::string& endpoint,
                            const std::shared_ptr<Region>& region,
                            Direction direction, size_t offset = 0) {
  JobRequest request;
  request.direction = direction;
  request.peer = peer;
  request.endpoint = endpoint;
  request.metadata = std::to_string(offset);
  request.spans = {{region, offset, kBytes}};
  auto job = session->Submit(std::move(request));
  Require(job.ok(), job.status().ToString());
  return std::move(*job);
}
void Finish(const std::unique_ptr<Job>& job) {
  const auto result = job->Wait();
  Require(result.status.ok(), result.status.ToString());
  Require(result.bytes == kBytes,
          "generic job reported wrong transferred bytes");
}

class GenericChannelsTest : public testing::TestWithParam<int> {};
// Long-term contract: one directed host budget is shared by both operation
// directions and all NIC routes; reverse initiators get their own p slots.
TEST_P(GenericChannelsTest, ReadWriteAndBothInitiatorsShareDirectedNicBudget) {
  const int p = GetParam();
  Child child([&] {
    Daemon a(p), b(p);
    auto sa = Connect(a), sb = Connect(b);
    auto ra = Allocate(sa), rb = Allocate(sb);
    auto ha = std::make_shared<BufferHandler>(ra);
    auto hb = std::make_shared<BufferHandler>(rb);
    auto ea = EndpointFor(sa, "opaque-A", ha),
         eb = EndpointFor(sb, "opaque-B", hb);
    const auto routes = [](const Daemon& daemon, const std::string& host) {
      const auto port = daemon.endpoint.substr(daemon.endpoint.rfind(':'));
      return Peer{host,
                  {{"127.0.0.1", "127.0.0.1" + port},
                   {"127.0.0.2", "127.0.0.2" + port}}};
    };
    auto pa = routes(a, "host-A"), pb = routes(b, "host-B");
    for (int round = 0; round < 4; ++round) {
      const bool from_a = round < 2;
      const auto direction =
          round % 2 == 0 ? Direction::kWrite : Direction::kRead;
      auto local = from_a ? ra : rb, remote = from_a ? rb : ra;
      auto source = direction == Direction::kWrite ? local : remote;
      auto destination = direction == Direction::kWrite ? remote : local;
      std::fill_n(source->data(), kBytes, 0x40 + round);
      std::fill_n(destination->data(), kBytes, 0);
      auto job = Submit(from_a ? sa : sb, from_a ? pb : pa,
                        from_a ? "opaque-B" : "opaque-A", local, direction);
      Finish(job);
      Require(
          std::all_of(destination->data(), destination->data() + kBytes,
                      [round](uint8_t byte) { return byte == 0x40 + round; }),
          "Read/Write payload direction is incorrect");
    }
    ha->gate = hb->gate = true;
    const int fa = ha->finishes, fb = hb->finishes;
    std::vector<std::unique_ptr<Job>> jobs;
    for (int i = 0; i < p + 2; ++i) {
      jobs.push_back(Submit(sa, pb, "opaque-B", ra,
                            i % 2 ? Direction::kRead : Direction::kWrite,
                            i * kBytes));
      jobs.push_back(Submit(sb, pa, "opaque-A", rb,
                            i % 2 ? Direction::kWrite : Direction::kRead,
                            (i + 4) * kBytes));
    }
    Await([&] { return ha->finishes >= fa + p && hb->finishes >= fb + p; },
          "both initiators did not fill their channel budgets");
    Require(a.stats().connections == static_cast<uint64_t>(p) &&
                b.stats().connections == static_cast<uint64_t>(p),
            "Read/Write or NIC routes multiplied directed host budget");
    ha->gate = hb->gate = false;
    for (const auto& job : jobs) Finish(job);
    Require(a.stats().connections <= static_cast<uint64_t>(p) &&
                b.stats().connections <= static_cast<uint64_t>(p),
            "idle channels exceed directed host budget");
    Require(
        a.stats().registered_regions == 1 && b.stats().registered_regions == 1,
        "operations registered or mapped the same allocation repeatedly");
    ea.reset();
    eb.reset();
    jobs.clear();
    ha.reset();
    hb.reset();
    ra.reset();
    rb.reset();
    Require(sa->Close().ok() && sb->Close().ok(), "Session drain failed");
  });
  EXPECT_EQ(child.Wait(), 0);
}
INSTANTIATE_TEST_SUITE_P(StreamBudgets, GenericChannelsTest,
                         testing::Values(1, 2));

// Long-term contract: normal release while a worker remains alive must retire
// daemon FD and mmap references, rather than waiting for worker EOF.
TEST(GenericMemoryTest, NormalReleaseAndRetireRemoveFdAndMapping) {
  Child child([] {
    Daemon daemon;
    auto session = Connect(daemon);
    const auto fd_baseline = MemfdFds(daemon.pid());
    const auto map_baseline = MemfdMappings(daemon.pid());
    for (int i = 0; i < 3; ++i) {
      auto region = Allocate(session, 8192);
      Require(daemon.stats().registered_regions == 1 &&
                  daemon.stats().mapped_bytes == 8192,
              "persistent region registration missing");
      Require(MemfdFds(daemon.pid()) == fd_baseline + 1 &&
                  MemfdMappings(daemon.pid()) == map_baseline + 1,
              "registered memfd ownership is not one FD and one mapping");
      if (i == 1)
        Require(region->Retire().ok(), "explicit region Retire failed");
      region.reset();
      Await(
          [&] {
            auto stats = daemon.stats();
            return stats.registered_regions == 0 && stats.mapped_bytes == 0 &&
                   MemfdFds(daemon.pid()) == fd_baseline &&
                   MemfdMappings(daemon.pid()) == map_baseline;
          },
          "normal allocation release leaked daemon FD or mapping");
    }
    Require(session->Close().ok(), "Session close failed");
  });
  EXPECT_EQ(child.Wait(), 0);
}

class TemporaryRegionHandler : public EndpointHandler {
 public:
  explicit TemporaryRegionHandler(std::shared_ptr<Session> session)
      : session(std::move(session)) {}
  absl::StatusOr<Prepared> Prepare(const PrepareRequest& request) override {
    auto region = session->AllocateRegion(request.bytes);
    if (!region.ok()) return region.status();
    retained = std::move(*region);
    return Prepared{request.ticket, {{retained, 0, request.bytes}}};
  }
  absl::Status Finish(uint64_t, const IoResult& result) override {
    if (!result.status.ok()) return result.status;
    Require(std::all_of(retained->data(), retained->data() + retained->size(),
                        [](uint8_t byte) { return byte == 0xA7; }),
            "temporary receive allocation lost payload");
    retained.reset();  // Last Region calls synchronous Retire from Finish.
    return absl::OkStatus();
  }
  std::shared_ptr<Session> session;
  std::shared_ptr<Region> retained;
};
TEST(GenericMemoryTest, LastRegionCanRetireInsideFinishWithoutDeadlock) {
  Child child([] {
    Daemon daemon;
    auto session = Connect(daemon);
    auto source = Allocate(session, kBytes);
    std::fill_n(source->data(), kBytes, 0xA7);
    auto handler = std::make_shared<TemporaryRegionHandler>(session);
    auto endpoint = EndpointFor(session, "temporary", handler);
    for (int i = 0; i < 3; ++i) {
      auto job = Submit(session, daemon.peer(), "temporary", source,
                        Direction::kWrite);
      Finish(job);
      Require(daemon.stats().registered_regions == 1,
              "Finish failed to retire its final Region reference");
    }
    endpoint.reset();
    handler.reset();
    source.reset();
    Require(session->Close().ok(), "Session close failed");
    Require(daemon.stats().registered_regions == 0 &&
                daemon.stats().mapped_bytes == 0,
            "temporary Finish cleanup leaked regions");
  });
  EXPECT_EQ(child.Wait(), 0);
}

class PreparedOnlyHandler : public EndpointHandler {
 public:
  explicit PreparedOnlyHandler(std::shared_ptr<Session> session)
      : session(std::move(session)) {}
  absl::StatusOr<Prepared> Prepare(const PrepareRequest& request) override {
    auto region = session->AllocateRegion(request.bytes);
    if (!region.ok()) return region.status();
    std::fill_n((*region)->data(), request.bytes, 0x6B);
    return Prepared{request.ticket, {{std::move(*region), 0, request.bytes}}};
  }
  absl::Status Finish(uint64_t, const IoResult& result) override {
    ++finishes;
    return result.status;
  }
  std::shared_ptr<Session> session;
  std::atomic<int> finishes{0};
};
TEST(GenericMemoryTest, PreparedSpansOwnRegionUntilPayloadStops) {
  Child child([] {
    Daemon daemon;
    auto session = Connect(daemon);
    auto destination = Allocate(session, kBytes);
    auto handler = std::make_shared<PreparedOnlyHandler>(session);
    auto endpoint = EndpointFor(session, "prepared-only", handler);
    auto job = Submit(session, daemon.peer(), "prepared-only", destination,
                      Direction::kRead);
    Finish(job);
    Require(std::all_of(destination->data(), destination->data() + kBytes,
                        [](uint8_t byte) { return byte == 0x6B; }),
            "Prepared-only Region expired before Read payload stopped");
    Require(handler->finishes == 1 && daemon.stats().registered_regions == 1,
            "Prepared-only region was not released at terminal Finish");
    endpoint.reset();
    handler.reset();
    job.reset();
    destination.reset();
    Require(session->Close().ok(), "Prepared-only Session close failed");
  });
  EXPECT_EQ(child.Wait(), 0);
}
class InvalidPreparedHandler : public EndpointHandler {
 public:
  InvalidPreparedHandler(std::shared_ptr<Region> region, bool out_of_bounds)
      : region(std::move(region)), out_of_bounds(out_of_bounds) {}
  absl::StatusOr<Prepared> Prepare(const PrepareRequest& request) override {
    return Prepared{
        request.ticket,
        {{region, out_of_bounds ? region->size() : 0, request.bytes}}};
  }
  absl::Status Finish(uint64_t, const IoResult& result) override {
    ++finishes;
    failed = !result.status.ok();
    return result.status;
  }
  std::shared_ptr<Region> region;
  bool out_of_bounds;
  std::atomic<int> finishes{0};
  std::atomic<bool> failed{false};
};
TEST(GenericMemoryTest,
     InvalidPreparedSpanFinishesOnceAndRejectsOtherSessionRegion) {
  Child child([] {
    Daemon daemon;
    auto a = Connect(daemon), b = Connect(daemon);
    auto ra = Allocate(a), rb = Allocate(b);
    // Region IDs are local to a worker session; an equal integer must never
    // authorize access to another session's Region.
    for (bool bounds : {false, true}) {
      auto handler =
          std::make_shared<InvalidPreparedHandler>(bounds ? ra : rb, bounds);
      auto endpoint = EndpointFor(a, "invalid-prepared", handler);
      auto job =
          Submit(a, daemon.peer(), "invalid-prepared", ra, Direction::kWrite);
      Require(!job->Wait().status.ok(),
              "invalid or other-session Prepared span was accepted");
      Require(handler->finishes == 1 && handler->failed,
              "failed Prepare validation did not release application pin "
              "exactly once");
    }
  });
  EXPECT_EQ(child.Wait(), 0);
}

TEST(GenericMemoryTest, WorkerSigkillRetiresRegionsWhileDaemonSurvives) {
  Daemon daemon;
  const auto fds = MemfdFds(daemon.pid()), maps = MemfdMappings(daemon.pid());
  int ready[2];
  ASSERT_EQ(pipe2(ready, O_CLOEXEC), 0);
  Fd reader(ready[0]), writer(ready[1]);
  Child worker([&] {
    auto session = Connect(daemon);
    auto region = Allocate(session, 65536);
    auto handler = std::make_shared<BufferHandler>(region);
    handler->gate = true;
    auto endpoint = EndpointFor(session, "killed-worker", handler);
    auto job = Submit(session, daemon.peer(), "killed-worker", region,
                      Direction::kWrite);
    Await([&] { return handler->finishes == 1; },
          "worker did not hold a prepared job before SIGKILL");
    Require(write(ready[1], "R", 1) == 1, "worker ready failed");
    for (;;) pause();
  });
  pollfd fd{ready[0], POLLIN, 0};
  ASSERT_EQ(poll(&fd, 1, 5000), 1);
  char signal;
  ASSERT_EQ(read(ready[0], &signal, 1), 1);
  ASSERT_EQ(daemon.stats().registered_regions, 1);
  worker.Stop();
  EXPECT_NO_THROW(Await(
      [&] {
        auto stats = daemon.stats();
        return stats.sessions == 0 && stats.endpoints == 0 &&
               stats.registered_regions == 0 && stats.mapped_bytes == 0 &&
               stats.queued == 0 && stats.prepared == 0 &&
               stats.inflight == 0 && MemfdFds(daemon.pid()) == fds &&
               MemfdMappings(daemon.pid()) == maps;
      },
      "SIGKILL worker left registry, FD or mmap resources in live daemon"));
}

// A queued cancellation must never Prepare; active cancellation cannot return
// while its responder's Finish still owns an application pin.
TEST(GenericChannelsTest, CancelQueuedAndActiveJobsDrainsResponderFinish) {
  Child child([] {
    Daemon daemon;
    auto session = Connect(daemon);
    auto region = Allocate(session);
    auto handler = std::make_shared<BufferHandler>(region);
    handler->gate = true;
    auto endpoint = EndpointFor(session, "cancel-target", handler);
    auto active = Submit(session, daemon.peer(), "cancel-target", region,
                         Direction::kWrite);
    Await([&] { return handler->finishes == 1; },
          "active job did not reach Finish");
    auto queued = Submit(session, daemon.peer(), "cancel-target", region,
                         Direction::kRead, kBytes);
    Require(queued->Cancel().ok(), "queued Cancel failed");
    Require(!queued->Wait().status.ok(), "queued Cancel reported success");
    Require(handler->finishes == 1,
            "cancelled queued job reached application Prepare");
    Require(active->Cancel().ok(), "active Cancel failed");
    auto drained =
        std::async(std::launch::async, [&] { return active->Wait(); });
    Require(drained.wait_for(20ms) == std::future_status::timeout,
            "active Wait returned before responder Finish released its pin");
    handler->gate = false;
    Require(drained.wait_for(5s) == std::future_status::ready,
            "active cancellation failed to drain");
    Require(!drained.get().status.ok(), "active cancellation reported success");
    endpoint.reset();
    active.reset();
    queued.reset();
    handler.reset();
    region.reset();
    Require(session->Close().ok(), "cancelled Session failed to close");
    Require(daemon.stats().registered_regions == 0 &&
                daemon.stats().mapped_bytes == 0,
            "cancelled jobs retained registered memory");
  });
  EXPECT_EQ(child.Wait(), 0);
}

class CallbackCloseHandler : public BufferHandler {
 public:
  CallbackCloseHandler(std::shared_ptr<Session> session,
                       std::shared_ptr<Region> region)
      : BufferHandler(std::move(region)), session(std::move(session)) {}
  absl::Status Finish(uint64_t cookie, const IoResult& result) override {
    auto status = BufferHandler::Finish(cookie, result);
    session_rejected = absl::IsFailedPrecondition(session->Close());
    endpoint_rejected = absl::IsFailedPrecondition(endpoint->Close());
    endpoint
        .reset();  // Destructor must defer its fence, not join this callback.
    return status;
  }
  std::shared_ptr<Session> session;
  std::unique_ptr<Endpoint> endpoint;
  std::atomic<bool> session_rejected{false}, endpoint_rejected{false};
};
TEST(GenericMemoryTest,
     FinishCanReleaseEndpointWhileExplicitCloseRejectsSelfWait) {
  Child child([] {
    Daemon daemon;
    auto session = Connect(daemon);
    auto region = Allocate(session);
    auto handler = std::make_shared<CallbackCloseHandler>(session, region);
    handler->endpoint = EndpointFor(session, "close-from-finish", handler);
    auto job = Submit(session, daemon.peer(), "close-from-finish", region,
                      Direction::kWrite);
    Finish(job);
    Require(handler->session_rejected && handler->endpoint_rejected,
            "callback explicit Close did not reject a self drain");
    Await([&] { return daemon.stats().endpoints == 0; },
          "Endpoint destructor from callback failed to retire endpoint");
    job.reset();
    handler.reset();
    region.reset();
    Require(session->Close().ok(), "external Session Close failed");
    auto stats = daemon.stats();
    Require(stats.sessions == 0 && stats.endpoints == 0 &&
                stats.registered_regions == 0 && stats.mapped_bytes == 0,
            "callback-originated cleanup retained service resources");
  });
  EXPECT_EQ(child.Wait(), 0);
}

TEST(GenericChannelsTest, ClosedIdleChannelReconnectsBeforeNextPayload) {
  Child child([] {
    Daemon a, b;
    auto sa = Connect(a), sb = Connect(b);
    auto ra = Allocate(sa), rb = Allocate(sb);
    auto handler = std::make_shared<BufferHandler>(rb);
    auto endpoint = EndpointFor(sb, "restart-target", handler);
    std::fill_n(ra->data(), kBytes, 0x81);
    Finish(Submit(sa, b.peer(), "restart-target", ra, Direction::kWrite));
    Require(std::all_of(rb->data(), rb->data() + kBytes,
                        [](uint8_t byte) { return byte == 0x81; }),
            "initial idle-channel write failed");
    b.Kill();
    endpoint.reset();
    handler.reset();
    rb.reset();
    sb->Close().IgnoreError();
    sb.reset();
    b.Restart();
    sb = Connect(b);
    rb = Allocate(sb);
    handler = std::make_shared<BufferHandler>(rb);
    endpoint = EndpointFor(sb, "restart-target", handler);
    std::fill_n(rb->data(), kBytes, 0x92);
    Finish(Submit(sa, b.peer(), "restart-target", ra, Direction::kRead));
    Require(std::all_of(ra->data(), ra->data() + kBytes,
                        [](uint8_t byte) { return byte == 0x92; }),
            "stale idle connection failed or replayed new Read payload");
    Require(a.stats().connections == 1,
            "reconnecting an idle route accumulated extra budget slots");
  });
  EXPECT_EQ(child.Wait(), 0);
}

TEST(GenericChannelsTest, ConflictingRoutesAcrossSessionsAreRejected) {
  Child child([] {
    Daemon daemon;
    auto a = Connect(daemon), b = Connect(daemon);
    auto region = Allocate(a), other = Allocate(b);
    auto handler = std::make_shared<BufferHandler>(region);
    auto endpoint = EndpointFor(a, "target", handler);
    Finish(Submit(a, daemon.peer("one-host"), "target", region,
                  Direction::kWrite));
    Peer conflict = daemon.peer("one-host");
    conflict.routes[0].local_ip = "127.0.0.2";
    JobRequest request;
    request.peer = conflict;
    request.endpoint = "target";
    request.metadata = "0";
    request.spans = {{other, 0, kBytes}};
    auto job = b->Submit(std::move(request));
    Require(!job.ok() || !(*job)->Wait().status.ok(),
            "same host_key accepted conflicting routes across workers");
  });
  EXPECT_EQ(child.Wait(), 0);
}

TEST(GenericSessionTest, ClosedSessionRejectsRepeatedEndpointAndJobAdmission) {
  Child child([] {
    Daemon daemon;
    auto session = Connect(daemon);
    auto region = Allocate(session);
    auto handler = std::make_shared<BufferHandler>(region);
    Require(session->Close().ok(), "initial Session Close failed");
    for (int attempt = 0; attempt < 3; ++attempt) {
      auto endpoint = session->RegisterEndpoint("same-closed-key", handler);
      Require(!endpoint.ok() && absl::IsUnavailable(endpoint.status()),
              "closed RegisterEndpoint changed into AlreadyExists on retry");
      JobRequest request;
      request.peer = daemon.peer();
      request.endpoint = "same-closed-key";
      request.metadata = "0";
      request.spans = {{region, 0, kBytes}};
      auto job = session->Submit(std::move(request));
      Require(!job.ok() && absl::IsUnavailable(job.status()),
              "closed Session admitted or retained a new job on retry");
    }
    auto stats = daemon.stats();
    Require(stats.sessions == 0 && stats.endpoints == 0 && stats.queued == 0 &&
                stats.inflight == 0 && stats.registered_regions == 0,
            "closed admission recreated service resources");
  });
  EXPECT_EQ(child.Wait(), 0);
}

TEST(GenericCapabilityTest, RequiredPspFailsAtBothPublicEntrypoints) {
  Child child([] {
    absl::SetFlag(&FLAGS_require_psp_tcp, true);
    auto session = Session::Connect("@nonexistent_psp_host_transport");
    Require(!session.ok() &&
                session.status().message().find("PSP") != std::string::npos,
            "client did not reject PSP before connecting");
    Service service({"@psp_host_transport_test", "127.0.0.1:0", 1});
    auto status = service.Start();
    Require(!status.ok() && status.message().find("PSP") != std::string::npos,
            "service silently used plaintext when PSP is required");
  });
  EXPECT_EQ(child.Wait(), 0);
}
TEST(GenericCapabilityTest, ExplicitSourceBindFailureNeverConnects) {
  Fd listener(socket(AF_INET, SOCK_STREAM | SOCK_CLOEXEC, 0));
  ASSERT_GE(listener.get(), 0);
  sockaddr_in address{};
  address.sin_family = AF_INET;
  address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  ASSERT_EQ(bind(listener.get(), reinterpret_cast<sockaddr*>(&address),
                 sizeof(address)),
            0);
  ASSERT_EQ(listen(listener.get(), 1), 0);
  socklen_t bytes = sizeof(address);
  ASSERT_EQ(getsockname(listener.get(), reinterpret_cast<sockaddr*>(&address),
                        &bytes),
            0);
  auto connection = transport::lib::ConnectToPeer(
      "127.0.0.1:" + std::to_string(ntohs(address.sin_port)), "192.0.2.1",
      false, nullptr, nullptr, true);
  EXPECT_FALSE(connection.ok());
  if (connection.ok()) Fd cleanup(*connection);
  pollfd pending{listener.get(), POLLIN, 0};
  EXPECT_EQ(poll(&pending, 1, 10), 0);
}
}  // namespace
}  // namespace tpu_raiden::host_leader
