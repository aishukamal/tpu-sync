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

// D2h()/H2d() with RAIDEN_ENABLE_ASYNC_DISPATCH=1 (what sglang's HiCache uses)
// must report a copy that failed, whether its dispatch failed or the issued
// copy failed later.

#include <cstdint>
#include <cstdlib>
#include <optional>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "tpu_sync/core/raw_transfer_core.h"
#include "tpu_sync/kv_cache/kv_cache_manager_base.h"
#include "xla/future.h"

namespace tpu_raiden::kv_cache {
namespace {

const absl::Status kDmaError = absl::InternalError("dma failed");

enum class Outcome { kOk, kDispatchError, kPending };

// Replaces the real per-layer dispatch: fails it, succeeds, or hands back a
// copy the test completes later.
class FakeDispatchManager : public KVCacheManagerBase {
 public:
  explicit FakeDispatchManager(Outcome outcome)
      : KVCacheManagerBase(/*num_layers=*/1, /*num_shards=*/1,
                           /*slice_byte_size=*/128, /*local_port=*/std::nullopt,
                           /*host_blocks_to_allocate=*/2),
        outcome_(outcome) {}

  absl::StatusOr<raiden::PjRtCopyFuture> H2dSyncDispatch(
      const std::vector<int64_t>&, const std::vector<int64_t>&,
      const std::vector<int64_t>&, std::optional<int64_t>,
      std::optional<size_t>, std::optional<size_t>) override {
    return Dispatch();
  }

  absl::StatusOr<raiden::PjRtCopyFuture> D2hSyncDispatch(
      const std::vector<int64_t>&, const std::vector<int64_t>&,
      const std::vector<int64_t>&, std::optional<int64_t>,
      std::optional<size_t>, std::optional<size_t>) override {
    return Dispatch();
  }

  bool HasPending() {
    absl::MutexLock lock(mu_);
    return pending_.has_value();
  }

  // Completes the copy handed out by a kPending dispatch.
  void CompletePending(absl::Status status) {
    absl::MutexLock lock(mu_);
    ASSERT_TRUE(pending_.has_value());
    pending_->Set(std::move(status));
  }

 private:
  absl::StatusOr<raiden::PjRtCopyFuture> Dispatch() {
    switch (outcome_) {
      case Outcome::kOk:
        return raiden::PjRtCopyFuture(std::vector<raiden::BufferHolder>{});
      case Outcome::kDispatchError:
        return kDmaError;
      case Outcome::kPending: {
        auto [promise, future] = xla::MakePromise();
        absl::MutexLock lock(mu_);
        pending_.emplace(std::move(promise));
        return raiden::PjRtCopyFuture(std::move(future), {});
      }
    }
    return absl::InternalError("unreachable");
  }

  const Outcome outcome_;
  absl::Mutex mu_;
  std::optional<xla::Promise<>> pending_ ABSL_GUARDED_BY(mu_);
};

class AsyncDispatchTest : public testing::Test {
 protected:
  void SetUp() override { setenv("RAIDEN_ENABLE_ASYNC_DISPATCH", "1", 1); }
  void TearDown() override { unsetenv("RAIDEN_ENABLE_ASYNC_DISPATCH"); }
};

// Polls like sglang's RaidenFutureEvent.query(): never blocks on the copy.
bool WaitReady(raiden::PjRtCopyFuture& f) {
  const absl::Time deadline = absl::Now() + absl::Seconds(10);
  while (!f.IsReady()) {
    if (absl::Now() > deadline) return false;
    absl::SleepFor(absl::Milliseconds(1));
  }
  return true;
}

TEST_F(AsyncDispatchTest, D2hReportsAFailedDispatch) {
  FakeDispatchManager manager(Outcome::kDispatchError);
  absl::StatusOr<raiden::PjRtCopyFuture> f = manager.D2h({0}, {0}, {1});
  ASSERT_TRUE(f.ok());  // async: the error can only arrive through the future
  ASSERT_TRUE(WaitReady(*f));
  EXPECT_EQ(f->PollError(), kDmaError);
  EXPECT_EQ(f->Await(), kDmaError);
}

TEST_F(AsyncDispatchTest, H2dReportsAFailedDispatch) {
  FakeDispatchManager manager(Outcome::kDispatchError);
  absl::StatusOr<raiden::PjRtCopyFuture> f = manager.H2d({0}, {0}, {1});
  ASSERT_TRUE(f.ok());
  ASSERT_TRUE(WaitReady(*f));
  EXPECT_EQ(f->PollError(), kDmaError);
  EXPECT_EQ(f->Await(), kDmaError);
}

TEST_F(AsyncDispatchTest, D2hReportsACopyThatFailsAfterDispatch) {
  FakeDispatchManager manager(Outcome::kPending);
  absl::StatusOr<raiden::PjRtCopyFuture> f = manager.D2h({0}, {0}, {1});
  ASSERT_TRUE(f.ok());
  // Wait until the worker has dispatched, then fail the issued copy.
  const absl::Time deadline = absl::Now() + absl::Seconds(10);
  while (!manager.HasPending() && absl::Now() < deadline) {
    absl::SleepFor(absl::Milliseconds(1));
  }
  ASSERT_TRUE(manager.HasPending());
  EXPECT_FALSE(f->IsReady());
  manager.CompletePending(kDmaError);
  ASSERT_TRUE(WaitReady(*f));
  EXPECT_EQ(f->PollError(), kDmaError);
  EXPECT_EQ(f->Await(), kDmaError);
}

TEST_F(AsyncDispatchTest, SuccessIsOk) {
  FakeDispatchManager manager(Outcome::kOk);
  absl::StatusOr<raiden::PjRtCopyFuture> f = manager.D2h({0}, {0}, {1});
  ASSERT_TRUE(f.ok());
  ASSERT_TRUE(WaitReady(*f));
  EXPECT_TRUE(f->PollError().ok());
  EXPECT_TRUE(f->Await().ok());
}

// Without async dispatch the same failure is returned by D2h() itself.
TEST(SyncDispatchTest, D2hReturnsAFailedDispatch) {
  unsetenv("RAIDEN_ENABLE_ASYNC_DISPATCH");
  FakeDispatchManager manager(Outcome::kDispatchError);
  EXPECT_EQ(manager.D2h({0}, {0}, {1}).status(), kDmaError);
}

}  // namespace
}  // namespace tpu_raiden::kv_cache
