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

#include <utility>

#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "tpu_sync/core/raiden_future.h"
#include "tpu_sync/core/raw_transfer_core.h"
#include "xla/future.h"

namespace raiden {
namespace {

// The future D2hAsyncDispatch / H2dAsyncDispatch return: promise-backed, with
// no PJRT events. WorkerLoop completes it with promise.Set(status), both for a
// dispatch that fails and for a copy that fails once issued.
struct AsyncDispatchFuture {
  xla::Promise<> promise;
  PjRtCopyFuture future;
};

AsyncDispatchFuture MakeAsyncDispatchFuture() {
  auto [promise, future] = xla::MakePromise();
  return {std::move(promise), PjRtCopyFuture(std::move(future), {})};
}

const absl::Status kDmaError = absl::InternalError("dma failed");

TEST(AsyncDispatchFutureTest, PollErrorReportsAFailedDispatch) {
  AsyncDispatchFuture f = MakeAsyncDispatchFuture();
  f.promise.Set(kDmaError);
  ASSERT_TRUE(f.future.IsReady());
  EXPECT_EQ(f.future.PollError(), kDmaError);
}

TEST(AsyncDispatchFutureTest, AwaitReportsAFailedDispatch) {
  AsyncDispatchFuture f = MakeAsyncDispatchFuture();
  f.promise.Set(kDmaError);
  EXPECT_EQ(f.future.Await(), kDmaError);
}

// RaidenFuture is what Python's ok() / error_message() / Await() call.
TEST(AsyncDispatchFutureTest, RaidenFutureReportsAFailedDispatch) {
  AsyncDispatchFuture f = MakeAsyncDispatchFuture();
  tpu_raiden::RaidenFuture rf{f.future};
  f.promise.Set(kDmaError);
  ASSERT_TRUE(rf.IsReady());
  EXPECT_FALSE(rf.PollError().ok());
  EXPECT_FALSE(rf.Await().ok());
}

TEST(AsyncDispatchFutureTest, PollErrorIsOkWhilePendingAndAfterSuccess) {
  AsyncDispatchFuture f = MakeAsyncDispatchFuture();
  EXPECT_FALSE(f.future.IsReady());
  EXPECT_TRUE(f.future.PollError().ok());
  f.promise.Set();
  ASSERT_TRUE(f.future.IsReady());
  EXPECT_TRUE(f.future.PollError().ok());
  EXPECT_TRUE(f.future.Await().ok());
}

// The error is on the future: Promise<void>::Set stores it as the value, not
// as an AsyncValue error, so the xla future and OnReady both see it.
TEST(AsyncDispatchFutureTest, ErrorIsStoredAsTheValue) {
  AsyncDispatchFuture f = MakeAsyncDispatchFuture();
  f.promise.Set(kDmaError);
  EXPECT_FALSE(f.future.future.async_value()->IsError());
  EXPECT_EQ(f.future.future.Await(), kDmaError);
}

TEST(AsyncDispatchFutureTest, OnReadyReportsAFailedDispatch) {
  AsyncDispatchFuture f = MakeAsyncDispatchFuture();
  absl::Status seen;
  f.future.OnReady(
      [&](absl::StatusOr<BufferHolders> holds) { seen = holds.status(); });
  f.promise.Set(kDmaError);
  EXPECT_EQ(seen, kDmaError);
}

}  // namespace
}  // namespace raiden
