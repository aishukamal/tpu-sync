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

#include "tpu_sync/core/raw_transfer_core.h"

#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/future.h"

namespace raiden {
namespace {

// A copy whose completion the test controls.
struct PendingCopy {
  xla::Promise<> promise;
  PjRtCopyFuture future;
};

PendingCopy MakePendingCopy() {
  auto [promise, future] = xla::MakePromise();
  return {std::move(promise), PjRtCopyFuture(std::move(future), {})};
}

TEST(FailAfterTest, ReportsOnlyOnceEveryIssuedCopyHasCompleted) {
  PendingCopy a = MakePendingCopy();
  PendingCopy b = MakePendingCopy();
  std::vector<PjRtCopyFuture> issued = {a.future, b.future};

  PjRtCopyFuture failed =
      FailAfter(issued, absl::InternalError("dispatch failed"));
  EXPECT_FALSE(failed.IsReady());

  a.promise.Set();
  EXPECT_FALSE(failed.IsReady());

  // An issued copy's own error does not replace the dispatch error.
  b.promise.Set(absl::UnavailableError("copy failed"));
  EXPECT_EQ(failed.future.Await(), absl::InternalError("dispatch failed"));
}

TEST(FailAfterTest, FailsAtOnceWhenEveryIssuedCopyIsDone) {
  PendingCopy a = MakePendingCopy();
  a.promise.Set();
  std::vector<PjRtCopyFuture> issued = {a.future};

  PjRtCopyFuture failed =
      FailAfter(issued, absl::InternalError("dispatch failed"));
  EXPECT_EQ(failed.future.Await(), absl::InternalError("dispatch failed"));
}

TEST(FailAfterTest, JoiningItWithOtherCopiesWaitsForThemToo) {
  PendingCopy issued_copy = MakePendingCopy();
  PendingCopy other_group_copy = MakePendingCopy();
  std::vector<PjRtCopyFuture> issued = {issued_copy.future};
  PjRtCopyFuture failed =
      FailAfter(issued, absl::InternalError("dispatch failed"));

  std::vector<PjRtCopyFuture> all = {failed, other_group_copy.future};
  PjRtCopyFuture joined = JoinPjRtCopyFutures(all);
  issued_copy.promise.Set();
  EXPECT_FALSE(joined.IsReady());

  other_group_copy.promise.Set();
  EXPECT_EQ(joined.future.Await(), absl::InternalError("dispatch failed"));
}

TEST(FailedCopyFutureTest, IsReadyWithTheError) {
  PjRtCopyFuture failed = FailedCopyFuture(absl::NotFoundError("no group"));
  EXPECT_EQ(failed.future.Await(), absl::NotFoundError("no group"));
}

TEST(OnReadyTest, FutureOnlyCopyReportsItsOutcome) {
  PendingCopy copy = MakePendingCopy();
  absl::Status seen = absl::UnknownError("not called");
  copy.future.OnReady([&seen](absl::StatusOr<BufferHolders> holds) {
    seen = holds.status();
  });
  EXPECT_EQ(seen, absl::UnknownError("not called"));
  copy.promise.Set(absl::DataLossError("bad"));
  EXPECT_EQ(seen, absl::DataLossError("bad"));
}

}  // namespace
}  // namespace raiden
