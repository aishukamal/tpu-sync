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

#include <cstdlib>
#include <optional>
#include <string>

#include <gtest/gtest.h>
#include "tpu_sync/kv_cache/reshard/pool_reshard_planner.h"

namespace tpu_raiden {
namespace kv_cache {
namespace reshard {
namespace {

class ScopedStreamsEnv {
 public:
  explicit ScopedStreamsEnv(const char* value) {
    if (const char* prev = std::getenv(kTransferStreamsEnvVar)) previous_ = prev;
    if (value == nullptr) {
      unsetenv(kTransferStreamsEnvVar);
    } else {
      setenv(kTransferStreamsEnvVar, value, 1);
    }
  }
  ~ScopedStreamsEnv() {
    if (previous_.has_value()) {
      setenv(kTransferStreamsEnvVar, previous_->c_str(), 1);
    } else {
      unsetenv(kTransferStreamsEnvVar);
    }
  }

 private:
  std::optional<std::string> previous_;
};

TEST(DefaultTransferStreamsTest, UnsetOrZeroKeepsOneStreamPerSourceRank) {
  { ScopedStreamsEnv env(nullptr); EXPECT_EQ(DefaultTransferStreams(8), 8); }
  { ScopedStreamsEnv env(""); EXPECT_EQ(DefaultTransferStreams(8), 8); }
  { ScopedStreamsEnv env("0"); EXPECT_EQ(DefaultTransferStreams(8), 8); }
}

TEST(DefaultTransferStreamsTest, PositiveValueCapsStreamsWithinAdmission) {
  { ScopedStreamsEnv env("1"); EXPECT_EQ(DefaultTransferStreams(8), 1); }
  { ScopedStreamsEnv env("4"); EXPECT_EQ(DefaultTransferStreams(8), 4); }
  { ScopedStreamsEnv env("16"); EXPECT_EQ(DefaultTransferStreams(8), 8); }
}

TEST(DefaultTransferStreamsTest, UnparsableValueFallsBack) {
  { ScopedStreamsEnv env("many"); EXPECT_EQ(DefaultTransferStreams(8), 8); }
  { ScopedStreamsEnv env("-2"); EXPECT_EQ(DefaultTransferStreams(8), 8); }
}

}  // namespace
}  // namespace reshard
}  // namespace kv_cache
}  // namespace tpu_raiden
