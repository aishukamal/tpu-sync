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

#include "tpu_sync/kv_cache/data_connection_policy.h"

#include <cstdlib>
#include <optional>
#include <string>
#include <vector>

#include <gtest/gtest.h>

namespace tpu_raiden {
namespace {

class ScopedEnv {
 public:
  ScopedEnv(const char* name, const char* value) : name_(name) {
    if (const char* prev = std::getenv(name)) previous_ = prev;
    if (value == nullptr) {
      unsetenv(name);
    } else {
      setenv(name, value, 1);
    }
  }
  ~ScopedEnv() {
    if (previous_.has_value()) {
      setenv(name_, previous_->c_str(), 1);
    } else {
      unsetenv(name_);
    }
  }

 private:
  const char* name_;
  std::optional<std::string> previous_;
};

constexpr char kEnv[] = "TPU_RAIDEN_DATA_CONN_REUSE";

TEST(DataConnectionPolicyTest, ParsesPolicyNamesCaseInsensitively) {
  EXPECT_EQ(ParseDataConnectionReuse("pooled"), DataConnectionReuse::kPooled);
  EXPECT_EQ(ParseDataConnectionReuse("Per_Transfer"),
            DataConnectionReuse::kPerTransfer);
  EXPECT_EQ(ParseDataConnectionReuse("per-transfer"), std::nullopt);
  EXPECT_EQ(ParseDataConnectionReuse(""), std::nullopt);
  EXPECT_EQ(DataConnectionReuseName(DataConnectionReuse::kPerTransfer),
            "per_transfer");
}

TEST(DataConnectionPolicyTest, DefaultsToPooledWhenUnsetOrEmpty) {
  {
    ScopedEnv env(kEnv, nullptr);
    EXPECT_EQ(ResolveDataConnectionReuse(), DataConnectionReuse::kPooled);
  }
  {
    ScopedEnv env(kEnv, "");
    EXPECT_EQ(ResolveDataConnectionReuse(), DataConnectionReuse::kPooled);
  }
}

TEST(DataConnectionPolicyTest, ReadsPerTransferFromEnvAndRejectsUnknown) {
  {
    ScopedEnv env(kEnv, "per_transfer");
    EXPECT_EQ(ResolveDataConnectionReuse(), DataConnectionReuse::kPerTransfer);
  }
  {
    ScopedEnv env(kEnv, "sometimes");
    EXPECT_EQ(ResolveDataConnectionReuse(), DataConnectionReuse::kPooled);
  }
}

TEST(DataConnectionPolicyTest, ReleaseIsANoOpWithoutTransportOrPeers) {
  ScopedEnv env(kEnv, "per_transfer");
  const std::vector<std::string> peers = {"127.0.0.1:1"};
  EXPECT_EQ(ReleaseDataConnectionsAfterSend(nullptr, peers, /*uuid=*/7), 0u);
  EXPECT_EQ(ReleaseDataConnectionsAfterSend(nullptr, {}, /*uuid=*/7), 0u);
}

}  // namespace
}  // namespace tpu_raiden
