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

#ifndef TPU_SYNC_HOST_LEADER_ADAPTER_H_
#define TPU_SYNC_HOST_LEADER_ADAPTER_H_

#include <memory>
#include <string>
#include <vector>

#include "tpu_sync/transport/block_transport_delegate.h"
#include "tpu_sync/transport/lib/transport_adapter.h"

namespace tpu_raiden::host_leader {
// Owned by BlockTransport; its worker endpoint drains before delegate teardown.
class Adapter : public transport::lib::TransportAdapter {
 public:
  Adapter(transport::BlockTransportDelegate* delegate, int rank_port,
          absl::Span<const std::string> rank_ips);
  ~Adapter() override;
  absl::StatusOr<transport::lib::Handle> Post(
      absl::Span<const std::string> peers,
      absl::Span<const transport::lib::Request> requests,
      absl::Span<const int> src_ids, absl::Span<const int> dst_ids,
      transport::lib::CompletionCallback on_complete) override;
  absl::StatusOr<transport::lib::Status> Poll(transport::lib::Handle) override;
  void Forget(uint64_t uuid);

 private:
  class Impl;
  std::shared_ptr<Impl> impl_;
};
}  // namespace tpu_raiden::host_leader
#endif
