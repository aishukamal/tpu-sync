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

#ifndef THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_KV_CACHE_DATA_CONNECTION_POLICY_H_
#define THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_KV_CACHE_DATA_CONNECTION_POLICY_H_

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>

#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "tpu_sync/transport/block_transport.h"

namespace tpu_raiden {

// What a sender does with its data-plane TCP connections to a peer once a send
// session to that peer has settled.
enum class DataConnectionReuse {
  // Connections stay in the transport's pool for the life of the process (the
  // historical behaviour): every (sender rank, receiver worker) pair the
  // process ever pushed to keeps `parallelism` sockets open, and the receiver
  // keeps one connection thread per socket.
  kPooled,
  // The sender closes its idle connections to the peer when the session
  // settles; the next session to that peer dials afresh. The open connection
  // count then follows the transfers in flight instead of the number of peers
  // ever seen, which is what keeps a prefill host within its socket budget and
  // a decode worker within its thread budget at high replica fan-out.
  kPerTransfer,
};

// Environment variable the policy is read from: "pooled" (default) or
// "per_transfer", case-insensitive. Read on every settle, so it only has to be
// set in the sender's environment; receivers need nothing.
inline constexpr absl::string_view kDataConnectionReuseEnvVar =
    "TPU_RAIDEN_DATA_CONN_REUSE";

std::optional<DataConnectionReuse> ParseDataConnectionReuse(
    absl::string_view value);

absl::string_view DataConnectionReuseName(DataConnectionReuse policy);

// The policy named by TPU_RAIDEN_DATA_CONN_REUSE; kPooled when the variable
// is unset or empty. An unparsable value is logged once and treated as kPooled.
DataConnectionReuse ResolveDataConnectionReuse();

// Applies the resolved policy after the send session `uuid` to `peers` has
// settled: under kPerTransfer closes the sender's idle connections to each
// peer through `transport`; a no-op under kPooled or when `transport` is null.
// Must be called once every push of the session has completed (so every
// stream has returned its socket to the pool) and outside the session's own
// lock. Returns the number of connections closed.
size_t ReleaseDataConnectionsAfterSend(transport::BlockTransport* transport,
                                       absl::Span<const std::string> peers,
                                       uint64_t uuid);

}  // namespace tpu_raiden

#endif  // THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_KV_CACHE_DATA_CONNECTION_POLICY_H_
