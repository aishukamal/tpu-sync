# Changelog

## [Unreleased]

### Added

- `TPU_RAIDEN_DATA_CONN_REUSE=per_transfer`: a sender closes its idle
  data-plane TCP connections to a peer when a send session to that peer
  settles (pool reshard and legacy pull-serve), so open connections follow the
  transfers in flight rather than every (rank, receiver) pair ever pushed to;
  the receiver's per-connection thread ends on the resulting EOF. Default
  `pooled` keeps the historical behaviour. `BlockTransport` /
  `RawBufferTransport` gained `CloseIdleConnections(peer)` and
  `IdleConnectionCount(peer)`; `ConnPool` gained `CloseIdle` / `IdleCount`.

[Unreleased]: https://github.com/tpu-raiden/tpu_raiden/compare/v0.1.0...HEAD
