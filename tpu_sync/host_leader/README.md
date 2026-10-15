# Host Transport：独立宿主 H2H 通道

## TL;DR

Host Transport 是独立、可选的字节传输层。每个宿主启动一个不初始化 TPU 的 daemon，
统一维护到远端宿主的 TCP 通道；worker 通过一条 abstract UDS session 注册 memfd
内存、endpoint 和传输 job。daemon 只处理 memory spans、opaque control bytes 和
Read/Write，不理解 UUID、pool、plan、expected pushes 或 DMA。

`TPU_RAIDEN_HOST_LEADER_SOCKET` 启用 Torch pool P/D bridge。bridge 保留原 plan、
staging lease、D2H/H2D 和 pool 完成语义；关闭环境变量时仍使用原 allocator 与
SocketTransportAdapter。通用 API 支持双端主动 Read/Write；当前 P/D bridge 仅接入
显式 pool `op6` push，尚未接入原 legacy pull、weights、JAX 或 host-only manager。

## 独立边界

- `protocol.{h,cc}` / `internal.h`：通用 frame、SCM_RIGHTS、socket I/O 和元数据编码。
- `memory.{h,cc}` / `session.{h,cc}`：worker 的 memfd / Region、Session、Endpoint、
  Job 和 completion；不依赖 PjRt、Torch 或 BlockTransport。
- `service.{h,cc}` / `daemon_main.cc`：内存注册表、peer 通道预算、Read/Write 执行与
  stats；不链接 RawBufferTransport 或 P/D delegate。
- `allocator.{h,cc}` / `adapter.{h,cc}`：Torch pool P/D bridge，负责 NUMA first touch、
  DMA map/unmap、opaque pool metadata、Begin/EndIncomingPush 和 pool progress。
  旧路径只在 allocator / transport factory 与 Forget 边界接入。

`Prepare` 必须立即解析已经 ready 的 buffer 并 pin，不能等待依赖另一 data job 的
数据准备，否则可能占住全部通道而死锁。未 ready 应返回 FailedPrecondition；
调用者在 Submit 前安排依赖。通用 Read 表示主动方写本地目标，Write 表示主动方读本地
源；远端 handler 分别提供可读源或可写目标。

## 环境前提与启动

需要 Linux memfd、SCM_RIGHTS 和 abstract AF_UNIX。worker 与本机 daemon 须处在能互相
访问 abstract UDS 的同一 network namespace，以相同 UID 运行；服务端通过 SO_PEERCRED
检查 UID。跨容器时须明确配置 network namespace。memfd FD 经 UDS 传递，不依赖共享
`/dev/shm` mount，也不创建有名称的 SHM 或每 worker 的 UDS 文件。

在仓库既有 Linux / Torch ABI 构建环境运行：

```bash
bazel build //tpu_sync/host_leader:daemon_main
bazel test //tpu_sync/host_leader:generic_test \
  //tpu_sync/host_leader:adapter_test --test_output=errors
```

每宿主独立启动 daemon；不能从已经初始化 TPU 的 worker fork：

```bash
bazel-bin/tpu_sync/host_leader/daemon_main \
  --socket=@raiden-host-transport-<本次运行> \
  --listen=0.0.0.0:19600 --streams=1
```

stderr 的 `HOST_TRANSPORT_READY` 表示开始接收。所有参与 P/D worker 设置：

```bash
export TPU_RAIDEN_HOST_LEADER_SOCKET=@raiden-host-transport-<本次运行>
export TPU_RAIDEN_DATA_CONN_REUSE=pooled
export TPU_RAIDEN_HOST_LEADER_PEERS='10.117.82.48=host-48@10.117.82.48>10.117.82.48:19600+10.117.83.48>10.117.83.48:19600;10.117.83.48=host-48@10.117.82.48>10.117.82.48:19600+10.117.83.48>10.117.83.48:19600'
```

peer 语法为 `rank_ip=host_key@local_ip>remote_endpoint+...`，以分号分隔 rank IP alias。
`local_ip` 是主动发起 job 的本机源 IP；`remote_endpoint` 是远端 leader 的 TCP 地址，
不能误填 rank listener。上例是同宿主双 NIC 的 P/D 实验示例；多宿主部署应填各主动
宿主自己的源 IP 和各远端宿主的 leader IP。属于同一远端宿主的所有 alias 必须使用
相同 `host_key` 和完全一致的有序 routes；service 也跨 session 校验此约束。
显式 source bind 失败直接报错，不回退到其他本地 IP。

P/D bridge 当前要求单目标 rank、layer-major、非空、已注册 pool plan 的 push；
不支持 persistent recovery SHM 组合。不要为未接入的 manager 配置该入口。
`FLAGS_require_psp_tcp=true` 在通用 Session / Service 公共入口显式拒绝，不能静默
使用普通 TCP。该实验不修改正式 wheel pin 或镜像发布输入。

## 通道预算与完成契约

对每个有向 `initiator-host → remote-host`，daemon 固定持有最多 `streams=p`
个 TCP slot。所有 worker、rank alias、routes、Read 和 Write 共享这组 slot；slot 按
编号选择 route。两端同时主动提交时，各有自己的 p 预算，合计最多 2p。
单向 P→D 部署的连接上限为 `H_P × H_D × p`，不是双向无向 host pair 的总上限。
UDS、原 rank 控制 listener 和入站 TCP 不计入 stats 的 outgoing connections。

Region 一次注册后复用 daemon mmap。memfd 固定 size，并要求 F_SEAL_GROW / F_SEAL_SHRINK；
不允许可随意 truncate 的外部 FD。worker 自己拥有 DMA 注册，daemon 只 mmap。
Job 持有本地 Span/Region；Session 不永久持有 Job/Region。远端 Prepare 返回的 spans
由 Session 保留至 I/O 停止，再于 Finish 前释放；如果 handler 在 Finish 中仍要访问
buffer，handler 须另持 Region 和应用 pin。

正常成功顺序是：payload I/O 停止 → 释放 daemon 的本 job 映射 I/O 引用 →
worker Finish → Finish ACK → 源 Job completion。Retire 停止该 region 新 admission，
等待已接收 I/O drain，关闭 daemon mmap/FD 后 ACK；它不等待业务 Finish，因此允许
Finish 释放最后一个 Region 并同步 Retire。释放 allocation 不会取消其他独立 job。

Cancel 对排队 job 立即取消；对已开始 job 标记取消并自然 drain 到完成或 deadline，
随后报告 Cancelled，不承诺立即中断 active payload。默认 job timeout 为 120 秒，
包含排队和数据阶段；跨宿主传剩余 duration。TCP connect 复用既有 ConnectToPeer 的
固定 3 秒 bound。只在新 job 开始前检查并重建已关闭的 idle channel；
发送后失败不自动重放 payload。

Job terminal 恒定保证本地 buffer 不再被 daemon I/O 访问；成功还保证远端 Finish ACK。
网络断开或 timeout 不能证明远端应用 Finish 已返回，远端会独立停止 I/O 后 Finish。
worker 控制发送/解析失败会 half-close 请求方向，继续等 daemon close-after-drain EOF，
不能把本地 IPC 错误视为允许提前复用 lease 的证据。

关闭 Endpoint/Session 应由外部 owner 发起；在自身 Prepare/Finish callback 中同步 Close
返回 FailedPrecondition，不能 drain 自己。callback 内释放最后 Endpoint/Session 引用时，
析构将清理交给独立 closer，避免自 join。正常关闭停止 admission，取消/排空对应任务，
完成 Finish 后关闭控制连接。worker 异常退出后 daemon 先中断并 drain 对应 payload，
再回收 session、endpoint、memfd/mmap；daemon 保持存活也应归零。
SIGTERM/SIGINT 触发 daemon Stop drain；SIGKILL 时由内核关闭全部 fd/mapping。
memfd 不产生需要扫描删除的名字残留。

## 资源验收

```bash
bazel-bin/tpu_sync/host_leader/daemon_main \
  --socket=@raiden-host-transport-<本次运行> --stats
```

成功时 stdout 仅输出一个 JSON 对象，exit 0：

```json
{"sessions":0,"endpoints":0,"registered_regions":0,"mapped_bytes":0,"queued":0,"prepared":0,"inflight":0,"connections":0}
```

stats 查询自身不计 worker session。停全部 worker 后保持同一 daemon PID 存活，
应等待前七项归零；outgoing idle connections 可以按已配置 host budget 保留。
同时比较 daemon 的 `/proc/<pid>/fd` 中 memfd 数量与 `/proc/<pid>/maps` 中 memfd
映射相对启动基线归零。不能同时杀掉 daemon 后用内核自动回收作为 worker 清理证明。

CPU 长期测试保护双端主动 Read/Write、有向 p 预算、双 NIC 共享、跨 session routes
冲突、strict bind、memfd 复用与 Retire、Finish 内释放最后 Region、Prepared spans
所有权、非法 span、取消、daemon EOF 与 worker SIGKILL 的 drain。P/D 测试继续检查
跨 rank 同 UUID 隔离、block/chunk 内容与 pool 完成一次。

真实 TPU 应使用父仓 `pd_disagg/scripts/run_all.sh`，检查完整 `49 requests / 0 failures`、
prefix cache、MTP、双端 bounded staging，并在 canonical 停 worker 后采集 daemon 存活
的资源归零。旧命名 SHM 原型的单宿主结果不能代替本实现的验收。

## 尚未覆盖的结论

当前未将 daemon bytes/latency 接入原 transport metrics，也未实现 PSP。
通用层已经支持 Read，P/D legacy pull bridge 尚未接入；本实现不改变旧 Pull 行为。
单宿主正确性和连接数不能证明多机吞吐、NIC 带宽或延迟收益。性能结论必须重复采样，
记录 count、avg、min、max、stddev、p90、p99。
