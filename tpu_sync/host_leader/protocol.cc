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

#include <arpa/inet.h>
#include <fcntl.h>
#include <netdb.h>
#include <poll.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <limits>
#include <stdexcept>

#include "absl/strings/str_cat.h"
#include "tpu_sync/host_leader/internal.h"
namespace tpu_raiden::host_leader {
Fd::~Fd() { reset(); }
Fd::Fd(Fd&& o) noexcept : fd_(o.release()) {}
Fd& Fd::operator=(Fd&& o) noexcept {
  if (this != &o) reset(o.release());
  return *this;
}
int Fd::release() {
  int fd = fd_;
  fd_ = -1;
  return fd;
}
void Fd::reset(int fd) {
  if (fd_ >= 0) close(fd_);
  fd_ = fd;
}
namespace internal {
namespace {
constexpr size_t kMaxFrame = 8 << 20;
struct Encoder {
  std::string data;
  void U(uint64_t n) {
    for (int i = 7; i >= 0; --i) data.push_back((n >> (i * 8)) & 255);
  }
  void S(const std::string& s) {
    U(s.size());
    data.append(s);
  }
};
struct Decoder {
  const std::string& data;
  size_t at = 0;
  uint64_t U() {
    if (data.size() - at < 8)
      throw std::runtime_error("truncated transport frame");
    uint64_t n = 0;
    for (int i = 0; i < 8; ++i) n = (n << 8) | static_cast<uint8_t>(data[at++]);
    return n;
  }
  size_t Count(size_t element = 1) {
    uint64_t n = U();
    if (n > kMaxFrame || n > (data.size() - at) / element)
      throw std::runtime_error("transport vector exceeds limit");
    return n;
  }
  std::string S() {
    size_t n = Count();
    if (n > data.size() - at)
      throw std::runtime_error("truncated transport string");
    auto s = data.substr(at, n);
    at += n;
    return s;
  }
};
std::string Encode(const Frame& f) {
  Encoder e;
  e.U(0x485452414e533032ULL);
  e.U(static_cast<uint64_t>(f.type));
  e.U(f.id);
  e.U(static_cast<uint64_t>(f.status.code()));
  e.S(std::string(f.status.message()));
  e.U(f.words.size());
  for (auto n : f.words) e.U(n);
  e.S(f.key);
  e.S(f.data);
  e.U(f.spans.size());
  for (const auto& s : f.spans) {
    e.U(s.region_id);
    e.U(s.offset);
    e.U(s.bytes);
  }
  e.S(f.peer.host_key);
  e.U(f.peer.routes.size());
  for (const auto& r : f.peer.routes) {
    e.S(r.local_ip);
    e.S(r.remote_endpoint);
  }
  if (e.data.size() > kMaxFrame)
    throw std::runtime_error("transport frame exceeds limit");
  return std::move(e.data);
}
Frame Decode(const std::string& data) {
  Decoder d{data};
  if (d.U() != 0x485452414e533032ULL)
    throw std::runtime_error("transport wire version mismatch");
  Frame f;
  f.type = static_cast<Type>(d.U());
  f.id = d.U();
  auto code = d.U();
  if (code > 16) throw std::runtime_error("invalid transport status");
  auto message = d.S();
  f.status = absl::Status(static_cast<absl::StatusCode>(code), message);
  f.words.resize(d.Count(8));
  for (auto& n : f.words) n = d.U();
  f.key = d.S();
  f.data = d.S();
  f.spans.resize(d.Count(24));
  for (auto& s : f.spans) {
    s.region_id = d.U();
    s.offset = d.U();
    s.bytes = d.U();
  }
  f.peer.host_key = d.S();
  f.peer.routes.resize(d.Count(16));
  for (auto& r : f.peer.routes) {
    r.local_ip = d.S();
    r.remote_endpoint = d.S();
  }
  if (d.at != data.size())
    throw std::runtime_error("trailing transport metadata");
  return f;
}
void Poll(int fd, short event, Deadline deadline) {
  int timeout = -1;
  if (deadline != InfiniteDeadline()) {
    auto remaining = std::chrono::duration_cast<std::chrono::milliseconds>(
                         deadline - std::chrono::steady_clock::now())
                         .count();
    if (remaining <= 0)
      Throw(absl::DeadlineExceededError("transport deadline exceeded"));
    timeout = static_cast<int>(std::min<int64_t>(remaining, INT32_MAX));
  }
  pollfd p{fd, event, 0};
  int rc;
  do {
    rc = poll(&p, 1, timeout);
  } while (rc < 0 && errno == EINTR);
  if (rc == 0)
    Throw(absl::DeadlineExceededError("transport deadline exceeded"));
  if (rc < 0) throw std::runtime_error("transport poll failed");
}
int Unix(const std::string& name, bool server) {
  sockaddr_un addr{};
  addr.sun_family = AF_UNIX;
  bool abstract = !name.empty() && name.front() == '@';
  if (name.empty() || name.size() >= sizeof(addr.sun_path))
    throw std::runtime_error("invalid transport UDS name");
  std::memcpy(addr.sun_path, name.data(), name.size());
  if (abstract) addr.sun_path[0] = 0;
  socklen_t n =
      offsetof(sockaddr_un, sun_path) + name.size() + (abstract ? 0 : 1);
  Fd fd(socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0));
  if (fd.get() < 0) throw std::runtime_error("cannot create transport IPC");
  int size = 16 << 20;
  setsockopt(fd.get(), SOL_SOCKET, SO_SNDBUF, &size, sizeof(size));
  setsockopt(fd.get(), SOL_SOCKET, SO_RCVBUF, &size, sizeof(size));
  int rc = server ? bind(fd.get(), reinterpret_cast<sockaddr*>(&addr), n)
                  : connect(fd.get(), reinterpret_cast<sockaddr*>(&addr), n);
  if (rc != 0 || (server && listen(fd.get(), 128) != 0))
    throw std::runtime_error(
        absl::StrCat("transport IPC ", name, ": ", std::strerror(errno)));
  return fd.release();
}
}  // namespace
namespace {
class StatusError : public std::runtime_error {
 public:
  explicit StatusError(absl::Status value)
      : std::runtime_error(value.ToString()), status(std::move(value)) {}
  absl::Status status;
};
}  // namespace
void Throw(const absl::Status& s) {
  if (!s.ok()) throw StatusError(s);
}
absl::Status Error(const std::exception& e) {
  if (const auto* status = dynamic_cast<const StatusError*>(&e))
    return status->status;
  return absl::UnavailableError(e.what());
}
Frame Reply(uint64_t id, absl::Status s) {
  Frame f;
  f.type = Type::kReply;
  f.id = id;
  f.status = std::move(s);
  return f;
}
void Read(int fd, void* ptr, size_t n, Deadline deadline) {
  auto* p = static_cast<uint8_t*>(ptr);
  while (n) {
    Poll(fd, POLLIN, deadline);
    ssize_t r = recv(fd, p, n, MSG_DONTWAIT);
    if (r < 0 && (errno == EINTR || errno == EAGAIN)) continue;
    if (r <= 0) throw std::runtime_error("transport peer closed during read");
    p += r;
    n -= r;
  }
}
void Write(int fd, const void* ptr, size_t n, Deadline deadline) {
  auto* p = static_cast<const uint8_t*>(ptr);
  while (n) {
    Poll(fd, POLLOUT, deadline);
    ssize_t r = send(fd, p, n, MSG_NOSIGNAL | MSG_DONTWAIT);
    if (r < 0 && (errno == EINTR || errno == EAGAIN)) continue;
    if (r <= 0) throw std::runtime_error("transport peer closed during write");
    p += r;
    n -= r;
  }
}
void SendFrame(int fd, const Frame& f, Deadline deadline) {
  auto bytes = Encode(f);
  Encoder e;
  e.U(bytes.size());
  Write(fd, e.data.data(), e.data.size(), deadline);
  Write(fd, bytes.data(), bytes.size(), deadline);
}
Frame ReceiveFrame(int fd, Deadline deadline) {
  std::string length(8, '\0');
  Read(fd, length.data(), 8, deadline);
  Decoder d{length};
  size_t n = d.U();
  if (n > kMaxFrame) throw std::runtime_error("transport frame exceeds limit");
  std::string bytes(n, '\0');
  Read(fd, bytes.data(), n, deadline);
  return Decode(bytes);
}
void SendPacket(int fd, const Frame& f) {
  auto bytes = Encode(f);
  Encoder header;
  header.U(bytes.size());
  iovec iov{header.data.data(), header.data.size()};
  msghdr msg{};
  msg.msg_iov = &iov;
  msg.msg_iovlen = 1;
  alignas(cmsghdr) char control[CMSG_SPACE(sizeof(int))]{};
  if (f.fd.get() >= 0) {
    msg.msg_control = control;
    msg.msg_controllen = sizeof(control);
    auto* c = CMSG_FIRSTHDR(&msg);
    c->cmsg_level = SOL_SOCKET;
    c->cmsg_type = SCM_RIGHTS;
    c->cmsg_len = CMSG_LEN(sizeof(int));
    int value = f.fd.get();
    std::memcpy(CMSG_DATA(c), &value, sizeof(value));
  }
  ssize_t n;
  do {
    n = sendmsg(fd, &msg, MSG_NOSIGNAL);
  } while (n < 0 && errno == EINTR);
  if (n <= 0) throw std::runtime_error("transport IPC header send failed");
  Write(fd, header.data.data() + n, header.data.size() - n);
  Write(fd, bytes.data(), bytes.size());
}
Frame ReceivePacket(int fd) {
  std::string header(8, '\0');
  iovec iov{header.data(), header.size()};
  msghdr msg{};
  msg.msg_iov = &iov;
  msg.msg_iovlen = 1;
  alignas(cmsghdr) char control[CMSG_SPACE(sizeof(int) * 8)]{};
  msg.msg_control = control;
  msg.msg_controllen = sizeof(control);
  ssize_t n;
  do {
    n = recvmsg(fd, &msg, MSG_CMSG_CLOEXEC);
  } while (n < 0 && errno == EINTR);
  std::vector<Fd> fds;
  for (auto* c = CMSG_FIRSTHDR(&msg); c; c = CMSG_NXTHDR(&msg, c))
    if (c->cmsg_level == SOL_SOCKET && c->cmsg_type == SCM_RIGHTS) {
      size_t count = (c->cmsg_len - CMSG_LEN(0)) / sizeof(int);
      auto* p = reinterpret_cast<int*>(CMSG_DATA(c));
      for (size_t i = 0; i < count; ++i) fds.emplace_back(p[i]);
    }
  if (n <= 0) throw std::runtime_error("transport IPC closed");
  if (msg.msg_flags & (MSG_TRUNC | MSG_CTRUNC) || fds.size() > 1)
    throw std::runtime_error("truncated transport IPC/FD header");
  Read(fd, header.data() + n, header.size() - n);
  Decoder d{header};
  size_t size = d.U();
  if (size > kMaxFrame)
    throw std::runtime_error("transport frame exceeds limit");
  std::string bytes(size, '\0');
  Read(fd, bytes.data(), size);
  Frame f = Decode(bytes);
  if (!fds.empty()) f.fd = std::move(fds.front());
  return f;
}
int ConnectUnix(const std::string& s) { return Unix(s, false); }
int ListenUnix(const std::string& s) { return Unix(s, true); }
int ListenTcp(const std::string& endpoint) {
  size_t colon = endpoint.rfind(':');
  if (colon == std::string::npos)
    throw std::runtime_error("transport listener requires host:port");
  std::string host = endpoint.substr(0, colon),
              port = endpoint.substr(colon + 1);
  if (host.empty() || port.empty())
    throw std::runtime_error("listener requires nonempty host and port");
  if (host.front() == '[') host = host.substr(1, host.size() - 2);
  addrinfo hints{};
  hints.ai_family = AF_UNSPEC;
  hints.ai_socktype = SOCK_STREAM;
  hints.ai_flags = AI_PASSIVE;
  addrinfo* addresses = nullptr;
  if (getaddrinfo(host.c_str(), port.c_str(), &hints, &addresses) != 0)
    throw std::runtime_error("transport listener resolution failed");
  int result = -1;
  for (auto* a = addresses; a; a = a->ai_next) {
    Fd fd(socket(a->ai_family, SOCK_STREAM | SOCK_CLOEXEC, 0));
    if (fd.get() < 0) continue;
    int one = 1;
    setsockopt(fd.get(), SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
    if (bind(fd.get(), a->ai_addr, a->ai_addrlen) == 0 &&
        listen(fd.get(), 128) == 0) {
      result = fd.release();
      break;
    }
  }
  freeaddrinfo(addresses);
  if (result < 0) throw std::runtime_error("cannot bind transport listener");
  return result;
}
std::vector<uint64_t> EncodeStats(const Stats& s) {
  return {s.sessions, s.endpoints, s.registered_regions, s.mapped_bytes,
          s.queued,   s.prepared,  s.inflight,           s.connections};
}
Stats DecodeStats(const std::vector<uint64_t>& v) {
  if (v.size() != 8) throw std::runtime_error("invalid transport stats");
  return {v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7]};
}
Executor::Executor(size_t n) {
  for (size_t i = 0; i < n; ++i)
    workers_.emplace_back([this] {
      while (true) {
        std::function<void()> fn;
        {
          std::unique_lock<std::mutex> l(mu_);
          cv_.wait(l, [this] { return stopping_ || !queue_.empty(); });
          if (stopping_ && queue_.empty()) return;
          fn = std::move(queue_.front());
          queue_.pop();
          ++active_;
        }
        fn();
        {
          std::lock_guard<std::mutex> l(mu_);
          --active_;
          if (queue_.empty() && active_ == 0) drained_.notify_all();
        }
      }
    });
}
void Executor::Post(std::function<void()> fn) {
  {
    std::lock_guard<std::mutex> l(mu_);
    if (stopping_) throw std::runtime_error("transport executor closed");
    queue_.push(std::move(fn));
  }
  cv_.notify_one();
}
void Executor::Drain() {
  std::unique_lock<std::mutex> l(mu_);
  drained_.wait(l, [this] { return queue_.empty() && active_ == 0; });
}
Executor::~Executor() {
  Drain();
  {
    std::lock_guard<std::mutex> l(mu_);
    stopping_ = true;
  }
  cv_.notify_all();
  for (auto& t : workers_) t.join();
}
}  // namespace internal
}  // namespace tpu_raiden::host_leader
