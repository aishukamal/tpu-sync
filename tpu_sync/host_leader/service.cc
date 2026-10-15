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
#include "tpu_sync/host_leader/service.h"

#include <fcntl.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <unistd.h>

#include <atomic>
#include <condition_variable>
#include <deque>
#include <future>
#include <iostream>
#include <map>
#include <mutex>
#include <set>
#include <stdexcept>
#include <thread>
#include <utility>

#include "absl/flags/flag.h"
#include "tpu_sync/common/detached_thread_group.h"
#include "tpu_sync/host_leader/internal.h"
#include "tpu_sync/transport/lib/socket/tcp_psp_helper.h"
#include "tpu_sync/transport/lib/socket/util.h"
namespace tpu_raiden::host_leader {
namespace {
using internal::Deadline;
using internal::Frame;
using internal::Type;
struct ServiceIoResult {
  absl::Status status;
  uint64_t bytes = 0;
};
struct Counters {
  std::atomic<uint64_t> sessions{0}, endpoints{0}, regions{0}, bytes{0},
      queued{0}, prepared{0}, inflight{0}, connections{0};
  Stats Get() {
    return {sessions.load(), endpoints.load(),  regions.load(),
            bytes.load(),    queued.load(),     prepared.load(),
            inflight.load(), connections.load()};
  }
};
struct Mapping {
  Fd fd;
  void* data = MAP_FAILED;
  uint64_t bytes = 0;
  Counters* counts;
  std::mutex mu;
  size_t users = 0;
  bool retired = false;
  std::function<void()> on_drained;
  Mapping(Fd f, uint64_t n, Counters* c)
      : fd(std::move(f)), bytes(n), counts(c) {
    int seals = fcntl(fd.get(), F_GET_SEALS);
    if (seals < 0 || (seals & (F_SEAL_GROW | F_SEAL_SHRINK)) !=
                         (F_SEAL_GROW | F_SEAL_SHRINK))
      internal::Throw(
          absl::InvalidArgumentError("region requires resize-sealed memfd"));
    struct stat st;
    if (fstat(fd.get(), &st) || st.st_size < 0 ||
        static_cast<uint64_t>(st.st_size) != bytes)
      internal::Throw(absl::InvalidArgumentError("registered fd size differs"));
    data =
        mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd.get(), 0);
    if (data == MAP_FAILED) throw std::runtime_error("daemon mmap failed");
    ++counts->regions;
    counts->bytes += bytes;
  }
  ~Mapping() {
    if (data != MAP_FAILED) {
      munmap(data, bytes);
      --counts->regions;
      counts->bytes -= bytes;
    }
  }
  void Retire(std::function<void()> done) {
    std::function<void()> run;
    {
      std::lock_guard<std::mutex> l(mu);
      retired = true;
      on_drained = std::move(done);
      if (!users) run = std::move(on_drained);
    }
    if (run) run();
  }
};
struct IoSpan {
  std::shared_ptr<Mapping> mapping;
  uint64_t offset, bytes;
  IoSpan(std::shared_ptr<Mapping> m, uint64_t o, uint64_t n)
      : mapping(std::move(m)), offset(o), bytes(n) {
    std::lock_guard<std::mutex> l(mapping->mu);
    if (mapping->retired)
      internal::Throw(absl::FailedPreconditionError("region retired"));
    ++mapping->users;
  }
  ~IoSpan() {
    std::function<void()> run;
    {
      std::lock_guard<std::mutex> l(mapping->mu);
      --mapping->users;
      if (!mapping->users) run = std::move(mapping->on_drained);
    }
    mapping.reset();
    if (run) run();
  }
  uint8_t* data() { return static_cast<uint8_t*>(mapping->data) + offset; }
};
struct Task {
  uint64_t id = 0, client_id = 0;
  std::string endpoint;
  Direction direction = Direction::kWrite;
  Deadline deadline;
  std::vector<std::shared_ptr<IoSpan>> spans;
  std::mutex mu;
  bool cancelled = false;
  int active_fd = -1;
  std::function<void()> cancel_queued;
  void Cancel(bool interrupt = true) {
    {
      std::lock_guard<std::mutex> l(mu);
      cancelled = true;
      if (interrupt && active_fd >= 0) shutdown(active_fd, SHUT_RDWR);
    }
    if (cancel_queued) cancel_queued();
  }
  void Attach(int fd) {
    std::lock_guard<std::mutex> l(mu);
    active_fd = fd;
    if (cancelled) shutdown(fd, SHUT_RDWR);
  }
  void Detach() {
    std::lock_guard<std::mutex> l(mu);
    active_fd = -1;
  }
  bool Cancelled() {
    std::lock_guard<std::mutex> l(mu);
    return cancelled;
  }
};
struct ServerSession;
struct Work {
  std::shared_ptr<Task> task;
  std::shared_ptr<ServerSession> session;
  Frame request;
};
bool SamePeer(const Peer& a, const Peer& b) {
  if (a.host_key != b.host_key || a.routes.size() != b.routes.size())
    return false;
  for (size_t i = 0; i < a.routes.size(); ++i)
    if (a.routes[i].local_ip != b.routes[i].local_ip ||
        a.routes[i].remote_endpoint != b.routes[i].remote_endpoint)
      return false;
  return true;
}
struct HostQueue {
  Peer peer;
  Counters* counts;
  std::mutex mu;
  std::condition_variable cv;
  bool stopped = false;
  std::deque<std::shared_ptr<Work>> queue;
  std::vector<std::thread> threads;
  std::function<void(std::shared_ptr<Work>, Fd&, size_t)> run;
  HostQueue(Peer p, size_t n, Counters* c,
            std::function<void(std::shared_ptr<Work>, Fd&, size_t)> fn)
      : peer(std::move(p)), counts(c), run(std::move(fn)) {
    for (size_t i = 0; i < n; ++i)
      threads.emplace_back([this, i] {
        Fd fd;
        while (true) {
          std::shared_ptr<Work> w;
          {
            std::unique_lock<std::mutex> l(mu);
            cv.wait(l, [&] { return stopped || !queue.empty(); });
            if (stopped && queue.empty()) break;
            w = std::move(queue.front());
            queue.pop_front();
            --counts->queued;
          }
          run(w, fd, i);
        }
        if (fd.get() >= 0) --counts->connections;
      });
  }
  void Push(std::shared_ptr<Work> w) {
    std::lock_guard<std::mutex> l(mu);
    if (stopped) throw std::runtime_error("host queue stopped");
    queue.push_back(std::move(w));
    ++counts->queued;
    cv.notify_one();
  }
  std::shared_ptr<Work> Remove(uint64_t id) {
    std::lock_guard<std::mutex> l(mu);
    for (auto it = queue.begin(); it != queue.end(); ++it)
      if ((*it)->task->id == id) {
        auto w = *it;
        queue.erase(it);
        --counts->queued;
        return w;
      }
    return nullptr;
  }
  void Stop() {
    std::vector<std::shared_ptr<Task>> cancel;
    {
      std::lock_guard<std::mutex> l(mu);
      stopped = true;
      for (auto& w : queue) cancel.push_back(w->task);
    }
    for (auto& t : cancel) t->Cancel();
    cv.notify_all();
    for (auto& t : threads)
      if (t.joinable()) t.join();
  }
  ~HostQueue() { Stop(); }
};
struct ServerSession : std::enable_shared_from_this<ServerSession> {
  Fd fd;
  Counters* counts;
  std::mutex mu, writer;
  bool hello = false, closing = false, dead = false, finished = false;
  uint64_t close_id = 0, next_region = 1;
  std::map<uint64_t, std::shared_ptr<Mapping>> regions;
  std::set<std::string> endpoints;
  std::map<uint64_t, std::shared_ptr<Task>> tasks;
  std::map<std::string, std::vector<uint64_t>> endpoint_closers;
  std::map<uint64_t, std::shared_ptr<std::promise<Frame>>> pending;
  std::atomic<uint64_t> next{1};
  std::function<void(std::shared_ptr<ServerSession>)> remove;
  explicit ServerSession(Fd f, Counters* c) : fd(std::move(f)), counts(c) {}
  void Send(const Frame& f) {
    std::lock_guard<std::mutex> l(writer);
    internal::SendPacket(fd.get(), f);
  }
  Frame Call(Frame f) {
    auto p = std::make_shared<std::promise<Frame>>();
    auto result = p->get_future();
    f.id = next.fetch_add(1);
    {
      std::lock_guard<std::mutex> l(mu);
      if (dead) throw std::runtime_error("worker disconnected");
      pending[f.id] = p;
    }
    try {
      Send(f);
    } catch (...) {
      std::lock_guard<std::mutex> l(mu);
      pending.erase(f.id);
      throw;
    }
    return result.get();
  }
  std::vector<std::shared_ptr<IoSpan>> Acquire(
      const std::vector<WireSpan>& spans, uint64_t expected) {
    std::vector<std::shared_ptr<IoSpan>> out;
    uint64_t total = 0;
    std::lock_guard<std::mutex> l(mu);
    if (closing) throw std::runtime_error("session closing");
    for (const auto& s : spans) {
      auto it = regions.find(s.region_id);
      if (it == regions.end() || !s.bytes || s.offset > it->second->bytes ||
          s.bytes > it->second->bytes - s.offset ||
          UINT64_MAX - total < s.bytes)
        internal::Throw(
            absl::InvalidArgumentError("invalid registered memory span"));
      total += s.bytes;
      out.push_back(std::make_shared<IoSpan>(it->second, s.offset, s.bytes));
    }
    if (total != expected)
      internal::Throw(absl::InvalidArgumentError("span bytes differ from job"));
    return out;
  }
  void AddTask(std::shared_ptr<Task> t) {
    std::lock_guard<std::mutex> l(mu);
    if (closing || (!t->endpoint.empty() && !endpoints.count(t->endpoint)))
      throw std::runtime_error("endpoint/session closing");
    tasks[t->id] = std::move(t);
  }
  void DoneTask(uint64_t id) {
    {
      std::lock_guard<std::mutex> l(mu);
      tasks.erase(id);
    }
    CheckDrain();
  }
  void CheckDrain() {
    std::vector<Frame> replies;
    std::map<uint64_t, std::shared_ptr<Mapping>> retired;
    bool finish = false, was_hello = false;
    uint64_t ack_close = 0;
    {
      std::lock_guard<std::mutex> l(mu);
      for (auto it = endpoint_closers.begin(); it != endpoint_closers.end();) {
        bool busy = false;
        for (const auto& [id, t] : tasks)
          if (t->endpoint == it->first) {
            busy = true;
            break;
          }
        if (!busy) {
          for (auto id : it->second) replies.push_back(internal::Reply(id));
          it = endpoint_closers.erase(it);
        } else
          ++it;
      }
      if (closing && tasks.empty() && !finished) {
        finished = true;
        retired.swap(regions);
        finish = true;
        was_hello = hello;
        if (!dead) ack_close = close_id;
        hello = false;
        counts->endpoints -= endpoints.size();
        endpoints.clear();
      }
    }
    for (auto& f : replies) try {
        Send(f);
      } catch (...) {
      }
    if (finish) {
      // No Task owns any actual data-I/O mapping at this point.
      retired.clear();
      if (ack_close) try {
          Send(internal::Reply(ack_close));
        } catch (...) {
        }
      {
        std::lock_guard<std::mutex> l(writer);
        shutdown(fd.get(), SHUT_RDWR);
      }
      if (was_hello) --counts->sessions;
      remove(shared_from_this());
    }
  }
  void BeginClose(uint64_t id, bool eof) {
    std::vector<std::shared_ptr<Task>> cancel;
    std::map<uint64_t, std::shared_ptr<std::promise<Frame>>> ps;
    {
      std::lock_guard<std::mutex> l(mu);
      closing = true;
      close_id = id;
      dead |= eof;
      for (auto& [n, t] : tasks) cancel.push_back(t);
      if (eof) ps.swap(pending);
    }
    for (auto& [n, p] : ps)
      p->set_value(
          internal::Reply(n, absl::UnavailableError("worker disconnected")));
    for (auto& t : cancel) t->Cancel();
    CheckDrain();
  }
};
void Transfer(int fd, const std::vector<std::shared_ptr<IoSpan>>& spans,
              bool write, Deadline deadline) {
  for (const auto& s : spans)
    if (write)
      internal::Write(fd, s->data(), s->bytes, deadline);
    else
      internal::Read(fd, s->data(), s->bytes, deadline);
}
}  // namespace
class Service::Impl {
 public:
  ServiceConfig config;
  Counters counts;
  Fd local, tcp;
  std::thread local_thread, tcp_thread;
  std::mutex mu;
  bool started = false, stopping = false;
  std::map<uint64_t, std::shared_ptr<ServerSession>> sessions;
  std::map<std::string, std::weak_ptr<ServerSession>> endpoints;
  std::map<std::string, std::unique_ptr<HostQueue>> hosts;
  std::set<int> accepted, raw_local;
  std::atomic<uint64_t> next{1};
  DetachedThreadGroup readers{"host transport readers"};
  explicit Impl(ServiceConfig c) : config(std::move(c)) {}
  void Remove(std::shared_ptr<ServerSession> s) {
    std::lock_guard<std::mutex> l(mu);
    for (auto it = endpoints.begin(); it != endpoints.end();)
      if (it->second.lock() == s)
        it = endpoints.erase(it);
      else
        ++it;
    for (auto it = sessions.begin(); it != sessions.end();)
      if (it->second == s)
        it = sessions.erase(it);
      else
        ++it;
  }
  void Local(Fd fd) {
    // Abstract UDS has no filesystem mode bits; retain the original same-UID
    // boundary.
    ucred credentials{};
    socklen_t len = sizeof(credentials);
    if (getsockopt(fd.get(), SOL_SOCKET, SO_PEERCRED, &credentials, &len) ||
        credentials.uid != geteuid())
      return;
    std::shared_ptr<ServerSession> s;
    const int control_fd = fd.get();
    struct Guard {
      std::function<void()> fn;
      ~Guard() { fn(); }
    } guard{[this, control_fd] {
      std::lock_guard<std::mutex> l(mu);
      raw_local.erase(control_fd);
    }};
    {
      std::lock_guard<std::mutex> l(mu);
      if (stopping) return;
      raw_local.insert(control_fd);
    }
    try {
      Frame first = internal::ReceivePacket(fd.get());
      if (first.type == Type::kStats) {
        auto r = internal::Reply(first.id);
        r.words = internal::EncodeStats(counts.Get());
        internal::SendPacket(fd.get(), r);
        return;
      }
      if (first.type != Type::kHello)
        throw std::runtime_error("worker must open a session");
      s = std::make_shared<ServerSession>(std::move(fd), &counts);
      s->remove = [this](auto p) { Remove(std::move(p)); };
      s->hello = true;
      ++counts.sessions;
      {
        std::lock_guard<std::mutex> l(mu);
        if (stopping) throw std::runtime_error("service stopping");
        sessions[next.fetch_add(1)] = s;
        raw_local.erase(control_fd);
      }
      s->Send(internal::Reply(first.id));
      while (true) {
        Frame f = internal::ReceivePacket(s->fd.get());
        if (f.type == Type::kReply) {
          std::shared_ptr<std::promise<Frame>> p;
          {
            std::lock_guard<std::mutex> l(s->mu);
            auto it = s->pending.find(f.id);
            if (it != s->pending.end()) {
              p = it->second;
              s->pending.erase(it);
            }
          }
          if (p) p->set_value(std::move(f));
          continue;
        }
        Frame reply = internal::Reply(f.id);
        bool respond = true;
        try {
          if (f.type == Type::kRegisterRegion) {
            {
              std::lock_guard<std::mutex> l(mu);
              if (stopping)
                internal::Throw(absl::UnavailableError("service stopping"));
            }
            if (f.words.size() != 1 || !f.words[0] || f.fd.get() < 0)
              throw std::runtime_error("invalid registration");
            auto m =
                std::make_shared<Mapping>(std::move(f.fd), f.words[0], &counts);
            {
              std::lock_guard<std::mutex> l(s->mu);
              if (s->closing) throw std::runtime_error("session closing");
              uint64_t id = s->next_region++;
              s->regions[id] = std::move(m);
              reply.words = {id};
            }
          } else if (f.type == Type::kRetireRegion) {
            if (f.words.size() != 1) throw std::runtime_error("bad Retire");
            std::shared_ptr<Mapping> m;
            {
              std::lock_guard<std::mutex> l(s->mu);
              auto it = s->regions.find(f.words[0]);
              if (it != s->regions.end()) {
                m = it->second;
                s->regions.erase(it);
              }
            }
            if (m) {
              respond = false;
              uint64_t id = f.id;
              // Callback owns the registry reference until actual I/O users
              // drain.
              auto holder = std::make_shared<std::shared_ptr<Mapping>>(m);
              auto* raw = m.get();
              m.reset();
              raw->Retire([s, holder, id] {
                holder->reset();
                try {
                  s->Send(internal::Reply(id));
                } catch (...) {
                }
              });
            }
          } else if (f.type == Type::kRegisterEndpoint) {
            if (f.key.empty()) throw std::runtime_error("endpoint required");
            {
              std::lock_guard<std::mutex> l(mu);
              if (stopping)
                internal::Throw(absl::UnavailableError("service stopping"));
              if (endpoints.count(f.key) && !endpoints[f.key].expired())
                internal::Throw(
                    absl::AlreadyExistsError("endpoint already registered"));
              endpoints[f.key] = s;
            }
            {
              std::lock_guard<std::mutex> l(s->mu);
              if (s->closing) throw std::runtime_error("session closing");
              s->endpoints.insert(f.key);
              ++counts.endpoints;
            }
          } else if (f.type == Type::kCloseEndpoint) {
            {
              std::lock_guard<std::mutex> l(mu);
              auto it = endpoints.find(f.key);
              if (it != endpoints.end() && it->second.lock() == s)
                endpoints.erase(it);
            }
            std::vector<std::shared_ptr<Task>> cancel;
            {
              std::lock_guard<std::mutex> l(s->mu);
              if (s->endpoints.erase(f.key)) --counts.endpoints;
              s->endpoint_closers[f.key].push_back(f.id);
              for (auto& [id, t] : s->tasks)
                if (t->endpoint == f.key) cancel.push_back(t);
            }
            for (auto& t : cancel) t->Cancel();
            respond = false;
            s->CheckDrain();
          } else if (f.type == Type::kSubmit) {
            if (f.words.size() != 4 || !f.words[2] || !f.words[3] ||
                f.words[3] > INT32_MAX ||
                (f.words[1] != 1 && f.words[1] != 2) ||
                f.peer.host_key.empty() || f.peer.routes.empty() ||
                f.key.empty())
              throw std::runtime_error("invalid transport job");
            for (const auto& r : f.peer.routes)
              if (r.local_ip.empty() || r.remote_endpoint.empty())
                throw std::runtime_error("route endpoints required");
            auto t = std::make_shared<Task>();
            t->id = next.fetch_add(1);
            t->client_id = f.words[0];
            t->direction = static_cast<Direction>(f.words[1]);
            t->deadline = std::chrono::steady_clock::now() +
                          std::chrono::milliseconds(f.words[3]);
            t->spans = s->Acquire(f.spans, f.words[2]);
            auto w = std::make_shared<Work>();
            w->task = t;
            w->session = s;
            w->request = std::move(f);
            HostQueue* queue;
            {
              std::lock_guard<std::mutex> l(mu);
              if (stopping)
                internal::Throw(absl::UnavailableError("service stopping"));
              auto it = hosts.find(w->request.peer.host_key);
              if (it != hosts.end() &&
                  !SamePeer(it->second->peer, w->request.peer))
                throw std::runtime_error(
                    "host_key routes conflict with existing session");
              if (it == hosts.end()) {
                auto q = std::make_unique<HostQueue>(
                    w->request.peer, config.streams, &counts,
                    [this](auto w, Fd& fd, size_t slot) {
                      Outgoing(std::move(w), fd, slot);
                    });
                queue = q.get();
                hosts.emplace(w->request.peer.host_key, std::move(q));
              } else
                queue = it->second.get();
            }
            std::weak_ptr<Work> weak = w;
            t->cancel_queued = [queue, weak] {
              if (auto current = weak.lock()) {
                auto removed = queue->Remove(current->task->id);
                if (removed) {
                  removed->task->spans.clear();
                  Frame done;
                  done.type = Type::kDone;
                  done.id = removed->task->client_id;
                  done.status = absl::CancelledError("job cancelled in queue");
                  done.words = {0};
                  try {
                    removed->session->Send(done);
                  } catch (...) {
                  }
                  removed->session->DoneTask(removed->task->id);
                }
              }
            };
            s->AddTask(t);
            try {
              queue->Push(w);
            } catch (...) {
              t->spans.clear();
              s->DoneTask(t->id);
              throw;
            }
          } else if (f.type == Type::kCancel) {
            if (f.words.size() != 1) throw std::runtime_error("bad Cancel");
            std::vector<std::shared_ptr<Task>> cancel;
            {
              std::lock_guard<std::mutex> l(s->mu);
              for (auto& [id, t] : s->tasks)
                if (t->client_id == f.words[0]) cancel.push_back(t);
            }
            for (auto& t : cancel) t->Cancel(false);
          } else if (f.type == Type::kClose) {
            respond = false;
            s->BeginClose(f.id, false);
          } else
            throw std::runtime_error("unknown worker request");
        } catch (const std::exception& e) {
          reply.status = internal::Error(e);
        }
        if (respond) s->Send(reply);
      }
    } catch (const std::exception&) {
      if (s) s->BeginClose(0, true);
    }
  }
  void Outgoing(std::shared_ptr<Work> w, Fd& fd, size_t slot) {
    auto t = w->task;
    ++counts.inflight;
    ServiceIoResult result;
    bool success = false;
    try {
      if (t->Cancelled())
        internal::Throw(absl::CancelledError("job cancelled in queue"));
      if (std::chrono::steady_clock::now() >= t->deadline)
        internal::Throw(
            absl::DeadlineExceededError("job queue deadline exceeded"));
      if (fd.get() >= 0) {
        pollfd p{fd.get(), POLLIN, 0};
        if (poll(&p, 1, 0) != 0) {
          fd.reset();
          --counts.connections;
        }
      }
      if (fd.get() < 0) {
        const auto& r =
            w->request.peer.routes[slot % w->request.peer.routes.size()];
        auto connected = transport::lib::ConnectToPeer(
            r.remote_endpoint, r.local_ip, false, nullptr, nullptr, true);
        if (!connected.ok()) internal::Throw(connected.status());
        fd.reset(*connected);
        ++counts.connections;
        std::cerr << "HOST_TRANSPORT_CONNECT host_key="
                  << w->request.peer.host_key << " slot=" << slot
                  << " route=" << r.local_ip << ">" << r.remote_endpoint
                  << "\n";
      }
      t->Attach(fd.get());
      if (t->Cancelled())
        internal::Throw(absl::CancelledError("job cancelled"));
      Frame request;
      request.type = Type::kSubmit;
      request.key = w->request.key;
      request.data = w->request.data;
      auto remaining = std::chrono::duration_cast<std::chrono::milliseconds>(
                           t->deadline - std::chrono::steady_clock::now())
                           .count();
      if (remaining <= 0)
        internal::Throw(
            absl::DeadlineExceededError("job deadline before send"));
      request.words = {static_cast<uint64_t>(t->direction), w->request.words[2],
                       static_cast<uint64_t>(remaining)};
      internal::SendFrame(fd.get(), request, t->deadline);
      auto ready = internal::ReceiveFrame(fd.get(), t->deadline);
      internal::Throw(ready.status);
      Transfer(fd.get(), t->spans, t->direction == Direction::kWrite,
               t->deadline);
      auto terminal = internal::ReceiveFrame(fd.get(), t->deadline);
      internal::Throw(terminal.status);
      result.bytes = w->request.words[2];
      success = true;
      if (t->Cancelled())
        result.status =
            absl::CancelledError("active job drained after cancellation");
    } catch (const std::exception& e) {
      result.status = internal::Error(e);
    }
    // Detach and Cancel use the same mutex, so old cancellation cannot shut
    // down a channel after this slot has begun a successor task.
    t->Detach();
    if (!success && fd.get() >= 0) {
      fd.reset();
      --counts.connections;
    }
    t->spans.clear();
    Frame done;
    done.type = Type::kDone;
    done.id = t->client_id;
    done.status = result.status;
    done.words = {result.bytes};
    try {
      w->session->Send(done);
    } catch (...) {
    }
    --counts.inflight;
    w->session->DoneTask(t->id);
  }
  void Remote(Fd fd) {
    {
      std::lock_guard<std::mutex> l(mu);
      if (stopping) return;
      accepted.insert(fd.get());
    }
    try {
      while (true) {
        Frame request =
            internal::ReceiveFrame(fd.get(), std::chrono::steady_clock::now() +
                                                 std::chrono::seconds(120));
        if (request.type != Type::kSubmit || request.words.size() != 3 ||
            !request.words[1] || !request.words[2] ||
            request.words[2] > INT32_MAX ||
            (request.words[0] != 1 && request.words[0] != 2))
          throw std::runtime_error("invalid remote job");
        auto task = std::make_shared<Task>();
        task->id = next.fetch_add(1);
        task->endpoint = request.key;
        task->direction = static_cast<Direction>(request.words[0]);
        task->deadline = std::chrono::steady_clock::now() +
                         std::chrono::milliseconds(request.words[2]);
        std::shared_ptr<ServerSession> s;
        {
          std::lock_guard<std::mutex> l(mu);
          auto it = endpoints.find(request.key);
          if (it != endpoints.end()) s = it->second.lock();
        }
        if (!s) {
          internal::SendFrame(
              fd.get(),
              internal::Reply(0, absl::NotFoundError("endpoint unavailable")),
              task->deadline);
          continue;
        }
        bool added = false, prepared = false, ready_sent = false;
        ServiceIoResult result;
        try {
          s->AddTask(task);
          added = true;
          task->Attach(fd.get());
          Frame prepare;
          prepare.type = Type::kPrepare;
          prepare.key = request.key;
          prepare.data = request.data;
          prepare.words = {task->id, request.words[0], request.words[1]};
          auto response = s->Call(std::move(prepare));
          internal::Throw(response.status);
          prepared = true;
          ++counts.prepared;
          task->spans = s->Acquire(response.spans, request.words[1]);
          if (task->Cancelled())
            internal::Throw(absl::CancelledError("receive cancelled"));
          internal::SendFrame(fd.get(), internal::Reply(0), task->deadline);
          ready_sent = true;
          ++counts.inflight;
          Transfer(fd.get(), task->spans, task->direction == Direction::kRead,
                   task->deadline);
          result.bytes = request.words[1];
        } catch (const std::exception& e) {
          result.status = internal::Error(e);
        }
        task->Detach();
        if (ready_sent) --counts.inflight;
        // Retire must only fence I/O. Finish may synchronously retire its last
        // Region.
        task->spans.clear();
        if (prepared) {
          --counts.prepared;
          Frame finish;
          finish.type = Type::kFinish;
          finish.words = {task->id, result.bytes};
          finish.status = result.status;
          try {
            auto response = s->Call(std::move(finish));
            if (result.status.ok()) result.status = response.status;
          } catch (const std::exception& e) {
            if (result.status.ok()) result.status = internal::Error(e);
          }
        }
        if (added) s->DoneTask(task->id);
        if (!result.status.ok() && ready_sent)
          throw std::runtime_error(result.status.ToString());
        internal::SendFrame(fd.get(), internal::Reply(0, result.status),
                            task->deadline);
      }
    } catch (const std::exception&) {
    }
    {
      std::lock_guard<std::mutex> l(mu);
      accepted.erase(fd.get());
    }
  }
  absl::Status Start() {
    if (absl::GetFlag(FLAGS_require_psp_tcp))
      return absl::FailedPreconditionError(
          "host transport does not support required PSP TCP");
    if (config.socket_name.empty() || config.listen_endpoint.empty() ||
        !config.streams || config.streams > 1024)
      return absl::InvalidArgumentError(
          "socket, listen endpoint and streams in [1,1024] required");
    try {
      local.reset(internal::ListenUnix(config.socket_name));
      tcp.reset(internal::ListenTcp(config.listen_endpoint));
      started = true;
      local_thread = std::thread([this] {
        while (true) {
          int f = accept4(local.get(), nullptr, nullptr, SOCK_CLOEXEC);
          if (f < 0) break;
          readers.Spawn([this, f] { Local(Fd(f)); });
        }
      });
      tcp_thread = std::thread([this] {
        while (true) {
          int f = accept4(tcp.get(), nullptr, nullptr, SOCK_CLOEXEC);
          if (f < 0) break;
          int one = 1;
          setsockopt(f, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
          readers.Spawn([this, f] { Remote(Fd(f)); });
        }
      });
      return absl::OkStatus();
    } catch (const std::exception& e) {
      return internal::Error(e);
    }
  }
  absl::Status Stop() {
    std::vector<std::shared_ptr<ServerSession>> all;
    {
      std::lock_guard<std::mutex> l(mu);
      if (stopping) return absl::OkStatus();
      stopping = true;
    }
    shutdown(local.get(), SHUT_RDWR);
    shutdown(tcp.get(), SHUT_RDWR);
    if (local_thread.joinable()) local_thread.join();
    if (tcp_thread.joinable()) tcp_thread.join();
    {
      std::lock_guard<std::mutex> l(mu);
      for (auto& [id, s] : sessions) all.push_back(s);
      for (int fd : accepted) shutdown(fd, SHUT_RDWR);
      for (int fd : raw_local) shutdown(fd, SHUT_RDWR);
    }
    for (auto& s : all) s->BeginClose(0, false);
    for (auto& [key, q] : hosts) q->Stop();
    readers.AwaitAllDone();
    hosts.clear();
    local.reset();
    tcp.reset();
    return absl::OkStatus();
  }
};
Service::Service(ServiceConfig c)
    : impl_(std::make_unique<Impl>(std::move(c))) {}
Service::~Service() { Stop(); }
absl::Status Service::Start() { return impl_->Start(); }
absl::Status Service::Stop() { return impl_->Stop(); }
}  // namespace tpu_raiden::host_leader
