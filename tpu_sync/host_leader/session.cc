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
#include "tpu_sync/host_leader/session.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <unistd.h>

#include <atomic>
#include <cerrno>
#include <condition_variable>
#include <future>
#include <iostream>
#include <map>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <utility>

#include "absl/flags/flag.h"
#include "tpu_sync/host_leader/internal.h"
#include "tpu_sync/transport/lib/socket/tcp_psp_helper.h"
namespace tpu_raiden::host_leader {
namespace {
using internal::Frame;
using internal::Type;
struct JobState {
  std::mutex mu;
  std::condition_variable cv;
  bool done = false;
  IoResult result;
  void Complete(IoResult r) {
    std::lock_guard<std::mutex> l(mu);
    if (!done) {
      done = true;
      result = std::move(r);
      cv.notify_all();
    }
  }
  IoResult Wait() {
    std::unique_lock<std::mutex> l(mu);
    cv.wait(l, [&] { return done; });
    return result;
  }
};
struct ClientState;
thread_local ClientState* current_callback = nullptr;
struct CallbackScope {
  explicit CallbackScope(ClientState* state) : previous(current_callback) {
    current_callback = state;
  }
  ~CallbackScope() { current_callback = previous; }
  ClientState* previous;
};
struct ClientState : std::enable_shared_from_this<ClientState> {
  Fd fd;
  std::mutex write_mu, mu;
  std::atomic<uint64_t> next{1};
  std::map<uint64_t, std::shared_ptr<std::promise<Frame>>> pending;
  std::map<uint64_t, std::shared_ptr<JobState>> jobs;
  std::map<std::string, std::weak_ptr<EndpointHandler>> handlers;
  struct Pin {
    uint64_t cookie;
    std::weak_ptr<EndpointHandler> handler;
    std::vector<Span> spans;
  };
  std::map<uint64_t, Pin> pins;
  std::function<bool(const std::shared_ptr<Region>&)> valid_region;
  internal::Executor callbacks{4};
  std::thread reader;
  std::mutex close_mu;
  bool closed = false;
  explicit ClientState(Fd f) : fd(std::move(f)) {}
  void Send(const Frame& f) {
    std::lock_guard<std::mutex> l(write_mu);
    internal::SendPacket(fd.get(), f);
  }
  Frame Call(Frame f) {
    auto p = std::make_shared<std::promise<Frame>>();
    auto future = p->get_future();
    f.id = next.fetch_add(1);
    {
      std::lock_guard<std::mutex> l(mu);
      if (closed) internal::Throw(absl::UnavailableError("session closed"));
      pending.emplace(f.id, p);
    }
    try {
      Send(f);
    } catch (...) {
      shutdown(fd.get(), SHUT_WR);
    }
    return future.get();
  }
  void Start() {
    reader = std::thread([this] { ReadLoop(); });
  }
  void ReadLoop() {
    absl::Status failure =
        absl::UnavailableError("daemon session closed after drain");
    try {
      while (true) {
        Frame f = internal::ReceivePacket(fd.get());
        if (f.type == Type::kReply) {
          std::shared_ptr<std::promise<Frame>> p;
          {
            std::lock_guard<std::mutex> l(mu);
            auto it = pending.find(f.id);
            if (it != pending.end()) {
              p = it->second;
              pending.erase(it);
            }
          }
          if (p) p->set_value(std::move(f));
        } else if (f.type == Type::kDone) {
          std::shared_ptr<JobState> s;
          {
            std::lock_guard<std::mutex> l(mu);
            auto it = jobs.find(f.id);
            if (it != jobs.end()) {
              s = it->second;
              jobs.erase(it);
            }
          }
          if (s) s->Complete({f.status, f.words.empty() ? 0 : f.words[0]});
        } else if (f.type == Type::kPrepare) {
          auto request = std::make_shared<Frame>(std::move(f));
          callbacks.Post([this, request] {
            CallbackScope callback_scope(this);
            Frame reply = internal::Reply(request->id);
            std::shared_ptr<EndpointHandler> h;
            uint64_t cookie = 0;
            bool owns_pin = false;
            try {
              {
                std::lock_guard<std::mutex> l(mu);
                auto it = handlers.find(request->key);
                if (it != handlers.end()) h = it->second.lock();
              }
              if (!h)
                internal::Throw(absl::NotFoundError("endpoint unavailable"));
              if (request->words.size() != 3)
                internal::Throw(absl::InvalidArgumentError("bad Prepare"));
              auto prepared = h->Prepare(
                  {request->words[0], static_cast<Direction>(request->words[1]),
                   request->words[2], request->data});
              if (!prepared.ok()) internal::Throw(prepared.status());
              cookie = prepared->cookie;
              owns_pin = true;
              uint64_t total = 0;
              for (const auto& s : prepared->spans) {
                if (!s.region || !valid_region(s.region) || !s.region->id() ||
                    !s.bytes || s.offset > s.region->size() ||
                    s.bytes > s.region->size() - s.offset ||
                    UINT64_MAX - total < s.bytes)
                  internal::Throw(
                      absl::InvalidArgumentError("invalid prepared span"));
                reply.spans.push_back({s.region->id(), s.offset, s.bytes});
                total += s.bytes;
              }
              if (total != request->words[2])
                internal::Throw(
                    absl::InvalidArgumentError("prepared byte count differs"));
              {
                std::lock_guard<std::mutex> l(mu);
                if (closed)
                  internal::Throw(
                      absl::UnavailableError("daemon drained during Prepare"));
                pins.emplace(request->words[0],
                             Pin{cookie, h, std::move(prepared->spans)});
                owns_pin = false;
              }
            } catch (const std::exception& e) {
              reply.status = internal::Error(e);
            }
            if (owns_pin) {
              try {
                h->Finish(cookie, {reply.status, 0});
              } catch (const std::exception& e) {
                reply.status = internal::Error(e);
              }
            }
            try {
              Send(reply);
            } catch (...) {
              shutdown(fd.get(), SHUT_WR);
            }
          });
        } else if (f.type == Type::kFinish) {
          auto request = std::make_shared<Frame>(std::move(f));
          callbacks.Post([this, request] {
            CallbackScope callback_scope(this);
            Frame reply = internal::Reply(request->id);
            std::shared_ptr<EndpointHandler> h;
            uint64_t cookie = 0;
            Pin pin;
            try {
              if (request->words.size() != 2)
                internal::Throw(absl::InvalidArgumentError("bad Finish"));
              {
                std::lock_guard<std::mutex> l(mu);
                auto it = pins.find(request->words[0]);
                if (it != pins.end()) {
                  pin = std::move(it->second);
                  cookie = pin.cookie;
                  h = pin.handler.lock();
                  pins.erase(it);
                }
              }
              pin.spans.clear();
              if (h)
                reply.status = h->Finish(
                    cookie,
                    {request->status,
                     request->words.size() > 1 ? request->words[1] : 0});
            } catch (const std::exception& e) {
              reply.status = internal::Error(e);
            }
            try {
              Send(reply);
            } catch (...) {
              shutdown(fd.get(), SHUT_WR);
            }
          });
        } else
          internal::Throw(
              absl::InvalidArgumentError("unexpected session packet"));
      }
    } catch (const std::exception& e) {
      failure = internal::Error(e);
    }
    // A local framing/send error is not a payload drain fence. Half-close the
    // request stream, then wait for the server's close-after-drain EOF.
    shutdown(fd.get(), SHUT_WR);
    char discard[4096];
    while (true) {
      ssize_t n = recv(fd.get(), discard, sizeof(discard), 0);
      if (n > 0) continue;
      if (n < 0 && errno == EINTR) continue;
      break;
    }
    std::map<uint64_t, std::shared_ptr<std::promise<Frame>>> ps;
    std::map<uint64_t, std::shared_ptr<JobState>> js;
    std::map<uint64_t, Pin> left;
    {
      std::lock_guard<std::mutex> l(mu);
      closed = true;
      handlers.clear();
      ps.swap(pending);
      js.swap(jobs);
      left.swap(pins);
    }
    for (auto& [id, p] : ps) p->set_value(internal::Reply(id, failure));
    for (auto& [id, j] : js) j->Complete({failure, 0});
    for (auto& [id, p] : left)
      callbacks.Post([this, p = std::move(p), failure]() mutable {
        CallbackScope callback_scope(this);
        p.spans.clear();
        if (auto h = p.handler.lock()) {
          try {
            h->Finish(p.cookie, {failure, 0});
          } catch (const std::exception& e) {
            std::cerr << "host transport EOF Finish: " << e.what() << "\n";
          }
        }
      });
  }
  absl::Status Close() {
    std::lock_guard<std::mutex> closing(close_mu);
    bool open;
    {
      std::lock_guard<std::mutex> l(mu);
      open = !closed;
    }
    absl::Status result;
    if (open) {
      try {
        Frame f;
        f.type = Type::kClose;
        result = Call(std::move(f)).status;
      } catch (const std::exception& e) {
        result = internal::Error(e);
      }
    }
    // The server ACKs Close only after all payload writers and Finish
    // callbacks.
    if (reader.joinable()) reader.join();
    callbacks.Drain();
    fd.reset();
    return result;
  }
};
}  // namespace
class Session::Impl {
 public:
  explicit Impl(std::shared_ptr<ClientState> s) : state(std::move(s)) {}
  std::shared_ptr<ClientState> state;
};
class Region::Impl {
 public:
  std::shared_ptr<Session> session;
  Fd fd;
  void* data = MAP_FAILED;
  size_t bytes = 0;
  std::atomic<uint64_t> id{0};
  std::mutex mu;
};
class Job::Impl {
 public:
  std::shared_ptr<Session> session;
  std::shared_ptr<JobState> state;
  std::vector<Span> spans;
  uint64_t id = 0;
};
class Endpoint::Impl {
 public:
  std::shared_ptr<Session> session;
  std::string key;
  bool closed = false;
};
Session::Session(std::unique_ptr<Impl> p) : impl_(std::move(p)) {}
Session::~Session() {
  // Destruction may originate inside Finish releasing its last Region. A
  // dedicated closer owns the state, so no callback/reader joins itself.
  auto state = std::move(impl_->state);
  std::thread([state = std::move(state)] { state->Close(); }).detach();
}
absl::StatusOr<std::shared_ptr<Session>> Session::Connect(
    const std::string& name) {
  if (absl::GetFlag(FLAGS_require_psp_tcp))
    return absl::FailedPreconditionError(
        "host transport does not support required PSP TCP");
  try {
    auto state = std::make_shared<ClientState>(Fd(internal::ConnectUnix(name)));
    state->Start();
    auto session =
        std::shared_ptr<Session>(new Session(std::make_unique<Impl>(state)));
    state->valid_region =
        [identity = session.get()](const std::shared_ptr<Region>& region) {
          return region->impl_->session.get() == identity;
        };
    Frame f;
    f.type = Type::kHello;
    auto reply = state->Call(std::move(f));
    if (!reply.status.ok()) {
      session->Close();
      return reply.status;
    }
    return session;
  } catch (const std::exception& e) {
    return internal::Error(e);
  }
}
absl::StatusOr<std::shared_ptr<Session>> Session::ForProcess(
    const std::string& name) {
  static auto* mu = new std::mutex;
  static auto* cache = new std::map<std::string, std::weak_ptr<Session>>;
  std::lock_guard<std::mutex> l(*mu);
  if (auto s = (*cache)[name].lock()) return s;
  auto s = Connect(name);
  if (s.ok()) (*cache)[name] = *s;
  return s;
}
absl::Status Session::Close() {
  if (current_callback == impl_->state.get())
    return absl::FailedPreconditionError(
        "Session::Close requires an external owner, not its own callback");
  return impl_->state->Close();
}
absl::StatusOr<std::shared_ptr<Region>> Session::AllocateRegion(size_t bytes) {
  if (!bytes) return absl::InvalidArgumentError("region must be nonempty");
  try {
    return RegisterRegion(internal::AllocateMemory(bytes), bytes);
  } catch (const std::exception& e) {
    return internal::Error(e);
  }
}
absl::StatusOr<std::shared_ptr<Region>> Session::RegisterRegion(Fd fd,
                                                                size_t bytes) {
  if (!bytes || fd.get() < 0)
    return absl::InvalidArgumentError("invalid region fd/size");
  try {
    int seals = fcntl(fd.get(), F_GET_SEALS);
    if (seals < 0 || (seals & (F_SEAL_GROW | F_SEAL_SHRINK)) !=
                         (F_SEAL_GROW | F_SEAL_SHRINK))
      return absl::InvalidArgumentError("region requires resize-sealed memfd");
    struct stat st;
    if (fstat(fd.get(), &st) || st.st_size < 0 ||
        static_cast<uint64_t>(st.st_size) != bytes)
      return absl::InvalidArgumentError("region fd size mismatch");
    auto p = std::make_unique<Region::Impl>();
    p->session = shared_from_this();
    p->bytes = bytes;
    p->fd = std::move(fd);
    p->data = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_SHARED,
                   p->fd.get(), 0);
    if (p->data == MAP_FAILED)
      return absl::UnavailableError("mmap worker region failed");
    auto region = std::shared_ptr<Region>(new Region(std::move(p)));
    Frame f;
    f.type = Type::kRegisterRegion;
    f.words = {bytes};
    f.fd = Fd(fcntl(region->impl_->fd.get(), F_DUPFD_CLOEXEC, 0));
    if (f.fd.get() < 0)
      return absl::UnavailableError("duplicate registered memfd failed");
    auto reply = impl_->state->Call(std::move(f));
    if (!reply.status.ok()) return reply.status;
    region->impl_->id = reply.words.at(0);
    return region;
  } catch (const std::exception& e) {
    return internal::Error(e);
  }
}
Region::Region(std::unique_ptr<Impl> p) : impl_(std::move(p)) {}
Region::~Region() { Retire(); }
uint8_t* Region::data() const { return static_cast<uint8_t*>(impl_->data); }
size_t Region::size() const { return impl_->bytes; }
uint64_t Region::id() const { return impl_->id; }
absl::Status Region::Retire() {
  std::lock_guard<std::mutex> l(impl_->mu);
  absl::Status status;
  if (impl_->id) {
    try {
      Frame f;
      f.type = Type::kRetireRegion;
      f.words = {impl_->id};
      status = impl_->session->impl_->state->Call(std::move(f)).status;
    } catch (const std::exception& e) {
      status = internal::Error(e);
    }
    impl_->id = 0;
  }
  if (impl_->data != MAP_FAILED) {
    munmap(impl_->data, impl_->bytes);
    impl_->data = MAP_FAILED;
  }
  impl_->fd.reset();
  return status;
}
absl::StatusOr<std::unique_ptr<Endpoint>> Session::RegisterEndpoint(
    const std::string& key, std::weak_ptr<EndpointHandler> h) {
  if (key.empty() || h.expired())
    return absl::InvalidArgumentError("endpoint key/handler required");
  try {
    {
      std::lock_guard<std::mutex> l(impl_->state->mu);
      if (impl_->state->closed) return absl::UnavailableError("session closed");
      if (impl_->state->handlers.count(key))
        return absl::AlreadyExistsError("endpoint already registered");
      impl_->state->handlers[key] = h;
    }
    Frame f;
    f.type = Type::kRegisterEndpoint;
    f.key = key;
    auto reply = impl_->state->Call(std::move(f));
    if (!reply.status.ok()) {
      std::lock_guard<std::mutex> l(impl_->state->mu);
      impl_->state->handlers.erase(key);
      return reply.status;
    }
    auto p = std::make_unique<Endpoint::Impl>();
    p->session = shared_from_this();
    p->key = key;
    return std::unique_ptr<Endpoint>(new Endpoint(std::move(p)));
  } catch (const std::exception& e) {
    return internal::Error(e);
  }
}
Endpoint::Endpoint(std::unique_ptr<Impl> p) : impl_(std::move(p)) {}
Endpoint::~Endpoint() {
  if (!impl_->closed &&
      current_callback == impl_->session->impl_->state.get()) {
    auto impl = std::move(impl_);
    std::thread([impl = std::move(impl)]() mutable {
      Endpoint endpoint(std::move(impl));
      endpoint.Close().IgnoreError();
    }).detach();
  } else {
    Close().IgnoreError();
  }
}
absl::Status Endpoint::Close() {
  if (current_callback == impl_->session->impl_->state.get())
    return absl::FailedPreconditionError(
        "Endpoint::Close requires an external owner, not its own callback");
  if (impl_->closed) return absl::OkStatus();
  impl_->closed = true;
  absl::Status result;
  try {
    Frame f;
    f.type = Type::kCloseEndpoint;
    f.key = impl_->key;
    result = impl_->session->impl_->state->Call(std::move(f)).status;
  } catch (const std::exception& e) {
    result = internal::Error(e);
  }
  {
    std::lock_guard<std::mutex> l(impl_->session->impl_->state->mu);
    impl_->session->impl_->state->handlers.erase(impl_->key);
  }
  return result;
}
absl::StatusOr<std::unique_ptr<Job>> Session::Submit(JobRequest r) {
  if (r.direction != Direction::kRead && r.direction != Direction::kWrite)
    return absl::InvalidArgumentError("unknown direction");
  if (r.peer.host_key.empty() || r.peer.routes.empty() || r.endpoint.empty() ||
      r.timeout.count() <= 0 || r.timeout.count() > INT32_MAX)
    return absl::InvalidArgumentError("job peer/endpoint/timeout required");
  for (const auto& route : r.peer.routes)
    if (route.local_ip.empty() || route.remote_endpoint.empty())
      return absl::InvalidArgumentError(
          "job route requires source and remote endpoint");
  try {
    Frame f;
    f.type = Type::kSubmit;
    f.key = std::move(r.endpoint);
    f.data = std::move(r.metadata);
    f.peer = std::move(r.peer);
    uint64_t bytes = 0;
    for (const auto& s : r.spans) {
      if (!s.region || s.region->impl_->session.get() != this ||
          !s.region->id() || !s.bytes || s.offset > s.region->size() ||
          s.bytes > s.region->size() - s.offset || UINT64_MAX - bytes < s.bytes)
        return absl::InvalidArgumentError("invalid job span");
      f.spans.push_back({s.region->id(), s.offset, s.bytes});
      bytes += s.bytes;
    }
    if (!bytes) return absl::InvalidArgumentError("job must transfer bytes");
    uint64_t id = impl_->state->next.fetch_add(1);
    f.words = {id, static_cast<uint64_t>(r.direction), bytes,
               static_cast<uint64_t>(r.timeout.count())};
    auto state = std::make_shared<JobState>();
    {
      std::lock_guard<std::mutex> l(impl_->state->mu);
      if (impl_->state->closed) return absl::UnavailableError("session closed");
      impl_->state->jobs[id] = state;
    }
    auto reply = impl_->state->Call(std::move(f));
    if (!reply.status.ok()) {
      std::lock_guard<std::mutex> l(impl_->state->mu);
      impl_->state->jobs.erase(id);
      return reply.status;
    }
    auto p = std::make_unique<Job::Impl>();
    p->session = shared_from_this();
    p->state = state;
    p->spans = std::move(r.spans);
    p->id = id;
    return std::unique_ptr<Job>(new Job(std::move(p)));
  } catch (const std::exception& e) {
    return internal::Error(e);
  }
}
Job::Job(std::unique_ptr<Impl> p) : impl_(std::move(p)) {}
Job::~Job() {
  Cancel();
  Wait();
}
uint64_t Job::id() const { return impl_->id; }
absl::Status Job::Cancel() {
  {
    std::lock_guard<std::mutex> l(impl_->state->mu);
    if (impl_->state->done) return absl::OkStatus();
  }
  try {
    Frame f;
    f.type = Type::kCancel;
    f.words = {id()};
    return impl_->session->impl_->state->Call(std::move(f)).status;
  } catch (const std::exception& e) {
    return internal::Error(e);
  }
}
IoResult Job::Wait() {
  auto result = impl_->state->Wait();
  impl_->spans.clear();
  return result;
}
absl::StatusOr<Stats> GetStats(const std::string& name) {
  try {
    Fd fd(internal::ConnectUnix(name));
    Frame f;
    f.type = Type::kStats;
    internal::SendPacket(fd.get(), f);
    auto reply = internal::ReceivePacket(fd.get());
    if (!reply.status.ok()) return reply.status;
    return internal::DecodeStats(reply.words);
  } catch (const std::exception& e) {
    return internal::Error(e);
  }
}
}  // namespace tpu_raiden::host_leader
