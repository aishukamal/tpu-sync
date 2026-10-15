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
#ifndef TPU_SYNC_HOST_LEADER_TEST_UTIL_H_
#define TPU_SYNC_HOST_LEADER_TEST_UTIL_H_
#include <arpa/inet.h>
#include <fcntl.h>
#include <signal.h>
#include <spawn.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "tpu_sync/host_leader/session.h"

namespace tpu_raiden::host_leader::test_util {
using namespace std::chrono_literals;
inline void Require(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}
inline void Await(const std::function<bool()>& condition,
                  const std::string& message) {
  for (int i = 0; i < 2000; ++i) {
    if (condition()) return;
    std::this_thread::sleep_for(5ms);
  }
  throw std::runtime_error(message);
}
class Child {
 public:
  Child(const std::vector<std::string>& command, const std::string& log) {
    Fd output(
        open(log.c_str(), O_CREAT | O_APPEND | O_WRONLY | O_CLOEXEC, 0600));
    Require(output.get() >= 0, "cannot open child log");
    posix_spawn_file_actions_t actions;
    Require(posix_spawn_file_actions_init(&actions) == 0,
            "spawn actions failed");
    posix_spawn_file_actions_adddup2(&actions, output.get(), STDOUT_FILENO);
    posix_spawn_file_actions_adddup2(&actions, output.get(), STDERR_FILENO);
    std::vector<char*> argv;
    for (const auto& part : command)
      argv.push_back(const_cast<char*>(part.c_str()));
    argv.push_back(nullptr);
    const int error = posix_spawn(&pid_, command.front().c_str(), &actions,
                                  nullptr, argv.data(), ::environ);
    posix_spawn_file_actions_destroy(&actions);
    Require(error == 0, "posix_spawn failed");
  }
  explicit Child(std::function<void()> function) {
    pid_ = fork();
    Require(pid_ >= 0, "fork failed");
    if (pid_ == 0) {
      try {
        function();
        _exit(0);
      } catch (const std::exception& error) {
        dprintf(STDERR_FILENO, "host transport test child: %s\n", error.what());
        _exit(1);
      }
    }
  }
  ~Child() { Stop(); }
  pid_t pid() const { return pid_; }
  int Wait() {
    int status = 0;
    for (int i = 0; i < 4000; ++i) {
      if (waitpid(pid_, &status, WNOHANG) == pid_) {
        pid_ = -1;
        return WIFEXITED(status) ? WEXITSTATUS(status) : 128 + WTERMSIG(status);
      }
      std::this_thread::sleep_for(5ms);
    }
    Stop();
    return 124;
  }
  void Stop() {
    if (pid_ > 0) {
      kill(pid_, SIGKILL);
      waitpid(pid_, nullptr, 0);
      pid_ = -1;
    }
  }

 private:
  pid_t pid_ = -1;
};
struct TempDir {
  TempDir() {
    char pattern[] = "/tmp/raiden_ht_test_XXXXXX";
    char* result = mkdtemp(pattern);
    Require(result != nullptr, "mkdtemp failed");
    path = result;
  }
  ~TempDir() { std::filesystem::remove_all(path); }
  std::string path;
};
inline int FreePort() {
  Fd listener(socket(AF_INET, SOCK_STREAM | SOCK_CLOEXEC, 0));
  Require(listener.get() >= 0, "socket failed");
  sockaddr_in address{};
  address.sin_family = AF_INET;
  address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  Require(bind(listener.get(), reinterpret_cast<sockaddr*>(&address),
               sizeof(address)) == 0,
          "bind failed");
  socklen_t bytes = sizeof(address);
  Require(getsockname(listener.get(), reinterpret_cast<sockaddr*>(&address),
                      &bytes) == 0,
          "getsockname failed");
  return ntohs(address.sin_port);
}
class Daemon {
 public:
  explicit Daemon(int streams = 1) : streams_(streams) {
    static std::atomic<unsigned> next{0};
    socket =
        "@raiden_ht_" + std::to_string(getpid()) + "_" + std::to_string(next++);
    endpoint = "127.0.0.1:" + std::to_string(FreePort());
    Restart();
  }
  void Restart() {
    if (child) child->Stop();
    const std::string executable =
        (std::filesystem::canonical("/proc/self/exe").parent_path() /
         "daemon_main")
            .string();
    const std::string log = temp.path + "/daemon.log";
    child = std::make_unique<Child>(
        std::vector<std::string>{
            executable, "--socket=" + socket,
            "--listen=0.0.0.0:" + endpoint.substr(endpoint.rfind(':') + 1),
            "--streams=" + std::to_string(streams_)},
        log);
    Await([&] { return GetStats(socket).ok(); },
          "daemon failed to start: " + log);
  }
  Stats stats() const {
    auto result = GetStats(socket);
    Require(result.ok(), result.status().ToString());
    return *result;
  }
  pid_t pid() const { return child->pid(); }
  void Kill() { child->Stop(); }
  Peer peer(const std::string& host = "test-host") const {
    return {host, {{"127.0.0.1", endpoint}}};
  }
  std::string socket, endpoint;
  TempDir temp;
  std::unique_ptr<Child> child;

 private:
  int streams_;
};
inline size_t MemfdFds(pid_t pid) {
  size_t count = 0;
  for (const auto& entry : std::filesystem::directory_iterator(
           "/proc/" + std::to_string(pid) + "/fd")) {
    std::error_code error;
    const auto target =
        std::filesystem::read_symlink(entry.path(), error).string();
    if (!error && target.find("memfd:") != std::string::npos) ++count;
  }
  return count;
}
inline size_t MemfdMappings(pid_t pid) {
  std::ifstream maps("/proc/" + std::to_string(pid) + "/maps");
  size_t count = 0;
  std::string line;
  while (std::getline(maps, line))
    if (line.find("memfd:") != std::string::npos) ++count;
  return count;
}
}  // namespace tpu_raiden::host_leader::test_util
#endif
