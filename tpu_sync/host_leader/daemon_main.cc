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
#include <pthread.h>
#include <signal.h>

#include <cstdlib>
#include <iostream>
#include <string>

#include "tpu_sync/host_leader/service.h"
#include "tpu_sync/host_leader/session.h"
namespace h = tpu_raiden::host_leader;
int main(int argc, char** argv) {
  h::ServiceConfig config;
  bool stats = false;
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg.rfind("--socket=", 0) == 0)
      config.socket_name = arg.substr(9);
    else if (arg.rfind("--listen=", 0) == 0)
      config.listen_endpoint = arg.substr(9);
    else if (arg.rfind("--streams=", 0) == 0) {
      try {
        config.streams = std::stoull(arg.substr(10));
      } catch (...) {
        std::cerr << "invalid streams\n";
        return 2;
      }
    } else if (arg == "--stats")
      stats = true;
    else {
      std::cerr << "unknown argument: " << arg << "\n";
      return 2;
    }
  }
  if (stats) {
    auto result = h::GetStats(config.socket_name);
    if (!result.ok()) {
      std::cerr << result.status() << "\n";
      return 1;
    }
    auto& s = *result;
    std::cout << "{\"sessions\":" << s.sessions
              << ",\"endpoints\":" << s.endpoints
              << ",\"registered_regions\":" << s.registered_regions
              << ",\"mapped_bytes\":" << s.mapped_bytes
              << ",\"queued\":" << s.queued << ",\"prepared\":" << s.prepared
              << ",\"inflight\":" << s.inflight
              << ",\"connections\":" << s.connections << "}\n";
    return 0;
  }
  sigset_t signals;
  sigemptyset(&signals);
  sigaddset(&signals, SIGINT);
  sigaddset(&signals, SIGTERM);
  pthread_sigmask(SIG_BLOCK, &signals, nullptr);
  h::Service service(config);
  auto status = service.Start();
  if (!status.ok()) {
    std::cerr << status << "\n";
    return 1;
  }
  std::cerr << "HOST_TRANSPORT_READY socket=" << config.socket_name
            << " listen=" << config.listen_endpoint
            << " streams=" << config.streams << "\n";
  int signal;
  sigwait(&signals, &signal);
  status = service.Stop();
  if (!status.ok()) {
    std::cerr << status << "\n";
    return 1;
  }
  return 0;
}
