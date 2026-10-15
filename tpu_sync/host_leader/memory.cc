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
#include <fcntl.h>
#include <linux/memfd.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <cerrno>
#include <cstring>
#include <stdexcept>

#include "tpu_sync/host_leader/internal.h"
namespace tpu_raiden::host_leader::internal {
Fd AllocateMemory(size_t bytes) {
  Fd fd(syscall(SYS_memfd_create, "raiden-host-transport",
                MFD_CLOEXEC | MFD_ALLOW_SEALING));
  if (fd.get() < 0 || ftruncate(fd.get(), bytes) ||
      fcntl(fd.get(), F_ADD_SEALS, F_SEAL_GROW | F_SEAL_SHRINK | F_SEAL_SEAL))
    throw std::runtime_error(std::string("memfd: ") + strerror(errno));
  return fd;
}
}  // namespace tpu_raiden::host_leader::internal
