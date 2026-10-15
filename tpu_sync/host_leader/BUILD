load("@rules_cc//cc:defs.bzl", "cc_binary", "cc_library", "cc_test")
package(default_visibility = ["//visibility:public"])
cc_library(
 name = "protocol",
 srcs = ["protocol.cc"],
 hdrs = ["protocol.h", "internal.h"],
 copts = ["-fexceptions"],
 deps = ["@com_google_absl//absl/status", "@com_google_absl//absl/strings"],
)
cc_library(
 name = "memory",
 srcs = ["memory.cc"],
 hdrs = ["memory.h"],
 copts = ["-fexceptions"],
 deps = [":protocol", "@com_google_absl//absl/status"],
)
cc_library(
 name = "session",
 srcs = ["session.cc"],
 hdrs = ["session.h"],
 copts = ["-fexceptions"],
 deps = [":memory", ":protocol", "//tpu_sync/transport/lib/socket:tcp_psp_helper",
 "@com_google_absl//absl/flags:flag", "@com_google_absl//absl/status", "@com_google_absl//absl/status:statusor"],
)
cc_library(
 name = "service",
 srcs = ["service.cc"],
 hdrs = ["service.h"],
 copts = ["-fexceptions"],
 deps = [":protocol", "//tpu_sync/common:detached_thread_group",
 "//tpu_sync/transport/lib/socket:util", "//tpu_sync/transport/lib/socket:tcp_psp_helper",
 "@com_google_absl//absl/flags:flag", "@com_google_absl//absl/status"],
)
cc_library(
 name = "allocator",
 srcs = ["allocator.cc"],
 hdrs = ["allocator.h"],
 copts = ["-fexceptions"],
 features = ["-use_header_modules"],
 deps = [":session", ":memory", "//tpu_sync/core:host_memory_allocator", "//tpu_sync/core:tpu_utils",
 "//tpu_sync/core:xla_pjrt_headers", "@com_google_absl//absl/status", "@com_google_absl//absl/strings"],
)
cc_library(
 name = "adapter",
 srcs = ["adapter.cc"],
 hdrs = ["adapter.h"],
 copts = ["-fexceptions"],
 features = ["-use_header_modules"],
 deps = [":allocator", ":session", ":memory", ":protocol", "//tpu_sync/core:numa_thread_pool",
 "//tpu_sync/transport:block_transport_delegate", "//tpu_sync/transport/lib:transport_adapter",
 "@com_google_absl//absl/flags:flag", "@com_google_absl//absl/log", "@com_google_absl//absl/status",
 "@com_google_absl//absl/status:statusor", "@com_google_absl//absl/strings", "@com_google_absl//absl/types:span"],
)
cc_binary(
 name = "daemon_main",
 srcs = ["daemon_main.cc"],
 copts = ["-fexceptions"],
 deps = [":service", ":session", "@com_google_absl//absl/status", "@com_google_absl//absl/status:statusor"],
)
cc_test(
 name = "generic_test",
 srcs = ["generic_test.cc", "test_util.h"],
 copts = ["-fexceptions"],
 data = [":daemon_main"],
 deps = [":session", ":service", ":protocol", "//tpu_sync/transport/lib/socket:util",
 "//tpu_sync/transport/lib/socket:tcp_psp_helper", "@com_google_absl//absl/flags:flag",
 "@com_google_absl//absl/status", "@com_google_absl//absl/status:statusor", "@com_google_googletest//:gtest_main"],
)
cc_test(
 name = "adapter_test",
 srcs = ["adapter_test.cc", "test_util.h"],
 copts = ["-fexceptions"],
 features = ["-use_header_modules"],
 data = [":daemon_main"],
 deps = [":adapter", ":allocator", ":session", ":protocol", "//tpu_sync/core:host_memory_allocator",
 "//tpu_sync/transport:block_transport_delegate", "//tpu_sync/transport/lib:transport_adapter",
 "@com_google_absl//absl/status", "@com_google_absl//absl/status:statusor",
 "@com_google_absl//absl/types:span", "@com_google_googletest//:gtest_main"],
)
