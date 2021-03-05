/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/core/tpu/tpu_model_server_initializer.h"
#include "tensorflow/core/tpu/tpu_api_dlsym_initializer.h"

#include <dlfcn.h>

#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/tpu/tpu_api_dlsym_set_fn.h"
#include "tensorflow/core/tpu/tpu_api.h"
#include "tensorflow/core/tpu/tpu_initializer_helper.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/stream_executor/tpu/tpu_platform.h"

namespace tensorflow {
namespace tpu {
namespace {
#if !defined(PLATFORM_GOOGLE)
#include "tensorflow/core/tpu/tpu_library_init_fns.inc"
Status InitializeTpuLib(void* library_handle) {
  LOG(INFO) << "hello there";
  Status s = InitializeTpuStructFns(library_handle);

  // Retrieve arguments from environment if applicable
  std::pair<std::vector<std::string>, std::vector<const char*> > args =
      GetLibTpuInitArguments();

  // TPU platform registration must only be performed after the library is
  // loaded. We do not want to register a TPU platform in XLA without the
  // supporting library providing the necessary APIs.
  if (s.ok()) {
    void (*initialize_fn)(bool init_library, int num_args, const char** args);
    initialize_fn = reinterpret_cast<decltype(initialize_fn)>(
        dlsym(library_handle, "TfTpu_Initialize"));
    (*initialize_fn)(/*init_library=*/true, args.second.size(),
                     args.second.data());

    RegisterTpuPlatform();
  }

  return s;
}



bool FindAndLoadTpuModelServer() {
  void* library = dlopen("libtpu.so", RTLD_NOW);
  if (library) {
    if (TryAcquireTpuLock()) {
      InitializeTpuLib(library);
    }
  }
  OpsApiFn()->TfTpu_InitializeTpuModelServerFn();
  return true;
}

static bool tpu_library_finder = FindAndLoadTpuModelServer();
#endif  // PLATFORM_GOOGLE
}  // namespace
}  // namespace tpu
}  // namespace tensorflow
