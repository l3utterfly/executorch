/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cassert>
#include <chrono>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include <executorch/examples/models/llama2/runner/runner.h>
#include <executorch/runtime/platform/log.h>
#include <executorch/runtime/platform/platform.h>
#include <executorch/runtime/platform/runtime.h>

#if defined(ET_USE_THREADPOOL)
#include <executorch/backends/xnnpack/threadpool/cpuinfo_utils.h>
#include <executorch/backends/xnnpack/threadpool/threadpool.h>
#endif

#include <fbjni/ByteBuffer.h>
#include <fbjni/fbjni.h>

#ifdef __ANDROID__
#include <android/log.h>

// For Android, write to logcat
void et_pal_emit_log_message(
    et_timestamp_t timestamp,
    et_pal_log_level_t level,
    const char* filename,
    const char* function,
    size_t line,
    const char* message,
    size_t length) {
  int android_log_level = ANDROID_LOG_UNKNOWN;
  if (level == 'D') {
    android_log_level = ANDROID_LOG_DEBUG;
  } else if (level == 'I') {
    android_log_level = ANDROID_LOG_INFO;
  } else if (level == 'E') {
    android_log_level = ANDROID_LOG_ERROR;
  } else if (level == 'F') {
    android_log_level = ANDROID_LOG_FATAL;
  }

  __android_log_print(android_log_level, "LLAMA", "%s", message);
}
#endif

using namespace torch::executor;

namespace executorch_jni {

class ExecuTorchLlamaCallbackJni
    : public facebook::jni::JavaClass<ExecuTorchLlamaCallbackJni> {
 public:
  constexpr static const char* kJavaDescriptor =
      "Lorg/pytorch/executorch/LlamaCallback;";

  void onResult(std::string result) const {
    static auto cls = ExecuTorchLlamaCallbackJni::javaClassStatic();
    static const auto method =
        cls->getMethod<void(facebook::jni::local_ref<jstring>)>("onResult");
    facebook::jni::local_ref<jstring> s = facebook::jni::make_jstring(result);
    method(self(), s);
  }

  void onStats(const Runner::Stats& result) const {
    static auto cls = ExecuTorchLlamaCallbackJni::javaClassStatic();
    static const auto method = cls->getMethod<void(jfloat)>("onStats");
    double eval_time =
        (double)(result.inference_end_ms - result.prompt_eval_end_ms);

    float tps = result.num_generated_tokens / eval_time *
        result.SCALING_FACTOR_UNITS_PER_SECOND;

    method(self(), tps);
  }
};

class ExecuTorchLlamaJni
    : public facebook::jni::HybridClass<ExecuTorchLlamaJni> {
 private:
  friend HybridBase;
  std::unique_ptr<Runner> runner_;

 public:
  constexpr static auto kJavaDescriptor =
      "Lorg/pytorch/executorch/LlamaModule;";

  static facebook::jni::local_ref<jhybriddata> initHybrid(
      facebook::jni::alias_ref<jclass>,
      facebook::jni::alias_ref<jstring> model_path,
      facebook::jni::alias_ref<jstring> tokenizer_path,
      jfloat temperature) {
    return makeCxxInstance(model_path, tokenizer_path, temperature);
  }

  ExecuTorchLlamaJni(
      facebook::jni::alias_ref<jstring> model_path,
      facebook::jni::alias_ref<jstring> tokenizer_path,
      jfloat temperature) {
#if defined(ET_USE_THREADPOOL)
    // Reserve 1 thread for the main thread.
    uint32_t num_performant_cores =
        torch::executorch::cpuinfo::get_num_performant_cores() - 1;
    if (num_performant_cores > 0) {
      ET_LOG(Info, "Resetting threadpool to %d threads", num_performant_cores);
      torch::executorch::threadpool::get_threadpool()->_unsafe_reset_threadpool(
          num_performant_cores);
    }
#endif

    runner_ = std::make_unique<Runner>(
        model_path->toStdString().c_str(),
        tokenizer_path->toStdString().c_str(),
        temperature);
  }

  jint generate(
      facebook::jni::alias_ref<jstring> prompt,
      facebook::jni::alias_ref<jstring> grammarStr,
      facebook::jni::alias_ref<ExecuTorchLlamaCallbackJni> callback) {
    runner_->generate(
        prompt->toStdString(),
        grammarStr->toStdString(),
        2048,
        [callback](std::string result) { callback->onResult(result); },
        [callback](const Runner::Stats& result) { callback->onStats(result); });
    return 0;
  }

  jint repl_start(
      // model settings
      facebook::jni::alias_ref<jstring> prompt,
      facebook::jni::alias_ref<jstring> antiPrompt,
      jint contextLength,

      // logs and storage
      facebook::jni::alias_ref<jstring> session_file,
      facebook::jni::alias_ref<jstring> prompt_cache_file,

      // callbacks
      facebook::jni::alias_ref<ExecuTorchLlamaCallbackJni> callback) {
    // buffer to handle utf8 string conversion byte-by-byte
    std::string buffer;

    runner_->start_repl(
        // model settings
        prompt->toStdString(),
        antiPrompt->toStdString(),
        contextLength,

        // logs and storage
        session_file->toStdString(),
        prompt_cache_file->toStdString(),

        // callbacks
        [buffer, &callback](std::string result) mutable {
          // Append the received bytes to the buffer.
          buffer.append(result);

          std::string response = "";

          while (!buffer.empty()) {
            size_t length;
            if ((buffer[0] & 0x80) == 0) {
              // Single-byte character (ASCII).
              length = 1;
            } else if ((buffer[0] & 0xE0) == 0xC0) {
              // Two-byte character.
              length = 2;
            } else if ((buffer[0] & 0xF0) == 0xE0) {
              // Three-byte character.
              length = 3;
            } else if ((buffer[0] & 0xF8) == 0xF0) {
              // Four-byte character.
              length = 4;
            } else {
              // Invalid byte, remove it from the buffer.
              buffer.erase(0, 1);
              continue;
            }

            if (buffer.size() < length) {
              // Buffer does not yet contain a complete character.
              break;
            }

            response += buffer.substr(0, length);

            // Remove the processed character from the buffer.
            buffer.erase(0, length);
          }

          if (response.size() > 0) {
            callback->onResult("REPL_MSG:" + response);
          }
        },
        [callback](std::string result) { callback->onResult(result); },
        [callback](const Runner::Stats& result) { callback->onStats(result); });
    return 0;
  }

  jint repl_enqueue_message(
      facebook::jni::alias_ref<jstring> msg,
      jint msgType,
      facebook::jni::alias_ref<jstring> grammar,
      facebook::jni::alias_ref<jstring> action) {
    runner_->repl_enqueue_message(
        msg->toStdString(),
        static_cast<Runner::MsgType>(msgType),
        grammar->toStdString(),
        action->toStdString());
    return 0;
  }

  facebook::jni::local_ref<facebook::jni::JString> infer(
      facebook::jni::alias_ref<jstring> prompt,
      facebook::jni::alias_ref<jstring> grammarStr) {
    Result<std::string> result =
        runner_->infer(prompt->toStdString(), grammarStr->toStdString(), 2048);

    if(!result.ok()) {
      return facebook::jni::make_jstring("Error code: " + std::to_string((int)result.error()));
    }
    
    return facebook::jni::make_jstring(result.get());
  }

  void stop() {
    runner_->stop();
  }

  jint load() {
    return static_cast<jint>(runner_->load());
  }

  static void registerNatives() {
    registerHybrid({
        makeNativeMethod("initHybrid", ExecuTorchLlamaJni::initHybrid),
        makeNativeMethod("generate", ExecuTorchLlamaJni::generate),
        makeNativeMethod("repl_start", ExecuTorchLlamaJni::repl_start),
        makeNativeMethod(
            "repl_enqueue_message", ExecuTorchLlamaJni::repl_enqueue_message),
        makeNativeMethod("infer", ExecuTorchLlamaJni::infer),
        makeNativeMethod("stop", ExecuTorchLlamaJni::stop),
        makeNativeMethod("load", ExecuTorchLlamaJni::load),
    });
  }
};

} // namespace executorch_jni

JNIEXPORT jint JNICALL JNI_OnLoad(JavaVM* vm, void*) {
  return facebook::jni::initialize(
      vm, [] { executorch_jni::ExecuTorchLlamaJni::registerNatives(); });
}
