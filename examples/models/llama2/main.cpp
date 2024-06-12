/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <iostream>  // Include the iostream header
#include <functional>
#include <string>
#include <gflags/gflags.h>

#include <executorch/examples/models/llama2/runner/runner.h>

#if defined(ET_USE_THREADPOOL)
#include <executorch/backends/xnnpack/threadpool/cpuinfo_utils.h>
#include <executorch/backends/xnnpack/threadpool/threadpool.h>
#endif

DEFINE_string(
    model_path,
    "llama2.pte",
    "Model serialized in flatbuffer format.");

DEFINE_string(tokenizer_path, "tokenizer.bin", "Tokenizer stuff.");

DEFINE_string(prompt, "The answer to the ultimate question is", "Prompt.");

DEFINE_double(
    temperature,
    0.0f,
    "Temperature; Default is 0.8f. 0 = greedy argmax sampling (deterministic). Lower temperature = more deterministic");

DEFINE_int32(
    seq_len,
    128,
    "Total number of tokens to generate (prompt + output). Defaults to max_seq_len. If the number of input tokens + seq_len > max_seq_len, the output will be truncated to max_seq_len tokens.");

DEFINE_int32(
    cpu_threads,
    -1,
    "Number of CPU threads for inference. Defaults to -1, which implies we'll use a heuristic to derive the # of performant cores for a specific device.");

int32_t main(int32_t argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  // Create a loader to get the data of the program file. There are other
  // DataLoaders that use mmap() or point32_t to data that's already in memory,
  // and users can create their own DataLoaders to load from arbitrary sources.
  const char* model_path = FLAGS_model_path.c_str();

  const char* tokenizer_path = FLAGS_tokenizer_path.c_str();

  const char* prompt = FLAGS_prompt.c_str();

  double temperature = FLAGS_temperature;

  int32_t seq_len = FLAGS_seq_len;

  int32_t cpu_threads = FLAGS_cpu_threads;

#if defined(ET_USE_THREADPOOL)
  uint32_t num_performant_cores = cpu_threads == -1
      ? torch::executorch::cpuinfo::get_num_performant_cores()
      : static_cast<uint32_t>(cpu_threads);
  ET_LOG(
      Info, "Resetting threadpool with num threads = %d", num_performant_cores);
  if (num_performant_cores > 0) {
    torch::executorch::threadpool::get_threadpool()->_unsafe_reset_threadpool(
        num_performant_cores);
  }
#endif
  std::vector<std::string> user_prompts = {
    "<|start_header_id|>user<|end_header_id|>\n\nWhat's the capital of France?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n", "REGEN"
  };
  // create llama runner
  ::torch::executor::Runner runner(model_path, tokenizer_path, temperature);

  // Define callbacks
  auto token_callback = [](const std::string& data) {
    printf("Token callback data: %s\n", data.c_str());
  };

  auto system_msg_callback = [&runner, &user_prompts](const std::string& data) {
    printf("System message callback data: %s\n", data.c_str());

    if(data == "REPL_READY:") {
      auto msg = user_prompts.front();

      if(msg == "REGEN") {
        // send a message
        runner.repl_enqueue_message("<|start_header_id|>assistant<|end_header_id|>\n\nThe capital of France is London.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nThat doesn't sound right...<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n", ::torch::executor::Runner::MsgType::USER, "", "REGEN");
      } else {
        // send a message
        runner.repl_enqueue_message(msg, ::torch::executor::Runner::MsgType::USER, "", "");
      }

      // Remove the first element
      user_prompts.erase(user_prompts.begin());
    }
  };

  auto stats_callback = [](const ::torch::executor::Runner::Stats& stats) {
    printf("Stats callback data: %ld\n", stats.model_load_start_ms);
  };

  std::string my_prompt = "<|start_header_id|>system<|end_header_id|>\n\nLayla is an AI Assistant created by Layla Network that is helpful, polite, and to the point. She is here to help the user with everyday tasks. Layla's favourite animal is the butterfly because it represents transformation, growth, and beauty.\n\nLayla and User are having a friendly conversation.\n\nYou ARE Layla. Embody the character and personality completely.<|eot_id|>";
  std::string antiPrompt = "<|eot_id|>";

  // start repl in a separate thread
  std::thread repl_thread(&::torch::executor::Runner::start_repl, &runner,
      my_prompt, antiPrompt, 8192,
      "session", "prompt_cache",
      token_callback, system_msg_callback, stats_callback);

  // Wait for the REPL thread to finish
  repl_thread.join();

  return 0;
}
