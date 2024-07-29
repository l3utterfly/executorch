/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// A simple llama2 runner that includes preprocessing and post processing logic.
// The module takes in a string as input and emits a string as output.

#include <executorch/examples/models/llama2/runner/runner.h>
#if ET_USE_TIKTOKEN
#include <executorch/examples/models/llama2/tokenizer/llama_tiktoken.h>
#else /* BPE */
#include <executorch/extension/llm/tokenizer/bpe_tokenizer.h>
#endif /* ET_USE_TIKTOKEN*/
#include <executorch/extension/evalue_util/print_evalue.h>
#include <executorch/extension/runner_util/managed_tensor.h>

#include <algorithm> // For std::max
#include <cstdint>
#include <ctime>
#include <filesystem> // C++17
#include <fstream>
#include <iostream>
#include <map> // Include this header for std::map
#include <memory>
#include <sstream>
#include <vector> // For std::vector

#ifdef USE_ATEN_LIB
#include <torch/torch.h>
#endif

#include <executorch/examples/models/llama2/runner/util.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/core/exec_aten/util/scalar_type_util.h>
#include <executorch/runtime/platform/log.h>

namespace torch::executor {
namespace {
static constexpr auto kTopp = 0.9f;
void printReport(const Runner::Stats& stats);
std::string statsToJsonString(const Runner::Stats& stats);
} // namespace

bool fileExists(const std::string& filename) {
  return std::filesystem::exists(filename);
}

void loadKvCacheBuffers(
    std::map<int, std::vector<uint8_t>>& kv_cache_buffers,
    std::vector<uint8_t>& full_buffer,
    std::vector<uint64_t>& session_tokens,
    const std::string& filename) {
  if (!fileExists(filename)) {
    // do nothing if the file does not exist
    return;
  }

  std::ifstream ifs(filename, std::ios::binary);
  if (!ifs) {
    throw std::runtime_error("Error opening file for reading: " + filename);
  }

  kv_cache_buffers.clear();
  session_tokens.clear();

  // Load session tokens
  size_t session_token_count;
  ifs.read(
      reinterpret_cast<char*>(&session_token_count),
      sizeof(session_token_count));
  session_tokens.resize(session_token_count);
  ifs.read(
      reinterpret_cast<char*>(session_tokens.data()),
      session_token_count * sizeof(uint64_t));

  // Load full_buffer
  size_t full_buffer_size;
  ifs.read(
      reinterpret_cast<char*>(&full_buffer_size), sizeof(full_buffer_size));
  full_buffer.resize(full_buffer_size);
  ifs.read(reinterpret_cast<char*>(full_buffer.data()), full_buffer_size);

  // Load kv_cache_buffers
  while (ifs) {
    int key;
    size_t buffer_size;

    ifs.read(reinterpret_cast<char*>(&key), sizeof(key));
    if (!ifs)
      break; // Exit if we've reached the end of file

    ifs.read(reinterpret_cast<char*>(&buffer_size), sizeof(buffer_size));
    if (!ifs)
      break; // Exit if we've reached the end of file

    std::vector<uint8_t> buffer(buffer_size);
    ifs.read(reinterpret_cast<char*>(buffer.data()), buffer_size);
    if (!ifs)
      break; // Exit if we've reached the end of file

    kv_cache_buffers[key] = buffer;
  }

  ifs.close();
}

void saveKvCacheBuffers(
    const std::map<int, std::vector<uint8_t>>& kv_cache_buffers,
    const std::vector<uint8_t>& full_buffer,
    const std::vector<uint64_t>& session_tokens,
    const std::string& filename) {
  std::ofstream ofs(
      filename,
      std::ios::binary | std::ios::trunc); // Open with trunc to overwrite
  if (!ofs) {
    throw std::runtime_error("Error opening file for writing: " + filename);
  }

  // Save session tokens
  size_t session_token_count = session_tokens.size();
  ofs.write(
      reinterpret_cast<const char*>(&session_token_count),
      sizeof(session_token_count));
  ofs.write(
      reinterpret_cast<const char*>(session_tokens.data()),
      session_token_count * sizeof(uint64_t));

  // Save full_buffer
  size_t full_buffer_size = full_buffer.size();
  ofs.write(
      reinterpret_cast<const char*>(&full_buffer_size),
      sizeof(full_buffer_size));
  ofs.write(
      reinterpret_cast<const char*>(full_buffer.data()), full_buffer_size);

  // Save kv_cache_buffers
  for (const auto& pair : kv_cache_buffers) {
    int key = pair.first;
    const std::vector<uint8_t>& buffer = pair.second;
    size_t buffer_size = buffer.size();

    ofs.write(reinterpret_cast<const char*>(&key), sizeof(key));
    ofs.write(reinterpret_cast<const char*>(&buffer_size), sizeof(buffer_size));
    ofs.write(reinterpret_cast<const char*>(buffer.data()), buffer_size);
  }

  ofs.close();
}

Runner::Runner(
    const std::string& model_path,
    const std::string& tokenizer_path,
    const float temperature)
    // NOTE: we observed ~2x loading performance increase on iPhone 15
    // and a ~5% improvement on Galaxy S22 by switching to
    // FileDataLoader instead of MmapDataLoader + UseMlockIgnoreErrors.
    : module_(std::make_unique<Module>(model_path, Module::LoadMode::File)),
      tokenizer_path_(tokenizer_path),
      temperature_(temperature) {
  ET_LOG(
      Info,
      "Creating LLaMa runner: model_path=%s, tokenizer_path=%s",
      model_path.c_str(),
      tokenizer_path.c_str());
}

bool Runner::is_loaded() const {
  return module_->is_loaded() && tokenizer_ && sampler_;
}

Error Runner::load() {
  if (is_loaded()) {
    return Error::Ok;
  }
  ET_CHECK_OK_OR_RETURN_ERROR(module_->load_method("forward"));

  // Read out metadata: vocab_size (expected by the model), BOS, EOS, n_BOS,
  // n_EOS max_seq_len from the model
  ET_LOG(Info, "Reading metadata from model");
  const auto method_names = module_->method_names();
  ET_CHECK_MSG(method_names.ok(), "Failed to read method names from model");
  model_methods_ = method_names.get();
  n_bos_ = getMetadataHelper<int64_t>("get_n_bos", 1);
  n_eos_ = getMetadataHelper<int64_t>("get_n_eos", 1);
  max_seq_len_ = getMetadataHelper<int64_t>("get_max_seq_len", 128);
  use_kv_cache_ = getMetadataHelper("use_kv_cache", true);
  use_sdpa_with_kv_cache_ = getMetadataHelper("use_sdpa_with_kv_cache", false);
  append_eos_ = getMetadataHelper("append_eos_to_prompt", false);
  enable_parallel_prefill_ = getMetadataHelper("enable_dynamic_shape", false);

  // Load tokenizer
#if ET_USE_TIKTOKEN
  tokenizer_ = get_tiktoken_for_llama();
#else
  tokenizer_ = std::make_unique<BPETokenizer>();
#endif
  tokenizer_->load(tokenizer_path_);

  vocab_size_ =
      getMetadataHelper<int64_t>("get_vocab_size", tokenizer_->vocab_size());
  bos_id_ = getMetadataHelper<int64_t>("get_bos_id", tokenizer_->bos_tok());
  eos_id_ = getMetadataHelper<int64_t>("get_eos_id", tokenizer_->eos_tok());

  // Create sampler
  sampler_ = std::make_unique<Sampler>(
      vocab_size_,
      temperature_,
      kTopp,
      static_cast<unsigned long long>(std::time(nullptr)));

  return Error::Ok;
}

template <typename T>
T Runner::getMetadataHelper(const std::string& method_name, T default_val) {
  T res = default_val;
  if (model_methods_.count(method_name)) {
    Result<std::vector<EValue>> outputs = module_->execute(method_name);
    if (outputs.ok()) {
      std::vector<EValue> outs = outputs.get();
      if (outs.size() > 0) {
        res = outs[0].to<T>();
      }
    }
  } else {
    ET_LOG(
        Info,
        "The model does not contain %s method, using default value %lld",
        method_name.c_str(),
        (long long)default_val);
  }
  ET_LOG(Info, "%s: %lld", method_name.c_str(), (long long)res);
  return res;
}

int32_t Runner::logitsToToken(
    const exec_aten::Tensor& logits_tensor,
    torch::executor::Grammar* grammar,
    const Tokenizer* tokenizer) {
  ET_CHECK_MSG(logits_tensor.dim() == 3, "Logits tensor must be 3D");
  auto num_tokens = logits_tensor.size(1);

  switch (logits_tensor.scalar_type()) {
    case ScalarType::Float: {
      float* logits = logits_tensor.mutable_data_ptr<float>();
      float* logits_last = logits;
      logits_last += (num_tokens - 1) * tokenizer_->vocab_size();
      return sampler_->sample(logits_last, grammar, tokenizer);
    }
    case ScalarType::Half: {
      exec_aten::Half* logits =
          logits_tensor.mutable_data_ptr<exec_aten::Half>();
      exec_aten::Half* logits_last = logits;
      logits_last += (num_tokens - 1) * tokenizer_->vocab_size();
      return sampler_->sample(logits_last, grammar, tokenizer);
    }
    default:
      ET_CHECK_MSG(
          false,
          "Unsupported dtype output %hhd",
          static_cast<int8_t>(logits_tensor.scalar_type()));
  }
}

Result<torch::executor::Tensor> Runner::prefill(
    const std::vector<uint64_t>& tokens,
    ManagedTensor& managed_tokens,
    ManagedTensor& managed_start_pos,
    std::function<void(const std::string&)> system_msg_callback) {
  // enable_parallel_prefill_ maybe set even when not using kv cache
  // When kv cache is not used, start pos is ignored
  int32_t num_tokens = tokens.size();
  if (enable_parallel_prefill_) {
    managed_tokens.resize({1, num_tokens});
    int64_t* tokens_ptr =
        managed_tokens.get_aliasing_tensor().mutable_data_ptr<int64_t>();
    for (int i = 0; i < num_tokens; i++) {
      // The following assumes batch size = 1
      tokens_ptr[i] = tokens[i];
    }
    std::vector<EValue> inputs;
    auto tokens_tensor = managed_tokens.get_aliasing_tensor();
    auto start_pos = managed_start_pos.get_aliasing_tensor();

    // inputs:[tokens, start_pos]
    inputs.push_back(tokens_tensor);
    inputs.push_back(start_pos);

    Result<std::vector<EValue>> outputs_res = module_->forward(inputs);
    ET_CHECK_OK_OR_RETURN_ERROR(outputs_res.error());
    ET_CHECK_MSG(
        outputs_res.get()[0].isTensor(),
        "Non Tensor Output returned from executing LLM");
    ET_CHECK_MSG(
        outputs_res.get()[0].toTensor().size(1) == num_tokens,
        "Expected number of output tokens %d does not match returned value %zu.",
        num_tokens,
        outputs_res.get()[0].toTensor().size(1));

    start_pos.mutable_data_ptr<int64_t>()[0] = num_tokens;

    uint64_t prev = tokens[0];
    uint64_t cur;
    for (int i = 1; i < num_tokens; i++) {
      cur = tokens[i];
      auto piece_res = tokenizer_->decode(prev, cur);
      ET_CHECK_OK_OR_RETURN_ERROR(piece_res.error());
      // util::safe_printf(piece_res.get().c_str());
      fflush(stdout);
      prev = cur;
      if (system_msg_callback) {
        system_msg_callback(
            "REPL_PROGRESS:" + std::to_string((float)i / (float)num_tokens));
      }
    }
    cur = logitsToToken(outputs_res.get()[0].toTensor());
    auto piece_res = tokenizer_->decode(prev, cur);
    ET_CHECK(piece_res.ok());
    const char* piece = piece_res.get().c_str();
    // util::safe_printf(piece);
    fflush(stdout);
    if (system_msg_callback) {
      // token_callback(piece_res.get().c_str());
    }

    // Return the logits tensor
    stats_.first_token_ms = util::time_in_ms();
    stats_.prompt_eval_end_ms = util::time_in_ms();
    return outputs_res.get()[0].toTensor();
  } else { // sequential prefill
    int64_t pos = 0; // position in the sequence
    int64_t cur_token = tokens[0];
    int64_t prev_token;
    // This is a hack to enable returning a logits tensor from prefill
    auto logits_tensor = managed_tokens.get_aliasing_tensor();
    while (pos < num_tokens) {
      // Run the model
      Result<torch::executor::Tensor> logits_res = run_model_step(
          cur_token, managed_tokens, managed_start_pos, num_tokens);

      ET_CHECK_OK_OR_RETURN_ERROR(logits_res.error());
      logits_tensor = logits_res.get();
      // Hack to enable returning a logits tensor from prefill

      prev_token = cur_token;

      long sample_start_time_ms = util::time_in_ms();
      cur_token = logitsToToken(logits_tensor);
      stats_.aggregate_sampling_time_ms +=
          util::time_in_ms() - sample_start_time_ms;

      // advance the state machine
      if (pos < num_tokens - 1) {
        // prefill, force the next token to be the next prompt token
        cur_token = tokens[pos + 1];
      }
      pos++;

      // print the token as string, decode it with the Tokenizer object
      auto piece_res = tokenizer_->decode(prev_token, cur_token);
      ET_CHECK(piece_res.ok());
      const char* piece = piece_res.get().c_str();
      // util::safe_printf(piece);
      fflush(stdout);
      if (system_msg_callback) {
        system_msg_callback(
            "REPL_PROGRESS:" + std::to_string((float)pos / (float)num_tokens));
      }
    }
    auto start_pos = managed_start_pos.get_aliasing_tensor();
    start_pos.mutable_data_ptr<int64_t>()[0] = num_tokens;
    stats_.first_token_ms = util::time_in_ms();
    stats_.prompt_eval_end_ms = util::time_in_ms();
    return logits_tensor;
  }
}

// Given an input token. Set up the inputs for the model and execute a single
// step. Returning the logits tensor.
Result<torch::executor::Tensor> Runner::run_model_step(
    int64_t input_token,
    ManagedTensor& managed_tokens,
    ManagedTensor& managed_start_pos,
    size_t max_seq_len) {
  // ET_LOG(Info, "Input token %" PRIu64, input_token);
  if (use_kv_cache_) {
    auto tokens = managed_tokens.get_aliasing_tensor();
    auto start_pos = managed_start_pos.get_aliasing_tensor();

    // When using kv-cache our input is always 1 token, so just update to the
    // latest.
    tokens.mutable_data_ptr<int64_t>()[0] = input_token;

    Result<std::vector<EValue>> outputs_res =
        module_->forward({tokens, start_pos});
    ET_CHECK_OK_OR_RETURN_ERROR(outputs_res.error());
    ET_CHECK_MSG(
        outputs_res.get().size() == 1,
        "More then one output returned from executing LLM.");
    ET_CHECK_MSG(
        outputs_res.get()[0].isTensor(),
        "Non Tensor Output returned from executing LLM");

    // Bump start_pos by 1
    start_pos.mutable_data_ptr<int64_t>()[0]++;

    // Return the logits tensor
    return outputs_res.get()[0].toTensor();
  } else { // no kv cache
    std::vector<EValue> inputs;
    auto tokens = managed_tokens.get_aliasing_tensor();
    (void)managed_start_pos; // unused

    // When not using kv-cache our input is the entire history of tokens we have
    // seen, so resize input to be 1 larger and append the new token to the end.
    // TODO does this work in ATen mode?
    tokens.mutable_data_ptr<int64_t>()[tokens.size(1) - 1] = input_token;

    // inputs:[tokens]
    inputs.push_back(tokens);

    Result<std::vector<EValue>> outputs_res = module_->forward(inputs);
    ET_CHECK_OK_OR_RETURN_ERROR(outputs_res.error());
    ET_CHECK_MSG(
        outputs_res.get().size() == 1,
        "More then one output returned from executing LLM.");
    ET_CHECK_MSG(
        outputs_res.get()[0].isTensor(),
        "Non Tensor Output returned from executing LLM");

    if (tokens.size(1) < max_seq_len) {
      // Resize the tokens tensor to be 1 larger for next step.
      // Note that this relies on the fact that underlying memory is the same
      // such that previous tokens stored there will still exist.
      // Not a good thing to rely upon.
      managed_tokens.resize({1, static_cast<int>(tokens.size(1) + 1)});
    }

    // Return the logits tensor
    return outputs_res.get()[0].toTensor();
  }
}

void Runner::repl_enqueue_message(
    std::string msg,
    MsgType type,
    std::string grammar,
    std::string action) {
  switch (type) {
    case SYSTEM:
      systemMessageQueue.enqueue(msg);
      break;

    default:
      messageQueue.enqueue(ReplMsg{msg, grammar, action});
      break;
  }
}

Error Runner::start_repl(
    // model settings
    const std::string& prompt,
    const std::string& antiPrompt,
    const int contextLength,

    // logs and storage
    std::string session_file,
    std::string prompt_cache_file,

    // callbacks
    std::function<void(const std::string&)> token_callback,
    std::function<void(const std::string&)> system_msg_callback,
    std::function<void(const Stats&)> stats_callback) {
  try {
    // Prepare the inputs.
    // Use ones-initialized inputs.
    ET_CHECK_MSG(!prompt.empty(), "Prompt cannot be null");
    if (!is_loaded()) {
      stats_.model_load_start_ms = util::time_in_ms();
      ET_CHECK_OK_OR_RETURN_ERROR(load());
      stats_.model_load_end_ms = util::time_in_ms();
    }

    // this is the context length
    int seq_len = contextLength;

    // First token time only measures the time it takes to encode the prompt and
    // return a response token.
    shouldStop_ = false;

    // Set the sequence length to the max seq length if not provided
    seq_len = (seq_len > 0 && seq_len <= max_seq_len_) ? seq_len : max_seq_len_;

    Result<std::vector<uint64_t>> encode_res =
        tokenizer_->encode(prompt, n_bos_, append_eos_ ? n_eos_ : 0);

    ET_CHECK_OK_OR_RETURN_ERROR(
        encode_res.error(), "Failed to encode prompt %s", prompt.c_str());

    // encode the (string) prompt into tokens sequence
    std::vector<uint64_t> prompt_tokens = encode_res.get();
    int num_prompt_tokens = prompt_tokens.size();

    system_msg_callback("REPL_LOG:seq_len=" + std::to_string(seq_len));
    system_msg_callback(
        "REPL_LOG:max_seq_len_=" + std::to_string(max_seq_len_));
    system_msg_callback(
        "REPL_LOG:num_prompt_tokens=" + std::to_string(num_prompt_tokens));

    if (num_prompt_tokens < 1) {
      system_msg_callback("REPL_ERROR:Expected at least 1 prompt token");
      return Error::Ok;
    }
    if (num_prompt_tokens > max_seq_len_) {
      system_msg_callback(
          "REPL_ERROR:Max context length exceeded for this model");
      return Error::Ok;
    }
    if (num_prompt_tokens > seq_len) {
      system_msg_callback(
          "REPL_ERROR:Prompt too long, increase the context length in your settings.");
      return Error::Ok;
    }

    std::unique_ptr<torch::executor::Grammar> grammar = nullptr;

    // start the main loop
    int64_t pos = 0; // position in the sequence
    uint32_t eot_id_ = 128009; // hard coded eot_id for llama3

    std::vector<int64_t> token_data; // allocate space for the tokens
    std::vector<exec_aten::SizesType> token_shape = {1, seq_len};

    std::vector<int64_t> start_pos_data; // allocate space for the tokens
    std::vector<exec_aten::SizesType> start_pos_shape = {1};

    token_data.resize(seq_len);
    if (use_kv_cache_) {
      // hard code these to size 1 as kv cache is locked to static size right
      // now.
      start_pos_data.resize(1);
      start_pos_data.push_back(0);
    }

    // initialize tensor wrappers
    ManagedTensor tokens_managed(
        token_data.data(), token_shape, ScalarType::Long);
    // Create with the max shape to approapriately set the capacity of this
    // tensor, then resize back to 1 for first input.
    tokens_managed.resize({1, 1});

    ManagedTensor start_pos_managed(
        start_pos_data.data(), start_pos_shape, ScalarType::Long);

    int64_t prev_token;
    int64_t cur_token = prompt_tokens[0];

    // Prefill first
    // Here feed all tokens to the model and get the next predicted token
    // after the prompt. After that we will enter generate loop.
    auto prefill_res = prefill(
        prompt_tokens, tokens_managed, start_pos_managed, system_msg_callback);
    ET_CHECK_OK_OR_RETURN_ERROR(prefill_res.error());
    exec_aten::Tensor& prefill_res_tensor = prefill_res.get();
    cur_token = logitsToToken(prefill_res_tensor);
    if (use_kv_cache_) {
      // Prefill could be parallel or sequential.
      // Parallel:
      //  kv cache:
      //    - tokens_managed should resized to 1 as inference expects one token
      //    at a time.
      //  no kv cache:
      //    - tokens_managed should be resized to prompt length + 1, as
      //    inference expects all tokens at once.
      // Sequential prefill:
      //  kv cache:
      //     - tokens_managed should be resized to 1, as inference expects one
      //     token at a time.
      //  no kv cache:
      //     - tokens_managed should be resized to prompt length + 1, as
      //     inference expects all tokens at once.
      tokens_managed.resize({1, 1});
    } else {
      tokens_managed.resize({1, num_prompt_tokens + 1});
    }
    pos = num_prompt_tokens;

    if (system_msg_callback) {
      system_msg_callback("REPL_LOG:starting repl...\n");
    }

    // Generate our tokens
    bool done = false;
    bool wait_for_input = true; // start by waiting for input
    std::string last_output = "";
    std::string input_suffix =
        "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n";

    std::vector<uint8_t> last_buffer;

    while (!done) {
      if (wait_for_input) {
        // clear last output
        last_output = "";

        // clear grammar (std::unique frees the memory automatically)
        grammar = nullptr;

        stats_.num_prompt_tokens = prompt_tokens.size();
        stats_.num_generated_tokens = pos - prompt_tokens.size();
        printReport(stats_);
        if (stats_callback) {
          stats_callback(stats_);
        }

        system_msg_callback("REPL_READY:");

        // save the kv cache buffers
        // saveKvCacheBuffers(kv_cache_buffers, prompt_tokens, session_file);

        ReplMsg replMsg;
        messageQueue.wait_dequeue(replMsg);

        wait_for_input = false;

        std::string message = replMsg.msg;

        if (system_msg_callback) {
          system_msg_callback(
              "REPL_LOG:received input message: " + message + "\n");
        }

        // parse grammar
        if (!replMsg.grammar.empty()) {
          if (system_msg_callback) {
            system_msg_callback(
                "REPL_LOG:received input with grammar: " + replMsg.grammar +
                "\n");
          }

          // this automatically frees the previous grammar
          grammar = std::make_unique<torch::executor::Grammar>(replMsg.grammar);
        }

        // we receive a special kill message here to end the repl
        if (message == "###KILL###")
          done = true;

        if (replMsg.action == "REGEN") {
          // tokenize the input suffix
          std::vector<uint64_t> input_suffix_tokens =
              tokenizer_->encode(input_suffix, 0, 0).get();

          // find the last input suffix in the prompt_tokens
          auto it = std::find_end(
              prompt_tokens.begin(),
              prompt_tokens.end(),
              input_suffix_tokens.begin(),
              input_suffix_tokens.end());

          // if we found it
          if (it != prompt_tokens.end()) {
            // move the iterator to the end of the input suffix
            it += input_suffix_tokens.size();

            // get the position of the last input suffix in the prompt tokens
            pos = std::distance(prompt_tokens.begin(), it);

            system_msg_callback(
                "REPL_LOG:rolling back to the last input suffix at position " +
                std::to_string(pos) + "\n");

            // we need to remove all the tokens after the input suffix (not
            // including the input suffix)
            prompt_tokens.erase(it, prompt_tokens.end());

            cur_token = prompt_tokens[pos];

            // rollback start pos
            auto start_pos = start_pos_managed.get_aliasing_tensor();
            start_pos.mutable_data_ptr<int64_t>()[0] = pos;
          }

          // if message is empty, we append the input suffix to start regen
          if (message.length() == 0) {
            message = input_suffix;
          }
        }

        // Add tokens to embd only if the input buffer is non-empty
        if (message.length() > 0) {
          stats_.inference_start_ms = util::time_in_ms();

          // tokenize message without any bos or eos
          Result<std::vector<uint64_t>> encode_res =
              tokenizer_->encode(message, 0, 0);

          ET_CHECK_OK_OR_RETURN_ERROR(
              encode_res.error(), "Failed to encode prompt %s", prompt.c_str());

          // extend prompt_tokens with the new encoded message
          std::vector<uint64_t> new_tokens = encode_res.get();
          prompt_tokens.insert(
              prompt_tokens.end(), new_tokens.begin(), new_tokens.end());

          system_msg_callback("REPL_LOG:current pos=" + std::to_string(pos));
          system_msg_callback(
              "REPL_LOG:prompt token size=" +
              std::to_string(prompt_tokens.size()));

          // if we have more prompt tokens than our current position, force the
          // current token to be from the prompt token
          if (pos < prompt_tokens.size()) {
            cur_token = prompt_tokens[pos];
          }
        }
      }

      // Run the model
      Result<torch::executor::Tensor> logits_res =
          run_model_step(cur_token, tokens_managed, start_pos_managed, seq_len);

      ET_CHECK_OK_OR_RETURN_ERROR(logits_res.error());
      exec_aten::Tensor& logits_tensor = logits_res.get();

      prev_token = cur_token;

      torch::executor::Grammar* grammarPtr = nullptr;
      if (grammar != nullptr && pos >= prompt_tokens.size() - 1) {
        // system_msg_callback("REPL_LOG:applying grammar...\n");
        grammarPtr = grammar.get();
      }

      // process the logits tensor
      long sample_start_time_ms = util::time_in_ms();
      cur_token = logitsToToken(logits_tensor, grammarPtr, tokenizer_.get());
      stats_.aggregate_sampling_time_ms +=
          util::time_in_ms() - sample_start_time_ms;

      // accept the grammar token
      if (grammarPtr != nullptr)
        grammarPtr->accept_token(cur_token, tokenizer_.get());

      pos++;

      // if we have more prompt tokens than our current position, force the
      // current token to be from the prompt token
      if (pos < prompt_tokens.size()) {
        cur_token = prompt_tokens[pos];

        // print progress
        if (system_msg_callback) {
          system_msg_callback(
              "REPL_PROGRESS:" +
              std::to_string((float)pos / (float)prompt_tokens.size()));
        }
      } else {
        // push current token into prompt_tokens
        prompt_tokens.push_back(cur_token);

        // print the token as string, decode it with the Tokenizer object
        auto piece_res = tokenizer_->decode(prev_token, cur_token);
        ET_CHECK(piece_res.ok());

        last_output += piece_res.get();

        if (system_msg_callback) {
          system_msg_callback("REPL_LOG:output: " + piece_res.get() + "\n");
        }

        if (token_callback && cur_token != eos_id_ && cur_token != eot_id_) {
          token_callback(piece_res.get());
        }

        // check for anti prompt
        if (!antiPrompt.empty()) {
          // Check if each of the reverse prompts appears at the end of the
          // output
          size_t search_start_pos =
              last_output.length() > static_cast<size_t>(antiPrompt.length())
              ? last_output.length() - static_cast<size_t>(antiPrompt.length())
              : 0;

          if (last_output.find(antiPrompt, search_start_pos) !=
              std::string::npos) {
            wait_for_input = true;
          }
        }

        // we have hit EOS, wait for user input
        if (cur_token == eos_id_ || cur_token == eot_id_) {
          wait_for_input = true;
        }
      }

      // check for system prompt that forces stop
      // try to dequeue systemMessage
      std::string systemMessage;
      if (systemMessageQueue.try_dequeue(systemMessage)) {
        // log
        system_msg_callback(
            "Processing system message: " + systemMessage + "\n");

        // stops eval, and passes back to the user
        if (systemMessage == "STOP") {
          wait_for_input = true;

          // remove all prompt tokens after the current pos
          prompt_tokens.erase(prompt_tokens.begin() + pos, prompt_tokens.end());
        }

        // kills the repl
        if (systemMessage == "KILL")
          done = true;
      }
    }
    stats_.inference_end_ms = util::time_in_ms();
    printf("\n");

    // final save of kv cache buffers
    // saveKvCacheBuffers(kv_cache_buffers, prompt_tokens, session_file);
  } catch (const std::exception& e) {
    system_msg_callback("REPL_ERROR:" + std::string(e.what()));
  }

  // log
  system_msg_callback("Executorch REPL exited");

  return Error::Ok;
}

Result<std::string> Runner::infer(
    const std::string& prompt,
    const std::string& grammarStr,
    int32_t seq_len) {
  // Prepare the inputs.
  // Use ones-initialized inputs.
  ET_CHECK_MSG(!prompt.empty(), "Prompt cannot be null");
  if (!is_loaded()) {
    stats_.model_load_start_ms = util::time_in_ms();
    ET_CHECK_OK_OR_RETURN_ERROR(load());
    stats_.model_load_end_ms = util::time_in_ms();
  }

  // First token time only measures the time it takes to encode the prompt and
  // return a response token.

  stats_.inference_start_ms = util::time_in_ms();

  // Set the sequence length to the max seq length if not provided
  seq_len = (seq_len > 0 && seq_len <= max_seq_len_) ? seq_len : max_seq_len_;

  Result<std::vector<uint64_t>> encode_res =
      tokenizer_->encode(prompt, n_bos_, append_eos_ ? n_eos_ : 0);

  ET_CHECK_OK_OR_RETURN_ERROR(
      encode_res.error(), "Failed to encode prompt %s", prompt.c_str());

  // encode the (string) prompt into tokens sequence
  std::vector<uint64_t> prompt_tokens = encode_res.get();
  int num_prompt_tokens = prompt_tokens.size();

  std::vector<uint64_t> embd;
  embd.reserve(seq_len);
  embd.resize(seq_len);

  ET_CHECK_MSG(num_prompt_tokens >= 1, "Expected at least 1 prompt token");
  ET_CHECK_MSG(
      num_prompt_tokens < max_seq_len_,
      "Max seq length exceeded - please increase max seq len value in .../llama2/model.py");

  ET_CHECK_MSG(
      num_prompt_tokens < seq_len,
      "Sequence length exceeded - please increase the seq_len value passed to generate()");

  // start the main loop
  int64_t pos = 0; // position in the sequence
  uint32_t eot_id_ = 128009; // hard coded eot_id for llama3

  std::vector<int64_t> token_data; // allocate space for the tokens
  std::vector<exec_aten::SizesType> token_shape = {1, seq_len};

  std::vector<int64_t> start_pos_data; // allocate space for the tokens
  std::vector<exec_aten::SizesType> start_pos_shape = {1};

  // parse our grammar if we have one
  std::unique_ptr<torch::executor::Grammar> grammar = nullptr;
  if (!grammarStr.empty()) {
    grammar = std::make_unique<torch::executor::Grammar>(grammarStr);
  }

  token_data.resize(seq_len);
  if (use_kv_cache_) {
    // hard code these to size 1 as kv cache is locked to static size right now.
    start_pos_data.resize(1);
    start_pos_data.push_back(0);
  }

  // initialize tensor wrappers
  ManagedTensor tokens_managed(
      token_data.data(), token_shape, ScalarType::Long);
  // Create with the max shape to approapriately set the capacity of this
  // tensor, then resize back to 1 for first input.
  tokens_managed.resize({1, 1});

  ManagedTensor start_pos_managed(
      start_pos_data.data(), start_pos_shape, ScalarType::Long);

  int64_t prev_token;
  int64_t cur_token = prompt_tokens[0];

  // Prefill first
  // Here feed all tokens to the model and get the next predicted token
  // after the prompt. After that we will enter generate loop.
  auto prefill_res =
      prefill(prompt_tokens, tokens_managed, start_pos_managed, nullptr);
  ET_CHECK_OK_OR_RETURN_ERROR(prefill_res.error());
  exec_aten::Tensor& prefill_res_tensor = prefill_res.get();
  cur_token = logitsToToken(prefill_res_tensor);
  if (use_kv_cache_) {
    // Prefill could be parallel or sequential.
    // Parallel:
    //  kv cache:
    //    - tokens_managed should resized to 1 as inference expects one token at
    //    a time.
    //  no kv cache:
    //    - tokens_managed should be resized to prompt length + 1, as inference
    //    expects all tokens at once.
    // Sequential prefill:
    //  kv cache:
    //     - tokens_managed should be resized to 1, as inference expects one
    //     token at a time.
    //  no kv cache:
    //     - tokens_managed should be resized to prompt length + 1, as inference
    //     expects all tokens at once.
    tokens_managed.resize({1, 1});
  } else {
    tokens_managed.resize({1, num_prompt_tokens + 1});
  }
  pos = num_prompt_tokens;

  std::string infer_result = "";

  // Generate our tokens
  while (pos < seq_len - 1) {
    // push cur token to embd (embd[pos] = cur_token)
    embd[pos] = cur_token;

    // Run the model
    Result<torch::executor::Tensor> logits_res =
        run_model_step(cur_token, tokens_managed, start_pos_managed, seq_len);

    ET_CHECK_OK_OR_RETURN_ERROR(logits_res.error());
    exec_aten::Tensor& logits_tensor = logits_res.get();

    prev_token = cur_token;

    long sample_start_time_ms = util::time_in_ms();
    cur_token = logitsToToken(
        logits_tensor, grammar ? grammar.get() : nullptr, tokenizer_.get());
    stats_.aggregate_sampling_time_ms +=
        util::time_in_ms() - sample_start_time_ms;

    if (grammar != nullptr) {
      grammar->accept_token(cur_token, tokenizer_.get());
    }

    pos++;

    // print the token as string, decode it with the Tokenizer object
    auto piece_res = tokenizer_->decode(prev_token, cur_token);
    ET_CHECK(piece_res.ok());
    const char* piece = piece_res.get().c_str();

    // same as printf("%s", piece), but skips "unsafe" bytes
    util::safe_printf(piece);
    fflush(stdout);

    if (cur_token != eos_id_ && cur_token != eot_id_) {
      infer_result += piece;
    }

    // data-dependent terminating condition: we have n_eos_ number of EOS
    if (cur_token == eos_id_ || cur_token == eot_id_) {
      printf("\n");
      ET_LOG(Info, "\nReached to the end of generation");
      break;
    }
  }
  stats_.inference_end_ms = util::time_in_ms();
  printf("\n");

  if (pos == seq_len) {
    ET_LOG(Info, "Sequence length (%i tokens) reached!", seq_len);
  }

  stats_.num_prompt_tokens = num_prompt_tokens;
  stats_.num_generated_tokens = pos - num_prompt_tokens;

  return infer_result;
}

Error Runner::generate(
    const std::string& prompt,
    const std::string& grammarStr,
    int32_t seq_len,
    std::function<void(const std::string&)> token_callback,
    std::function<void(const Stats&)> stats_callback) {
  // Prepare the inputs.
  // Use ones-initialized inputs.
  ET_CHECK_MSG(!prompt.empty(), "Prompt cannot be null");
  if (!is_loaded()) {
    stats_.model_load_start_ms = util::time_in_ms();
    ET_CHECK_OK_OR_RETURN_ERROR(load());
    stats_.model_load_end_ms = util::time_in_ms();
  }

  // First token time only measures the time it takes to encode the prompt and
  // return a response token.

  stats_.inference_start_ms = util::time_in_ms();
  shouldStop_ = false;

  // Set the sequence length to the max seq length if not provided
  seq_len = (seq_len > 0 && seq_len <= max_seq_len_) ? seq_len : max_seq_len_;

  Result<std::vector<uint64_t>> encode_res =
      tokenizer_->encode(prompt, n_bos_, append_eos_ ? n_eos_ : 0);

  ET_CHECK_OK_OR_RETURN_ERROR(
      encode_res.error(), "Failed to encode prompt %s", prompt.c_str());

  // encode the (string) prompt into tokens sequence
  std::vector<uint64_t> prompt_tokens = encode_res.get();
  int num_prompt_tokens = prompt_tokens.size();

  std::vector<uint64_t> embd;
  embd.reserve(seq_len);
  embd.resize(seq_len);

  ET_CHECK_MSG(num_prompt_tokens >= 1, "Expected at least 1 prompt token");
  ET_CHECK_MSG(
      num_prompt_tokens < max_seq_len_,
      "Max seq length exceeded - please increase max seq len value in .../llama2/model.py");

  ET_CHECK_MSG(
      num_prompt_tokens < seq_len,
      "Sequence length exceeded - please increase the seq_len value passed to generate()");

  // start the main loop
  int64_t pos = 0; // position in the sequence
  uint32_t eot_id_ = 128009; // hard coded eot_id for llama3

  std::vector<int64_t> token_data; // allocate space for the tokens
  std::vector<exec_aten::SizesType> token_shape = {1, seq_len};

  std::vector<int64_t> start_pos_data; // allocate space for the tokens
  std::vector<exec_aten::SizesType> start_pos_shape = {1};

  // parse our grammar if we have one
  std::unique_ptr<torch::executor::Grammar> grammar = nullptr;
  if (!grammarStr.empty()) {
    grammar = std::make_unique<torch::executor::Grammar>(grammarStr);
  }

  token_data.resize(seq_len);
  if (use_kv_cache_) {
    // hard code these to size 1 as kv cache is locked to static size right now.
    start_pos_data.resize(1);
    start_pos_data.push_back(0);
  }

  // initialize tensor wrappers
  ManagedTensor tokens_managed(
      token_data.data(), token_shape, ScalarType::Long);
  // Create with the max shape to approapriately set the capacity of this
  // tensor, then resize back to 1 for first input.
  tokens_managed.resize({1, 1});

  ManagedTensor start_pos_managed(
      start_pos_data.data(), start_pos_shape, ScalarType::Long);

  int64_t prev_token;
  int64_t cur_token = prompt_tokens[0];

  // Prefill first
  // Here feed all tokens to the model and get the next predicted token
  // after the prompt. After that we will enter generate loop.
  auto prefill_res =
      prefill(prompt_tokens, tokens_managed, start_pos_managed, token_callback);
  ET_CHECK_OK_OR_RETURN_ERROR(prefill_res.error());
  exec_aten::Tensor& prefill_res_tensor = prefill_res.get();
  cur_token = logitsToToken(prefill_res_tensor);
  if (use_kv_cache_) {
    // Prefill could be parallel or sequential.
    // Parallel:
    //  kv cache:
    //    - tokens_managed should resized to 1 as inference expects one token at
    //    a time.
    //  no kv cache:
    //    - tokens_managed should be resized to prompt length + 1, as inference
    //    expects all tokens at once.
    // Sequential prefill:
    //  kv cache:
    //     - tokens_managed should be resized to 1, as inference expects one
    //     token at a time.
    //  no kv cache:
    //     - tokens_managed should be resized to prompt length + 1, as inference
    //     expects all tokens at once.
    tokens_managed.resize({1, 1});
  } else {
    tokens_managed.resize({1, num_prompt_tokens + 1});
  }
  pos = num_prompt_tokens;

  // Generate our tokens
  while (pos < seq_len - 1) {
    // push cur token to embd (embd[pos] = cur_token)
    embd[pos] = cur_token;

    // Run the model
    Result<torch::executor::Tensor> logits_res =
        run_model_step(cur_token, tokens_managed, start_pos_managed, seq_len);

    ET_CHECK_OK_OR_RETURN_ERROR(logits_res.error());
    exec_aten::Tensor& logits_tensor = logits_res.get();

    prev_token = cur_token;

    long sample_start_time_ms = util::time_in_ms();
    cur_token = logitsToToken(
        logits_tensor, grammar ? grammar.get() : nullptr, tokenizer_.get());
    stats_.aggregate_sampling_time_ms +=
        util::time_in_ms() - sample_start_time_ms;

    if (grammar != nullptr) {
      grammar->accept_token(cur_token, tokenizer_.get());
    }

    pos++;

    // print the token as string, decode it with the Tokenizer object
    auto piece_res = tokenizer_->decode(prev_token, cur_token);
    ET_CHECK(piece_res.ok());
    const char* piece = piece_res.get().c_str();

    // same as printf("%s", piece), but skips "unsafe" bytes
    util::safe_printf(piece);
    fflush(stdout);

    if (token_callback && cur_token != eos_id_ && cur_token != eot_id_) {
      token_callback(piece);
    }

    if (shouldStop_) {
      break;
    }

    // data-dependent terminating condition: we have n_eos_ number of EOS
    if (pos >= num_prompt_tokens || cur_token == eos_id_ ||
        cur_token == eot_id_) {
      printf("\n");
      ET_LOG(Info, "\nReached to the end of generation");
      break;
    }
  }
  stats_.inference_end_ms = util::time_in_ms();
  printf("\n");

  if (pos == seq_len) {
    ET_LOG(Info, "Sequence length (%i tokens) reached!", seq_len);
  }

  stats_.num_prompt_tokens = num_prompt_tokens;
  stats_.num_generated_tokens = pos - num_prompt_tokens;
  printReport(stats_);
  if (stats_callback) {
    stats_callback(stats_);
  }

  return Error::Ok;
}

namespace {
void printReport(const Runner::Stats& stats) {
  printf("PyTorchObserver %s\n", statsToJsonString(stats).c_str());

  ET_LOG(
      Info,
      "\tPrompt Tokens: %" PRIu64 "    Generated Tokens: %" PRIu64,
      stats.num_prompt_tokens,
      stats.num_generated_tokens);

  ET_LOG(
      Info,
      "\tModel Load Time:\t\t%f (seconds)",
      ((double)(stats.model_load_end_ms - stats.model_load_start_ms) /
       stats.SCALING_FACTOR_UNITS_PER_SECOND));
  double inference_time_ms =
      (double)(stats.inference_end_ms - stats.inference_start_ms);
  ET_LOG(
      Info,
      "\tTotal inference time:\t\t%f (seconds)\t\t Rate: \t%f (tokens/second)",
      inference_time_ms / stats.SCALING_FACTOR_UNITS_PER_SECOND,

      (stats.num_generated_tokens) /
          (double)(stats.inference_end_ms - stats.inference_start_ms) *
          stats.SCALING_FACTOR_UNITS_PER_SECOND);
  double prompt_eval_time =
      (double)(stats.prompt_eval_end_ms - stats.inference_start_ms);
  ET_LOG(
      Info,
      "\t\tPrompt evaluation:\t%f (seconds)\t\t Rate: \t%f (tokens/second)",
      prompt_eval_time / stats.SCALING_FACTOR_UNITS_PER_SECOND,
      (stats.num_prompt_tokens) / prompt_eval_time *
          stats.SCALING_FACTOR_UNITS_PER_SECOND);

  double eval_time =
      (double)(stats.inference_end_ms - stats.prompt_eval_end_ms);
  ET_LOG(
      Info,
      "\t\tGenerated %" PRIu64
      " tokens:\t%f (seconds)\t\t Rate: \t%f (tokens/second)",
      stats.num_generated_tokens,
      eval_time / stats.SCALING_FACTOR_UNITS_PER_SECOND,
      stats.num_generated_tokens / eval_time *
          stats.SCALING_FACTOR_UNITS_PER_SECOND);

  // Time to first token is measured from the start of inference, excluding
  // model load time.
  ET_LOG(
      Info,
      "\tTime to first generated token:\t%f (seconds)",
      ((double)(stats.first_token_ms - stats.inference_start_ms) /
       stats.SCALING_FACTOR_UNITS_PER_SECOND));

  ET_LOG(
      Info,
      "\tSampling time over %" PRIu64 " tokens:\t%f (seconds)",
      stats.num_prompt_tokens + stats.num_generated_tokens,
      (double)stats.aggregate_sampling_time_ms /
          stats.SCALING_FACTOR_UNITS_PER_SECOND);
}

std::string statsToJsonString(const Runner::Stats& stats) {
  std::stringstream ss;
  ss << "{\"prompt_tokens\":" << stats.num_prompt_tokens << ","
     << "\"generated_tokens\":" << stats.num_generated_tokens << ","
     << "\"model_load_start_ms\":" << stats.model_load_start_ms << ","
     << "\"model_load_end_ms\":" << stats.model_load_end_ms << ","
     << "\"inference_start_ms\":" << stats.inference_start_ms << ","
     << "\"inference_end_ms\":" << stats.inference_end_ms << ","
     << "\"prompt_eval_end_ms\":" << stats.prompt_eval_end_ms << ","
     << "\"first_token_ms\":" << stats.first_token_ms << ","
     << "\"aggregate_sampling_time_ms\":" << stats.aggregate_sampling_time_ms
     << "," << "\"SCALING_FACTOR_UNITS_PER_SECOND\":"
     << stats.SCALING_FACTOR_UNITS_PER_SECOND << "}";
  return ss.str();
}
} // namespace

void Runner::stop() {
  shouldStop_ = true;
}

// explicit instantiation of template methods
template int64_t Runner::getMetadataHelper<int64_t>(
    const std::string& method_name,
    int64_t default_val);
template bool Runner::getMetadataHelper<bool>(
    const std::string& method_name,
    bool default_val);
} // namespace torch::executor
