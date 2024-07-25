#pragma once

#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <string>
#include <vector>
#ifdef USE_ATEN_LIB
#include <torch/torch.h>
#endif

#include <executorch/extension/llm/tokenizer/tokenizer.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>

namespace torch {
namespace executor {

// grammar element type
enum llama_gretype {
  // end of rule definition
  LLAMA_GRETYPE_END = 0,

  // start of alternate definition for rule
  LLAMA_GRETYPE_ALT = 1,

  // non-terminal element: reference to rule
  LLAMA_GRETYPE_RULE_REF = 2,

  // terminal element: character (code point)
  LLAMA_GRETYPE_CHAR = 3,

  // inverse char(s) ([^a], [^a-b] [^abc])
  LLAMA_GRETYPE_CHAR_NOT = 4,

  // modifies a preceding LLAMA_GRETYPE_CHAR or LLAMA_GRETYPE_CHAR_ALT to
  // be an inclusive range ([a-z])
  LLAMA_GRETYPE_CHAR_RNG_UPPER = 5,

  // modifies a preceding LLAMA_GRETYPE_CHAR or
  // LLAMA_GRETYPE_CHAR_RNG_UPPER to add an alternate char to match ([ab],
  // [a-zA])
  LLAMA_GRETYPE_CHAR_ALT = 6,

  // any character (.)
  LLAMA_GRETYPE_CHAR_ANY = 7,
};

typedef struct llama_grammar_element {
  enum llama_gretype type;
  uint32_t value; // Unicode code point or rule ID
} llama_grammar_element;

namespace grammar_parser {

struct parse_state {
  std::map<std::string, uint32_t> symbol_ids;
  std::vector<std::vector<llama_grammar_element>> rules;

  std::vector<const llama_grammar_element*> c_rules();
};

parse_state parse(const char* src);
void print_grammar(FILE* file, const parse_state& state);

} // namespace grammar_parser

struct llama_partial_utf8 {
  uint32_t value; // bit value so far (unshifted)
  int n_remain; // num bytes remaining; -1 indicates invalid sequence
};

struct llama_grammar {
  const std::vector<std::vector<llama_grammar_element>> rules;
  std::vector<std::vector<const llama_grammar_element*>> stacks;

  // buffer for partially generated UTF-8 sequence from accepted tokens
  llama_partial_utf8 partial_utf8;
};

class Grammar {
 public:
  Grammar(std::string grammar);

 public:
  template <typename T>
  void sample_grammar(T* probabilities, const Tokenizer* tokenizer);

  void accept_token(uint32_t token, const Tokenizer* tokenizer);

 private:
  llama_grammar* grammar;
};

} // namespace executor
} // namespace torch
