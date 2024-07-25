#include <executorch/examples/models/llama2/sampler/grammar.h>

#include <algorithm>
#include <array>
#include <cassert>
#include <cctype>
#include <cfloat>
#include <cinttypes>
#include <climits>
#include <cmath>
#include <cstdarg>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <cwchar>
#include <exception>
#include <forward_list>
#include <fstream>
#include <functional>
#include <future>
#include <initializer_list>
#include <locale>
#include <map>
#include <memory>
#include <mutex>
#include <numeric>
#include <queue>
#include <random>
#include <regex>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

namespace torch {
namespace executor {

namespace grammar_parser {
// NOTE: assumes valid utf8 (but checks for overrun)
// copied from llama.cpp
static std::pair<uint32_t, const char*> decode_utf8(const char* src) {
  static const int lookup[] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 3, 4};
  uint8_t first_byte = static_cast<uint8_t>(*src);
  uint8_t highbits = first_byte >> 4;
  int len = lookup[highbits];
  uint8_t mask = (1 << (8 - len)) - 1;
  uint32_t value = first_byte & mask;
  const char* end = src + len; // may overrun!
  const char* pos = src + 1;
  for (; pos < end && *pos; pos++) {
    value = (value << 6) + (static_cast<uint8_t>(*pos) & 0x3F);
  }
  return std::make_pair(value, pos);
}

static uint32_t get_symbol_id(parse_state& state, const char* src, size_t len) {
  uint32_t next_id = static_cast<uint32_t>(state.symbol_ids.size());
  auto result = state.symbol_ids.emplace(std::string(src, len), next_id);
  return result.first->second;
}

static uint32_t generate_symbol_id(
    parse_state& state,
    const std::string& base_name) {
  uint32_t next_id = static_cast<uint32_t>(state.symbol_ids.size());
  state.symbol_ids[base_name + '_' + std::to_string(next_id)] = next_id;
  return next_id;
}

static void add_rule(
    parse_state& state,
    uint32_t rule_id,
    const std::vector<llama_grammar_element>& rule) {
  if (state.rules.size() <= rule_id) {
    state.rules.resize(rule_id + 1);
  }
  state.rules[rule_id] = rule;
}

static bool is_digit_char(char c) {
  return '0' <= c && c <= '9';
}

static bool is_word_char(char c) {
  return ('a' <= c && c <= 'z') || ('A' <= c && c <= 'Z') || c == '-' ||
      is_digit_char(c);
}

static std::pair<uint32_t, const char*> parse_hex(const char* src, int size) {
  const char* pos = src;
  const char* end = src + size;
  uint32_t value = 0;
  for (; pos < end && *pos; pos++) {
    value <<= 4;
    char c = *pos;
    if ('a' <= c && c <= 'f') {
      value += c - 'a' + 10;
    } else if ('A' <= c && c <= 'F') {
      value += c - 'A' + 10;
    } else if ('0' <= c && c <= '9') {
      value += c - '0';
    } else {
      break;
    }
  }
  if (pos != end) {
    throw std::runtime_error(
        "expecting " + std::to_string(size) + " hex chars at " + src);
  }
  return std::make_pair(value, pos);
}

static const char* parse_space(const char* src, bool newline_ok) {
  const char* pos = src;
  while (*pos == ' ' || *pos == '\t' || *pos == '#' ||
         (newline_ok && (*pos == '\r' || *pos == '\n'))) {
    if (*pos == '#') {
      while (*pos && *pos != '\r' && *pos != '\n') {
        pos++;
      }
    } else {
      pos++;
    }
  }
  return pos;
}

static const char* parse_name(const char* src) {
  const char* pos = src;
  while (is_word_char(*pos)) {
    pos++;
  }
  if (pos == src) {
    throw std::runtime_error(std::string("expecting name at ") + src);
  }
  return pos;
}

static const char* parse_int(const char* src) {
  const char* pos = src;
  while (is_digit_char(*pos)) {
    pos++;
  }
  if (pos == src) {
    throw std::runtime_error(std::string("expecting integer at ") + src);
  }
  return pos;
}

static std::pair<uint32_t, const char*> parse_char(const char* src) {
  if (*src == '\\') {
    switch (src[1]) {
      case 'x':
        return parse_hex(src + 2, 2);
      case 'u':
        return parse_hex(src + 2, 4);
      case 'U':
        return parse_hex(src + 2, 8);
      case 't':
        return std::make_pair('\t', src + 2);
      case 'r':
        return std::make_pair('\r', src + 2);
      case 'n':
        return std::make_pair('\n', src + 2);
      case '\\':
      case '"':
      case '[':
      case ']':
        return std::make_pair(src[1], src + 2);
      default:
        throw std::runtime_error(std::string("unknown escape at ") + src);
    }
  } else if (*src) {
    return decode_utf8(src);
  }
  throw std::runtime_error("unexpected end of input");
}

const char* parse_alternates(
    parse_state& state,
    const char* src,
    const std::string& rule_name,
    uint32_t rule_id,
    bool is_nested);

static const char* parse_sequence(
    parse_state& state,
    const char* src,
    const std::string& rule_name,
    std::vector<llama_grammar_element>& out_elements,
    bool is_nested) {
  size_t last_sym_start = out_elements.size();
  const char* pos = src;

  auto handle_repetitions = [&](int min_times, int max_times) {
    if (last_sym_start == out_elements.size()) {
      throw std::runtime_error(
          std::string("expecting preceding item to */+/?/{ at ") + pos);
    }

    // apply transformation to previous symbol (last_sym_start to end) according
    // to the following rewrite rules: S{m,n} --> S S S (m times) S'(n-m)
    //            S'(x)   ::= S S'(x-1) |
    //            (... n-m definitions of these S' rules ...)
    //            S'(1)   ::= S |
    // S{m,} -->  S S S (m times) S'
    //            S'     ::= S S' |
    // S*     --> S{0,}
    //        --> S'     ::= S S' |
    // S+     --> S{1,}
    //        --> S S'
    //            S'     ::= S S' |
    // S?     --> S{0,1}
    //        --> S'
    //            S'     ::= S |

    std::vector<llama_grammar_element> previous_elements(
        out_elements.begin() + last_sym_start, out_elements.end());
    if (min_times == 0) {
      out_elements.resize(last_sym_start);
    } else {
      // Repeat the previous elements (min_times - 1) times
      for (int i = 1; i < min_times; i++) {
        out_elements.insert(
            out_elements.end(),
            previous_elements.begin(),
            previous_elements.end());
      }
    }

    uint32_t last_rec_rule_id = 0;
    auto n_opt = max_times < 0 ? 1 : max_times - min_times;

    std::vector<llama_grammar_element> rec_rule(previous_elements);
    for (int i = 0; i < n_opt; i++) {
      rec_rule.resize(previous_elements.size());
      uint32_t rec_rule_id = generate_symbol_id(state, rule_name);
      if (i > 0 || max_times < 0) {
        rec_rule.push_back(
            {LLAMA_GRETYPE_RULE_REF,
             max_times < 0 ? rec_rule_id : last_rec_rule_id});
      }
      rec_rule.push_back({LLAMA_GRETYPE_ALT, 0});
      rec_rule.push_back({LLAMA_GRETYPE_END, 0});
      add_rule(state, rec_rule_id, rec_rule);
      last_rec_rule_id = rec_rule_id;
    }
    if (n_opt > 0) {
      out_elements.push_back({LLAMA_GRETYPE_RULE_REF, last_rec_rule_id});
    }
  };

  while (*pos) {
    if (*pos == '"') { // literal string
      pos++;
      last_sym_start = out_elements.size();
      while (*pos != '"') {
        if (!*pos) {
          throw std::runtime_error("unexpected end of input");
        }
        auto char_pair = parse_char(pos);
        pos = char_pair.second;
        out_elements.push_back({LLAMA_GRETYPE_CHAR, char_pair.first});
      }
      pos = parse_space(pos + 1, is_nested);
    } else if (*pos == '[') { // char range(s)
      pos++;
      enum llama_gretype start_type = LLAMA_GRETYPE_CHAR;
      if (*pos == '^') {
        pos++;
        start_type = LLAMA_GRETYPE_CHAR_NOT;
      }
      last_sym_start = out_elements.size();
      while (*pos != ']') {
        if (!*pos) {
          throw std::runtime_error("unexpected end of input");
        }
        auto char_pair = parse_char(pos);
        pos = char_pair.second;
        enum llama_gretype type = last_sym_start < out_elements.size()
            ? LLAMA_GRETYPE_CHAR_ALT
            : start_type;

        out_elements.push_back({type, char_pair.first});
        if (pos[0] == '-' && pos[1] != ']') {
          if (!pos[1]) {
            throw std::runtime_error("unexpected end of input");
          }
          auto endchar_pair = parse_char(pos + 1);
          pos = endchar_pair.second;
          out_elements.push_back(
              {LLAMA_GRETYPE_CHAR_RNG_UPPER, endchar_pair.first});
        }
      }
      pos = parse_space(pos + 1, is_nested);
    } else if (is_word_char(*pos)) { // rule reference
      const char* name_end = parse_name(pos);
      uint32_t ref_rule_id = get_symbol_id(state, pos, name_end - pos);
      pos = parse_space(name_end, is_nested);
      last_sym_start = out_elements.size();
      out_elements.push_back({LLAMA_GRETYPE_RULE_REF, ref_rule_id});
    } else if (*pos == '(') { // grouping
      // parse nested alternates into synthesized rule
      pos = parse_space(pos + 1, true);
      uint32_t sub_rule_id = generate_symbol_id(state, rule_name);
      pos = parse_alternates(state, pos, rule_name, sub_rule_id, true);
      last_sym_start = out_elements.size();
      // output reference to synthesized rule
      out_elements.push_back({LLAMA_GRETYPE_RULE_REF, sub_rule_id});
      if (*pos != ')') {
        throw std::runtime_error(std::string("expecting ')' at ") + pos);
      }
      pos = parse_space(pos + 1, is_nested);
    } else if (*pos == '.') { // any char
      last_sym_start = out_elements.size();
      out_elements.push_back({LLAMA_GRETYPE_CHAR_ANY, 0});
      pos = parse_space(pos + 1, is_nested);
    } else if (*pos == '*') {
      pos = parse_space(pos + 1, is_nested);
      handle_repetitions(0, -1);
    } else if (*pos == '+') {
      pos = parse_space(pos + 1, is_nested);
      handle_repetitions(1, -1);
    } else if (*pos == '?') {
      pos = parse_space(pos + 1, is_nested);
      handle_repetitions(0, 1);
    } else if (*pos == '{') {
      pos = parse_space(pos + 1, is_nested);

      if (!is_digit_char(*pos)) {
        throw std::runtime_error(std::string("expecting an int at ") + pos);
      }
      const char* int_end = parse_int(pos);
      int min_times = std::stoul(std::string(pos, int_end - pos));
      pos = parse_space(int_end, is_nested);

      int max_times = -1;

      if (*pos == '}') {
        max_times = min_times;
        pos = parse_space(pos + 1, is_nested);
      } else if (*pos == ',') {
        pos = parse_space(pos + 1, is_nested);

        if (is_digit_char(*pos)) {
          const char* int_end = parse_int(pos);
          max_times = std::stoul(std::string(pos, int_end - pos));
          pos = parse_space(int_end, is_nested);
        }

        if (*pos != '}') {
          throw std::runtime_error(std::string("expecting '}' at ") + pos);
        }
        pos = parse_space(pos + 1, is_nested);
      } else {
        throw std::runtime_error(std::string("expecting ',' at ") + pos);
      }
      handle_repetitions(min_times, max_times);
    } else {
      break;
    }
  }
  return pos;
}

const char* parse_alternates(
    parse_state& state,
    const char* src,
    const std::string& rule_name,
    uint32_t rule_id,
    bool is_nested) {
  std::vector<llama_grammar_element> rule;
  const char* pos = parse_sequence(state, src, rule_name, rule, is_nested);
  while (*pos == '|') {
    rule.push_back({LLAMA_GRETYPE_ALT, 0});
    pos = parse_space(pos + 1, true);
    pos = parse_sequence(state, pos, rule_name, rule, is_nested);
  }
  rule.push_back({LLAMA_GRETYPE_END, 0});
  add_rule(state, rule_id, rule);
  return pos;
}

static const char* parse_rule(parse_state& state, const char* src) {
  const char* name_end = parse_name(src);
  const char* pos = parse_space(name_end, false);
  size_t name_len = name_end - src;
  uint32_t rule_id = get_symbol_id(state, src, name_len);
  const std::string name(src, name_len);

  if (!(pos[0] == ':' && pos[1] == ':' && pos[2] == '=')) {
    throw std::runtime_error(std::string("expecting ::= at ") + pos);
  }
  pos = parse_space(pos + 3, true);

  pos = parse_alternates(state, pos, name, rule_id, false);

  if (*pos == '\r') {
    pos += pos[1] == '\n' ? 2 : 1;
  } else if (*pos == '\n') {
    pos++;
  } else if (*pos) {
    throw std::runtime_error(std::string("expecting newline or end at ") + pos);
  }
  return parse_space(pos, true);
}

parse_state parse(const char* src) {
  try {
    parse_state state;
    const char* pos = parse_space(src, true);
    while (*pos) {
      pos = parse_rule(state, pos);
    }
    // Validate the state to ensure that all rules are defined
    for (const auto& rule : state.rules) {
      for (const auto& elem : rule) {
        if (elem.type == LLAMA_GRETYPE_RULE_REF) {
          // Ensure that the rule at that location exists
          if (elem.value >= state.rules.size() ||
              state.rules[elem.value].empty()) {
            // Get the name of the rule that is missing
            for (const auto& kv : state.symbol_ids) {
              if (kv.second == elem.value) {
                throw std::runtime_error(
                    "Undefined rule identifier '" + kv.first + "'");
              }
            }
          }
        }
      }
    }
    return state;
  } catch (const std::exception& err) {
    fprintf(stderr, "%s: error parsing grammar: %s\n", __func__, err.what());
    return parse_state();
  }
}

static void print_grammar_char(FILE* file, uint32_t c) {
  if (0x20 <= c && c <= 0x7f) {
    fprintf(file, "%c", static_cast<char>(c));
  } else {
    // cop out of encoding UTF-8
    fprintf(file, "<U+%04X>", c);
  }
}

static bool is_char_element(llama_grammar_element elem) {
  switch (elem.type) {
    case LLAMA_GRETYPE_CHAR:
      return true;
    case LLAMA_GRETYPE_CHAR_NOT:
      return true;
    case LLAMA_GRETYPE_CHAR_ALT:
      return true;
    case LLAMA_GRETYPE_CHAR_RNG_UPPER:
      return true;
    case LLAMA_GRETYPE_CHAR_ANY:
      return true;
    default:
      return false;
  }
}

static void print_rule_binary(
    FILE* file,
    const std::vector<llama_grammar_element>& rule) {
  for (auto elem : rule) {
    switch (elem.type) {
      case LLAMA_GRETYPE_END:
        fprintf(file, "END");
        break;
      case LLAMA_GRETYPE_ALT:
        fprintf(file, "ALT");
        break;
      case LLAMA_GRETYPE_RULE_REF:
        fprintf(file, "RULE_REF");
        break;
      case LLAMA_GRETYPE_CHAR:
        fprintf(file, "CHAR");
        break;
      case LLAMA_GRETYPE_CHAR_NOT:
        fprintf(file, "CHAR_NOT");
        break;
      case LLAMA_GRETYPE_CHAR_RNG_UPPER:
        fprintf(file, "CHAR_RNG_UPPER");
        break;
      case LLAMA_GRETYPE_CHAR_ALT:
        fprintf(file, "CHAR_ALT");
        break;
      case LLAMA_GRETYPE_CHAR_ANY:
        fprintf(file, "CHAR_ANY");
        break;
    }
    switch (elem.type) {
      case LLAMA_GRETYPE_END:
      case LLAMA_GRETYPE_ALT:
      case LLAMA_GRETYPE_RULE_REF:
        fprintf(file, "(%u) ", elem.value);
        break;
      case LLAMA_GRETYPE_CHAR:
      case LLAMA_GRETYPE_CHAR_NOT:
      case LLAMA_GRETYPE_CHAR_RNG_UPPER:
      case LLAMA_GRETYPE_CHAR_ALT:
      case LLAMA_GRETYPE_CHAR_ANY:
        fprintf(file, "(\"");
        print_grammar_char(file, elem.value);
        fprintf(file, "\") ");
        break;
    }
  }
  fprintf(file, "\n");
}

static void print_rule(
    FILE* file,
    uint32_t rule_id,
    const std::vector<llama_grammar_element>& rule,
    const std::map<uint32_t, std::string>& symbol_id_names) {
  if (rule.empty() || rule.back().type != LLAMA_GRETYPE_END) {
    throw std::runtime_error(
        "malformed rule, does not end with LLAMA_GRETYPE_END: " +
        std::to_string(rule_id));
  }
  fprintf(file, "%s ::= ", symbol_id_names.at(rule_id).c_str());
  for (size_t i = 0, end = rule.size() - 1; i < end; i++) {
    llama_grammar_element elem = rule[i];
    switch (elem.type) {
      case LLAMA_GRETYPE_END:
        throw std::runtime_error(
            "unexpected end of rule: " + std::to_string(rule_id) + "," +
            std::to_string(i));
      case LLAMA_GRETYPE_ALT:
        fprintf(file, "| ");
        break;
      case LLAMA_GRETYPE_RULE_REF:
        fprintf(file, "%s ", symbol_id_names.at(elem.value).c_str());
        break;
      case LLAMA_GRETYPE_CHAR:
        fprintf(file, "[");
        print_grammar_char(file, elem.value);
        break;
      case LLAMA_GRETYPE_CHAR_NOT:
        fprintf(file, "[^");
        print_grammar_char(file, elem.value);
        break;
      case LLAMA_GRETYPE_CHAR_RNG_UPPER:
        if (i == 0 || !is_char_element(rule[i - 1])) {
          throw std::runtime_error(
              "LLAMA_GRETYPE_CHAR_RNG_UPPER without preceding char: " +
              std::to_string(rule_id) + "," + std::to_string(i));
        }
        fprintf(file, "-");
        print_grammar_char(file, elem.value);
        break;
      case LLAMA_GRETYPE_CHAR_ALT:
        if (i == 0 || !is_char_element(rule[i - 1])) {
          throw std::runtime_error(
              "LLAMA_GRETYPE_CHAR_ALT without preceding char: " +
              std::to_string(rule_id) + "," + std::to_string(i));
        }
        print_grammar_char(file, elem.value);
        break;
      case LLAMA_GRETYPE_CHAR_ANY:
        fprintf(file, ".");
        break;
    }
    if (is_char_element(elem)) {
      switch (rule[i + 1].type) {
        case LLAMA_GRETYPE_CHAR_ALT:
        case LLAMA_GRETYPE_CHAR_RNG_UPPER:
        case LLAMA_GRETYPE_CHAR_ANY:
          break;
        default:
          fprintf(file, "] ");
      }
    }
  }
  fprintf(file, "\n");
}

void print_grammar(FILE* file, const parse_state& state) {
  try {
    std::map<uint32_t, std::string> symbol_id_names;
    for (const auto& kv : state.symbol_ids) {
      symbol_id_names[kv.second] = kv.first;
    }
    for (size_t i = 0, end = state.rules.size(); i < end; i++) {
      // fprintf(file, "%zu: ", i);
      // print_rule_binary(file, state.rules[i]);
      print_rule(file, uint32_t(i), state.rules[i], symbol_id_names);
      // fprintf(file, "\n");
    }
  } catch (const std::exception& err) {
    fprintf(stderr, "\n%s: error printing grammar: %s\n", __func__, err.what());
  }
}

std::vector<const llama_grammar_element*> parse_state::c_rules() {
  std::vector<const llama_grammar_element*> ret;
  ret.reserve(rules.size());
  for (const auto& rule : rules) {
    ret.push_back(rule.data());
  }
  return ret;
}
} // namespace grammar_parser

struct llama_grammar_candidate {
  size_t index;
  const uint32_t* code_points;
  llama_partial_utf8 partial_utf8;
};

// function declarations

static std::vector<llama_grammar_candidate>
llama_grammar_reject_candidates_for_stack(
    const std::vector<std::vector<llama_grammar_element>>& rules,
    const std::vector<const llama_grammar_element*>& stack,
    const std::vector<llama_grammar_candidate>& candidates);

// implementations

// returns true iff pos points to the end of one of the definitions of a rule
static bool llama_grammar_is_end_of_sequence(const llama_grammar_element* pos) {
  switch (pos->type) {
    case LLAMA_GRETYPE_END:
      return true; // NOLINT
    case LLAMA_GRETYPE_ALT:
      return true; // NOLINT
    default:
      return false;
  }
}

// transforms a grammar pushdown stack into N possible stacks, all ending
// at a character range (terminal element)
static void llama_grammar_advance_stack(
    const std::vector<std::vector<llama_grammar_element>>& rules,
    const std::vector<const llama_grammar_element*>& stack,
    std::vector<std::vector<const llama_grammar_element*>>& new_stacks) {
  if (stack.empty()) {
    if (std::find(new_stacks.begin(), new_stacks.end(), stack) ==
        new_stacks.end()) {
      new_stacks.emplace_back(stack);
    }
    return;
  }

  const llama_grammar_element* pos = stack.back();

  switch (pos->type) {
    case LLAMA_GRETYPE_RULE_REF: {
      const size_t rule_id = static_cast<size_t>(pos->value);
      const llama_grammar_element* subpos = rules[rule_id].data();
      do {
        // init new stack without the top (pos)
        std::vector<const llama_grammar_element*> new_stack(
            stack.begin(), stack.end() - 1);
        if (!llama_grammar_is_end_of_sequence(pos + 1)) {
          // if this rule ref is followed by another element, add that to stack
          new_stack.push_back(pos + 1);
        }
        if (!llama_grammar_is_end_of_sequence(subpos)) {
          // if alternate is nonempty, add to stack
          new_stack.push_back(subpos);
        }
        llama_grammar_advance_stack(rules, new_stack, new_stacks);
        while (!llama_grammar_is_end_of_sequence(subpos)) {
          // scan to end of alternate def
          subpos++;
        }
        if (subpos->type == LLAMA_GRETYPE_ALT) {
          // there's another alternate def of this rule to process
          subpos++;
        } else {
          break;
        }
      } while (true);
      break;
    }
    case LLAMA_GRETYPE_CHAR:
    case LLAMA_GRETYPE_CHAR_NOT:
    case LLAMA_GRETYPE_CHAR_ANY:
      if (std::find(new_stacks.begin(), new_stacks.end(), stack) ==
          new_stacks.end()) {
        // only add the stack if it's not a duplicate of one we already have
        new_stacks.emplace_back(stack);
      }
      break;
    default:
      // end of alternate (LLAMA_GRETYPE_END, LLAMA_GRETYPE_ALT) or middle of
      // char range (LLAMA_GRETYPE_CHAR_ALT, LLAMA_GRETYPE_CHAR_RNG_UPPER);
      // stack should never be left on those
      assert(false);
  }
}

// returns true iff some continuation of the given partial UTF-8 sequence could
// satisfy the char range at pos (regular or inverse range) asserts that pos is
// pointing to a char range element
static bool llama_grammar_match_partial_char(
    const llama_grammar_element* pos,
    const llama_partial_utf8 partial_utf8) {
  bool is_positive_char =
      pos->type == LLAMA_GRETYPE_CHAR || pos->type == LLAMA_GRETYPE_CHAR_ANY;
  assert(is_positive_char || pos->type == LLAMA_GRETYPE_CHAR_NOT);

  uint32_t partial_value = partial_utf8.value;
  int n_remain = partial_utf8.n_remain;

  // invalid sequence or 7-bit char split across 2 bytes (overlong)
  if (n_remain < 0 || (n_remain == 1 && partial_value < 2)) {
    return false;
  }

  // range of possible code points this partial UTF-8 sequence could complete to
  uint32_t low = partial_value << (n_remain * 6);
  uint32_t high = low | ((1 << (n_remain * 6)) - 1);

  if (low == 0) {
    if (n_remain == 2) {
      low = 1 << 11;
    } else if (n_remain == 3) {
      low = 1 << 16;
    }
  }

  do {
    if (pos[1].type == LLAMA_GRETYPE_CHAR_RNG_UPPER) {
      // inclusive range, e.g. [a-z]
      if (pos->value <= high && low <= pos[1].value) {
        return is_positive_char;
      }
      pos += 2;
    } else if (pos->type == LLAMA_GRETYPE_CHAR_ANY) {
      // Any character matches "."
      return true;
    } else {
      // exact char match, e.g. [a] or "a"
      if (low <= pos->value && pos->value <= high) {
        return is_positive_char;
      }
      pos += 1;
    }
  } while (pos->type == LLAMA_GRETYPE_CHAR_ALT);

  return !is_positive_char;
}

// returns true iff chr satisfies the char range at pos (regular or inverse
// range) asserts that pos is pointing to a char range element
static std::pair<bool, const llama_grammar_element*> llama_grammar_match_char(
    const llama_grammar_element* pos,
    const uint32_t chr) {
  bool found = false;
  bool is_positive_char =
      pos->type == LLAMA_GRETYPE_CHAR || pos->type == LLAMA_GRETYPE_CHAR_ANY;

  assert(is_positive_char || pos->type == LLAMA_GRETYPE_CHAR_NOT); // NOLINT

  do {
    if (pos[1].type == LLAMA_GRETYPE_CHAR_RNG_UPPER) {
      // inclusive range, e.g. [a-z]
      found = found || (pos->value <= chr && chr <= pos[1].value);
      pos += 2;
    } else if (pos->type == LLAMA_GRETYPE_CHAR_ANY) {
      // Any character matches "."
      found = true;
      pos += 1;
    } else {
      // exact char match, e.g. [a] or "a"
      found = found || pos->value == chr;
      pos += 1;
    }
  } while (pos->type == LLAMA_GRETYPE_CHAR_ALT);

  return std::make_pair(found == is_positive_char, pos);
}

static std::vector<llama_grammar_candidate> llama_grammar_reject_candidates(
    const std::vector<std::vector<llama_grammar_element>>& rules,
    const std::vector<std::vector<const llama_grammar_element*>>& stacks,
    const std::vector<llama_grammar_candidate>& candidates) {
  assert(!stacks.empty()); // REVIEW

  if (candidates.empty()) {
    return std::vector<llama_grammar_candidate>();
  }

  auto rejects = llama_grammar_reject_candidates_for_stack(
      rules, stacks.front(), candidates);

  for (size_t i = 1, size = stacks.size(); i < size; ++i) {
    rejects =
        llama_grammar_reject_candidates_for_stack(rules, stacks[i], rejects);
  }
  return rejects;
}

static std::vector<llama_grammar_candidate>
llama_grammar_reject_candidates_for_stack(
    const std::vector<std::vector<llama_grammar_element>>& rules,
    const std::vector<const llama_grammar_element*>& stack,
    const std::vector<llama_grammar_candidate>& candidates) {
  std::vector<llama_grammar_candidate> rejects;
  rejects.reserve(candidates.size());

  if (stack.empty()) {
    for (const auto& tok : candidates) {
      if (*tok.code_points != 0 || tok.partial_utf8.n_remain != 0) {
        rejects.push_back(tok);
      }
    }
    return rejects;
  }

  const llama_grammar_element* stack_pos = stack.back();

  std::vector<llama_grammar_candidate> next_candidates;
  next_candidates.reserve(candidates.size());

  for (const auto& tok : candidates) {
    if (*tok.code_points == 0) {
      // reached end of full codepoints in token, reject iff it ended in a
      // partial sequence that cannot satisfy this position in grammar
      if (tok.partial_utf8.n_remain != 0 &&
          !llama_grammar_match_partial_char(stack_pos, tok.partial_utf8)) {
        rejects.push_back(tok);
      }
    } else if (llama_grammar_match_char(stack_pos, *tok.code_points).first) {
      next_candidates.push_back(
          {tok.index, tok.code_points + 1, tok.partial_utf8});
    } else {
      rejects.push_back(tok);
    }
  }

  const auto* stack_pos_after = llama_grammar_match_char(stack_pos, 0).second;

  // update top of stack to next element, if any
  std::vector<const llama_grammar_element*> stack_after(
      stack.begin(), stack.end() - 1);
  if (!llama_grammar_is_end_of_sequence(stack_pos_after)) {
    stack_after.push_back(stack_pos_after);
  }
  std::vector<std::vector<const llama_grammar_element*>> next_stacks;
  llama_grammar_advance_stack(rules, stack_after, next_stacks);

  auto next_rejects =
      llama_grammar_reject_candidates(rules, next_stacks, next_candidates);
  for (const auto& tok : next_rejects) {
    rejects.push_back({tok.index, tok.code_points - 1, tok.partial_utf8});
  }

  return rejects;
}

static bool llama_grammar_detect_left_recursion(
    const std::vector<std::vector<llama_grammar_element>>& rules,
    size_t rule_index,
    std::vector<bool>* rules_visited,
    std::vector<bool>* rules_in_progress,
    std::vector<bool>* rules_may_be_empty) {
  if ((*rules_in_progress)[rule_index]) {
    return true;
  }

  (*rules_in_progress)[rule_index] = true;

  const std::vector<llama_grammar_element>& rule = rules[rule_index];

  // First check if the rule might produce the empty string. This could be done
  // combined with the second step but it's more readable as two steps.
  bool at_rule_start = true;
  for (size_t i = 0; i < rule.size(); i++) {
    if (llama_grammar_is_end_of_sequence(&rule[i])) {
      if (at_rule_start) {
        (*rules_may_be_empty)[rule_index] = true;
        break;
      }
      at_rule_start = true;
    } else {
      at_rule_start = false;
    }
  }

  // Second, recurse into leftmost nonterminals (or next-leftmost as long as the
  // previous nonterminal may be empty)
  bool recurse_into_nonterminal = true;
  for (size_t i = 0; i < rule.size(); i++) {
    if (rule[i].type == LLAMA_GRETYPE_RULE_REF && recurse_into_nonterminal) {
      if (llama_grammar_detect_left_recursion(
              rules,
              (size_t)rule[i].value,
              rules_visited,
              rules_in_progress,
              rules_may_be_empty)) {
        return true;
      }
      if (!((*rules_may_be_empty)[(size_t)rule[i].value])) {
        recurse_into_nonterminal = false;
      }
    } else if (llama_grammar_is_end_of_sequence(&rule[i])) {
      recurse_into_nonterminal = true;
    } else {
      recurse_into_nonterminal = false;
    }
  }

  (*rules_in_progress)[rule_index] = false;
  (*rules_visited)[rule_index] = true;
  return false;
}

struct llama_grammar* llama_grammar_init(
    const llama_grammar_element** rules,
    size_t n_rules,
    size_t start_rule_index) {
  const llama_grammar_element* pos;

  // copy rule definitions into vectors
  std::vector<std::vector<llama_grammar_element>> vec_rules(n_rules);
  for (size_t i = 0; i < n_rules; i++) {
    for (pos = rules[i]; pos->type != LLAMA_GRETYPE_END; pos++) {
      vec_rules[i].push_back(*pos);
    }
    vec_rules[i].push_back({LLAMA_GRETYPE_END, 0});
  }

  // Check for left recursion
  std::vector<bool> rules_visited(n_rules);
  std::vector<bool> rules_in_progress(n_rules);
  std::vector<bool> rules_may_be_empty(n_rules);
  for (size_t i = 0; i < n_rules; i++) {
    if (rules_visited[i]) {
      continue;
    }
    if (llama_grammar_detect_left_recursion(
            vec_rules,
            i,
            &rules_visited,
            &rules_in_progress,
            &rules_may_be_empty)) {
      fprintf(
          stderr,
          "unsupported grammar, left recursion detected for nonterminal at index %zu\n",
          i);
      return nullptr;
    }
  }

  // loop over alternates of start rule to build initial stacks
  std::vector<std::vector<const llama_grammar_element*>> stacks;
  pos = vec_rules[start_rule_index].data();
  do {
    std::vector<const llama_grammar_element*> stack;
    if (!llama_grammar_is_end_of_sequence(pos)) {
      // if alternate is nonempty, add to stack
      stack.push_back(pos);
    }
    llama_grammar_advance_stack(vec_rules, stack, stacks);
    while (!llama_grammar_is_end_of_sequence(pos)) {
      // scan to end of alternate def
      pos++;
    }
    if (pos->type == LLAMA_GRETYPE_ALT) {
      // there's another alternate def of this rule to process
      pos++;
    } else {
      break;
    }
  } while (true);

  // Important: vec_rules has to be moved here, not copied, because stacks
  // contains pointers to elements of vec_rules. If vec_rules were copied into
  // llama_grammar then the pointers would be invalidated when the local
  // vec_rules goes out of scope.
  return new llama_grammar{std::move(vec_rules), std::move(stacks), {}};
}

// Decodes a UTF-8 string which may end in an incomplete sequence. Adds a
// terminating 0 for use as pointer. If an invalid sequence is encountered,
// returns `llama_partial_utf8.n_remain == -1`.
std::pair<std::vector<uint32_t>, llama_partial_utf8> decode_utf8(
    const std::string& src,
    llama_partial_utf8 partial_start) {
  static const int lookup[] = {1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 2, 2, 3, 4};
  const char* pos = src.c_str();
  std::vector<uint32_t> code_points;
  // common english strings have the same number of codepoints and bytes. `+ 1`
  // for the terminating 0.
  code_points.reserve(src.size() + 1);
  uint32_t value = partial_start.value;
  int n_remain = partial_start.n_remain;

  // continue previous decode, if applicable
  while (*pos != 0 && n_remain > 0) {
    uint8_t next_byte = static_cast<uint8_t>(*pos);
    if ((next_byte >> 6) != 2) {
      // invalid sequence, abort
      code_points.push_back(0);
      return std::make_pair(std::move(code_points), llama_partial_utf8{0, -1});
    }
    value = (value << 6) + (next_byte & 0x3F);
    ++pos;
    --n_remain;
  }

  if (partial_start.n_remain > 0 && n_remain == 0) {
    code_points.push_back(value);
  }

  // decode any subsequent utf-8 sequences, which may end in an incomplete one
  while (*pos != 0) {
    uint8_t first_byte = static_cast<uint8_t>(*pos);
    uint8_t highbits = first_byte >> 4;
    n_remain = lookup[highbits] - 1;

    if (n_remain < 0) {
      // invalid sequence, abort
      code_points.clear();
      code_points.push_back(0);
      return std::make_pair(
          std::move(code_points), llama_partial_utf8{0, n_remain});
    }

    uint8_t mask = (1 << (7 - n_remain)) - 1;
    value = first_byte & mask;
    ++pos;
    while (*pos != 0 && n_remain > 0) {
      value = (value << 6) + (static_cast<uint8_t>(*pos) & 0x3F);
      ++pos;
      --n_remain;
    }
    if (n_remain == 0) {
      code_points.push_back(value);
    }
  }
  code_points.push_back(0);

  return std::make_pair(
      std::move(code_points), llama_partial_utf8{value, n_remain});
}

// takes a set of possible pushdown stacks on a grammar, which are required to
// be positioned at a character range (see `llama_grammar_advance_stack`), and
// produces the N possible stacks if the given char is accepted at those
// positions
void llama_grammar_accept(
    const std::vector<std::vector<llama_grammar_element>>& rules,
    const std::vector<std::vector<const llama_grammar_element*>>& stacks,
    const uint32_t chr,
    std::vector<std::vector<const llama_grammar_element*>>& new_stacks) {
  new_stacks.clear();

  for (const auto& stack : stacks) {
    if (stack.empty()) {
      continue;
    }

    auto match = llama_grammar_match_char(stack.back(), chr);
    if (match.first) {
      const llama_grammar_element* pos = match.second;

      // update top of stack to next element, if any
      std::vector<const llama_grammar_element*> new_stack(
          stack.begin(), stack.end() - 1);
      if (!llama_grammar_is_end_of_sequence(pos)) {
        new_stack.push_back(pos);
      }
      llama_grammar_advance_stack(rules, new_stack, new_stacks);
    }
  }
}

Grammar::Grammar(std::string grammar) {
  auto parsed_grammar = grammar_parser::parse(grammar.c_str());

  // will be empty (default) if there are parse errors
  if (parsed_grammar.rules.empty()) {
    throw std::runtime_error("Failed to parse grammar");
  }

  // Ensure that there is a "root" node.
  if (parsed_grammar.symbol_ids.find("root") ==
      parsed_grammar.symbol_ids.end()) {
    throw std::runtime_error("grammar does not contain a 'root' symbol");
  }

  std::vector<const llama_grammar_element*> grammar_rules(
      parsed_grammar.c_rules());

  this->grammar = llama_grammar_init(
      grammar_rules.data(),
      grammar_rules.size(),
      parsed_grammar.symbol_ids.at("root"));

  if (this->grammar == nullptr) {
    throw std::runtime_error("Failed to initialize llama_grammar");
  }
}

template <typename T>
void Grammar::sample_grammar(T* probabilities, const Tokenizer* tokenizer) {
  bool allow_eog = false;
  for (const auto& stack : grammar->stacks) {
    if (stack.empty()) {
      allow_eog = true;
      break;
    }
  }

  std::vector<std::pair<std::vector<uint32_t>, llama_partial_utf8>>
      candidates_decoded;
  candidates_decoded.reserve(tokenizer->vocab_size());

  std::vector<llama_grammar_candidate> candidates_grammar;
  candidates_grammar.reserve(tokenizer->vocab_size());

  for (size_t i = 0; i < tokenizer->vocab_size(); ++i) {
    const uint32_t id = i;

    auto piece_res = tokenizer->decode(id, id);
    ET_CHECK(piece_res.ok());

    const std::string& piece = piece_res.get();

    if (id == tokenizer->eos_tok()) {
      if (!allow_eog) {
        probabilities[i] = -INFINITY;
      }
    } else if (piece.empty() || piece[0] == 0) {
      probabilities[i] = -INFINITY;
    } else {
      candidates_decoded.push_back(decode_utf8(piece, grammar->partial_utf8));
      candidates_grammar.push_back(
          {i,
           candidates_decoded.back().first.data(),
           candidates_decoded.back().second});
    }
  }

  const auto rejects = llama_grammar_reject_candidates(
      grammar->rules, grammar->stacks, candidates_grammar);
  for (const auto& reject : rejects) {
    probabilities[reject.index] = -INFINITY;
  }
}

template void Grammar::sample_grammar<float>(
    float* logits,
    const Tokenizer* tokenizer);
template void Grammar::sample_grammar<exec_aten::Half>(
    exec_aten::Half* logits,
    const Tokenizer* tokenizer);

void Grammar::accept_token(uint32_t token, const Tokenizer* tokenizer) {
  assert(tokenizer != nullptr);

  if (token == tokenizer->eos_tok()) {
    for (const auto& stack : grammar->stacks) {
      if (stack.empty()) {
        return;
      }
    }

    // should never reach here
    assert(false);
  }

  auto piece_res = tokenizer->decode(token, token);
  ET_CHECK(piece_res.ok());

  const std::string& piece = piece_res.get();

  // Note terminating 0 in decoded string
  const auto decoded = decode_utf8(piece, grammar->partial_utf8);
  const auto& code_points = decoded.first;
  std::vector<std::vector<const llama_grammar_element*>> tmp_new_stacks;
  for (auto it = code_points.begin(), end = code_points.end() - 1; it != end;
       ++it) {
    llama_grammar_accept(grammar->rules, grammar->stacks, *it, tmp_new_stacks);
    grammar->stacks = tmp_new_stacks;
  }
  grammar->partial_utf8 = decoded.second;
  assert(!grammar->stacks.empty());
}

} // namespace executor
} // namespace torch
