"""Language data for the code-aware tokenizer (Milestone 1b).

Pure data, no logic: the keyword sets we promise to keep whole, the code idioms we
want as single tokens, and the whitespace ladder. Kept separate from the tokenizer
itself so the *spec* is readable and auditable on its own.

The design spec being served here:
  1. whitespace runs of 1, 3, 7, 11, 15, 19, 23, 27 spaces are tokens
  2. naturally-joined punctuation (");") is one token
  3. operators ==, ===, !=, !== are single tokens
  4. C++/Python/JS idioms handled cleanly
  5. every keyword of those three languages is a single token
  6. snake_case / camelCase / PascalCase identifiers are SPLIT
  7. the tokenizer is case sensitive
"""

from __future__ import annotations

# --- Requirement 1 -----------------------------------------------------------
# Space-run lengths that must exist as single tokens. These are 4n-1, not 4n, and
# that is the whole trick: in a GPT-4-style pre-tokenization regex the clause
# `\s+(?!\S)` stops one space short of a word, because that final space glues to
# the word after it (" def" is one piece). So a 4-space indent pre-tokenizes as
# 3 spaces + " def", an 8-space indent as 7 spaces + " def", and so on. This
# ladder therefore covers indent depths 1..7 at exactly one token each, while
# leaving ordinary prose (where a space always glues to its word) untouched.
SPACE_LADDER = [1, 3, 7, 11, 15, 19, 23, 27]

# --- Requirement 5 -----------------------------------------------------------
PYTHON_KEYWORDS = [
    "False", "None", "True", "and", "as", "assert", "async", "await", "break",
    "class", "continue", "def", "del", "elif", "else", "except", "finally",
    "for", "from", "global", "if", "import", "in", "is", "lambda", "nonlocal",
    "not", "or", "pass", "raise", "return", "try", "while", "with", "yield",
    # soft keywords (contextual, but we want them whole all the same)
    "match", "case", "type",
]

CPP_KEYWORDS = [
    "alignas", "alignof", "and", "and_eq", "asm", "atomic_cancel",
    "atomic_commit", "atomic_noexcept", "auto", "bitand", "bitor", "bool",
    "break", "case", "catch", "char", "char8_t", "char16_t", "char32_t",
    "class", "compl", "concept", "const", "consteval", "constexpr",
    "constinit", "const_cast", "continue", "co_await", "co_return", "co_yield",
    "decltype", "default", "delete", "do", "double", "dynamic_cast", "else",
    "enum", "explicit", "export", "extern", "false", "float", "for", "friend",
    "goto", "if", "inline", "int", "long", "mutable", "namespace", "new",
    "noexcept", "not", "not_eq", "nullptr", "operator", "or", "or_eq",
    "private", "protected", "public", "register", "reinterpret_cast",
    "requires", "return", "short", "signed", "sizeof", "static",
    "static_assert", "static_cast", "struct", "switch", "synchronized",
    "template", "this", "thread_local", "throw", "true", "try", "typedef",
    "typeid", "typename", "union", "unsigned", "using", "virtual", "void",
    "volatile", "wchar_t", "while", "xor", "xor_eq",
]

JS_KEYWORDS = [
    "async", "await", "break", "case", "catch", "class", "const", "continue",
    "debugger", "default", "delete", "do", "else", "enum", "export", "extends",
    "false", "finally", "for", "function", "get", "if", "implements", "import",
    "in", "instanceof", "interface", "let", "new", "null", "of", "package",
    "private", "protected", "public", "return", "set", "static", "super",
    "switch", "this", "throw", "true", "try", "typeof", "undefined", "var",
    "void", "while", "with", "yield",
]


def all_keywords() -> list[str]:
    """Every keyword across the three languages, deduplicated, longest first."""
    seen = dict.fromkeys(PYTHON_KEYWORDS + CPP_KEYWORDS + JS_KEYWORDS)
    return sorted(seen, key=len, reverse=True)


def protected_keywords() -> list[str]:
    """Keywords requirement 6 would otherwise shred.

    Requirement 6 splits identifiers on underscores and on digit boundaries, which
    is exactly what `static_assert`, `co_await` and `char16_t` are made of. Those
    two requirements are in direct conflict, and the conflict is resolved in the
    pre-tokenizer: these strings get their own alternation branch that matches
    before the identifier rules, so they survive whole while `make_shared` — a
    user identifier, not a keyword — still splits.
    """
    return [k for k in all_keywords() if "_" in k or any(c.isdigit() for c in k)]


# --- Requirements 2, 3, 4 ----------------------------------------------------
# Punctuation runs merge on their own (the regex has a "run of punctuation"
# clause), so these mostly emerge from the corpus rather than being forced. The
# list is what we AUDIT for afterwards — the promise we check we kept.
OPERATORS = [
    "==", "===", "!=", "!==", "<=", ">=", "&&", "||", "->", "=>", "::",
    "++", "--", "+=", "-=", "*=", "/=", "%=", "<<", ">>", "**", "//",
    "?.", "??", "...", ":=", "|>", "<=>",
]

# Requirement 2: sequences that are "truly different" glued together.
CLOSERS = [");", "};", "});", "}));", "];", ");\n", "}\n", "()", "[]", "{}"]

# Requirement 4: the per-language gotchas.
PY_IDIOMS = ['"""', "'''", 'f"', "f'", "self", "self.", "__init__", "):", "->"]
CPP_IDIOMS = ["#include", "#define", "#ifndef", "#endif", "#pragma", "::", "<<",
              ">>", "->", "*/", "/*", "//", "&&", "std::"]
JS_IDIOMS = ["```", "${", "=>", "});", "()", "===", "!==", "??", "?."]


def audit_targets() -> dict[str, list[str]]:
    """The full checklist the trained tokenizer is graded against."""
    return {
        "space ladder (req 1)": [" " * n for n in SPACE_LADDER],
        "closers (req 2)": CLOSERS,
        "operators (req 3)": OPERATORS,
        "python idioms (req 4)": PY_IDIOMS,
        "c++ idioms (req 4)": CPP_IDIOMS,
        "js idioms (req 4)": JS_IDIOMS,
        "python keywords (req 5)": PYTHON_KEYWORDS,
        "c++ keywords (req 5)": CPP_KEYWORDS,
        "js keywords (req 5)": JS_KEYWORDS,
    }


# --- Requirement 6 -----------------------------------------------------------
# Identifiers that must NOT survive as one token. Used as a negative test.
MUST_SPLIT = [
    "make_shared", "unique_ptr", "shared_ptr", "value_type", "push_back",
    "getElementById", "addEventListener", "XMLHttpRequest", "parseJSONResponse",
    "HTTPServer", "MyClassName", "snake_case_name", "camelCaseName",
]
