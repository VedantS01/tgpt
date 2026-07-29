"""Measure tokenizers against each other and against the design spec (Milestone 1b).

    python -m scripts.eval_tokenizer

Three things get reported, because "is this tokenizer better" is three questions:

  1. COMPRESSION — bytes per token on held-out text, per language. Higher is better.
     This is the number that matters: it is effective context length and inference
     cost per unit of code, and it is what a bad tokenizer silently taxes you on.
  2. SPEC COMPLIANCE — for each requirement, the fraction of the promised strings
     that really are single tokens. Promises, checked rather than assumed.
  3. ROUND-TRIP — decode(encode(x)) == x on deliberately nasty inputs.
"""

from __future__ import annotations

import os

from tgpt import langspec
from tgpt.code_tokenizer import CodeTokenizer
from tgpt.tokenizer import Tokenizer

HELDOUT_DIR = "data/heldout"

NASTY = [
    "def f():\n\treturn 1  # tab indent\n",
    "x = 1\r\nif x:\r\n    pass\r\n",                       # CRLF
    's = "he said \\"hi\\"" + \'nested \\\'quotes\\\'\'',   # nested quotes
    "emoji = '🙂🇯🇵' # 日本語 in a string literal",
    "    \t  mixed = 'tabs and spaces'",
    "a===b!==c&&d||e??f?.g",
    "#include <vector>\ntemplate<typename T> struct A { T* p; };",
    "`tmpl ${a.b(c)} ${'x'}`",
    "",
    " ",
    "\n\n\n",
]


def compression(tok, name: str) -> dict[str, float]:
    out = {}
    for fname in sorted(os.listdir(HELDOUT_DIR)):
        path = os.path.join(HELDOUT_DIR, fname)
        text = open(path, encoding="utf-8").read()
        n = len(tok.encode(text))
        out[fname.replace(".txt", "")] = len(text.encode("utf-8")) / max(n, 1)
    return out


def compliance(tok) -> list[tuple[str, int, int, list[str]]]:
    rows = []
    for label, targets in langspec.audit_targets().items():
        is_ladder = "ladder" in label
        ok, bad = 0, []
        for t in targets:
            # Keywords and operators realistically appear with a leading space
            # (" return", " => "), and that glued form is what the tokenizer needs
            # to be efficient at — so either form counts. Whitespace runs are the
            # exception: for those the bare form IS the claim being made.
            forms = [t] if is_ladder else [t, " " + t]
            if any(len(tok.encode(f)) == 1 for f in forms):
                ok += 1
            else:
                bad.append(t)
        rows.append((label, ok, len(targets), bad))
    return rows


def must_split(tok) -> tuple[int, list[str]]:
    """Requirement 6: these identifiers must NOT be a single token."""
    bad = [s for s in langspec.MUST_SPLIT if len(tok.encode(s)) == 1]
    return len(langspec.MUST_SPLIT) - len(bad), bad


def roundtrip(tok) -> list[str]:
    return [s for s in NASTY if tok.decode(tok.encode(s)) != s]


def main():
    toks: list[tuple[str, object]] = []
    sp = "tokenizer/tgpt.model"
    if os.path.exists(sp):
        toks.append(("M1 sentencepiece/wiki 16k", Tokenizer(sp)))
    for f in sorted(os.listdir("tokenizer")):
        if f.startswith("tgpt-code-") and f.endswith(".json"):
            toks.append((f"M1b bytelevel/code {f[10:-5]}", CodeTokenizer(f"tokenizer/{f}")))

    print("=" * 78)
    print("1. COMPRESSION — bytes per token on held-out text (higher is better)")
    print("=" * 78)
    langs = sorted(x.replace(".txt", "") for x in os.listdir(HELDOUT_DIR))
    print(f"{'tokenizer':<30}" + "".join(f"{l:>11}" for l in langs))
    base = None
    for name, tok in toks:
        c = compression(tok, name)
        if base is None:
            base = c
        print(f"{name:<30}" + "".join(f"{c[l]:>11.2f}" for l in langs))
    if len(toks) > 1:
        last = compression(toks[-1][1], "")
        print(f"{'  vs baseline':<30}" + "".join(
            f"{(last[l]/base[l]-1)*100:>+10.0f}%" for l in langs))

    for name, tok in toks:
        print("\n" + "=" * 78)
        print(f"2. SPEC COMPLIANCE — {name}")
        print("=" * 78)
        for label, ok, total, bad in compliance(tok):
            mark = "PASS" if ok == total else "    "
            print(f"  {mark} {label:<28} {ok:>3}/{total:<3}"
                  + (f"   missing: {bad[:6]}" if bad else ""))
        ok6, bad6 = must_split(tok)
        print(f"  {'PASS' if not bad6 else '    '} identifiers split (req 6)   "
              f"{ok6:>3}/{len(langspec.MUST_SPLIT):<3}"
              + (f"   NOT split: {bad6}" if bad6 else ""))
        fails = roundtrip(tok)
        print(f"  {'PASS' if not fails else 'FAIL'} round-trip (nasty inputs) "
              f"  {len(NASTY)-len(fails):>3}/{len(NASTY):<3}"
              + (f"   failed: {fails}" if fails else ""))

    print("\n" + "=" * 78)
    print("3. WORKED EXAMPLE")
    print("=" * 78)
    sample = "class Foo:\n    def make_shared(self, xs):\n        return [x*2 for x in xs]\n"
    for name, tok in toks:
        ids = tok.encode(sample)
        pieces = tok.pieces(sample) if hasattr(tok, "pieces") else tok.sp.encode(sample, out_type=str)
        print(f"\n{name}  ({len(ids)} tokens)\n  {pieces}")


if __name__ == "__main__":
    main()
