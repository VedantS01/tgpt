"""Build a mixed code + prose corpus for the code-aware tokenizer (Milestone 1b).

    python -m scripts.fetch_code_corpus

The point of the whole exercise: a tokenizer only ever learns tokens for text it
has actually seen. No regex or flag will produce a `\"\"\"` token or a 4-space
indent token from a corpus of Wikipedia articles. So before tuning anything, the
corpus has to contain real code.

Sources, all openly available without authentication:
  python  codeparrot/codeparrot-clean-valid  (whole files, permissive licenses)
  js      code-search-net/code_search_net    (function bodies)
  c++     release tarballs of well-known MIT/BSL C++ libraries
  prose   the WikiText-103 corpus already fetched in M1

C++ comes from tarballs because the obvious dataset choices are unusable here:
The Stack and StarCoder are gated behind authentication, and codeparrot's
github-code / the-vault are script-based datasets, which `datasets` 5.0 dropped
support for entirely.

Each language is written to its own file, and a held-out slice is kept aside for
measuring bytes-per-token on text the tokenizer never trained on.
"""

from __future__ import annotations

import io
import os
import tarfile
import urllib.request

OUT_DIR = "data/code"
HELDOUT_DIR = "data/heldout"

# (repo, tag) — MIT / BSL / Apache licensed, and stylistically varied on purpose:
# headers, templates, macros, and plain implementation files.
CPP_REPOS = [
    ("fmtlib/fmt", "10.2.1"),
    ("nlohmann/json", "v3.11.3"),
    ("gabime/spdlog", "v1.13.0"),
    ("catchorg/Catch2", "v3.5.2"),
    ("google/googletest", "v1.14.0"),
    ("google/leveldb", "1.23"),
]
CPP_EXTS = (".h", ".hpp", ".cc", ".cpp", ".cxx", ".hh")

# Rough per-language budgets. A tokenizer needs representative statistics, not
# volume — a few tens of MB per language is plenty for a 32k vocabulary.
PY_TARGET_MB = 60
JS_TARGET_MB = 25
PROSE_TARGET_MB = 40
HELDOUT_MB = 1.0


def _write_stream(path: str, chunks, target_mb: float, heldout_path: str, heldout_mb: float):
    """Write `chunks` into `path`, diverting the first `heldout_mb` to `heldout_path`."""
    target, held = target_mb * 1e6, heldout_mb * 1e6
    n_held = n_train = 0
    with open(heldout_path, "w", encoding="utf-8") as fh, open(path, "w", encoding="utf-8") as ft:
        for text in chunks:
            if not text or not text.strip():
                continue
            if not text.endswith("\n"):
                text += "\n"
            if n_held < held:
                fh.write(text)
                n_held += len(text)
            else:
                ft.write(text)
                n_train += len(text)
                if n_train >= target:
                    break
    print(f"  {path}: {n_train/1e6:.1f} MB   (held out {n_held/1e6:.2f} MB -> {heldout_path})")


def fetch_python():
    from datasets import load_dataset

    ds = load_dataset("codeparrot/codeparrot-clean-valid", split="train", streaming=True)
    _write_stream(
        f"{OUT_DIR}/python.txt", (r["content"] for r in ds),
        PY_TARGET_MB, f"{HELDOUT_DIR}/python.txt", HELDOUT_MB,
    )


def fetch_javascript():
    from datasets import load_dataset

    ds = load_dataset("code-search-net/code_search_net", "javascript", split="train", streaming=True)
    _write_stream(
        f"{OUT_DIR}/javascript.txt", (r["whole_func_string"] for r in ds),
        JS_TARGET_MB, f"{HELDOUT_DIR}/javascript.txt", HELDOUT_MB,
    )


def _cpp_files():
    for repo, tag in CPP_REPOS:
        url = f"https://codeload.github.com/{repo}/tar.gz/refs/tags/{tag}"
        print(f"  downloading {repo}@{tag}")
        with urllib.request.urlopen(url, timeout=120) as resp:
            blob = resp.read()
        with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as tar:
            for member in tar.getmembers():
                if not member.isfile() or not member.name.endswith(CPP_EXTS):
                    continue
                f = tar.extractfile(member)
                if f is None:
                    continue
                try:
                    yield f.read().decode("utf-8")
                except UnicodeDecodeError:
                    continue


def fetch_cpp():
    _write_stream(
        f"{OUT_DIR}/cpp.txt", _cpp_files(),
        1e9, f"{HELDOUT_DIR}/cpp.txt", HELDOUT_MB,  # take everything; these repos are small
    )


def fetch_prose():
    src = "data/corpus.txt"
    if not os.path.exists(src):
        from tgpt.data import download_corpus
        download_corpus(src, source="wikitext-103")

    def chunks():
        with open(src, encoding="utf-8") as f:
            while block := f.read(1 << 20):
                yield block

    _write_stream(
        f"{OUT_DIR}/prose.txt", chunks(),
        PROSE_TARGET_MB, f"{HELDOUT_DIR}/prose.txt", HELDOUT_MB,
    )


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(HELDOUT_DIR, exist_ok=True)
    for name, fn in [("python", fetch_python), ("javascript", fetch_javascript),
                     ("c++", fetch_cpp), ("prose", fetch_prose)]:
        print(f"{name}:")
        fn()
    total = sum(os.path.getsize(os.path.join(OUT_DIR, f)) for f in os.listdir(OUT_DIR))
    print(f"\ncorpus total: {total/1e6:.1f} MB in {OUT_DIR}/")


if __name__ == "__main__":
    main()
