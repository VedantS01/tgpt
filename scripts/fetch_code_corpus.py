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

Documents inside each file are separated by a NUL byte (`tgpt.data.DOC_SEP`).
The tokenizer does not care — it sees a stream either way — but Milestone 2 needs
to know where one source file ends and the next begins, so it can put an EOS token
there and keep whole documents on one side of the train/val split.
"""

from __future__ import annotations

import io
import os
import tarfile
import urllib.request

from tgpt.data import DOC_SEP

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
    """Write `chunks` into `path`, diverting the first `heldout_mb` to `heldout_path`.

    Each chunk is one document (a source file, a function, an article) and is
    written separated by DOC_SEP, so M2 can recover the boundaries. The held-out
    files get no separators — they are only ever read as plain text for the
    bytes-per-token measurement.
    """
    target, held = target_mb * 1e6, heldout_mb * 1e6
    n_held = n_train = n_docs = 0
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
                if n_docs:
                    ft.write(DOC_SEP)
                ft.write(text)
                n_docs += 1
                n_train += len(text)
                if n_train >= target:
                    break
    print(f"  {path}: {n_train/1e6:.1f} MB, {n_docs:,} documents"
          f"   (held out {n_held/1e6:.2f} MB -> {heldout_path})")


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
    """Take the prose slice from the WikiText-103 corpus fetched in M1.

    `download_corpus` marks article boundaries with DOC_SEP, so this streams whole
    articles. An older corpus.txt written before that convention existed has no
    separators, so it is regenerated rather than silently treated as one document.
    """
    from tgpt.data import DOC_SEP as SEP, download_corpus

    src = "data/corpus.txt"
    if not os.path.exists(src) or SEP not in open(src, encoding="utf-8").read(1 << 24):
        download_corpus(src, source="wikitext-103")

    def articles():
        buf = ""
        with open(src, encoding="utf-8") as f:
            while block := f.read(1 << 20):
                buf += block
                *done, buf = buf.split(SEP)
                yield from done
        if buf:
            yield buf

    _write_stream(
        f"{OUT_DIR}/prose.txt", articles(),
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
