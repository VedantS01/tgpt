"""The model corpus — a declarative registry of sources, and the fetcher for them.

The spec for this project asks the model to train on code plus several
high-quality content sources, all openly available: open code datasets, GitHub,
StackOverflow and similar, Medium-style web writing, Wikipedia, wikiHow and
instruction manuals. This module is that list, made executable.

Every source is one `Source` row: where it comes from, how many bytes of it to
take, and how to turn a dataset record into a document. Adding a source is one
row; changing the mixture is one number. `fetch_all` streams them, so the corpus
size is bounded by the budget rather than by the (often multi-terabyte) dataset.

Conventions that the rest of the pipeline depends on:
  - one file per source in `data/corpus/`, documents separated by DOC_SEP
  - the first ~1 MB of each source is diverted to `data/heldout/` and never
    trained on, so tokenizer compression is always measured on unseen text
  - sources are resumable: a file already at its budget is skipped, because a
    2 GB fetch should not restart from zero because source 14 timed out

**Availability is the hard part, not bandwidth.** Datasets rot: `bigcode/the-stack`
and `nampdn-ai/tiny-codes` are gated behind authentication, and `datasets` 5.0
dropped script-based datasets entirely, which killed `armanc/scientific_papers`
and `codeparrot/github-code`. Every source below was probed before being listed
here, and a source that fails at fetch time is logged and skipped rather than
taking the whole run down with it.
"""

from __future__ import annotations

import io
import os
import tarfile
import time
import urllib.request
from dataclasses import dataclass, field
from typing import Callable, Iterator

from .data import DOC_SEP

OUT_DIR = "data/corpus"
HELDOUT_DIR = "data/heldout"
HELDOUT_MB = 1.0


@dataclass(frozen=True)
class Source:
    name: str                       # -> data/corpus/<name>.txt
    group: str                      # code | wikipedia | help | qa  (for the mixture report)
    budget_mb: float
    dataset: str = ""               # HuggingFace dataset id ("" => a tarball source)
    config: str | None = None
    split: str = "train"
    render: Callable[[dict], str] | None = None   # record -> document text
    note: str = ""


# --- record renderers --------------------------------------------------------
# Each turns one dataset record into one document. Kept as named functions rather
# than lambdas so a failure points at something readable.

def _content(r):  return r["content"]
def _code(r):     return r["code"]
def _text(r):     return r["text"]
def _func(r):     return r["whole_func_string"]


def _wikipedia(r):
    # Keep the title: it is the only thing marking what the article is about, and
    # it gives the model a cheap, consistent "document starts here" pattern.
    return f"{r['title']}\n\n{r['text']}"


def _stackexchange(r):
    return f"{r['INSTRUCTION']}\n\n{r['RESPONSE']}"


def _code_qa(r):
    # Markdown-fenced code inside a natural-language answer — the shape most real
    # programming help takes, and a form the code tokenizer should handle well.
    return f"{r['query']}\n\n{r['answer']}"


def _rosetta(r):
    return f"# {r['task_name']} ({r['language_name']})\n{r['code']}"


# --- the registry ------------------------------------------------------------
# budget_mb is at scale 1.0; `--total-mb` rescales every row proportionally.

SOURCES: list[Source] = [
    # --- code: the bulk of the corpus, and the point of the custom tokenizer ---
    Source("python", "code", 400, "codeparrot/codeparrot-clean", render=_content,
           note="whole Python files, permissively licensed"),
    Source("python_github", "code", 120, "angie-chen55/python-github-code", render=_code,
           note="a second, independently filtered Python scrape"),
    Source("javascript", "code", 160, "code-search-net/code_search_net", "javascript", render=_func),
    Source("java", "code", 80, "code-search-net/code_search_net", "java", render=_func),
    Source("go", "code", 60, "code-search-net/code_search_net", "go", render=_func),
    Source("php", "code", 60, "code-search-net/code_search_net", "php", render=_func),
    Source("ruby", "code", 40, "code-search-net/code_search_net", "ruby", render=_func),
    Source("cpp", "code", 100, "", note="release tarballs of well-known C++ libraries"),
    Source("rosetta", "code", 30, "christopher/rosetta-code", render=_rosetta,
           note="the same task in many languages — cheap breadth"),
    Source("code_qa", "code", 100, "m-a-p/CodeFeedback-Filtered-Instruction", render=_code_qa,
           note="programming questions answered with fenced code"),

    # --- wikipedia ---
    Source("wikipedia", "wikipedia", 600, "wikimedia/wikipedia", "20231101.en", render=_wikipedia,
           note="the full English dump, streamed to budget"),

    # --- help articles, how-tos, instructional writing ---
    Source("wikihow", "help", 160, "HuggingFaceTB/cosmopedia", "wikihow", render=_text,
           note="wikiHow-style how-tos (synthetic; see docs)"),
    Source("openstax", "help", 90, "HuggingFaceTB/cosmopedia", "openstax", render=_text,
           note="open textbook prose"),
    Source("khanacademy", "help", 60, "HuggingFaceTB/cosmopedia", "khanacademy", render=_text,
           note="explanatory teaching material"),
    Source("web_edu", "help", 190, "HuggingFaceFW/fineweb-edu", "sample-10BT", render=_text,
           note="real web writing, filtered for educational value"),

    # --- question-and-answer prose ---
    Source("stackexchange", "qa", 150, "donfu/oa-stackexchange", render=_stackexchange,
           note="StackExchange Q&A as plain text, not HTML"),
]

# Sources deliberately NOT included, so the reasoning survives:
#
#   ccdv/arxiv-summarization  — the only openly-available full-text arXiv corpus,
#       but it is lowercased with LaTeX stripped to `@xcite` markers. The spec
#       requires a case-sensitive model; 70 MB of caseless text works against that
#       more than the domain coverage is worth.
#   HuggingFaceH4/stack-exchange-preferences — richer than the one above, but
#       ships raw `<p>` HTML. donfu's is the same content already cleaned.
#   bigcode/the-stack*, nampdn-ai/tiny-codes — gated, need an authenticated token.
#   armanc/scientific_papers, codeparrot/github-code — script-based datasets,
#       which `datasets` 5.0 removed support for.


# --- C++ from release tarballs ----------------------------------------------
# No open C++ dataset survives the gating/script problems above, so C++ comes
# straight from source releases. Chosen to be stylistically varied: header-only
# template libraries, macro-heavy test frameworks, and plain implementation code.

CPP_REPOS = [
    ("fmtlib/fmt", "10.2.1"),
    ("nlohmann/json", "v3.11.3"),
    ("gabime/spdlog", "v1.13.0"),
    ("catchorg/Catch2", "v3.5.2"),
    ("google/googletest", "v1.14.0"),
    ("google/leveldb", "1.23"),
    ("google/re2", "2023-11-01"),
    ("google/benchmark", "v1.8.3"),
    ("abseil/abseil-cpp", "20230802.1"),
    ("Tencent/rapidjson", "v1.1.0"),
    ("yhirose/cpp-httplib", "v0.14.3"),
    ("leethomason/tinyxml2", "9.0.0"),
    ("USCiLab/cereal", "v1.3.2"),
    ("skypjack/entt", "v3.12.2"),
    ("ThePhD/sol2", "v3.3.0"),
    ("Neargye/magic_enum", "v0.9.5"),
    ("doctest/doctest", "v2.4.11"),
    ("chriskohlhoff/asio", "asio-1-28-0"),
    ("ericniebler/range-v3", "0.12.0"),
    ("google/glog", "v0.6.0"),
    ("gflags/gflags", "v2.2.2"),
    ("jbeder/yaml-cpp", "0.8.0"),
    ("zeux/pugixml", "v1.14"),
    ("nothings/stb", "master"),
]
CPP_EXTS = (".h", ".hpp", ".cc", ".cpp", ".cxx", ".hh", ".ipp")


def _cpp_documents() -> Iterator[str]:
    for repo, tag in CPP_REPOS:
        ref = f"refs/tags/{tag}" if tag != "master" else "refs/heads/master"
        url = f"https://codeload.github.com/{repo}/tar.gz/{ref}"
        try:
            with urllib.request.urlopen(url, timeout=180) as resp:
                blob = resp.read()
        except Exception as e:
            print(f"    skip {repo}@{tag}: {type(e).__name__}")
            continue
        n = 0
        try:
            with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as tar:
                for member in tar.getmembers():
                    if not member.isfile() or not member.name.endswith(CPP_EXTS):
                        continue
                    f = tar.extractfile(member)
                    if f is None:
                        continue
                    try:
                        yield f.read().decode("utf-8")
                        n += 1
                    except UnicodeDecodeError:
                        continue
        except tarfile.TarError as e:
            print(f"    skip {repo}@{tag}: {type(e).__name__}")
            continue
        print(f"    {repo}@{tag}: {n} files")


def _hf_documents(src: Source) -> Iterator[str]:
    """Stream a HuggingFace dataset, retrying once if the connection drops."""
    from datasets import load_dataset

    for attempt in (1, 2):
        try:
            ds = load_dataset(src.dataset, src.config, split=src.split, streaming=True)
            for row in ds:
                try:
                    yield src.render(row)
                except (KeyError, TypeError):
                    continue
            return
        except Exception as e:
            # A mid-stream failure has already yielded most of its budget; the
            # retry starts over, and the byte cap stops it either way.
            print(f"    stream error ({type(e).__name__}: {str(e)[:80]}) "
                  f"— {'retrying' if attempt == 1 else 'giving up on this source'}")
            if attempt == 2:
                return
            time.sleep(5)


def documents(src: Source) -> Iterator[str]:
    return _cpp_documents() if not src.dataset else _hf_documents(src)


# --- fetching ----------------------------------------------------------------


def fetch_source(src: Source, scale: float = 1.0, out_dir: str = OUT_DIR,
                 heldout_dir: str = HELDOUT_DIR, force: bool = False) -> int:
    """Write one source to disk under its byte budget. Returns bytes written."""
    budget = src.budget_mb * scale * 1e6
    path = os.path.join(out_dir, f"{src.name}.txt")
    held_path = os.path.join(heldout_dir, f"{src.name}.txt")

    # Resume: a 2 GB fetch should not restart because source 14 timed out.
    if not force and os.path.exists(path) and os.path.getsize(path) >= budget * 0.98:
        print(f"  {src.name}: already at budget ({os.path.getsize(path)/1e6:.0f} MB) — skipping")
        return os.path.getsize(path)

    held_target = HELDOUT_MB * 1e6
    n_held = n_train = n_docs = 0
    last_report = 0.0
    tmp = path + ".tmp"
    with open(held_path, "w", encoding="utf-8") as fh, open(tmp, "w", encoding="utf-8") as ft:
        for text in documents(src):
            if not text or not text.strip():
                continue
            if not text.endswith("\n"):
                text += "\n"
            if n_held < held_target:
                fh.write(text)
                n_held += len(text)
                continue
            if n_docs:
                ft.write(DOC_SEP)
            ft.write(text)
            n_docs += 1
            n_train += len(text)
            if n_train - last_report > 50e6:
                last_report = n_train
                print(f"    {src.name}: {n_train/1e6:.0f}/{budget/1e6:.0f} MB", flush=True)
            if n_train >= budget:
                break
    os.replace(tmp, path)
    print(f"  {src.name}: {n_train/1e6:.1f} MB, {n_docs:,} documents "
          f"(held out {n_held/1e6:.2f} MB)")
    return n_train


def fetch_all(total_mb: float | None = None, only: list[str] | None = None,
              force: bool = False, workers: int = 4, out_dir: str = OUT_DIR,
              heldout_dir: str = HELDOUT_DIR) -> dict:
    """Fetch every source (or `only` of them), scaled so the total hits `total_mb`.

    Sources are fetched concurrently. This is network-latency-bound rather than
    CPU-bound, so a handful of threads turns a multi-hour sequential download into
    something much shorter; each source writes to its own file, so they never
    contend. A source that fails takes only itself down.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(heldout_dir, exist_ok=True)

    chosen = [s for s in SOURCES if not only or s.name in only]
    if only:
        unknown = set(only) - {s.name for s in SOURCES}
        if unknown:
            raise ValueError(f"unknown source(s): {sorted(unknown)}")
    nominal = sum(s.budget_mb for s in chosen)
    scale = (total_mb / nominal) if total_mb else 1.0
    print(f"fetching {len(chosen)} sources with {workers} workers, "
          f"target {nominal*scale:.0f} MB (scale {scale:.2f})\n", flush=True)

    def run(src: Source) -> int:
        try:
            return fetch_source(src, scale, out_dir, heldout_dir, force)
        except Exception as e:
            print(f"  {src.name}: FAILED ({type(e).__name__}: {str(e)[:120]})", flush=True)
            return 0

    written: dict[str, int] = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(run, s): s for s in chosen}
        for fut in as_completed(futures):
            written[futures[fut].name] = fut.result()
    return written


def report(written: dict[str, int]) -> None:
    by_group: dict[str, int] = {}
    for src in SOURCES:
        if src.name in written:
            by_group[src.group] = by_group.get(src.group, 0) + written[src.name]
    total = sum(by_group.values()) or 1
    print(f"\ncorpus total: {total/1e6:.0f} MB")
    for group, n in sorted(by_group.items(), key=lambda kv: -kv[1]):
        print(f"  {group:<12} {n/1e6:8.0f} MB   {n/total*100:5.1f}%")
    failed = [k for k, v in written.items() if v == 0]
    if failed:
        print(f"  failed/empty: {failed}")
