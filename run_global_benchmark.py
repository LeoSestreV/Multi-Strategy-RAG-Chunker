#!/usr/bin/env python3
"""
run_global_benchmark.py
=======================
Exécute toutes les stratégies de chunking en parallèle et génère un rapport
comparatif (CSV + JSON). Non-intrusif : ne modifie aucun fichier dans les
dossiers Strategy_*.

Usage:
    python run_global_benchmark.py                  # lance les pipelines puis agrège
    python run_global_benchmark.py --skip-run       # agrège uniquement les résultats existants
"""

import argparse
import csv
import importlib
import importlib.util
import json
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("benchmark")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent
BIOTXT_DIR = ROOT_DIR / "BioTxt"
BENCHMARK_CSV = ROOT_DIR / "benchmark_results.csv"
BENCHMARK_JSON = ROOT_DIR / "benchmark_summary.json"


# ===================================================================
# 1. Discovery
# ===================================================================
def discover_strategies() -> list[Path]:
    """Return sorted list of Strategy_* directories that contain a main.py."""
    dirs = sorted(
        p for p in ROOT_DIR.iterdir()
        if p.is_dir() and p.name.startswith("Strategy_") and (p / "main.py").exists()
    )
    return dirs


# ===================================================================
# 2. Dynamic import helpers  (isolés pour éviter les conflits config.py)
# ===================================================================
def _load_module_from_path(module_name: str, file_path: Path):
    """Load a Python module from an absolute file path, isolated by name."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


def load_strategy(strategy_dir: Path) -> tuple[Any, Any]:
    """
    Dynamically load the config + Pipeline from a strategy directory.
    Returns (config_instance, pipeline_instance).

    Each module is loaded under a unique namespace to avoid collisions
    between the different config.py / main.py / etc.
    """
    name = strategy_dir.name  # e.g. "Strategy_Semantic"

    # Temporarily insert the strategy dir at the front of sys.path so that
    # relative imports inside main.py (from config import …) resolve correctly.
    str_dir = str(strategy_dir)
    sys.path.insert(0, str_dir)

    try:
        # Load config.py
        config_mod = _load_module_from_path(f"{name}.config", strategy_dir / "config.py")
        # Find the *Config dataclass (only one per file)
        config_cls = None
        for attr_name in dir(config_mod):
            obj = getattr(config_mod, attr_name)
            if isinstance(obj, type) and attr_name.endswith("Config"):
                config_cls = obj
                break
        if config_cls is None:
            raise ImportError(f"No *Config dataclass found in {strategy_dir / 'config.py'}")
        config_instance = config_cls()

        # Fix input_dir / output_dir to be absolute (relative to strategy dir)
        if hasattr(config_instance, "input_dir"):
            config_instance.input_dir = str(
                (strategy_dir / config_instance.input_dir).resolve()
            )
        if hasattr(config_instance, "output_dir"):
            config_instance.output_dir = str(
                (strategy_dir / config_instance.output_dir).resolve()
            )

        # Load preprocessing, embeddings, chunking first (dependencies of main)
        for dep in ("preprocessing", "embeddings", "chunking"):
            dep_path = strategy_dir / f"{dep}.py"
            if dep_path.exists():
                _load_module_from_path(f"{name}.{dep}", dep_path)
                # Also register under the plain name so `from preprocessing import …` works
                sys.modules[dep] = sys.modules[f"{name}.{dep}"]

        # Register config module under plain name for main.py imports
        sys.modules["config"] = config_mod

        # Load main.py
        main_mod = _load_module_from_path(f"{name}.main", strategy_dir / "main.py")
        pipeline = main_mod.Pipeline(config_instance)

        return config_instance, pipeline

    finally:
        # Clean up sys.path and plain-name aliases
        if str_dir in sys.path:
            sys.path.remove(str_dir)
        for plain in ("config", "preprocessing", "embeddings", "chunking", "main"):
            sys.modules.pop(plain, None)


# ===================================================================
# 3. Run a single strategy pipeline
# ===================================================================
def run_strategy(strategy_dir: Path) -> dict:
    """
    Run the pipeline for one strategy on all BioTxt/*.txt files.
    Returns a result dict with timing, output paths, and status.
    """
    name = strategy_dir.name
    log.info("[START]  %s", name)
    t0 = time.perf_counter()

    try:
        config, pipeline = load_strategy(strategy_dir)
    except Exception as exc:
        log.error("[FAIL]   %s — import error: %s", name, exc)
        return {"strategy": name, "status": "error", "error": str(exc), "elapsed": 0.0}

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    txt_files = sorted(BIOTXT_DIR.glob("*.txt"))
    if not txt_files:
        log.warning("[WARN]   %s — no .txt files in %s", name, BIOTXT_DIR)
        return {"strategy": name, "status": "no_input", "elapsed": 0.0}

    results_files: list[str] = []
    for fp in txt_files:
        fname = fp.name
        try:
            raw_text = fp.read_text(encoding="utf-8")
            chunks = pipeline.run_to_dict(raw_text)

            # Build JSON result — mirrors the format of each strategy's run_pipeline.py
            result = {
                "source_file": fname,
                "strategy": _strategy_label(name),
                "timestamp": datetime.now().isoformat(),
                "config": _safe_config_dict(config),
                "total_chunks": len(chunks),
                "chunks": chunks,
            }

            suffix = name.replace("Strategy_", "").lower()
            out_name = f"{fp.stem}_{suffix}.json"
            out_path = output_dir / out_name
            out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
            results_files.append(str(out_path))
            log.info("         %s | %s -> %d chunks", name, fname, len(chunks))

        except Exception as exc:
            log.error("         %s | %s FAILED: %s", name, fname, exc)

    elapsed = time.perf_counter() - t0
    log.info("[DONE]   %s in %.1fs", name, elapsed)

    return {
        "strategy": name,
        "status": "ok",
        "elapsed": elapsed,
        "output_files": results_files,
    }


def _strategy_label(dir_name: str) -> str:
    """Strategy_Semantic -> semantic_chunking, etc."""
    mapping = {
        "Strategy_Semantic": "semantic_chunking",
        "Strategy_Token": "token_based_chunking",
        "Strategy_Recursive": "recursive_character_chunking",
        "Strategy_Sentence": "sentence_based_chunking",
        "Strategy_FixedSize": "fixed_size_chunking",
        "Strategy_Proposition": "proposition_based_chunking",
    }
    return mapping.get(dir_name, dir_name.lower())


def _safe_config_dict(config) -> dict:
    """Convert a config dataclass to dict, handling non-serializable fields."""
    d = asdict(config)
    # Remove internal / path fields
    for key in ("input_dir", "output_dir", "spacy_model", "ollama_base_url"):
        d.pop(key, None)
    # Ensure everything is JSON-serializable
    return {k: (list(v) if isinstance(v, tuple) else v) for k, v in d.items()}


# ===================================================================
# 4. Aggregation — read existing JSON outputs
# ===================================================================
def aggregate_results(strategy_dirs: list[Path]) -> list[dict]:
    """
    Read all JSON outputs from Strategy_*/output/ and compute statistics
    per strategy.
    """
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
    except Exception:
        log.warning("tiktoken unavailable — token counts will be approximated (words * 1.3)")
        enc = None

    all_stats: list[dict] = []

    for sdir in strategy_dirs:
        name = sdir.name
        output_dir = sdir / "output"
        if not output_dir.exists():
            log.warning("No output/ directory for %s — skipping", name)
            continue

        json_files = sorted(output_dir.glob("*.json"))
        if not json_files:
            log.warning("No JSON files in %s/output/ — skipping", name)
            continue

        chunks_all: list[dict] = []
        config_snapshot: dict = {}
        strategy_label = ""

        for jf in json_files:
            try:
                data = json.loads(jf.read_text(encoding="utf-8"))
            except Exception as exc:
                log.error("Failed to read %s: %s", jf, exc)
                continue
            if not config_snapshot:
                config_snapshot = data.get("config", {})
                strategy_label = data.get("strategy", name)
            chunks_all.extend(data.get("chunks", []))

        if not chunks_all:
            continue

        stats = _compute_stats(name, strategy_label, chunks_all, config_snapshot, enc)
        all_stats.append(stats)

    return all_stats


def _compute_stats(
    dir_name: str,
    strategy_label: str,
    chunks: list[dict],
    config: dict,
    enc,
) -> dict:
    """Compute all benchmark statistics for a list of chunks."""
    n = len(chunks)
    texts = [c.get("text", "") for c in chunks]
    lengths = [len(t) for t in texts]

    # --- Volume ---
    total_chunks = n

    # --- Granularity (characters) ---
    max_chunk = max(lengths) if lengths else 0
    min_chunk = min(lengths) if lengths else 0
    avg_chunk = sum(lengths) / n if n else 0

    # --- Contenu: phrases, mots, tokens ---
    sentence_counts = []
    word_counts = []
    token_counts = []

    for c in chunks:
        sents = c.get("sentences", [])
        sentence_counts.append(len(sents))

        words = c.get("text", "").split()
        word_counts.append(len(words))

        if enc is not None:
            token_counts.append(len(enc.encode(c.get("text", ""))))
        else:
            # Fallback approximation
            token_counts.append(int(len(words) * 1.3))

    avg_sentences = sum(sentence_counts) / n if n else 0
    avg_words = sum(word_counts) / n if n else 0
    avg_tokens = sum(token_counts) / n if n else 0

    # --- Qualité: coherence_score ---
    coherence_scores = [c.get("coherence_score", 0.0) for c in chunks]
    avg_coherence = sum(coherence_scores) / n if n else 0

    # --- Densité d'information (Proposition only) ---
    avg_proposition_count = None
    if dir_name == "Strategy_Proposition":
        prop_counts = [c.get("proposition_count", 0) for c in chunks]
        avg_proposition_count = sum(prop_counts) / n if n else 0

    return {
        "strategy": dir_name,
        "strategy_label": strategy_label,
        "config": config,
        # Volume
        "total_chunks": total_chunks,
        # Granularity
        "max_chunk_chars": max_chunk,
        "min_chunk_chars": min_chunk,
        "avg_chunk_chars": round(avg_chunk, 1),
        # Content
        "avg_sentences_per_chunk": round(avg_sentences, 2),
        "avg_words_per_chunk": round(avg_words, 2),
        "avg_tokens_per_chunk": round(avg_tokens, 2),
        # Quality
        "avg_coherence_score": round(avg_coherence, 4),
        # Proposition density
        "avg_proposition_count": (
            round(avg_proposition_count, 2) if avg_proposition_count is not None else None
        ),
    }


# ===================================================================
# 5. Report generation
# ===================================================================
def write_csv(stats: list[dict], path: Path) -> None:
    """Write benchmark_results.csv."""
    fieldnames = [
        "strategy",
        "total_chunks",
        "max_chunk_chars",
        "min_chunk_chars",
        "avg_chunk_chars",
        "avg_sentences_per_chunk",
        "avg_words_per_chunk",
        "avg_tokens_per_chunk",
        "avg_coherence_score",
        "avg_proposition_count",
        "execution_time_s",
    ]

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in stats:
            writer.writerow(row)

    log.info("CSV written to %s", path)


def write_json_summary(stats: list[dict], timings: dict[str, float], path: Path) -> None:
    """Write benchmark_summary.json with per-strategy metadata."""
    summary = {
        "generated_at": datetime.now().isoformat(),
        "input_dir": str(BIOTXT_DIR),
        "strategies": [],
    }

    for s in stats:
        entry = {
            "strategy": s["strategy"],
            "strategy_label": s["strategy_label"],
            "config": s["config"],
            "statistics": {k: v for k, v in s.items()
                          if k not in ("strategy", "strategy_label", "config")},
        }
        if s["strategy"] in timings:
            entry["statistics"]["execution_time_s"] = round(timings[s["strategy"]], 2)
        summary["strategies"].append(entry)

    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("JSON summary written to %s", path)


# ===================================================================
# 6. Pretty-print table to terminal
# ===================================================================
def print_summary_table(stats: list[dict]) -> None:
    """Print a readable comparison table to stdout."""
    if not stats:
        log.warning("No statistics to display.")
        return

    header = (
        f"{'Strategy':<28} {'Chunks':>6} {'MaxCh':>6} {'MinCh':>6} "
        f"{'AvgCh':>7} {'Sents':>6} {'Words':>7} {'Tokens':>7} "
        f"{'Coher':>7} {'Props':>6} {'Time':>7}"
    )
    sep = "-" * len(header)

    print(f"\n{sep}")
    print("  BENCHMARK RESULTS — Multi-Strategy RAG Chunker")
    print(sep)
    print(header)
    print(sep)

    for s in stats:
        props = (
            f"{s['avg_proposition_count']:>6.1f}"
            if s.get("avg_proposition_count") is not None
            else "   n/a"
        )
        time_str = (
            f"{s['execution_time_s']:>6.1f}s"
            if s.get("execution_time_s") is not None
            else "    n/a"
        )
        print(
            f"{s['strategy']:<28} {s['total_chunks']:>6} {s['max_chunk_chars']:>6} "
            f"{s['min_chunk_chars']:>6} {s['avg_chunk_chars']:>7.1f} "
            f"{s['avg_sentences_per_chunk']:>6.2f} {s['avg_words_per_chunk']:>7.1f} "
            f"{s['avg_tokens_per_chunk']:>7.1f} {s['avg_coherence_score']:>7.4f} "
            f"{props} {time_str}"
        )

    print(sep)
    print()


# ===================================================================
# 7. Main orchestration
# ===================================================================
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run all chunking strategies and generate a comparative benchmark."
    )
    parser.add_argument(
        "--skip-run",
        action="store_true",
        help="Skip pipeline execution; only aggregate existing JSON outputs.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=2,
        help="Max parallel strategies to run simultaneously (default: 2).",
    )
    args = parser.parse_args()

    strategy_dirs = discover_strategies()
    if not strategy_dirs:
        log.error("No Strategy_* directories found in %s", ROOT_DIR)
        sys.exit(1)

    log.info("Discovered %d strategies: %s",
             len(strategy_dirs), [d.name for d in strategy_dirs])

    # ------------------------------------------------------------------
    # Phase 1: Execute pipelines (unless --skip-run)
    # ------------------------------------------------------------------
    timings: dict[str, float] = {}

    if not args.skip_run:
        log.info("=== Phase 1: Running pipelines (max_workers=%d) ===", args.max_workers)
        global_t0 = time.perf_counter()

        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            future_to_name = {
                executor.submit(run_strategy, sd): sd.name for sd in strategy_dirs
            }
            for future in as_completed(future_to_name):
                strat_name = future_to_name[future]
                try:
                    result = future.result()
                    if result.get("status") == "ok":
                        timings[strat_name] = result["elapsed"]
                    else:
                        log.warning("Strategy %s finished with status: %s",
                                    strat_name, result.get("status"))
                except Exception as exc:
                    log.error("Strategy %s raised an exception: %s", strat_name, exc)

        global_elapsed = time.perf_counter() - global_t0
        log.info("All pipelines finished in %.1fs", global_elapsed)
    else:
        log.info("--skip-run: skipping pipeline execution")

    # ------------------------------------------------------------------
    # Phase 2: Aggregate results from JSON outputs
    # ------------------------------------------------------------------
    log.info("=== Phase 2: Aggregating results ===")
    stats = aggregate_results(strategy_dirs)

    # Inject execution times into stats
    for s in stats:
        s["execution_time_s"] = round(timings.get(s["strategy"], 0.0), 2) if timings else None

    # ------------------------------------------------------------------
    # Phase 3: Generate reports
    # ------------------------------------------------------------------
    log.info("=== Phase 3: Generating reports ===")
    write_csv(stats, BENCHMARK_CSV)
    write_json_summary(stats, timings, BENCHMARK_JSON)
    print_summary_table(stats)

    log.info("Benchmark complete.")


if __name__ == "__main__":
    main()
