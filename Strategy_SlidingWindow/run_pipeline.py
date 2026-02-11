import json
import os
import glob
import time
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

from config import SlidingWindowConfig
from main import Pipeline


def process_file(filepath: str, pipeline: Pipeline, config: SlidingWindowConfig,
                 output_dir: str) -> str:
    filename = os.path.basename(filepath)
    t0 = time.perf_counter()

    with open(filepath, "r", encoding="utf-8") as f:
        raw_text = f.read()

    chunks = pipeline.run_to_dict(raw_text)

    result = {
        "source_file": filename,
        "strategy": "sliding_window_chunking",
        "timestamp": datetime.now().isoformat(),
        "config": {
            "embedding_model": config.embedding_model,
            "window_size": config.window_size,
            "overlap_ratio": config.overlap_ratio,
            "step_size": int(config.window_size * (1 - config.overlap_ratio)),
        },
        "total_chunks": len(chunks),
        "chunks": chunks
    }

    out_name = f"{os.path.splitext(filename)[0]}_sliding.json"
    out_path = os.path.join(output_dir, out_name)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    elapsed = time.perf_counter() - t0
    return f"  {filename} -> {len(chunks)} chunks in {elapsed:.1f}s -> {out_name}"


def main():
    config = SlidingWindowConfig()
    pipeline = Pipeline(config)

    input_dir = os.path.abspath(config.input_dir)
    output_dir = os.path.abspath(config.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    txt_files = sorted(glob.glob(os.path.join(input_dir, "*.txt")))

    if not txt_files:
        print(f"No .txt files found in {input_dir}")
        return

    print(f"Found {len(txt_files)} file(s) in {input_dir}")
    t0 = time.perf_counter()

    with ThreadPoolExecutor(max_workers=config.max_workers) as executor:
        futures = [
            executor.submit(process_file, fp, pipeline, config, output_dir)
            for fp in txt_files
        ]
        for future in futures:
            print(future.result())

    elapsed = time.perf_counter() - t0
    print(f"\nDone in {elapsed:.1f}s.")


if __name__ == "__main__":
    main()
