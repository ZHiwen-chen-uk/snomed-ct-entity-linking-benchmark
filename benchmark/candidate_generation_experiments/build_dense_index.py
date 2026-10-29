"""One-time: encode every SNOMED description form with SapBERT into a memmap.

Reuses candidate_booster.build_synindex (already cached), so this only pays the
encode cost. CPU: ~1-2h for ~665k forms; progress is printed and written
incrementally. Run once, then test_dense_retrieval.py is fast.

    python benchmark/candidate_generation_experiments/build_dense_index.py
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BENCHMARK = HERE.parent
PROJECT = BENCHMARK.parent
sys.path.insert(0, str(BENCHMARK))
sys.path.insert(0, str(HERE))

from candidate_booster import build_synindex
from dense_retriever import build_index

SNOMED_DIR = PROJECT / "SnomedCT_InternationalRF2_PRODUCTION_20260601T120000Z"


def main():
    print("loading syn_index (cached)...", flush=True)
    syn_index = build_synindex(str(SNOMED_DIR))
    print(f"syn_index forms: {len(syn_index)}", flush=True)
    build_index(syn_index, batch_size=128)


if __name__ == "__main__":
    main()
