"""Build the code package that goes in with the dissertation.

The package copies the repository's directory layout instead of grouping scripts
by topic, because each script finds its imports with `HERE.parent` and `sys.path`
insertions. Flattening the tree would stop all of them running.

It contains the experiment, analysis and figure scripts the dissertation draws
on, plus the local modules they import. Those are worked out here as an import
closure, so nothing needed goes missing and nothing unused comes along. It also
contains the aggregate result tables the scripts wrote, which are what the figure
script reads and where the numbers in the write-up come from, together with a
README, a manifest with one SHA-256 per file, and the package versions used.

Files carrying corpus content stay out. MIMIC-IV comes under a PhysioNet
credentialed data-use agreement, so note text, and records with one row per
mention, cannot be redistributed. The README lists each of them next to the
command that regenerates it.

    python build_reproducibility_package.py
"""
import ast
import hashlib
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BM = ROOT / "benchmark"
CGE = BM / "candidate_generation_experiments"
OUT = ROOT / "reproducibility_package"

# Directories searched when resolving a local import.
SEARCH = [BM, CGE, BM / "group_a_transformer", CGE / "stage0_ner",
          BM / "replication_bc5cdr"]

# Scripts to package. Everything the dissertation's numbers, tables or figures
# come from, plus the exploratory runs it does not report (labelled in README).
SEEDS = [
    CGE / p for p in [
        # stage 1, candidate retrieval
        "build_dense_index.py", "dense_retriever.py", "retrieval_fusion.py",
        "test_candidate_generation.py", "test_dense_retrieval.py",
        "test_rrf_fusion.py", "abbrev_expander.py", "test_abbrev_expansion.py",
        "abbrev_full_scale.py", "colab_full_candidate_eval.py",
        "run_full_local.py", "grf_over_dense_test.py",
        # stage 2, candidate ranking
        "stage2_gated.py", "stage2_rank_over_rrf.py", "stage2_reranker.py",
        "stage2_full_reranker.py", "stage2_reranker_full.py",
        "stage2_unified.py", "stage2_feature_hygiene.py",
        "stage2_lono_training.py", "stage2_index_dose_response.py",
        "stage3_ontology_constraint.py",
        "stage2_export_crossencoder_data.py", "colab_crossencoder.py",
        "stage2_export_llm_tail.py", "stage2_llm_tail.py",
        "colab_llm_tail_ranker.py",
        # end to end and the official metric
        "stage0_chain_e2e.py", "stage0_chain_e2e_lono.py",
        "stage0_challenge_metric.py",
        # analysis instruments
        "stage_audit_fullscale.py", "stage_oracle_substitution.py",
        "stage_oracle_substitution_lono.py", "stage2_error_taxonomy.py",
        "stage2_bootstrap_ci.py", "stage2_abstention.py",
        "stage_error_examples.py", "dataset_stats.py",
        # figures
        "make_figures.py", "make_fig_dataset_split.py",
    ]
] + [
    CGE / "stage0_ner" / p for p in
    ["build_bio_data.py", "ner_common.py", "run_ner.py"]
] + [
    # the semantic-type predictor: its inference wrapper arrives through the
    # import closure, but the scripts that built and trained it do not
    BM / "group_a_transformer" / p for p in
    ["build_semtype_dataset.py", "train_transformer_semtype.py"]
] + [
    BM / "replication_bc5cdr" / p for p in
    ["replicate_blind_spot.py", "probe_determinism.py", "diag_nondet.py"]
]

EXTRA_FILES = [
    CGE / "run_fullscale_chain.sh",
    CGE / "COLAB_CANDIDATE_RUNBOOK.md",
    CGE / "COLAB_CROSSENCODER_RUNBOOK.md",
    CGE / "COLAB_LLM_TAIL_RUNBOOK.md",
]

# Result tables to ship. Aggregate metrics only.
RESULT_DIRS = [CGE / "stage2_out", CGE / "stage0_ner" / "result",
               CGE]                       # the stage-1 retrieval tables sit here
# .json carries the span-detection metrics; the predicted-span .jsonl files are
# excluded by name below, since those are per-note derived annotations.
RESULT_SUFFIXES = {".csv", ".md", ".json"}

REQUIREMENTS = """numpy==1.20.1
pandas==1.2.4
scikit-learn==0.24.1
scipy==1.6.2
torch==2.4.1
transformers==4.30.2
matplotlib==3.5.1
"""

# Files withheld under the data-use agreement: these carry note text, or one row
# per mention of a MIMIC-IV note.
WITHHELD = {
    "stage_error_examples_all.csv", "stage_error_examples_shortlist.csv",
    "STAGE_ERROR_EXAMPLES.md", "abbrev_acronym_slice_misses.csv",
    "abbrev_new_gold_examples.csv", "dense_new_gold_examples.csv",
    "hybrid_candidate_details.csv",
    "stage0_pred_spans_vanilla.jsonl", "stage0_pred_spans_section_aware.jsonl",
    "GRF_OVER_DENSE_RESULTS.md",
}


def local_imports(path):
    """Module names imported by `path` that resolve to a file in SEARCH."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except SyntaxError:
        return []
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names.add(node.module.split(".")[0])
    out = []
    for n in names:
        for d in SEARCH:
            cand = d / f"{n}.py"
            if cand.exists():
                out.append(cand)
                break
    return out


def closure(seeds):
    """Seeds plus every local module reachable from them."""
    seen, stack = set(), list(seeds)
    while stack:
        p = stack.pop()
        p = p.resolve()
        if p in seen or not p.exists():
            continue
        seen.add(p)
        stack.extend(local_imports(p))
    return sorted(seen)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    missing = [p for p in SEEDS + EXTRA_FILES if not p.exists()]
    if missing:
        print("MISSING seeds, fix the list before shipping:")
        for m in missing:
            print("   ", m.relative_to(ROOT))
        return 1

    if OUT.exists():
        shutil.rmtree(OUT)

    copied = []
    for src in closure(SEEDS) + [p.resolve() for p in EXTRA_FILES]:
        rel = src.relative_to(ROOT)
        dst = OUT / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        copied.append(rel)

    withheld_found = []
    for d in RESULT_DIRS:
        for src in sorted(d.iterdir()):
            if not src.is_file():
                continue
            # dotfile caches are keyed by note id and character offset
            if src.name.startswith("."):
                continue
            if src.name in WITHHELD:
                withheld_found.append(src.relative_to(ROOT))
                continue
            if src.suffix not in RESULT_SUFFIXES:
                continue
            rel = src.relative_to(ROOT)
            dst = OUT / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            if src.suffix == ".md":
                text = src.read_text(encoding="utf-8")
                dst.write_text(text.replace(str(ROOT), "<project root>"),
                               encoding="utf-8")
            else:
                shutil.copy2(src, dst)
            copied.append(rel)

    # ship the README, the environment and this builder, so that the package is
    # entirely regenerable and documents itself
    readme = ROOT / "reproducibility_README.md"
    if not readme.exists():
        print(f"MISSING {readme.name}, the package needs its README")
        return 1
    shutil.copy2(readme, OUT / "README.md")
    (OUT / "requirements.txt").write_text(REQUIREMENTS, encoding="utf-8")
    shutil.copy2(Path(__file__), OUT / Path(__file__).name)
    copied += [Path("README.md"), Path("requirements.txt"),
               Path(Path(__file__).name)]

    code = [c for c in copied if c.suffix == ".py"]
    print(f"packaged {len(copied)} files ({len(code)} python) "
          f"into {OUT.relative_to(ROOT)}")
    print(f"withheld under the data-use agreement: {len(withheld_found)}")
    for w in withheld_found:
        print("   ", w.name)

    lines = ["path,bytes,sha256"]
    total = 0
    for rel in sorted(set(copied)):
        p = OUT / rel
        total += p.stat().st_size
        lines.append(f"{rel.as_posix()},{p.stat().st_size},{sha256(p)}")
    (OUT / "MANIFEST.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"manifest written, {total/1024:.0f} KB of content")

    archive = shutil.make_archive(str(ROOT / OUT.name), "zip",
                                  root_dir=ROOT, base_dir=OUT.name)
    print(f"archive {Path(archive).name}, "
          f"{Path(archive).stat().st_size/1024:.0f} KB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
