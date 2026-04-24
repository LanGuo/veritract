"""EBM-NLP benchmark dataset from Nye et al. 2018.

Downloads the EBM-NLP 2.0 corpus (Nye et al., 2018), which contains 4993
clinical trial abstracts annotated with P/I/O spans by crowdworkers and medical
experts. Uses the expert-annotated (gold) test set — 183 abstracts with all
three elements.

Field mapping:
  interventions  hierarchical label 3 (Pharmacological) → drug
  participants   hierarchical label 3 (Sample-size)     → sample_size
  outcomes       starting_spans (first span)             → outcome

Source text is the token-joined abstract so GT spans are guaranteed verbatim
matches, making grounding evaluation meaningful.

Run once to build the cache:
    python benchmarks/ebmnlp_dataset.py --build

Reference:
    Nye et al. (2018). A corpus with multi-level annotations of patients,
    interventions and outcomes to support language processing for medical
    literature. ACL 2018.
"""
from __future__ import annotations

import io
import json
import sys
import tarfile
import urllib.request
from pathlib import Path

_CACHE_PATH = Path(__file__).parent / "ebmnlp_cache.json"
_GITHUB_TAR = "https://github.com/bepnye/EBM-NLP/archive/refs/heads/master.tar.gz"

# Hierarchical label values used for GT extraction
_SAMPLE_SIZE_LABEL = 3   # participants hierarchical: "Sample-size"
_DRUG_LABEL = 3          # interventions hierarchical: "Pharmacological"


def _get_spans(tokens: list[str], labels: list[int], target: int | None = None) -> list[str]:
    """Extract contiguous labeled token spans.

    If target is None, any non-zero label is a match.
    Returns list of space-joined span strings.
    """
    spans: list[str] = []
    cur: list[str] = []
    for tok, lab in zip(tokens, labels):
        match = (lab != 0) if target is None else (lab == target)
        if match:
            cur.append(tok)
        elif cur:
            spans.append(" ".join(cur))
            cur = []
    if cur:
        spans.append(" ".join(cur))
    return spans


def _read_member(tf: tarfile.TarFile, members: dict, path: str) -> str | None:
    m = members.get(path)
    if not m:
        return None
    f = tf.extractfile(m)
    return f.read().decode().strip() if f else None


def _load_inner_tar() -> tuple[tarfile.TarFile, dict]:
    """Download GitHub archive and return inner ebm_nlp_2_00 tarfile + member index."""
    print("Downloading EBM-NLP from GitHub…")
    resp = urllib.request.urlopen(_GITHUB_TAR, timeout=120)
    outer_data = resp.read()
    print(f"  Downloaded {len(outer_data) / 1024 / 1024:.1f} MB")

    outer = tarfile.open(fileobj=io.BytesIO(outer_data))
    inner_bytes = outer.extractfile("EBM-NLP-master/ebm_nlp_2_00.tar.gz").read()
    inner = tarfile.open(fileobj=io.BytesIO(inner_bytes))
    members = {m.name: m for m in inner.getmembers()}
    return inner, members


def build_cache(out: Path = _CACHE_PATH) -> list[dict]:
    """Parse EBM-NLP gold test set, save cache. Returns samples list."""
    inner, members = _load_inner_tar()
    names = list(members.keys())

    # Find PMIDs that have all required gold test annotations
    def pmid_set(pattern: str) -> set[str]:
        return {
            n.split("/")[-1].split(".")[0]
            for n in names
            if pattern in n and n.endswith(".ann")
        }

    all_pmids = (
        pmid_set("starting_spans/participants/test/gold")
        & pmid_set("starting_spans/interventions/test/gold")
        & pmid_set("starting_spans/outcomes/test/gold")
        & pmid_set("hierarchical_labels/participants/test/gold")
        & pmid_set("hierarchical_labels/interventions/test/gold")
    )
    print(f"  {len(all_pmids)} PMIDs with complete gold test annotations")

    def read(path: str) -> str | None:
        return _read_member(inner, members, path)

    samples = []
    for pmid in sorted(all_pmids):
        tokens_raw = read(f"ebm_nlp_2_00/documents/{pmid}.tokens")
        if not tokens_raw:
            continue
        tokens = tokens_raw.split("\n")
        src_text = " ".join(tokens)

        def get_labels(phase: str, element: str) -> list[int] | None:
            raw = read(
                f"ebm_nlp_2_00/annotations/aggregated/{phase}/{element}/test/gold/{pmid}.AGGREGATED.ann"
            )
            if not raw:
                return None
            return [int(x) for x in raw.split("\n")]

        hl_p = get_labels("hierarchical_labels", "participants")
        hl_i = get_labels("hierarchical_labels", "interventions")
        ss_o = get_labels("starting_spans", "outcomes")

        if not (hl_p and hl_i and ss_o):
            continue

        sample_size_spans = _get_spans(tokens, hl_p, target=_SAMPLE_SIZE_LABEL)
        drug_spans = _get_spans(tokens, hl_i, target=_DRUG_LABEL)
        # Fall back to any intervention if no pharmacological span found
        if not drug_spans:
            ss_i = get_labels("starting_spans", "interventions")
            if ss_i:
                drug_spans = _get_spans(tokens, ss_i)
        outcome_spans = _get_spans(tokens, ss_o)

        if not (sample_size_spans and drug_spans and outcome_spans):
            continue

        samples.append({
            "id": pmid,
            "pmid": pmid,
            "text": src_text,
            "ground_truth": {
                "drug": drug_spans[0],
                "sample_size": sample_size_spans[0],
                "outcome": outcome_spans[0],
            },
            "all_gt_spans": {
                "drug": drug_spans,
                "sample_size": sample_size_spans,
                "outcome": outcome_spans,
            },
        })

    print(f"  {len(samples)} samples with all three fields populated")
    out.write_text(json.dumps(samples, indent=2))
    print(f"Saved {len(samples)} samples to {out}")
    return samples


def load_cache(path: Path = _CACHE_PATH) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(
            f"Cache not found at {path}. Run:\n"
            "  python benchmarks/ebmnlp_dataset.py --build"
        )
    return json.loads(path.read_text())


def get_samples() -> list[dict]:
    return load_cache()


SCHEMA = {
    "type": "object",
    "properties": {
        "drug": {"type": "string"},
        "sample_size": {"type": "string"},
        "outcome": {"type": "string"},
    },
    "required": ["drug", "sample_size", "outcome"],
}

FIELDS = list(SCHEMA["properties"].keys())


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--out", default=str(_CACHE_PATH))
    args = parser.parse_args()

    if args.build:
        build_cache(out=Path(args.out))
    else:
        samples = load_cache(Path(args.out))
        print(f"{len(samples)} samples in cache")
        s = samples[0]
        print(f"\nPMID: {s['pmid']}")
        print(f"Text (first 300): {s['text'][:300]}")
        print(f"Ground truth: {json.dumps(s['ground_truth'], indent=2)}")
        print(f"All GT spans: {json.dumps(s['all_gt_spans'], indent=2)}")
