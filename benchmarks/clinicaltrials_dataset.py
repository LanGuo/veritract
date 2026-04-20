"""Real-world benchmark dataset from ClinicalTrials.gov API v2.

Downloads completed Phase III/IV RCT records with drug interventions, parses
structured fields as ground truth, then fetches the corresponding PubMed abstract
(via NCT→PMID link) to use as the extraction source text.

Run once to build the cache:
    python benchmarks/clinicaltrials_dataset.py --build

Then use SAMPLES from the module for benchmarking.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import requests

_CACHE_PATH = Path(__file__).parent / "clinicaltrials_cache.json"
_CT_API = "https://clinicaltrials.gov/api/v2/studies"
_PUBMED_FETCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
_PUBMED_SEARCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"


def _fetch_trials(page_size: int = 200) -> list[dict]:
    """Fetch completed Phase 3/4 interventional drug trials from ClinicalTrials.gov."""
    params = {
        "filter.overallStatus": "COMPLETED",
        "query.intr": "DRUG",
        "query.term": "phase 3 randomized",
        "pageSize": page_size,
    }
    resp = requests.get(_CT_API, params=params, timeout=30)
    resp.raise_for_status()
    return resp.json().get("studies", [])


def _parse_trial(study: dict) -> dict | None:
    """Extract structured fields from a ClinicalTrials.gov study record."""
    p = study.get("protocolSection", {})
    id_mod = p.get("identificationModule", {})
    design_mod = p.get("designModule", {})
    arms_mod = p.get("armsInterventionsModule", {})
    outcomes_mod = p.get("outcomesModule", {})
    dates_mod = p.get("statusModule", {})
    refs_mod = p.get("referencesModule") or {}

    nct_id = id_mod.get("nctId", "")
    title = id_mod.get("officialTitle") or id_mod.get("briefTitle", "")

    # Drug name — prefer Drug-type interventions
    interventions = arms_mod.get("interventions", [])
    drug = next(
        (i["name"] for i in interventions if i.get("type", "").upper() == "DRUG"),
        next((i["name"] for i in interventions), None),
    )

    # Enrollment
    enroll = design_mod.get("enrollmentInfo", {})
    sample_size = enroll.get("count")

    # Primary outcome
    primary_outcomes = outcomes_mod.get("primaryOutcomes", [])
    outcome = primary_outcomes[0].get("measure") if primary_outcomes else None

    # Duration: prefer human-readable timeFrame from the primary outcome,
    # fall back to start→completion date range from the status module.
    time_frame = primary_outcomes[0].get("timeFrame", "") if primary_outcomes else ""
    start = dates_mod.get("startDateStruct", {}).get("date", "")
    end = dates_mod.get("completionDateStruct", {}).get("date", "")

    # PMIDs from references
    refs = refs_mod.get("references", [])
    pmids = [
        r.get("pmid") for r in refs
        if r.get("pmid") and r.get("type") in ("RESULT", "BACKGROUND")
    ]

    if not all([drug, sample_size, outcome]):
        return None

    return {
        "nct_id": nct_id,
        "title": title,
        "ground_truth": {
            "drug": drug,
            "sample_size": str(sample_size),
            "outcome": outcome,
            "duration": time_frame or (f"{start} to {end}" if start and end else ""),
        },
        "pmids": pmids[:3],
    }


def _fetch_abstract(pmid: str) -> str | None:
    """Fetch PubMed abstract text for a given PMID."""
    try:
        resp = requests.get(
            _PUBMED_FETCH,
            params={"db": "pubmed", "id": pmid, "retmode": "xml", "rettype": "abstract"},
            timeout=20,
        )
        resp.raise_for_status()
        # Extract AbstractText from XML without requiring lxml
        import re, html
        texts = re.findall(r"<AbstractText[^>]*>(.*?)</AbstractText>", resp.text, re.DOTALL)
        if texts:
            return html.unescape(" ".join(t.strip() for t in texts))
    except Exception:
        pass
    return None


def _nct_to_pmid(nct_id: str) -> str | None:
    """Search PubMed for the NCT ID to find a linked PMID."""
    try:
        resp = requests.get(
            _PUBMED_SEARCH,
            params={"db": "pubmed", "term": nct_id, "retmode": "json", "retmax": 3},
            timeout=15,
        )
        resp.raise_for_status()
        ids = resp.json().get("esearchresult", {}).get("idlist", [])
        return ids[0] if ids else None
    except Exception:
        return None


def build_cache(target: int = 50, out: Path = _CACHE_PATH) -> list[dict]:
    """Download trials, find PubMed abstracts, save cache. Returns samples list."""
    print(f"Fetching trials from ClinicalTrials.gov…")
    studies = _fetch_trials(page_size=300)
    print(f"  Got {len(studies)} raw records")

    parsed = [_parse_trial(s) for s in studies]
    parsed = [p for p in parsed if p is not None]
    print(f"  {len(parsed)} records with drug + sample_size + outcome")

    samples = []
    for trial in parsed:
        if len(samples) >= target:
            break

        # Try PMIDs from references first, then search by NCT ID
        pmids = trial.pop("pmids")
        abstract = None
        used_pmid = None

        for pmid in pmids:
            abstract = _fetch_abstract(pmid)
            if abstract and len(abstract) > 100:
                used_pmid = pmid
                break

        if not abstract:
            pmid = _nct_to_pmid(trial["nct_id"])
            if pmid:
                abstract = _fetch_abstract(pmid)
                if abstract and len(abstract) > 100:
                    used_pmid = pmid

        if not abstract:
            continue

        trial["text"] = abstract
        trial["id"] = trial["nct_id"]
        trial["pmid"] = used_pmid
        samples.append(trial)
        print(f"  [{len(samples):2d}/{target}] {trial['nct_id']} (pmid:{used_pmid}) — {trial['title'][:60]}")
        time.sleep(0.35)  # NCBI rate limit: 3 req/s without API key

    out.write_text(json.dumps(samples, indent=2))
    print(f"\nSaved {len(samples)} samples to {out}")
    return samples


def load_cache(path: Path = _CACHE_PATH) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(
            f"Cache not found at {path}. Run:\n"
            "  python benchmarks/clinicaltrials_dataset.py --build"
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
        "duration": {"type": "string"},
    },
    "required": ["drug", "sample_size", "outcome", "duration"],
}

FIELDS = list(SCHEMA["properties"].keys())


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--target", type=int, default=50)
    parser.add_argument("--out", default=str(_CACHE_PATH))
    args = parser.parse_args()
    if args.build:
        build_cache(target=args.target, out=Path(args.out))
    else:
        samples = load_cache(Path(args.out))
        print(f"{len(samples)} samples in cache")
        print(json.dumps(samples[0], indent=2))
