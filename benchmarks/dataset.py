"""Benchmark dataset: clinical trial abstracts with ground truth field values."""
from __future__ import annotations

SAMPLES: list[dict] = [
    {
        "id": "ct_001",
        "text": (
            "A randomized, double-blind, placebo-controlled trial enrolled 312 patients "
            "with moderate-to-severe rheumatoid arthritis. Participants received adalimumab "
            "40mg subcutaneously every two weeks or placebo for 24 weeks. The primary "
            "endpoint was ACR20 response rate at week 24. Adalimumab achieved ACR20 in "
            "62% of patients versus 14% with placebo (p<0.001)."
        ),
        "ground_truth": {
            "drug": "adalimumab",
            "sample_size": "312 patients",
            "outcome": "ACR20 response rate",
            "duration": "24 weeks",
        },
    },
    {
        "id": "ct_002",
        "text": (
            "This phase III study randomized 528 adults with type 2 diabetes to receive "
            "semaglutide 1mg weekly or sitagliptin 100mg daily for 56 weeks. The primary "
            "outcome was change in HbA1c from baseline. Semaglutide reduced HbA1c by "
            "1.5% versus 0.9% for sitagliptin (p<0.0001)."
        ),
        "ground_truth": {
            "drug": "semaglutide",
            "sample_size": "528 adults",
            "outcome": "change in HbA1c",
            "duration": "56 weeks",
        },
    },
    {
        "id": "ct_003",
        "text": (
            "We conducted a multicenter RCT of 180 patients with hypertension. Subjects "
            "were assigned to lisinopril 10mg once daily or hydrochlorothiazide 25mg once "
            "daily for 12 months. The primary endpoint was reduction in systolic blood "
            "pressure at 6 months. Mean SBP reduction was 18.3 mmHg with lisinopril "
            "versus 15.1 mmHg with HCTZ."
        ),
        "ground_truth": {
            "drug": "lisinopril",
            "sample_size": "180 patients",
            "outcome": "reduction in systolic blood pressure",
            "duration": "12 months",
        },
    },
    {
        "id": "ct_004",
        "text": (
            "A single-blind trial of 95 children aged 6-12 with asthma compared fluticasone "
            "propionate inhaler 100mcg twice daily versus salbutamol as-needed over 8 weeks. "
            "Primary outcomes included forced expiratory volume in 1 second (FEV1) and "
            "symptom-free days. Fluticasone improved FEV1 by 12% (p=0.003)."
        ),
        "ground_truth": {
            "drug": "fluticasone propionate",
            "sample_size": "95 children",
            "outcome": "forced expiratory volume in 1 second",
            "duration": "8 weeks",
        },
    },
    {
        "id": "ct_005",
        "text": (
            "Four hundred sixty-two patients with depression were randomized to escitalopram "
            "10mg or placebo for 8 weeks. The Hamilton Depression Rating Scale (HDRS) "
            "score was the primary outcome. Escitalopram led to a mean reduction of "
            "11.2 points on HDRS versus 7.4 for placebo."
        ),
        "ground_truth": {
            "drug": "escitalopram",
            "sample_size": "462 patients",
            "outcome": "Hamilton Depression Rating Scale score",
            "duration": "8 weeks",
        },
    },
    {
        "id": "ct_006",
        "text": (
            "This open-label RCT enrolled 240 patients with Crohn's disease who had failed "
            "anti-TNF therapy. Patients received ustekinumab 260mg IV induction then 90mg "
            "subcutaneously every 8 weeks for 44 weeks. The primary endpoint was clinical "
            "remission at week 44, achieved in 53.1% of patients."
        ),
        "ground_truth": {
            "drug": "ustekinumab",
            "sample_size": "240 patients",
            "outcome": "clinical remission",
            "duration": "44 weeks",
        },
    },
    {
        "id": "ct_007",
        "text": (
            "A controlled trial of 150 adults with insomnia disorder compared zolpidem "
            "10mg versus cognitive behavioral therapy for insomnia (CBT-I) over 6 weeks. "
            "Sleep onset latency (SOL) was the primary measure. CBT-I reduced SOL by "
            "31 minutes versus 20 minutes for zolpidem at week 6."
        ),
        "ground_truth": {
            "drug": "zolpidem",
            "sample_size": "150 adults",
            "outcome": "sleep onset latency",
            "duration": "6 weeks",
        },
    },
    {
        "id": "ct_008",
        "text": (
            "We enrolled 388 patients with metastatic non-small cell lung cancer into a "
            "randomized trial of pembrolizumab 200mg every 3 weeks plus chemotherapy "
            "versus chemotherapy alone for up to 35 cycles. Progression-free survival was "
            "significantly longer in the pembrolizumab arm (median 9.0 vs 4.9 months)."
        ),
        "ground_truth": {
            "drug": "pembrolizumab",
            "sample_size": "388 patients",
            "outcome": "progression-free survival",
            "duration": "35 cycles",
        },
    },
    {
        "id": "ct_009",
        "text": (
            "A double-blind crossover study of 72 healthy volunteers evaluated rosuvastatin "
            "20mg versus atorvastatin 40mg over two 4-week periods with a 2-week washout. "
            "LDL-cholesterol reduction was the primary endpoint. Rosuvastatin reduced LDL "
            "by 49.4% versus 43.2% for atorvastatin."
        ),
        "ground_truth": {
            "drug": "rosuvastatin",
            "sample_size": "72 healthy volunteers",
            "outcome": "LDL-cholesterol reduction",
            "duration": "4-week periods",
        },
    },
    {
        "id": "ct_010",
        "text": (
            "Three hundred patients with chronic migraine were randomized to onabotulinumtoxinA "
            "155 units or placebo injected every 12 weeks for 52 weeks. The primary endpoint "
            "was frequency of headache days per month. Active treatment reduced headache days "
            "by 8.4 per month versus 6.6 for placebo."
        ),
        "ground_truth": {
            "drug": "onabotulinumtoxinA",
            "sample_size": "300 patients",
            "outcome": "frequency of headache days per month",
            "duration": "52 weeks",
        },
    },
]

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
