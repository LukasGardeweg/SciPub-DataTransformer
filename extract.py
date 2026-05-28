"""
Reads PDFs listed in data/metadata.json, extracts structured data via Groq API,
and writes one CSV row per outcome variable x group x timepoint to data/dataset.csv.

Resumes from where it left off if interrupted (progress tracked in data/processed.json).

Setup:
    pip install groq pdfplumber
    Add to .env:  GROQ_API_KEY=gsk_...  (free at console.groq.com)

Usage:
    python extract.py
    python extract.py --doi 10.xxxx/some-doi   # process single paper by DOI
"""

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import fitz  # pymupdf
from groq import Groq

# ── Config ──────────────────────────────────────────────────────────────────
MODEL         = "llama-3.3-70b-versatile"
MAX_PDF_CHARS = 20_000  # Groq free tier limit (~12k tokens input); results section is prioritised

DATA_DIR      = Path("data")
PDF_DIR       = DATA_DIR / "pdfs"
META_FILE     = DATA_DIR / "metadata.json"
OUTPUT_CSV    = DATA_DIR / "dataset.csv"
PROGRESS_FILE = DATA_DIR / "processed.json"
# ────────────────────────────────────────────────────────────────────────────

CSV_COLUMNS = [
    # study-level (repeated per row)
    "doi", "title", "year", "journal", "authors",
    "study_design", "n_total", "n_intervention", "n_control",
    "duration_weeks", "intervention_type", "intervention_description",
    "frequency_per_week", "intensity_description",
    "population_description", "age_mean", "age_sd", "sex", "fitness_level",
    # outcome-level
    "outcome_variable", "outcome_unit", "measurement_timepoint", "group_label",
    "mean_value", "sd_value",
    "pre_mean", "pre_sd", "post_mean", "post_sd",
    "between_group_p_value", "within_group_p_value",
    "effect_size_value", "effect_size_type",
    "notes", "source_pdf",
]

SYSTEM_PROMPT = """\
You are a scientific data extraction assistant specializing in sports science intervention studies.

Extract structured data from the paper and return ONLY valid JSON with this exact structure:

{
  "study": {
    "doi": string or null,
    "title": string,
    "year": integer or null,
    "journal": string or null,
    "authors": string,
    "study_design": string,
    "n_total": integer or null,
    "n_intervention": integer or null,
    "n_control": integer or null,
    "duration_weeks": number or null,
    "intervention_type": string,
    "intervention_description": string,
    "frequency_per_week": number or null,
    "intensity_description": string or null,
    "population_description": string,
    "age_mean": number or null,
    "age_sd": number or null,
    "sex": string or null,
    "fitness_level": string or null
  },
  "outcomes": [
    {
      "outcome_variable": string,
      "outcome_unit": string or null,
      "measurement_timepoint": string,
      "group_label": string,
      "mean_value": number or null,
      "sd_value": number or null,
      "pre_mean": number or null,
      "pre_sd": number or null,
      "post_mean": number or null,
      "post_sd": number or null,
      "between_group_p_value": number or null,
      "within_group_p_value": number or null,
      "effect_size_value": number or null,
      "effect_size_type": string or null,
      "notes": string
    }
  ]
}

Rules:
- Create one "outcomes" entry per outcome_variable x group_label x measurement_timepoint combination
- Focus on the PRIMARY reported outcomes first; include secondary ones only if space permits
- Limit to a maximum of 30 outcome entries total
- Use null for any value not explicitly stated -- never invent or interpolate numbers
- Do not calculate effect sizes or p-values yourself; only report what the paper states
- Return ONLY the JSON object, no markdown fences, no explanation text\
"""


def load_env() -> None:
    env_file = Path(__file__).parent / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


def extract_pdf_text(pdf_path: Path) -> str:
    doc   = fitz.open(str(pdf_path))
    pages = [page.get_text() for page in doc]
    full  = "\n".join(pages)
    lower = full.lower()

    # always keep the opening (~4000 chars): abstract, participants, study design
    head = full[:4000]

    # find results / first table and take as much as possible
    results_start = len(full)
    for marker in ("results", "table 1", "▶table"):
        idx = lower.find(marker)
        if idx != -1 and idx < results_start:
            results_start = idx

    results_part = full[results_start : results_start + (MAX_PDF_CHARS - 4000)]

    return head + "\n\n[...]\n\n" + results_part


def repair_truncated_json(raw: str) -> dict | None:
    """Try to recover a truncated JSON by salvaging complete outcome entries."""
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    # find last complete outcome object
    last_close = raw.rfind("},")
    if last_close == -1:
        last_close = raw.rfind("}")
    if last_close == -1:
        return None
    truncated = raw[:last_close + 1]
    # close open arrays/objects
    for closing in ("]}", "]}"):
        try:
            return json.loads(truncated + closing)
        except json.JSONDecodeError:
            pass
    return None


def call_groq(client: Groq, text: str, doi: str) -> dict | None:
    try:
        response = client.chat.completions.create(
            model=MODEL,
            temperature=0,
            max_tokens=4096,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Extract data from this paper (DOI: {doi}):\n\n{text}"},
            ],
        )
        raw = response.choices[0].message.content.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        result = repair_truncated_json(raw)
        if result is None:
            print("  JSON parse error: could not recover")
        return result
    except Exception as e:
        print(f"  API error: {e}")
        return None


def flatten(data: dict, source_pdf: str) -> list[dict]:
    study = data.get("study", {})
    rows  = []
    for outcome in data.get("outcomes", []):
        row = {col: study.get(col) for col in CSV_COLUMNS if col in study}
        row.update(outcome)
        row["source_pdf"] = source_pdf
        rows.append(row)
    return rows


def process_entry(
    entry: dict,
    client: Groq,
    writer: csv.DictWriter,
    csv_file,
) -> bool:
    doi      = entry["doi"]
    pdf_path = Path(entry["pdf_file"])

    if not pdf_path.exists():
        print(f"  PDF not found: {pdf_path}")
        return False

    try:
        text = extract_pdf_text(pdf_path)
        print(f"  Extracted {len(text):,} chars")
    except Exception as e:
        print(f"  PDF read error: {e}")
        return False

    data = call_groq(client, text, doi)
    if data is None:
        return False

    rows = flatten(data, str(pdf_path))
    writer.writerows(rows)
    csv_file.flush()
    print(f"  Wrote {len(rows)} rows")
    return True


def main() -> None:
    load_env()
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        sys.exit(
            "GROQ_API_KEY not set.\n"
            "Get a free key at https://console.groq.com\n"
            "Then add to .env:  GROQ_API_KEY=gsk_..."
        )

    client = Groq(api_key=api_key)

    parser = argparse.ArgumentParser()
    parser.add_argument("--doi", help="Process a single paper by DOI")
    args = parser.parse_args()

    if not META_FILE.exists():
        sys.exit(f"No metadata file at {META_FILE}. Run scrape.py first.")

    metadata    = json.loads(META_FILE.read_text())
    pdf_entries = [m for m in metadata if m.get("pdf_file")]

    if args.doi:
        pdf_entries = [m for m in pdf_entries if m["doi"] == args.doi]
        if not pdf_entries:
            sys.exit(f"DOI not found in metadata or no PDF available: {args.doi}")

    processed: set[str] = set()
    if PROGRESS_FILE.exists():
        processed = set(json.loads(PROGRESS_FILE.read_text()))

    pending = [e for e in pdf_entries if e["doi"] not in processed]
    print(f"{len(pending)} papers to process ({len(processed)} already done)")

    write_header = not OUTPUT_CSV.exists() or OUTPUT_CSV.stat().st_size == 0
    with OUTPUT_CSV.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        if write_header:
            writer.writeheader()

        for i, entry in enumerate(pending):
            doi   = entry["doi"]
            title = entry.get("title", "")
            cites = entry.get("citations", "?")
            print(f"[{i+1}/{len(pending)}] {cites}x  {title[:65]}...")

            ok = process_entry(entry, client, writer, f)
            if ok:
                processed.add(doi)
                PROGRESS_FILE.write_text(json.dumps(list(processed)))

            time.sleep(1)

    print(f"\nDone -> {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
