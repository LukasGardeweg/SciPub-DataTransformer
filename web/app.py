"""
SportsEvidence Community Web App

Run:
    python web/app.py

Environment variables (in .env):
    GROQ_API_KEY    — required for extraction
    ADMIN_PASSWORD  — admin panel password (default: changeme)
    SECRET_KEY      — Flask session key (auto-generated if missing)
"""

import csv
import hashlib
import os
import secrets
import sqlite3
import sys
from datetime import datetime
from functools import wraps
from pathlib import Path

from flask import (
    Flask,
    flash,
    redirect,
    render_template,
    request,
    send_from_directory,
    session,
    url_for,
)

sys.path.insert(0, str(Path(__file__).parent.parent))
from extract import CSV_COLUMNS, call_groq, extract_pdf_text, flatten, load_env

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent.parent
DATA_DIR   = BASE_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
DB_PATH    = DATA_DIR / "community.db"
OUTPUT_CSV = DATA_DIR / "dataset.csv"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# ── App ───────────────────────────────────────────────────────────────────────
load_env()

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY") or secrets.token_hex(32)
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024  # 32 MB


# ── Database ──────────────────────────────────────────────────────────────────
def get_db() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    conn = get_db()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS requests (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            doi             TEXT NOT NULL,
            title           TEXT,
            submitter_name  TEXT,
            submitter_email TEXT,
            pdf_filename    TEXT NOT NULL,
            status          TEXT NOT NULL DEFAULT 'pending',
            submitted_at    TEXT NOT NULL DEFAULT (datetime('now')),
            reviewed_at     TEXT,
            review_notes    TEXT
        )
    """)
    conn.commit()
    conn.close()


# ── Auth ──────────────────────────────────────────────────────────────────────
def require_admin(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if not session.get("admin"):
            return redirect(url_for("admin_login"))
        return f(*args, **kwargs)
    return decorated


# ── Helpers ───────────────────────────────────────────────────────────────────
def get_stats() -> dict:
    dois, row_count = set(), 0
    if OUTPUT_CSV.exists():
        with OUTPUT_CSV.open(encoding="utf-8") as f:
            for row in csv.DictReader(f):
                row_count += 1
                if row.get("doi"):
                    dois.add(row["doi"])

    conn = get_db()
    pending  = conn.execute("SELECT COUNT(*) FROM requests WHERE status='pending'").fetchone()[0]
    approved = conn.execute("SELECT COUNT(*) FROM requests WHERE status='approved'").fetchone()[0]
    conn.close()

    return {
        "study_count": len(dois),
        "row_count":   row_count,
        "pending":     pending,
        "approved":    approved,
    }


def doi_exists_in_dataset(doi: str) -> bool:
    if not OUTPUT_CSV.exists():
        return False
    with OUTPUT_CSV.open(encoding="utf-8") as f:
        return any(r.get("doi", "").lower() == doi.lower() for r in csv.DictReader(f))


# ── Public routes ─────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html", **get_stats())


@app.route("/browse")
def browse():
    q = request.args.get("doi", "").strip()
    studies: dict = {}

    if OUTPUT_CSV.exists():
        with OUTPUT_CSV.open(encoding="utf-8") as f:
            for row in csv.DictReader(f):
                doi = row.get("doi", "")
                if q and q.lower() not in doi.lower():
                    continue
                if doi not in studies:
                    studies[doi] = {
                        "doi":               doi,
                        "title":             row.get("title", ""),
                        "year":              row.get("year", ""),
                        "journal":           row.get("journal", ""),
                        "authors":           row.get("authors", ""),
                        "n_total":           row.get("n_total", ""),
                        "intervention_type": row.get("intervention_type", ""),
                        "rows":              0,
                    }
                studies[doi]["rows"] += 1

    return render_template(
        "browse.html",
        studies=list(studies.values()),
        q=q,
        not_found=bool(q and not studies),
    )


@app.route("/submit", methods=["GET", "POST"])
def submit():
    if request.method == "POST":
        doi   = request.form.get("doi", "").strip()
        title = request.form.get("title", "").strip()
        name  = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip()
        pdf   = request.files.get("pdf")

        errors = []
        if not doi:
            errors.append("DOI is required.")
        if not pdf or not pdf.filename:
            errors.append("A PDF file is required.")
        elif not pdf.filename.lower().endswith(".pdf"):
            errors.append("Only PDF files are accepted.")

        if not errors and doi_exists_in_dataset(doi):
            flash("A dataset for this DOI already exists.", "warning")
            return redirect(url_for("browse", doi=doi))

        if not errors:
            conn = get_db()
            dup = conn.execute(
                "SELECT id FROM requests WHERE doi=? AND status='pending'", (doi,)
            ).fetchone()
            conn.close()
            if dup:
                errors.append("A pending request for this DOI already exists.")

        if errors:
            for e in errors:
                flash(e, "error")
            return render_template("submit.html", doi=doi, title=title, name=name, email=email)

        safe_doi = doi.replace("/", "_").replace(".", "-")
        ts       = datetime.now().strftime("%Y%m%d%H%M%S")
        filename = f"{safe_doi}_{ts}.pdf"
        pdf.save(str(UPLOAD_DIR / filename))

        conn = get_db()
        conn.execute(
            "INSERT INTO requests (doi, title, submitter_name, submitter_email, pdf_filename) "
            "VALUES (?,?,?,?,?)",
            (doi, title, name, email, filename),
        )
        conn.commit()
        conn.close()

        flash(
            "Your submission has been received! We will review it shortly and notify you by email.",
            "success",
        )
        return redirect(url_for("submit"))

    return render_template("submit.html", doi=request.args.get("doi", ""))


# ── Admin routes ──────────────────────────────────────────────────────────────
@app.route("/admin/login", methods=["GET", "POST"])
def admin_login():
    if request.method == "POST":
        pw       = request.form.get("password", "")
        expected = os.environ.get("ADMIN_PASSWORD", "changeme")
        if (
            hashlib.sha256(pw.encode()).hexdigest()
            == hashlib.sha256(expected.encode()).hexdigest()
        ):
            session["admin"] = True
            return redirect(url_for("admin"))
        flash("Wrong password.", "error")
    return render_template("admin_login.html")


@app.route("/admin/logout")
def admin_logout():
    session.pop("admin", None)
    return redirect(url_for("index"))


@app.route("/admin")
@require_admin
def admin():
    tab = request.args.get("tab", "pending")
    conn = get_db()
    reqs = conn.execute(
        "SELECT * FROM requests WHERE status=? ORDER BY submitted_at DESC", (tab,)
    ).fetchall()
    counts = {
        s: conn.execute(
            "SELECT COUNT(*) FROM requests WHERE status=?", (s,)
        ).fetchone()[0]
        for s in ("pending", "approved", "rejected")
    }
    conn.close()
    return render_template(
        "admin.html",
        requests=reqs,
        tab=tab,
        counts=counts,
        has_groq=bool(os.environ.get("GROQ_API_KEY")),
    )


@app.route("/admin/pdf/<filename>")
@require_admin
def serve_pdf(filename):
    return send_from_directory(str(UPLOAD_DIR), filename)


@app.route("/admin/approve/<int:req_id>", methods=["POST"])
@require_admin
def approve(req_id: int):
    conn = get_db()
    req  = conn.execute("SELECT * FROM requests WHERE id=?", (req_id,)).fetchone()

    if not req or req["status"] != "pending":
        flash("Request not found or already processed.", "error")
        conn.close()
        return redirect(url_for("admin"))

    pdf_path = UPLOAD_DIR / req["pdf_filename"]

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        flash("GROQ_API_KEY is not set in .env — cannot run extraction.", "error")
        conn.close()
        return redirect(url_for("admin"))

    try:
        from groq import Groq

        client = Groq(api_key=api_key)
        text   = extract_pdf_text(pdf_path)
        data   = call_groq(client, text, req["doi"])

        if data is None:
            flash("Extraction failed — the API response could not be parsed.", "error")
            conn.close()
            return redirect(url_for("admin"))

        rows         = flatten(data, str(pdf_path))
        write_header = not OUTPUT_CSV.exists() or OUTPUT_CSV.stat().st_size == 0

        with OUTPUT_CSV.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
            if write_header:
                writer.writeheader()
            writer.writerows(rows)

        conn.execute(
            "UPDATE requests SET status='approved', reviewed_at=? WHERE id=?",
            (datetime.now().isoformat(), req_id),
        )
        conn.commit()
        flash(f"Approved — {len(rows)} rows added to the dataset.", "success")

    except Exception as e:
        flash(f"Processing error: {e}", "error")

    conn.close()
    return redirect(url_for("admin"))


@app.route("/admin/reject/<int:req_id>", methods=["POST"])
@require_admin
def reject(req_id: int):
    reason = request.form.get("reason", "").strip()
    conn   = get_db()
    conn.execute(
        "UPDATE requests SET status='rejected', reviewed_at=?, review_notes=? WHERE id=?",
        (datetime.now().isoformat(), reason, req_id),
    )
    conn.commit()
    conn.close()
    flash("Request rejected.", "info")
    return redirect(url_for("admin"))


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    init_db()
    app.run(debug=True, port=5000)
