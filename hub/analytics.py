"""The view-models the hub's screens read: the split, accuracy against the
historical outcomes, queues, monthly volume and agent statistics. Pure
functions over the in-memory book; the historical outcome is read here
and on the case page's historical drawer, nowhere else."""
from collections import Counter, defaultdict

from fincrime.hub import pipeline
from fincrime.hub.agents import AGENTS, triage
from fincrime.utils import dates

SEGMENTS = (   # key, label, who handles it, colour slot
    ("auto_clear", "Cleared by agents", "no analyst", "s1"),
    ("auto_reject", "Returned to customer (documents)", "no analyst", "s2"),
    ("auto_rfi", "Returned to customer (address RFI)", "no analyst", "s2"),
    ("tp_flag", "Confirmed match → decision queue", "AI-adjudicated, a person decides", "s3"),
    ("inconclusive", "Date of birth not decisive", "human adjudication", "s4"),
    ("sanctions", "Sanctions guard", "human adjudication", "s5"),
    ("fraud", "Fraud referral (documents)", "human adjudication", "s5"),
    ("below_gate", "Below the confidence gate", "human adjudication", "s4"),
)
HOLD_CODES = ("DOB_TRANSPOSED", "DOB_YEAR_ONLY", "DOB_RANGE", "DOB_MISSING", "DOB_NEAR", "NAME_WEAK", "SCRIPT_CONFLICT")


def _pct(a, b):
    return round(100.0 * a / b, 1) if b else 0.0


def composition(book):
    split = pipeline.split(book)
    segs = []
    for key, label, who, slot in SEGMENTS:
        n = split.get(key, 0)
        if n:
            segs.append({"key": key, "label": label, "who": who, "slot": slot, "n": n, "pct": _pct(n, split["n"])})
    return split, segs


def accuracy_numbers(book):
    auto = [c for c in book if c.get("status") == triage.STATUS_AUTO_CLEARED or
            (c.get("decision") and c["decision"].get("basis") == "auto" and (c.get("screening") or {}).get("decision") == "auto_clear")]
    auto_clear = [c for c in book if (c.get("screening") or {}).get("decision") == "auto_clear"]
    precision_n = sum(1 for c in auto_clear if c["history"]["outcome"] == "FP")
    tp = [c for c in book if c["history"]["outcome"] == "TP"]
    tp_flagged = sum(1 for c in tp if (c.get("screening") or {}).get("verdict") == "same person")
    tp_cleared = sum(1 for c in tp if (c.get("screening") or {}).get("decision") == "auto_clear")
    held = [c for c in book if (c.get("screening") or {}).get("reason_code") in HOLD_CODES]
    fp = [c for c in book if c["history"]["outcome"] == "FP"]
    fp_cleared = sum(1 for c in fp if (c.get("screening") or {}).get("decision") == "auto_clear")
    return {"auto_clear": len(auto_clear), "precision": _pct(precision_n, len(auto_clear)), "precision_n": precision_n,
            "tp_total": len(tp), "tp_flagged": tp_flagged, "tp_recall": _pct(tp_flagged, len(tp)), "tp_cleared": tp_cleared,
            "fp_total": len(fp), "fp_cleared": fp_cleared, "fp_recall": _pct(fp_cleared, len(fp)),
            "held": len(held), "auto_status": len(auto)}


def minutes_saved(book):
    saved = sum(c["history"]["minutes"] for c in book if c.get("group") == "automated")
    total = sum(c["history"]["minutes"] for c in book)
    return {"minutes": saved, "hours": round(saved / 60.0), "total_minutes": total, "pct": _pct(saved, total)}


def monthly(book):
    months = defaultdict(lambda: {"n": 0, "automated": 0, "adjudicated": 0})
    for c in book:
        m = (c["history"]["review_date"] or "")[:7]
        row = months[m]
        row["n"] += 1
        if c.get("group") == "automated":
            row["automated"] += 1
        s = c.get("screening") or {}
        if c.get("group") == "automated" or s.get("decision") == "auto_confirm":
            row["adjudicated"] += 1
    out = []
    for m in sorted(months):
        r = months[m]
        out.append({"month": m, "label": dates.parse(m + "-01").strftime("%b %Y") if dates.parse(m + "-01") else m, "n": r["n"],
                    "automated": r["automated"], "adjudicated": r["adjudicated"], "pct": _pct(r["adjudicated"], r["n"])})
    return out


def queues(book, now=None):
    now = now or dates.now()
    rows = defaultdict(lambda: {"n": 0, "overdue": 0, "p1": 0, "oldest_h": 0.0})
    for c in book:
        if c.get("group") != "human_open":
            continue
        q = c.get("queue") or "Unassigned"
        row = rows[q]
        row["n"] += 1
        if c.get("priority") == "P1":
            row["p1"] += 1
        due = dates.parse(c.get("due_at"))
        if due and due < now:
            row["overdue"] += 1
        received = dates.parse((c.get("synth") or {}).get("received_at"))
        if received:
            row["oldest_h"] = max(row["oldest_h"], round((now - received).total_seconds() / 3600, 1))
    order = [triage.QUEUE_L1, triage.QUEUE_L2, triage.QUEUE_EDD, triage.QUEUE_INTAKE]
    return [{"queue": q, **rows[q]} for q in order if q in rows] + [{"queue": q, **r} for q, r in rows.items() if q not in order]


def agent_stats(book):
    stats = []
    for a in AGENTS:
        steps = [s for c in book for s in c.get("trail", []) if s.get("agent") == a["key"]]
        if a["key"] == "claude":
            steps = [s for c in book for s in c.get("trail", []) if s.get("kind") == "claude"]
        decisions = Counter(s.get("decision", "") for s in steps)
        ms = [s.get("elapsed_ms", 0) for s in steps if s.get("elapsed_ms") is not None]
        stats.append({"agent": a, "runs": len(steps), "decisions": decisions.most_common(4),
                      "mean_ms": round(sum(ms) / len(ms), 2) if ms else 0.0, "needs_human": sum(1 for s in steps if s.get("needs_human"))})
    return stats


def latest_decisions(book, n=8):
    rows = [c for c in book if c.get("decision") and c["decision"].get("basis") == "human"]
    rows.sort(key=lambda c: c["decision"]["at"], reverse=True)
    return rows[:n]


def overview(book, settings):
    split, segs = composition(book)
    acc = accuracy_numbers(book)
    return {"split": split, "segments": segs, "accuracy": acc, "saved": minutes_saved(book), "monthly": monthly(book),
            "queues": queues(book), "agents": agent_stats(book), "latest": latest_decisions(book),
            "open_human": sum(1 for c in book if c.get("group") == "human_open"),
            "overdue": sum(1 for c in book if c.get("group") == "human_open" and dates.parse(c.get("due_at")) and dates.parse(c.get("due_at")) < dates.now()),
            "gate": settings.get("gate", 85), "last_run": settings.get("last_run", {})}


# ── accuracy page ────────────────────────────────────────────────────────────

AGENT_ROWS = (("Auto-clear", "auto_clear"), ("Confirmed match → decision queue", "auto_confirm"),
              ("Human adjudication", "needs_review"))
HISTORY_COLS = ("FP", "TP", "Escalated", "Pending")
HISTORY_LABEL = {"FP": "False positive (cleared)", "TP": "True positive", "Escalated": "Escalated to L2", "Pending": "Pending documents"}


def _agent_row(c):
    s = c.get("screening") or {}
    return s.get("decision", "needs_review")


def accuracy(book):
    matrix = {r: {h: 0 for h in HISTORY_COLS} for _, r in AGENT_ROWS}
    for c in book:
        matrix[_agent_row(c)][c["history"]["outcome"]] += 1
    rows = [{"label": label, "key": key, "cells": [matrix[key][h] for h in HISTORY_COLS], "total": sum(matrix[key].values())}
            for label, key in AGENT_ROWS]
    totals = [sum(matrix[k][h] for _, k in AGENT_ROWS) for h in HISTORY_COLS]
    by_variance = defaultdict(lambda: {"n": 0, "automated": 0, "adjudicated": 0, "detected": 0, "cleared_fp": 0, "fp": 0, "minutes": 0})
    for c in book:
        v = c["history"]["name_variance"]
        r = by_variance[v]
        r["n"] += 1
        r["minutes"] += c["history"]["minutes"]
        if c.get("group") == "automated":
            r["automated"] += 1
        s = c.get("screening") or {}
        if c.get("group") == "automated" or s.get("decision") == "auto_confirm":
            r["adjudicated"] += 1
        if v in (c.get("identity") or {}).get("variance_detected", []):
            r["detected"] += 1
        if c["history"]["outcome"] == "FP":
            r["fp"] += 1
            if s.get("decision") == "auto_clear":
                r["cleared_fp"] += 1
    variance_rows = []
    for v, r in sorted(by_variance.items(), key=lambda kv: -kv[1]["n"]):
        variance_rows.append({"variance": v, "n": r["n"], "adjudicated_pct": _pct(r["adjudicated"], r["n"]),
                              "automated_pct": _pct(r["automated"], r["n"]),
                              "detected_pct": _pct(r["detected"], r["n"]) if not v.startswith("None") else None,
                              "fp_cleared_pct": _pct(r["cleared_fp"], r["fp"]) if r["fp"] else None,
                              "avg_minutes": round(r["minutes"] / r["n"]) if r["n"] else 0})
    by_country = defaultdict(lambda: {"n": 0, "adjudicated": 0, "automated": 0})
    for c in book:
        r = by_country[c["country_name"]]
        r["n"] += 1
        s = c.get("screening") or {}
        if c.get("group") == "automated":
            r["automated"] += 1
        if c.get("group") == "automated" or s.get("decision") == "auto_confirm":
            r["adjudicated"] += 1
    country_rows = [{"country": k, "n": r["n"], "adjudicated_pct": _pct(r["adjudicated"], r["n"]), "automated_pct": _pct(r["automated"], r["n"])}
                    for k, r in sorted(by_country.items())]
    held = [c for c in book if (c.get("screening") or {}).get("dob", {}).get("class") == "transposed"]
    unverified = [c for c in book if (c.get("screening") or {}).get("decision") == "auto_clear" and c["history"]["outcome"] != "FP"]
    missed = [c for c in book if c["history"]["outcome"] == "TP" and (c.get("screening") or {}).get("verdict") != "same person"]
    resurfaced = [c for c in book if c["history"]["impact"].startswith("Potential missed match")]
    resurfaced_flagged = sum(1 for c in resurfaced if (c.get("screening") or {}).get("verdict") == "same person")
    reconciled = [c for c in book if c["history"]["impact"].startswith("Review delayed") and (c.get("identity") or {}).get("address", {}).get("resolved")]
    return {"matrix": rows, "history_cols": [HISTORY_LABEL[h] for h in HISTORY_COLS], "totals": totals,
            "numbers": accuracy_numbers(book), "variance_rows": variance_rows, "country_rows": country_rows,
            "held": held, "unverified": unverified, "missed": missed, "saved": minutes_saved(book),
            "resurfaced": {"n": len(resurfaced), "flagged": resurfaced_flagged},
            "reconciled": {"n": sum(1 for c in book if c["history"]["impact"].startswith("Review delayed")), "resolved": len(reconciled)}}


# ── queue pages ──────────────────────────────────────────────────────────────

def screening_rows(book):
    rows = [c for c in book if c.get("group") == "human_open"]
    order = {"P1": 0, "P2": 1, "P3": 2}
    rows.sort(key=lambda c: (order.get(c.get("priority"), 3), c.get("due_at") or ""))
    return rows


def intake_rows(book):
    rows = [c for c in book if (c.get("intake") or {}).get("decision") in ("Reject", "Refer") or
            c.get("status") in (triage.STATUS_AUTO_RFI, "Pending (customer)")]
    rows.sort(key=lambda c: ((c.get("intake") or {}).get("decision") != "Refer", c.get("synth", {}).get("received_at", "")))
    return rows


def audit_rows(book, limit=500):
    rows = []
    for c in book:
        for a in c.get("audit", []):
            rows.append({"at": a["at"], "actor": a["actor"], "action": a["action"], "detail": a.get("detail", ""), "case": c["id"],
                         "name": (c.get("identity") or {}).get("name", {}).get("canonical", c["customer"]["name"])})
    rows.sort(key=lambda r: r["at"], reverse=True)
    return rows[:limit]
