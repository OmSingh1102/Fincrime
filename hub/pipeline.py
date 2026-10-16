"""The run over the book, and the settings that shape it.

Settings live in one stored record (kind "hub_settings"), changed by the
team lead on the Agents page, each change logged. run_all() puts every
case without a human decision through the four agents; a case somebody has
decided keeps its decision and its trail.
"""
import threading
import time

from fincrime.engines.errors import WorkflowError
from fincrime.hub import documents
from fincrime.hub.agents import RUN_ORDER, adjudication, identity, intake, triage
from fincrime.utils import dates

LOCK = threading.Lock()
RUNNERS = {"identity": identity.run, "intake": intake.run, "adjudication": adjudication.run, "triage": triage.run}

SETTING_LABELS = {"gate": "Confidence gate", "doc_strictness": "Document strictness", "sanctions_guard": "Sanctions always to a human",
                  "auto_rfi": "Auto-RFI on an unresolved address", "claude": "Claude agents",
                  "agent_identity": "Identity Normalisation Agent", "agent_intake": "Document Intake Agent",
                  "agent_adjudication": "Watchlist Adjudication Agent", "agent_triage": "Triage & Routing Agent"}


def default_settings():
    return {"id": "hub_settings", "gate": 85, "doc_strictness": "standard", "sanctions_guard": True, "auto_rfi": True,
            "claude": True, "agents": {"identity": True, "intake": True, "adjudication": True, "triage": True},
            "history": [], "last_run": {}}


def _as_bool(value):
    return str(value).strip().lower() in ("1", "true", "on", "yes")


def set_setting(record, actor, key, value):
    """Change one setting; returns (before, after). Logged on the record."""
    if key == "gate":
        try:
            after = int(value)
        except (TypeError, ValueError):
            raise WorkflowError("The gate must be a whole number.")
        if not 50 <= after <= 99:
            raise WorkflowError("The gate must be between 50 and 99.")
        before = record.get("gate", 85)
        record["gate"] = after
    elif key == "doc_strictness":
        if value not in ("lenient", "standard", "strict"):
            raise WorkflowError("Document strictness is lenient, standard or strict.")
        before = record.get("doc_strictness", "standard")
        record["doc_strictness"] = value
        after = value
    elif key in ("sanctions_guard", "auto_rfi", "claude"):
        before = bool(record.get(key, True))
        after = _as_bool(value)
        record[key] = after
    elif key.startswith("agent_") and key[6:] in RUN_ORDER:
        agents = record.setdefault("agents", {})
        before = bool(agents.get(key[6:], True))
        after = _as_bool(value)
        agents[key[6:]] = after
    else:
        raise WorkflowError("Unknown setting.")
    if before != after:
        record.setdefault("history", []).append({"at": dates.now_str(), "by": actor, "key": key,
                                                 "label": SETTING_LABELS.get(key, key), "from": before, "to": after})
    return before, after


def run_case(case, settings):
    """The four agents over one case. A case with a human decision keeps it."""
    if (case.get("decision") and case["decision"].get("basis") == "human") or case.get("locked"):
        return False
    if not case.get("files"):
        documents.synthesise(case)
    case["trail"] = [s for s in case.get("trail", []) if s.get("kind") == "claude"]
    case["decision"] = None
    case["customer_message"] = ""
    rules_steps = []
    for key in RUN_ORDER:
        rules_steps.append(RUNNERS[key](case, settings))
    case["trail"] = rules_steps + case["trail"]
    if case["status"] in triage.AUTOMATED:
        case["decision"] = {"outcome": case["screening"].get("suggested", "") if case["status"] == triage.STATUS_AUTO_CLEARED else "",
                            "action": {triage.STATUS_AUTO_CLEARED: "Clear", triage.STATUS_AUTO_REJECTED: "Re-upload requested",
                                       triage.STATUS_AUTO_RFI: "RFI sent"}[case["status"]],
                            "basis": "auto", "by": "Agents", "at": dates.now_str(), "comment": ""}
    return True


def split(book):
    """How the book divides: automated, human, by status and by reason."""
    out = {"n": len(book), "automated": 0, "human": 0, "closed_by_human": 0, "by_status": {}, "by_reason": {},
           "auto_clear": 0, "auto_reject": 0, "auto_rfi": 0, "tp_flag": 0, "sanctions": 0, "fraud": 0, "inconclusive": 0, "below_gate": 0}
    for c in book:
        status = c.get("status", "")
        out["by_status"][status] = out["by_status"].get(status, 0) + 1
        if c.get("group") == "automated":
            out["automated"] += 1
            if status == triage.STATUS_AUTO_CLEARED:
                out["auto_clear"] += 1
            elif status == triage.STATUS_AUTO_REJECTED:
                out["auto_reject"] += 1
            elif status == triage.STATUS_AUTO_RFI:
                out["auto_rfi"] += 1
            continue
        if c.get("group") == "human_closed":
            out["closed_by_human"] += 1
        out["human"] += 1
        s = c.get("screening") or {}
        code = s.get("reason_code", "")
        if "fraud_referral" in s.get("hard_stops", []) or (c.get("intake") or {}).get("decision") == "Refer":
            key = "fraud"
        elif "sanctions" in s.get("hard_stops", []):
            key = "sanctions"
        elif s.get("decision") == "auto_confirm" or s.get("verdict") == "same person":
            key = "tp_flag"
        elif code == "BELOW_GATE":
            key = "below_gate"
        else:
            key = "inconclusive"
        out[key] += 1
        out["by_reason"][code] = out["by_reason"].get(code, 0) + 1
    out["automated_pct"] = round(100.0 * out["automated"] / max(1, out["n"]), 1)
    # AI-adjudicated: the agents decided the record (cleared, returned to the
    # customer, or confirmed and routed); human adjudication is what is left.
    out["adjudicated"] = out["automated"] + out["tp_flag"]
    out["adjudicated_pct"] = round(100.0 * out["adjudicated"] / max(1, out["n"]), 1)
    out["human_adjudication"] = out["n"] - out["adjudicated"]
    out["human_pct"] = round(100.0 - out["adjudicated_pct"], 1)
    return out


def run_all(book, settings, reason="boot", actor="System"):
    """Every case without a human decision, under a lock. Returns the summary."""
    with LOCK:
        t0 = time.perf_counter()
        ran = sum(1 for c in book if run_case(c, settings))
        summary = split(book)
        summary.update({"ran": ran, "seconds": round(time.perf_counter() - t0, 3), "at": dates.now_str(), "reason": reason,
                        "by": actor, "gate": settings.get("gate", 85)})
        settings["last_run"] = summary
        return summary
