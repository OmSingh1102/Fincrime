"""The hub's agents and the shape of a trail step.

Four deterministic agents run over every case at boot, in this order:
identity (what is the legal name and address?), intake (are the uploads
acceptable?), adjudication (is the watchlist record the same person?) and
triage (who, if anyone, needs to look?). The Claude agents run on demand
from the case page and append their own steps.
"""
import time

from fincrime.utils import dates

AGENTS = [
    {"key": "identity", "label": "Identity Normalisation Agent", "short": "Identity", "kind": "rules", "icon": "user-check",
     "purpose": "Reads the CRM record, the passport's visual and machine-readable zones and the secondary ID, and "
                "produces the legal screening name, the other names to screen as AKAs, and the normalised address.",
     "inputs": "CRM name and address · passport VIZ and MRZ · secondary ID · proof of address",
     "outputs": "Canonical name · AKAs · what changed and why · address agreement across documents"},
    {"key": "intake", "label": "Document Intake Agent", "short": "Intake", "kind": "rules", "icon": "file-text",
     "purpose": "Checks each upload the way an onboarding system should: camera photo or flat scan, edges and MRZ "
                "present, check digits, visual zone against the MRZ, expiry, proof-of-address type, age and address.",
     "inputs": "The uploads and their capture data · the identity agent's name and address findings",
     "outputs": "Accept · Reject with a reason code and a customer message · Refer (possible fraud) to a human"},
    {"key": "adjudication", "label": "Watchlist Adjudication Agent", "short": "Adjudication", "kind": "rules", "icon": "shield",
     "purpose": "Compares the normalised client with the watchlist record: name similarity over every form, the date "
                "of birth read from the record itself, and the jurisdiction. Clears or confirms only above the gate.",
     "inputs": "Canonical name and AKAs · date of birth · current address · the watchlist record",
     "outputs": "Verdict with confidence · auto-clear, confirmed match for a human, or needs review · reasons"},
    {"key": "triage", "label": "Triage & Routing Agent", "short": "Triage", "kind": "rules", "icon": "layers",
     "purpose": "Turns the other agents' findings into a status, a queue and a priority: closed without an analyst, "
                "back to the customer, or to L1, L2 or EDD with an SLA.",
     "inputs": "Intake decision · adjudication decision · risk rating · watchlist category",
     "outputs": "Status · queue · priority · SLA"},
    {"key": "claude", "label": "Claude agents", "short": "Claude", "kind": "claude", "icon": "sparkles",
     "purpose": "On demand from a case: a second opinion on a match, a draft compliance narrative, and a customer "
                "message for a document rejection or a request for information.",
     "inputs": "The normalised case, the watchlist record and the agents' findings - never the historical outcome",
     "outputs": "Advisory text and a suggested outcome; the analyst decides"},
]
AGENT_BY_KEY = {a["key"]: a for a in AGENTS}
RUN_ORDER = ("identity", "intake", "adjudication", "triage")


def step(key, started, decision, *, confidence=None, findings=(), reasons=(), needs_human=False, reason_code="",
         inputs=None, **extra):
    """One trail entry. `started` is a time.perf_counter() reading."""
    out = {"agent": key, "label": AGENT_BY_KEY[key]["label"], "kind": AGENT_BY_KEY[key]["kind"], "at": dates.now_str(),
           "elapsed_ms": round((time.perf_counter() - started) * 1000, 2), "decision": decision,
           "confidence": confidence, "findings": list(findings), "reasons": list(reasons), "needs_human": bool(needs_human),
           "reason_code": reason_code, "inputs": inputs or {}}
    out.update(extra)
    return out


def finding(code, text, tone="neutral"):
    """tone: good | neutral | warning | serious | critical."""
    return {"code": code, "text": text, "tone": tone}


def skipped(key, started):
    return step(key, started, "Skipped", findings=[finding("OFF", "This agent is switched off on the Agents page.", "neutral")])
