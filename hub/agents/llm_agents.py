"""The Claude agents, on demand from a case: a second opinion on the
watchlist match, a draft compliance narrative, and a draft customer
message. Advisory: nothing here changes a status. Each call reads the
normalised case, the watchlist record and the rule agents' findings -
never the historical outcome - and leaves a step on the trail and a row
in the audit.
"""
import json
import time

from fincrime.engines.errors import WorkflowError
from fincrime.hub import llm
from fincrime.hub.agents import finding, step
from fincrime.hub.agents.intake import REASONS
from fincrime.utils import dates

RESULT_LABELS = ["False Positive", "Potential Match", "True Positive"]
VERDICT_TONE = {"different person": "good", "same person": "serious", "unclear": "warning"}

TASKS = {
    "second_opinion": {"label": "Second opinion on the match", "action": "Claude second opinion"},
    "narrative": {"label": "Draft compliance narrative", "action": "Claude narrative drafted"},
    "customer_message": {"label": "Draft customer message", "action": "Claude customer message drafted"},
}

SECOND_OPINION_SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": ["different person", "same person", "unclear"]},
        "confidence": {"type": "integer", "description": "0-100, how sure you are of the verdict"},
        "summary": {"type": "string", "description": "one or two sentences an auditor can read"},
        "reasons": {"type": "array", "items": {"type": "string"}, "description": "short, one fact each"},
        "missing_data": {"type": "array", "items": {"type": "string"}, "description": "decisive fields absent on either side"},
        "suggested_outcome": {"type": "string", "enum": RESULT_LABELS},
        "what_would_resolve": {"type": "array", "items": {"type": "string"},
                               "description": "the documents or checks that would settle it, when unclear"},
    },
    "required": ["verdict", "confidence", "summary", "reasons", "missing_data", "suggested_outcome", "what_would_resolve"],
    "additionalProperties": False,
}
NARRATIVE_SCHEMA = {
    "type": "object",
    "properties": {
        "narrative": {"type": "string", "description": "two to four sentences: what was checked, what was found, the outcome"},
        "rationale_points": {"type": "array", "items": {"type": "string"}},
        "regulatory_note": {"type": "string", "description": "one sentence on the control or rule applied, or empty"},
        "record_corrections": {"type": "array", "items": {"type": "string"},
                               "description": "CRM fields to correct from the documents, or empty"},
    },
    "required": ["narrative", "rationale_points", "regulatory_note", "record_corrections"],
    "additionalProperties": False,
}
MESSAGE_SCHEMA = {
    "type": "object",
    "properties": {
        "subject": {"type": "string"},
        "body": {"type": "string", "description": "at most 120 words, plain English, no legal threats"},
        "channel": {"type": "string", "enum": ["email", "app"]},
    },
    "required": ["subject", "body", "channel"],
    "additionalProperties": False,
}

SYSTEM_SECOND_OPINION = """You assist a KYC screening analyst. You receive one onboarding client, normalised from their documents, one watchlist record a screening vendor returned as a possible match, and the findings of the deterministic agents that already compared them. Decide whether the record could be the same person as the client.

Judge only from the data given and never invent facts. A similar name is why the record came back and is not, by itself, evidence of identity. A full date of birth that differs by years points to a different person; an identical full date of birth with the same name points to the same person; a year-only, range or missing date of birth is not decisive, and a day/month transposition cannot be told from two people born in the same year without another identifier. A different city in the same country is weak evidence either way. When a decisive field is missing on either side, list it under missing_data, lean to "unclear", and say in what_would_resolve which document or check would settle it. suggested_outcome follows the verdict: "different person" is False Positive, "same person" is True Positive, "unclear" is Potential Match. Keep the summary to one or two sentences an auditor can read; each reason is one fact."""

SYSTEM_NARRATIVE = """You draft the compliance note an analyst records when closing a KYC screening case. Write in the register of the examples: two to four plain sentences stating what was checked, what was found, the outcome and any correction to the record; no marketing language, no speculation, no facts beyond the data given. Name the person as the documents spell them. The proposed outcome is the analyst's; write the note that supports it, and if the data does not support it, say so in the note."""

SYSTEM_MESSAGE = """You write the message an onboarding team sends a customer whose upload was not accepted or whose address could not be confirmed. Plain English, at most 120 words, warm but direct: say exactly what was wrong and exactly what to send (a photo of the physical document taken with the phone camera, all four corners and both code lines visible, no glare, no scan or PDF; or a bank statement or utility bill from the last six months showing the name and the address). Use the customer's first name only. No legal language, no threats, no mention of watchlists or screening."""


def _identity_brief(case):
    name, addr = case["identity"]["name"], case["identity"]["address"]
    return {"legal_name": name["canonical"], "aka": [a["name"] for a in name["aka"]], "date_of_birth": case["customer"]["dob"],
            "customer_type": case["customer"]["type"], "nationality": case["country_name"], "risk_rating": case["customer"]["risk"],
            "current_address": addr["current"]["display"], "address_reconciled": addr["resolved"], "address_flags": addr["flags"],
            "name_changes": [f"{c['rule']}: {c['from']} -> {c['to']}" for c in name["changes"]]}


def _record_brief(case):
    wc = case["wc"]
    return {"name": wc["name_raw"], "aka": wc["aka"], "date_of_birth": wc["dob"]["raw"].split(" (")[0],
            "jurisdiction": wc["location"]["raw"], "category": wc["category"], "narrative": wc["narrative"],
            "vendor_match_score": wc["score"], "matched_attributes": wc["matched_attributes"]}


def _findings_brief(case):
    sc = case["screening"]
    return {"date_of_birth": sc["dob"]["detail"], "name_similarity": sc["name"]["score"], "name_facts": sc["name"]["facts"],
            "place": sc["place"]["detail"], "decision": sc["decision"], "verdict": sc["verdict"], "confidence": sc["confidence"],
            "reasons": sc["reasons"], "hard_stops": sc["hard_stops"],
            "documents": (case.get("intake") or {}).get("reasons", []) or ["accepted"]}


def brief(case, which, book=None, proposed=""):
    """The user turn for each task, as JSON. The historical outcome of the
    case is never included; the narrative task reads other cases' notes as
    style examples only."""
    if which == "second_opinion":
        body = {"client": _identity_brief(case), "watchlist_record": _record_brief(case), "agent_findings": _findings_brief(case)}
    elif which == "narrative":
        outcome = proposed or case["screening"]["suggested"]
        family = {"False Positive": "FP", "True Positive": "TP", "Potential Match": "Escalated"}.get(outcome, "FP")
        examples = []
        for other in (book or []):
            if other["id"] != case["id"] and other["history"]["outcome"] == family and other["history"]["notes"]:
                examples.append(other["history"]["notes"])
            if len(examples) == 3:
                break
        body = {"client": _identity_brief(case), "watchlist_record": _record_brief(case), "agent_findings": _findings_brief(case),
                "proposed_outcome": outcome, "style_examples_from_other_cases": examples}
    else:
        intake = case.get("intake") or {}
        codes = intake.get("codes") or []
        body = {"customer_first_name": (case["identity"]["name"].get("given") or [case["customer"]["name"].split()[0]])[0].title(),
                "documents": [f["label"] for f in case["files"]],
                "problems": [{"code": c, "label": REASONS[c]["label"]} for c in codes if c in REASONS],
                "address_notes": case["identity"]["address"]["notes"],
                "template_in_use": case.get("customer_message", ""), "status": case["status"]}
    return json.dumps(body, ensure_ascii=False, indent=1, sort_keys=True, default=str)


def run(case, which, actor, settings, book=None, proposed=""):
    """One Claude agent over one case. Writes case["ai"][which], a trail
    step and an audit row; returns (result, message)."""
    if which not in TASKS:
        raise WorkflowError("Unknown Claude task.")
    why = llm.why_not(settings)
    if why:
        raise WorkflowError(why)
    if not case.get("identity") or not case.get("screening"):
        raise WorkflowError("The rule agents have not run on this case yet.")
    started = time.perf_counter()
    system, schema = {"second_opinion": (SYSTEM_SECOND_OPINION, SECOND_OPINION_SCHEMA), "narrative": (SYSTEM_NARRATIVE, NARRATIVE_SCHEMA),
                      "customer_message": (SYSTEM_MESSAGE, MESSAGE_SCHEMA)}[which]
    text = brief(case, which, book, proposed)
    try:
        data, usage, seconds = llm.call(system, text, schema)
    except WorkflowError as exc:
        llm.record(0, str(exc))
        raise
    llm.record(seconds)
    result = {"model": llm.MODEL, "at": dates.now_str(), "by": actor, "seconds": seconds, "tokens": usage}
    if which == "second_opinion":
        verdict = data.get("verdict") if data.get("verdict") in VERDICT_TONE else "unclear"
        try:
            confidence = max(0, min(100, int(data.get("confidence") or 0)))
        except (TypeError, ValueError):
            confidence = 0
        suggested = data.get("suggested_outcome")
        result.update(verdict=verdict, confidence=confidence, summary=str(data.get("summary") or "").strip(),
                      reasons=[str(r) for r in (data.get("reasons") or [])][:8],
                      missing_data=[str(r) for r in (data.get("missing_data") or [])][:8],
                      suggested_outcome=suggested if suggested in RESULT_LABELS else {"different person": "False Positive", "same person": "True Positive", "unclear": "Potential Match"}[verdict],
                      what_would_resolve=[str(r) for r in (data.get("what_would_resolve") or [])][:6])
        decision = f"Claude: {verdict} ({confidence}%)"
        findings = [finding("CLAUDE", result["summary"], VERDICT_TONE[verdict])] + \
                   [finding("WHY", r, "neutral") for r in result["reasons"][:4]]
        reasons = result["reasons"] + [f"Missing: {m}" for m in result["missing_data"]]
        message = f"Claude's second opinion on {case['id']}: {verdict}, {confidence}% — suggested {result['suggested_outcome']}."
        agree = (verdict == case["screening"]["verdict"])
        findings.append(finding("AGREE" if agree else "DIFFER", "Agrees with the Watchlist Adjudication Agent." if agree else
                                f"Differs from the Watchlist Adjudication Agent ({case['screening']['verdict']}).", "good" if agree else "warning"))
    elif which == "narrative":
        result.update(narrative=str(data.get("narrative") or "").strip(), rationale_points=[str(r) for r in (data.get("rationale_points") or [])][:6],
                      regulatory_note=str(data.get("regulatory_note") or "").strip(),
                      record_corrections=[str(r) for r in (data.get("record_corrections") or [])][:6], proposed=proposed or case["screening"]["suggested"])
        confidence = None
        decision = "Claude: narrative drafted"
        findings = [finding("CLAUDE", result["narrative"], "neutral")]
        reasons = result["rationale_points"]
        message = f"Claude drafted the compliance narrative for {case['id']}; it is in the decision panel to edit."
    else:
        result.update(subject=str(data.get("subject") or "").strip(), body=str(data.get("body") or "").strip(),
                      channel=data.get("channel") if data.get("channel") in ("email", "app") else "email")
        confidence = None
        decision = "Claude: customer message drafted"
        findings = [finding("CLAUDE", result["subject"] + " — " + result["body"][:160], "neutral")]
        reasons = []
        message = f"Claude drafted the customer message for {case['id']}."
    case.setdefault("ai", {})[which] = result
    trail_step = step("claude", started, decision, confidence=confidence, findings=findings, reasons=reasons, needs_human=True,
                      reason_code="", inputs={"brief_chars": len(text), "task": which}, label=f"Claude · {TASKS[which]['label']}",
                      model=llm.MODEL, seconds=seconds, tokens=usage, by=actor)
    trail_step["elapsed_ms"] = round(seconds * 1000, 1)
    case.setdefault("trail", []).append(trail_step)
    case.setdefault("audit", []).append({"at": result["at"], "actor": actor, "action": TASKS[which]["action"],
                                         "detail": f"{decision} [{llm.MODEL}, {seconds} s]"})
    return result, message
