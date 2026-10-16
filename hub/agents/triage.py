"""Triage & Routing Agent: the other agents' findings become a status, a
queue, a priority and an SLA."""
import time

from fincrime.hub.agents import finding, skipped, step
from fincrime.hub.agents.intake import customer_message
from fincrime.utils import dates

STATUS_AUTO_CLEARED = "Auto-cleared"
STATUS_AUTO_REJECTED = "Auto-rejected (documents)"
STATUS_AUTO_RFI = "Auto-RFI (address)"
STATUS_REVIEW = "Needs review"
AUTOMATED = (STATUS_AUTO_CLEARED, STATUS_AUTO_REJECTED, STATUS_AUTO_RFI)
QUEUE_CLOSED, QUEUE_INTAKE, QUEUE_L1, QUEUE_L2, QUEUE_EDD = "Auto-closed", "Intake (customer)", "L1 review", "L2 compliance", "EDD"
SLA_HOURS = {"P1": 24, "P2": 48, "P3": 72}
SENIOR_PEP = ("PEP - Level 1", "PEP - State-Owned")


def run(case, settings):
    started = time.perf_counter()
    intake = case.get("intake") or {"decision": "Accept", "reason_code": "", "address_unresolved": False}
    screening = case.get("screening") or {"decision": "needs_review", "reason_code": "", "hard_stops": [], "verdict": "unclear", "dob": {"class": "missing"}}
    risk, group, category = case["customer"]["risk"], case["wc"]["group"], case["wc"]["category"]
    received = case["synth"].get("received_at") or dates.now_str()
    reasons, findings = [], []
    if not settings["agents"].get("triage", True):
        status, queue, priority = (STATUS_AUTO_CLEARED, QUEUE_CLOSED, "P3") if screening["decision"] == "auto_clear" else (STATUS_REVIEW, QUEUE_L1, "P3")
        case.update(status=status, queue=queue, priority=priority, sla_hours=72, due_at=dates.add_hours(received, 72),
                    group="automated" if status in AUTOMATED else "human_open")
        return skipped("triage", started)
    tp_flag = screening["decision"] == "auto_confirm" or (screening["verdict"] == "same person")
    inconclusive = screening["dob"]["class"] in ("year_match", "range_match", "missing", "year_near", "range_near", "near")
    message = ""
    if intake["decision"] == "Refer":
        status, queue, priority = STATUS_REVIEW, QUEUE_L2, "P1"
        reasons.append("Possible fraud upload: L2 looks at the documents first; no customer contact.")
    elif screening["decision"] == "auto_clear":
        if intake["decision"] == "Reject":
            status, queue, priority = STATUS_AUTO_REJECTED, QUEUE_INTAKE, "P3"
            message = intake.get("message", "")
            reasons.append("Watchlist record cleared by the agent; the documents go back to the customer with the reason.")
        else:
            status, queue, priority = STATUS_AUTO_CLEARED, QUEUE_CLOSED, "P3"
            reasons.append("Watchlist record cleared and the documents accepted: no analyst needed.")
    elif intake["decision"] == "Reject" and not screening["hard_stops"] and not tp_flag:
        status, queue, priority = STATUS_AUTO_REJECTED, QUEUE_INTAKE, "P3"
        message = intake.get("message", "")
        reasons.append("The documents go back to the customer first; screening is reviewed when valid documents arrive.")
    elif (settings.get("auto_rfi", True) and inconclusive and intake.get("address_unresolved") and not screening["hard_stops"]
          and not (risk == "High" and group in ("PEP", "Law Enforcement", "Sanctions"))):
        status, queue, priority = STATUS_AUTO_RFI, QUEUE_INTAKE, "P3"
        message = customer_message("ADDR_UNRESOLVED", settings.get("doc_strictness", "standard"))
        reasons.append("The date of birth is not decisive and the addresses on file disagree: the customer is asked for a current proof of address before an analyst spends time.")
    else:
        status = STATUS_REVIEW
        senior = any(category.startswith(p) for p in SENIOR_PEP)
        if tp_flag and (risk == "High" or senior or group == "Sanctions"):
            queue = QUEUE_EDD
            reasons.append("Confirmed match on a high-risk or senior-PEP subject: EDD decides the relationship.")
        elif tp_flag or group == "Sanctions" or risk == "High" or (intake["decision"] == "Reject" and screening["hard_stops"]):
            queue = QUEUE_L2
            reasons.append("Confirmed match, sanctions hit or high risk: L2 compliance review.")
        else:
            queue = QUEUE_L1
            reasons.append("The agent could not clear or confirm the record: an L1 analyst adjudicates.")
        if group == "Sanctions" or (tp_flag and risk == "High") or "fraud_referral" in screening["hard_stops"]:
            priority = "P1"
        elif group in ("PEP", "Law Enforcement", "Regulatory Enforcement") or risk == "High":
            priority = "P2"
        else:
            priority = "P3"
    sla = SLA_HOURS[priority]
    case.update(status=status, queue=queue, priority=priority, sla_hours=sla, due_at=dates.add_hours(received, sla),
                group="automated" if status in AUTOMATED else "human_open")
    if message:
        case["customer_message"] = message
    findings.append(finding("STATUS", f"{status} → {queue}, priority {priority}, SLA {sla} h.", "good" if status in AUTOMATED else "warning"))
    return step("triage", started, status, confidence=None, findings=findings, reasons=reasons,
                needs_human=(status == STATUS_REVIEW), reason_code=screening.get("reason_code", "") or intake.get("reason_code", ""),
                inputs={"intake": intake["decision"], "adjudication": screening["decision"], "risk": risk, "category": category},
                queue=queue, priority=priority)
