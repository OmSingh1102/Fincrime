"""What an analyst does to a case. Every action checks the role and the
case's state, writes the decision and an audit row, and returns the
message the screen shows. A refusal is a WorkflowError."""
from fincrime.engines.errors import WorkflowError
from fincrime.hub.agents import triage
from fincrime.utils import dates

ALL = ("analyst", "l2", "lead")
SENIOR = ("l2", "lead")

ACTIONS = {
    "clear": {"label": "Clear as false positive", "outcome": "False Positive", "status": "Closed – False positive",
              "roles": ALL, "group": "human_closed"},
    "confirm_edd": {"label": "Confirm match – EDD approved", "outcome": "True Positive", "status": "Closed – True positive (EDD approved)",
                    "roles": SENIOR, "group": "human_closed"},
    "reject": {"label": "Confirm match – application rejected", "outcome": "True Positive", "status": "Closed – Application rejected",
               "roles": SENIOR, "group": "human_closed"},
    "exit": {"label": "Confirm match – relationship exited", "outcome": "True Positive", "status": "Closed – Relationship exited",
             "roles": SENIOR, "group": "human_closed"},
    "block": {"label": "Confirm match – account blocked", "outcome": "True Positive", "status": "Closed – Account blocked",
              "roles": SENIOR, "group": "human_closed"},
    "escalate": {"label": "Escalate to L2 compliance", "outcome": "Potential Match", "status": "Escalated", "queue": triage.QUEUE_L2,
                 "roles": ALL, "group": "human_open", "priority": "P1"},
    "pend": {"label": "Pend – awaiting customer documents", "outcome": "", "status": "Pending (customer)", "queue": triage.QUEUE_INTAKE,
             "roles": ALL, "group": "human_open"},
    "rfi": {"label": "Request for information sent", "outcome": "", "status": "Pending (customer)", "queue": triage.QUEUE_INTAKE,
            "roles": ALL, "group": "human_open"},
}
INTAKE_ACTIONS = {"accept": "Documents accepted by the analyst (override)", "resend": "Customer message re-sent",
                  "refer": "Referred to L2 as a possible fraud upload"}


def _require(cond, msg):
    if not cond:
        raise WorkflowError(msg)


def _audit(case, actor, action, detail=""):
    case.setdefault("audit", []).append({"at": dates.now_str(), "actor": actor, "action": action, "detail": detail})


def is_closed(case):
    return (case.get("status") or "").startswith("Closed")


def allowed(role, action):
    spec = ACTIONS.get(action)
    return bool(spec) and role in spec["roles"]


def decide(case, actor, role, action, comment=""):
    """Record the analyst's decision on a case. Returns the flash message."""
    spec = ACTIONS.get(action)
    _require(spec, "Unknown action.")
    _require(not is_closed(case), f"{case['id']} is already closed ({case['status']}).")
    _require(role in spec["roles"], f"{spec['label']} needs {' or '.join(r.upper() if r != 'lead' else 'Team Lead' for r in spec['roles'])} access.")
    if action in ("rfi", "pend"):
        _require((comment or "").strip() or action == "pend", "Say what the customer must send.")
    case["decision"] = {"outcome": spec["outcome"], "action": spec["label"], "basis": "human", "by": actor,
                        "at": dates.now_str(), "comment": (comment or "").strip()}
    case["status"] = spec["status"]
    case["group"] = spec["group"]
    case["queue"] = spec.get("queue", "")
    if spec.get("priority"):
        case["priority"] = spec["priority"]
        case["sla_hours"] = triage.SLA_HOURS[spec["priority"]]
        case["due_at"] = dates.add_hours(dates.now_str(), case["sla_hours"])
    if action == "rfi":
        case["customer_message"] = comment.strip()
    case["locked"] = True
    detail = spec["label"] + (f": {comment.strip()}" if comment and comment.strip() else "")
    _audit(case, actor, "Decision", detail)
    return f"{case['id']}: {spec['label']}."


def intake_override(case, actor, role, action, settings):
    """The analyst overrides, re-sends or refers a document decision."""
    _require(action in INTAKE_ACTIONS, "Unknown intake action.")
    _require(not is_closed(case), f"{case['id']} is already closed.")
    intake = case.get("intake") or {}
    if action == "accept":
        _require(intake.get("decision") in ("Reject", "Refer"), "Nothing to override: the documents were accepted.")
        intake["override"] = {"by": actor, "at": dates.now_str(), "was": intake.get("decision")}
        intake["decision"] = "Accept"
        case["locked"] = True
        trail_step = triage.run(case, settings)
        case["trail"].append(trail_step)
        _audit(case, actor, "Intake override", f"Documents accepted; was {intake['override']['was']} ({intake.get('reason_code', '')}). Now {case['status']} → {case['queue']}.")
        return f"{case['id']}: documents accepted; the case is now {case['status']} ({case['queue']})."
    if action == "refer":
        _require(role in SENIOR or True, "")
        intake["decision"] = "Refer"
        intake["reason_code"] = intake.get("reason_code") or "ANALYST_REFERRAL"
        case["locked"] = True
        case["screening"]["hard_stops"] = sorted(set(case["screening"].get("hard_stops", [])) | {"fraud_referral"})
        trail_step = triage.run(case, settings)
        case["trail"].append(trail_step)
        _audit(case, actor, "Fraud referral", "Documents referred to L2 as a possible fraud upload.")
        return f"{case['id']}: referred to L2 compliance as a possible fraud upload."
    _require(bool(case.get("customer_message")), "There is no customer message on this case.")
    _audit(case, actor, "Customer message re-sent", case["customer_message"][:120])
    return f"{case['id']}: the customer message was sent again."
