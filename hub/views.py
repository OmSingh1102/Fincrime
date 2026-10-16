"""The hub's Flask blueprint: its own sign-in, its screens and its actions.

Mounted at /hub by app.py. The session keys are the hub's own (hub_user,
hub_name, hub_role) so a console session and a hub session do not collide;
the console's before_request hook skips every endpoint of this blueprint
and the guard below takes over.
"""
import csv
import hmac
import io
import os
from functools import wraps
from urllib.parse import urlsplit

from flask import (Blueprint, Response, abort, current_app, flash, jsonify, redirect, render_template, request, session,
                   url_for)

from fincrime.engines.errors import WorkflowError
from fincrime.hub import actions, analytics, documents, llm, pipeline, state
from fincrime.hub.agents import AGENTS, RUN_ORDER, llm_agents, triage
from fincrime.hub.agents.intake import REASONS
from fincrime.utils import dates

bp = Blueprint("hub", __name__)

PRODUCT = "Straive KYC Operations Hub"
TAGLINE = "AI agents with a human in the loop"
DEMO_PASSWORD = os.environ.get("STRAIVE_DEMO_PASSWORD", "1234")
DEMO_MODE = os.environ.get("DEMO_MODE", "1") == "1"
SHOW_HINT = "STRAIVE_DEMO_PASSWORD" not in os.environ
OPEN_ENDPOINTS = {"hub.sign_in", "hub.sign_out"}
PAGE = 50

NAV = (("hub.overview", "Overview", "grid"), ("hub.intake", "Intake & documents", "file-text"),
       ("hub.screening", "Screening review", "shield"), ("hub.cases", "Cases", "list"), ("hub.agents", "Agents", "cpu"),
       ("hub.accuracy", "Accuracy", "bar-chart"), ("hub.audit", "Audit log", "book"))

STATUS_TONE = {triage.STATUS_AUTO_CLEARED: "good", triage.STATUS_AUTO_REJECTED: "info", triage.STATUS_AUTO_RFI: "info",
               triage.STATUS_REVIEW: "warning", "Escalated": "serious", "Pending (customer)": "info"}
DOB_LABEL = {"exact": "Exact date", "transposed": "Day/month transposed", "conflict": "Date conflict", "near": "Within two years",
             "year_match": "Year only", "year_conflict": "Year conflict", "year_near": "Year ± 1", "range_match": "Birth range",
             "range_conflict": "Outside the range", "range_near": "Edge of the range", "missing": "Not listed"}
DOB_TONE = {"exact": "serious", "transposed": "warning", "conflict": "good", "near": "warning", "year_match": "warning",
            "year_conflict": "good", "year_near": "warning", "range_match": "warning", "range_conflict": "good",
            "range_near": "warning", "missing": "neutral"}
REASON_LABEL = {"EXACT_MATCH": "Confirmed match", "DOB_CONFLICT": "Date of birth conflict", "DOB_TRANSPOSED": "Day/month transposed",
                "DOB_YEAR_ONLY": "Year of birth only", "DOB_RANGE": "Birth range only", "DOB_MISSING": "No date of birth",
                "DOB_NEAR": "Dates within two years", "NAME_WEAK": "Exact date, weak name", "SCRIPT_CONFLICT": "Native-script names differ",
                "SANCTIONS_GUARD": "Sanctions guard", "FRAUD_REFERRAL": "Fraud referral", "BELOW_GATE": "Below the gate",
                "AGENT_OFF": "Agent switched off"}
GROUP_LABEL = {"automated": "Automated", "human_open": "Open for a person", "human_closed": "Closed by a person"}
ACTION_HELP = {"clear": "Record a false positive and close the case.", "confirm_edd": "Same person; the relationship continues under EDD.",
               "reject": "Same person; the application is declined.", "exit": "Same person; the relationship is exited.",
               "block": "Same person; the account is blocked.", "escalate": "Hand the case to L2 compliance.",
               "pend": "Wait for the customer's documents.", "rfi": "Send the customer a request for information (the comment is the message)."}


# ── session helpers ──────────────────────────────────────────────────────────

def _analyst():
    return state.ANALYST_BY_KEY.get(session.get("hub_user"))


def _actor():
    return session.get("hub_user", "")


def _role():
    return (_analyst() or {}).get("role", "")


def _safe_next(target, fallback=None):
    if target and target.startswith("/") and target[1:2] not in ("/", "\\"):
        return target
    return fallback or url_for("hub.overview")


def _back(fallback=None):
    ref = request.referrer or ""
    parts = urlsplit(ref)
    if ref and (not parts.netloc or parts.netloc == request.host) and (parts.path or "").startswith("/hub"):
        return parts.path + (("?" + parts.query) if parts.query else "")
    return fallback or url_for("hub.overview")


def _sign_in(analyst):
    for key in ("hub_user", "hub_name", "hub_role"):
        session.pop(key, None)
    session["hub_user"] = analyst["key"]
    session["hub_name"] = analyst["name"]
    session["hub_role"] = analyst["role"]


def role_required(*roles):
    def decorator(view):
        @wraps(view)
        def wrapper(*args, **kwargs):
            if _role() not in roles:
                names_ = [{"lead": "Team Lead", "l2": "L2 Compliance", "analyst": "Analyst"}[r] for r in roles]
                flash(f"That action needs {' or '.join(names_)} access.", "error")
                return redirect(_back())
            return view(*args, **kwargs)
        return wrapper
    return decorator


def llm_status():
    """Claude availability for the chips and buttons."""
    return llm.status(state.SETTINGS)


@bp.before_request
def _guard():
    if request.endpoint in OPEN_ENDPOINTS:
        return None
    if not _analyst():
        for key in ("hub_user", "hub_name", "hub_role"):
            session.pop(key, None)
        return redirect(url_for("hub.sign_in", next=request.path))
    return None


@bp.context_processor
def _inject():
    a = _analyst()
    counts = {}
    if a:
        counts = {"hub.intake": sum(1 for c in state.BOOK if c.get("queue") == triage.QUEUE_INTAKE and not c.get("status", "").startswith("Closed")),
                  "hub.screening": sum(1 for c in state.BOOK if c.get("group") == "human_open" and c.get("queue") != triage.QUEUE_INTAKE)}
    nav = [{"endpoint": e, "label": l, "icon": i, "active": (request.endpoint == e) or (e == "hub.cases" and request.endpoint == "hub.case"),
            "count": counts.get(e, 0)} for e, l, i in NAV]
    return {"product": PRODUCT, "tagline": TAGLINE, "hub_user": a, "hub_role": (a or {}).get("role", ""),
            "hub_analysts": state.ANALYSTS, "hub_nav": nav, "hub_version": state.version(),
            "state_version": state.version(), "poll_url": url_for("hub.state_version"), "demo_mode": DEMO_MODE,
            "demo_password": DEMO_PASSWORD if SHOW_HINT else "", "llm": llm_status(), "settings": state.SETTINGS,
            "status_tone": STATUS_TONE, "dob_label": DOB_LABEL, "dob_tone": DOB_TONE, "reason_label": REASON_LABEL,
            "group_label": GROUP_LABEL, "reasons": REASONS, "actions_spec": actions.ACTIONS, "action_help": ACTION_HELP,
            "now": dates.now_str(), "triage": triage, "claude_routes": "hub.case_ai" in current_app.view_functions}


@bp.app_template_filter("hub_dt")
def _hub_dt(stamp):
    d = dates.parse(stamp)
    return d.strftime("%d %b %Y %H:%M") if d else (stamp or "")


@bp.app_template_filter("hub_date")
def _hub_date(stamp):
    d = dates.parse(stamp)
    return d.strftime("%d %b %Y") if d else (stamp or "")


@bp.app_template_filter("hub_hours")
def _hub_hours(stamp):
    """Hours since `stamp`, as '3 h' / '2 d'."""
    d = dates.parse(stamp)
    if not d:
        return ""
    h = (dates.now() - d).total_seconds() / 3600.0
    return f"{h / 24:.0f} d" if h >= 48 else f"{h:.0f} h"


@bp.app_template_filter("hub_left")
def _hub_left(stamp):
    """Time until `stamp`, as '9 h' / '2 d' (absolute: the caller says overdue or left)."""
    d = dates.parse(stamp)
    if not d:
        return ""
    h = abs((d - dates.now()).total_seconds()) / 3600.0
    return f"{h / 24:.0f} d" if h >= 48 else f"{h:.0f} h"


@bp.app_template_filter("hub_pct")
def _hub_pct(value):
    try:
        return f"{float(value):.0f}%"
    except (TypeError, ValueError):
        return "-"


@bp.app_template_filter("hub_num")
def _hub_num(value):
    try:
        return f"{int(value):,}"
    except (TypeError, ValueError):
        return str(value)


# ── sign-in ──────────────────────────────────────────────────────────────────

@bp.route("/sign-in", methods=["GET", "POST"])
def sign_in():
    next_url = request.values.get("next", "")
    error, selected = "", request.form.get("analyst") or state.ANALYSTS[0]["key"]
    status = 200
    if request.method == "POST":
        analyst = state.ANALYST_BY_KEY.get(request.form.get("analyst", ""))
        if not analyst:
            error, status = "Choose an analyst profile to sign in as.", 400
        elif not hmac.compare_digest((request.form.get("password") or "").encode("utf-8"), DEMO_PASSWORD.encode("utf-8")):
            error, status = "Incorrect password.", 401
        else:
            _sign_in(analyst)
            return redirect(_safe_next(next_url))
    elif _analyst():
        return redirect(_safe_next(next_url))
    return render_template("hub/sign_in.html", error=error, selected=selected, next=next_url), status


@bp.route("/sign-out", methods=["POST"])
def sign_out():
    for key in ("hub_user", "hub_name", "hub_role"):
        session.pop(key, None)
    return redirect(url_for("hub.sign_in"))


@bp.route("/switch", methods=["POST"])
def switch():
    if not DEMO_MODE:
        abort(404)
    analyst = state.ANALYST_BY_KEY.get(request.form.get("analyst", ""))
    if analyst:
        _sign_in(analyst)
        flash(f"Now working as {analyst['name']} ({analyst['role_label']}).", "success")
    return redirect(_safe_next(request.form.get("next", "")))


@bp.route("/state/version")
def state_version():
    return jsonify({"v": state.version()})


# ── screens ──────────────────────────────────────────────────────────────────

@bp.route("/")
def overview():
    vm = analytics.overview(state.BOOK, state.SETTINGS)
    return render_template("hub/overview.html", vm=vm, agents=AGENTS)


def _filter(rows, country="", group="", queue="", priority="", reason="", q="", decision="", outcome="", status=""):
    q = (q or "").strip().lower()
    out = []
    for c in rows:
        if country and c["country"] != country:
            continue
        if group and c["wc"]["group"] != group:
            continue
        if queue and c.get("queue") != queue:
            continue
        if priority and c.get("priority") != priority:
            continue
        if reason and (c.get("screening") or {}).get("reason_code") != reason:
            continue
        if decision and (c.get("intake") or {}).get("decision") != decision and not (decision == "RFI" and c.get("status") in (triage.STATUS_AUTO_RFI, "Pending (customer)")):
            continue
        if outcome and c["history"]["outcome"] != outcome:
            continue
        if status and c.get("group") != status:
            continue
        if q:
            hay = " ".join([c["id"], c["customer"]["name"], (c.get("identity") or {}).get("name", {}).get("canonical", ""),
                            c["wc"]["name_raw"], c["wc"]["category"]]).lower()
            if q not in hay:
                continue
        out.append(c)
    return out


def _args(*keys):
    return {k: request.args.get(k, "") for k in keys}


@bp.route("/intake")
def intake():
    rows = analytics.intake_rows(state.BOOK)
    f = _args("country", "decision", "reason", "q")
    shown = [c for c in _filter(rows, country=f["country"], decision=f["decision"], q=f["q"])
             if not f["reason"] or (c.get("intake") or {}).get("reason_code") == f["reason"] or
             (f["reason"] == "ADDR_UNRESOLVED" and c.get("status") == triage.STATUS_AUTO_RFI)]
    counts = {"Reject": sum(1 for c in rows if (c.get("intake") or {}).get("decision") == "Reject"),
              "Refer": sum(1 for c in rows if (c.get("intake") or {}).get("decision") == "Refer"),
              "RFI": sum(1 for c in rows if c.get("status") in (triage.STATUS_AUTO_RFI, "Pending (customer)"))}
    codes = sorted({(c.get("intake") or {}).get("reason_code") for c in rows if (c.get("intake") or {}).get("reason_code")})
    return render_template("hub/intake.html", rows=shown, total=len(rows), counts=counts, codes=codes, f=f)


@bp.route("/screening")
def screening():
    rows = [c for c in analytics.screening_rows(state.BOOK) if c.get("queue") != triage.QUEUE_INTAKE]
    f = _args("country", "group", "queue", "priority", "reason", "q")
    shown = _filter(rows, **f)
    options = {"reasons": sorted({(c.get("screening") or {}).get("reason_code", "") for c in rows} - {""}),
               "queues": [triage.QUEUE_L1, triage.QUEUE_L2, triage.QUEUE_EDD], "groups": sorted({c["wc"]["group"] for c in rows})}
    return render_template("hub/screening.html", rows=shown, total=len(rows), f=f, options=options)


@bp.route("/screening/bulk", methods=["POST"])
@role_required("analyst", "l2", "lead")
def screening_bulk():
    ids = request.form.getlist("case_id")
    action, comment = request.form.get("action", ""), request.form.get("comment", "")
    if not ids:
        flash("Tick at least one case first.", "error")
        return redirect(_back(url_for("hub.screening")))
    done, errors = 0, []
    for cid in ids:
        c = state.case(cid)
        if c is None:
            errors.append(f"{cid}: not found")
            continue
        try:
            actions.decide(c, _actor(), _role(), action, comment)
            state.save_case(c)
            done += 1
        except WorkflowError as exc:
            errors.append(f"{cid}: {exc}")
    spec = actions.ACTIONS.get(action, {"label": action})
    if done:
        flash(f"{spec['label']} recorded for {done} case(s).", "success")
    for e in errors[:5]:
        flash(e, "error")
    return redirect(_back(url_for("hub.screening")))


@bp.route("/cases")
def cases():
    f = _args("status", "country", "outcome", "q")
    shown = _filter(state.BOOK, status=f["status"], country=f["country"], outcome=f["outcome"], q=f["q"])
    try:
        page = max(1, int(request.args.get("page", "1")))
    except ValueError:
        page = 1
    pages = max(1, (len(shown) + PAGE - 1) // PAGE)
    page = min(page, pages)
    rows = shown[(page - 1) * PAGE: page * PAGE]
    return render_template("hub/cases.html", rows=rows, total=len(shown), page=page, pages=pages, f=f)


@bp.route("/search")
def search():
    q = (request.args.get("q") or "").strip()
    if q.upper() in state.BY_ID:
        return redirect(url_for("hub.case", case_id=q.upper()))
    if q.isdigit() and f"KYC-2026-{q}" in state.BY_ID:
        return redirect(url_for("hub.case", case_id=f"KYC-2026-{q}"))
    return redirect(url_for("hub.cases", q=q))


def _case(case_id):
    return state.case(case_id) or abort(404)


@bp.route("/case/<case_id>")
def case(case_id):
    c = _case(case_id)
    role = _role()
    allowed = [k for k, spec in actions.ACTIONS.items() if role in spec["roles"]]
    labels = (c.get("identity") or {}).get("variance_detected", [])
    playbook = [p for p in state.PLAYBOOK if p["variance"] in labels and p["country"] == c["country_name"]] or \
               [p for p in state.PLAYBOOK if p["variance"] in labels]
    return render_template("hub/case.html", c=c, allowed=allowed, playbook=playbook[:4], closed=actions.is_closed(c))


@bp.route("/files/<case_id>/<file_id>.svg")
def file_svg(case_id, file_id):
    c = _case(case_id)
    svg = documents.render(c, file_id)
    if svg is None:
        abort(404)
    return Response(svg, mimetype="image/svg+xml", headers={"Cache-Control": "no-store"})


@bp.route("/case/<case_id>/decide", methods=["POST"])
@role_required("analyst", "l2", "lead")
def case_decide(case_id):
    c = _case(case_id)
    try:
        msg = actions.decide(c, _actor(), _role(), request.form.get("action", ""), request.form.get("comment", ""))
    except WorkflowError as exc:
        flash(str(exc), "error")
        return redirect(url_for("hub.case", case_id=case_id))
    state.save_case(c)
    flash(msg, "success")
    return redirect(url_for("hub.case", case_id=case_id))


@bp.route("/case/<case_id>/intake", methods=["POST"])
@role_required("analyst", "l2", "lead")
def case_intake(case_id):
    c = _case(case_id)
    try:
        msg = actions.intake_override(c, _actor(), _role(), request.form.get("action", ""), state.SETTINGS)
    except WorkflowError as exc:
        flash(str(exc), "error")
        return redirect(_back(url_for("hub.case", case_id=case_id)))
    state.save_case(c)
    flash(msg, "success")
    return redirect(_back(url_for("hub.case", case_id=case_id)))


@bp.route("/case/<case_id>/ai/<which>", methods=["POST"])
@role_required("analyst", "l2", "lead")
def case_ai(case_id, which):
    """One Claude agent over the case: second_opinion, narrative or
    customer_message. Advisory; the result lands on the case page."""
    c = _case(case_id)
    try:
        _, msg = llm_agents.run(c, which, _actor(), state.SETTINGS, state.BOOK, proposed=request.form.get("proposed", ""))
    except WorkflowError as exc:
        flash(str(exc), "error")
        return redirect(url_for("hub.case", case_id=case_id))
    state.save_case(c)
    flash(msg, "success")
    return redirect(url_for("hub.case", case_id=case_id) + "#claude")


@bp.route("/agents")
def agents():
    stats = analytics.agent_stats(state.BOOK)
    split = pipeline.split(state.BOOK)
    groups = {}
    for p in state.PLAYBOOK:
        groups.setdefault((p["country"], p["field"]), []).append(p)
    return render_template("hub/agents.html", agents=AGENTS, stats=stats, split=split, playbook_groups=groups,
                           history=list(reversed(state.SETTINGS.get("history", [])))[:12], last_run=state.SETTINGS.get("last_run", {}),
                           can_edit=(_role() == "lead"), run_order=RUN_ORDER)


@bp.route("/agents/settings", methods=["POST"])
@role_required("lead")
def agents_settings():
    changes = []
    keys = ["gate", "doc_strictness", "sanctions_guard", "auto_rfi", "claude"] + [f"agent_{k}" for k in RUN_ORDER]
    try:
        for key in keys:
            values = request.form.getlist(key)
            if not values:
                continue
            before, after = pipeline.set_setting(state.SETTINGS, _actor(), key, values[-1])
            if before != after:
                changes.append(f"{pipeline.SETTING_LABELS.get(key, key)} {before} → {after}")
    except WorkflowError as exc:
        flash(str(exc), "error")
        return redirect(url_for("hub.agents"))
    state.save_settings()
    if changes:
        flash("Settings saved: " + "; ".join(changes) + ". Re-run the pipeline to apply them to the open book.", "success")
    else:
        flash("Nothing changed.", "success")
    return redirect(url_for("hub.agents"))


@bp.route("/agents/rerun", methods=["POST"])
@role_required("lead")
def agents_rerun():
    before, after = state.rerun(actor=_actor())
    flash(f"Pipeline re-run over {after['ran']} undecided case(s) in {after['seconds']} s at gate {after['gate']}: "
          f"AI-adjudicated {before.get('adjudicated_pct', '?')}% → {after['adjudicated_pct']}% "
          f"(no analyst {before.get('automated_pct', '?')}% → {after['automated_pct']}%).", "success")
    return redirect(url_for("hub.agents"))


@bp.route("/accuracy")
def accuracy():
    return render_template("hub/accuracy.html", vm=analytics.accuracy(state.BOOK))


@bp.route("/audit")
def audit():
    rows = analytics.audit_rows(state.BOOK)
    if request.args.get("format") == "csv":
        buf = io.StringIO()
        w = csv.writer(buf)
        w.writerow(["at", "actor", "action", "case", "name", "detail"])
        for r in rows:
            w.writerow([r["at"], r["actor"], r["action"], r["case"], r["name"], r["detail"]])
        return Response(buf.getvalue(), mimetype="text/csv; charset=utf-8",
                        headers={"Content-Disposition": 'attachment; filename="hub_audit.csv"'})
    return render_template("hub/audit.html", rows=rows, actors=sorted({r["actor"] for r in rows}), kinds=sorted({r["action"] for r in rows}))


@bp.route("/reset", methods=["POST"])
@role_required("lead")
def reset():
    summary = state.reset()
    flash(f"Hub reset: the {summary['n']} cases are back to the agents' run ({summary['adjudicated_pct']}% AI-adjudicated).", "success")
    return redirect(url_for("hub.overview"))
