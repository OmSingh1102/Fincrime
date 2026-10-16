"""The hub's in-memory book and its persistence.

The 1,000 cases are built from the CSV at boot and run through the agents;
only cases somebody touched (a decision, an override, a Claude step) and
the settings record are written to the SQLite store, under their own kinds,
and laid back over the fresh run on the next boot - the same contract the
console uses. Reset drops only the hub's kinds.
"""
import os

from fincrime.data.state_store import StateStore
from fincrime.hub import documents, loader, pipeline
from fincrime.utils import dates

HUB_SEED_VERSION = "2026-10-07.1"
KINDS = ("hub_case", "hub_settings")
VERSION_KEY = "hub_seed_version"

BOOK, BY_ID = [], {}
SETTINGS = {}
ANALYSTS = loader.analysts()
ANALYST_BY_KEY = {a["key"]: a for a in ANALYSTS}
PLAYBOOK = loader.playbook()
STATE = {"version": 0, "booted_at": "", "boot": {}, "restored": 0}


def _serving():
    """Only the process that serves requests may stamp or clear the store
    (the dev reloader's parent must not)."""
    debug = os.environ.get("DEBUG", "1") == "1"
    return (not debug) or os.environ.get("WERKZEUG_RUN_MAIN") == "true"


def bump():
    STATE["version"] += 1


def version():
    return STATE["version"]


def boot(log=None):
    """Build the book, run the agents, lay stored state back on top."""
    BOOK[:] = loader.load_cases()
    BY_ID.clear()
    BY_ID.update({c["id"]: c for c in BOOK})
    SETTINGS.clear()
    SETTINGS.update(pipeline.default_settings())
    stored = StateStore.setting(VERSION_KEY)
    if stored and stored != HUB_SEED_VERSION and _serving():
        if log:
            log("Hub seed version changed (%s -> %s); clearing the hub's stored state.", stored, HUB_SEED_VERSION)
        StateStore.clear(kinds=KINDS)
        stored = ""
    if not stored and _serving():
        StateStore.set_setting(VERSION_KEY, HUB_SEED_VERSION)
    StateStore.restore("hub_settings", {"hub_settings": SETTINGS})
    for key, value in pipeline.default_settings().items():
        SETTINGS.setdefault(key, value)
    for c in BOOK:
        documents.synthesise(c)
    summary = pipeline.run_all(BOOK, SETTINGS, reason="boot")
    restored = StateStore.restore("hub_case", BY_ID)
    STATE.update(booted_at=dates.now_str(), boot=summary, restored=restored)
    bump()
    if log:
        log("Hub: %d cases, %s%% AI-adjudicated (%s%% without an analyst) in %ss; %d stored record(s) restored.",
            summary["n"], summary["adjudicated_pct"], summary["automated_pct"], summary["seconds"], restored)
    return summary


def reload(log=None):
    return boot(log)


def reset(log=None):
    """Drop the hub's stored state and rebuild. Returns the boot summary."""
    StateStore.clear(kinds=KINDS)
    return boot(log)


def save_case(case):
    bump()
    StateStore.save("hub_case", case)


def save_settings():
    bump()
    StateStore.save("hub_settings", SETTINGS)


def rerun(actor="System", reason="settings changed"):
    """Re-run the agents over every case nobody decided; returns (before, after)."""
    before = dict(SETTINGS.get("last_run") or pipeline.split(BOOK))
    after = pipeline.run_all(BOOK, SETTINGS, reason=reason, actor=actor)
    save_settings()
    return before, after


def case(case_id):
    return BY_ID.get(case_id)
