"""The hub's screens and actions on the Flask test client. The engine is
covered in test_hub.py; this file is about the blueprint: its own sign-in,
every page, the bulk outcome, the settings, the resets, and the two apps
leaving each other alone.

Run:  .venv\\Scripts\\python.exe -m unittest -v tests.test_hub_ui
"""
import os
import re
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
_DB = os.path.join(tempfile.mkdtemp(prefix="fcc-hub-ui-"), "state.db")
os.environ.setdefault("FCC_STATE_DB", _DB)
os.environ["CONSOLE_CASES"] = "demo"      # the scripted book these tests are written against
from fincrime.data.state_store import StateStore  # noqa: E402
if StateStore.path != os.environ["FCC_STATE_DB"]:
    StateStore.path = os.environ["FCC_STATE_DB"]
import app as fcc  # noqa: E402
from fincrime.hub import state as hub_state  # noqa: E402
from fincrime.hub.agents import triage  # noqa: E402

PAGES = ("/hub/", "/hub/intake", "/hub/screening", "/hub/cases", "/hub/agents", "/hub/accuracy", "/hub/audit")
FORBIDDEN = ("TikTok", "TT Live", "TT Shop", "tenant", "ttlive", "ttshop")


def flash_text(response):
    return " | ".join(re.findall(r'role="status"[^>]*>\s*<span[^>]*>[^<]*</span>\s*<span>([^<]+)</span>',
                                 response.data.decode("utf-8")))


class HubUiTests(unittest.TestCase):

    def setUp(self):
        fcc.app.testing = True
        fcc.app.secret_key = "test"
        StateStore.clear()
        fcc._load_seed()
        hub_state.reload()
        self.c = fcc.app.test_client()

    def login(self, key="m.chen"):
        r = self.c.post("/hub/sign-in", data={"analyst": key, "password": "1234"})
        self.assertEqual(r.status_code, 302)
        return r

    def first(self, **where):
        for c in hub_state.BOOK:
            if all(c.get(k) == v for k, v in where.items()):
                return c
        raise AssertionError(f"no case with {where}")

    # sign-in and the gate ---------------------------------------------------

    def test_signed_out_is_sent_to_the_hub_sign_in(self):
        r = self.c.get("/hub/")
        self.assertEqual(r.status_code, 302)
        self.assertIn("/hub/sign-in", r.headers["Location"])
        self.assertEqual(self.c.get("/hub/sign-in").status_code, 200)
        r = self.c.post("/hub/sign-in", data={"analyst": "m.chen", "password": "nope"})
        self.assertEqual(r.status_code, 401)
        self.assertEqual(self.c.post("/hub/sign-in", data={"analyst": "nobody", "password": "1234"}).status_code, 400)

    def test_every_page_renders_for_each_role(self):
        for key in ("m.chen", "j.vance", "h.pham"):
            self.login(key)
            for path in PAGES:
                r = self.c.get(path)
                self.assertEqual(r.status_code, 200, f"{key} {path}")
                html = r.data.decode("utf-8")
                for word in FORBIDDEN:
                    self.assertNotIn(word, html, f"{word} on {path}")
                self.assertIn('class="hub"', html)
                self.assertIn("/hub/state/version", html)
            c = hub_state.BOOK[0]
            self.assertEqual(self.c.get(f"/hub/case/{c['id']}").status_code, 200)
            self.assertEqual(self.c.get("/hub/case/KYC-0000-0000").status_code, 404)
            self.c.post("/hub/sign-out")

    def test_documents_are_served_and_the_viewer_is_wired(self):
        self.login()
        c = next(x for x in hub_state.BOOK if x["synth"]["capture"] == "flat_scan")
        r = self.c.get(f"/hub/files/{c['id']}/passport.svg")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.mimetype, "image/svg+xml")
        self.assertIn(b"SPECIMEN", r.data)
        self.assertIn(b'data-capture="flat_scan"', r.data)
        self.assertEqual(self.c.get(f"/hub/files/{c['id']}/nope.svg").status_code, 404)
        html = self.c.get(f"/hub/case/{c['id']}").data.decode("utf-8")
        self.assertEqual(html.count("window.viewerInit = function"), 1)
        self.assertIn('id="hub-stage"', html)
        self.assertIn("flat scan", html.lower())

    def test_the_two_apps_leave_each_other_alone(self):
        self.login()
        self.assertEqual(self.c.get("/kyc/list").status_code, 302)        # no console session
        self.assertEqual(self.c.get("/hub/").status_code, 200)
        r = self.c.post("/login", data={"analyst": "amara.iyer", "password": "1234"})
        self.assertEqual(r.status_code, 302)
        self.assertEqual(self.c.get("/kyc/list").status_code, 200)
        self.assertEqual(self.c.get("/hub/").status_code, 302)            # the console sign-in cleared the hub session
        html = self.c.get("/").data.decode("utf-8")
        self.assertIn("/hub/sign-in", html) if "auth-option" in html else None
        self.c.post("/logout")
        self.assertIn("/hub/sign-in", self.c.get("/").data.decode("utf-8"))

    def test_state_version_and_search(self):
        self.login()
        r = self.c.get("/hub/state/version")
        self.assertEqual(r.status_code, 200)
        self.assertIn("v", r.get_json())
        c = hub_state.BOOK[5]
        r = self.c.get("/hub/search", query_string={"q": c["id"].lower()})
        self.assertIn(f"/hub/case/{c['id']}", r.headers["Location"])
        r = self.c.get("/hub/search", query_string={"q": "nguyen"})
        self.assertIn("/hub/cases", r.headers["Location"])
        self.assertEqual(self.c.get("/hub/cases", query_string={"q": "nguyen", "page": "2"}).status_code, 200)

    # actions ----------------------------------------------------------------

    def test_decision_from_the_case_page(self):
        self.login()
        c = self.first(status=triage.STATUS_REVIEW, queue=triage.QUEUE_L1)
        r = self.c.post(f"/hub/case/{c['id']}/decide", data={"action": "clear", "comment": "Year only; different person"}, follow_redirects=True)
        self.assertIn("Clear as false positive", flash_text(r))
        self.assertEqual(c["status"], "Closed – False positive")
        self.assertEqual(c["decision"]["by"], "m.chen")
        r = self.c.post(f"/hub/case/{c['id']}/decide", data={"action": "clear"}, follow_redirects=True)
        self.assertIn("already closed", flash_text(r))
        self.assertIn("toast error", r.data.decode("utf-8"))

    def test_role_refusal_lands_back_where_the_analyst_was(self):
        self.login("m.chen")
        c = self.first(status=triage.STATUS_REVIEW, queue=triage.QUEUE_EDD)
        r = self.c.post(f"/hub/case/{c['id']}/decide", data={"action": "confirm_edd"},
                        headers={"Referer": f"http://localhost/hub/case/{c['id']}"})
        self.assertEqual(r.status_code, 302)
        self.assertIn(f"/hub/case/{c['id']}", r.headers["Location"])
        r = self.c.get(r.headers["Location"])
        self.assertIn("needs L2 or Team Lead access", flash_text(r))
        self.login("j.vance")
        r = self.c.post(f"/hub/case/{c['id']}/decide", data={"action": "confirm_edd", "comment": "EDD approved"}, follow_redirects=True)
        self.assertIn("EDD approved", flash_text(r))
        self.assertTrue(c["status"].startswith("Closed"))

    def test_bulk_outcome_on_ticked_cases(self):
        self.login()
        rows = [c for c in hub_state.BOOK if c["status"] == triage.STATUS_REVIEW and c["queue"] == triage.QUEUE_L1][:3]
        r = self.c.post("/hub/screening/bulk", data={"case_id": [c["id"] for c in rows], "action": "escalate", "comment": "L2 please"},
                        headers={"Referer": "http://localhost/hub/screening?queue=L1+review"}, follow_redirects=True)
        self.assertIn("Escalate to L2 compliance recorded for 3 case(s)", flash_text(r))
        self.assertTrue(all(c["status"] == "Escalated" and c["queue"] == triage.QUEUE_L2 for c in rows))
        r = self.c.post("/hub/screening/bulk", data={"action": "clear"}, follow_redirects=True)
        self.assertIn("Tick at least one case", flash_text(r))
        html = self.c.get("/hub/screening").data.decode("utf-8")
        self.assertIn('id="pick-all"', html)
        self.assertIn('name="pick"', html)

    def test_intake_override_from_the_intake_page(self):
        self.login()
        c = next(x for x in hub_state.BOOK if x["status"] == triage.STATUS_AUTO_REJECTED and x["screening"]["decision"] == "auto_clear")
        r = self.c.post(f"/hub/case/{c['id']}/intake", data={"action": "accept"},
                        headers={"Referer": "http://localhost/hub/intake"}, follow_redirects=True)
        self.assertIn("documents accepted", flash_text(r))
        self.assertEqual(c["status"], triage.STATUS_AUTO_CLEARED)
        self.assertEqual(c["audit"][-1]["action"], "Intake override")
        self.assertIn("Intake override", self.c.get("/hub/audit").data.decode("utf-8"))
        r = self.c.get("/hub/audit", query_string={"format": "csv"})
        self.assertEqual(r.mimetype, "text/csv")
        self.assertIn(c["id"].encode(), r.data)

    # settings, re-run, persistence, reset ------------------------------------

    def test_settings_are_the_leads_and_move_the_split(self):
        self.login("m.chen")
        r = self.c.post("/hub/agents/settings", data={"gate": "95"}, headers={"Referer": "http://localhost/hub/agents"}, follow_redirects=True)
        self.assertIn("needs Team Lead access", flash_text(r))
        self.assertEqual(hub_state.SETTINGS["gate"], 85)
        self.login("h.pham")
        before = hub_state.SETTINGS["last_run"]["adjudicated_pct"]
        r = self.c.post("/hub/agents/settings", data={"gate": "95", "sanctions_guard": ["0", "1"], "auto_rfi": ["0"]}, follow_redirects=True)
        self.assertIn("Confidence gate 85 → 95", flash_text(r))
        self.assertIn("Auto-RFI on an unresolved address True → False", flash_text(r))
        self.assertEqual(hub_state.SETTINGS["gate"], 95)
        r = self.c.post("/hub/agents/rerun", follow_redirects=True)
        text = flash_text(r)
        self.assertIn("→", text)
        self.assertLess(hub_state.SETTINGS["last_run"]["adjudicated_pct"], before)
        r = self.c.post("/hub/agents/settings", data={"gate": "200"}, follow_redirects=True)
        self.assertIn("between 50 and 99", flash_text(r))
        html = self.c.get("/hub/agents").data.decode("utf-8")
        self.assertIn("Confidence gate", html)
        self.assertIn('value="95"', html)

    def test_decisions_and_settings_survive_a_reload_and_a_reset_clears_them(self):
        self.login("h.pham")
        c = self.first(status=triage.STATUS_REVIEW, queue=triage.QUEUE_L1)
        self.c.post(f"/hub/case/{c['id']}/decide", data={"action": "clear", "comment": "ok"})
        self.c.post("/hub/agents/settings", data={"gate": "90"})
        hub_state.reload()
        again = hub_state.BY_ID[c["id"]]
        self.assertEqual(again["status"], "Closed – False positive")
        self.assertEqual(hub_state.SETTINGS["gate"], 90)
        r = self.c.post("/hub/reset", follow_redirects=True)
        self.assertIn("Hub reset", flash_text(r))
        self.assertEqual(hub_state.BY_ID[c["id"]]["status"], triage.STATUS_REVIEW)
        self.assertEqual(hub_state.SETTINGS["gate"], 85)
        self.login("m.chen")
        r = self.c.post("/hub/reset", follow_redirects=True)
        self.assertIn("needs Team Lead access", flash_text(r))

    def test_console_reset_reseeds_the_hub_too(self):
        self.login("m.chen")
        c = self.first(status=triage.STATUS_REVIEW, queue=triage.QUEUE_L1)
        self.c.post(f"/hub/case/{c['id']}/decide", data={"action": "clear", "comment": "ok"})
        self.c.post("/login", data={"analyst": "amara.iyer", "password": "1234"})
        r = self.c.post("/reset", follow_redirects=True)
        self.assertEqual(r.status_code, 200)
        self.assertEqual(hub_state.BY_ID[c["id"]]["status"], triage.STATUS_REVIEW)

    def test_overview_numbers_match_the_book(self):
        self.login()
        html = self.c.get("/hub/").data.decode("utf-8")
        split = hub_state.SETTINGS["last_run"]
        self.assertIn(f"{split['adjudicated_pct']}", html)
        self.assertIn("Where the split comes from", html)
        self.assertIn("Cleared by agents", html)
        acc = self.c.get("/hub/accuracy").data.decode("utf-8")
        self.assertIn("Auto-clear precision", acc)
        self.assertIn("100", acc)


if __name__ == "__main__":
    unittest.main()
