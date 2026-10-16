"""The hub's Claude agents through the seam: configured or not, the three
tasks store their result on the case with a trail step and an audit row,
a gateway failure writes nothing, and the Agents-page switch turns the
buttons off. The suite never calls the real gateway.

Run:  .venv\\Scripts\\python.exe -m unittest -v tests.test_hub_llm
"""
import os
import re
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
_DB = os.path.join(tempfile.mkdtemp(prefix="fcc-hub-llm-"), "state.db")
os.environ.setdefault("FCC_STATE_DB", _DB)
os.environ["CONSOLE_CASES"] = "demo"      # the scripted book these tests are written against
from fincrime.data.state_store import StateStore  # noqa: E402
if StateStore.path != os.environ["FCC_STATE_DB"]:
    StateStore.path = os.environ["FCC_STATE_DB"]
import app as fcc  # noqa: E402
from fincrime.engines.errors import WorkflowError  # noqa: E402
from fincrime.hub import llm, state as hub_state  # noqa: E402
from fincrime.hub.agents import llm_agents, triage  # noqa: E402


def flash_text(response):
    return " | ".join(re.findall(r'role="status"[^>]*>\s*<span[^>]*>[^<]*</span>\s*<span>([^<]+)</span>',
                                 response.data.decode("utf-8")))


ANSWERS = {
    "second_opinion": {"verdict": "unclear", "confidence": 58, "summary": "Only the year of birth is listed; the names agree.",
                       "reasons": ["Year of birth matches", "Surname and given names agree"], "missing_data": ["full date of birth on the record"],
                       "suggested_outcome": "Potential Match", "what_would_resolve": ["the record's full date of birth", "an ID number"]},
    "narrative": {"narrative": "The watchlist record lists only a year of birth, which matches the client's; the names agree after normalisation. Escalated for a full-date check.",
                  "rationale_points": ["year only", "names agree"], "regulatory_note": "Retained under the screening procedure.", "record_corrections": []},
    "customer_message": {"subject": "One more document for your application", "body": "Hi Javier, please send a recent utility bill.", "channel": "email"},
}


class HubLlmTests(unittest.TestCase):

    def setUp(self):
        fcc.app.testing = True
        fcc.app.secret_key = "test"
        StateStore.clear()
        fcc._load_seed()
        hub_state.reload()
        self.c = fcc.app.test_client()
        self.saved = (llm.KEY, llm.call, dict(llm._state))
        self.calls = []

        def fake(system, text, schema, max_tokens=3000):
            self.calls.append((system, text, schema))
            which = "second_opinion" if "verdict" in schema["properties"] else "narrative" if "narrative" in schema["properties"] else "customer_message"
            return dict(ANSWERS[which]), {"input": 1200, "output": 300, "cached": 900}, 9.5

        self.fake = fake

    def tearDown(self):
        llm.KEY, llm.call = self.saved[0], self.saved[1]
        llm._state.clear()
        llm._state.update(self.saved[2])

    def login(self, key="m.chen"):
        self.assertEqual(self.c.post("/hub/sign-in", data={"analyst": key, "password": "1234"}).status_code, 302)

    def case(self):
        return next(c for c in hub_state.BOOK if c["status"] == triage.STATUS_REVIEW and c["queue"] == triage.QUEUE_L1)

    def test_not_configured(self):
        llm.KEY = ""
        self.login()
        c = self.case()
        html = self.c.get(f"/hub/case/{c['id']}").data.decode("utf-8")
        self.assertIn("Claude is not configured", html)
        self.assertIn("Claude not configured", html)
        r = self.c.post(f"/hub/case/{c['id']}/ai/second_opinion", follow_redirects=True)
        self.assertIn("Claude is not configured", flash_text(r))
        self.assertNotIn("second_opinion", c["ai"])
        self.assertFalse(any(s["kind"] == "claude" for s in c["trail"]))

    def test_three_tasks_through_the_seam(self):
        llm.KEY, llm.call = "test-key", self.fake
        self.login("j.vance")
        c = self.case()
        for which in ("second_opinion", "narrative", "customer_message"):
            r = self.c.post(f"/hub/case/{c['id']}/ai/{which}", follow_redirects=True)
            self.assertEqual(r.status_code, 200)
            self.assertIn("Claude", flash_text(r))
            self.assertIn(which, c["ai"])
            self.assertEqual(c["ai"][which]["by"], "j.vance")
            self.assertEqual(c["ai"][which]["seconds"], 9.5)
        self.assertEqual(c["ai"]["second_opinion"]["verdict"], "unclear")
        self.assertEqual(c["ai"]["second_opinion"]["suggested_outcome"], "Potential Match")
        self.assertEqual(c["ai"]["customer_message"]["channel"], "email")
        claude_steps = [s for s in c["trail"] if s["kind"] == "claude"]
        self.assertEqual(len(claude_steps), 3)
        self.assertTrue(all(s["needs_human"] for s in claude_steps))
        self.assertEqual(len(c["trail"]), 7)
        self.assertEqual([a["action"] for a in c["audit"][-3:]],
                         ["Claude second opinion", "Claude narrative drafted", "Claude customer message drafted"])
        self.assertEqual(llm.status(hub_state.SETTINGS)["calls"], 3)
        # the briefs never carry the historical outcome
        for _, text, _ in self.calls:
            self.assertNotIn(c["history"]["disposition"], text)
            self.assertNotIn("history", text)
        self.assertIn("style_examples_from_other_cases", self.calls[1][1])
        html = self.c.get(f"/hub/case/{c['id']}").data.decode("utf-8")
        self.assertIn("Claude: unclear", html)
        self.assertIn("Prefilled with Claude", html)
        self.assertIn("One more document for your application", html)
        # a human decision keeps the Claude steps, and a re-run leaves them alone
        self.c.post(f"/hub/case/{c['id']}/decide", data={"action": "escalate", "comment": "as Claude suggests"})
        hub_state.rerun(actor="h.pham")
        self.assertEqual(len([s for s in c["trail"] if s["kind"] == "claude"]), 3)
        hub_state.reload()
        self.assertIn("narrative", hub_state.BY_ID[c["id"]]["ai"])

    def test_a_gateway_failure_writes_nothing(self):
        llm.KEY = "test-key"

        def boom(system, text, schema, max_tokens=3000):
            raise WorkflowError("The model gateway answered 503: busy")
        llm.call = boom
        self.login()
        c = self.case()
        r = self.c.post(f"/hub/case/{c['id']}/ai/narrative", follow_redirects=True)
        self.assertIn("answered 503", flash_text(r))
        self.assertIn("toast error", r.data.decode("utf-8"))
        self.assertEqual(c["ai"], {})
        self.assertEqual(len(c["trail"]), 4)
        self.assertIn("503", llm.status(hub_state.SETTINGS)["last_error"])
        r = self.c.post(f"/hub/case/{c['id']}/ai/nonsense", follow_redirects=True)
        self.assertIn("Unknown Claude task", flash_text(r))

    def test_switched_off_on_the_agents_page(self):
        llm.KEY, llm.call = "test-key", self.fake
        self.login("h.pham")
        self.c.post("/hub/agents/settings", data={"claude": ["0"]})
        self.assertFalse(hub_state.SETTINGS["claude"])
        c = self.case()
        r = self.c.post(f"/hub/case/{c['id']}/ai/second_opinion", follow_redirects=True)
        self.assertIn("switched off", flash_text(r))
        self.assertEqual(self.calls, [])
        html = self.c.get("/hub/").data.decode("utf-8")
        self.assertIn("Claude off", html)

    def test_brief_shapes(self):
        c = self.case()
        import json
        b = json.loads(llm_agents.brief(c, "second_opinion"))
        self.assertEqual(set(b), {"client", "watchlist_record", "agent_findings"})
        self.assertEqual(b["client"]["legal_name"], c["identity"]["name"]["canonical"])
        self.assertNotIn("annotation", json.dumps(b))
        m = json.loads(llm_agents.brief(c, "customer_message"))
        self.assertIn("customer_first_name", m)
        with self.assertRaises(WorkflowError):
            llm_agents.run(c, "nope", "m.chen", hub_state.SETTINGS)


if __name__ == "__main__":
    unittest.main()
