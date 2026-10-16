"""The AI Agent on the file book: Claude's read of the ID and the proof of
address (stood in for here, so no call leaves the machine), the fixed checks,
and what the agent does with the case.

Run:  .venv\\Scripts\\python.exe -m unittest -v tests.test_ai_agent
"""
import copy
import os
import sys
import tempfile
import unittest
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
_DB = os.path.join(tempfile.mkdtemp(prefix="fcc-agent-"), "state.db")
os.environ.setdefault("FCC_STATE_DB", _DB)
os.environ["CONSOLE_CASES"] = "demo"      # the suite's default; this class switches to the file book itself
from fincrime.data.state_store import StateStore  # noqa: E402
if StateStore.path != os.environ["FCC_STATE_DB"]:
    StateStore.path = os.environ["FCC_STATE_DB"]
import app as fcc  # noqa: E402
from fincrime.data import recon_book  # noqa: E402
from fincrime.documents.raster import svg_to_png  # noqa: E402
from fincrime.documents.svg_cards import ISO3  # noqa: E402
from fincrime.engines import ai_agent  # noqa: E402
from fincrime.engines.errors import WorkflowError  # noqa: E402


def perfect_read(case, **id_overrides):
    """What the document images print, as a flawless reader would return it."""
    v = recon_book.view(case["recon"]["id"])
    pp, s = v["documents"]["passport"], v["synth"]
    idr = {"readable": True, "problems": [], "surname": pp["viz"]["surname"], "given_names": pp["viz"]["given"],
           "dob": v["customer"]["dob"], "nationality_code": ISO3.get(case["client"]["nationality"], ""),
           "document_number": pp["number"], "expiry": s["passport_expiry"], "address": "", "issued_on": s["passport_issued"],
           "note": "", "edges_visible": True, "mrz_visible": True}
    idr.update(id_overrides)
    holder = v["identity"].get("name", {}).get("canonical", v["customer"]["name"])     # as the proof of address prints it
    poa = {"readable": True, "problems": [], "surname": holder, "given_names": "", "dob": "", "nationality_code": "",
           "document_number": "", "expiry": "", "address": v["documents"]["poa"]["address"], "issued_on": s["poa_issued_on"],
           "note": ""}
    return {"id": idr, "poa": poa}


VERIFY = {"decision": "verify", "confidence": 93, "reason": "Clear passport; every detail matches the record."}


class Stub:
    """Stands in for llm.call: records what it was sent, answers `answer` to
    the read and `verdict` to the verdict call."""

    def __init__(self, answer=None, error=None, verdict=None):
        self.answer, self.error, self.verdict, self.sent = answer, error, verdict or VERIFY, []

    def __call__(self, system, content, schema, max_tokens=3000):
        self.sent.append(content)
        if self.error:
            raise self.error
        if schema is ai_agent.VERDICT_SCHEMA:
            return self.verdict, {"input": 900, "output": 60, "cached": 0}, 3.0
        return self.answer, {"input": 3000, "output": 200, "cached": 0}, 8.5


class AiAgentTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        fcc.app.testing = True
        fcc.app.secret_key = "test"
        cls.book_before = fcc.CASE_BOOK
        fcc.CASE_BOOK = "file"

    @classmethod
    def tearDownClass(cls):
        fcc.CASE_BOOK = cls.book_before
        StateStore.clear()
        fcc._load_seed()

    def setUp(self):
        StateStore.clear()
        fcc._load_seed()
        self.K = fcc.KYC_BY_ID

    def run_agent(self, case, answer=None, error=None, verdict=None, **id_overrides):
        stub = Stub(answer or perfect_read(case, **id_overrides), error, verdict)
        alert, wc = fcc._linked(case)
        return ai_agent.run_case(case, alert, wc, fcc._render_document, call=stub, sleep=lambda s: None), stub

    def pick(self, test):
        return next(c for c in fcc.KYC_CASES if ai_agent.eligible(c) and test(c))

    # ── the clean path ──────────────────────────────────────────────────────
    def test_a_clean_case_is_verified_with_no_person(self):
        c = self.clean(lambda c: c["screening"]["workflow"] == "auto-verified"
                       and not ai_agent.address_update(c, perfect_read(c)))      # its record's address needs no correction
        rec, stub = self.run_agent(c)
        self.assertEqual(rec["decision"], "verified", rec["reasons"])
        self.assertEqual((c["status"], c["closed_by"], c["previous_reviewer"]), ("Verified", "System User", "AI Agent"))
        self.assertEqual(c["confirmations"]["dob"]["confirmed_by"], "AI Agent")
        self.assertIn("Data confirmed by AI Agent", c["audit"][-1]["detail"])
        self.assertEqual([b["type"] for b in stub.sent[0]], ["text", "image", "text", "image"])
        self.assertEqual(stub.sent[0][1]["source"]["media_type"], "image/png")

    def test_a_case_the_rules_held_goes_to_screening(self):
        c = self.K["KYC-2026-3012"]
        c["current_reviewer"] = ""                                  # a starter case; free it for the agent
        rec, _ = self.run_agent(c)
        self.assertEqual(rec["decision"], "screening", rec["reasons"])
        self.assertEqual((c["status"], c["screening"]["workflow"]), ("DataScreening", "human-in-the-loop"))
        self.assertEqual(fcc.ALERT_BY_ID[c["screening"]["alert_case_id"]]["alerts"][0]["outcome"], "Pending")

    # ── what sends a case to a person ───────────────────────────────────────
    def assert_human(self, case, needle, **id_overrides):
        rec, _ = self.run_agent(case, **id_overrides)
        self.assertEqual(rec["decision"], "human")
        self.assertEqual((case["status"], case["screening"]["workflow"]), ("DataConfirming", "human-in-the-loop"))
        self.assertTrue(any(needle in r for r in rec["reasons"]), rec["reasons"])
        self.assertEqual((case["audit"][-1]["actor"], case["audit"][-1]["action"]), ("AI Agent", "Sent to a person"))
        return rec

    def clean(self, also=lambda c: True):
        """A case every check passes on a perfect read."""
        return self.pick(lambda c: also(c) and all(x["ok"] for x in ai_agent.checks(c, perfect_read(c))[0]))

    def test_unreadable_id(self):
        self.assert_human(self.clean(), "ID could not be read: blurred", readable=False, problems=["blurred"])
        self.assert_human(self.clean(), "ID could not be read: cut off", problems=["cut_off"])

    def test_date_of_birth_missing_or_different(self):
        self.assert_human(self.clean(), "Date of birth missing on the ID", dob="")
        self.assert_human(self.clean(), "Date of birth on the ID 1999-01-02", dob="1999-01-02")

    def test_expired_id(self):
        self.assert_human(self.clean(), "ID expired on 2025-03-01", expiry="2025-03-01")

    def test_name_that_is_somebody_else(self):
        self.assert_human(self.clean(), "Name on the ID differs from the CRM", surname="OKONKWO", given_names="CHIDI")

    def test_vendor_declined_and_old_proof_of_address(self):
        declined = self.pick(lambda c: c["vendor_check"]["result"] == "Declined")
        self.assert_human(declined, "ID declined by the vendor")
        old = self.pick(lambda c: ai_agent.workflow.poa_failures(c))
        self.assert_human(old, "Issued within the last 6 months")

    def test_gateway_error_goes_to_a_person(self):
        c = self.clean()
        rec, _ = self.run_agent(c, error=WorkflowError("The model gateway answered 500: busy"))
        self.assertEqual(rec["decision"], "human")
        self.assertIn("could not read the documents", rec["reasons"][0])
        self.assertEqual(c["status"], "DataConfirming")

    def test_rate_limit_waits_and_tries_again(self):
        for failure in ("The model gateway answered 429: too many connections", "Could not reach the model gateway."):
            c = self.clean(lambda c: "ai_agent" not in c)
            calls = []

            def flaky(system, content, schema, max_tokens=3000):
                if schema is ai_agent.VERDICT_SCHEMA:
                    return VERIFY, {}, 1.0
                calls.append(1)
                if len(calls) == 1:
                    raise WorkflowError(failure)
                return perfect_read(c), {}, 5.0
            alert, wc = fcc._linked(c)
            rec = ai_agent.run_case(c, alert, wc, fcc._render_document, call=flaky, sleep=lambda s: None)
            self.assertEqual(len(calls), 2, failure)
            self.assertNotEqual(rec["decision"], "human", rec["reasons"])

    # ── bad photos of the ID ────────────────────────────────────────────────
    def capture(self, kind):
        return self.pick(lambda c: recon_book.view(c["recon"]["id"])["synth"]["capture"] == kind)

    def test_a_phone_on_the_passport_passes_the_vendor_but_not_the_agent(self):
        c = self.capture("obstructed")
        self.assertEqual(c["vendor_check"]["result"], "Approved")                   # the vendor missed it, as on real uploads
        self.assertNotIn("Vendor rejected", c["manual_review_reason"])
        self.assertEqual(c["documents"][0]["quality"], "covered")
        hub = recon_book.hub_documents
        self.assertEqual(c["recon"]["drawn"], hub.POA_REDRAWN if hub.poa_wraps(recon_book.view(c["recon"]["id"])) else hub.REDRAWN)
        rec = self.assert_human(c, "ID could not be read: something covers part of it", problems=["covered"], dob="",
                                expiry="", mrz_visible=False)
        self.assertEqual(rec["resubmit"], ai_agent.RESUBMIT_UNCLEAR)
        self.assertIn(ai_agent.RESUBMIT_UNCLEAR, ai_agent.vocab.RESUBMIT_ID_REASONS)
        # the picture hides the dates: a reader that names no problem still cannot verify it
        rec = self.assert_human(self.capture("angled"), "Date of birth missing on the ID", dob="", expiry="")
        self.assertIn("Expiry date not readable", rec["reasons"])

    def test_a_slight_tilt_alone_stops_nothing(self):
        rec, _ = self.run_agent(self.clean(), problems=["angled"])
        self.assertEqual(rec["decision"], "verified", rec["reasons"])
        rec = self.assert_human(self.clean(), "something covers part of it, photographed at an angle", problems=["angled", "covered"])
        self.assertEqual(rec["resubmit"], ai_agent.RESUBMIT_UNCLEAR)

    def test_edges_and_mrz_out_of_shot(self):
        rec = self.assert_human(self.clean(), "its edges are not all in the picture", edges_visible=False)
        self.assertEqual(rec["resubmit"], ai_agent.RESUBMIT_CROPPED)
        rec = self.assert_human(self.clean(), "the MRZ is not fully visible", mrz_visible=False)
        self.assertEqual(rec["resubmit"], ai_agent.RESUBMIT_UNCLEAR)

    def test_the_suggested_resubmit_reason(self):
        for overrides, want in ((dict(problems=["damaged"]), ai_agent.RESUBMIT_DAMAGED),
                                (dict(readable=False, problems=["blurred", "too_dark"]), ai_agent.RESUBMIT_ILLEGIBLE),
                                (dict(problems=["angled", "covered"]), ai_agent.RESUBMIT_UNCLEAR),
                                (dict(problems=["covered", "angled"], edges_visible=False), ai_agent.RESUBMIT_UNCLEAR),
                                (dict(problems=["cut_off", "angled"], edges_visible=False), ai_agent.RESUBMIT_CROPPED),
                                (dict(expiry="2025-09-19"), ai_agent.RESUBMIT_EXPIRED),
                                (dict(surname="OKONKWO", given_names="CHIDI"), "")):   # not the photo: nothing to retake
            rec, _ = self.run_agent(self.clean(), **overrides)
            self.assertEqual((rec["decision"], rec["resubmit"]), ("human", want), overrides)
            self.assertTrue(not want or want in ai_agent.vocab.RESUBMIT_ID_REASONS)

    # ── Claude's own verdict, beside the rules ──────────────────────────────
    def test_the_verdict_sits_beside_the_rules_and_decides_nothing(self):
        c = self.clean()
        rec, stub = self.run_agent(c)
        self.assertEqual(rec["decision"], "verified")
        self.assertEqual((rec["verdict"]["decision"], rec["verdict"]["confidence"], rec["verdict"]["agrees"]), ("verify", 93, True))
        # the verdict call is text only and carries the read and the record, never the rules' result
        sent = stub.sent[1]
        self.assertEqual([b["type"] for b in sent], ["text"])
        self.assertIn(c["client"]["legal_name"], sent[0]["text"])
        self.assertIn('"readable": true', sent[0]["text"])
        self.assertNotIn("verified", sent[0]["text"].replace("Approved", ""))
        # the image read never sees the record: a hidden date of birth cannot be filled in from it
        self.assertNotIn(c["client"]["dob"], " ".join(b.get("text", "") for b in stub.sent[0]))
        # a verdict that differs is recorded as such and changes nothing
        other = self.clean()
        rec, _ = self.run_agent(other, verdict={"decision": "send_to_person", "confidence": 61, "reason": "Unsure."})
        self.assertEqual((rec["decision"], other["status"]), ("verified", "Verified"))
        self.assertFalse(rec["verdict"]["agrees"])
        # a person-bound case where Claude would verify also disagrees
        rec, _ = self.run_agent(self.clean(), dob="")
        self.assertEqual(rec["decision"], "human")
        self.assertFalse(rec["verdict"]["agrees"])

    def test_a_failed_or_odd_verdict_changes_nothing(self):
        c = self.clean()
        rec, _ = self.run_agent(c, verdict={"decision": "maybe", "confidence": 50, "reason": ""})
        self.assertEqual(rec["decision"], "verified")
        self.assertIn("not verify or send_to_person", rec["verdict"]["error"])
        self.assertNotIn("agrees", rec["verdict"])
        rec, _ = self.run_agent(self.clean(), error=WorkflowError("The model gateway answered 500: busy"))
        self.assertNotIn("verdict", rec)                                   # no read, nothing to judge

    def test_a_failed_verdict_is_asked_again_once_per_run_three_times_in_all(self):
        c = self.clean()
        rec, _ = self.run_agent(c)
        rec["verdict"] = {"error": "Could not reach the model gateway.", "policy": ai_agent.POLICY_VERSION}   # failed before tries counted
        status = c["status"]
        busy = Stub(error=WorkflowError("The model gateway answered 500: busy"))
        for run in (1, 2):
            self.assertTrue(ai_agent.needs_verdict(c))
            with mock.patch.object(ai_agent.llm, "call", busy), mock.patch.object(ai_agent, "eligible", lambda x: False):
                fcc.AI_AGENT.update(running=True, threads=[], busy=set())
                fcc._agent_worker()
            self.assertEqual(c["ai_agent"]["verdict"]["tries"], run + 1)          # one try per run
        self.assertFalse(ai_agent.needs_verdict(c))                               # three tries in all: no more
        self.assertEqual((len(busy.sent), c["status"]), (2, status))
        # a later success replaces the failure
        c["ai_agent"]["verdict"]["tries"] = 1
        with mock.patch.object(ai_agent.llm, "call", Stub()), mock.patch.object(ai_agent, "eligible", lambda x: False):
            fcc.AI_AGENT.update(running=True, threads=[], busy=set())
            fcc._agent_worker()
        self.assertEqual((c["ai_agent"]["verdict"]["decision"], c["ai_agent"]["verdict"]["agrees"]), ("verify", True))
        self.assertFalse(ai_agent.needs_verdict(c))

    def test_earlier_reads_get_a_verdict_from_the_stored_read(self):
        c = self.clean()
        self.run_agent(c)
        del c["ai_agent"]["verdict"]                                       # as read before verdicts existed
        status = c["status"]
        self.assertTrue(ai_agent.needs_verdict(c))
        stub = Stub(verdict={"decision": "verify", "confidence": 88, "reason": "Fine."})
        with mock.patch.object(ai_agent.llm, "call", stub), mock.patch.object(ai_agent, "eligible", lambda x: False):
            fcc.AI_AGENT.update(running=True, threads=[], busy=set())
            fcc._agent_worker()
        self.assertEqual((c["ai_agent"]["verdict"]["confidence"], c["ai_agent"]["verdict"]["agrees"]), (88, True))
        self.assertEqual(c["status"], status)
        self.assertEqual([[b["type"] for b in sent] for sent in stub.sent], [["text"]])   # no images sent again
        self.assertFalse(ai_agent.needs_verdict(c))
        # the AI Screening page counts it, the case list filters on it
        self.login("leah.fernandez")
        page = self.c.get("/ai-screening").data.decode("utf-8")
        self.assertIn('id="ai-verdict-row"', page)
        self.assertIn("agrees with the rules on 1 of 1 (100%)", page)
        self.login("amara.iyer")
        self.assertIn('id="ai-verdict"', self.c.get(f"/kyc/detail/{c['id']}").data.decode("utf-8"))
        listing = self.c.get("/kyc/list?verdict=agree").data.decode("utf-8")
        self.assertIn(f'data-case-id="{c["id"]}"', listing)
        self.assertNotIn(f'data-case-id="{c["id"]}"', self.c.get("/kyc/list?verdict=disagree").data.decode("utf-8"))

    def test_a_verdict_under_an_older_policy_is_redone(self):
        # the policy carries Vietnam's 2025 mergers, from the address rules' own list
        c = self.clean()
        rec, stub = self.run_agent(c)
        self.assertIn("Bac Giang → Bac Ninh", stub.sent[1][0]["text"])
        self.assertIn("Binh Duong", ai_agent.POLICY)
        self.assertEqual(rec["verdict"]["policy"], ai_agent.POLICY_VERSION)
        self.assertFalse(ai_agent.needs_verdict(c))
        del rec["verdict"]["policy"]                                         # as given under the first policy
        self.assertTrue(ai_agent.needs_verdict(c))
        self.login("leah.fernandez")
        page = self.c.get("/ai-screening").data.decode("utf-8")
        self.assertIn("1 earlier read(s) waiting for a verdict", page)       # waiting, not counted as judged
        self.assertNotIn("agrees with the rules on", page)
        stub = Stub(verdict={"decision": "verify", "confidence": 91, "reason": "Fine."})
        with mock.patch.object(ai_agent.llm, "call", stub), mock.patch.object(ai_agent, "eligible", lambda x: False):
            fcc.AI_AGENT.update(running=True, threads=[], busy=set())
            fcc._agent_worker()
        self.assertEqual((rec["verdict"]["confidence"], rec["verdict"]["policy"]), (91, ai_agent.POLICY_VERSION))
        self.assertEqual([[b["type"] for b in sent] for sent in stub.sent], [["text"]])   # no images sent again
        self.assertIn("agrees with the rules on 1 of 1 (100%)", self.c.get("/ai-screening").data.decode("utf-8"))

    # ── an Aadhaar card has no age limit; older decisions are checked again ──
    def poa_kind(self, kind):
        return lambda c: (ai_agent.workflow.poa_docs(c) or [{}])[0].get("kind") == kind

    def old_read(self, case):
        read = perfect_read(case)
        read["poa"]["issued_on"] = "2024-10-14"                             # an Aadhaar card's date of issue
        return read

    def test_an_aadhaar_card_has_no_age_limit(self):
        card = self.clean(self.poa_kind("aadhaar"))
        rec, _ = self.run_agent(card, answer=self.old_read(card))
        self.assertEqual(rec["decision"], "verified", rec["reasons"])
        self.assertIn("except an Aadhaar card", ai_agent.POLICY)
        # the Data Confirm checks agree, whatever the upload's date
        other = self.clean(self.poa_kind("aadhaar"))
        ai_agent.workflow.poa_docs(other)[0]["issued_on"] = "2025-01-02"
        self.assertEqual(ai_agent.workflow.poa_failures(other), [])
        # a bill read as two years old still goes to a person
        bill = self.clean(self.poa_kind("utility_bill"))
        rec, _ = self.run_agent(bill, answer=self.old_read(bill))
        self.assertEqual(rec["decision"], "human")
        self.assertTrue(any("Issued within the last 6 months" in r for r in rec["reasons"]), rec["reasons"])

    def test_an_unreadable_proof_of_address_date_goes_to_a_person(self):
        # the record carries a date, but the document's own date must be legible
        bill = self.clean(self.poa_kind("utility_bill"))
        read = perfect_read(bill)
        read["poa"]["issued_on"] = ""
        rec, _ = self.run_agent(bill, answer=read)
        self.assertEqual(rec["decision"], "human")
        self.assertTrue(any("Issue date readable: the date on the document could not be read" in r for r in rec["reasons"]),
                        rec["reasons"])
        # an Aadhaar card has no age limit, so its date does not matter
        card = self.clean(self.poa_kind("aadhaar"))
        read = perfect_read(card)
        read["poa"]["issued_on"] = ""
        self.assertEqual(self.run_agent(card, answer=read)[0]["decision"], "verified")

    def test_a_case_sent_to_a_person_under_older_checks_is_checked_again(self):
        card = self.clean(self.poa_kind("aadhaar"))
        touched = self.clean(lambda c: self.poa_kind("aadhaar")(c) and c["id"] != card["id"])
        bill = self.clean(self.poa_kind("utility_bill"))
        with mock.patch.object(ai_agent.vocab, "POA_NO_AGE_LIMIT", ()):           # the checks before the fix
            for c in (card, touched, bill):
                self.assertEqual(self.run_agent(c, answer=self.old_read(c))[0]["decision"], "human")
                del c["ai_agent"]["rules"]                                    # decided under the first rules
        self.login("amara.iyer")
        self.c.post(f"/kyc/{touched['id']}/reserve")                         # a person has it now: left alone
        self.assertEqual([ai_agent.needs_recheck(c) for c in (card, touched, bill)], [True, False, True])
        stub = Stub()
        with mock.patch.object(ai_agent.llm, "call", stub), mock.patch.object(ai_agent, "eligible", lambda x: False):
            fcc.AI_AGENT.update(running=True, threads=[], busy=set())
            fcc._agent_worker()
        self.assertEqual(stub.sent, [])                                       # no model call, no images
        self.assertEqual((card["status"], card["ai_agent"]["decision"]), ("Verified", "verified"))
        self.assertIn(("AI Agent", "Checked again"), [(e["actor"], e["action"]) for e in card["audit"]])
        self.assertEqual((touched["status"], touched["ai_agent"]["decision"]), ("DataConfirming", "human"))
        self.assertEqual((bill["status"], bill["ai_agent"]["decision"], bill["ai_agent"]["rules"]),
                         ("DataConfirming", "human", ai_agent.RULES_VERSION))   # still a person's, checked once
        self.assertFalse(ai_agent.needs_recheck(bill))

    def test_the_proof_of_address_is_in_the_customers_name(self):
        card = self.clean(self.poa_kind("aadhaar"))
        order = [c["check"] for c in ai_agent.checks(card, perfect_read(card))[0]]
        self.assertEqual(order[order.index("Proof of address readable") + 1], "Name on proof of address")
        self.assertIn("An S/O, D/O, W/O or C/O line in an address names a relative", self.run_agent(card)[1].sent[0][2]["text"])
        self.assertEqual(card["ai_agent"]["decision"], "verified", card["ai_agent"]["reasons"])
        # the relative an Aadhaar's address names is not the holder
        other = self.clean(lambda c: self.poa_kind("aadhaar")(c) and c["id"] != card["id"])
        read = perfect_read(other)
        read["poa"]["surname"] = "RAKESH TIWARI"
        self.assert_human(other, "Name on the proof of address differs from the CRM", answer=read)
        bill = self.clean(self.poa_kind("utility_bill"))
        read = perfect_read(bill)
        read["poa"]["surname"] = ""
        self.assert_human(bill, "No holder name could be read on the proof of address", answer=read)
        v = ai_agent.poa_name_verdict
        self.assertTrue(v("SEEMA VERMA", {"surname": "Seema Verma Pandey", "given_names": ""}))        # a married surname added
        self.assertTrue(v("JEFF WILLIAMS", {"surname": "WILLIAMS", "given_names": "JEFFREY DEAN"}))   # nickname
        self.assertTrue(v("SAKSHI", {"surname": "SAKSHI", "given_names": ""}))                        # a single name
        self.assertFalse(v("PRIYA SHARMA", {"surname": "SHARMA", "given_names": ""}))                 # a surname alone is not her
        self.assertIsNone(v("PRIYA SHARMA", {"surname": "", "given_names": ""}))

    def test_a_read_without_the_proof_of_address_name_starts_again(self):
        unnamed, named = self.clean(self.poa_kind("utility_bill")), self.clean(self.poa_kind("aadhaar"))
        touched = self.clean(lambda c: self.poa_kind("utility_bill")(c) and c["id"] != unnamed["id"])
        for c in (unnamed, named, touched):
            self.assertEqual(self.run_agent(c)[0]["decision"], "verified", c["ai_agent"]["reasons"])
            c["ai_agent"]["rules"] = 3                                       # decided under the checks before the name
            if c is not named:
                c["ai_agent"]["read"]["poa"]["surname"] = ""                 # a read that was not asked for it
            StateStore.save("kyc", c)
        self.assertEqual([ai_agent.needs_reread(c) for c in (unnamed, named, touched)], [True, False, True])
        ai_agent.workflow._audit(touched, "amara.iyer", "Reopened")          # a person has worked on it since
        StateStore.save("kyc", touched)
        self.assertFalse(ai_agent.needs_reread(touched))
        fcc._load_seed()
        self.assertNotIn("ai_agent", fcc.KYC_BY_ID[unnamed["id"]])           # back to its seeded state, read again
        self.assertEqual(fcc.KYC_BY_ID[named["id"]]["ai_agent"]["decision"], "verified")
        self.assertEqual(fcc.KYC_BY_ID[touched["id"]]["ai_agent"]["decision"], "verified")

    # ── the name rules ──────────────────────────────────────────────────────
    def test_name_rules(self):
        v = ai_agent.name_verdict
        self.assertEqual(v("NGUYEN NGOC PHUONG", "NGUYỄN", "NGỌC PHƯƠNG"), (True, "NGUYEN NGOC PHUONG"))
        self.assertEqual(v("JEFF WILLIAMS", "WILLIAMS", "JEFFREY DEAN"), (True, "JEFFREY DEAN WILLIAMS"))   # nickname
        self.assertEqual(v("PILLAI NEETHU", "PILLAI", "NEETHU"), (True, "PILLAI NEETHU"))                    # order
        self.assertEqual(v("BALAJI K", "KRISHNAN", "BALAJI"), (True, "BALAJI KRISHNAN"))                     # patronymic initial
        self.assertEqual(v("PRABHU FNU", "", "PRABHU"), (True, "PRABHU"))                                    # single name
        self.assertEqual(v("VÅ© CÃ´ng KhÃ¡nh", "VŨ", "CÔNG KHÁNH")[0], True)                                # mojibake
        self.assertFalse(v("PATRICIA WILSON", "ROBERTS", "PATRICIA MARIE")[0])                              # another surname
        self.assertTrue(v("KEVIN NGUYEN", "NGUYEN", "KHOA QUOC")[0])                                        # the hub knows Kevin for Khoa
        self.assertFalse(v("BRANDON NGUYEN", "NGUYEN", "KHOA QUOC")[0])                                     # another given name

    def test_a_record_that_keeps_the_fathers_name_is_not_the_holder(self):
        # an Indian passport's second given name is the father's: a record
        # that keeps only it names the father (the file's "Father's name as
        # middle name"); one that drops it keeps the holder
        v = ai_agent.name_verdict
        self.assertFalse(v("PRAKASH MORE", "MORE", "MANASI PRAKASH", country="IN")[0])
        self.assertTrue(v("HITESH BHATT", "BHATT", "HITESH DINESHBHAI", country="IN")[0])
        self.assertTrue(v("M PRAKASH MORE", "MORE", "MANASI PRAKASH", country="IN")[0])                    # her initial kept
        # not held to it: a Vietnamese personal name (its last), a longer name, a name often dropped
        self.assertTrue(v("NGUYEN MINHKHANG", "NGUYỄN", "VĂN MINH KHANG", country="VN")[0])
        self.assertTrue(v("PHUONG NGUYEN", "NGUYỄN", "NGỌC PHƯƠNG", country="VN")[0])
        self.assertTrue(v("DURGA BANDARU", "BANDARU", "VENKATA LAKSHMI DURGA", country="IN")[0])
        self.assertTrue(v("SALMAN PATHAN", "PATHAN", "MOHAMMED SALMAN", country="IN")[0])
        # the agent sends the book's case to a person, on the ID and on the proof of address alike
        c = self.K["KYC-2026-3052"]
        rec = self.assert_human(c, "Name on the ID differs from the CRM")
        self.assertIn("Name on the proof of address differs from the CRM", rec["reasons"])

    def test_a_case_confirmed_on_the_fathers_name_starts_again(self):
        father, kept = self.K["KYC-2026-3052"], self.clean()
        with mock.patch.object(ai_agent, "_own_name_kept", lambda *a, **k: True):   # the checks before the fix
            for c in (father, kept):
                self.assertEqual(self.run_agent(c)[0]["decision"], "verified", c["ai_agent"]["reasons"])
                c["ai_agent"]["rules"] = 4
                StateStore.save("kyc", c)
        self.assertEqual(father["client"]["legal_name"], "MANASI PRAKASH MORE")            # the confirmation rewrote it
        fcc._load_seed()
        self.assertNotIn("ai_agent", fcc.KYC_BY_ID[father["id"]])                         # back to its seeded state, read again
        self.assertEqual(fcc.KYC_BY_ID[father["id"]]["client"]["legal_name"], "PRAKASH MORE")
        self.assertEqual(fcc.KYC_BY_ID[kept["id"]]["ai_agent"]["decision"], "verified")    # the rest is kept

    # ── the verified address written back ──────────────────────────────────
    def test_the_verified_address_corrects_the_record(self):
        # a PIN keyed wrong: corrected in the record's own words
        c = self.K["KYC-2026-3014"]
        self.assertTrue(c["client"]["address"].endswith("Uttar Pradesh 226001"))
        rec, _ = self.run_agent(c)
        self.assertEqual(rec["decision"], "verified", rec["reasons"])
        self.assertTrue(c["client"]["address"].endswith("Uttar Pradesh 226010"), c["client"]["address"])
        self.assertEqual(rec["address_updated"]["from"][-6:], "226001")
        self.assertEqual((c["change_log"][-1]["by"], c["change_log"][-1]["field"]), ("AI Agent", "address"))
        self.assertIn(("AI Agent", "Address updated"), [(e["actor"], e["action"]) for e in c["audit"]])
        self.login("amara.iyer")
        page = self.c.get(f"/kyc/detail/{c['id']}").data.decode("utf-8")
        self.assertIn('id="ai-agent-address"', page)
        # a city under its old name: the document's address, without the relative's line
        c = self.K["KYC-2026-3032"]                                           # Madras; the Aadhaar reads S/O: Krishnan, ... Chennai
        self.assertIn("Madras", c["client"]["address"])
        self.assertEqual(self.run_agent(c)[0]["decision"], "verified")
        self.assertIn("Chennai", c["client"]["address"])
        self.assertNotIn("S/O", c["client"]["address"])
        # an address written another way only is left as it is
        same = self.clean(lambda x: x["id"] not in ("KYC-2026-3014", "KYC-2026-3032")
                          and not ai_agent.address_update(x, perfect_read(x)))
        before = same["client"]["address"]
        self.assertEqual(self.run_agent(same)[0]["decision"], "verified")
        self.assertEqual((same["client"]["address"], same["ai_agent"].get("address_updated")), (before, None))

    def test_an_address_confirmed_before_is_corrected_with_no_model_call(self):
        c = self.K["KYC-2026-3014"]
        with mock.patch.object(ai_agent, "address_update", lambda case, got: ""):    # confirmed before the write-back
            self.run_agent(c)
        c["ai_agent"]["rules"] = 4
        self.assertTrue(ai_agent.needs_address_update(c))
        stub = Stub()
        with mock.patch.object(ai_agent.llm, "call", stub), mock.patch.object(ai_agent, "eligible", lambda x: False):
            fcc.AI_AGENT.update(running=True, threads=[], busy=set())
            fcc._agent_worker()
        self.assertEqual(stub.sent, [])                                       # no model call, no images
        self.assertTrue(c["client"]["address"].endswith("226010"))
        self.assertEqual(c["ai_agent"]["rules"], ai_agent.RULES_VERSION)
        self.assertFalse(ai_agent.needs_address_update(c))

    def test_a_garbled_record_name_repairs_from_the_files_text(self):
        # the console keeps the record's name in capitals, and capitals change
        # the bytes a garbled name stands for: it repairs from the file's text
        v, filed = ai_agent.name_verdict, "VÃµ Gia SÆ¡n"
        self.assertEqual(filed.upper(), "VÃΜ GIA SÆ¡N")
        self.assertFalse(v(filed.upper(), "VÕ", "GIA SƠN")[0])
        self.assertEqual(v(filed.upper(), "VÕ", "GIA SƠN", filed=filed), (True, "VO GIA SON"))
        self.assertFalse(v("NGUYEN VAN AN", "VÕ", "GIA SƠN", filed=filed)[0])        # an edited record: the file no longer applies
        # a byte the file itself lost stands for exactly one letter
        filed = "Phan Ngá»c Äáº¡t"
        self.assertEqual(v(filed.upper(), "PHAN", "NGỌC ĐẠT", filed=filed), (True, "PHAN NGOC DAT"))
        self.assertFalse(v(filed.upper(), "PHAN", "NGỌC TÂM", filed=filed)[0])
        self.assertFalse(v(filed.upper(), "PHAN", "NGỌC ĐẠT ANH", filed=filed)[0])
        # the agent finds the file's text for a case of the book
        c = self.K["KYC-2026-3353"]
        self.assertEqual(ai_agent._filed_name(c), "VÃµ Gia SÆ¡n")
        name = next(x for x in ai_agent.checks(c, perfect_read(c))[0] if x["check"] == "Name")
        self.assertTrue(name["ok"], name)

    def test_only_auto_verified_individuals_are_taken(self):
        self.assertTrue(ai_agent.eligible(self.K["KYC-2026-3011"]))      # auto-verified, even though Amara holds it
        self.assertFalse(ai_agent.eligible(self.K["KYC-2026-3012"]))     # an alert left: the analysts' from the start
        self.assertFalse(ai_agent.eligible(self.K["KYC-2026-3029"]))     # a company
        self.assertTrue(svg_to_png('<svg xmlns="http://www.w3.org/2000/svg" width="4" height="4"/>').startswith(b"\x89PNG"))


    # ── in the console ──────────────────────────────────────────────────────
    def login(self, key):
        self.c = fcc.app.test_client()
        r = self.c.post("/login", data={"analyst": key, "password": "1234"})
        self.assertEqual(r.status_code, 302)

    def test_the_agent_verifies_an_auto_verified_case_by_itself(self):
        self.login("amara.iyer")
        c = self.K["KYC-2026-3011"]                                     # auto-verified, reserved by Amara at intake
        page = self.c.get(f"/kyc/detail/{c['id']}").data.decode("utf-8")
        self.assertNotIn('id="ai-agent"', page)                          # no card, no button before the agent runs
        self.assertNotIn("Run AI agent", page)
        original = ai_agent.eligible
        with mock.patch.object(ai_agent, "eligible", lambda x: x is c and original(x)),                 mock.patch.object(ai_agent.llm, "call", Stub(perfect_read(c))):
            fcc.AI_AGENT.update(running=True, threads=[], busy=set())
            fcc._agent_worker()
        self.assertEqual((c["status"], c["closed_by"], c["previous_reviewer"]), ("Verified", "System User", "AI Agent"))
        self.assertTrue(all(w.get("to") for w in c.get("work", [])), c.get("work"))      # Amara's turn closed
        page = self.c.get(f"/kyc/detail/{c['id']}").data.decode("utf-8")
        self.assertIn('id="ai-agent-decision"', page)
        self.assertIn("Handled by AI", page)
        self.assertNotIn("Run AI agent", page)
        self.assertIn(c["id"], self.c.get("/kyc/list?agent=verified").data.decode("utf-8"))

    def test_the_resubmit_dialog_arrives_filled_in(self):
        c = self.capture("obstructed")
        c["current_reviewer"] = ""
        self.run_agent(c, problems=["covered"], dob="", expiry="")
        self.login("amara.iyer")
        page = self.c.get(f"/kyc/detail/{c['id']}").data.decode("utf-8")
        self.assertIn('id="ai-agent-resubmit"', page)
        self.assertIn('id="rs-suggested"', page)
        self.assertIn('value="Resubmit ID" onchange="rsToggle()" checked', page)
        self.assertIn(f"<option selected>{ai_agent.RESUBMIT_UNCLEAR}</option>", page)
        self.c.post(f"/kyc/{c['id']}/reserve")
        self.c.post(f"/kyc/{c['id']}/resubmit", data={"decision": "Resubmit ID", "id_reason": ai_agent.RESUBMIT_UNCLEAR})
        self.assertEqual((c["status"], c["resubmit"]["id_reason"]), ("ResubmitID", ai_agent.RESUBMIT_UNCLEAR))
        self.assertNotIn('id="ai-agent-resubmit"', self.c.get(f"/kyc/detail/{c['id']}").data.decode("utf-8"))

    def test_a_watchlist_record_lists_every_place_the_file_gives(self):
        c = self.K["KYC-2026-3887"]                                          # Singapore; Dhaka, Bangladesh
        alert, wc = fcc._linked(c)
        m = wc["matches"][0]
        self.assertEqual(m["country"], "Singapore; Bangladesh")
        self.assertEqual([(x["city"], x["country"]) for x in m["key_data"]["locations"]], [("", "Singapore"), ("Dhaka", "Bangladesh")])
        self.assertEqual(m["key_data"]["location"]["country"], "Singapore")                # the first stays the main one
        self.assertEqual(fcc._linked(self.K["KYC-2026-3014"])[1]["matches"][0]["country"], "India")   # one place: as before
        # a match stored when it kept only the first place is brought in line
        stale = copy.deepcopy(wc)
        stale["matches"][0]["country"] = "Singapore"
        stale["matches"][0]["key_data"].pop("locations")
        StateStore.save("wc", stale)
        fcc._load_seed()
        m = fcc.WC_BY_ID[wc["id"]]["matches"][0]
        self.assertEqual((m["country"], len(m["key_data"]["locations"])), ("Singapore; Bangladesh", 2))
        self.login("amara.iyer")
        page = self.c.get(f"/wc/{wc['id']}/match/{m['id']}").data.decode("utf-8")
        self.assertIn("Dhaka", page)
        self.assertIn("Singapore; Bangladesh", page)

    def test_an_old_proof_of_address_is_reported_once(self):
        bill = self.clean(self.poa_kind("utility_bill"))
        read = perfect_read(bill)
        read["poa"]["issued_on"] = ai_agent.workflow.poa_docs(bill)[0]["issued_on"] = "2025-12-01"   # the record and the document agree
        rec, _ = self.run_agent(bill, answer=read)
        note = next(r for r in rec["reasons"] if "Issued within the last 6 months" in r)
        self.assertEqual(note.count("Issued within the last 6 months"), 1, note)
        # a document whose own date is old where the record's is not: both are said
        other = self.clean(lambda c: self.poa_kind("utility_bill")(c) and c["id"] != bill["id"])
        rec, _ = self.run_agent(other, answer=self.old_read(other))
        self.assertIn("the document reads 2024-10-14", " ".join(rec["reasons"]))

    def test_a_stored_case_takes_its_sex_from_the_file(self):
        c = self.K["KYC-2026-3140"]                                          # Jennifer Smith
        alert, wc = fcc._linked(c)
        self.assertEqual((c["client"]["gender"], alert["screening_info"]["gender"], wc["gender"]), ("F", "Female", "Female"))
        stale = [copy.deepcopy(r) for r in (c, alert, wc)]                     # as stored under the coin toss
        stale[0]["client"]["gender"], stale[1]["screening_info"]["gender"], stale[2]["gender"] = "M", "Male", "Male"
        for kind, record in zip(("kyc", "alert", "wc"), stale):
            StateStore.save(kind, record)
        fcc._load_seed()
        c = fcc.KYC_BY_ID["KYC-2026-3140"]
        alert, wc = fcc._linked(c)
        self.assertEqual((c["client"]["gender"], alert["screening_info"]["gender"], wc["gender"]), ("F", "Female", "Female"))
        self.login("amara.iyer")
        self.assertIn("Female", self.c.get(f"/wc/{wc['id']}").data.decode("utf-8"))

    def test_a_case_stored_from_an_older_drawing_starts_again(self):
        c = self.capture("damaged")
        self.run_agent(c, problems=["damaged"])
        stale = dict(c, recon=dict(c["recon"], drawn=""))                   # as stored before the redraw
        StateStore.save("kyc", stale)
        kept = self.clean()
        self.run_agent(kept)
        StateStore.save("kyc", kept)
        fcc._load_seed()
        self.assertNotIn("ai_agent", fcc.KYC_BY_ID[c["id"]])                 # back to its seeded state, read again
        self.assertEqual(fcc.KYC_BY_ID[c["id"]]["status"], "DataConfirming")
        self.assertEqual(fcc.KYC_BY_ID[kept["id"]]["ai_agent"]["decision"], "verified")    # the rest is kept

    def test_the_worker_works_the_queue_and_stops(self):
        chosen = [c["id"] for c in fcc.KYC_CASES if ai_agent.eligible(c)][:3]
        original = ai_agent.eligible
        only = lambda c: c["id"] in chosen and original(c)                  # noqa: E731
        answers = [perfect_read(self.K[i]) for i in chosen]                  # the worker takes them in book order

        def call(system, content, schema, max_tokens=3000):
            if schema is ai_agent.VERDICT_SCHEMA:
                return VERIFY, {"input": 10, "output": 5}, 1.0
            return answers.pop(0), {"input": 10, "output": 5}, 1.0
        with mock.patch.object(ai_agent, "eligible", only), mock.patch.object(ai_agent.llm, "call", call):
            fcc.AI_AGENT.update(running=True, threads=[], busy=set())
            fcc._agent_worker()
        self.assertFalse(fcc.AI_AGENT["running"])
        self.assertEqual(fcc.AI_AGENT["stopped"], "Queue finished")
        self.assertTrue(all(self.K[i].get("ai_agent") for i in chosen))

    def test_start_and_pause_are_for_the_team_lead(self):
        self.login("amara.iyer")
        r = self.c.post("/ai-agent/start", follow_redirects=True)
        self.assertIn("That action needs QC access", r.data.decode("utf-8"))
        self.assertFalse(fcc.AI_AGENT["running"])
        self.login("leah.fernandez")
        with mock.patch.object(fcc, "_agent_worker", lambda: None), mock.patch.object(fcc.ai_review, "configured", lambda: True):
            self.c.post("/ai-agent/start")
            self.assertTrue(fcc.AI_AGENT["running"])
            page = self.c.get("/ai-screening").data.decode("utf-8")
            self.assertIn('id="ai-agent-card"', page)
            self.assertIn("Running", page)
            self.c.post("/ai-agent/pause")
        self.assertFalse(fcc.AI_AGENT["running"])
        self.assertIn("Paused by Leah Fernandez", self.c.get("/ai-screening").data.decode("utf-8"))


if __name__ == "__main__":
    unittest.main()
