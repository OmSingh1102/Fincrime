"""The AI Agent: it works an L1 Data Confirming case the way an analyst does,
and hands the case to a person whenever it cannot be sure.

1. Claude reads the passport data page and the proof of address from their
   images (one call through Straive's LLM Foundry, fincrime/hub/llm.call):
   what is printed, whether it can be read, and what spoils the capture.
2. Fixed checks - code, not the model - compare that read with the case:
   readable, name, date of birth, nationality, ID number, expiry, the
   vendor's result, the proof of address and its address.
3. Every check passes: the agent confirms the data through the analysts'
   own path (workflow.confirm_data), the AI screening rules re-run, and a
   case they leave nothing on verifies itself. Any check fails, or Claude
   cannot be reached: the case stays in Data Confirming, labelled
   human-in-the-loop, with the reasons and what the agent read.
4. Claude's own verdict, a second opinion beside the rules: a text-only call
   gets the read, the customer's record and the bank's document policy, and
   answers verify / send to a person with a confidence. It is shown and
   counted (does it agree with the rules?) but never acts on the case. It is
   a separate call so the record cannot leak into the read: a hidden date of
   birth must stay unread, not be filled in from the CRM.

It only takes individuals whose screening the rules already
cleared (the auto-verified workflow); a company (KYB) and every case with an
alert left stay with people. The result sits on the case as case["ai_agent"].
"""
import base64
import json
import re
import time

from fincrime.data import vocab
from fincrime.documents.raster import svg_to_png
from fincrime.documents.svg_cards import ISO3
from fincrime.engines import workflow
from fincrime.engines.errors import WorkflowError
from fincrime.hub import addresses, llm, names, similarity
from fincrime.utils import dates

ACTOR = vocab.AI_AGENT
MODEL = llm.MODEL
ALPHA2 = {v: k for k, v in ISO3.items()}
SPOILERS = {"blurred": "blurred", "cut_off": "cut off", "glare": "glare", "covered": "something covers part of it",
            "angled": "photographed at an angle or from too far", "damaged": "the document is cracked or worn",
            "too_dark": "too dark"}
# A tilt is common on good photos (the model sometimes names one), so on its
# own it stops nothing; it is listed when something else on the ID is wrong.
SOFT = {"angled"}
FILLERS = {"FNU", "LNU", "NFN", "NLN", "-", "."}
RETRIES = 3           # the gateway allows few concurrent calls; a 429 or a dropped connection waits and tries again
RETRY_WAIT = 10.0

SYSTEM = ("You read identity documents for a KYC check at a bank. You are given images of a customer's documents. "
          "Report only what is printed and legible in the images. Never guess, infer or complete a value: if a field "
          "is not printed, cut off, blurred, hidden by glare or covered by something, return an empty string for it. "
          "Mark a document unreadable when any of name, date of birth, document number or expiry on an ID cannot be "
          "read with certainty. Name every capture problem you see: blurred; cut_off (part of the document is outside "
          "the picture); glare; covered (an object such as a phone, a finger or paper lies over part of it); angled "
          "(photographed at a steep angle or from so far away that the text is small; a slight tilt is normal, not a "
          "problem); damaged (cracked, torn, worn or faded); too_dark. "
          "Dates as YYYY-MM-DD.")

_DOC = {"type": "object", "additionalProperties": False, "properties": {
    "readable": {"type": "boolean"},
    "problems": {"type": "array", "items": {"type": "string", "enum": ["blurred", "cut_off", "glare", "covered", "angled",
                                                                      "damaged", "too_dark", "other"]}},
    "surname": {"type": "string", "description": "Surname as printed (on a proof of address: the holder's full name)"},
    "given_names": {"type": "string", "description": "Given names as printed, else empty"},
    "dob": {"type": "string"},
    "nationality_code": {"type": "string", "description": "Three-letter code printed on the ID (e.g. VNM), else empty"},
    "document_number": {"type": "string"},
    "expiry": {"type": "string"},
    "address": {"type": "string", "description": "The address printed on the document, else empty"},
    "issued_on": {"type": "string", "description": "Issue or statement date, else empty"},
    "note": {"type": "string", "description": "One short sentence on anything wrong with the document, else empty"}},
    "required": ["readable", "problems", "surname", "given_names", "dob", "nationality_code", "document_number", "expiry", "address",
                 "issued_on", "note"]}
# The ID's read also says whether the whole document and its MRZ are in shot.
_ID = dict(_DOC, properties=dict(_DOC["properties"],
    edges_visible={"type": "boolean", "description": "All four corners and edges of the document are inside the picture"},
    mrz_visible={"type": "boolean", "description": "Both machine-readable (MRZ) lines at the bottom of a passport page are fully "
                                                   "visible and legible, with nothing over them"}),
           required=_DOC["required"] + ["edges_visible", "mrz_visible"])
SCHEMA = {"type": "object", "additionalProperties": False, "properties": {"id": _ID, "poa": _DOC},
          "required": ["id", "poa"]}

# Claude's own verdict: advisory, beside the rules' decision.
VERDICT_SYSTEM = ("You are a senior KYC analyst at a bank giving a second opinion. You get a customer's record, the "
                  "bank's document policy, the ID-verification vendor's result, and what an earlier reading step read "
                  "from the customer's ID and proof of address. Decide whether you would verify this customer's identity "
                  "and address from these documents (verify) or send the case to a person (send_to_person). Judge only "
                  "what is given; a field that was not read was not legible. Give your confidence (0-100) that your "
                  "decision is right and one short sentence why.")
VERDICT_SCHEMA = {"type": "object", "additionalProperties": False, "properties": {
    "decision": {"type": "string", "enum": ["verify", "send_to_person"]},
    "confidence": {"type": "integer", "description": "0-100: how sure you are that your decision is right"},
    "reason": {"type": "string", "description": "One short sentence"}},
    "required": ["decision", "confidence", "reason"]}
def _vn_mergers():
    """Vietnam's 2025 mergers (the address rules' list) in words."""
    joined, seen = {}, set()
    for old, new in addresses.VN_MERGED.items():
        if old.replace(" - ", " ") not in seen:
            seen.add(old.replace(" - ", " "))
            joined.setdefault(new.title(), []).append(old.title())
    return "; ".join(f"{', '.join(olds)} → {new}" for new, olds in sorted(joined.items()))


POLICY = ("The ID must be fully legible (all four edges and the MRZ in shot, nothing covering it, not blurred or "
          "damaged), not expired, and its name, date of birth, nationality and number must match the record (a "
          "spelling variant, a nickname, another word order or an added middle name is acceptable). The proof of "
          "address must be legible, issued within the last 6 months (except an Aadhaar card: it is a permanent "
          "card, so its date of issue does not matter), and show the customer's registered address. "
          "On 1 July 2025 Vietnam merged provinces and abolished district-level units, so an address may name a "
          "former province or district where the other names the province it joined; that is the same place. "
          f"Former → current: {_vn_mergers()}.")
# Bumped when POLICY changes: verdicts given under an older one are redone.
POLICY_VERSION = 3
# A verdict call that failed (a busy gateway) is asked again on later agent
# runs, one try per run, this many tries in all.
VERDICT_TRIES = 3
# Bumped when the checks change: a case they sent to a person, and no one has
# touched since, is checked again on its stored read (recheck). 4: the name
# on the proof of address. 5: the holder's own given name (_own_name_kept),
# and the verified address written to the record (address_update). 6: an old
# proof of address reported once when its read and recorded dates agree.
RULES_VERSION = 6
POA_NAME_RULES = 4    # the first checks to want the proof of address's name: reads before them may lack it (needs_reread)
OWN_NAME_RULES = 5    # the first to want the holder's own given name: a case confirmed before may name the father
VERDICT_LABELS = {"verify": "Verify", "send_to_person": "Send to a person"}

# What an analyst asks the customer for when the ID itself is the problem
# (vocab.RESUBMIT_ID_REASONS); the agent suggests it, the analyst decides.
RESUBMIT_CROPPED = "Cropped"
RESUBMIT_DAMAGED = "This ID is not in a suitable state"
RESUBMIT_ILLEGIBLE = "The document you have provided is illegible"
RESUBMIT_UNCLEAR = ("Please resubmit a clear, good quality valid proof of identity and ensure all information on the "
                    "document is visible")
RESUBMIT_EXPIRED = "The document you have submitted is expired"


# ── which case, which images ─────────────────────────────────────────────────

def eligible(case):
    """An individual (not a business) in Data Confirming whose screening the rules
    cleared (auto-verified): the agent confirms it so it verifies itself.
    Everything else is an analyst's from the start."""
    return (case["tenant"] == "ttlive" and case["status"] == "DataConfirming" and not case.get("agency")
            and case["screening"].get("workflow") == "auto-verified")


def _id_doc(case):
    return next((d for d in case["documents"] if d.get("category") == "Passport"), None) or case["documents"][0]


def _image(png):
    return {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": base64.b64encode(png).decode()}}


def read(case, render, call=None, sleep=time.sleep):
    """Claude's read of the ID and the proof of address: (read, usage, seconds)."""
    call = call or llm.call
    content = [{"type": "text", "text": "Image 1 is the customer's ID (passport data page). Read it into `id`."},
               _image(svg_to_png(render(case, _id_doc(case))))]
    poa = workflow.poa_docs(case)
    if poa:
        content += [{"type": "text", "text": "Image 2 is the customer's proof of address. Read it into `poa`, with the "
                                             "holder's (account holder's, cardholder's) full name as printed in `surname`. "
                                             "An S/O, D/O, W/O or C/O line in an address names a relative, not the holder."},
                    _image(svg_to_png(render(case, poa[0])))]
    else:
        content.append({"type": "text", "text": "There is no proof of address: return `poa` unreadable with empty fields."})
    return _ask(call, SYSTEM, content, SCHEMA, 3000, sleep)


def _ask(call, system, content, schema, max_tokens, sleep):
    """One model call; a busy gateway (429) or a dropped connection is worth
    another try."""
    for attempt in range(RETRIES):
        try:
            return call(system, content, schema, max_tokens=max_tokens)
        except WorkflowError as exc:
            if not any(m in str(exc) for m in (" 429", "Could not reach")) or attempt == RETRIES - 1:
                raise
            sleep(RETRY_WAIT * (attempt + 1))


# ── Claude's own verdict (advisory) ──────────────────────────────────────────

def verdict_input(case, got, now=None):
    """What the verdict call is told: today, the policy, the record, the
    vendor's result and the read. Never the rules' checks or decision."""
    now = now or dates.now()
    client, vc = case["client"], case.get("vendor_check") or {}
    poa = workflow.poa_docs(case)
    record = {"legal_name": client["legal_name"], "date_of_birth": client["dob"], "nationality": client["nationality"],
              "id_type": client.get("id_type", ""), "id_number": client.get("id_number", ""),
              "registered_address": client.get("address", ""), "proof_of_address_required": bool(case.get("poa_required")),
              "proof_of_address_document": poa[0]["name"] if poa else ""}
    vendor = vc.get("result", "") + (f" ({vc['vendor_reason']})" if vc.get("vendor_reason") else "")
    return (f"Today is {now:%Y-%m-%d}.\nPolicy: {POLICY}\nCustomer record: {json.dumps(record, ensure_ascii=False)}\n"
            f"ID-verification vendor: {vendor or 'no result'}\n"
            f"Read from the documents: {json.dumps(got, ensure_ascii=False)}")


def opinion(case, got, call=None, sleep=time.sleep):
    """Claude's verdict on a read: {decision, confidence, reason, ...} or
    {error}. Touches nothing."""
    started = time.monotonic()
    out = {"at": dates.now_str(), "model": MODEL, "policy": POLICY_VERSION}
    try:
        ans, usage, seconds = _ask(call or llm.call, VERDICT_SYSTEM, [{"type": "text", "text": verdict_input(case, got)}],
                                   VERDICT_SCHEMA, 2000, sleep)
        decision = ans.get("decision") if isinstance(ans, dict) else None
        if decision not in VERDICT_LABELS:
            raise WorkflowError("The model's verdict was not verify or send_to_person.")
        out.update(decision=decision, confidence=max(0, min(100, int(ans.get("confidence") or 0))),
                   reason=str(ans.get("reason") or "")[:300], usage=usage, seconds=seconds)
    except (WorkflowError, TypeError, ValueError) as exc:
        out["error"] = str(exc)
    out.setdefault("seconds", round(time.monotonic() - started, 1))
    return out


def rules_say(record):
    """The rules' decision in the verdict's words: the documents passed every
    check (verified, or confirmed with an alert left) or went to a person."""
    return "verify" if record.get("decision") in ("verified", "screening") else "send_to_person"


def attach_verdict(record, verdict):
    """Put the verdict on the agent's record, with whether it agrees; a failed
    call counts its tries, after the failed tries before it."""
    before = record.get("verdict")
    if verdict.get("error"):
        again = before is not verdict and bool(before) and bool(before.get("error")) \
            and before.get("policy", 1) == verdict.get("policy", 1)
        verdict["tries"] = before.get("tries", 1) + 1 if again else 1
    else:
        verdict["agrees"] = verdict["decision"] == rules_say(record)
    record["verdict"] = verdict
    return verdict


def needs_verdict(case):
    """A case the agent read before verdicts existed, whose verdict was given
    under an older policy, or whose verdict call failed fewer than
    VERDICT_TRIES times: it gets one from its stored read, with no new look
    at the images."""
    rec = case.get("ai_agent") or {}
    if not rec.get("read") or rec.get("error"):
        return False
    verdict = rec.get("verdict")
    if verdict is None or verdict.get("policy", 1) != POLICY_VERSION:
        return True
    return bool(verdict.get("error")) and verdict.get("tries", 1) < VERDICT_TRIES


# ── the checks ───────────────────────────────────────────────────────────────

def _name_key(text):
    return sorted(similarity.tokens(text))


# Names a holder often goes without (a Telugu prefix, Mohammed): a CRM that
# drops one keeps the holder's own name all the same.
DROPPED_FIRST = names.GIVEN_PREFIXES | {"MOHAMMED", "MOHAMMAD", "MUHAMMAD", "MOHD", "MD"}


def _own_name_kept(given, rest, country=""):
    """An ID with two given names, the second of them the father's (an
    Indian passport's MANASI PRAKASH): the CRM must keep the first, or its
    initial. A CRM that keeps only the second names the father (PRAKASH
    MORE): another person. A Vietnamese passport's personal name is its
    last, and a longer name is often called by its last (VENKATA LAKSHMI
    DURGA as DURGA), so neither is held to this."""
    if country == "VN" or len(given) != 2 or given[0] in DROPPED_FIRST:
        return True
    own = given[0]
    return any(names.given_match(own, b) >= 0.9 or (names.is_initial(b) and b == own[0]) for b in rest)


def name_verdict(crm, surname, given, filed="", country=""):
    """Is the CRM name the person the ID names? The hub's own name rules
    decide: the ID's surname is among the CRM's names (any typing, a
    compound part, a spelling variant) and the holder's own given name
    agrees (exact, nickname, short form or spelling variant; an initial
    alone is not enough; _own_name_kept). `filed` is the name as the file
    wrote it before the console put it in capitals: broken characters
    (UTF-8 read as Windows-1252) repair from that text, since capitals
    change the bytes they stand for, and a byte the file itself lost stands
    for one letter. `country` is the ID's. Returns (same, value to
    confirm)."""
    source = filed if filed and filed.upper() == crm else crm
    crm_fixed, repaired = names.repair_mojibake(source)
    c, s, g = similarity.tokens(crm_fixed), similarity.tokens(surname), similarity.tokens(given)
    if not c or not (s or g):
        return False, crm
    if repaired is False:
        found = _lost_letters_match(names.mojibake_lost(source), s + g)
        if found:
            return True, " ".join(found)
    if _name_key(crm_fixed) == sorted(s + g):
        return True, (similarity.fold(" ".join(c)) if repaired else crm)
    if not s:
        # a single-name passport: the CRM's other tokens are fillers (FNU),
        # repeats or initials the record system added
        core = {t for t in c if t not in FILLERS and not names.is_initial(t)}
        return (bool(core) and core <= set(g)), " ".join(g)
    sur_ok = names.surname_score(c, s) >= 0.8 or (
        # a patronymic initial: BALAJI K for KRISHNAN BALAJI
        any(names.is_initial(t) and t[0] == s[0][0] for t in c))
    rest = [t for t in c if t not in names.surname_forms(s)]
    giv_ok = not g or (any(names.given_match(a, b) >= 0.9 for a in g for b in rest) and _own_name_kept(g, rest, country))
    if not (sur_ok and giv_ok):
        return False, crm
    # the ID's spelling, in the order the CRM keeps (surname first or last)
    surname_first = c[0] in names.surname_forms(s)
    return True, " ".join(s + g if surname_first else g + s)


def _lost_letters_match(lossy, printed):
    """The ID's name tokens, in the record's order, when each token of a
    record name that lost characters matches a different one of them: a lost
    character stands for exactly one letter, every other letter is the same.
    None when they do not."""
    if not lossy:
        return None
    want = [re.compile("".join("[A-Z]" if ch == "\ufffd" else re.escape(similarity.fold(ch)) for ch in tok) + "$")
            for tok in lossy.split()]
    left, found = list(printed), []
    if len(want) != len(left):
        return None
    for w in want:
        hit = next((t for t in left if w.match(t)), None)
        if hit is None:
            return None
        left.remove(hit)
        found.append(hit)
    return found


def poa_name_verdict(crm, poa, filed="", country=""):
    """Is the proof of address in the CRM name? A proof of address prints the
    holder's name in one line, with no surname field, so each of its names is
    tried as the surname, the rest as the given names, under the ID's rules
    (name_verdict). True, False, or None when no name was read."""
    printed = similarity.tokens(" ".join((poa.get("surname", ""), poa.get("given_names", ""))))
    if not printed:
        return None
    if poa.get("surname") and poa.get("given_names") and name_verdict(crm, poa["surname"], poa["given_names"], filed, country)[0]:
        return True
    if len(printed) == 1:
        return name_verdict(crm, "", printed[0], filed, country)[0]
    return any(name_verdict(crm, t, " ".join(printed[:i] + printed[i + 1:]), filed, country)[0] for i, t in enumerate(printed))


def _filed_name(case):
    """The record's name as the reconciliation log wrote it, for a case
    built from the file; '' for any other case."""
    from fincrime.data import recon_book                 # lazy: recon_book imports the seed builders
    v = recon_book.VIEWS.get((case.get("recon") or {}).get("id", ""))
    return v["customer"]["name"] if v else ""


def _poa_address(read, country):
    """The demo's proof-of-address images print the passport country after
    every address, so an overseas one reads '..., CA 92683, USA, Vietnam'.
    Drop that tail when the address before it names another country."""
    name = vocab.COUNTRY_NAMES.get(country, "")
    head, _, tail = read.rpartition(",")
    if name and head and similarity.fold(tail) == similarity.fold(name)             and addresses.normalise(head, country)["country"] not in ("", country):
        return head.strip()
    return read


# What makes a registered address out of date when the proof of address
# matches it: a city or district under an older or other name (Bangalore,
# Secunderabad) or Vietnam's pre-2025 units, where the document names the
# current ones (addresses.normalise flags).
STALE_ADDRESS = {"legacy_city", "district_alias", "neighbourhood_alias", "pre2025_unit"}


def address_update(case, got):
    """The proof of address's address, when the record's should be corrected
    to it as an analyst would: the documents' postal code where the record's
    was keyed wrong (226001 for 226010), or the current place names where
    the record keeps older ones. '' when the two differ only in how they are
    written (abbreviations, ZIP+4, the order of a flat and wing)."""
    country, now = case["client"]["nationality"], case["client"].get("address", "")
    read = _poa_address(((got or {}).get("poa") or {}).get("address", ""), country)
    reg, poa = addresses.normalise(now, country), addresses.normalise(read, country)
    if addresses.compare(reg, poa) != "same":
        return ""
    wrong_code = bool(reg["postal5"] and poa["postal5"]) and reg["postal5"] != poa["postal5"]
    stale = (set(reg["flags"]) & STALE_ADDRESS) - set(poa["flags"])
    if not wrong_code and not stale:
        return ""
    if not stale:
        # only the code was keyed wrong: correct it in the record's own words (the state the bill leaves out stays)
        text = re.sub(rf"\b{reg['postal5']}(-\d{{4}})?\b", poa["postal5"], now, count=1)
    else:
        # the document's address, as the record keeps one: no relative's line (S/O: Krishnan, ...), no country after its own
        text = addresses.RELATION.sub("", read).strip()
        head, _, tail = text.rpartition(",")
        if head and addresses.COUNTRY_WORDS.get(similarity.fold(tail).strip()) == country:
            text = head.strip()
    return "" if similarity.fold(text) == similarity.fold(now) else text


def write_address(case, record, address):
    """Correct the record's registered address to the proof of address's."""
    before = case["client"].get("address", "")
    case["client"]["address"] = address
    workflow._change(case, ACTOR, "address", before, address)
    workflow._audit(case, ACTOR, "Address updated", f"Registered address corrected to the proof of address's: {address} (was {before}).")
    record["address_updated"] = {"from": before, "to": address}


def resubmit_reason(got, record_checks):
    """The Resubmit ID reason that fits what the agent found on the ID, or ""
    when the ID itself is fine (a name or address mismatch is not the
    customer's photo to retake)."""
    idr = (got or {}).get("id") or {}
    problems = set(idr.get("problems") or [])
    failed = {c["check"] for c in record_checks if not c["ok"]}
    if "ID readable" in failed:
        if "cut_off" in problems:
            return RESUBMIT_CROPPED
        if "damaged" in problems:
            return RESUBMIT_DAMAGED
        if problems & {"blurred", "too_dark"}:
            return RESUBMIT_ILLEGIBLE
        if problems & {"covered", "angled", "glare"}:
            return RESUBMIT_UNCLEAR
        if idr.get("edges_visible") is False:        # out of shot, and nothing else named
            return RESUBMIT_CROPPED
        return RESUBMIT_UNCLEAR
    expiry = next((c for c in record_checks if c["check"] == "Expiry"), None)
    if expiry and not expiry["ok"] and expiry["note"].startswith("ID expired"):
        return RESUBMIT_EXPIRED
    return ""


def _check(out, check, crm, document, ok, note=""):
    out.append({"check": check, "crm": crm or "", "document": document or "", "ok": bool(ok), "note": note})


def checks(case, got, now=None):
    """Every check the agent makes on Claude's read, in order; each says why
    it failed. Also returns the values to confirm when they all pass."""
    now = now or dates.now()
    client = case["client"]
    idr, poa = got.get("id") or {}, got.get("poa") or {}
    out = []
    named = [p for p in idr.get("problems", []) if p in SPOILERS]
    spoilt = [SPOILERS[p] for p in named if p not in SOFT]
    if idr.get("edges_visible") is False and "cut off" not in spoilt:
        spoilt.append("its edges are not all in the picture")
    if idr.get("mrz_visible") is False and _id_doc(case).get("category") == "Passport":
        spoilt.append("the MRZ is not fully visible")
    if spoilt:
        spoilt += [SPOILERS[p] for p in named if p in SOFT]
    ok = bool(idr.get("readable")) and not spoilt
    _check(out, "ID readable", "", "readable" if ok else "not readable", ok,
           "" if ok else f"ID could not be read: {', '.join(spoilt) or idr.get('note') or 'unclear'}")

    printed = " ".join(x for x in (idr.get("surname", ""), idr.get("given_names", "")) if x)
    same, name_value = name_verdict(client["legal_name"], idr.get("surname", ""), idr.get("given_names", ""),
                                    filed=_filed_name(case), country=client["nationality"])
    _check(out, "Name", client["legal_name"], printed, same,
           "" if same else ("Name not readable on the ID" if not printed else "Name on the ID differs from the CRM"))

    dob = idr.get("dob", "")
    ok = bool(dob) and dob == client["dob"]
    _check(out, "Date of birth", client["dob"], dob or "not on the ID", ok,
           "" if ok else ("Date of birth missing on the ID" if not dob else f"Date of birth on the ID {dob}, CRM {client['dob']}"))

    code = (idr.get("nationality_code") or "").upper()
    nat = ALPHA2.get(code, code[:2] if len(code) == 2 else "")
    ok = nat == client["nationality"]
    _check(out, "Nationality", client["nationality"], code, ok, "" if ok else "Nationality on the ID differs from the CRM")

    number = similarity.squash(similarity.fold(idr.get("document_number", "")))
    ok = bool(number) and number == similarity.squash(similarity.fold(client.get("id_number", "")))
    _check(out, "ID number", client.get("id_number", ""), idr.get("document_number", ""), ok,
           "" if ok else "ID number on the document differs from the CRM")

    expiry = dates.parse(idr.get("expiry", ""))
    ok = expiry is not None and expiry >= now
    _check(out, "Expiry", "", idr.get("expiry", "") or "not readable", ok,
           "" if ok else (f"ID expired on {idr['expiry']}" if expiry else "Expiry date not readable"))

    vc = case.get("vendor_check") or {}
    ok = vc.get("result") != "Declined"
    _check(out, "Vendor check", "", vc.get("result", ""), ok,
           "" if ok else "ID declined by the vendor" + (f": {vc['vendor_reason']}" if vc.get("vendor_reason") else ""))

    if case.get("poa_required"):
        _check(out, "Proof of address readable", "", "readable" if poa.get("readable") else "not readable",
               poa.get("readable"), "" if poa.get("readable") else "Proof of address could not be read")
        if poa.get("readable"):
            holder = " ".join(x for x in (poa.get("surname", ""), poa.get("given_names", "")) if x)
            same = poa_name_verdict(client["legal_name"], poa, filed=_filed_name(case), country=client["nationality"])
            _check(out, "Name on proof of address", client["legal_name"], holder or "not read", same,
                   "" if same else ("No holder name could be read on the proof of address" if same is None
                                    else "Name on the proof of address differs from the CRM"))
        country = client.get("nationality", "")
        verdict = addresses.compare(addresses.normalise(client.get("address", ""), country),
                                    addresses.normalise(_poa_address(poa.get("address", ""), country), country))
        _check(out, "Address", client.get("address", ""), poa.get("address", ""), verdict == "same",
               "" if verdict == "same" else f"Address on the proof of address does not match the CRM ({verdict.replace('_', ' ')})")
        failures = workflow.poa_failures(case, now)
        issued = dates.parse(poa.get("issued_on", ""))
        kind = (workflow.poa_docs(case) or [{}])[0].get("kind", "")
        if issued is not None and (now - issued).days > vocab.POA_MAX_AGE_DAYS and kind not in vocab.POA_NO_AGE_LIMIT \
                and not any(f["label"] == "Issued within the last 6 months" and poa["issued_on"] in f["detail"] for f in failures):
            # the document's own date, when the upload's record does not already report that date too old
            failures.append({"label": "Issued within the last 6 months", "detail": f"the document reads {poa['issued_on']}"})
        if issued is None and poa.get("readable") and kind not in vocab.POA_NO_AGE_LIMIT:
            # the record's date proves nothing about the document: its own date must be legible
            failures.append({"label": "Issue date readable", "detail": "the date on the document could not be read"})
        _check(out, "Proof of address checks", "", "; ".join(f["detail"] for f in failures) or "passed", not failures,
               "; ".join(f"{f['label']}: {f['detail']}" for f in failures))
    values = {"name": name_value, "dob": client["dob"], "nationality": client["nationality"]}
    return out, values


# ── one case ─────────────────────────────────────────────────────────────────

def perceive(case, render, call=None, sleep=time.sleep):
    """The slow half, safe to run outside any lock: Claude's read of the
    documents, or the error that stopped it. Touches nothing."""
    started = time.monotonic()
    out = {"at": dates.now_str(), "model": MODEL, "read": {}, "usage": {}, "seconds": 0.0, "error": ""}
    try:
        got, usage, seconds = read(case, render, call=call, sleep=sleep)
        out.update(read=got, usage=usage, seconds=seconds)
    except WorkflowError as exc:
        out["error"] = str(exc)
    out["seconds"] = out["seconds"] or round(time.monotonic() - started, 1)
    if not out["error"]:
        out["verdict"] = opinion(case, out["read"], call=call, sleep=sleep)
    return out


def decide(case, alert_case, wc, perceived):
    """The quick half: check the read and act on it. Mutates the case (and,
    when it confirms, the screening and watchlist cases the rules touch);
    the caller saves them."""
    record = dict(perceived, checks=[], reasons=[], rules=RULES_VERSION)
    values = {}
    if record["error"]:
        record["reasons"] = [f"AI Agent could not read the documents: {record['error']}"]
    else:
        record["checks"], values = checks(case, record["read"])
        record["reasons"] = [c["note"] for c in record["checks"] if not c["ok"]]
    if not record["reasons"]:
        workflow._audit(case, ACTOR, "Documents read", f"{MODEL} read the ID and the proof of address; every check passed.")
        try:
            workflow.confirm_data(case, ACTOR, values, "Confirmed by the AI Agent against the ID and the proof of address.",
                                  alert_case=alert_case, wc=wc)
        except WorkflowError as exc:
            record["reasons"] = [str(exc)]
        if not record["reasons"]:
            address = address_update(case, record["read"])
            if address:
                write_address(case, record, address)
    if record["reasons"]:
        record["decision"] = "human"
        record["resubmit"] = resubmit_reason(record["read"], record["checks"])
        case["screening"]["workflow"] = "human-in-the-loop"
        workflow._audit(case, ACTOR, "Sent to a person", "; ".join(record["reasons"]))
    else:
        record["decision"] = "verified" if case["status"] == "Verified" else "screening"
    if record.get("verdict"):
        attach_verdict(record, record["verdict"])            # shown and counted; the rules above decided
    case["ai_agent"] = record
    return record


def needs_recheck(case):
    """A case the agent sent to a person under older checks, still in Data
    Confirming, and untouched since: no one reserved it or acted on it."""
    rec = case.get("ai_agent") or {}
    return (rec.get("decision") == "human" and bool(rec.get("read")) and not rec.get("error")
            and rec.get("rules", 1) != RULES_VERSION and case["status"] == "DataConfirming"
            and not case.get("current_reviewer")
            and all(e.get("actor") == ACTOR for e in case.get("audit", []) if e.get("at", "") >= rec.get("at", "")))


def _untouched(case):
    """No person has reserved or acted on the case since the agent's read."""
    rec = case.get("ai_agent") or {}
    return not case.get("current_reviewer") and all(
        e.get("actor") in (ACTOR, vocab.SYSTEM_USER) for e in case.get("audit", []) if e.get("at", "") >= rec.get("at", ""))


def needs_reread(case):
    """A case read before the checks wanted the proof of address's name, whose
    readable proof of address gave none (the read was not asked for it), and
    no person has touched since: it starts again and is read again
    (app._load_seed), rather than sent to a person for a name it printed."""
    rec = case.get("ai_agent") or {}
    poa = (rec.get("read") or {}).get("poa") or {}
    return (bool(case.get("poa_required")) and rec.get("rules", 1) < POA_NAME_RULES and bool(poa.get("readable"))
            and not (poa.get("surname") or poa.get("given_names")) and _untouched(case))


def confirmed_on_older_names(stored, seed):
    """A case the agent confirmed before the checks wanted the holder's own
    given name, whose stored read the name checks now refuse against the
    record as the file gave it (`seed`: the confirmation rewrote the name),
    and no person has touched since: it starts again and is read again
    (app._load_seed), and goes to a person."""
    rec = stored.get("ai_agent") or {}
    if (rec.get("decision") not in ("verified", "screening") or rec.get("rules", 1) >= OWN_NAME_RULES
            or not rec.get("read") or not _untouched(stored)):
        return False
    found, _ = checks(seed, rec["read"])
    return any(not c["ok"] for c in found if c["check"] in ("Name", "Name on proof of address"))


def needs_address_update(case):
    """A case the agent confirmed before it wrote the verified address back,
    whose record keeps an address its proof of address corrects, and no
    person has touched since (update_address)."""
    rec = case.get("ai_agent") or {}
    return (rec.get("decision") in ("verified", "screening") and rec.get("rules", 1) < RULES_VERSION
            and bool(rec.get("read")) and not rec.get("address_updated") and _untouched(case)
            and bool(address_update(case, rec["read"])))


def update_address(case):
    """The address write-back on a case confirmed before it existed: no new
    look at the images; the record's checks are now this version's."""
    rec = case["ai_agent"]
    write_address(case, rec, address_update(case, rec["read"]))
    rec["rules"] = RULES_VERSION
    return rec


def recheck(case, alert_case, wc):
    """The current checks on the stored read, with no new look at the images.
    A case that now passes them is confirmed as decide() confirms one; a case
    that still fails stays with a person, its reasons brought up to date."""
    old = case["ai_agent"]
    found, _ = checks(case, old["read"])
    if any(not c["ok"] for c in found):
        old.update(checks=found, reasons=[c["note"] for c in found if not c["ok"]], rules=RULES_VERSION,
                   resubmit=resubmit_reason(old["read"], found))
        return old
    workflow._audit(case, ACTOR, "Checked again", "The agent's checks changed; the stored read now passes every one.")
    case["screening"]["workflow"] = "auto-verified"
    return decide(case, alert_case, wc, {k: v for k, v in old.items()
                                         if k in ("at", "model", "read", "usage", "seconds", "error", "verdict")})


def run_case(case, alert_case, wc, render, call=None, sleep=time.sleep):
    """Read, check, decide - one case, start to finish."""
    return decide(case, alert_case, wc, perceive(case, render, call=call, sleep=sleep))
