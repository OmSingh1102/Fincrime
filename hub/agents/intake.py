"""Document Intake Agent: are the uploads acceptable, and if not, what does
the customer have to do? A possible fraud upload is never bounced to the
customer; it goes to a person."""
import time

from fincrime.data import vocab
from fincrime.hub import documents
from fincrime.hub.agents import finding, skipped, step
from fincrime.hub.similarity import fold, squash, tokens
from fincrime.utils import dates

POA_MAX_DAYS = {"lenient": 365, "standard": 183, "strict": 90}

REASONS = {
    "DOC_FLAT_SCAN": {"label": "Flat scan or PDF instead of a camera photo", "decision": "Reject", "tone": "serious",
                      "message": "The ID you uploaded looks like a scan or a PDF rather than a photo. Please photograph the physical "
                                 "document with your phone camera, in good light, so the whole page and both code lines at the bottom are visible."},
    "DOC_PARTIAL": {"label": "Partial ID image", "decision": "Reject", "tone": "serious",
                    "message": "Part of the ID is cut off. Please photograph the whole document with all four corners and the two code "
                               "lines at the bottom of the page visible."},
    "DOC_GLARE": {"label": "Glare over the data page", "decision": "Reject", "tone": "warning",
                  "message": "A reflection covers part of the ID. Please retake the photo away from direct light and without flash."},
    "DOC_BLUR": {"label": "Blurred image", "decision": "Reject", "tone": "warning",
                 "message": "The photo is too blurred to read. Please hold the phone steady, let the camera focus, and retake it."},
    "DOC_OBSTRUCTED": {"label": "Something lies on the ID", "decision": "Reject", "tone": "warning",
                       "message": "Something is lying on top of your ID, so part of it cannot be read. Please take everything off the "
                                  "document and retake the photo so the whole page is visible."},
    "DOC_ANGLE": {"label": "ID not photographed straight on", "decision": "Reject", "tone": "warning",
                  "message": "The photo of your ID is taken at an angle or from too far away. Please lay the document flat, hold the "
                             "phone straight above it so the page fills the frame, and retake it in good light."},
    "DOC_DAMAGED": {"label": "Damaged ID", "decision": "Reject", "tone": "serious",
                    "message": "Your ID is cracked or worn, so some of its details cannot be read. Please upload an undamaged "
                               "government-issued photo ID, such as a valid passport or national ID card."},
    "DOC_EXPIRED": {"label": "Expired passport", "decision": "Reject", "tone": "serious",
                    "message": "The passport you uploaded has expired. Please upload a valid passport or another government-issued photo ID."},
    "DOC_MRZ_CHECK_FAIL": {"label": "MRZ check digits fail", "decision": "Refer", "tone": "critical", "message": ""},
    "DOC_VIZ_MRZ_MISMATCH": {"label": "Data page and MRZ disagree", "decision": "Refer", "tone": "critical", "message": ""},
    "POA_TOO_OLD": {"label": "Proof of address older than allowed", "decision": "Reject", "tone": "warning",
                    "message": "Your proof of address is older than {months} months. Please upload a bank statement or utility bill "
                               "issued in the last {months} months that shows your name and current address."},
    "POA_PO_BOX": {"label": "PO Box given as the residential address", "decision": "Reject", "tone": "warning",
                   "message": "A PO Box cannot be accepted as a residential address. Please give the street address where you live "
                              "and upload a recent document (bank statement or utility bill) that shows it. You can keep the PO Box as a mailing address."},
    "ADDR_UNRESOLVED": {"label": "Addresses on file point at different residences", "decision": "Note", "tone": "warning",
                        "message": "The address on your documents does not match the address you registered with us. Please upload a "
                                   "current proof of address (bank statement, utility bill or lease) for the address where you live, "
                                   "or update your registered address."},
}


def customer_message(code, strictness="standard"):
    r = REASONS.get(code)
    if not r or not r["message"]:
        return ""
    months = {"lenient": 12, "standard": 6, "strict": 3}[strictness]
    return r["message"].format(months=months)


def run(case, settings):
    started = time.perf_counter()
    if not settings["agents"].get("intake", True):
        case["intake"] = {"decision": "Accept", "reason_code": "", "codes": [], "message": "", "address_unresolved": False, "off": True}
        return skipped("intake", started)
    strictness = settings.get("doc_strictness", "standard")
    s, docs, files = case["synth"], case["documents"], case["files"]
    identity = case.get("identity") or {}
    address = identity.get("address") or {}
    passport = next(f for f in files if f["id"] == "passport")
    defects = set(passport.get("defects") or [])
    findings, codes = [], []

    def hit(code, text=None, tone=None):
        r = REASONS[code]
        codes.append(code)
        findings.append(finding(code, text or r["label"], tone or r["tone"]))

    # the passport photo
    if "flat_scan" in defects:
        hit("DOC_FLAT_SCAN", "The passport image is a flat scan or PDF, not a camera capture (no perspective, uniform lighting, scanner border).")
    elif "partial" in defects:
        hit("DOC_PARTIAL", "The passport image is cut off: the machine-readable zone and the lower edge are missing.")
    elif "glare" in defects:
        hit("DOC_GLARE", "A reflection covers the lower part of the data page; the MRZ cannot be read in full.")
    elif "blur" in defects:
        hit("DOC_BLUR", "The passport image is too blurred to read the data page reliably.")
    elif "obstructed" in defects:
        hit("DOC_OBSTRUCTED", "A phone lies on the data page; the dates of birth and expiry and part of the MRZ are hidden.")
    elif "angled" in defects:
        hit("DOC_ANGLE", "The passport is photographed at an angle in dim light, and a phone hides its lower corner.")
    elif "damaged" in defects:
        hit("DOC_DAMAGED", "The passport is cracked and worn; the number, the dates and the MRZ's dates cannot be read.")
    else:
        findings.append(finding("CAPTURE", "Camera-captured image; all four corners and the MRZ visible.", "good"))
    # expiry, from the printed dates
    received = dates.parse(s["received_at"])
    expiry = dates.parse(s["passport_expiry"])
    if expiry and received and expiry < received:
        hit("DOC_EXPIRED", f"The passport expired on {s['passport_expiry']}.")
    else:
        findings.append(finding("EXPIRY", f"Passport valid until {s['passport_expiry']}.", "good"))
    # MRZ check digits and VIZ against MRZ (when readable)
    mrz_readable = not ({"partial", "glare", "blur", "obstructed", "angled", "damaged"} & defects)
    if mrz_readable:
        ok, problems, line2 = documents.mrz_checks(case)
        if ok:
            findings.append(finding("MRZ", "MRZ check digits verify (document number, date of birth, expiry).", "good"))
        else:
            hit("DOC_MRZ_CHECK_FAIL", "MRZ check digits do not verify: " + ", ".join(problems) +
                ". The page may have been altered; a person must look at it.")
        viz, mrz = docs["passport"]["viz"], docs["passport"]["mrz"]
        viz_sur, viz_given = tokens(viz["surname"]), tokens(viz["given"])
        mrz_sur, mrz_given = [fold(t) for t in mrz.get("primary_tokens", [])], [fold(t) for t in mrz.get("secondary", [])]
        sur_joined, mrz_sur_joined = squash("".join(viz_sur)), squash("".join(mrz_sur))
        sur_ok = sur_joined == mrz_sur_joined or (not viz_sur and not mrz_sur) \
            or (mrz.get("truncated") and sur_joined.startswith(mrz_sur_joined))
        given_joined, mrz_joined = squash("".join(viz_given)), squash("".join(mrz_given))
        given_ok = given_joined == mrz_joined or (mrz.get("truncated") and given_joined.startswith(mrz_joined)) \
            or given_joined.replace("JR", "").replace("SR", "").replace("III", "").replace("II", "") == mrz_joined.replace("JR", "").replace("SR", "").replace("III", "").replace("II", "")
        if sur_ok and given_ok:
            findings.append(finding("VIZ", "Visual zone and MRZ carry the same name" + (" (MRZ truncated at 39 characters, tolerated)." if mrz.get("truncated") else "."), "good"))
        else:
            hit("DOC_VIZ_MRZ_MISMATCH", f"The printed name ({viz['surname']} / {viz['given']}) and the MRZ ({mrz.get('primary', '')} / {' '.join(mrz.get('secondary', []))}) disagree.")
    else:
        findings.append(finding("MRZ", "MRZ not readable on this capture; verified after re-upload.", "warning"))
    # secondary ID against the passport
    name = identity.get("name") or {}
    if "surname_conflict" in name.get("flags", []):
        findings.append(finding("ID2", "The secondary ID carries another surname (married or maiden name); both are screened.", "warning"))
    elif docs["secondary"]["name"]:
        findings.append(finding("ID2", f"{docs['secondary']['type']}: name agrees with the passport after normalisation.", "good"))
    # proof of address
    poa = docs["poa"]
    max_days = POA_MAX_DAYS.get(strictness, 183)
    if poa["kind"] in vocab.POA_NO_AGE_LIMIT:
        findings.append(finding("POA", f"{poa['type']}: a permanent card, no age limit.", "good"))
    elif s["poa_age_days"] > max_days:
        hit("POA_TOO_OLD", f"{poa['type']} issued {s['poa_age_days']} days ago; the limit is {max_days} days.")
    else:
        findings.append(finding("POA", f"{poa['type']} issued {s['poa_age_days']} days ago (within {max_days} days).", "good"))
    if poa["kind"] == "military_orders":
        findings.append(finding("POA", "Military orders / LES accepted as proof of address for a service member.", "neutral"))
    if address.get("registered", {}).get("po_box"):
        hit("POA_PO_BOX", "The registered address is a PO Box; the documents show a street address, which must be captured as the residence.")
    address_unresolved = False
    if address:
        if address.get("resolved"):
            findings.append(finding("ADDR", "Proof of address agrees with the registered address" +
                                    (" after normalisation (" + ", ".join(sorted(set(f.replace('_', ' ') for f in address['flags'] if f not in ('po_box', 'po_box_registered')))) + ")." if address["flags"] else "."), "good"))
        for n in address.get("notes", []):
            if "explain" in n.lower() or "governs" in n.lower() or "permanent" in n.lower() or "home of record" in n.lower() or "overseas" in n.lower():
                findings.append(finding("ADDR", "Accepted with a note: " + n, "neutral"))
        if address.get("conflict"):
            address_unresolved = True
            hit("ADDR_UNRESOLVED", "The documents point at different residences and nothing on file explains it.")
    # decision
    refer = [c for c in codes if REASONS[c]["decision"] == "Refer"]
    reject = [c for c in codes if REASONS[c]["decision"] == "Reject"]
    if "DOC_GLARE" in reject and strictness == "lenient":
        reject.remove("DOC_GLARE")
    if "DOC_BLUR" in reject and strictness == "lenient":
        reject.remove("DOC_BLUR")
    if refer:
        decision, code = "Refer", refer[0]
    elif reject:
        decision, code = "Reject", reject[0]
    else:
        decision, code = "Accept", ""
    message = customer_message(code, strictness) if decision == "Reject" else ""
    case["intake"] = {"decision": decision, "reason_code": code, "codes": codes, "message": message,
                      "address_unresolved": address_unresolved, "off": False,
                      "reasons": [REASONS[c]["label"] for c in codes]}
    confidence = 99 if decision == "Accept" and not codes else 96 if decision == "Reject" else 70
    reasons = []
    if decision == "Accept":
        reasons.append("Every upload passed its checks" + ("; notes recorded for the reviewer." if codes else "."))
    elif decision == "Reject":
        reasons.append(f"{REASONS[code]['label']}: the customer is asked to upload again; no analyst time spent.")
    else:
        reasons.append(f"{REASONS[code]['label']}: possible manipulation, so a person reviews it before anything is sent to the customer.")
    return step("intake", started, decision, confidence=confidence, findings=findings, reasons=reasons,
                needs_human=(decision == "Refer"), reason_code=code, inputs={"uploads": [f["label"] for f in files],
                                                                           "capture": s["capture"], "poa_age_days": s["poa_age_days"]},
                customer_message=message)
