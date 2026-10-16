"""Watchlist Adjudication Agent: is the watchlist record the same person as
the client? Judged from the normalised identity, the date of birth read
from the record itself, and the jurisdiction - never from the annotations
or the historical outcome in the log."""
import time
import unicodedata

from fincrime.hub import addresses, names
from fincrime.hub.agents import finding, skipped, step
from fincrime.hub.similarity import fold, squash, tokens
from fincrime.utils import dates

VERDICTS = ("different person", "same person", "unclear")
SUGGESTED = {"different person": "False Positive", "same person": "True Positive", "unclear": "Potential Match"}
COUNTRY_NAME = {"IN": "India", "VN": "Vietnam", "US": "United States", "AE": "United Arab Emirates", "CA": "Canada",
                "AU": "Australia", "DE": "Germany", "KR": "South Korea", "SG": "Singapore", "QA": "Qatar", "OM": "Oman",
                "GB": "United Kingdom", "HK": "Hong Kong"}


# ── date of birth ────────────────────────────────────────────────────────────

def dob_analysis(client_iso, wc):
    """The decision table over the raw date strings."""
    c = dates.parse(client_iso)
    cy = c.year if c else None
    kind = wc.get("kind", "none")
    out = {"class": "missing", "delta": None, "client": client_iso, "listed": wc.get("raw", ""), "detail": ""}
    if not c:
        out["detail"] = "The client's date of birth is missing."
        return out
    if kind == "full":
        w = dates.parse(wc["date"])
        if not w:
            return out
        delta = abs(cy - w.year)
        if w == c:
            out.update({"class": "exact", "delta": 0, "detail": f"Full date of birth identical ({client_iso})."})
        elif w.year == c.year and w.month == c.day and w.day == c.month and c.month != c.day:
            out.update({"class": "transposed", "delta": 0,
                        "detail": f"Same year; day and month swapped ({client_iso} vs {wc['date']}) — a transposition, or two people born in the same year."})
        elif delta >= 2:
            out.update({"class": "conflict", "delta": delta, "detail": f"Full date of birth differs by {delta} years ({client_iso} vs {wc['date']})."})
        else:
            out.update({"class": "near", "delta": delta, "detail": f"Dates of birth differ by less than two years ({client_iso} vs {wc['date']})."})
    elif kind == "year":
        delta = abs(cy - wc["year"])
        if delta == 0:
            out.update({"class": "year_match", "delta": 0, "detail": f"Only the year is listed and it matches ({wc['year']}); day and month cannot be checked."})
        elif delta >= 2:
            out.update({"class": "year_conflict", "delta": delta, "detail": f"Only the year is listed and it differs by {delta} years ({cy} vs {wc['year']})."})
        else:
            out.update({"class": "year_near", "delta": delta, "detail": f"Only the year is listed and it differs by one ({cy} vs {wc['year']})."})
    elif kind == "range":
        lo, hi = wc["year"], wc["year_to"]
        if lo <= cy <= hi:
            out.update({"class": "range_match", "delta": 0, "detail": f"The record gives a birth range {lo}–{hi} that includes {cy}; not decisive."})
        else:
            delta = min(abs(cy - lo), abs(cy - hi))
            cls = "range_conflict" if delta >= 2 else "range_near"
            out.update({"class": cls, "delta": delta, "detail": f"The record gives a birth range {lo}–{hi}; the client was born in {cy}."})
    else:
        out["detail"] = "The record lists no date of birth."
    return out


# ── name ─────────────────────────────────────────────────────────────────────

def _nfc_tokens(text):
    return [t for t in unicodedata.normalize("NFC", text or "").upper().replace(",", " ").split() if t]


def name_analysis(identity_name, wc):
    """How alike the client's names and the record's are, 0–1, with the facts."""
    wc_sur = tokens(wc["surname"])
    wc_given, wc_suffix = names._suffix(tokens(wc["given"]))
    client_sur_lists = identity_name.get("screen_surnames") or []
    client_given = identity_name.get("screen_given") or []
    mononym_client = identity_name.get("mononym") and not client_sur_lists
    mononym_wc = wc.get("mononym") or not wc_given
    facts = []
    # surname
    if mononym_client or mononym_wc:
        pool_c = set(client_given) | {t for lst in client_sur_lists for t in lst}
        pool_w = set(wc_sur) | set(wc_given)
        best = 0.0
        for a in pool_c:
            for b in pool_w:
                best = max(best, names.given_match(a, b))
        sur = best
        given = best if (mononym_client and mononym_wc) else best * 0.85
        facts.append("Single-name record compared against every name token." if best else "No name token in common.")
    else:
        sur = max((names.surname_score(lst, wc_sur) for lst in client_sur_lists), default=0.0)
        if sur >= 1.0:
            facts.append(f"Surname agrees ({wc['surname']}).")
        elif sur >= 0.9:
            facts.append(f"Surname is a spelling variant ({' / '.join(' '.join(l) for l in client_sur_lists[:2])} vs {wc['surname']}).")
        elif sur >= 0.8:
            facts.append(f"One component of a compound surname matches ({wc['surname']}).")
        elif sur >= 0.6:
            facts.append("Surname given as an initial on one side.")
        else:
            facts.append(f"Surname differs ({' / '.join(' '.join(l) for l in client_sur_lists[:2])} vs {wc['surname']}).")
        if wc_given:
            scores = []
            for b in wc_given:
                scores.append(max((names.given_match(a, b) for a in client_given), default=0.0))
            given = sum(scores) / len(scores)
            weak = [b for b, sc in zip(wc_given, scores) if sc < 0.6]
            if given >= 0.95:
                facts.append(f"Given names agree ({wc['given']}).")
            elif weak:
                facts.append(f"Given name(s) on the record not on the client's documents: {' '.join(weak)}.")
            else:
                facts.append("Given names agree in part (short forms or initials).")
        else:
            given = 0.6
            facts.append("The record carries no given names.")
    # the full alias forms the record lists ("AKA: FARHEEN KURESHI")
    alias_hit = False
    client_all = set(client_given) | {t for lst in client_sur_lists for t in lst}
    for alias in wc.get("aka", []):
        at = tokens(alias)
        if at and all(any(names.given_match(a, b) >= 0.9 for a in client_all) for b in at):
            alias_hit = True
    if alias_hit:
        facts.append("An alias on the record matches the client's name in full.")
    score = 0.55 * sur + 0.45 * given
    if alias_hit:
        score = max(score, 0.97)
    flags = []
    if "common_surname" in identity_name.get("flags", []) and given < 0.9:
        score *= 0.85
        flags.append("common_surname")
        facts.append("A high-frequency family name; the given names carry the weight.")
    suffix_c, suffix_w = identity_name.get("suffix", ""), wc_suffix
    if suffix_c and suffix_w and suffix_c != suffix_w:
        score = min(score, 0.85)
        flags.append("suffix_differs")
        facts.append(f"Generational suffix differs ({suffix_c} vs {suffix_w}): the record may be a relative.")
    # the native-script alias on the record against the passport's own script
    native = wc.get("native", "")
    if native and identity_name.get("native"):
        a, b = _nfc_tokens(identity_name["native"]), _nfc_tokens(native)
        if [fold(t) for t in a] == [fold(t) for t in b] and a != b:
            diff = next(((x, y) for x, y in zip(a, b) if x != y), None)
            score *= 0.9
            flags.append("script_conflict")
            facts.append(f"The names fold to the same ASCII but differ in Vietnamese script ({diff[0]} vs {diff[1]}): two different names.")
    return {"score": round(min(score, 1.0), 3), "surname": round(sur, 2), "given": round(given, 2), "facts": facts, "flags": flags}


# ── place ────────────────────────────────────────────────────────────────────

def place_analysis(identity_address, wc_location, case_country):
    """The client's residence against every place the record lists (its
    `places`: a home, an office, where the person went)."""
    current = identity_address.get("current") or {}
    cur_city = fold(current.get("city") or "")
    cur_country = current.get("country") or case_country
    cur_country_name = COUNTRY_NAME.get(cur_country, cur_country)
    places = wc_location.get("places") or [wc_location]
    wc_cities = {fold(p.get(k) or "") for p in places for k in ("city", "former")}
    wc_cities = {addresses.VN_CITY.get(c, addresses.IN_LEGACY_CITY.get(c, addresses._US_GROUP.get(c, c))) for c in wc_cities if c}
    cur_cities = {addresses.VN_CITY.get(cur_city, addresses.IN_LEGACY_CITY.get(cur_city, addresses._US_GROUP.get(cur_city, cur_city)))}
    sec = identity_address.get("secondary") or {}
    if sec.get("city_key"):
        cur_cities.add(fold(sec["city_key"]))
    if wc_cities & cur_cities and wc_cities:
        return {"state": "agree", "detail": f"The record's jurisdiction ({wc_location.get('raw')}) names the client's city."}
    same = next((p for p in places if p.get("country") and cur_country_name and fold(p["country"]) == fold(cur_country_name)), None)
    if same:
        return {"state": "partial", "detail": f"Same country ({same['country']}), different city ({same.get('city') or 'not listed'} vs {current.get('city') or '-'})."}
    if any(p.get("country") for p in places):
        return {"state": "differ", "detail": f"The record's jurisdiction ({wc_location.get('raw')}) differs from the client's residence ({current.get('city') or ''}, {cur_country_name})."}
    return {"state": "unknown", "detail": "No jurisdiction on the record."}


# ── the agent ────────────────────────────────────────────────────────────────

def run(case, settings):
    started = time.perf_counter()
    gate = int(settings.get("gate", 85))
    wc = case["wc"]
    identity = case.get("identity") or {}
    if not settings["agents"].get("adjudication", True):
        case["screening"] = {"dob": dob_analysis(case["customer"]["dob"], wc["dob"]), "name": {"score": 0, "facts": [], "flags": []},
                             "place": {"state": "unknown", "detail": ""}, "verdict": "unclear", "confidence": 0,
                             "decision": "needs_review", "suggested": "Potential Match", "reasons": ["The adjudication agent is switched off."],
                             "hard_stops": [], "reason_code": "AGENT_OFF", "off": True}
        return skipped("adjudication", started)
    dob = dob_analysis(case["customer"]["dob"], wc["dob"])
    name = name_analysis(identity.get("name") or {}, wc)
    place = place_analysis(identity.get("address") or {}, wc["location"], case["country"])
    risk = case["customer"]["risk"]
    group = wc["group"]
    hard_stops = []
    if settings.get("sanctions_guard", True) and group == "Sanctions":
        hard_stops.append("sanctions")
    if (case.get("intake") or {}).get("decision") == "Refer":
        hard_stops.append("fraud_referral")
    cls, delta, ns = dob["class"], dob["delta"] or 0, name["score"]
    if "script_conflict" in name["flags"] and cls not in ("conflict", "year_conflict", "range_conflict"):
        hard_stops.append("script_conflict")       # blocks a confirmation, never a clear: the date of birth decides those
    reasons = [dob["detail"]] + name["facts"] + [place["detail"]]
    place_bonus = place["state"] == "agree"
    if cls in ("conflict", "year_conflict", "range_conflict"):
        verdict = "different person"
        if cls == "conflict":
            confidence = 96 if delta >= 10 else 88
        elif cls == "year_conflict":
            confidence = 92 if delta >= 10 else 85
        else:
            confidence = 80
        if place_bonus and ns >= 0.9:
            confidence -= 8
            reasons.append("Same city and a close name: a stronger coincidence, so the confidence is lowered.")
        code = "DOB_CONFLICT"
    elif cls == "exact":
        if ns >= 0.80 and "script_conflict" not in name["flags"]:
            verdict, confidence, code = "same person", 95 + (3 if place_bonus else 0) - (10 if ns < 0.9 else 0), "EXACT_MATCH"
            reasons.append("Identical full date of birth and the name agrees: treated as a confirmed match for a person to decide.")
        else:
            verdict, confidence, code = "unclear", 60, "NAME_WEAK" if "script_conflict" not in name["flags"] else "SCRIPT_CONFLICT"
            reasons.append("Identical date of birth but the name does not agree well enough to call it the same person.")
    elif cls == "transposed":
        verdict, code = "unclear", "DOB_TRANSPOSED"
        confidence = 70 if (ns >= 0.9 and place_bonus) else 60 if ns >= 0.9 else 55
        reasons.append("A day/month transposition cannot be told from two people born in the same year without another identifier.")
    elif cls in ("near", "year_near", "range_near"):
        verdict, confidence, code = "unclear", 50, "DOB_NEAR"
    else:
        verdict, code = "unclear", {"year_match": "DOB_YEAR_ONLY", "range_match": "DOB_RANGE", "missing": "DOB_MISSING"}.get(cls, "DOB_MISSING")
        confidence = int(round(40 + 30 * (1 - ns) + (10 if place["state"] == "differ" else 0)))
        confidence = min(confidence, 80)
        reasons.append("The date of birth is not decisive, so the record cannot be cleared or confirmed on the data alone.")
    confidence = max(0, min(100, int(confidence)))
    # decision
    if hard_stops and verdict != "unclear":
        if "sanctions" in hard_stops:
            reasons.append("Sanctions list record: the guard sends every sanctions hit to a person, whatever the verdict.")
            code = "SANCTIONS_GUARD"
        if "fraud_referral" in hard_stops:
            reasons.append("The intake agent referred the documents as a possible fraud upload; nothing is cleared until a person has looked.")
            code = "FRAUD_REFERRAL"
    if verdict == "different person" and confidence >= gate and not hard_stops:
        decision, label = "auto_clear", "Auto-clear (false positive)"
    elif verdict == "same person" and confidence >= gate and not hard_stops:
        decision, label = "auto_confirm", "Confirmed match — for a human"
    else:
        decision, label = "needs_review", "Needs review"
        if verdict != "unclear" and confidence < gate and not hard_stops:
            reasons.append(f"Confidence {confidence} is below the gate of {gate}.")
            code = "BELOW_GATE"
    suggested = SUGGESTED[verdict]
    case["screening"] = {"dob": dob, "name": name, "place": place, "verdict": verdict, "confidence": confidence,
                         "decision": decision, "suggested": suggested, "reasons": reasons, "hard_stops": hard_stops,
                         "reason_code": code, "gate": gate, "off": False}
    tone = {"auto_clear": "good", "auto_confirm": "serious", "needs_review": "warning"}[decision]
    findings = [finding("DOB", dob["detail"], "good" if cls == "exact" else "neutral" if cls in ("conflict", "year_conflict") else "warning"),
                finding("NAME", f"Name similarity {int(ns * 100)}%: " + " ".join(name["facts"]), "good" if ns >= 0.9 else "warning" if ns >= 0.7 else "serious"),
                finding("PLACE", place["detail"], "neutral"),
                finding("VERDICT", f"{verdict.capitalize()} at {confidence}% confidence (gate {gate}); suggested outcome {suggested}.", tone)]
    for h in hard_stops:
        findings.append(finding("STOP", {"sanctions": "Sanctions guard: a person decides.", "fraud_referral": "Fraud referral from intake: a person decides.",
                                         "script_conflict": "Native-script names differ: a person decides."}[h], "critical"))
    return step("adjudication", started, label, confidence=confidence, findings=findings, reasons=reasons,
                needs_human=(decision != "auto_clear"), reason_code=code,
                inputs={"client": (identity.get("name") or {}).get("canonical", case["customer"]["name"]), "client_dob": case["customer"]["dob"],
                        "record": wc["name_raw"], "record_dob": wc["dob"]["raw"], "jurisdiction": wc["location"]["raw"],
                        "category": wc["category"], "vendor_score": wc["score"]},
                verdict=verdict, suggested=suggested, hard_stops=hard_stops)
