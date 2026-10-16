"""Identity Normalisation Agent: the legal name, the AKAs and the address,
with every rule that fired and the variance type it maps to."""
import time

from fincrime.hub import addresses, names
from fincrime.hub.agents import finding, skipped, step

ADDRESS_LABELS = {   # address flags → the playbook's variance labels, by country
    "po_box": "PO Box vs residential address", "military": "Military APO/FPO address",
    "dl_address_stale": "No address on passport – stale DL address", "usps_normalised": "USPS abbreviation / unit format",
    "zip4_trimmed": "ZIP vs ZIP+4 / ZIP error", "zip_mismatch": "ZIP vs ZIP+4 / ZIP error",
    "neighbourhood_alias": "Neighbourhood vs USPS city name", "legacy_city": "Legacy city name",
    "co_prefix_stripped": "Aadhaar C/O & landmark format", "landmark_stripped": "Aadhaar C/O & landmark format",
    "pin_mismatch": "PIN code mismatch", "passport_address_outdated": "Outdated address on passport",
    "unit_format": "Unit / flat number format", "district_alias": "District vs locality naming",
    "pre2025_unit": "Pre-2025 administrative address", "permanent_vs_temporary": "Permanent vs temporary residence",
    "vn_abbreviation": "Diacritics / abbreviation / translation", "alley_format": "Alley (hẻm/ngõ) numbering format",
}


def _address_labels(case, address):
    labels = []
    flags = set(address["flags"])
    for flag, label in ADDRESS_LABELS.items():
        if flag in flags and label not in labels:
            labels.append(label)
    if "overseas_residence" in flags:
        labels.append("NRI – overseas vs Indian address" if case["country"] == "IN" else "Overseas residence vs CCCD address")
    if "address_conflict" in flags and case["country"] == "US":
        labels.append("Dual / seasonal residence")
    if any("Thu Duc" in c for c in address["changes"]) or any("Quan 2" in c or "'2'" in c for c in address["changes"]):
        labels.append("Renamed administrative unit")
    return labels


def run(case, settings):
    started = time.perf_counter()
    if not settings["agents"].get("identity", True):
        # Without the agent the CRM record is screened as keyed: the last
        # token is taken for the surname and nothing is repaired.
        crm = case["customer"]["ubo"] if case["customer"]["corporate"] and case["customer"]["ubo"] else case["customer"]["name"]
        toks = crm.split()
        pseudo = {"surname": toks[-1] if toks else "", "given": " ".join(toks[:-1]), "blank_surname": not toks, "raw": crm}
        name = names.analyse(case["country"], crm, pseudo, {"line1": "", "truncated": False}, "", False, "")
        name["changes"], name["variance_detected"], name["aka"] = [], [], []
        address = addresses.analyse(case)
        case["identity"] = {"name": name, "address": address, "variance_detected": [], "off": True}
        return skipped("identity", started)
    pp = case["documents"]["passport"]
    name = names.analyse(case["country"], case["customer"]["name"], pp["viz"], pp["mrz"], case["documents"]["secondary"]["name"],
                         case["customer"]["corporate"], case["customer"]["ubo"])
    address = addresses.analyse(case)
    labels = list(name["variance_detected"]) + _address_labels(case, address)
    case["identity"] = {"name": name, "address": address, "variance_detected": labels, "off": False}
    findings = []
    for c in name["changes"]:
        arrow = f"{c['from']} → {c['to']}" if c["from"] and c["to"] else (c["to"] or c["from"])
        findings.append(finding("NAME", f"{c['rule']}: {arrow}" + (f" ({c['note']})" if c.get("note") else ""), "neutral"))
    if "encoding_unrepairable" in name["flags"]:
        findings.append(finding("NAME", "CRM name bytes were lost in a bad encoding; the passport spelling is used.", "warning"))
    if "surname_conflict" in name["flags"]:
        findings.append(finding("NAME", "More than one surname across the documents; every surname is screened.", "warning"))
    for n in address["notes"]:
        tone = "serious" if ("conflict" in n.lower() or "nothing on file" in n.lower() or "PO Box" in n) else "neutral"
        findings.append(finding("ADDR", n, tone))
    if not findings:
        findings.append(finding("OK", "CRM record, passport and secondary ID agree; nothing to repair.", "good"))
    reasons = [f"Legal name screened: {name['canonical']}" + (f" (AKA {', '.join(a['name'] for a in name['aka'][:3])})" if name["aka"] else "")]
    reasons.append("Current address: " + (address["current"]["display"] or "not evidenced") +
                   (" — reconciled across documents" if address["resolved"] else " — not reconciled"))
    confidence = 98 if not name["flags"] else 90 if "encoding_unrepairable" not in name["flags"] else 80
    decision = "Normalised" if (name["changes"] or address["changes"]) else "Consistent"
    return step("identity", started, decision, confidence=confidence, findings=findings, reasons=reasons,
                needs_human=False, reason_code="", inputs={"crm_name": case["customer"]["name"], "viz": pp["viz"]["raw"],
                                                           "mrz": pp["mrz"]["line1"], "secondary": case["documents"]["secondary"]["name"]},
                variance_detected=labels)
