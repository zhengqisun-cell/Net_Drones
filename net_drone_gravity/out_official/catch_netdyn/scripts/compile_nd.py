"""Tables of the net-dynamics catch batches.
python compile_nd.py select <tune_tag>                     -> selection_<tag>.txt / .json (rule fixed beforehand: among
       the candidates with at most 6 of 120 trials breaking the family's tier lines, the one with the most balls caught)
python compile_nd.py table <tag> [ref_candidate] [name]    -> table_<name>.txt / .json (per family, candidate, group:
       caught / launched, all caught, >= 8 caught, tier-violation trials and items, aborts, miss reasons, seeds whose
       catches differ from ref_candidate)"""
import sys, os, json
from collections import Counter
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.abspath(os.path.join(HERE, ".."))
MAX_VIOL = 6


def load(tag):
    return json.load(open(os.path.join(OUT, "runs", f"{tag}.json")))["results"]


def gstats(ts):
    ok = [t for t in ts if "error" not in t]
    items = Counter(x for t in ok for x in t["tier_viol"])
    reasons = Counter(b["reason"] for t in ok for b in t["balls"] if b["outcome"] not in ("caught", "not_launched"))
    return {"trials": len(ok), "errors": len(ts) - len(ok), "caught": sum(t["n_caught"] for t in ok),
            "launched": sum(t["n_launched"] for t in ok),
            "all_caught": sum(t["n_caught"] == t["n_launched"] for t in ok),
            "ge8": sum(t["n_caught"] >= 8 for t in ok),
            "viol": sum(bool(t["tier_viol"]) for t in ok), "items": dict(items),
            "aborts": sum(t["aborted"] for t in ok), "miss_reasons": dict(reasons),
            "disc_min_mm": min((t["disc_min_mm"] for t in ok), default=None),
            "guard_min_mm": min((t["guard_min_mm"] for t in ok), default=None),
            "motor_max": max((t["motor_max"] for t in ok), default=None),
            "tilt_max_deg": max((t["tilt_max_deg"] for t in ok), default=None),
            "netdyn_fallback": sum((t.get("netdyn") or {}).get("fallback", 0) for t in ok),
            "netdyn_go": sum((t.get("netdyn") or {}).get("go", 0) for t in ok)}


def total(gr):
    ts = [t for g in gr.values() for t in g]
    return gstats(ts)


def fmt(name, s):
    return (f"{name:12s} {s['caught']:4d}/{s['launched']:<4d} all {s['all_caught']:3d} >=8 {s['ge8']:3d} "
            f"viol {s['viol']:3d}/{s['trials']} {s['items']} abort {s['aborts']} disc {s['disc_min_mm']} guard "
            f"{s['guard_min_mm']} motor {s['motor_max']} tilt {s['tilt_max_deg']} miss {s['miss_reasons']}"
            + (f" errors {s['errors']}" if s["errors"] else ""))


def select(tag):
    R = load(tag)
    out, L = {}, [f"selection on {tag}: among candidates with <= {MAX_VIOL} trials breaking the tier lines, most caught"]
    for fam, cc in R.items():
        tot = {c: total(gr) for c, gr in cc.items()}
        okc = [c for c, s in tot.items() if s["viol"] <= MAX_VIOL and s["errors"] == 0]
        best = max(okc, key=lambda c: (tot[c]["caught"], -tot[c]["viol"])) if okc else None
        out[fam] = {"selected": best, "totals": tot}
        L.append(f"== {fam}: selected {best}")
        for c, s in sorted(tot.items(), key=lambda kv: -kv[1]["caught"]):
            L.append("  " + fmt(c, s) + ("   <- selected" if c == best else "")
                     + ("   (over the violation limit)" if s["viol"] > MAX_VIOL else ""))
    open(os.path.join(OUT, f"selection_{tag}.txt"), "w").write("\n".join(L) + "\n")
    json.dump(out, open(os.path.join(OUT, f"selection_{tag}.json"), "w"), indent=1)
    print("\n".join(L))


def table(tag, ref=None, name=None):
    R = load(tag)
    name = name or tag
    out, L = {}, [f"catch table {tag} (reference for changed seeds: {ref})"]
    for fam, cc in R.items():
        L.append(f"== {fam}")
        for c, gr in cc.items():
            out.setdefault(fam, {})[c] = {"total": total(gr), "groups": {}}
            L.append("  " + fmt(c, total(gr)))
            for g, ts in gr.items():
                s = gstats(ts)
                if ref and ref in cc and c != ref and g in cc[ref]:
                    rm = {t["seed"]: t["n_caught"] for t in cc[ref][g] if "error" not in t}
                    s["changed_caught"] = [(t["seed"], rm[t["seed"]], t["n_caught"]) for t in ts
                                           if "error" not in t and t["seed"] in rm and rm[t["seed"]] != t["n_caught"]]
                out[fam][c]["groups"][g] = s
                L.append("      " + fmt(g, s) + (f"  changed vs {ref}: {s['changed_caught']}" if "changed_caught" in s
                                                  else ""))
    open(os.path.join(OUT, f"table_{name}.txt"), "w").write("\n".join(L) + "\n")
    json.dump(out, open(os.path.join(OUT, f"table_{name}.json"), "w"), indent=1)
    print("\n".join(L))


if __name__ == "__main__":
    if sys.argv[1] == "select":
        select(sys.argv[2])
    else:
        table(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None, sys.argv[4] if len(sys.argv) > 4 else None)
