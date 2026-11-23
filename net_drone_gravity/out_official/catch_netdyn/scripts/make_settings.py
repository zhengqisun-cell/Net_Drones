"""Settings files (--catch-limits) of the selected candidates and of the net-dynamics planner per family:
python make_settings.py -> settings/<family>_<candidate>.json (the family's original file + the candidate's overrides;
the original files are not changed)."""
import os, json, copy
import nd_runs as R

SELECTED = {"x1_T0": "ff_ah+", "x5_T0": "ff", "x5_T1": "ff_ah+", "x5_T0_10": "nd", "x5_T1_10": "ff_ah+"}
THRUST_NOTE = {1: "", 5: " -- use with --thrust-scale 5"}
DESC = {"ff": "feedforward only (netdyn plan off): drone and net air drag at the reference velocity, 70 ms "
              "acceleration lead",
        "ff_ah+": "feedforward only (netdyn plan off, drag + 70 ms lead) and the horizontal / total acceleration limits "
                  "+0.5 m/s^2",
        "nd": "net-dynamics planner (netdyn: net model from catch_netdyn/ident, model state, formation = reference) "
              "with the drag and lead feedforward"}


def main():
    out = os.path.join(R.OUT, "settings")
    os.makedirs(out, exist_ok=True)
    made = []
    for fam, (th, tier, path, _) in R.FAMILIES.items():
        for cand in sorted({SELECTED[fam], "nd"}):
            d = json.load(open(path))
            ov = copy.deepcopy(R.candidates(fam)[cand])
            for k, v in ov.items():
                if k == "limits":
                    d["limits"] = {**d.get("limits", {}), **v}
                else:
                    d[k] = v
            sel = "selected on seeds 101-130 (catch_netdyn/selection_*)" if cand == SELECTED[fam] else \
                "net-dynamics planner, not selected (for comparison)"
            d["note"] = (f"catch_netdyn, {fam} (tier {tier}), {sel}: {DESC[cand]}; based on "
                         f"{os.path.relpath(path, R.OO).replace(os.sep, '/')}{THRUST_NOTE[th]}")
            fn = os.path.join(out, f"{fam}_{cand.replace('+', 'plus')}.json")
            json.dump(d, open(fn, "w"), indent=1)
            made.append(os.path.basename(fn))
    print(made)


if __name__ == "__main__":
    main()
