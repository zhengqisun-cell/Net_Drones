"""Model table for net_sim.CATCH_NETDYN_MODEL from the fits: per thrust (x1: n = 0..5, x5: n = 0..10)
  "fh" / "fv": formation-centre offset from the reference (ident_fit_ref.py, model_ref_*.json): horizontal 2 modes,
               vertical 1 mode;
  "h" / "v":   net-centre offset from the formation centre (ident_fit.py, model_fit_*.json): horizontal 2 modes for the
               empty net, 1 mode with balls; vertical 1 mode.
python build_model_table.py -> ident/model_table.json"""
import os, json
HERE = os.path.dirname(os.path.abspath(__file__))
ID = os.path.abspath(os.path.join(HERE, "..", "ident"))
fits = {}
for f in ("model_fit_x1.json", "model_fit_x5.json", "model_fit_x5b.json",
          "model_ref_x1.json", "model_ref_x5.json", "model_ref_x5b.json"):
    for k, v in json.load(open(os.path.join(ID, f))).items():
        fits[k.replace("x5b", "x5")] = v


def modes(key, m):
    e = fits[key]["models"][str(m)]
    return [[round(e["omega"][i], 4), round(e["zeta"][i], 4), round(e["g"][i], 5), round(e["d"][i], 5)] for i in range(m)]


tab = {}
for th, ns in (("x1", range(6)), ("x5", range(11))):
    tab[th] = [{"fh": modes(f"{th}_n{n}_form_h", 2), "fv": modes(f"{th}_n{n}_form_v", 1),
                "h": modes(f"{th}_n{n}_horizontal", 2 if n == 0 else 1), "v": modes(f"{th}_n{n}_vertical", 1)}
               for n in ns]
json.dump(tab, open(os.path.join(ID, "model_table.json"), "w"), indent=1)
print(json.dumps(tab)[:300])
