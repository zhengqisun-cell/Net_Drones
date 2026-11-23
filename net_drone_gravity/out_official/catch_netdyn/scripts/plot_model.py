"""Identified model parameters vs ball count: python plot_model.py -> ident/model_params.png"""
import os, json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__))
ID = os.path.abspath(os.path.join(HERE, "..", "ident"))
tab = json.load(open(os.path.join(ID, "model_table.json")))
val = json.load(open(os.path.join(ID, "validate.json")))["groups"]
fig, ax = plt.subplots(1, 3, figsize=(14, 3.8))
col = {"x1": "#1f4e79", "x5": "#c0504d"}
for th in ("x1", "x5"):
    ns = range(len(tab[th]))
    w = [tab[th][n]["h"][-1][0] if n == 0 else tab[th][n]["h"][0][0] for n in ns]
    z = [tab[th][n]["h"][-1][1] if n == 0 else tab[th][n]["h"][0][1] for n in ns]
    ax[0].plot(ns, w, "o-", color=col[th], label=f"{th} net swing (horizontal)")
    ax[0].plot(ns, [tab[th][n]["fh"][1][0] for n in ns], "s--", color=col[th], alpha=0.6,
               label=f"{th} formation fast mode")
    ax[1].plot(ns, z, "o-", color=col[th], label=f"{th} net swing")
    ax[2].plot(ns, [val[f"{th}_n{n}_xy"]["net_rms_mm"] for n in ns], "o-", color=col[th], label=f"{th} net centre")
    ax[2].plot(ns, [val[f"{th}_n{n}_xy"]["form_rms_mm"] for n in ns], "s--", color=col[th], alpha=0.6,
               label=f"{th} formation centre")
ax[0].set_ylabel("rad/s")
ax[0].set_title("natural frequency (empty net: second mode, 9.6 rad/s, zeta 0.04, not shown)", fontsize=8)
ax[1].set_ylabel("zeta")
ax[1].set_title("damping ratio of the net swing", fontsize=9)
ax[2].set_ylabel("mm rms")
ax[2].set_title("open-loop model error (horizontal, cascade model)", fontsize=9)
for a in ax:
    a.set_xlabel("balls in the net")
    a.grid(alpha=0.3)
    a.legend(fontsize=7)
fig.tight_layout()
fig.savefig(os.path.join(ID, "model_params.png"), dpi=110)
