"""Validation of the planner's model (net_sim.CATCH_NETDYN_MODEL, cascade reference -> formation -> net) on the
identification runs: the model is driven by the logged reference only (open loop, as the planner's "model" state),
started at rest 2 s before the move; per run a constant offset (static net / formation offset at hover) is removed.
Errors of the net-centre and formation-centre trajectories (mm, over the move and the 6 s after it), per thrust and
ball count, horizontal moves (both axes) and vertical / oblique moves (z). Also plots of a few runs.
python ident_validate.py -> ident/validate.json / .txt / validate_*.png"""
import os, sys, json
import numpy as np
from scipy.signal import savgol_filter
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ident_fit as F
REPO = os.path.abspath(os.path.join(F.OUT, "..", ".."))
sys.path.insert(0, REPO)
import net_sim as N

DT = F.DT


def simulate(r, th, n):
    """Model offsets (net, formation) from the reference, per axis, from rest at the window start."""
    rc = r["ref_c"].astype(float)
    rv = savgol_filter(rc, 9, 3, deriv=1, delta=DT, axis=0)
    ra = savgol_filter(rc, 9, 3, deriv=2, delta=DT, axis=0)
    tab = N.CATCH_NETDYN_MODEL[th]
    out_n, out_f = np.zeros_like(rc), np.zeros_like(rc)
    for ax in range(3):
        parts = ("v", "fv") if ax == 2 else ("h", "fh")
        mdl = N.NDModel(N._nd_modes(tab, n, parts[0]), N._nd_modes(tab, n, parts[1]))
        Ph, Gh = mdl.disc(DT)
        s = np.zeros(mdl.ns)
        for i in range(len(rc)):
            out_n[i, ax], out_f[i, ax] = mdl.Cn @ s, mdl.Cf @ s
            s = Ph @ s + Gh @ np.array([ra[i, ax], rv[i, ax]])
    return rc, out_n, out_f


def main():
    rows, lines, res = [], [], {}
    plots = []
    for name in ("x1", "x5", "x5b"):
        th = name[:2]
        for r in F.load(name):
            n = int(r["label"].split("_n")[1].split("_")[0])
            kind = r["label"].split("_")[2][0]
            rc, mn, mf = simulate(r, th, n)
            net = r["net_c"].astype(float) - rc
            form = r["pos"].mean(axis=1).astype(float) - rc
            t = r["t"].astype(float)
            w = t >= r["t_move"] - 1e-9
            pre = (t >= r["t_move"] - 1.0) & ~w
            axes = (0, 1) if kind == "h" else (2,) if kind == "v" else (0, 2)
            for ax in axes:
                en = (net[:, ax] - net[pre, ax].mean()) - mn[:, ax]
                ef = (form[:, ax] - form[pre, ax].mean()) - mf[:, ax]
                rows.append({"thrust": th, "n": n, "part": "z" if ax == 2 else "xy", "label": r["label"], "ax": ax,
                             "net_rms": float(np.sqrt(np.mean(en[w] ** 2))), "net_max": float(np.abs(en[w]).max()),
                             "form_rms": float(np.sqrt(np.mean(ef[w] ** 2))), "form_max": float(np.abs(ef[w]).max()),
                             "net_sig": float(np.sqrt(np.mean((net[w, ax] - net[pre, ax].mean()) ** 2))),
                             "form_sig": float(np.sqrt(np.mean((form[w, ax] - form[pre, ax].mean()) ** 2)))})
            if r["label"] in ("x1_n0_h1_0", "x5_n5_h2_0", "x5_n10_h1_0", "x1_n3_o1_0.4"):
                plots.append((r["label"], t - r["t_move"], net, mn, form, mf, axes[0]))
    for th in ("x1", "x5"):
        for n in range(11):
            for part in ("xy", "z"):
                sel = [x for x in rows if x["thrust"] == th and x["n"] == n and x["part"] == part]
                if not sel:
                    continue
                k = f"{th}_n{n}_{part}"
                q = lambda key: round(1e3 * float(np.sqrt(np.mean([x[key] ** 2 for x in sel]))), 1)
                m_ = lambda key: round(1e3 * max(x[key] for x in sel), 1)
                res[k] = {"runs": len(sel), "net_rms_mm": q("net_rms"), "net_max_mm": m_("net_max"),
                          "net_signal_rms_mm": q("net_sig"), "form_rms_mm": q("form_rms"),
                          "form_max_mm": m_("form_max"), "form_signal_rms_mm": q("form_sig")}
                lines.append(f"{k:12s} runs {len(sel):2d}  net centre: error rms {res[k]['net_rms_mm']:5.1f} max "
                             f"{res[k]['net_max_mm']:5.1f} (offset rms {res[k]['net_signal_rms_mm']:5.1f})   formation: "
                             f"error rms {res[k]['form_rms_mm']:5.1f} max {res[k]['form_max_mm']:5.1f} "
                             f"(offset rms {res[k]['form_signal_rms_mm']:5.1f})")
                print(lines[-1], flush=True)
    note = ("offsets from the reference; error = logged offset - model offset (model driven open loop by the logged "
            "reference from rest 2 s before the move, constant hover offset removed), over the move and the 6 s after it")
    json.dump({"note": note, "groups": res, "runs": rows}, open(os.path.join(F.OUT, "ident", "validate.json"), "w"),
              indent=1)
    open(os.path.join(F.OUT, "ident", "validate.txt"), "w").write(note + "\n" + "\n".join(lines) + "\n")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(len(plots), 2, figsize=(11, 2.6 * len(plots)), squeeze=False)
    for i, (lab, tt, net, mn, form, mf, ax) in enumerate(plots):
        pre = (tt >= -1.0) & (tt < 0)
        for j, (y, ym, nm) in enumerate(((net, mn, "net centre"), (form, mf, "formation centre"))):
            a = axs[i, j]
            a.plot(tt, 1e3 * (y[:, ax] - y[pre, ax].mean()), color="#1f4e79", lw=1.4, label="simulation")
            a.plot(tt, 1e3 * ym[:, ax], color="#c0504d", lw=1.2, ls="--", label="model")
            a.set_title(f"{lab}: {nm} offset from the reference, axis {'xyz'[ax]}", fontsize=9)
            a.set_ylabel("mm")
            a.grid(alpha=0.3)
            a.set_xlim(-0.5, 6)
            if i == 0 and j == 0:
                a.legend(fontsize=8)
    axs[-1, 0].set_xlabel("time from move start (s)")
    axs[-1, 1].set_xlabel("time from move start (s)")
    fig.tight_layout()
    fig.savefig(os.path.join(F.OUT, "ident", "validate_examples.png"), dpi=110)


if __name__ == "__main__":
    main()
