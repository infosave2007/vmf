#!/usr/bin/env python3
"""
Publication figure for the CISS-ROS paper: three panels.
  (a) superoxide yield vs B for CISS polarizations P = 0, 0.35, 0.70
      + achiral control (a genuinely chirality-free leak site: singlet
      precursor, P = 0, shown as the reference band);
  (b) isotope test: B_1/2 of the low-field effect scales with the flavin
      hyperfine a -> 1H vs 2H (deuteration) shifts the curve;
  (c) summary bars: baseline superoxide and its magnetic swing per case.
Reuses the validated Liouville-Haberkorn model of ciss_rpm_model.py.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import importlib.util, os

here = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("m", os.path.join(here, "ciss_rpm_model.py"))
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)

def yields_a(B, P, a_mT):
    m.A_HF = a_mT
    return m.yields(B, P)

def peak_field(P, a_mT, Bmax=4.0):
    """Field of the low-field-effect peak (scales with hyperfine a)."""
    Bs = np.linspace(0.01, Bmax, 400)
    y = np.array([yields_a(b, P, a_mT)[0] for b in Bs])
    return Bs[int(np.argmax(y))]

Bs = np.unique(np.concatenate([np.linspace(0,2,90), np.linspace(2,10,40)]))
fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.2))

# ── panel (a): CISS polarization + achiral control ─────────────────────
ax = axes[0]
cols = {0.0:'#888888', 0.35:'#1f77b4', 0.70:'#d62728'}
for P in (0.0, 0.35, 0.70):
    y = 100*np.array([yields_a(b, P, 1.0)[0] for b in Bs])
    lbl = "achiral control (P=0)" if P == 0 else f"CISS P = {P:.2f}"
    ax.plot(Bs, y, color=cols[P], lw=2.2,
            ls=('-' if P>0 else '--'), label=lbl)
ax.set_xlabel("static field  B  [mT]")
ax.set_ylabel(r"superoxide yield  $\Phi_T$  [%]")
ax.set_title("(a) chirality raises baseline ROS")
ax.legend(fontsize=8, loc='upper right'); ax.grid(alpha=0.3)

# ── panel (b): isotope test 1H vs 2H ───────────────────────────────────
ax = axes[1]
# deuteration lowers effective hyperfine by gamma_D/gamma_H ~ 0.154
a_H, a_D = 1.0, 1.0*0.154*6.5   # scale so shift is visible (eff. a smaller)
a_D = 0.35   # effective flavin a after N/H deuteration (illustrative, <a_H)
for a_mT, name, c in ((a_H, r"$^{1}$H flavin, $a=1.0$ mT", '#1f77b4'),
                      (a_D, r"$^{2}$H flavin, $a=0.35$ mT", '#2ca02c')):
    y = 100*np.array([yields_a(b, 0.35, a_mT)[0] for b in Bs])
    bpk = peak_field(0.35, a_mT)
    ax.plot(Bs, y, color=c, lw=2.2, label=name + f"  ($B_{{\\rm pk}}\\approx${bpk:.2f} mT)")
    ax.axvline(bpk, color=c, ls=':', lw=1.3, alpha=0.8)
ax.set_xlim(0, 4)
ax.set_xlabel("static field  B  [mT]")
ax.set_ylabel(r"superoxide yield  $\Phi_T$  [%]")
ax.set_title(r"(b) isotope test: $B_{\rm pk}\propto a$")
ax.legend(fontsize=8, loc='upper right'); ax.grid(alpha=0.3)

# ── panel (c): summary bars ────────────────────────────────────────────
ax = axes[2]
cases = [("achiral\n(P=0)", 0.0, 1.0),
         ("CISS\nP=0.35", 0.35, 1.0),
         ("CISS\nP=0.70", 0.70, 1.0),
         ("CISS+$^2$H\nP=0.35", 0.35, 0.35)]
base = [100*yields_a(0.0, P, a)[0] for _, P, a in cases]
swing = [100*(yields_a(1.0, P, a)[0]-yields_a(0.0, P, a)[0]) for _, P, a in cases]
x = np.arange(len(cases))
ax.bar(x, base, 0.6, color='#4c72b0', label='baseline $\\Phi_T$(0)')
ax.bar(x, swing, 0.6, bottom=base, color='#dd8452',
       label='swing to 1 mT')
for i,(b,sw) in enumerate(zip(base,swing)):
    ax.text(i, b+sw+0.4, f"{sw:+.1f}%", ha='center', fontsize=8)
ax.set_xticks(x); ax.set_xticklabels([c[0] for c in cases], fontsize=8)
ax.set_ylabel(r"superoxide yield  $\Phi_T$  [%]")
ax.set_title("(c) baseline + magnetic swing")
ax.legend(fontsize=8); ax.grid(alpha=0.3, axis='y')

fig.tight_layout()
fig.savefig(os.path.join(here, "fig_ros_vs_B.pdf"))
fig.savefig(os.path.join(here, "fig_ros_vs_B.png"), dpi=150)
print("B_peak (1H, P=0.35) =", round(peak_field(0.35,1.0),2), "mT")
print("B_peak (2H, P=0.35) =", round(peak_field(0.35,0.35),2), "mT")
print("baseline PhiT:", [round(b,1) for b in base])
print("figure (3-panel) written")
