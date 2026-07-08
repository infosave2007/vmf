#!/usr/bin/env python3
"""Dose-plan figure: predicted superoxide change vs field, with the
recommended experimental set-points marked. Reuses the radical-pair model."""
import numpy as np, matplotlib, importlib.util, os
matplotlib.use("Agg"); import matplotlib.pyplot as plt
here=os.path.dirname(os.path.abspath(__file__))
spec=importlib.util.spec_from_file_location("m",os.path.join(here,"ciss_rpm_model.py"))
m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
m.A_HF=1.0
Bs=np.unique(np.concatenate([np.linspace(0,2,120),np.linspace(2,10,50)]))
y0=m.yields(0.0,0.35)[0]
dphi=np.array([100*(m.yields(b,0.35)[0]-y0) for b in Bs])
fig,ax=plt.subplots(figsize=(6.6,4.0))
ax.plot(Bs,dphi,'-',color='#1f77b4',lw=2.2)
ax.axhline(0,color='k',lw=0.8)
setpts=[0.0,0.2,0.5,1.0,3.0]
for b in setpts:
    d=100*(m.yields(b,0.35)[0]-y0)
    ax.plot(b,d,'o',color='#d62728',ms=7)
    ax.annotate(f"{b:g} mT",(b,d),textcoords="offset points",
                xytext=(6,6),fontsize=8)
ax.set_xlabel("applied static field  B  [mT]")
ax.set_ylabel(r"predicted change in superoxide  $\Delta\Phi_T$  [%]")
ax.set_title("Dose plan: field set-points to scan")
ax.grid(alpha=0.3); ax.set_xlim(-0.2,10)
fig.tight_layout()
fig.savefig(os.path.join(here,"fig_dose_plan.pdf"))
fig.savefig(os.path.join(here,"fig_dose_plan.png"),dpi=150)
print("set-point predictions (dPhiT %):",
      [round(100*(m.yields(b,0.35)[0]-y0),2) for b in setpts])
print("dose-plan figure written")
