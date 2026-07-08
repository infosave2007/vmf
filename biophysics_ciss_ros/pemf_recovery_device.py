#!/usr/bin/env python3
"""
Design calculation for an experimental weak-field ROS-modulation coil
(Helmholtz pair) sized to the radical-pair low-field effect of the
CISS-ROS model. Honest engineering + the hormesis caveat.

Physics anchor: the flavin-superoxide radical-pair low-field effect peaks
at B_pk ~ hyperfine scale (0.14-0.23 mT here) and relaxes over 0-3 mT
(Usselman et al. measured a real mT-scale ROS shift in cells). So a
research applicator must deliver a TUNABLE, uniform 0-3 mT over a limb.
"""
import math

mu0 = 4e-7*math.pi

print("="*68)
print("  Weak-field ROS-modulation coil — Helmholtz design")
print("="*68)

# ── target field and geometry ──────────────────────────────────────────
R = 0.15          # coil radius [m] (30 cm coils) -> fits a forearm/calf
# Helmholtz: separation = R, B_center = (4/5)^1.5 * mu0 * N*I / R
k_helm = (4/5)**1.5
print(f"\nHelmholtz pair, radius R = {R*100:.0f} cm, spacing = {R*100:.0f} cm")
print(f"Uniform-field region: ~{R*100*0.6:.0f} cm diameter (limb fits)\n")
print(f"  {'B target':>9} {'N*I (A-turns)':>14} {'I at N=100':>11}")
for B in (0.2e-3, 1.0e-3, 3.0e-3):
    NI = B*R/(k_helm*mu0)
    print(f"  {B*1e3:>7.1f} mT {NI:>14.0f} {NI/100:>9.2f} A")

# ── power for a concrete build ─────────────────────────────────────────
N = 100
I_max = 3.0e-3*R/(k_helm*mu0)/N     # current for 3 mT
wire_A = 0.5e-6                      # AWG20, 0.5 mm^2
turn_len = 2*math.pi*R
L_wire = 2*N*turn_len               # both coils
rho_cu = 1.68e-8
Rcoil = rho_cu*L_wire/wire_A
P = I_max**2*Rcoil
print(f"\nConcrete build: N = {N} turns/coil, AWG20 copper")
print(f"  wire length (both coils): {L_wire:.0f} m, resistance {Rcoil:.1f} ohm")
print(f"  current for 3 mT: {I_max:.2f} A -> dissipation {P:.1f} W")
print(f"  runs off a 12 V bench/battery supply with a current driver")

# ── exposure context (safety perspective, not medical advice) ──────────
print(f"\nExposure context (static field magnitude):")
for name, b in (("Earth's field", 0.05e-3), ("this device (max)", 3e-3),
                ("fridge magnet surface", 5e-3), ("MRI scanner", 1.5)):
    print(f"  {name:<22} {b*1e3:>8.2f} mT")
print(f"  -> 3 mT is ~60x Earth, ~500x below a fridge magnet, far below MRI;")
print(f"     static mT fields carry no established acute hazard, but this is")
print(f"     a RESEARCH instrument, not a medical device — see caveats.")

# ── map to the ROS model prediction ────────────────────────────────────
print(f"\nPredicted effect (from the radical-pair model):")
print(f"  tuning B across 0.2->3 mT shifts superoxide partitioning by a few")
print(f"  percent (matching Usselman et al.); the SIGN and best setting are")
print(f"  what the instrument would MEASURE, not assume.")

print(f"""
{'='*68}
  CRITICAL CAVEAT — why 'less ROS' is NOT automatically 'better recovery'
{'='*68}
  Exercise ROS are SIGNALLING molecules, not just damage: the acute
  superoxide/H2O2 burst DRIVES mitochondrial biogenesis, antioxidant
  upregulation and training adaptation (redox hormesis). Randomized
  trials show that blunting exercise ROS with high-dose antioxidants
  (vitamin C/E) can IMPAIR endurance adaptation and insulin sensitivity.
  Therefore:
   - a device that simply suppresses ROS could REDUCE training gains;
   - the plausible useful window is TIMING/MAGNITUDE modulation (e.g.
     trimming a pathological post-exercise excess in injury/overtraining
     while preserving the adaptive signal), which is exactly the open
     question such an instrument is built to TEST;
   - effect sizes here are few-percent and unproven in vivo.
  Honest status: this is an experimental research applicator to MEASURE
  whether weak-field ROS modulation affects recovery markers (e.g. CK,
  soreness, HRV) under proper controls (sham coil, blinding) — not a
  treatment with a guaranteed benefit. Self-experimentation carries the
  usual risks; human protocols need ethics oversight.
""")
print("="*68)
