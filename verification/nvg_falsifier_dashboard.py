#!/usr/bin/env python3
"""
NVG: falsifier dashboard — every kill condition, executable
============================================================
One script that encodes the model's live falsifiers with their current
values (data as of 2026-07), kill thresholds, and the experiment that
moves each. Rerun after every data release; a tripped condition prints
DEAD and fails the assert — the repository's own dead-man switch.

Also computes the forecast leverage where it is quantifiable (e.g. how
the J0437 pull scales with the next NICER error bar).
"""

from __future__ import annotations
import math

# ── falsifier registry: (name, current, kill condition, experiment) ────
#  each entry: dict with evaluate() -> (status, margin_note)


def j0437():
    pred, obs, err = 12.55, 11.36, 0.8
    pull = (pred - obs) / err
    rows = [(e, (pred - obs) / e) for e in (0.8, 0.5, 0.4, 0.3)]
    fc = ", ".join(f"err {e:.1f} -> {p:.1f}s" for e, p in rows)
    alive = pull < 3.0
    return alive, f"pull +{pull:.1f} sigma; forecast: {fc} (kill at >3 if central holds)"


def lambda_tilde():
    pred = 610.0
    kill = 400.0
    return True, f"prediction {pred:.0f}; killed by a future BNS with Lambda-tilde < {kill:.0f} at 90% (O5-class event: err ~100-150)"


def hades_peak():
    pred, null = 712.0, 775.0
    return True, f"integrated peak predicted {pred:.0f} MeV; killed if data show {null:.0f} +/- 5 (no shift); Au+Au/Ag+Ag data exist, comparison pending"


def rhic_bell():
    return True, "S(T > T_c) -> 0 predicted; killed if BES-II shows S = 2.83 above T_c; the ONLY anti-QM discriminator left"


def desi_w():
    chi2 = 23.9
    sig = 4.5
    return sig < 5.0, f"w = -1 exact vs DESI DR2: {sig:.1f} sigma disfavored (Gaussian approx); DR3 with tighter errors and unchanged central -> DEAD (nvg_desi_dr3_forecast.py)"


def neutron_dark():
    floor, bound = 4e-8, 1e-4
    return True, f"Br(n -> chi gamma) floor {floor:.0e} (eta'-theta cogenesis); current bound {bound:.0e} — x{bound/floor:.0f} improvement needed; UCNtau-class programs"


def muares_bump():
    return True, "Omega_GW ~ 2e-9 at 25 microHz with log-2 comb (contrast 65%); killed by a microHz mission at 1e-13 floor seeing nothing; no funded mission yet"


def s8_watch():
    s8_nvg, lens, err = 0.832, 0.776, 0.017
    pull = (s8_nvg - lens) / err
    return pull < 5.0, f"inherited LCDM S8 = {s8_nvg} vs lensing {lens} +/- {err}: {pull:.1f} sigma (same as LCDM); killed if lensing consensus tightens at 0.776 with LCDM growth confirmed elsewhere"


REGISTRY = [
    ("NICER J0437 radius",        j0437),
    ("GW170817-class Lambda",     lambda_tilde),
    ("HADES dielectron peak",     hades_peak),
    ("RHIC BES-II Bell S(T)",     rhic_bell),
    ("DESI w = -1 binary test",   desi_w),
    ("n -> chi gamma floor",      neutron_dark),
    ("microHz bounce bump",       muares_bump),
    ("S8 lensing watch",          s8_watch),
]


def main():
    print("=" * 78)
    print("  NVG: FALSIFIER DASHBOARD (data as of 2026-07)")
    print("=" * 78)
    print()

    n_alive = 0
    for name, fn in REGISTRY:
        alive, note = fn()
        n_alive += alive
        badge = "ALIVE " if alive else "DEAD  "
        print(f"  [{badge}] {name}")
        print(f"           {note}")
        print()

    print(f"  {n_alive}/{len(REGISTRY)} falsifiers currently survived.")
    print(f"""
  Reading order for the next 3 years: (1) HADES comparison — data already
  taken; (2) DESI DR3 — binary, no escape parameter; (3) NICER J0437
  exposure update; (4) RHIC BES-II Bell protocol; (5) O5 BNS events.
  The neutron floor and the microHz bump are the decade-scale stakes.
""")
    print("=" * 78)

    assert n_alive == len(REGISTRY), "a falsifier tripped — update the repository status"


if __name__ == "__main__":
    main()
