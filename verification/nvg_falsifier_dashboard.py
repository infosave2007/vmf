#!/usr/bin/env python3
"""Falsifier registry status (process/reporting surface only).

The former dashboard embedded a hand-maintained table of favourable current
values and asserted that every row was alive.  Those values had no executable
producer, so they could drift away from the maintained calculations.  This
module now reports the evidence contract for each falsifier and leaves its
status unassessed until a traceable producer and sourced observation are
connected.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class FalsifierSpec:
    name: str
    producer: str
    experiment: str
    evidence_status: str = "UNASSESSED_NO_LIVE_PRODUCER"


# This is metadata, not a registry of current numerical results.  Keep the
# producer path explicit so a future data release can be connected without
# turning this process check into a scientific success assertion.
REGISTRY = (
    FalsifierSpec("NICER radius comparison", "verification/nvg_tidal_deformability.py", "new NICER radius likelihood"),
    FalsifierSpec("BNS tidal deformability", "verification/nvg_tidal_deformability.py", "new BNS event likelihood"),
    FalsifierSpec("Heavy-ion spectral comparison", "verification/nvg_hades_dielectron_sim.py", "sourced HADES/CBM spectrum"),
    FalsifierSpec("RHIC BES-II Bell protocol", "verification/nvg_bell_contextual.py", "sourced BES-II result"),
    FalsifierSpec("DESI dark-energy test", "verification/nvg_dark_energy_desi.py", "sourced DESI likelihood"),
    FalsifierSpec("Neutron portal search", "verification/nvg_adm_bl_cogenesis.py", "sourced neutron-decay limit"),
    FalsifierSpec("Stochastic GW spectrum", "verification/nvg_gw_spectrum_template.py", "sourced detector search"),
    FalsifierSpec("Weak-lensing S8 comparison", "verification/nvg_black_hole_entropy.py", "sourced weak-lensing likelihood"),
)


def evaluate_registry() -> list[dict[str, str]]:
    """Return structured evidence statuses without manufacturing results."""

    return [asdict(spec) for spec in REGISTRY]


def main() -> int:
    print("=" * 78)
    print("  NVG FALSIFIER REGISTRY (NO LIVE SCIENTIFIC VERDICT)")
    print("=" * 78)
    print("Each row is unassessed until its producer and a sourced observation are connected.")
    print()

    for row in evaluate_registry():
        print(f"  [{row['evidence_status']}] {row['name']}")
        print(f"      producer: {row['producer']}")
        print(f"      next evidence: {row['experiment']}")

    print()
    print("Registry process completed; no current-value or all-alive claim is emitted.")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
