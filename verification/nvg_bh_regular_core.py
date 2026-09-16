#!/usr/bin/env python3
"""Compatibility entry point for a selected regular metric, not NVG collapse."""
from __future__ import annotations
import json
import math
from scipy.constants import G, c, electron_volt
import regular_core_geometry_audit as geometry


def calculate_bh_interior(m_tot_solar, eps_max_mev_fm3):
    """Former argument name eps_max now denotes an ASSUMED central density.

    No maximum of the matter EOS is inferred. Returned r0 is the Hayward
    crossover (2 M l^2)^(1/3), not an event horizon or a junction surface.
    """
    mass_solar=float(m_tot_solar)
    energy=float(eps_max_mev_fm3)
    if not math.isfinite(mass_solar) or mass_solar<=0 or not math.isfinite(energy) or energy<=0:
        raise ValueError("positive finite mass and assumed central density required")
    solar_mass_kg=1.989e30  # declared conventional input
    mass=G*mass_solar*solar_mass_kg/c**2/1000  # geometrized km
    energy_geom=energy*(1e6*electron_volt/1e-45)*G/c**4*1e6  # km^-2
    length=math.sqrt(3/(8*math.pi*energy_geom))
    scale=(2*mass*length**2)**(1/3)
    return {
        "evidence_status":geometry.EVIDENCE_STATUS,
        "observed_likelihood":None, "assumed_mass_solar":mass_solar,
        "assumed_central_density_MeV_fm3":energy,
        "length_km":length, "crossover_radius_km":scale,
        "central_curvature":geometry.curvature_invariants(0,mass,length),
        "central_stress_geometric_units":geometry.stress_tensor(0,mass,length),
        "crossover_stress_geometric_units":geometry.stress_tensor(scale,mass,length),
        "limitations":["Selected metric, not a matter EOS density bound",
                       "No collapse, stable interior, evaporation, or cosmological bounce derivation"]}


def main():
    result=calculate_bh_interior(10,859**4/197.3269804**3)
    print(json.dumps(result,indent=2,allow_nan=False))
    return result


if __name__=="__main__":
    main()
