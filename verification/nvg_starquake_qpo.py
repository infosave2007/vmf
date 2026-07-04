#!/usr/bin/env python3
"""
NVG Magnetar Starquake QPO Frequencies — NO INDEPENDENT CONTENT (retracted).

HONESTY NOTE: an earlier version claimed an "avg 0.17% match" to the SGR
1806-20 QPOs. That match was manufactured: the "standard GR shear mode
frequencies" [49, 71, 251, 409, 1703, 5013] Hz are NOT literature values —
they are exactly the OBSERVED QPOs [18, 26, 92, 150, 625, 1840] Hz divided
by the NVG factor sqrt(0.135) = 0.367. Multiplying them back and reporting
the roundoff as agreement is circular by construction.

What survives as genuine (conditional) content: IF eps_eff = 0.135 rescales
crustal shear speeds, NVG predicts a UNIFORM ratio ~0.367 between observed
QPO frequencies and an INDEPENDENTLY computed GR baseline. Until such a
baseline (from a crust shear-mode calculation not derived from these same
data) is plugged in, this script demonstrates internal scaling only and
must not be cited as an observational confirmation.
"""

import math

def calculate_qpo_frequencies(m_omega: float) -> list[float]:
    # eps_eff scales with m_omega: eps_eff = exp(ln(0.135) * m_omega / 859)
    eps_eff_ratio = math.exp(math.log(0.135) * m_omega / 859.0)
    # The crustal shear speed and frequency scale as sqrt(eps_eff)
    factor = math.sqrt(eps_eff_ratio)
    
    # WARNING: these are NOT independent GR values — they equal the observed
    # SGR 1806-20 QPOs divided by sqrt(0.135). See the module docstring.
    standard_frequencies = [49.0, 71.0, 251.0, 409.0, 1703.0, 5013.0]
    
    return [freq * factor for freq in standard_frequencies]

def main():
    print("=" * 80)
    print(" NVG MAGNETAR QPO STARQUAKE FREQUENCY COHERENCE")
    print("=" * 80)
    
    m_omega_center = 859.0
    m_omega_err = 8.0
    
    observed_qpos = [18.0, 26.0, 92.0, 150.0, 625.0, 1840.0]
    modes = ["2t1", "2t2", "10t2", "16t2", "68t2", "n=1 radial"]
    
    pred_center = calculate_qpo_frequencies(m_omega_center)
    pred_lower = calculate_qpo_frequencies(m_omega_center + m_omega_err) # Higher m_omega -> smaller eps_eff -> lower freq
    pred_upper = calculate_qpo_frequencies(m_omega_center - m_omega_err) # Lower m_omega -> larger eps_eff -> higher freq
    
    print(f"QCD Anchor M_Omega_0 : {m_omega_center} +/- {m_omega_err} MeV")
    print(f"NVG Scaling Factor   : {math.sqrt(math.exp(math.log(0.135) * m_omega_center / 859.0)):.4f}")
    print()
    print(f"  {'Mode':<15} | {'Std Freq (Hz)':<15} | {'NVG Pred (Hz)':<20} | {'Observed (Hz)':<15} | {'Error (%)'}")
    print("-" * 78)
    
    errors = []
    for i in range(len(observed_qpos)):
        std = [49.0, 71.0, 251.0, 409.0, 1703.0, 5013.0][i]
        obs = observed_qpos[i]
        pred = pred_center[i]
        lower = pred_lower[i]
        upper = pred_upper[i]
        
        err_pct = abs(pred - obs) / obs * 100.0
        errors.append(err_pct)
        
        pred_range_str = f"{pred:.1f} ({lower:.1f}-{upper:.1f})"
        print(f"  {modes[i]:<15} | {std:<15.1f} | {pred_range_str:<20} | {obs:<15.1f} | {err_pct:>8.2f}%")
        
    avg_error = sum(errors) / len(errors)
    print("-" * 78)
    print(f"Average deviation      : {avg_error:.2f}% — BY CONSTRUCTION (baseline was")
    print(f"                         reverse-engineered from these same observations).")
    print(f"Status                 : ⚠️ NO INDEPENDENT CONTENT — requires an external GR")
    print(f"                         crust shear-mode baseline before any claim can be made.")
    print("=" * 80)

if __name__ == "__main__":
    main()
