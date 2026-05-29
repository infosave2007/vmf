#!/usr/bin/env python3
"""
NVG Verification: Primordial Gravitational Wave Background Comb.
Calculates the redshifted bounce frequencies for the sequence of N=77 cycles
and checks which cycles fall in the Pulsar Timing Array (PTA) nHz band.
"""

def calculate_comb_frequencies(m_omega: float) -> dict[int, float]:
    # Redshifted bounce frequency scales linearly with m_omega: f_0 = 145 nHz at 859 MeV
    f_0 = 145.0 * (m_omega / 859.0)
    
    # Calculate frequencies for the last 30 cycles (from k=48 to 77)
    frequencies = {}
    for k in range(48, 78):
        # f_GW(k) = f_0 * (3/4)^(77 - k)
        frequencies[k] = f_0 * (0.75 ** (77 - k))
    return frequencies

def main():
    print("=" * 70)
    print(" NVG PRIMORDIAL GRAVITATIONAL WAVE BACKGROUND COMB")
    print("=" * 70)
    
    m_omega_center = 859.0
    m_omega_err = 8.0
    
    freq_center = calculate_comb_frequencies(m_omega_center)
    freq_lower = calculate_comb_frequencies(m_omega_center - m_omega_err)
    freq_upper = calculate_comb_frequencies(m_omega_center + m_omega_err)
    
    # PTA sensitivity band: 1.0 nHz to 1000.0 nHz
    pta_min = 1.0
    pta_max = 1000.0
    
    print(f"QCD Anchor M_Omega_0 : {m_omega_center} +/- {m_omega_err} MeV")
    print(f"Bounce Frequency today: {145.0 * (m_omega_center/859.0):.1f} nHz")
    print()
    print(f"  {'Cycle (k)':<12} | {'NVG Pred Frequency (nHz)':<35} | {'In PTA Band?'}")
    print("-" * 68)
    
    pta_cycles = []
    for k in sorted(freq_center.keys()):
        val = freq_center[k]
        low = freq_lower[k]
        upp = freq_upper[k]
        
        in_pta = pta_min <= val <= pta_max
        if in_pta:
            pta_cycles.append(k)
            
        in_pta_str = "✅ YES" if in_pta else "NO (Too Low)" if val < pta_min else "NO (Too High)"
        range_str = f"{val:.2f} ({low:.2f} - {upp:.2f})"
        print(f"  Cycle {k:<7} | {range_str:<35} | {in_pta_str}")
        
    print("-" * 68)
    print(f"Result: Cycles {pta_cycles[0]} to {pta_cycles[-1]} fall in the Pulsar Timing Array band.")
    print("This forms a discrete, testable frequency comb of bounce signals.")
    print("=" * 70)

if __name__ == "__main__":
    main()
