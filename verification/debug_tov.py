import numpy as np
from nvg_full_ns_eos import UnifiedEOS, solve_tov, nvg_core_eos, n_0
eps_t, P_t, _ = nvg_core_eos(1.8 * n_0)
print(f"P_trans at 1.8 n_0: {P_t:.2f} MeV/fm3")
eos = UnifiedEOS(1.8, 800.0)
masses = []
for P in np.logspace(0, 3.5, 30):
    M, R = solve_tov(eos, P)
    masses.append(M)
    print(f"P={P:.1f}, R={R:.2f}, M={M:.3f}")
print("M_max:", max(masses))
