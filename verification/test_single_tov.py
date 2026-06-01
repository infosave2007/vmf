import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import solve_ivp

conv_MeV_fm3_to_geo = 1.3234e-6
M_sun_km = 1.4766

# Simple test EOS similar to NL3
eps_arr = np.logspace(-3.0, 3.5, 500)
p_arr = 0.0005 * (eps_arr ** 2.1)

eps_of_p = interp1d(p_arr, eps_arr, bounds_error=False, fill_value=(eps_arr[0], eps_arr[-1]))

p_c = 10.0
r0 = 1.0e-3
e_c = float(eps_of_p(p_c))
m0 = 4.0 * np.pi * r0**3 * e_c * conv_MeV_fm3_to_geo / 3.0

def rhs(radius, state):
    mass, pressure = state
    energy = float(eps_of_p(pressure))
    denom = radius * (radius - 2.0 * mass)
    if denom <= 0.0:
        return [0.0, 0.0]
        
    dmdr = 4.0 * np.pi * radius**2 * energy * conv_MeV_fm3_to_geo
    dpdr = -(energy + pressure) * (mass + 4.0 * np.pi * radius**3 * pressure * conv_MeV_fm3_to_geo) / denom
    return [dmdr, dpdr]

def stop_condition(radius, state):
    return state[1]
stop_condition.terminal = True
stop_condition.direction = -1

sol = solve_ivp(rhs, [r0, 30.0], [m0, p_c], events=stop_condition, max_step=0.25, rtol=1.0e-5, atol=1.0e-8)

for i in range(len(sol.t)):
    print(f"r={sol.t[i]:.2f}, m={sol.y[0, i]:.4f}, p={sol.y[1, i]:.4f}")

print("Status:", sol.status)
print("Message:", sol.message)
print("t_events:", sol.t_events)
print("y_events:", sol.y_events)
print("Last t:", sol.t[-1])
print("Last y:", sol.y[:, -1])
