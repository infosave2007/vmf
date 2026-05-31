import numpy as np
from scipy.integrate import odeint

# Cosmological constants today
Omega_m0 = 0.315
Omega_W0 = 1.0 - Omega_m0
beta = 0.12

# To match W field density today, we need 0.5 * g0^2 * W^2 + V(W) + 0.5 * W_dot^2 = Omega_W0
# Let's assume V(W) is constant V0 for simplicity in the late universe (or just 0.5 g0^2 W^2).
# Actually, if V(W) is a Mexican hat, it sits near the minimum. 
# For quintessence, let's just use V(W) = V0, and ignore g0^2 W^2 for a moment, or use a linear potential.
# If we just have V(W) = V0, it's a cosmological constant.
# Let's use a simple potential: V(W) = V0 + 0.5 * m^2 * W^2
# For it to roll, m^2 must be small (tachyonic or just light).
# What if we use exactly V(W) = V0, but with the source term -Q/W_dot?
# That would mean the field is driven purely by the matter melting!

def field_system(y, a, V0, m2, beta):
    rho_m, W, W_dot_a = y  # W_dot_a is dW/da
    
    # We need to compute H to convert d/dt to d/da.
    # W_dot = dW/dt = a * H * dW/da
    # K = 0.5 * W_dot^2 = 0.5 * a^2 * H^2 * W_dot_a^2
    # P_W = 0.5 * g0^2 * W^2 + V0
    
    # It's an implicit equation for H:
    # H^2 = rho_m + 0.5 * a^2 * H^2 * W_dot_a^2 + 0.5 * m2 * W^2 + V0
    # H^2 (1 - 0.5 * a^2 * W_dot_a^2) = rho_m + 0.5 * m2 * W^2 + V0
    num = rho_m + 0.5 * m2 * W^2 + V0
    den = 1.0 - 0.5 * a**2 * W_dot_a**2
    if den <= 0 or num <= 0:
        return [0, 0, 0]
    H2 = num / den
    H = np.sqrt(H2)
    
    W_dot = a * H * W_dot_a
    rho_W = 0.5 * W_dot**2 + 0.5 * m2 * W^2 + V0
    
    Q = beta * H * rho_m
    
    # d(rho_m)/da
    drho_m_da = Q / (a * H) - 3.0 * rho_m / a
    
    # W EOM: W_ddot + 3H W_dot + m2 W = - Q / W_dot
    # W_ddot = d(W_dot)/dt = a * H * d(W_dot)/da
    # d(W_dot)/da = - (3H W_dot + m2 W + Q / W_dot) / (a * H)
    # But we track W_dot_a = W_dot / (aH). 
    # Let's just track W_dot directly as the 3rd variable for simplicity, but W_dot requires knowing a.
    pass

# Better to integrate in time t! Or use N = ln(a)
# dN = H dt => d/dt = H d/dN
def sys_N(y, N, V0, m2, beta):
    rho_m, W, W_prime = y  # prime = d/dN
    
    # H^2 = rho_m + 0.5 * H^2 * W_prime^2 + 0.5 * m2 * W^2 + V0
    num = rho_m + 0.5 * m2 * W^2 + V0
    den = 1.0 - 0.5 * W_prime**2
    if den <= 0 or num <= 0:
        # Field rolls too fast, K > rho_W!
        return [0, 0, 0]
    H = np.sqrt(num / den)
    
    W_dot = H * W_prime
    
    Q = beta * H * rho_m
    
    # drho_m / dN = Q/H - 3 rho_m = (beta - 3) rho_m
    drho_m_dN = (beta - 3.0) * rho_m
    
    # W_ddot = H * d(H W_prime)/dN = H * (H' W_prime + H W_prime')
    # W_ddot + 3H W_dot + m2 W + V'(W) = - Q / W_dot
    # H (H' W_prime + H W_prime') + 3H^2 W_prime + m2 W = - beta * H * rho_m / (H W_prime)
    # (H'/H) W_prime + W_prime_prime + 3 W_prime + m2 W / H^2 = - beta * rho_m / (H^2 W_prime)
    
    # We need H'/H.
    # H^2 = rho_tot => 2H H' = d(rho_tot)/dt = H d(rho_tot)/dN
    # d(rho_tot)/dN = -3 rho_m - 3 rho_W - 3 P_W = -3 rho_m - 3 (W_dot^2) = -3 rho_m - 3 H^2 W_prime^2
    # So 2H H' = H (-3 rho_m - 3 H^2 W_prime^2)
    # H'/H = -1.5 * (rho_m / H^2 + W_prime^2)
    
    H_prime_over_H = -1.5 * (rho_m / H**2 + W_prime**2)
    
    # Now solve for W_prime_prime:
    # W_prime_prime = - (H'/H) W_prime - 3 W_prime - m2 W / H^2 - beta * rho_m / (H^2 W_prime)
    W_prime_prime = - H_prime_over_H * W_prime - 3.0 * W_prime - (m2 * W) / H**2
    
    if abs(W_prime) > 1e-10:
        W_prime_prime -= beta * rho_m / (H**2 * W_prime)
    else:
        # If W_prime is 0, the source -Q/W_dot is singular!
        # The physical meaning is the field is instantly accelerated by the mass melting.
        # W_dot ~ sqrt(Q t) -> we must initialize with W_prime != 0
        pass
        
    dW_dN = W_prime
    
    return [drho_m_dN, dW_dN, W_prime_prime]

N_vals = np.linspace(0, -np.log(1.0/0.4), 1000)
# Init
W0 = 1.0
m2 = 0.0  # Massless field driven purely by melting!
V0 = Omega_W0  # Cosmological constant base
# We need W_prime(0) > 0 so that W_prime doesn't diverge. 
W_p0 = -0.05
y0 = [Omega_m0, W0, W_p0]

sol = odeint(sys_N, y0, N_vals, args=(V0, m2, beta))

rho_m_vals = sol[:, 0]
W_vals = sol[:, 1]
W_p_vals = sol[:, 2]

a_vals = np.exp(N_vals)
H2_vals = (rho_m_vals + 0.5 * m2 * W_vals**2 + V0) / (1.0 - 0.5 * W_p_vals**2)
rho_W_vals = 0.5 * H2_vals * W_p_vals**2 + 0.5 * m2 * W_vals**2 + V0
K_vals = 0.5 * H2_vals * W_p_vals**2
P_W_vals = K_vals - (0.5 * m2 * W_vals**2 + V0)
w_W_vals = P_W_vals / rho_W_vals

# Effective w(a) for CPL
rho_tot = rho_m_vals + rho_W_vals
rho_m_std = Omega_m0 * a_vals**-3
rho_DE_eff = rho_tot - rho_m_std

w_eff_vals = np.zeros_like(a_vals)
for i in range(len(a_vals)-1):
    da = a_vals[i+1] - a_vals[i]
    drho = rho_DE_eff[i+1] - rho_DE_eff[i]
    w_eff_vals[i] = -1.0 - (a_vals[i]/3.0) * (1.0 / rho_DE_eff[i]) * (drho / da)
w_eff_vals[-1] = w_eff_vals[-2]

x_vals = 1.0 - a_vals
X = np.vstack([np.ones_like(x_vals), x_vals]).T
w0_pred, wa_pred = np.linalg.lstsq(X, w_eff_vals, rcond=None)[0]

print(f"w0 = {w0_pred:.4f}, wa = {wa_pred:.4f}")
print(f"w_W today = {w_W_vals[0]:.4f}")
print(f"w_W at a=0.4 = {w_W_vals[-1]:.4f}")
