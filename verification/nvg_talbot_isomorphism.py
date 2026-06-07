"""
===============================================================================
 NVG / VMF VERIFICATION: NON-LINEAR TALBOT-VACUUM ISOMORPHISM
===============================================================================

 Numerical validation of the theoretical isomorphism between:
 1. The optical Talbot effect (paraxial wave equation / Fresnel propagation)
 2. The temporal phase evolution of the VMF vacuum condensate (Gross-Pitaevskii)

 Focuses on the non-linear regime (g |Psi|^2 Psi term) as a unique, falsifiable
 prediction for vacuum neural processors and analog gravity systems.

 Author: NVG Research Laboratory / Oleg Kirichenko
"""

import numpy as np
import matplotlib.pyplot as plt
import os

# Physical parameters of the system
LAMBDA_G = 530e-9        # Green channel wavelength of OLED (530 nm)
LAMBDA_R = 635e-9        # Red channel wavelength (635 nm)
LAMBDA_B = 470e-9        # Blue channel wavelength (470 nm)
PIXEL_PITCH = 50.3e-6    # Pixel pitch (Samsung S24 Ultra, 50.3 um)
GRATING_PERIOD = 4 * PIXEL_PITCH  # Grating period a = 201.2 um

# Calculate linear Talbot distance for RGB channels
def calculate_z_talbot(period, lam):
    return 2 * (period ** 2) / lam

z_T_R = calculate_z_talbot(GRATING_PERIOD, LAMBDA_R)
z_T_G = calculate_z_talbot(GRATING_PERIOD, LAMBDA_G)
z_T_B = calculate_z_talbot(GRATING_PERIOD, LAMBDA_B)

print("=" * 78)
print("  OLED SUBPIXEL TALBOT DISTANCE CALCULATION (a = 201.2 um):")
print("=" * 78)
print(f"  Red   (635 nm): z_T = {z_T_R*1e3:.4f} mm")
print(f"  Green (530 nm): z_T = {z_T_G*1e3:.4f} mm")
print(f"  Blue  (470 nm): z_T = {z_T_B*1e3:.4f} mm")
print("-" * 78)

# =============================================================================
# WAVE PROPAGATION MODELING (LINEAR & NON-LINEAR SSFM)
# =============================================================================

# Computational grid
N = 512  # number of grid points
L_x = N * PIXEL_PITCH / 4  # size of the spatial domain
x = np.linspace(-L_x/2, L_x/2, N, endpoint=False)
dx = x[1] - x[0]

# Input field with a periodic phase (phase confinement state)
W_0 = 1.0
E_0 = W_0 * np.exp(1j * np.sin(2 * np.pi * x / GRATING_PERIOD))

# Linear paraxial propagator (exact spectral method)
def propagate_linear(E, z, lam):
    k = 2 * np.pi / lam
    fx = np.fft.fftfreq(N, d=dx)
    kx = 2 * np.pi * fx
    kz_paraxial = k - (kx**2) / (2 * k)
    E_k = np.fft.fft(E)
    E_k_propagated = E_k * np.exp(1j * kz_paraxial * z)
    return np.fft.ifft(E_k_propagated)

# Non-linear propagator using Symmetric Split-Step Fourier Method (SSFM)
# Models NLSE / Gross-Pitaevskii: i dE/dz = -1/(2k) d^2E/dx^2 - gamma_nl |E|^2 E
def propagate_nonlinear(E, z, lam, gamma_nl, steps=200):
    k = 2 * np.pi / lam
    dz = z / steps
    fx = np.fft.fftfreq(N, d=dx)
    kx = 2 * np.pi * fx
    # Linear step propagator: exp(i * (k - kx^2 / (2k)) * dz)
    H_linear = np.exp(1j * (k - (kx**2) / (2 * k)) * dz)
    
    E_curr = E.copy().astype(complex)
    for _ in range(steps):
        # 1. Non-linear half-step
        E_curr *= np.exp(1j * gamma_nl * np.abs(E_curr)**2 * (dz / 2))
        # 2. Linear full-step
        E_curr = np.fft.ifft(np.fft.fft(E_curr) * H_linear)
        # 3. Non-linear half-step
        E_curr *= np.exp(1j * gamma_nl * np.abs(E_curr)**2 * (dz / 2))
        
    return E_curr

# Simulate Talbot carpets
z_steps = 300
z_arr = np.linspace(0, z_T_G, z_steps)

carpet_linear = np.zeros((z_steps, N))
carpet_nonlinear = np.zeros((z_steps, N))

# Attractive vacuum interaction parameter: gamma_nl < 0 corresponds to self-focusing
GAMMA_NL = -15.0  # normalized non-linear strength

print("  Simulating linear and non-linear propagation carpets...")
for i, z in enumerate(z_arr):
    # Linear propagation
    E_lin = propagate_linear(E_0, z, LAMBDA_G)
    carpet_linear[i, :] = np.abs(E_lin)**2
    
    # Non-linear propagation using SSFM
    steps = max(10, int(250 * z / z_T_G))
    E_nl = propagate_nonlinear(E_0, z, LAMBDA_G, GAMMA_NL, steps=steps)
    carpet_nonlinear[i, :] = np.abs(E_nl)**2

# Find focal positions (first local maximum of peak intensity along z-axis)
max_int_lin = np.max(carpet_linear, axis=1)
max_int_nl = np.max(carpet_nonlinear, axis=1)

def find_primary_focus(max_int_curve):
    for i in range(1, len(max_int_curve)-1):
        if max_int_curve[i] > max_int_curve[i-1] and max_int_curve[i] > max_int_curve[i+1]:
            if max_int_curve[i] > 1.2:  # threshold
                return z_arr[i], i
    return None, None

z_focus_lin, idx_focus_lin = find_primary_focus(max_int_lin)
z_focus_nl, idx_focus_nl = find_primary_focus(max_int_nl)

print("-" * 78)
print("  FOCAL PLANE COMPRESSION PREDICTION:")
print("-" * 78)
print(f"  Linear Primary Focus:     z = {z_focus_lin*1e3:.4f} mm")
print(f"  Non-Linear Primary Focus:  z = {z_focus_nl*1e3:.4f} mm (gamma = {GAMMA_NL})")
focus_shift_pct = (z_focus_lin - z_focus_nl) / z_focus_lin * 100
print(f"  Predicted Focus Shift:    {focus_shift_pct:+.2f}% (Compression/Self-Focusing)")
print("-" * 78)

# Linear coherence recovery check
E_zt_lin = propagate_linear(E_0, z_T_G, LAMBDA_G)
fidelity = np.abs(np.vdot(E_0, E_zt_lin)) / (np.linalg.norm(E_0) * np.linalg.norm(E_zt_lin))
print(f"  Linear Coherence Recovery at z = z_T: Fidelity = {fidelity:.6f}")
assert fidelity > 0.999, "Error: Linear coherence recovery failed!"

# =============================================================================
# VISUALIZATION AND PLOT GENERATION
# =============================================================================

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12), gridspec_kw={'height_ratios': [1.2, 1.2, 1]})

# 1. 2D Linear Talbot Carpet
im1 = ax1.imshow(carpet_linear, extent=[x[0]*1e6, x[-1]*1e6, z_T_G*1e3, 0], aspect='auto', cmap='viridis')
ax1.set_title("Linear Talbot Carpet (GP Equation with g = 0 / Linear Wave Propagation)", fontsize=12, fontweight='bold')
ax1.set_ylabel("Propagation $z$ (mm)", fontsize=10)
ax1.axhline(y=z_focus_lin*1e3, color='r', linestyle='--', linewidth=1.5, label=f"Primary Focus = {z_focus_lin*1e3:.2f} mm")
ax1.legend(loc='upper right')
fig.colorbar(im1, ax=ax1, label="Intensity $|E|^2$")

# 2. 2D Non-Linear Talbot Carpet (Self-focusing)
im2 = ax2.imshow(carpet_nonlinear, extent=[x[0]*1e6, x[-1]*1e6, z_T_G*1e3, 0], aspect='auto', cmap='viridis')
ax2.set_title(f"Non-Linear Talbot Carpet (GP Equation with g < 0 / Self-Focusing, $\\gamma_{{NL}} = {GAMMA_NL}$)", fontsize=12, fontweight='bold')
ax2.set_ylabel("Propagation $z$ (mm)", fontsize=10)
ax2.axhline(y=z_focus_nl*1e3, color='r', linestyle='--', linewidth=1.5, label=f"Compressed Focus = {z_focus_nl*1e3:.2f} mm")
ax2.legend(loc='upper right')
fig.colorbar(im2, ax=ax2, label="Intensity $|E|^2$")

# 3. 1D Intensity Profile Comparison at Focal Planes
ax3.plot(x*1e6, carpet_linear[idx_focus_lin, :], 'b-', label=f"Linear Focus (z = {z_focus_lin*1e3:.2f} mm)", linewidth=2)
ax3.plot(x*1e6, carpet_nonlinear[idx_focus_nl, :], 'r-', label=f"Non-Linear Focus (z = {z_focus_nl*1e3:.2f} mm)", linewidth=2)
ax3.set_title("Intensity Profile Comparison at Respective Focal Planes", fontsize=11, fontweight='bold')
ax3.set_xlabel("Transverse Coordinate $x$ (um)", fontsize=10)
ax3.set_ylabel("Normalized Intensity $|E|^2$", fontsize=10)
ax3.set_xlim(x[0]*1e6, x[-1]*1e6)
ax3.legend(loc='upper right')
ax3.grid(True, linestyle=':', alpha=0.6)

plt.tight_layout()

# Save the generated figure to both verification/ and article/
output_dir = "/Users/oleg/Documents/NVG-Research/verification"
os.makedirs(output_dir, exist_ok=True)
fig_path = os.path.join(output_dir, "fig_talbot_isomorphism.png")
plt.savefig(fig_path, dpi=300)

article_dir = "/Users/oleg/Documents/NVG-Research/article"
os.makedirs(article_dir, exist_ok=True)
article_fig_path = os.path.join(article_dir, "fig_talbot_isomorphism.png")
plt.savefig(article_fig_path, dpi=300)

plt.close()

print(f"  Plot successfully saved to: {fig_path}")
print(f"  Plot successfully saved to: {article_fig_path}")
print("=" * 78)
