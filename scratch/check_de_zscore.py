import math
import numpy as np

def calculate_desi_alignment(w0_pred, wa_pred):
    w0_desi = -0.752
    wa_desi = -0.860
    sig_w0 = 0.057
    sig_wa = 0.200
    rho_corr = -0.85
    
    cov_00 = sig_w0**2
    cov_11 = sig_wa**2
    cov_01 = rho_corr * sig_w0 * sig_wa
    
    cov_matrix = np.array([
        [cov_00, cov_01],
        [cov_01, cov_11]
    ])
    
    inv_cov = np.linalg.inv(cov_matrix)
    diff = np.array([w0_pred - w0_desi, wa_pred - wa_desi])
    chi2 = float(np.dot(diff, np.dot(inv_cov, diff)))
    z_score = math.sqrt(chi2)
    p_value = math.exp(-chi2 / 2.0)
    
    return chi2, z_score, p_value

# Test for different kinetic fractions f_kin
for f_kin in np.linspace(0.01, 0.15, 15):
    w0 = -1.0 + 2.0 * f_kin
    # For CPL fit over a = 0.4 to 1.0:
    a_vals = np.linspace(0.4, 1.0, 100)
    w_vals = -1.0 + (2.0 * f_kin * a_vals**-6) / (1.0 - f_kin + f_kin * a_vals**-6)
    
    x = 1.0 - a_vals
    X = np.vstack([np.ones_like(x), x]).T
    w0_pred, wa_pred = np.linalg.lstsq(X, w_vals, rcond=None)[0]
    
    chi2, z, p = calculate_desi_alignment(w0_pred, wa_pred)
    print(f"f_kin = {f_kin:.3f} | w0 = {w0_pred:.3f}, wa = {wa_pred:.3f} | chi2 = {chi2:.2f} | Z = {z:.2f} sigma")
