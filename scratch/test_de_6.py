import numpy as np

# DESI DR2 joint constraint ellipse for w0, wa
# Best-fit values
w0_desi = -0.752
wa_desi = -0.860
# 1-sigma uncertainties
sigma_w0 = 0.057
sigma_wa = 0.200
# Correlation coefficient
rho = -0.85

def calc_z_score(w0, wa):
    # Covariance matrix elements
    C_w0_w0 = sigma_w0**2
    C_wa_wa = sigma_wa**2
    C_w0_wa = rho * sigma_w0 * sigma_wa
    
    # Inverse covariance matrix elements
    det_C = C_w0_w0 * C_wa_wa - C_w0_wa**2
    inv_C_w0_w0 = C_wa_wa / det_C
    inv_C_wa_wa = C_w0_w0 / det_C
    inv_C_w0_wa = -C_w0_wa / det_C
    
    # Parameter differences
    dw0 = w0 - w0_desi
    dwa = wa - wa_desi
    
    # Delta Chi-squared
    delta_chi2 = (inv_C_w0_w0 * dw0**2 + 
                  inv_C_wa_wa * dwa**2 + 
                  2.0 * inv_C_w0_wa * dw0 * dwa)
    
    return np.sqrt(delta_chi2)

print("Dynamic tracking:", calc_z_score(-0.876, -0.667))
