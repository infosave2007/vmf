import numpy as np

def main():
    # Parameters
    a_turn = 2.5
    c_DE = -1.0 / (3.0 * (a_turn - 1.0)) # Wait, let's see: -1.0 / (3 * 1.5) = -1 / 4.5 = -0.222?
    # Ah, the summary says: "a_turn ≈ 2.5 => c_DE ≈ -0.336"
    # Let's check: what is the formula? If c_DE ≈ -0.336, how does it relate to a_turn ≈ 2.5?
    # Maybe c_DE = -1.0 / (a_turn + 0.47) or similar? Or maybe c_DE = -0.336 is derived differently?
    # Let's solve: -1.0 / (3 * (a_turn - 1.0))? No.
    # Let's check what nvg_dark_energy_desi.py currently says or write a script to fit c_DE and k to w0 = -0.888 and wa = -0.597.
    
    # Let's write the fit function and see what we get for c_DE = -0.336 and k = 0.45.
    c_DE_val = -0.336
    k_val = 0.45
    
    a = np.linspace(0.4, 1.0, 100)
    w_a = -1.0 - (c_DE_val * a) / (3.0 * (1.0 - c_DE_val * (1.0 - a))) - k_val * (1.0 - a)
    
    # We want to fit w(a) = w0 + wa * (1 - a)
    # Let x = 1 - a
    x = 1.0 - a
    # Perform linear fit: y = w0 + wa * x
    # design matrix X = [ones, x]
    X = np.vstack([np.ones_like(x), x]).T
    w0_fit, wa_fit = np.linalg.lstsq(X, w_a, rcond=None)[0]
    
    print(f"For c_DE = {c_DE_val}, k = {k_val}:")
    print(f"  w0_fit = {w0_fit:.4f} (target: -0.888)")
    print(f"  wa_fit = {wa_fit:.4f} (target: -0.597)")

if __name__ == "__main__":
    main()
