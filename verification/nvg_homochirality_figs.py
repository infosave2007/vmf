#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import os

def create_figures():
    # Setup styling
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 14,
        'axes.labelsize': 16,
        'axes.titlesize': 18,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'legend.fontsize': 14,
        'lines.linewidth': 2.5
    })

    # Constants
    k_B_eV = 8.617333e-5      # eV/K
    T_room = 300.0            # K
    kT = k_B_eV * T_room      # eV
    
    delta_E_ew = 1e-17        # Standard Electroweak PVED (eV)
    delta_E_qcd = 1.45e-14    # NVG Topological PVED (eV)
    
    adv_ew = delta_E_ew / kT
    adv_qcd = delta_E_qcd / kT
    
    # Reactions axis (up to 10^15 over 1 billion years)
    # We use a log scale for reactions
    reactions = np.logspace(11, 15, 1000)
    
    ee_ew = (1.0 - np.exp(-adv_ew * reactions)) * 100
    ee_qcd = (1.0 - np.exp(-adv_qcd * reactions)) * 100

    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.semilogx(reactions, ee_qcd, color='#e74c3c', label='NVG Topological PVED ($\Delta E \sim 10^{-14}$ eV)')
    ax.semilogx(reactions, ee_ew, color='#3498db', linestyle='--', label='Electroweak PVED ($\Delta E \sim 10^{-17}$ eV)')
    
    # Highlight the 99% homochirality threshold
    ax.axhline(99.0, color='#2c3e50', linestyle=':', alpha=0.8, label='99% Homochirality Threshold')
    
    ax.set_xlabel('Number of Amplification Interactions (Prebiotic Epoch)')
    ax.set_ylabel('Enantiomeric Excess (ee) [%]')
    ax.set_title('Thermodynamic Amplification of Homochirality')
    
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left')
    
    ax.set_ylim(-2, 105)
    
    # Annotate points
    ax.annotate('Complete Homochirality\nAchieved', xy=(1e14, 99.5), xytext=(1e12, 80),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
                fontsize=12, color='#e74c3c')
                
    ax.annotate('Fails to break\nsymmetry', xy=(5e14, 2), xytext=(5e12, 20),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
                fontsize=12, color='#3498db')

    plt.tight_layout()
    
    os.makedirs('preprints/figures', exist_ok=True)
    plt.savefig('preprints/figures/fig_homochirality_amplification.png', dpi=300)
    print("Saved preprints/figures/fig_homochirality_amplification.png")
    
if __name__ == "__main__":
    create_figures()
