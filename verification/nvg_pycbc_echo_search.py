"""
NVG GW Echo Matched Filtering Search Script
Downloads open LIGO data for GW150914 and applies a matched filter
using the pure NVG parameter-free echo template.

Requires: pip install pycbc lalsuite gwosc
"""
import os
import numpy as np
import matplotlib.pyplot as plt

# Try to import PyCBC, handle gracefully if not installed
try:
    from pycbc.catalog import Catalog
    from pycbc.types import TimeSeries
    from pycbc.filter import matched_filter
    from pycbc.psd import interpolate, inverse_spectrum_truncation
    PYCBC_AVAILABLE = True
except ImportError:
    PYCBC_AVAILABLE = False

def generate_nvg_template_pycbc(duration_sec=1.0, sample_rate=4096, mass_solar=65.0):
    """Generates the NVG echo template and returns a PyCBC TimeSeries."""
    t = np.linspace(0, duration_sec, int(duration_sec * sample_rate))
    waveform = np.zeros_like(t)
    
    # Primary Kerr Ringdown
    f_qnm = 251.0
    tau_qnm = 3.6e-3
    
    # NVG Echo Parameters
    delta_t_echo = 0.022 * (mass_solar / 65.0) # Scales linearly with mass
    R_eff = 0.95 # Slow geometric attenuation
    
    num_echoes = int(duration_sec / delta_t_echo)
    for n in range(1, num_echoes + 1):
        t_shift = n * delta_t_echo
        idx = t >= t_shift
        t_echo = t[idx] - t_shift
        
        amplitude = (R_eff ** n)
        echo_signal = amplitude * np.exp(-t_echo / tau_qnm) * np.sin(2 * np.pi * f_qnm * t_echo)
        
        # Phase shift by pi at the boundary
        if n % 2 != 0:
            echo_signal *= -1
            
        waveform[idx] += echo_signal
        
    return TimeSeries(waveform, delta_t=1.0/sample_rate)

def run_search():
    if not PYCBC_AVAILABLE:
        print("PyCBC is not installed.")
        print("To run this script, please execute in your terminal:")
        print("  pip install pycbc lalsuite gwosc")
        return

    print("Downloading GW150914 data from GWOSC (this may take a minute)...")
    cat = Catalog('GWTC-1-confident')
    merger = cat['GW150914']
    
    # Get 32 seconds of LIGO Hanford data around the merger
    strain = merger.strain('H1')
    
    print("Conditioning data...")
    # Remove low frequency seismic noise
    strain = strain.highpass_fir(15, 512)
    
    print("Estimating Power Spectral Density (PSD)...")
    # Estimate PSD to weight the SNR appropriately by frequency
    psd = strain.psd(4)
    psd = interpolate(psd, strain.delta_f)
    psd = inverse_spectrum_truncation(psd, int(4 * strain.sample_rate), low_frequency_cutoff=15)
    
    print("Generating NVG Parameter-Free Template...")
    # Create the template and resize to match strain data length
    template = generate_nvg_template_pycbc(duration_sec=1.5, sample_rate=strain.sample_rate, mass_solar=65.0)
    template.resize(len(strain))
    
    print("Performing Matched Filtering...")
    # The actual matched filter operation
    snr = matched_filter(template, strain, psd=psd, low_frequency_cutoff=20)
    
    # Remove edges corrupted by filtering
    snr = snr.crop(4 + 4, 4) 
    
    print("Plotting results...")
    plt.figure(figsize=(12, 6))
    merger_time = merger.time
    
    # Shift time axis so t=0 is the exact merger time
    snr_time = snr.sample_times - merger_time
    
    plt.plot(snr_time, abs(snr), color='navy', lw=1.5)
    plt.xlim(0, 0.5) # Look at the first 500ms after merger
    
    # Dynamic ylim to focus on the noise floor and potential peaks
    plt.ylim(0, max(abs(snr).numpy()[(snr_time > 0) & (snr_time < 0.5)]) * 1.2)
    
    plt.xlabel('Time after GW150914 merger (s)')
    plt.ylabel('Signal-to-Noise Ratio (SNR)')
    plt.title('NVG Echo Matched Filter SNR (GW150914, H1 Observatory)')
    plt.grid(True, alpha=0.3)
    
    # Mark expected echo times based on pure NVG geometry
    for i in range(1, 20):
        plt.axvline(x=i*0.022, color='red', linestyle=':', alpha=0.5, 
                    label='Expected NVG Echo' if i==1 else "")
        
    plt.legend()
    plt.tight_layout()
    plt.savefig('nvg_gw150914_echo_snr.png', dpi=300)
    
    print("Search complete! Plot saved to 'nvg_gw150914_echo_snr.png'")
    print("Look for peaks aligning with the red dotted lines (multiples of 22ms).")

if __name__ == '__main__':
    run_search()
