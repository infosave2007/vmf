import argparse
import os
import sys
import csv
import traceback
import numpy as np
import matplotlib.pyplot as plt

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
    
    # Primary Kerr Ringdown - highly simplified scaling
    # f_qnm is roughly inversely proportional to mass
    # For 65 M_sun, f_qnm was ~251 Hz. 
    # tau_qnm is roughly proportional to mass.
    f_qnm = 251.0 * (65.0 / mass_solar)
    tau_qnm = 3.6e-3 * (mass_solar / 65.0)
    
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

def process_event(merger_name, catalog_name, output_csv):
    """Processes a single event, computes max SNR, and saves to CSV."""
    cat = Catalog(catalog_name)
    merger = cat[merger_name]
    
    # Extract mass
    # Total mass is usually available as mass_1_source + mass_2_source
    try:
        m1 = merger.data['mass_1_source']
        m2 = merger.data['mass_2_source']
        total_mass = m1 + m2
    except KeyError:
        # Fallback if mass is not explicitly named this way
        try:
            total_mass = merger.data['total_mass_source']
        except KeyError:
            print(f"Skipping {merger_name}: Could not determine total mass.")
            return False
            
    # Try H1, fallback to L1
    detector = 'H1'
    try:
        strain = merger.strain(detector)
    except Exception:
        try:
            detector = 'L1'
            strain = merger.strain(detector)
        except Exception:
            print(f"Skipping {merger_name}: No H1 or L1 strain data found.")
            return False
            
    print(f"Processing {merger_name} (Total Mass: {total_mass:.1f} M_sun) using {detector} data...")
    
    try:
        # Condition data
        strain = strain.highpass_fir(15, 512)
        psd = strain.psd(4)
        psd = interpolate(psd, strain.delta_f)
        psd = inverse_spectrum_truncation(psd, int(4 * strain.sample_rate), low_frequency_cutoff=15)
        
        # Generate dynamically scaled template
        template = generate_nvg_template_pycbc(duration_sec=1.5, sample_rate=strain.sample_rate, mass_solar=total_mass)
        template.resize(len(strain))
        
        # Filter
        snr = matched_filter(template, strain, psd=psd, low_frequency_cutoff=20)
        snr = snr.crop(4 + 4, 4) 
        
        # Extract max SNR in the post-merger window [0, 0.5s]
        merger_time = merger.time
        snr_time = snr.sample_times - merger_time
        
        post_merger_idx = (snr_time > 0.0) & (snr_time < 0.5)
        snr_post_merger = abs(snr)[post_merger_idx]
        
        if len(snr_post_merger) == 0:
            max_snr = 0.0
        else:
            max_snr = float(max(snr_post_merger))
            
        print(f"[{merger_name}] Max Echo SNR: {max_snr:.2f}")
        
        # Save to CSV
        file_exists = os.path.isfile(output_csv)
        with open(output_csv, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            if not file_exists:
                writer.writerow(['Event', 'Catalog', 'Detector', 'Total_Mass', 'Max_Echo_SNR'])
            writer.writerow([merger_name, catalog_name, detector, total_mass, max_snr])
            
        return True
        
    except Exception as e:
        print(f"Error processing {merger_name}: {e}")
        # traceback.print_exc()
        return False

def main():
    parser = argparse.ArgumentParser(description="Scan GWTC catalogs for NVG Echoes")
    parser.add_argument('--test-mode', action='store_true', help='Only run on 3 events for testing')
    parser.add_argument('--output', type=str, default='gwtc3_nvg_results.csv', help='Output CSV file')
    args = parser.parse_args()
    
    if not PYCBC_AVAILABLE:
        print("PyCBC is not installed. Please install it.")
        sys.exit(1)
        
    catalogs_to_scan = ['GWTC-3-confident', 'GWTC-2.1-confident']
    
    # Load already processed events to allow resuming
    processed_events = set()
    if os.path.exists(args.output):
        with open(args.output, 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                processed_events.add(row['Event'])
                
    print(f"Found {len(processed_events)} already processed events in {args.output}")
    
    processed_count = 0
    max_to_process = 3 if args.test_mode else 9999
    
    for cat_name in catalogs_to_scan:
        try:
            print(f"Loading catalog {cat_name}...")
            cat = Catalog(cat_name)
        except Exception as e:
            print(f"Could not load catalog {cat_name}: {e}")
            continue
            
        for merger_name in cat.names:
            if processed_count >= max_to_process:
                print("Reached maximum events to process.")
                return
                
            if merger_name in processed_events:
                print(f"Skipping {merger_name} (already processed)")
                continue
                
            success = process_event(merger_name, cat_name, args.output)
            if success:
                processed_count += 1
                processed_events.add(merger_name)

if __name__ == '__main__':
    main()
