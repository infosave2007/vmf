import csv
import numpy as np

def calculate_stats(csv_file='gwtc3_nvg_results.csv'):
    snrs = []
    top_candidates = []
    
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            snr = float(row['Max_Echo_SNR'])
            snrs.append(snr)
            top_candidates.append({
                'Event': row['Event'],
                'Catalog': row['Catalog'],
                'Max_Echo_SNR': snr
            })
            
    snrs = np.array(snrs)
    mean_snr = np.mean(snrs)
    std_snr = np.std(snrs)
    
    print(f"Total events: {len(snrs)}")
    print(f"Background Mean SNR: {mean_snr:.2f}")
    print(f"Background Std Dev (sigma): {std_snr:.2f}")
    
    top_candidates.sort(key=lambda x: x['Max_Echo_SNR'], reverse=True)
    
    print("\nTop Candidates Deviation:")
    for row in top_candidates[:5]:
        z_score = (row['Max_Echo_SNR'] - mean_snr) / std_snr
        print(f"{row['Event']:<20} | SNR: {row['Max_Echo_SNR']:.2f} | Deviation: +{z_score:.2f} sigma")

if __name__ == '__main__':
    calculate_stats()
