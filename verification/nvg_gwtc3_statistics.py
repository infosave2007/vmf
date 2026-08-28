from __future__ import annotations

import csv
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
REPOSITORY_ROOT = HERE.parent
DEFAULT_INPUT = HERE / "gwtc3_nvg_results.csv"


def resolve_input_path(value: str | Path | None = None) -> Path:
    """Resolve a GWTC result path relative to the repository, not caller cwd."""

    path = DEFAULT_INPUT if value is None else Path(value)
    return path if path.is_absolute() else REPOSITORY_ROOT / path


def calculate_stats(csv_file=None):
    csv_file = resolve_input_path(csv_file)
    snrs = []
    top_candidates = []
    
    with csv_file.open('r') as f:
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
