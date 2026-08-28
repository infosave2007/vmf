from __future__ import annotations

import csv
import matplotlib.pyplot as plt
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPOSITORY_ROOT = HERE.parent
DEFAULT_INPUT = HERE / "gwtc3_nvg_results.csv"
DEFAULT_OUTPUT = HERE / "nvg_gwtc3_snr_distribution.png"


def resolve_path(value: str | Path | None, default: Path) -> Path:
    """Resolve relative inputs/outputs against the repository root."""

    path = default if value is None else Path(value)
    return path if path.is_absolute() else REPOSITORY_ROOT / path

def plot_results(csv_file=None, plot_file=None):
    csv_file = resolve_path(csv_file, DEFAULT_INPUT)
    plot_file = resolve_path(plot_file, DEFAULT_OUTPUT)
    if not csv_file.exists():
        print(f"Error: {csv_file} not found. Run mass scan first.")
        return
        
    data = []
    with csv_file.open('r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append({
                'Event': row['Event'],
                'Catalog': row['Catalog'],
                'Max_Echo_SNR': float(row['Max_Echo_SNR'])
            })
            
    if len(data) == 0:
        print("CSV is empty.")
        return
        
    print(f"Loaded {len(data)} events from {csv_file}.")
    
    # Sort by SNR descending
    data_sorted = sorted(data, key=lambda x: x['Max_Echo_SNR'], reverse=True)
    
    print("\n--- Top 5 Candidates for NVG Echoes ---")
    print(f"{'Event':<20} | {'Catalog':<20} | {'Max_Echo_SNR'}")
    print("-" * 60)
    for row in data_sorted[:5]:
        print(f"{row['Event']:<20} | {row['Catalog']:<20} | {row['Max_Echo_SNR']:.2f}")
    
    # Plot histogram of SNRs
    snrs = [row['Max_Echo_SNR'] for row in data]
    plt.figure(figsize=(10, 6))
    plt.hist(snrs, bins=20, color='indigo', alpha=0.7, edgecolor='black')
    
    # Mark top 3 events with text
    top_3 = data_sorted[:3]
    for row in top_3:
        plt.axvline(x=row['Max_Echo_SNR'], color='red', linestyle='--', alpha=0.7)
        plt.text(row['Max_Echo_SNR'], plt.ylim()[1]*0.9, f" {row['Event']}", 
                 rotation=90, va='top', color='darkred', fontweight='bold')
                 
    plt.xlabel('Maximum Echo SNR')
    plt.ylabel('Number of Events')
    plt.title('Distribution of NVG Echo SNRs across GWTC Catalog')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plot_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_file, dpi=300)
    print(f"\nPlot saved to {plot_file}")

if __name__ == '__main__':
    plot_results()
