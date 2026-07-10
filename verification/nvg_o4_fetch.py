#!/usr/bin/env python3
"""
Fetch O4 open strain by GPS from GWOSC (bypasses pycbc's per-event catalog, which
does not yet serve O4 events). Returns a conditioned pycbc TimeSeries around a merger.

GWOSC serves O4a/O4b as 4096 s bulk HDF5 chunks (~7 MB gzip). We locate the chunk
covering [gps-16, gps+16] with gwosc.locate, download it (cached), read /strain/Strain,
and slice the 32 s window. This is the ONLY change needed to extend the echo search to
O4 -- the matched-filter / time-slide analysis is unchanged.
"""
from __future__ import annotations
import numpy as np
import fsspec
import h5py
from gwosc.locate import get_urls
from pycbc.types import TimeSeries

PAD = 16.0   # seconds each side of the merger
_FS = fsspec.filesystem("http", block_size=1 << 20)


def fetch_strain(det: str, gps: float):
    """Return a raw pycbc TimeSeries of ~32 s centered on gps, or None if unavailable.

    Reads only the needed ~1 MB slice from the remote 4096 s GWOSC HDF5 via HTTP
    range requests (no full-file download)."""
    try:
        urls = get_urls(det, gps - PAD, gps + PAD)
    except Exception:
        return None
    if not urls:
        return None
    try:
        with _FS.open(urls[0]) as fobj:
            with h5py.File(fobj, "r") as f:
                strain = f["strain"]["Strain"]
                x0 = float(strain.attrs.get("Xstart", f["meta"]["GPSstart"][()]))
                dt = float(strain.attrs.get("Xspacing", 1.0 / 4096.0))
                n = strain.shape[0]
                i_lo = int(round((gps - PAD - x0) / dt))
                i_hi = int(round((gps + PAD - x0) / dt))
                if i_lo < 0 or i_hi > n:
                    return None              # window straddles chunk boundary
                data = strain[i_lo:i_hi].astype(np.float64)
                start = x0 + i_lo * dt
    except Exception:
        return None
    if data.size == 0 or not np.all(np.isfinite(data)):
        return None                          # gap / NaN in the segment
    return TimeSeries(data, delta_t=dt, epoch=start)


if __name__ == "__main__":
    # end-to-end smoke test on one O4a event
    gps = 1376089759.8   # GW230814_230901
    for det in ("H1", "L1"):
        ts = fetch_strain(det, gps)
        if ts is None:
            print(f"  {det}: unavailable")
        else:
            print(f"  {det}: OK  dur={len(ts)*ts.delta_t:.0f}s  fs={ts.sample_rate:.0f}Hz  "
                  f"rms={float(np.std(ts)):.2e}")
