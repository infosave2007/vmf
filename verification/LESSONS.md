# Lessons learned — GW echo search (read before any future scan)

Practical caveats discovered while testing the NVG post-merger echo prediction against
LVK open data (O1–O4b). Ignoring these silently inflates significance and manufactures
fake "candidates". Every one of them cost real debugging — do not relearn them.

## 1. Primary-signal leakage is the dominant systematic (MANDATORY control)

**Effect.** For a loud or massive BBH, the inspiral–merger–ringdown produces a broad
matched-filter response to the echo-comb template (a train of ~250 Hz damped sinusoids).
That response reaches the ±8 ms on-source window in *both* detectors. An L1-vs-H1
time-slide background (which moves L1's primary response out of the window) then scores
the zero-lag as a *coherent* excess — indistinguishable from an echo.

**Symptom.** The lowest-p events are the loudest/most massive ones. In the O4a/O4b run
(`nvg_echo_o4_search.py`) the naive stack showed **p = 0.0085 (2.39σ)**, entirely from
loud/massive events (e.g. GW250114, netSNR 79). It is NOT an echo.

**Mandatory control — do at least one before reporting anything:**
- **Best:** residual subtraction — subtract the best-fit IMR waveform (parameter
  estimation) and search the residual. This is what a full LVK pipeline does.
- **Quick discriminant (used here):** the **gap-tooth test** — drop the FIRST comb
  tooth and search only teeth at `2·dt, 3·dt, …`. The primary sits at the merger and can
  only bleed into the first-tooth region; a genuine echo comb keeps its later teeth. If
  the low-p vanishes when the first tooth is removed → leakage, not an echo
  (`nvg_o4_lowp_deepdive.py`). Result: GW250114 p 0.001→0.50 (leakage); the two
  "survivors" are massive systems (M≈50–78 M⊙) where one-tooth removal is insufficient
  (broad ringdown bleeds into teeth 2–3) → residual leakage, still not a detection.

## 2. Do NOT gate the primary in the time domain

Zeroing the strain around the merger (even with a cosine taper) injects a broadband edge
transient and corrupts the PSD estimate → the matched filter rings on the gate edge and
returns nonsensical SNR² ~ 10⁵–10⁶. A naive gate is invalid. Use the template-based
gap-tooth discriminant or proper inpainting instead.

## 3. Use a coincidence (time-slide) background, never off-source-only

A single-segment off-source background inflates significance because the on-source sits at
the loudest, most non-stationary time. GW150914 gave a spurious p≈0.005 (2.8σ) with an
off-source background; the L1-vs-H1 time-slide background collapsed it to p≈0.8 (it was
single-detector; `nvg_echo_timeslide_background.py`). Always: time-slide background +
single-detector decomposition (a one-detector excess is instrumental, never an echo).

## 4. Remember the base rate

Per-event p-values are only look-elsewhere-corrected over the delay scan, not over the
event population. Finding a couple of events at p~0.01 across ~40 events is expected:
Poisson P(≥2 | μ = 0.01·40) ≈ 5% (~1.9σ). Do not read marginal per-event p as a detection.

## 5. "Coherent" here is a weak label

The network statistic `|SNR_H1|² + |SNR_L1|²` does not enforce phase/antenna-pattern
consistency, so two co-located ~2σ noise bumps register as "coherent". A real claim needs
a true coherent-network statistic (Gürsel–Tinto / coherent SNR), DQ CAT2/3 vetoes,
hardware-injection checks, and injection-recovery efficiency — a full pipeline.

## 6. Data access

pycbc's per-event catalog (`pycbc.catalog.Merger`) does not serve O4 strain. Fetch O4
directly from GWOSC: `gwosc.locate.get_urls(det, gps-16, gps+16)` → lazy HTTP-range read
of the 4096 s HDF5 with fsspec+h5py (`nvg_o4_fetch.py`, ~1.1 MB/event, not the full
~130 MB file). The identical analysis engine (`analyse_strains`) then runs unchanged.

---
**Bottom line (as of 2026-07, O1–O4b open data):** no evidence for NVG echoes; the naive
O4 excess is primary-signal leakage; a deep, quotable upper limit requires
residual-subtracted data (institutional pipeline, future work).
