# Repair acceptance record — 2026-09-08

State: REPAIR AND NUMERICAL ACCEPTANCE COMPLETE.
Independent stellar validation and the integrated run both passed. No empirical confirmation,
parameter fitting, commit, or push is implied.

## Completed independent physics check

The executable code in `.docs/SOURCE_COMPLETE_RESPONSE_AUDIT.md` was rerun in
the supported Python 3.12 environment. It uses only mpmath and the declared
decimal inputs, not production EOS/TOV functions or saved result tables.
The 80/120-digit calculations agree to 28 digits. All 1000 sampled densities
from 0.01 to 10 n0 have positive scalar curvature and compressibility and
0 < cs2 < 1; this is a sampled check, not a continuous interval theorem.

At n0, P = 18.7236168238308755 MeV/fm3 and epsilon/n - MN =
124.3174696235643115 MeV. Its components in this effective functional are:
scalar mass shift -0.7193689209, Fermi kinetic energy +21.8212110074,
scalar potential +0.2727231104, vector energy +102.9429044266 MeV per baryon.
Thus the fixed parameterization fails nuclear saturation independently of
the stellar solver. These are not observable QCD mass fractions.

The field minimum is n/n0 = 2.1410263101189605 and
W/W0 = 0.9989323820179688, not W = 0. The on-shell identity
C = 2 lambda W0^2 - 3 n B/W and thermodynamic derivative identities pass
independent high-precision differentiation. Their assumptions and limits
are stated in `SOURCE_COMPLETE_MODEL_RU.md`.

## Completed implementation and environment checks

- The upstream arbitrary-precision context is lexically scoped and restored,
  including exceptional exits. Physical action and parameters are unchanged.
- Runtime contracts now reside under `verification/contracts/`; accepted
  inputs no longer require the ignored/private Lunacy directory.
- A fresh `.venv-research` uses Python 3.12.14 with 55 resolved package pins.
  PyCBC is required by existing echo integrity tests and was installed.
  `pip check`, offline pin reconciliation, and full test collection succeed.
  Old environments and global packages were not modified.
- Upstream plus resonator focused suites: 50 passed in 19.26 seconds.
  Resonator regeneration matches byte for byte in the supported environment.
  Its modes and response results are unchanged; provenance is portable.
  Failed resonator results cannot overwrite artifacts; NaN/Inf serialization
  is rejected. The response remains conditional and physical power blocked.
- Registry validation passes with 396 entries, 292 scripts, 91 manifest
  artifacts and 9 data-provenance records. Stellar artifact registration is
  promoted to maintained mathematical output after independent validation;
  evidence weight remains zero and physical neutron-star claims blocked.

## Stellar regression and completed numerical acceptance

The diagnosed centre recurrence used source-unit coefficients as if radius
were already geometric. The correct formal coordinate is
rho = sqrt(K_CONV) * r_km, with m_km = m_formal / sqrt(K_CONV).
Final source-unit pressure is multiplied by K_CONV. Converting only the
final mass/pressure without converting radial powers changes the actual
central state and is not a small integration-tolerance error.

Repair work also covers actual distinct RK4 step sizes, the exact-vacuum
derivative order, producer/validator schema agreement, deterministic proof
identity, formal-to-consumed coefficient binding and positive mutation-test
controls. Numerical convergence is separate from mathematical certification:
an ulp box around a computed tangent or a replay of the same rounded
arithmetic is not a rigorous global ODE error bound.

The independent regeneration comparison and integrated front door both
finished with exit 0. Earlier failed attempts remain recorded below.

First complete-build attempt stopped after both grids and independent
pressure trajectories in an external logging wrapper: it accessed args[0]
for a zero-argument gamma2 control. This was not a physics or numerical-gate
failure, but it prevented final validation and writing. The attempt is not
accepted; the original log is retained. After the logging fix, the complete
second run finished successfully in 627.154 s and wrote self-validated
artifacts. Production source and test identities did not change. The
focused stellar suite then passed all 70 tests in 374.62 s, including
positive controls and deliberately corrupted coefficient/trajectory data.

Fresh result: first mass maximum M = 2.7425498519222313 solar masses,
R = 19.673224944429776 km at nc/n0 = 2.485075342166858,
Lambda = 52.69331973210884. This is a mathematical one-component
sequence, not an observationally validated neutron-star prediction.
Numerical convergence passes. All three independent interval-diagnostic
replays pass, while all three aggregate rounding-cell checks are false;
neither a rigorous global error bound nor correct rounding is claimed.

Frozen stellar SHA256 identities:

- producer: b22c9e4860f3a9a3d7cd2204f0e7529f67ede10cd178cd9dcfceabbcc8f81a17
- test: 30ed8a95679ab5b5c35d1bbf5e4da9a6ec6a4ee697ba2f9fdce43d85fcf6a196
- JSON: 4b46bcac26e5a4b884d36b0d864de4307f13c2efc3b7267c0a1923a19d9611b8
- report: 91d365515a92764482d3b4253866aeee1fe87aba81f095a5363666cb1ebbfa2a
- canonical scientific payload: 8bd852937d6ff5e8a512bd6bd932bf6970ed9dd435ea9f827e21bffb5eed9dec

The independent fresh-process public `validate_contract` completed in
1137.1875465420017 s with `(True, [])`. No evaluator injection, monkeypatch,
stored-data fallback or artifact write was used. It regenerated the whole
scientific result and matched canonical bytes; additional assertions
confirmed unchanged JSON/report/source/test bytes and exact report rendering.
Log: `scratch/stellar-independent-acceptance-2026-09-08.log`.

## Additional test-induced result regression

The non-stellar suite exposed another cause of changing project results:
the beta/hyperon CLI test wrote its quick 61/121 grid into the canonical
120/241 JSON. The test now uses a temporary output and asserts that the
canonical bytes remain unchanged. The producer adds only an output-path
argument; its equations and numerical parameters are unchanged.

The full canonical beta artifact was regenerated in Python 3.12.14. Its
composition, onset and Urca results are bit-identical to HEAD; changed
numerical leaves have maximum absolute difference 2.274e-13. The focused
suite passed all 7 tests. JSON SHA256:
68f49c1d1df1e69475b76654957d8cfb599f807ab176f63807744e0a04b905ef.

The first non-stellar run finished with 259 passes and six setup errors in
719.27 seconds. All six were the same predictive-ledger prerequisite:
the cooling artifact no longer matched live source/README provenance during
edits. An independent full-payload comparison found only source/README
hashes, scanned documentation lines and the payload identity had changed,
not physical numerical quantities. After README and beta source freeze,
the cooling producer was genuinely regenerated and validated, then the
predictive ledger was regenerated; all six controls subsequently passed
in 7.90 s. The complete front door was rerun from the frozen state and passed.

Both READMEs now distinguish historical proposals, synthetic figures,
calibrations and maintained results. The unwanted top banners were removed
without discarding the ordinary provenance link. All original Markdown
link/image targets were preserved. Final focused documentation checks:
18 passed; no test was weakened or edited for the wording cleanup.

## Integrated acceptance and remaining limits

`verification/run_verification.py` exited 0. Registry/provenance, predictive
ledger and canonical runtime ledger passed. The complete semantic suite
passed all 335 tests in 1112.99 s with 276 warnings (dependency deprecations
and a pre-existing string escape warning in the echo figure title);
no tests failed or were skipped. The optional process-only smoke runner
is separate and was not run by this front door. Linux CI was not executed
on this macOS host. Log: `scratch/final-front-door-2026-09-08.log`.

The old calibrated canonical EOS still gives Mmax = 2.048 solar masses,
R1.4 = 12.550 km, Lambda1.4 = 519.4, with explicitly conditional/in-sample
comparison. It is not the source-complete one-component EOS. The difference
between its maximum mass and 2.742550 does not compare identical models.

Independent integration review confirmed all 55 installed constraint pins,
clean `pip check`, all available artifact hash pins and portable runtime
contracts. After the two stellar status promotions, registry validation
passed (396 entries, 292 scripts, 91 artifacts, 9 input records) and the
13 organization/provenance tests passed in 2.26 s. All manifest paths and
SHA pins were rechecked; no mismatch. The four stellar source/test/result/
report identities and the beta JSON identity listed above remain unchanged.
`git diff --check` passes. No production, test or generated numerical bytes
changed after acceptance.

The repair establishes a reproducible mathematical effective model and
removes concrete regressions and false evidential claims. It does not
establish an empirically valid universal theory, physical neutron-star EOS,
black-hole interior or energy-producing device. Fixing the failed nuclear
saturation comparison requires a justified change of interactions or
explicit calibration followed by independent testing, not hidden fitting.
Unrelated pre-existing worktree edits are preserved. No commit or push.
