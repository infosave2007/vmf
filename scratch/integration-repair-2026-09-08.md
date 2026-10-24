# Integration and reproducibility repair — 2026-09-08

This record covers upstream precision, portable contracts, registry/provenance,
and the Python environment. It does not certify the unfinished full stellar
generation or observational agreement. Existing unrelated edits and all older
virtual environments were preserved.

## Upstream source and numerical behavior

`reduced_lapse_scale_action` now evaluates inside `mp.workdps(50)`, restoring
the caller's precision on successful and exceptional exits. Tests cover
vacuum and finite-density calls at ambient precision 15/37/80, a successful
finite-density control before an injected arithmetic exception, and restoration
after the exception. The pre/post precision-only change produced identical
JSON/report bytes in the original Python 3.9 diagnostic environment.

The action and observable-contract snapshots were copied byte for byte from
the private run directory into maintained `verification/contracts/` inputs.
Upstream passport references now use the maintained action file. A positive
test copies only the declared passport dependencies into a temporary tree and
validates successfully without a `Lunacy` directory. Report commands now point
to maintained producer/tests rather than private evidence directories.

Final source identities (subsequent environment regeneration does not change
these source/test bytes):

| File | SHA-256 |
|---|---|
| `verification/source_complete_solution_audit.py` | `b2bf146b71716e56232c56b7afab2b09126d6b32c8e397989f037440c5dae5a0` |
| `verification/test_source_complete_solution_audit.py` | `78a012fd46e35f3e1952c9bef4ea9240a89450cdcf6a01526a51fbbe900eb38a` |
| `verification/contracts/source_complete_action.md` | `25e42abb91028f09a76cccddc05ca4a7458bc0609df7f41fb932d165cc73606e` |
| `verification/contracts/observable_predictions.md` | `ebcc4e0cae1bf981f3fb2dd0bf6567a177dffa75aa25a160f82c2f86cbe87cef` |

The upstream JSON was regenerated in the supported Python 3.12 environment.
Against the Python 3.9 reconstruction, the only serialized numerical changes
were the two symmetric covariance off-diagonals:
`1.964262327458533e-05` → `1.9642623274585327e-05` (one ulp,
relative `1.725e-16`). The report and all other numerical output were identical.
No physical parameter or acceptance threshold changed.

Final upstream JSON/report hashes are respectively
`caac569976b33a729ed3f3fc5806f257ba3eb130144d786a127c48786ee4e5dc`
and `f19a8fb70247064371edccc899e3fd722af58e46f0230622813a77e2621106c8`.

## Registered ownership and scientific limits

The four stellar/resonator producers/tests and their four JSON/report outputs
are now registered. Both producers depend on the upstream action producer and
maintained contract inputs. Contract snapshots are zero-weight validation data;
the resonator fingerprint file is zero-weight utility metadata. The parent
owns its content. Historical workflow prose inside the exact contract
snapshots is not current execution authority.

All source-complete surfaces have zero independent empirical weight. Stellar
outputs describe a mathematical one-component sequence; physical neutron-star
claims remain blocked. Resonator outputs describe an EFT spectrum and a
conditional passive response with numerical physical power blocked.

The parent regenerated and validated the resonator in Python 3.12. Its
registry entries are maintained, with JSON/report hashes
`bf2db7984a041f6641b4d7000bcceae20729032bc6cc7097bbb4dbd285b270a0`
and `2d127e5d7453978a5dcc444676a686171bcc8284cc38d26fabd7fe9b3e9d5c85`.
All six source-complete artifact-manifest hashes match their current files.

The existing stellar JSON/report are still explicitly quarantined in the
registry and marked `stale_requires_regeneration_*` in the artifact manifest.
Their recorded hashes identify the actual old bytes and do not assert current
acceptance. No stellar or resonator source/test/artifact was edited by this
integration worker.

Both READMEs link `SOURCE_COMPLETE_MODEL_RU.md`, describe the effective
one-component scope and nuclear-saturation mismatch, and link the three
maintained calculation stages. No empirical confirmation was added.

## Supported environment

The fresh interpreter is
`/Users/oleg/Documents/NVG-Research/.venv-research/bin/python`, built from the
available bundled Python 3.12.14 arm64 runtime. Packages were installed only
into this workspace environment after a scoped network escalation was
approved. Global installations and `.venv`, `.venv_arm64`, `.venv_x64` were not
changed. The broken old arm64 interpreter was not deleted.

`requirements.txt` now directly declares mpmath, SymPy and PyCBC as well as the
existing requirements. PyCBC is a demonstrated front-door dependency:
`test_p2_s3_integrity.py` imports `nvg_echo_timeslide_background.py`, whose
PyCBC imports exit with code 1 when the package is absent. Installing it fixed
full-suite collection; no test was skipped, disabled, or modified to bypass
that dependency. Optional GW fetching's `gwosc` package was not added.

`constraints-python312.txt` records all 55 resolved direct/transitive package
versions in this environment, including NumPy 1.26.4, SciPy 1.13.1,
mpmath 1.3.0, SymPy 1.14.0, CAMB 1.6.0, pytest 8.4.2, Matplotlib 3.9.4,
and PyCBC 2.9.0. Every pin equals the installed version; an offline repeat of
the requirements/constraints installation and `pip check` succeed. These are
version constraints, not cross-platform wheel hashes. Linux CI execution has
not been run locally.

CI and both READMEs now use Python 3.12 and the same requirements/constraints.
The documented supported command after installation is
`.venv-research/bin/python verification/run_verification.py`. The front door
was inspected: it validates registry/provenance, regenerates the predictive
and canonical ledgers, then runs all semantic tests. Its complete run is
reserved for the parent's final serial verification after stellar generation.

## Verification completed

- Upstream focused suite: 21 passed in Python 3.12, including byte regeneration.
- Organization/provenance/public-artifact integration: 18 passed in 14.11 s.
- Organization plus PyCBC-dependent integrity tests: 15 passed in 1.69 s.
- Final README/registry checks: 12 passed in 0.38 s.
- Full semantic suite collection: 326 tests collected in 36.84 s, exit 0
  (snapshot while the stellar worker was still completing its changes).
- Registry CLI: valid, 396 entries / 292 scripts / 91 manifest artifacts /
  9 data-provenance entries.
- Parent reported final supported-environment upstream + resonator focused
  suites: 50 passed in 19.26 s, with resonator bytes matching regeneration.

## Remaining finalization owned by the parent

After the stellar worker's final source freeze, full generation and independent
validation succeed, compute SHA-256 of the final stellar JSON and report;
update only their two artifact-manifest hashes and change their manifest
statuses to mathematical-result/report status. Update the corresponding two
registry descriptions, clear their stale quarantine reasons, and set status
to `maintained`. Keep role `validation`, evidence weight zero, and all physical
neutron-star blockers. If a new maintained stellar contract input is added,
register that input and its producer dependency as well.

Then rerun `verification/registry.py`, the repository-organization tests, and
the full front door in `.venv-research`. A failure or interruption is not a
successful full run. Until that work finishes the stellar quarantine remains
the truthful status.

## Bilingual README semantic repair (final follow-up)

Targeted both READMEs without editing historical articles, production code,
figures, or unrelated user changes. Removed the top `Evidence status` lines;
the `PUBLICATION_STATUS.md` link remains in ordinary verification prose.
The maintained source-complete model link and its effective one-component
scope, saturation mismatch and lack of empirical confirmation remain explicit.

Corrected repeated claims that historical melting/bounce ansatzes are derived
by the present action, that 859 MeV is a direct field measurement or a
parameter-free lattice determination, and that cosmological or fundamental
problems are already solved. Numerical values remain as historical inputs,
conditional template outputs or calibration values where appropriate.
Synthetic figures, prescribed optical input/output shapes, and separate
neutrino/axion/cosmological extensions are not promoted to evidence for the
current action. HADES outreach/templates are not collaboration endorsement
or measured peaks.

The distinctions were checked against registry roles and maintained scripts,
including the explicit Cas A slope/normalization, imposed Bell/arrow-of-time
curves, sigma-term mass budget, conditional recondensation and core-frequency
templates, and the separate strong-CP/neutrino assumptions. All Markdown
link/image target occurrences were preserved: 95 in English, 109 in Russian.

Final README freeze:

- `README.md`: `c4b06406ecd4c6bc6b914f6189ed6d31f7d8528e3d1eaa649723de1c3f6dfe3c`
- `README_RU.md`: `725a8e0711dca52a7e4470878721d0bde9358382a976fe18eeb89fc5b1ed39e1`

`git diff --check -- README.md README_RU.md` passes. Supported-environment
public-artifact, repository-organization and final-consistency tests pass:
18 passed in 1.34 s. No tests required the removed banner or strong claims;
none were modified or weakened for this cleanup. The parent was notified of
the final hashes before downstream cooling regeneration.

A read-only manifest SHA scan at this handoff found five mismatches:
`fig_p3_s2_cooling_gmode_dependency_audit.png`, `fig_s8_no_go_map.png`,
`nvg_beta_hyperon_urca_audit_p2s2_results.json`,
`nvg_cooling_gmode_dependency_audit_p3s2_results.json`, and
`renormalization_covariance_audit_results.json` (all under `verification/`).
No mismatched output was rehashed by this worker. Attribution and legitimate
producer regeneration remain with the parent; the cooling JSON was already
dirty at task entry. The parent independently identified a beta test writing
a quick-grid result over the full canonical artifact and assigned its fix
and canonical regeneration to the physics worker. That is a concrete
test-induced divergence, not a harmless numerical-precision change.

The parent's later nonstellar suite reported 259 passed / 6 setup errors;
the errors were deterministic cooling-provenance mismatches while beta and
README dependencies were changing. The parent found only dependency/README
claim-scan/hash leaf differences, not cooling physics differences, and owns
serial regeneration and final re-verification. This note does not claim a
successful full-suite or front-door run.

## Authorized nonstellar manifest finalization

After the parent froze the canonical beta and cooling outputs, regenerated
`nvg_renormalization_covariance_audit.py` in `.venv-research`, then ran its
focused suite: 9 passed in 1.10 s, including exact byte regeneration of all
three outputs. Its result, inventory and report were unchanged by that run;
the result also matched current HEAD bytes. Only its manifest SHA was stale.
Its status remains `COMPLETE_WITH_RENORM_BLOCKS`, with missing operator and
renormalization metadata still blocked; no inputs or physics were refitted.

Updated exactly six authorized SHA fields in `artifact_manifest.json`, without
changing statuses, producers, roles, unrelated entries, or any stellar pins:

| Artifact under `verification/` | Accepted file SHA-256 | Regeneration justification |
| --- | --- | --- |
| `fig_p3_s2_cooling_gmode_dependency_audit.png` | `7c94cbc2c0aa89466e0fca2e82d0028b82e8b012357b988d170ff3fe4c845cdf` | Parent regenerated the actual cooling producer after dependency repair. |
| `fig_s8_no_go_map.png` | `525daf274debc844b638b7c639c9339001056bd772d9a44f206f0af584f64c30` | Parent compared HEAD and regenerated PNG: same 1890x756 dimensions, identical pixels and Matplotlib 3.9.4 metadata; source-data JSON unchanged, binary encoding only. |
| `nvg_beta_hyperon_urca_audit_p2s2_results.json` | `68f49c1d1df1e69475b76654957d8cfb599f807ab176f63807744e0a04b905ef` | Physics worker regenerated and validated the full canonical grid after moving the quick test output out of the canonical path; statuses unchanged. |
| `nvg_cooling_gmode_dependency_audit_p3s2_results.json` | `e8ec1c38fc44cd2f9f56617767f24c365df242c8317bfd6d3adf66331921530a` | Parent regenerated the producer after beta and README freeze and passed its strict provenance assertion. |
| `renormalization_covariance_audit_results.json` | `05fea92c60c33009e4653bf28b1c7f8963c53be7d68fbdf23a3ed082feaa9c5b` | Direct producer regeneration and 9 focused tests above; inventory/report needed no hash changes. |
| `predictive_research_ledger.json` | `3a97a054e90f91964f19c705595f203cc905ad24246779d1081cb65a0d9c8de5` | Parent explicitly authorized this sixth entry after real producer regeneration and 6 passing predictive integration tests, including deterministic CLI regeneration. |

File hashes above are distinct from internal semantic payload hashes. The
parent reported cooling payload
`e9baefc978c19486a3214d14ab5ef863d97a3c39010499dc987fc1fc05f6f480`
and predictive payload
`e827976553f6168a7d4e3d10afc2c49c2d86ff49f1fadac58a58699ea201ec2f`.

After these bounded updates, a read-only scan of all existing manifest-pinned
files found zero SHA mismatches. Registry validation passed (396 entries /
292 scripts / 91 artifacts / 9 inputs), organization tests passed (7 in
0.25 s), and manifest `git diff --check` was clean. README hashes and old
quarantined stellar bytes remain unchanged. Zero hash mismatches does not
accept the stale stellar result; its quarantine remains until actual full
producer acceptance. The final serial front-door run remains parent-owned.
