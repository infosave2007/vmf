# Repair and reproducibility plan — 2026-09-08

User authorization: fix the diagnosed regressions and establish a usable, scientifically honest model. Continue with GPT-6 as explicitly requested in the preceding diagnosis. Do not invent empirical confirmation, fit target results, restore contradicted claims, add publication banners, or overwrite unrelated edits. No commit/push was requested in this turn.

Baseline diagnosis: `scratch/result-divergence-audit-2026-09-08.md`.

## Acceptance

- Correct the dimensional centre launch, actual pressure mesh refinement, exact-zero derivative ordering, and identified producer/validator mismatches.
- Keep mathematical error limits and independent equations. A historical assistant-created timing limit is performance telemetry, not scientific acceptance or a reason to truncate a necessary verification. Record actual timing; use bounded resource guards sufficient to complete work safely. Do not report an interrupted run as complete.
- Scientific output and proof identities must be deterministic and exclude wall-clock telemetry. Use positive baseline controls for mutation tests and explicit rejection reasons.
- Preserve the source-complete action and parameter values. Document its applicability, parameter provenance, and unsupported physical claims. Correctness of calculations is distinct from agreement with observations.
- Bring registry/manifests/environment documentation into agreement with maintained files; preserve existing unrelated changes.
- After the final code freeze, run focused suites, full source generation, independent validation/regeneration and the repository front door; record actual failures and fix in scope. No static success flags or threshold relaxation to obtain PASS.

## Ownership

- Stellar worker: `verification/source_complete_stellar_prediction.py`, its focused test, and regenerated stellar JSON/report. Own the complete numerical and schema repair. Coordinate upstream digest changes before final freeze.
- Integration worker: upstream precision isolation and its focused tests; registry/manifest/dependency/environment integration. Do not edit stellar or resonator producer/test files. Notify parent immediately when upstream bytes are frozen. Coordinate artifact hash refresh after regeneration.
- Physics worker: independently derived, source-backed model document with meaningful calculated consequences and limits; no production source edits or invented particle content.
- Parent: global acceptance, dependency pins in resonator as needed, integration conflict resolution, independent targeted checks, final verification and user report.

## Execution state

Status: REPAIR AND NUMERICAL ACCEPTANCE COMPLETE.
The fresh stellar producer completed and wrote self-validated results in
627.154 s. Independent unmodified-API regeneration completed in 1137.188 s
and reproduced the entire canonical scientific payload, with no source,
test, result or report changes. The focused stellar suite passed all 70
tests in 374.62 s. The complete scientific front door passed: 335 tests,
276 warnings (dependency deprecations and a pre-existing string escape),
1112.99 s for pytest, exit 0.
The two stellar artifact statuses are now maintained mathematical outputs
with evidence weight zero; physical neutron-star claims remain blocked.
After metadata promotion, registry/provenance validation passed again
(396 entries, 292 scripts, 91 artifacts, 9 inputs), and the 13 organization
and provenance tests passed in 2.26 s. All pinned artifact hashes match;
the stellar and beta numerical file hashes are unchanged. `git diff --check`
passes. No commit/push was performed.
The empirical theory is not confirmed: the fixed parameterization fails
nuclear saturation and is not being retuned to hide that result.
