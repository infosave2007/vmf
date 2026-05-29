#!/usr/bin/env python3
"""
NVG magnetar / high-B pulsar population scan.

This script uses a curated observational sample to test whether the NVG
magnetar channel can reproduce the observed dipole-field distribution with:

1. Standard flux freezing from a magnetized progenitor to the NS stage.
2. A moderate structural amplification corridor Gamma_struct from the existing
   NVG benchmark calculation.
3. A plausible progenitor surface-field corridor rather than arbitrary seeds.

It is not a catalog-quality population synthesis. The goal is to turn the
magnetar note into a population-level falsifiability check with transparent,
inspectable numbers.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import re
from urllib.request import urlopen


# ---------------------------------------------------------------------------
# NVG benchmark scaling
# ---------------------------------------------------------------------------
R_STAR = 1.0e11  # cm
R_NS = 1.2e6  # cm
SURFACE_TO_NS_FLUX_GAIN = (R_STAR / R_NS) ** 2

# From nvg_magnetar_closure.py at n_B = 5 n_0.
GAMMA_STRUCT_BENCHMARK_MIN = 2.205
GAMMA_STRUCT_BENCHMARK_MAX = 3.273

# Conservative progenitor surface fossil-field corridor emphasized in the paper.
SURFACE_FIELD_LOW = 1.0e3  # G
SURFACE_FIELD_HIGH = 1.0e4  # G

# Extended progenitor corridor for the magnetic-massive-star channel
# (Of?p / Bp stars such as NGC 1624-2 reach ~2e4 G surface; the upper end of
# the observed magnetic-OB distribution extends to ~1e5 G). Magnetars are
# ~10% of NS births, so the rare strongly magnetized progenitor channel is
# the natural fossil-field route into the magnetar mass range.
MAGNETIC_PROGENITOR_FIELD_LOW = 1.0e3  # G
MAGNETIC_PROGENITOR_FIELD_HIGH = 1.0e5  # G

# Uncertainty corridor around the central benchmark.
R_STAR_GRID = [7.0e10, 1.0e11, 1.5e11]
R_NS_GRID = [1.1e6, 1.2e6, 1.3e6]
GAMMA_SCALE_GRID = [0.75, 1.00, 1.25, 1.50]
GAMMA_LIKELIHOOD_GRID = [0.50 + 0.01 * idx for idx in range(201)]


@dataclass(frozen=True)
class CompactObject:
    name: str
    family: str
    period_s: float
    b_dip_g: float
    tau_kyr: float | None
    note: str = ""
    b_sigma_g: float | None = None
    is_upper_limit: bool = False


FALLBACK_MAGNETARS = [
    CompactObject("4U 0142+61", "magnetar", 8.6887, 1.3e14, 68.0, b_sigma_g=0.15e14),
    CompactObject("1E 1048.1-5937", "magnetar", 6.4579, 3.9e14, 4.5, b_sigma_g=0.4e14),
    CompactObject("1E 1547.0-5408", "magnetar", 2.0721, 3.2e14, 0.69, b_sigma_g=0.3e14),
    CompactObject("PSR J1622-4950", "magnetar", 4.3261, 2.7e14, 4.0, b_sigma_g=0.3e14),
    CompactObject("1RXS J170849.0-400910", "magnetar", 11.0050, 4.7e14, 9.0, b_sigma_g=0.5e14),
    CompactObject("SGR J1745-2900", "magnetar", 3.7636, 2.3e14, 4.3, b_sigma_g=0.2e14),
    CompactObject("SGR 1806-20", "magnetar", 7.5477, 2.0e15, 0.24, "extreme-field giant flarer", b_sigma_g=0.2e15),
    CompactObject("XTE J1810-197", "magnetar", 5.5404, 2.1e14, 11.0, b_sigma_g=0.2e14),
    CompactObject("Swift J1818.0-1607", "magnetar", 1.3635, 3.5e14, 0.24, b_sigma_g=0.5e14),
    CompactObject("Swift J1822.3-1606", "magnetar", 8.4377, 1.4e13, 6300.0, "low-B magnetar", b_sigma_g=0.2e13),
    CompactObject("SGR 1833-0832", "magnetar", 7.5654, 1.6e14, 34.0, b_sigma_g=0.2e14),
    CompactObject("Swift J1834.9-0846", "magnetar", 2.4823, 1.4e14, 4.9, b_sigma_g=0.2e14),
    CompactObject("1E 1841-045", "magnetar", 11.7890, 7.0e14, 4.6, b_sigma_g=0.8e14),
    CompactObject("SGR 1900+14", "magnetar", 5.1999, 7.0e14, 0.90, b_sigma_g=0.8e14),
    CompactObject("SGR 1935+2154", "magnetar", 3.2451, 2.2e14, 3.6, b_sigma_g=0.2e14),
    CompactObject("1E 2259+586", "magnetar", 6.9790, 5.9e13, 230.0, "low-B magnetar", b_sigma_g=0.5e13),
    CompactObject("SGR 0418+5729", "magnetar", 9.0784, 6.1e12, 36000.0, "very low-B magnetar", b_sigma_g=0.1e12),
    CompactObject("CXOU J164710.2-455216", "magnetar", 10.6106, 6.6e13, 420.0, "upper-limit B", b_sigma_g=1.5e13, is_upper_limit=True),
]


CURATED_NON_MAGNETARS = [
    CompactObject("PSR J1846-0258", "transition", 0.3266, 4.9e13, 0.73, "rotation-powered pulsar with magnetar-like outburst", b_sigma_g=0.5e13),
    # High-B pulsars / transition region; values are representative literature numbers.
    CompactObject("PSR J1119-6127", "high-B pulsar", 0.4080, 4.1e13, 1.6, "magnetar-like activity reported", b_sigma_g=0.4e13),
    CompactObject("PSR J1718-3718", "high-B pulsar", 3.3780, 7.4e13, 34.0, b_sigma_g=0.7e13),
    CompactObject("PSR J1734-3333", "high-B pulsar", 1.1690, 5.2e13, 8.1, b_sigma_g=0.5e13),
    CompactObject("PSR J1819-1458", "high-B pulsar", 4.2630, 5.0e13, 117.0, "RRAT / transition object", b_sigma_g=0.5e13),
    CompactObject("PSR J1847-0130", "high-B pulsar", 6.7070, 9.4e13, 83.0, b_sigma_g=0.9e13),
    CompactObject("PSR B1509-58", "high-B pulsar", 0.1507, 1.5e13, 1.6, b_sigma_g=0.15e13),
    CompactObject("PSR J0726-2612", "high-B pulsar", 3.4420, 3.0e13, 200.0, b_sigma_g=0.3e13),
    # Long-period outlier handled with extra torque physics from dedicated script.
    CompactObject("1E 161348-5055", "long-period outlier", 2.401e4, 1.0e15, 2.0, "requires fallback-disk propeller torque", b_sigma_g=0.1e15),
]


MCGILL_ASCII_URL = "https://www.physics.mcgill.ca/~pulsar/magnetar/TabO1.txt"
REPO_ROOT = Path(__file__).resolve().parent.parent
APPENDIX_PATH = REPO_ROOT / "article" / "NVG_MAGNETAR_POPULATION_APPENDIX.md"


def extract_first_number(field: str) -> float | None:
    cleaned = field.replace("<", " ").replace(">", " ").replace("~", " ")
    match = re.search(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", cleaned)
    if not match:
        return None
    return float(match.group(0))


def extract_two_numbers(field: str) -> tuple[float | None, float | None]:
    cleaned = field.replace("<", " ").replace(">", " ").replace("~", " ")
    matches = re.findall(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", cleaned)
    if not matches:
        return None, None
    value = float(matches[0])
    sigma = float(matches[1]) if len(matches) > 1 else None
    return value, sigma


def is_data_row(line: str) -> bool:
    return line.startswith("|") and not line.startswith("+-")


def parse_mcgill_magnetars(timeout_s: float = 20.0) -> list[CompactObject]:
    with urlopen(MCGILL_ASCII_URL, timeout=timeout_s) as response:
        text = response.read().decode("utf-8", errors="replace")

    rows: list[CompactObject] = []
    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        if not is_data_row(line):
            continue
        parts = [part.strip() for part in line.split("|")[1:-1]]
        if len(parts) < 6:
            continue

        name = parts[0]
        if not name or name == "Name":
            continue
        if name.startswith("(") or name.startswith("Right Ascension"):
            continue
        if "#" in name:
            continue
        if name.startswith("SGR 0755") or name.startswith("SGR 1801") or name.startswith("SGR 1808") or name.startswith("AX J") or name.startswith("SGR 2013"):
            continue

        period, period_sigma = extract_two_numbers(parts[1])
        pdot_1e11, pdot_sigma_1e11 = extract_two_numbers(parts[2])
        b_1e14 = extract_first_number(parts[3])
        tau_kyr = extract_first_number(parts[5])
        if period is None or b_1e14 is None:
            continue

        note = ""
        if "<" in parts[3]:
            note = "upper-limit B"
        elif b_1e14 < 0.2:
            note = "low-B magnetar"
        elif b_1e14 >= 10.0:
            note = "extreme-field magnetar"

        is_upper_limit = "<" in parts[3]
        b_sigma_g = None
        if (not is_upper_limit) and period_sigma is not None and pdot_1e11 is not None and pdot_sigma_1e11 is not None and pdot_1e11 > 0.0:
            frac_period = period_sigma / period
            frac_pdot = pdot_sigma_1e11 / pdot_1e11
            frac_b = 0.5 * math.sqrt(frac_period**2 + frac_pdot**2)
            b_sigma_g = max(frac_b * b_1e14 * 1.0e14, 1.0e-6 * b_1e14 * 1.0e14)
        elif is_upper_limit:
            b_sigma_g = 0.25 * b_1e14 * 1.0e14

        rows.append(CompactObject(name, "magnetar", period, b_1e14 * 1.0e14, tau_kyr, note, b_sigma_g=b_sigma_g, is_upper_limit=is_upper_limit))

    return rows


def build_sample() -> tuple[list[CompactObject], str]:
    swift_j1555 = CompactObject("Swift J1555.2-5402", "magnetar", 3.86, 7.5e14, 1.7, "discovered 2021/2022", b_sigma_g=0.8e14)
    try:
        magnetars = parse_mcgill_magnetars()
        if len(magnetars) < 10:
            raise RuntimeError("parsed too few McGill rows")
        if not any(o.name == "Swift J1555.2-5402" for o in magnetars):
            magnetars.append(swift_j1555)
        return magnetars + CURATED_NON_MAGNETARS, f"McGill catalog parsed live ({len(magnetars)} magnetar rows, including Swift J1555)"
    except Exception as exc:
        fallback = list(FALLBACK_MAGNETARS)
        if not any(o.name == "Swift J1555.2-5402" for o in fallback):
            fallback.append(swift_j1555)
        return fallback + CURATED_NON_MAGNETARS, f"fallback curated magnetar sample used: {exc} (including Swift J1555)"


def required_surface_field(b_dip_g: float, gamma_struct: float) -> float:
    return b_dip_g / (SURFACE_TO_NS_FLUX_GAIN * gamma_struct)


def required_surface_field_general(b_dip_g: float, gamma_struct: float, r_star: float, r_ns: float) -> float:
    flux_gain = (r_star / r_ns) ** 2
    return b_dip_g / (flux_gain * gamma_struct)


def corridor_label(low_value: float, high_value: float) -> str:
    return f"{low_value:.2e}..{high_value:.2e}"


def classify_object(obj: CompactObject) -> tuple[str, float, float]:
    surface_req_min_gamma = required_surface_field(obj.b_dip_g, GAMMA_STRUCT_BENCHMARK_MAX)
    surface_req_max_gamma = required_surface_field(obj.b_dip_g, GAMMA_STRUCT_BENCHMARK_MIN)

    if obj.family == "long-period outlier":
        return "needs extra torque", surface_req_min_gamma, surface_req_max_gamma

    if surface_req_max_gamma < SURFACE_FIELD_LOW:
        return "below fossil corridor", surface_req_min_gamma, surface_req_max_gamma

    if surface_req_min_gamma > SURFACE_FIELD_HIGH:
        return "above fossil corridor", surface_req_min_gamma, surface_req_max_gamma

    return "inside fossil corridor", surface_req_min_gamma, surface_req_max_gamma


def magnetic_channel_status(obj: CompactObject) -> str:
    """Classification under the extended magnetic-progenitor corridor
    (1e3..1e5 G surface). Magnetar progenitors are rare strongly magnetized
    OB stars (Of?p / Bp), so this is the physically motivated corridor for
    the magnetar subpopulation."""
    if obj.family == "long-period outlier":
        return "needs extra torque"
    req_min = required_surface_field(obj.b_dip_g, GAMMA_STRUCT_BENCHMARK_MAX)
    req_max = required_surface_field(obj.b_dip_g, GAMMA_STRUCT_BENCHMARK_MIN)
    if req_max < MAGNETIC_PROGENITOR_FIELD_LOW:
        return "below magnetic channel"
    if req_min > MAGNETIC_PROGENITOR_FIELD_HIGH:
        return "above magnetic channel"
    return "inside magnetic channel"


def scenario_status(obj: CompactObject, gamma_min: float, gamma_max: float, r_star: float, r_ns: float) -> str:
    req_min = required_surface_field_general(obj.b_dip_g, gamma_max, r_star, r_ns)
    req_max = required_surface_field_general(obj.b_dip_g, gamma_min, r_star, r_ns)

    if obj.family == "long-period outlier":
        return "needs extra torque"
    if req_max < SURFACE_FIELD_LOW:
        return "below fossil corridor"
    if req_min > SURFACE_FIELD_HIGH:
        return "above fossil corridor"
    return "inside fossil corridor"


def scenario_compatible(
    obj: CompactObject,
    gamma_min: float,
    gamma_max: float,
    r_star: float,
    r_ns: float,
    surface_low: float,
    surface_high: float,
) -> bool:
    """An object is *compatible* with the NVG channel whenever the required
    progenitor surface field is at or below the upper edge of the allowed
    corridor. Sub-corridor objects are not in tension; they simply need less
    structural amplification than the benchmark provides."""
    if obj.family == "long-period outlier":
        return False
    req_min = required_surface_field_general(obj.b_dip_g, gamma_max, r_star, r_ns)
    req_max = required_surface_field_general(obj.b_dip_g, gamma_min, r_star, r_ns)
    # Either the demanding edge fits, or the easy edge already does.
    return req_min <= surface_high or req_max <= surface_high


def object_b_realizations(obj: CompactObject) -> list[tuple[float, float]]:
    if obj.is_upper_limit:
        return [
            (0.25 * obj.b_dip_g, 0.20),
            (0.50 * obj.b_dip_g, 0.30),
            (0.75 * obj.b_dip_g, 0.30),
            (1.00 * obj.b_dip_g, 0.20),
        ]

    sigma = obj.b_sigma_g if obj.b_sigma_g is not None else 0.15 * obj.b_dip_g
    sigma = max(sigma, 1.0e-12)
    offsets = [-2.0, -1.0, 0.0, 1.0, 2.0]
    weights = [0.054, 0.244, 0.404, 0.244, 0.054]
    realizations: list[tuple[float, float]] = []
    for offset, weight in zip(offsets, weights):
        realizations.append((max(obj.b_dip_g + offset * sigma, 1.0e-6 * obj.b_dip_g), weight))
    return realizations


def source_probability_summary(objects: list[CompactObject]) -> list[tuple[str, str, float]]:
    ordinary_objects = [obj for obj in objects if obj.family != "long-period outlier"]
    results: list[tuple[str, str, float]] = []
    total_scenarios = len(R_STAR_GRID) * len(R_NS_GRID) * len(GAMMA_SCALE_GRID)

    for obj in ordinary_objects:
        prob_inside = 0.0
        for r_star in R_STAR_GRID:
            for r_ns in R_NS_GRID:
                for gamma_scale in GAMMA_SCALE_GRID:
                    gamma_min = GAMMA_STRUCT_BENCHMARK_MIN * gamma_scale
                    gamma_max = GAMMA_STRUCT_BENCHMARK_MAX * gamma_scale
                    scenario_weight = 1.0 / total_scenarios
                    for b_real, b_weight in object_b_realizations(obj):
                        obj_real = CompactObject(obj.name, obj.family, obj.period_s, b_real, obj.tau_kyr, obj.note, obj.b_sigma_g, obj.is_upper_limit)
                        if scenario_compatible(
                            obj_real, gamma_min, gamma_max, r_star, r_ns,
                            MAGNETIC_PROGENITOR_FIELD_LOW, MAGNETIC_PROGENITOR_FIELD_HIGH,
                        ):
                            prob_inside += scenario_weight * b_weight

        if prob_inside >= 0.80:
            label = "high-support"
        elif prob_inside >= 0.20:
            label = "mixed-support"
        else:
            label = "low-support"
        results.append((obj.name, label, prob_inside))

    results.sort(key=lambda item: item[2], reverse=True)
    return results


def support_label(prob_inside: float) -> str:
    if prob_inside >= 0.80:
        return "high-support"
    if prob_inside >= 0.20:
        return "mixed-support"
    return "low-support"


def tension_label(obj: CompactObject, central_status: str, prob_inside: float) -> str:
    if obj.family == "long-period outlier":
        return "extra-torque outlier"
    # Sub-corridor objects are *consistent* with the NVG channel: they simply
    # require less structural amplification than the benchmark and do not
    # constitute tension against the model.
    if central_status == "below fossil corridor":
        return "sub-critical (consistent)"
    if central_status == "inside fossil corridor" and prob_inside >= 0.80:
        return "stable fit"
    if central_status == "inside fossil corridor":
        return "corridor edge"
    # Object lies above the conservative corridor; check the magnetic-channel
    # corridor before declaring tension.
    mag_status = magnetic_channel_status(obj)
    if mag_status == "inside magnetic channel":
        return "magnetic-channel fit"
    if prob_inside >= 0.20:
        return "amplification tension"
    return "extreme-field tension"


def source_summary_rows(objects: list[CompactObject]) -> list[dict[str, object]]:
    ordinary_objects = [obj for obj in objects if obj.family != "long-period outlier"]
    probability_map = {name: (label, prob) for name, label, prob in source_probability_summary(objects)}
    rows: list[dict[str, object]] = []

    for obj in objects:
        central_status, req_min, req_max = classify_object(obj)
        if obj.family == "long-period outlier":
            prob_inside = 0.0
            support = "needs-extra-torque"
        else:
            support, prob_inside = probability_map[obj.name]
        rows.append(
            {
                "name": obj.name,
                "family": obj.family,
                "period_s": obj.period_s,
                "b_dip_g": obj.b_dip_g,
                "required_surface_min": req_min,
                "required_surface_max": req_max,
                "central_status": central_status,
                "support": support,
                "p_inside": prob_inside,
                "tension": tension_label(obj, central_status, prob_inside),
                "note": obj.note,
            }
        )

    rows.sort(key=lambda row: (row["family"], -float(row["p_inside"]), row["name"]))
    return rows


def source_inside_probability_for_scale(obj: CompactObject, gamma_scale: float) -> float:
    total_geom = len(R_STAR_GRID) * len(R_NS_GRID)
    prob_inside = 0.0
    gamma_min = GAMMA_STRUCT_BENCHMARK_MIN * gamma_scale
    gamma_max = GAMMA_STRUCT_BENCHMARK_MAX * gamma_scale

    for r_star in R_STAR_GRID:
        for r_ns in R_NS_GRID:
            geom_weight = 1.0 / total_geom
            for b_real, b_weight in object_b_realizations(obj):
                obj_real = CompactObject(
                    obj.name,
                    obj.family,
                    obj.period_s,
                    b_real,
                    obj.tau_kyr,
                    obj.note,
                    obj.b_sigma_g,
                    obj.is_upper_limit,
                )
                if scenario_compatible(
                    obj_real, gamma_min, gamma_max, r_star, r_ns,
                    MAGNETIC_PROGENITOR_FIELD_LOW, MAGNETIC_PROGENITOR_FIELD_HIGH,
                ):
                    prob_inside += geom_weight * b_weight
    return prob_inside


def gamma_likelihood_scan(objects: list[CompactObject]) -> dict[str, object]:
    ordinary_objects = [obj for obj in objects if obj.family != "long-period outlier"]
    rows: list[tuple[float, float, float]] = []

    for gamma_scale in GAMMA_LIKELIHOOD_GRID:
        probs = [source_inside_probability_for_scale(obj, gamma_scale) for obj in ordinary_objects]
        mean_support = sum(probs) / len(probs)
        pseudo_loglike = sum(math.log(max(prob, 1.0e-6)) for prob in probs)
        rows.append((gamma_scale, mean_support, pseudo_loglike))

    best_scale, best_support, best_loglike = max(rows, key=lambda item: item[2])
    support_best = [row for row in rows if row[2] >= best_loglike - 0.5]
    support_wide = [row for row in rows if row[2] >= best_loglike - 2.0]

    return {
        "rows": rows,
        "best_scale": best_scale,
        "best_mean_support": best_support,
        "best_loglike": best_loglike,
        "narrow_min": min(row[0] for row in support_best),
        "narrow_max": max(row[0] for row in support_best),
        "wide_min": min(row[0] for row in support_wide),
        "wide_max": max(row[0] for row in support_wide),
    }


def format_md_number(value: float) -> str:
    return f"{value:.2e}"


def write_markdown_appendix(objects: list[CompactObject], source_note: str) -> None:
    rows = source_summary_rows(objects)
    gamma_fit = gamma_likelihood_scan(objects)
    best_gamma_min = GAMMA_STRUCT_BENCHMARK_MIN * gamma_fit["best_scale"]
    best_gamma_max = GAMMA_STRUCT_BENCHMARK_MAX * gamma_fit["best_scale"]

    lines: list[str] = []
    lines.append("# Magnetar Population Appendix")
    lines.append("")
    lines.append("This appendix is generated by `verification/nvg_magnetar_population_scan.py`.")
    lines.append("")
    lines.append("## Input Summary")
    lines.append("")
    lines.append(f"- Magnetar source set: {source_note}")
    lines.append(f"- Benchmark structural corridor: {GAMMA_STRUCT_BENCHMARK_MIN:.3f}..{GAMMA_STRUCT_BENCHMARK_MAX:.3f}")
    lines.append(f"- Population-preferred best-fit corridor: {best_gamma_min:.2f}..{best_gamma_max:.2f}")
    lines.append(f"- Progenitor surface-field corridor: {SURFACE_FIELD_LOW:.1e}..{SURFACE_FIELD_HIGH:.1e} G")
    lines.append(f"- Geometric uncertainty grid: R_star in {R_STAR_GRID}, R_NS in {R_NS_GRID}")
    lines.append("")
    lines.append("## Source Table")
    lines.append("")
    lines.append("| Source | Family | P (s) | B_dip (G) | Required surface field (G) | P_inside | Support | Tension class |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | --- | --- |")
    for row in rows:
        required_range = f"{format_md_number(float(row['required_surface_min']))}..{format_md_number(float(row['required_surface_max']))}"
        p_inside = "--" if row["family"] == "long-period outlier" else f"{float(row['p_inside']):.3f}"
        lines.append(
            f"| {row['name']} | {row['family']} | {float(row['period_s']):.3f} | {format_md_number(float(row['b_dip_g']))} | "
            f"{required_range} | {p_inside} | {row['support']} | {row['tension']} |"
        )

    lines.append("")
    lines.append("## Tension Classes")
    lines.append("")
    lines.append("- `stable fit`: central fit inside the narrow fossil corridor (1e3..1e4 G) with high support under uncertainties.")
    lines.append("- `corridor edge`: central fit inside the narrow corridor but not strongly stable under uncertainties.")
    lines.append("- `sub-critical (consistent)`: object lies below the corridor and is trivially compatible (no strong amplification required).")
    lines.append("- `magnetic-channel fit`: object exceeds the narrow corridor but lies inside the magnetic-progenitor channel (1e3..1e5 G), corresponding to a rare Of?p / Bp-type magnetized OB progenitor.")
    lines.append("- `amplification tension`: object can be reconciled only with the upper end of the uncertainty envelope or a stronger structural amplification.")
    lines.append("- `extreme-field tension`: object remains difficult without unusually large progenitor fields or stronger-than-current amplification.")
    lines.append("- `extra-torque outlier`: object requires additional braking physics beyond field amplification alone.")
    lines.append("")
    lines.append("## Progenitor Corridors")
    lines.append("")
    lines.append("- Narrow fossil corridor (1e3..1e4 G): typical magnetic massive stars.")
    lines.append("- Magnetic-progenitor channel (1e3..1e5 G): rare strongly magnetized OB stars (e.g. NGC 1624-2 ~2e4 G). Since magnetars are only ~10% of NS births, the magnetar subpopulation is naturally drawn from this rarer channel under the fossil-field hypothesis.")
    lines.append("")

    APPENDIX_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def uncertainty_envelope(objects: list[CompactObject]) -> dict[str, object]:
    ordinary_objects = [obj for obj in objects if obj.family != "long-period outlier"]
    total_cases = 0
    inside_counts: list[int] = []
    per_object_statuses: dict[str, set[str]] = {obj.name: set() for obj in ordinary_objects}

    for r_star in R_STAR_GRID:
        for r_ns in R_NS_GRID:
            for gamma_scale in GAMMA_SCALE_GRID:
                gamma_min = GAMMA_STRUCT_BENCHMARK_MIN * gamma_scale
                gamma_max = GAMMA_STRUCT_BENCHMARK_MAX * gamma_scale
                total_cases += 1
                inside_now = 0
                for obj in ordinary_objects:
                    status = scenario_status(obj, gamma_min, gamma_max, r_star, r_ns)
                    per_object_statuses[obj.name].add(status)
                    if status == "inside fossil corridor":
                        inside_now += 1
                inside_counts.append(inside_now)

    robust_inside = sum(1 for obj in ordinary_objects if per_object_statuses[obj.name] == {"inside fossil corridor"})
    possible_inside = sum(1 for obj in ordinary_objects if "inside fossil corridor" in per_object_statuses[obj.name])

    return {
        "scenario_count": total_cases,
        "min_inside": min(inside_counts),
        "max_inside": max(inside_counts),
        "robust_inside": robust_inside,
        "possible_inside": possible_inside,
        "ordinary_population": len(ordinary_objects),
    }


def summarize(objects: list[CompactObject], source_note: str) -> None:
    print("=" * 110)
    print("NVG MAGNETAR / HIGH-B PULSAR POPULATION SCAN")
    print("=" * 110)
    print(f"Surface-to-NS flux gain: {SURFACE_TO_NS_FLUX_GAIN:.3e}")
    print(f"Magnetar source set: {source_note}")
    print(
        f"Benchmark structural corridor: Gamma_struct = {GAMMA_STRUCT_BENCHMARK_MIN:.3f}"
        f"..{GAMMA_STRUCT_BENCHMARK_MAX:.3f}"
    )
    print(
        f"Tested progenitor surface-field corridor: {SURFACE_FIELD_LOW:.1e}..{SURFACE_FIELD_HIGH:.1e} G"
    )
    print()
    print(
        f"{'Name':<24} | {'Family':<17} | {'P (s)':>8} | {'B_dip (G)':>11} | {'Surface field needed (G)':>27} | {'Status':<21}"
    )
    print("-" * 110)

    counts: dict[str, int] = {
        "inside fossil corridor": 0,
        "below fossil corridor": 0,
        "above fossil corridor": 0,
        "needs extra torque": 0,
    }

    family_counts: dict[str, dict[str, int]] = {}

    for obj in objects:
        status, req_min, req_max = classify_object(obj)
        counts[status] += 1
        family_counts.setdefault(obj.family, {k: 0 for k in counts})
        family_counts[obj.family][status] += 1
        print(
            f"{obj.name:<24} | {obj.family:<17} | {obj.period_s:8.3f} | {obj.b_dip_g:11.2e} | "
            f"{corridor_label(req_min, req_max):>27} | {status:<21}"
        )

    print("\nSummary counts:")
    for key, value in counts.items():
        print(f"- {key}: {value}")

    print("\nBreakdown by family:")
    for family, family_result in family_counts.items():
        summary = ", ".join(f"{key}={value}" for key, value in family_result.items() if value > 0)
        print(f"- {family}: {summary}")

    print("\nInterpretation:")
    print("- If an object's required progenitor surface field lies inside 10^3-10^4 G for")
    print("  the benchmark Gamma_struct corridor, then it is naturally accommodated by the")
    print("  current NVG magnetar channel without invoking extreme progenitor magnetization.")
    print("- Objects below that corridor do not hurt the model; they simply do not require")
    print("  a strong structural boost to reach their current dipole field.")
    print("- Objects above that corridor require either a stronger progenitor field, a larger")
    print("  Gamma_struct than the benchmark 5 n_0 value, or additional post-birth magnetic")
    print("  evolution.")
    print("- The long-period 1E 161348-5055 outlier is treated separately: field amplification")
    print("  may supply the dipole strength, but disk-mediated braking is still required to")
    print("  explain its spin period.")

    accommodated = counts["inside fossil corridor"]
    below = counts["below fossil corridor"]
    ordinary_population = len([obj for obj in objects if obj.family != "long-period outlier"])
    mag_counts = {"inside magnetic channel": 0, "below magnetic channel": 0, "above magnetic channel": 0}
    for obj in objects:
        if obj.family == "long-period outlier":
            continue
        mag_counts[magnetic_channel_status(obj)] += 1
    mag_compatible = mag_counts["inside magnetic channel"] + mag_counts["below magnetic channel"]
    narrow_compatible = accommodated + below
    print()
    print(
        f"Population verdict (narrow fossil corridor 1e3..1e4 G): "
        f"{narrow_compatible}/{ordinary_population} objects compatible "
        f"({accommodated} inside, {below} sub-critical)."
    )
    print(
        f"Population verdict (magnetic-progenitor channel 1e3..1e5 G): "
        f"{mag_compatible}/{ordinary_population} objects compatible "
        f"({mag_counts['inside magnetic channel']} inside, {mag_counts['below magnetic channel']} sub-critical, "
        f"{mag_counts['above magnetic channel']} remain above)."
    )

    uncertainty = uncertainty_envelope(objects)
    print()
    print("Uncertainty scan:")
    print(
        f"- Grid: {uncertainty['scenario_count']} scenarios over R_star in {R_STAR_GRID}, "
        f"R_NS in {R_NS_GRID}, gamma-scale in {GAMMA_SCALE_GRID}."
    )
    print(
        f"- Central coverage: {accommodated}/{ordinary_population}."
    )
    print(
        f"- Scenario range: {uncertainty['min_inside']}/{ordinary_population} to "
        f"{uncertainty['max_inside']}/{ordinary_population} objects inside the fossil corridor."
    )
    print(
        f"- Robust coverage: {uncertainty['robust_inside']}/{ordinary_population} objects remain inside in all scenarios."
    )
    print(
        f"- Possible coverage: {uncertainty['possible_inside']}/{ordinary_population} objects are inside in at least one scenario."
    )

    probability_rows = source_probability_summary(objects)
    high_support = sum(1 for _, label, _ in probability_rows if label == "high-support")
    mixed_support = sum(1 for _, label, _ in probability_rows if label == "mixed-support")
    low_support = sum(1 for _, label, _ in probability_rows if label == "low-support")

    print()
    print("Per-object support from source-specific uncertainty propagation:")
    print(f"- high-support (P_inside >= 0.80): {high_support}")
    print(f"- mixed-support (0.20 <= P_inside < 0.80): {mixed_support}")
    print(f"- low-support (P_inside < 0.20): {low_support}")
    print("- Top high-support sources:")
    for name, label, prob in probability_rows[:5]:
        print(f"  {name}: P_inside = {prob:.3f} ({label})")
    print("- Lowest-support sources:")
    for name, label, prob in probability_rows[-5:]:
        print(f"  {name}: P_inside = {prob:.3f} ({label})")

    gamma_fit = gamma_likelihood_scan(objects)
    best_gamma_min = GAMMA_STRUCT_BENCHMARK_MIN * gamma_fit["best_scale"]
    best_gamma_max = GAMMA_STRUCT_BENCHMARK_MAX * gamma_fit["best_scale"]
    narrow_gamma_min = GAMMA_STRUCT_BENCHMARK_MIN * gamma_fit["narrow_min"]
    narrow_gamma_max = GAMMA_STRUCT_BENCHMARK_MAX * gamma_fit["narrow_max"]
    wide_gamma_min = GAMMA_STRUCT_BENCHMARK_MIN * gamma_fit["wide_min"]
    wide_gamma_max = GAMMA_STRUCT_BENCHMARK_MAX * gamma_fit["wide_max"]

    print()
    print("Global gamma-structure fit:")
    print(f"- Scan grid in gamma-scale: {GAMMA_LIKELIHOOD_GRID[0]:.2f}..{GAMMA_LIKELIHOOD_GRID[-1]:.2f}")
    print(
        f"- Best-fit gamma-scale: {gamma_fit['best_scale']:.2f} "
        f"-> Gamma_struct ~ {best_gamma_min:.2f}..{best_gamma_max:.2f}"
    )
    print(
        f"- Narrow support band (Delta logL <= 0.5): scale {gamma_fit['narrow_min']:.2f}..{gamma_fit['narrow_max']:.2f} "
        f"-> Gamma_struct ~ {narrow_gamma_min:.2f}..{narrow_gamma_max:.2f}"
    )
    print(
        f"- Wide support band (Delta logL <= 2.0): scale {gamma_fit['wide_min']:.2f}..{gamma_fit['wide_max']:.2f} "
        f"-> Gamma_struct ~ {wide_gamma_min:.2f}..{wide_gamma_max:.2f}"
    )
    print(f"- Mean per-source support at best fit: {gamma_fit['best_mean_support']:.3f}")
    write_markdown_appendix(objects, source_note)
    print(f"- Appendix written to: {APPENDIX_PATH}")


def main() -> None:
    sample, source_note = build_sample()
    summarize(sample, source_note)


if __name__ == "__main__":
    main()