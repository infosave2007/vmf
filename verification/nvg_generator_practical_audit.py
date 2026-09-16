#!/usr/bin/env python3
"""Closed-boundary, conventional waste-heat TEG audit.

This calculator intentionally keeps the legacy NVG arithmetic as an audit
record only.  No vacuum/CISS/NVG gain is accepted as an energy source.  The
usable result is a configurable conventional Seebeck heat-engine estimate.
The default temperatures and heat flow are a model scenario, not a rating for
the TEC1-12706 Peltier hardware.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RESULT_PATH = ROOT / "verification" / "nvg_generator_practical_audit_results.json"
REPORT_PATH = ROOT / "verification" / "NVG_GENERATOR_PRACTICAL_AUDIT_RU.md"

LITERATURE_CONTEXT = {
    "title": "Towards tellurium-free thermoelectric modules for power generation from low-grade heat",
    "url": "https://www.nature.com/articles/s41467-021-21391-1",
    "doi": "10.1038/s41467-021-21391-1",
    "note": (
        "Primary-literature context only: a research module reported about 6.5--7.0% "
        "at a temperature difference near 250 K; this is not a rating or validation "
        "of the local TEC1-12706 model."
    ),
}

FORBIDDEN_GAIN_KEYS = {
    "vacuum_gain",
    "vacuum_energy",
    "nvg_gain",
    "ciss_gain",
    "casimir_gain",
    "theta_gain",
    "parametric_gain",
}

DEFAULT_CONFIG: dict[str, float | None] = {
    "hot_k": 450.0,
    "cold_k": 300.0,
    "zt": 0.9,
    "heat_per_module_w": 200.0,
    "contact_derate": 0.4,
    "converter_efficiency": 0.90,
    "auxiliary_w": 0.0,
    "system_auxiliary_w": 0.0,
    "seebeck_v_per_k": 0.055,
    "example_requested_net_w": 10.0,
    "available_heat_w": None,
}


def _finite(value: float) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def _require_positive(name: str, value: float) -> float:
    if not _finite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return float(value)


def assert_no_speculative_gain(value: Any, path: str = "config") -> None:
    """Fail closed if a caller tries to smuggle a speculative gain parameter."""

    if isinstance(value, dict):
        for key, child in value.items():
            normalized = str(key).strip().lower()
            if normalized in FORBIDDEN_GAIN_KEYS:
                raise ValueError(f"unsupported speculative gain key at {path}.{key}")
            assert_no_speculative_gain(child, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            assert_no_speculative_gain(child, f"{path}[{index}]")


def legacy_peltier_audit() -> dict[str, float | str | bool]:
    """Reproduce the declared Peltier arithmetic, with and without free G.

    The old diode expression was proportional to the claimed output.  It is
    retained as an explicitly output-dependent diagnostic, while the
    no-gain loss floor contains only drive losses that do not depend on that
    unsupported output claim.
    """

    n_pellets = 254.0
    pellet_width_m = 1.4e-3
    active_area_m2 = n_pellets * 2.0 * pellet_width_m**2
    carriers = 5.0e15 * active_area_m2
    energy_cycle_j = carriers * (0.15 * 1.60217663e-19) * 0.05
    frequency_hz = 150.0e3
    power_without_gain_w = energy_cycle_j * frequency_hz
    legacy_gain = 4.5e5
    power_with_gain_w = power_without_gain_w * legacy_gain

    voltage_v = 12.0
    resistance_ohm = 2.0
    duty = 0.35
    current_peak_a = voltage_v / resistance_ohm
    current_rms_a = current_peak_a * math.sqrt(duty)
    p_ohmic_w = current_rms_a**2 * resistance_ohm
    p_mosfet_w = current_rms_a**2 * 0.015
    transition_s = 2.2 * 10.0 * 7.16e-9
    p_switching_w = 0.5 * 112.0 * current_peak_a * frequency_hz * transition_s
    # Legacy report arithmetic: 1.2 W per 100 W of *claimed* output.  This is
    # not a component rating and must not be folded into an output-independent
    # no-gain floor.
    no_gain_diode_w = power_without_gain_w / 100.0 * 1.2
    output_dependent_diode_w = power_with_gain_w / 100.0 * 1.2
    p_control_w = 1.2
    no_gain_loss_floor_w = p_ohmic_w + p_mosfet_w + p_switching_w + p_control_w
    losses_without_gain_w = no_gain_loss_floor_w + no_gain_diode_w
    losses_with_legacy_gain_w = no_gain_loss_floor_w + output_dependent_diode_w

    return {
        "status": "ARITHMETIC_ONLY_UNSUPPORTED_SOURCE",
        "active_surface_area_m2": active_area_m2,
        "active_carriers": carriers,
        "energy_per_cycle_j": energy_cycle_j,
        "frequency_hz": frequency_hz,
        "power_without_gain_w": power_without_gain_w,
        "legacy_gain": legacy_gain,
        "power_with_legacy_gain_w": power_with_gain_w,
        "input_voltage_v": voltage_v,
        "current_rms_a": current_rms_a,
        # Keep the historical aggregate name as the loss at the legacy-G
        # point, but expose the decomposition needed for an honest audit.
        "modeled_losses_w": losses_with_legacy_gain_w,
        "no_gain_loss_floor_w": no_gain_loss_floor_w,
        "diode_loss_without_gain_w": no_gain_diode_w,
        "legacy_output_dependent_diode_loss_w": output_dependent_diode_w,
        "modeled_losses_without_gain_w": losses_without_gain_w,
        "modeled_losses_with_legacy_gain_w": losses_with_legacy_gain_w,
        "net_without_gain_floor_w": power_without_gain_w - no_gain_loss_floor_w,
        "net_without_gain_w": power_without_gain_w - losses_without_gain_w,
        "net_with_legacy_gain_w": power_with_gain_w - losses_with_legacy_gain_w,
        "loss_semantics": (
            "no_gain_loss_floor is output-independent; diode terms are retained "
            "separately because the legacy expression scales with claimed output"
        ),
        "dimensional_basis": "energy_per_cycle_j [J] times frequency_hz [1/s] gives power [W]",
        "energy_source": "none_declared; G is not accepted as a source",
        "target_dependent": False,
    }


def ioffe_efficiency(hot_k: float, cold_k: float, zt: float, derate: float) -> dict[str, float]:
    hot_k = _require_positive("hot_k", hot_k)
    cold_k = _require_positive("cold_k", cold_k)
    if hot_k <= cold_k:
        raise ValueError("hot_k must exceed cold_k")
    if not _finite(zt) or zt < 0.0:
        raise ValueError("zt must be finite and non-negative")
    if not _finite(derate) or not 0.0 < derate <= 1.0:
        raise ValueError("derate must be in (0, 1]")
    carnot = (hot_k - cold_k) / hot_k
    root = math.sqrt(1.0 + zt)
    ioffe = (root - 1.0) / (root + cold_k / hot_k)
    device_efficiency = carnot * ioffe * derate
    return {
        "hot_k": hot_k,
        "cold_k": cold_k,
        "temperature_difference_k": hot_k - cold_k,
        "zt": float(zt),
        "carnot_efficiency": carnot,
        "ioffe_factor": ioffe,
        "device_efficiency": device_efficiency,
    }


def conventional_teg(
    *,
    hot_k: float = 450.0,
    cold_k: float = 300.0,
    zt: float = 0.9,
    heat_per_module_w: float = 200.0,
    contact_derate: float = 0.4,
    converter_efficiency: float = 0.90,
    auxiliary_w: float = 0.0,
    seebeck_v_per_k: float = 0.055,
) -> dict[str, float | str | bool]:
    """Estimate one conventional TEG module.

    ``auxiliary_w`` is deliberately per module.  System-level auxiliaries are
    applied by :func:`module_requirement`, where the parallel module count is
    known; this prevents silently multiplying a shared fan, pump or controller.
    """

    eff = ioffe_efficiency(hot_k, cold_k, zt, contact_derate)
    heat = _require_positive("heat_per_module_w", heat_per_module_w)
    if not _finite(converter_efficiency) or not 0.0 < converter_efficiency <= 1.0:
        raise ValueError("converter_efficiency must be in (0, 1]")
    if not _finite(auxiliary_w) or auxiliary_w < 0.0:
        raise ValueError("auxiliary_w must be finite and non-negative")
    if not _finite(seebeck_v_per_k) or seebeck_v_per_k <= 0.0:
        raise ValueError("seebeck_v_per_k must be finite and positive")
    gross = heat * eff["device_efficiency"]
    converter_out = gross * float(converter_efficiency)
    net = converter_out - float(auxiliary_w)
    open_circuit_v = float(seebeck_v_per_k) * eff["temperature_difference_k"]
    return {
        "status": "CONVENTIONAL_MODEL_REQUIRES_MEASURED_MODULE_AND_CONTACTS",
        **eff,
        "heat_per_module_w": heat,
        "contact_derate": float(contact_derate),
        "converter_efficiency": float(converter_efficiency),
        "auxiliary_w": float(auxiliary_w),
        "gross_electrical_w": gross,
        "converter_output_w": converter_out,
        "net_electrical_w": net,
        "open_circuit_voltage_v_model": open_circuit_v,
        "energy_source": "pre-existing waste heat at hot_k; heat sink at cold_k",
        "auxiliary_scope": "per_module",
        "model_scope": "scenario arithmetic; not a manufacturer rating",
        "manufacturer_rating_used": False,
        "independent_evidence_weight": 0.0,
    }


def sensitivity_grid() -> dict[str, Any]:
    rows: list[dict[str, float]] = []
    for hot_k in (350.0, 400.0, 450.0):
        for zt in (0.5, 0.9, 1.2):
            for derate in (0.25, 0.4, 0.6):
                case = conventional_teg(
                    hot_k=hot_k,
                    cold_k=300.0,
                    zt=zt,
                    heat_per_module_w=200.0,
                    contact_derate=derate,
                    converter_efficiency=1.0,
                    auxiliary_w=0.0,
                )
                rows.append(
                    {
                        "hot_k": hot_k,
                        "cold_k": 300.0,
                        "zt": zt,
                        "contact_derate": derate,
                        "gross_electrical_w": float(case["gross_electrical_w"]),
                    }
                )
    powers = [row["gross_electrical_w"] for row in rows]
    return {
        "grid_definition": {
            "hot_k": [350.0, 400.0, 450.0],
            "cold_k": [300.0],
            "zt": [0.5, 0.9, 1.2],
            "contact_derate": [0.25, 0.4, 0.6],
            "heat_per_module_w": 200.0,
            "converter_efficiency": 1.0,
            "auxiliary_w": 0.0,
            "system_auxiliary_w": 0.0,
        },
        "rows": rows,
        "gross_power_envelope_w": {"min": min(powers), "max": max(powers)},
        "selection_rule": "Cartesian parameter grid; no row was fitted to a target output",
        "target_independent": True,
        "power_basis": "gross_electrical_w_per_module",
        "requested_target_not_used": True,
    }


def module_requirement(
    requested_net_w: float,
    case: dict[str, float | str],
    *,
    available_heat_w: float | None = None,
    system_auxiliary_w: float = 0.0,
) -> dict[str, float | int | str | bool]:
    if not _finite(requested_net_w) or requested_net_w < 0.0:
        raise ValueError("requested_net_w must be finite and non-negative")
    per_module_net = float(case["net_electrical_w"])
    heat_per_module = float(case["heat_per_module_w"])
    if not _finite(per_module_net):
        raise ValueError("case net_electrical_w must be finite")
    if not _finite(heat_per_module) or heat_per_module <= 0.0:
        raise ValueError("case heat_per_module_w must be finite and positive")
    if not _finite(system_auxiliary_w) or system_auxiliary_w < 0.0:
        raise ValueError("system_auxiliary_w must be finite and non-negative")
    system_aux = float(system_auxiliary_w)
    if per_module_net <= 0.0:
        return {
            "status": "INFEASIBLE_NONPOSITIVE_NET_PER_MODULE",
            "feasible": False,
            "requested_net_w": float(requested_net_w),
            "modules": 0,
            "thermal_input_w": 0.0,
            "system_auxiliary_w": system_aux,
            "target_dependent": True,
        }
    required_module_output = float(requested_net_w) + system_aux
    modules = int(math.ceil(required_module_output / per_module_net)) if required_module_output else 0
    thermal = modules * heat_per_module
    per_module_aux = float(case.get("auxiliary_w", 0.0))
    if not _finite(per_module_aux) or per_module_aux < 0.0:
        raise ValueError("case auxiliary_w must be finite and non-negative")
    total_auxiliary = modules * per_module_aux + system_aux
    predicted_net = modules * per_module_net - system_aux
    if available_heat_w is not None:
        if not _finite(available_heat_w) or available_heat_w < 0.0:
            raise ValueError("available_heat_w must be finite and non-negative")
        if thermal > float(available_heat_w) + 1.0e-12:
            return {
                "status": "INFEASIBLE_HEAT_BUDGET",
                "feasible": False,
                "requested_net_w": float(requested_net_w),
                "modules": modules,
                "thermal_input_w": thermal,
                "system_auxiliary_w": system_aux,
                "auxiliary_total_w": total_auxiliary,
                "predicted_net_w": predicted_net,
                "available_heat_w": float(available_heat_w),
                "shortfall_heat_w": thermal - float(available_heat_w),
                "module_topology": "parallel; thermal input sums across modules",
                "target_dependent": True,
            }
    return {
        "status": "FEASIBLE_IF_HEAT_AND_SINK_ARE_AVAILABLE",
        "feasible": True,
        "requested_net_w": float(requested_net_w),
        "modules": modules,
        "thermal_input_w": thermal,
        "predicted_net_w": predicted_net,
        "per_module_net_w": per_module_net,
        "system_auxiliary_w": system_aux,
        "auxiliary_total_w": total_auxiliary,
        "module_topology": "parallel; thermal input sums across modules",
        "target_dependent": True,
        "purpose": "illustrative target sizing, not independent evidence",
    }


def family_classifications() -> list[dict[str, Any]]:
    return [
        {
            "family": "peltier_vacuum",
            "classification": "REJECTED_UNSUPPORTED_SOURCE",
            "independent_evidence_weight": 0.0,
            "reason": "legacy G is an unmeasured multiplier; no external heat/work boundary",
            "dimension_check": "energy_per_cycle_j times frequency_hz has units of W; G is dimensionless but unmeasured",
            "empirical_status": "no calorimetric source measurement; shared assumptions do not count as evidence",
            "salvage": "electrically driven TEC calorimetry with hot/cold reservoirs and a no-gain control",
        },
        {
            "family": "magnetic_parametric",
            "classification": "REJECTED_STORED_FIELD_TURNOVER",
            "independent_evidence_weight": 0.0,
            "reason": "cycling field energy and pump work are not a net source",
            "dimension_check": "field energy is J and turnover rate is Hz; output cannot exceed measured electrical/mechanical drive",
            "empirical_status": "coil, switching and recharge losses are not independently metered",
            "salvage": "low-voltage magnetometry and a complete field-energy input/output ledger",
        },
        {
            "family": "hydrovoltaic_ciss",
            "classification": "SENSOR_SCALE_ONLY_UNVERIFIED_BOOST",
            "independent_evidence_weight": 0.0,
            "reason": "evaporation/chemical input and CISS multipliers are unclosed",
            "dimension_check": "voltage times current is W only after charge, chemical and mass flux are measured",
            "empirical_status": "CISS/polarization multipliers are assumptions, not independent measurements",
            "salvage": "sealed evaporation/materials test with chemical, mass and electrical calorimetry",
        },
        {
            "family": "theta_haloscope",
            "classification": "DETECTOR_OR_NULL_EXPERIMENT_ONLY",
            "independent_evidence_weight": 0.0,
            "reason": "prospective signal channel, not an energy source",
            "dimension_check": "reported RF power must follow calibrated coupling, Q, bandwidth and temperature units",
            "empirical_status": "static geometry/calibration tables do not establish a theta signal",
            "predicted_signal_vs_noise": "expected receiver signal is below conventional thermal/electronics noise until blinded injection and recovery succeed",
            "salvage": "blind haloscope/null experiment with measured Q, loss tangent, Tsys and synthetic-tone recovery",
        },
        {
            "family": "antigravity_theta",
            "classification": "REJECTED_DIMENSIONAL_FORCE_ANSATZ",
            "independent_evidence_weight": 0.0,
            "reason": "dimensionally inconsistent force ratio and unsupported coupling; ordinary matter channel is null",
            "dimension_check": "dimensionless couplings cannot be divided by SI G*m^2 without a consistent natural-unit conversion",
            "empirical_status": "no force/torque anomaly survives magnetic, species and reversal controls",
            "predicted_signal_vs_noise": "corrected force/torque estimate is below conventional magnetic and torsion systematics",
            "salvage": "species-ratio comagnetometer or torsion null measurement with blind sign reversal",
        },
        {
            "family": "high_speed_magnetic",
            "classification": "REJECTED_DRIVE_LIMITED_SYSTEMATIC",
            "independent_evidence_weight": 0.0,
            "reason": "high-speed magnetic/theta link is bandwidth- and drive-limited; shared motion drives the background",
            "dimension_check": "bit rate, field, torque and power must use measured bandwidth and SI units",
            "empirical_status": "predicted anomaly is not separated from ordinary magnetic pickup, vibration or thermal drift",
            "predicted_signal_vs_noise": "conventional field/torque backgrounds exceed the putative signal by many orders until cancellation is demonstrated",
            "salvage": "phase detector/comagnetometer with ±source reversal, independent field monitor and preregistered threshold",
        },
        {
            "family": "conventional_seebeck_teg",
            "classification": "CONVENTIONAL_HEAT_ENGINE_MODEL",
            "independent_evidence_weight": 0.0,
            "reason": "local arithmetic is unmeasured model output; accept only after measured hot-side heat, cold sink and parasitics",
            "dimension_check": "heat flow in W times bounded efficiency gives gross electrical W; all auxiliaries are subtracted",
            "empirical_status": "manufacturer-rated generator performance and contacts are not supplied",
            "salvage": "calibrated waste-heat TEG, calorimetry and zero/reversed-gradient controls",
        },
    ]


def calculate(config_overrides: dict[str, float] | None = None) -> dict[str, Any]:
    config = dict(DEFAULT_CONFIG)
    if config_overrides:
        unknown = sorted(set(config_overrides) - set(config))
        if unknown:
            raise ValueError(f"unknown calculation configuration: {', '.join(unknown)}")
        config.update(config_overrides)
    assert_no_speculative_gain(config)
    teg = conventional_teg(
        **{
            k: v
            for k, v in config.items()
            if k not in {"example_requested_net_w", "available_heat_w", "system_auxiliary_w"}
        }
    )
    requirements = module_requirement(
        float(config["example_requested_net_w"]),
        teg,
        available_heat_w=config["available_heat_w"],
        system_auxiliary_w=float(config["system_auxiliary_w"] or 0.0),
    )
    return {
        "schema": "nvg-generator-practical-audit/v1",
        "status": "PASS_CONVENTIONAL_ONLY",
        "config": config,
        "legacy_peltier": legacy_peltier_audit(),
        "conventional_teg": teg,
        "sensitivity": sensitivity_grid(),
        "example_module_requirement": requirements,
        "families": family_classifications(),
        "literature_context": LITERATURE_CONTEXT,
        "evidence_policy": "No local numerical scenario receives independent-evidence weight without a measured source-to-load balance",
        "safety": {
            "mains_or_high_voltage_build": False,
            "required_controls": [
                "guarded heat-flow measurement",
                "cold-sink temperature and auxiliary-power logging",
                "zero-gradient and reversed-gradient controls",
            ],
        },
    }


def russian_report(data: dict[str, Any]) -> str:
    teg = data["conventional_teg"]
    legacy = data["legacy_peltier"]
    env = data["sensitivity"]["gross_power_envelope_w"]
    req = data["example_module_requirement"]
    return f"""# Практический аудит генератора NVG

## Границы и классификация

Калькулятор не принимает вакуумную, CISS, Casimir-, theta- или иную NVG-мощность
как источник энергии. Итоговая классификация `PASS_CONVENTIONAL_ONLY` означает
только то, что разрешён обычный тепловой сценарий; это не публикационный статус
и не подтверждение новой физики. Все локальные числа являются арифметикой модели:
в JSON вес независимого локального доказательства равен нулю, включая
conventional Seebeck TEG. Литературный контекст ниже не меняет этот вес.

Параметры TEC1-12706 здесь не являются паспортной характеристикой изделия:
`ZT`, тепловой поток, коэффициент Зеебека и derate контактов заданы как
сценарные входы. Нужны модуль, рассчитанный производителем для режима генератора,
и измерение его тепловых контактов.

## Контроль старого утверждения

По литералам `calc_peltier_generator.py`: без свободного множителя `G` получается
`{float(legacy['power_without_gain_w']):.6g} W` на 150 kHz, при заявленном
`G={float(legacy['legacy_gain']):.3g}` — `{float(legacy['power_with_legacy_gain_w']):.3f} W`.
Независимый от заявленного выхода no-gain loss floor драйвера составляет
`{float(legacy['no_gain_loss_floor_w']):.3f} W`; legacy-формула диода добавляет
`{float(legacy['legacy_output_dependent_diode_loss_w']):.3f} W` только как
выходозависимый диагностический член. Поэтому `{float(legacy['net_without_gain_w']):.3f} W`
без `G` уже отрицательны, а `{float(legacy['net_with_legacy_gain_w']):.3f} W`
при `G` не являются доказательством источника. У TEC без горячего и холодного
резервуаров остаётся электрически питаемый нагреватель.

## Спекулятивные каналы и безопасная полезность

Вакуумный/COP-канал использует `J × Hz = W`, но свободный `G` не измерен и не
замыкает баланс. Магнитный и высокоскоростной каналы считают оборот запасённой
энергии выходом, не вычитая возбуждение, перезарядку, переключение и pickup;
сигнал должен быть отделён от обычных полей, вибрации и дрейфа. В CISS/
гидровольтаике `V × A` становится мощностью только после учёта испарения,
ионного/химического потока и заряда; множители поляризации не являются данными.
Антигравитационная формула дополнительно смешивает безразмерную связь с SI
`G m²`, поэтому заявленная сила размерностно недействительна и обычная материя
даёт нулевой канал. Theta/haloscope остаётся приёмником: таблицы `Q`, геометрии
и калибровок — статические входы, а ожидаемый RF-сигнал пока ниже обычного
теплового и электронного шума. Безопасная полезность этих веток — только
калориметрия, магнитометрия/фазовый детектор, материаловедческий тест или
слепой null-эксперимент с заранее заданным порогом.

## Единственный разрешённый энергетический вариант

Внешний источник: уже имеющееся отходящее тепло (например, горячая поверхность
или выхлоп) с `T_h={float(teg['hot_k']):.0f} K`; теплоотвод поддерживает
`T_c={float(teg['cold_k']):.0f} K`. При `ZT={float(teg['zt']):.2f}`, derate
контактов `{float(teg['contact_derate']):.2f}` и тепловом потоке
`{float(teg['heat_per_module_w']):.0f} W/module` модель даёт Carnot
`{float(teg['carnot_efficiency'])*100:.1f}%`, gross `{float(teg['gross_electrical_w']):.2f} W`;
после DC--DC с КПД `{float(teg['converter_efficiency'])*100:.0f}%` net
`{float(teg['net_electrical_w']):.2f} W/module`. Это оценка, а не обещание
производительности. Пример запроса `{float(req['requested_net_w']):.1f} W` требует
`{int(req['modules'])}` параллельных модулей и около `{float(req['thermal_input_w']):.0f} W`
теплового потока; общий auxiliary составляет `{float(req.get('auxiliary_total_w', 0.0)):.2f} W`,
из них системный — `{float(req.get('system_auxiliary_w', 0.0)):.2f} W`.
Потоки тепла суммируются по модулям, но общий вентилятор/насос не умножается.
Этот запрос — target-dependent иллюстрация sizing, не evidence.

Статическая сетка чувствительности без подгонки цели (gross, 200 W/module) даёт
`{float(env['min']):.2f}--{float(env['max']):.2f} W` для 350--450 K,
`ZT=0.5--1.2` и derate 0.25--0.60. Таблица не использует запрошенную цель;
паразитные вентиляторы, насосы и утечки добавляются в net и могут сделать
запрошенную мощность **infeasible**.

## Проверка и контекст

Измерять нужно теплоприток горячей стороны, теплоприёмник, мощность нагрузки,
преобразователь и вспомогательные цепи; повторить при нулевом и обратном
градиенте. Положительный net считается только после полного баланса. В качестве
контекста (не как доказательство локальной модели) указана первичная работа:
[{LITERATURE_CONTEXT['title']}]({LITERATURE_CONTEXT['url']}) — исследовательский
модуль сообщал около 6.5--7.0% при ΔT порядка 250 K.

Классификации всех семейств и воспроизводимая арифметика находятся в JSON рядом
с этим отчётом; ни один локальный численный сценарий не получает независимый
вес доказательства без измеренного баланса.
"""


def write_outputs(data: dict[str, Any], result_path: Path = RESULT_PATH, report_path: Path = REPORT_PATH) -> None:
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path.write_text(russian_report(data), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="write canonical JSON and Russian report")
    parser.add_argument("--json", action="store_true", help="print deterministic JSON to stdout")
    for name, help_text in (
        ("hot_k", "hot-side temperature in K"),
        ("cold_k", "cold-side temperature in K"),
        ("zt", "dimensionless figure of merit"),
        ("heat_per_module_w", "available heat per module in W"),
        ("contact_derate", "contact/mismatch derate in (0,1]"),
        ("converter_efficiency", "DC-DC converter efficiency in (0,1]"),
        ("auxiliary_w", "fan/pump/control load in W"),
        ("system_auxiliary_w", "shared system auxiliary load in W"),
        ("seebeck_v_per_k", "model Seebeck coefficient in V/K"),
        ("example_requested_net_w", "requested net power for module sizing in W"),
        ("available_heat_w", "available waste-heat budget for module sizing in W"),
    ):
        parser.add_argument(f"--{name.replace('_', '-')}", type=float, default=None, help=help_text)
    args = parser.parse_args(argv)
    overrides = {
        key: value
        for key, value in vars(args).items()
        if key in DEFAULT_CONFIG and value is not None
    }
    data = calculate(overrides)
    if args.write:
        write_outputs(data)
    if args.json or not args.write:
        print(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
