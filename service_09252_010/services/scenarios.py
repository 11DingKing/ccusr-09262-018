"""情景预测报告：在基准数据版本上叠加假设参数外推，假设与版本随结果一同落库。

核心规则：
- 假设参数（每度量增长率 growth 或固定值 override）必须齐全且恰好覆盖全部
  指标依赖的度量；缺参即拒绝（422），且任何失败都发生在唯一一次 INSERT 之前，
  事务回滚，绝不留下半成品报告；
- 报告行同时固化 assumptions 与 pins（基准数据版本、指标版本、规则版本），
  读取时可原样复现当时输入；
- recompute 严格按固化的版本与假设重算，核对输入/结果指纹——指标更新、
  规则回滚、迟到数据均不影响旧报告复现。
"""
from __future__ import annotations

import math

from ..domain.conversion import convert_rows, rule_key
from ..domain.errors import (
    NotFoundError,
    StateError,
    ValidationError,
)
from ..domain.fingerprint import fingerprint
from ..domain.formulas import evaluate, formula_measures, validate_formula
from ..domain.models import Principal, ScenarioReport, SnapshotRow
from ..domain.periods import iter_periods, next_periods, period_key
from ..persistence.database import Database
from ..persistence.store import Store
from ..ports import Clock, IdGenerator
from .access import AccessPolicy

_ABSENT = object()  # 期间槽位尚无记录（区别于已记录的缺失值 None）


def _require_number(value: object, label: str) -> float:
    """假设参数取值：必须为有限数值（布尔与 NaN/Inf 拒绝）。"""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValidationError(f"假设参数 {label} 必须为数值，得到 {value!r}")
    if not math.isfinite(value):
        raise ValidationError(f"假设参数 {label} 必须为有限数值")
    return float(value)


def normalize_assumptions(raw: object, needed_measures: set[str]) -> dict:
    """校验并规范化假设参数。

    每个指标依赖的度量都必须恰好给出一条假设：growth（每期增长率）或
    override（预测期固定值）之一；多给、少给、给错形态都拒绝。
    """
    if not isinstance(raw, dict):
        raise ValidationError("assumptions 必须为对象")
    measures = raw.get("measures")
    if not isinstance(measures, dict) or not measures:
        raise ValidationError("assumptions.measures 必须为非空对象")
    unknown = sorted(set(measures) - needed_measures)
    if unknown:
        raise ValidationError(
            "假设参数包含指标未依赖的度量",
            detail={"unknown_measures": unknown},
        )
    missing = sorted(needed_measures - set(measures))
    if missing:
        raise ValidationError(
            "假设参数缺失：以下度量未给出增长率或固定值",
            detail={"missing_measures": missing},
        )
    norm: dict[str, dict[str, float]] = {}
    for measure, entry in measures.items():
        if not isinstance(entry, dict):
            raise ValidationError(f"度量 {measure} 的假设必须为对象")
        keys = set(entry)
        if keys == {"growth"}:
            norm[measure] = {
                "growth": _require_number(entry["growth"], f"{measure}.growth")
            }
        elif keys == {"override"}:
            norm[measure] = {
                "override": _require_number(entry["override"], f"{measure}.override")
            }
        else:
            raise ValidationError(
                f"度量 {measure} 的假设必须且只能包含 growth 或 override 之一"
            )
    return {"measures": norm}


class ScenarioService:
    def __init__(self, db: Database, clock: Clock, ids: IdGenerator) -> None:
        self.db = db
        self.clock = clock
        self.ids = ids

    # ---- 生成（缺参拒绝，参数与版本随结果一次落库）----
    def generate(self, principal: Principal, *, project_id: str | None = None,
                 scenario_name: str | None = None,
                 history_start: str | None = None,
                 history_end: str | None = None,
                 horizon_months: object = None,
                 target_caliber: str | None = None,
                 assumptions: object = None) -> dict:
        """生成情景预测报告。任一必填参数缺失即拒绝，且不写入任何记录。"""
        params = {
            "project_id": project_id,
            "scenario_name": scenario_name,
            "history_start": history_start,
            "history_end": history_end,
            "horizon_months": horizon_months,
            "target_caliber": target_caliber,
            "assumptions": assumptions,
        }
        missing = [k for k, v in params.items()
                   if v is None or (isinstance(v, str) and not v.strip())]
        if missing:
            raise ValidationError("缺少必填参数，已拒绝生成",
                                  detail={"missing": sorted(missing)})
        assert isinstance(project_id, str) and isinstance(history_start, str)
        assert isinstance(history_end, str) and isinstance(target_caliber, str)
        assert isinstance(scenario_name, str)
        # 纯参数校验先行：窗口与预测期数均不依赖数据库
        iter_periods(history_start, history_end)
        if isinstance(horizon_months, bool) \
                or not isinstance(horizon_months, int):
            raise ValidationError("horizon_months 必须为正整数")
        horizon: int = horizon_months
        next_periods(history_end, horizon)  # 期数必须为正

        with self.db.uow() as uow:
            store = Store(uow.conn)
            AccessPolicy(store).require(principal, project_id, "*", "calculate")
            pins, lines, snapshot_rows, assumptions_norm = self._compute(
                store, project_id=project_id, history_start=history_start,
                history_end=history_end, horizon_months=horizon,
                target_caliber=target_caliber, assumptions=assumptions,
                for_reverify=False,
            )
            now = self.clock.now()
            input_payload = {
                "project_id": project_id,
                "scenario_name": scenario_name,
                "history_window": [history_start, history_end],
                "horizon_months": horizon,
                "target_caliber": target_caliber,
                "assumptions": assumptions_norm,
                "pins": pins,
                "snapshot": snapshot_rows,
            }
            report = ScenarioReport(
                id=self.ids.new_id("scenario"),
                project_id=project_id,
                scenario_name=scenario_name,
                baseline_version_no=pins["data_version"],
                history_start=history_start,
                history_end=history_end,
                horizon_months=horizon,
                target_caliber=target_caliber,
                assumptions=assumptions_norm,
                pins=pins,
                lines=lines,
                input_fingerprint=fingerprint(input_payload),
                result_fingerprint=fingerprint(lines),
                created_by=principal.institution_id,
                created_at=now,
            )
            # 全请求唯一的写操作：此前任何失败都令事务回滚，不留半成品
            store.add_scenario_report(report)
        return self._report_dict(report, report.lines, [])

    # ---- 读取（返回完整假设与版本，可复现当时输入）----
    def get(self, principal: Principal, report_id: str) -> dict:
        with self.db.read() as conn:
            store = Store(conn)
            report = store.get_scenario_report(report_id)
            if report is None:
                raise NotFoundError(f"情景预测报告不存在: {report_id}")
            policy = AccessPolicy(store)
            categories = sorted({l["category"] for l in report.lines})
            visible = set(policy.granted_categories(
                principal, report.project_id, "view", categories))
        lines = [l for l in report.lines if l["category"] in visible]
        redacted = sorted(c for c in categories if c not in visible)
        return self._report_dict(report, lines, redacted)

    def list(self, principal: Principal, project_id: str) -> list[dict]:
        with self.db.read() as conn:
            store = Store(conn)
            reports = store.list_scenario_reports(project_id)
            policy = AccessPolicy(store)
            result = []
            for report in reports:
                categories = sorted({l["category"] for l in report.lines})
                visible = set(policy.granted_categories(
                    principal, project_id, "view", categories))
                lines = [l for l in report.lines if l["category"] in visible]
                redacted = sorted(c for c in categories if c not in visible)
                result.append(self._report_dict(report, lines, redacted))
        return result

    # ---- 重算（按固化的假设与版本复现旧输入）----
    def recompute(self, principal: Principal, report_id: str) -> dict:
        """按报告固化的假设参数与版本重算，核对输入/结果指纹。

        指标定义更新、规则回滚或迟到数据均不影响——全部输入按 pins 取版本，
        假设参数取落库时的原文。
        """
        with self.db.read() as conn:
            store = Store(conn)
            report = store.get_scenario_report(report_id)
            if report is None:
                raise NotFoundError(f"情景预测报告不存在: {report_id}")
            AccessPolicy(store).require(
                principal, report.project_id, "*", "view")
            pins, lines, snapshot_rows, _ = self._compute(
                store, project_id=report.project_id,
                history_start=report.history_start,
                history_end=report.history_end,
                horizon_months=report.horizon_months,
                target_caliber=report.target_caliber,
                assumptions=report.assumptions,
                for_reverify=True,
                pins_indicators=report.pins["indicators"],
                pins_rules=report.pins["rules"],
                data_version_no=report.baseline_version_no,
            )
            input_fp = fingerprint({
                "project_id": report.project_id,
                "scenario_name": report.scenario_name,
                "history_window": [report.history_start, report.history_end],
                "horizon_months": report.horizon_months,
                "target_caliber": report.target_caliber,
                "assumptions": report.assumptions,
                "pins": pins,
                "snapshot": snapshot_rows,
            })
            result_fp = fingerprint(lines)
        return {
            "report_id": report_id,
            "input_match": input_fp == report.input_fingerprint,
            "result_match": result_fp == report.result_fingerprint,
            "stored_result_fingerprint": report.result_fingerprint,
            "recomputed_result_fingerprint": result_fp,
            "reproduced_inputs": {
                "scenario_name": report.scenario_name,
                "assumptions": report.assumptions,
                "pins": report.pins,
                "baseline_version_no": report.baseline_version_no,
                "history_window": [report.history_start, report.history_end],
                "horizon_months": report.horizon_months,
                "target_caliber": report.target_caliber,
            },
        }

    # ---- 内部：外推计算（生成与重算共用，差异仅在版本来源）----
    @staticmethod
    def _needed_measures(store: Store,
                         versions: dict[str, int] | None) -> set[str]:
        """指标依赖的度量集合。versions 为 None 时取各指标最新版本。"""
        needed: set[str] = set()
        if versions is None:
            indicators = store.list_indicators()
            if not indicators:
                raise ValidationError("尚未登记任何指标，无法生成情景预测")
            for indicator in indicators:
                version = store.latest_indicator_version(indicator.code)
                assert version is not None
                validate_formula(version.formula)
                needed.update(formula_measures(version.formula))
        else:
            for code, ver in versions.items():
                version = store.get_indicator_version(code, ver)
                if version is None:
                    raise StateError(f"重算失败：指标 {code} v{ver} 已不存在")
                needed.update(formula_measures(version.formula))
        return needed

    def _compute(self, store: Store, *, project_id: str, history_start: str,
                 history_end: str, horizon_months: int, target_caliber: str,
                 assumptions: dict, for_reverify: bool,
                 pins_indicators: dict[str, int] | None = None,
                 pins_rules: dict[str, int] | None = None,
                 data_version_no: int | None = None,
                 ) -> tuple[dict, list[dict], list[dict], dict]:
        """外推计算。返回 (pins, lines, snapshot_rows, 规范化假设)，不写库。"""
        if data_version_no is None:
            seq = store.latest_batch_seq(project_id)
            if seq is None:
                raise ValidationError("项目尚无任何数据版本，无法生成情景预测")
        else:
            seq = data_version_no

        lo, hi = period_key(history_start), period_key(history_end)
        snapshot_rows: list[dict] = []
        for obs in store.snapshot(project_id, seq):
            if obs.retracted or not (lo <= period_key(obs.period) <= hi):
                continue
            snapshot_rows.append({
                "measure": obs.measure,
                "period": obs.period,
                "caliber": obs.caliber,
                "value": obs.value,
                "evidence_id": obs.evidence_id,
            })

        # 换算规则：生成取当前生效版本，重算严格按固化版本
        if for_reverify:
            assert pins_rules is not None
            rules_params: dict[str, tuple[float, float]] = {}
            for key, ver in pins_rules.items():
                rule = store.get_rule_by_version(key, ver)
                if rule is None:
                    raise StateError(f"重算失败：规则 {key} v{ver} 已不存在")
                rules_params[key] = (rule.factor, rule.offset)
            rules_used = dict(pins_rules)
        else:
            active = {r.rule_key: r for r in store.active_rules()}
            rules_params = {k: (r.factor, r.offset) for k, r in active.items()}
            rules_used = {}
            for row in snapshot_rows:
                if row["caliber"] == target_caliber or row["value"] is None:
                    continue
                key = rule_key(row["measure"], row["caliber"], target_caliber)
                rule = active.get(key)
                if rule is not None:
                    rules_used[key] = rule.version_no

        rows = [SnapshotRow(**r) for r in snapshot_rows]
        converted, missing_keys = convert_rows(rows, rules_params, target_caliber)
        if missing_keys:
            detail = {"missing_rules": sorted(missing_keys)}
            if for_reverify:
                raise StateError("重算失败：缺少固化的口径换算规则",
                                 detail=detail)
            raise ValidationError("缺少口径换算规则，已拒绝生成", detail=detail)

        # 合并期间取值（与计算服务同规则：缺失不覆盖实值，冲突实值报错）
        values: dict[str, dict[str, float | None]] = {}
        evidence_by_measure: dict[str, set[str]] = {}
        for r in converted:
            slot = values.setdefault(r.measure, {})
            existing = slot.get(r.period, _ABSENT)
            if existing is _ABSENT or existing is None:
                slot[r.period] = r.value
            elif r.value is None:
                pass
            elif existing != r.value:
                raise ValidationError(
                    f"度量 {r.measure} 在 {r.period} 存在多个口径换算结果，"
                    "数据口径不唯一",
                )
            if r.evidence_id:
                evidence_by_measure.setdefault(r.measure, set()).add(
                    r.evidence_id)

        # 指标版本与假设校验
        needed = self._needed_measures(store, pins_indicators)
        normalized = normalize_assumptions(assumptions, needed)["measures"]

        # 基线：历史窗口内已换算值的均值；无有效基线视同输入不全，拒绝
        baselines: dict[str, float] = {}
        no_baseline: list[str] = []
        for measure in sorted(needed):
            present = [v for v in values.get(measure, {}).values()
                       if v is not None]
            if present:
                baselines[measure] = sum(present) / len(present)
            else:
                no_baseline.append(measure)
        if no_baseline:
            err = StateError if for_reverify else ValidationError
            raise err("历史窗口内缺少有效基线数据，已拒绝生成",
                      detail={"measures_without_baseline": no_baseline})

        # 外推：growth 复利外推，override 固定值
        future_periods = next_periods(history_end, horizon_months)
        projected: dict[str, dict[str, float | None]] = {}
        for measure in sorted(needed):
            assumption = normalized[measure]
            base = baselines[measure]
            if "override" in assumption:
                series = [assumption["override"]] * len(future_periods)
            else:
                growth = assumption["growth"]
                series = [base * (1.0 + growth) ** t
                          for t in range(1, len(future_periods) + 1)]
            projected[measure] = dict(zip(future_periods, series))

        # 逐指标在预测区间求值
        lines: list[dict] = []
        indicator_pins: dict[str, int] = {}
        if for_reverify:
            assert pins_indicators is not None
            codes = sorted(pins_indicators)
        else:
            codes = [i.code for i in store.list_indicators()]
        for code in codes:
            if for_reverify:
                assert pins_indicators is not None
                version = store.get_indicator_version(code, pins_indicators[code])
                if version is None:
                    raise StateError(f"重算失败：指标 {code} 已不存在")
                indicator = store.get_indicator(code)
                assert indicator is not None
            else:
                indicator = store.get_indicator(code)
                assert indicator is not None
                version = store.latest_indicator_version(code)
                assert version is not None
            validate_formula(version.formula)
            deps = formula_measures(version.formula)
            scoped = {m: projected.get(m, {}) for m in deps}
            result = evaluate(version.formula, version.missing_policy,
                              future_periods, scoped)
            indicator_pins[code] = version.version_no
            lines.append({
                "code": code,
                "name": indicator.name,
                "category": indicator.category,
                "unit": indicator.unit,
                "version_no": version.version_no,
                "value": result.value,
                "covered_periods": list(result.covered_periods),
                "missing_periods": list(result.missing_periods),
                "notes": list(result.notes),
                "basis": {
                    m: {"baseline": baselines[m],
                        "assumption": normalized[m]}
                    for m in deps
                },
                "evidence_ids": sorted({
                    eid for m in deps
                    for eid in evidence_by_measure.get(m, set())
                }),
            })

        pins = {
            "data_version": seq,
            "indicators": indicator_pins,
            "rules": rules_used,
        }
        return pins, lines, snapshot_rows, {"measures": normalized}

    @staticmethod
    def _report_dict(report: ScenarioReport, lines: list[dict],
                     redacted: list[str]) -> dict:
        return {
            "report_id": report.id,
            "project_id": report.project_id,
            "scenario_name": report.scenario_name,
            "baseline_version_no": report.baseline_version_no,
            "history_window": [report.history_start, report.history_end],
            "horizon_months": report.horizon_months,
            "target_caliber": report.target_caliber,
            "assumptions": report.assumptions,
            "pins": report.pins,
            "lines": lines,
            "input_fingerprint": report.input_fingerprint,
            "result_fingerprint": report.result_fingerprint,
            "created_by": report.created_by,
            "created_at": report.created_at,
            "redacted_categories": redacted,
        }
