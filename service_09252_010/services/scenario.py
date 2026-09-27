"""情景预测服务：固化假设参数与计算版本，拒绝缺参，支持按固化输入复算。

生成路径严格“先校验后落库”：基线报告、授权、假设参数、预测窗口任一不满足即
抛错，绝不写入半成品情景报告。报告保存：

- 基线报告引用与基线 pins（指标/规则/数据版本）、基线输入指纹；
- 原始假设参数（读取即可复现当时输入）；
- 计算引擎版本 ENGINE_VERSION 与输入/结果指纹。

recompute 用库内固化的假设与基线行重放投影，核对指纹——即使后续登记了新的
指标版本或新数据版本，旧情景报告仍按固化版本复现一致结果。
"""
from __future__ import annotations

from ..domain.errors import NotFoundError, ValidationError
from ..domain.fingerprint import fingerprint
from ..domain.forecast import ENGINE_VERSION, project_forecast
from ..domain.models import Principal, ScenarioReport
from ..persistence.database import Database
from ..persistence.store import Store
from ..ports import Clock, IdGenerator
from .access import AccessPolicy


class ScenarioService:
    def __init__(self, db: Database, clock: Clock, ids: IdGenerator) -> None:
        self.db = db
        self.clock = clock
        self.ids = ids

    # ---- 生成 ----
    def create(self, principal: Principal, *, baseline_report_id: str,
               scenario_name: str, model: str, assumptions: dict,
               horizon_start: str, horizon_end: str) -> dict:
        if not scenario_name or not str(scenario_name).strip():
            raise ValidationError("scenario_name 缺失")
        if not isinstance(assumptions, dict):
            raise ValidationError("assumptions 必须为参数对象")

        # 读取基线、授权、校验假设、落库全部在同一事务：校验抛错即回滚，
        # 绝不留下半成品情景报告（INSERT 在校验通过之后才发生）。
        with self.db.uow() as uow:
            store = Store(uow.conn)
            baseline = store.get_report(baseline_report_id)
            if baseline is None:
                raise NotFoundError(f"基线报告不存在: {baseline_report_id}")
            AccessPolicy(store).require(
                principal, baseline.project_id, "*", "view"
            )
            # 假设缺失在此抛 ValidationError（422），事务回滚、无任何写入。
            projection = project_forecast(
                model, assumptions, horizon_start, horizon_end,
                baseline.lines, (baseline.window_start, baseline.window_end),
            )
            lines = [self._line_dict(l) for l in projection.lines]
            input_fp = fingerprint({
                "baseline_report_id": baseline_report_id,
                "model": model,
                "assumptions": assumptions,
                "horizon": [horizon_start, horizon_end],
                "pins": baseline.pins,
                "baseline_input_fingerprint": baseline.input_fingerprint,
            })
            result_fp = fingerprint(lines)

            scenario = ScenarioReport(
                id=self.ids.new_id("scenario"),
                project_id=baseline.project_id,
                scenario_name=scenario_name.strip(),
                model=model,
                assumptions=assumptions,
                horizon_start=horizon_start,
                horizon_end=horizon_end,
                baseline_report_id=baseline_report_id,
                pins=baseline.pins,
                baseline_input_fingerprint=baseline.input_fingerprint,
                lines=lines,
                input_fingerprint=input_fp,
                result_fingerprint=result_fp,
                calculation_version=ENGINE_VERSION,
                created_by=principal.institution_id,
                created_at=self.clock.now(),
            )
            store.add_scenario_report(scenario)
        return self._dict(scenario, lines, [])

    # ---- 查询（按授权类别过滤行）----
    def get(self, principal: Principal, scenario_id: str) -> dict:
        with self.db.read() as conn:
            store = Store(conn)
            scenario = store.get_scenario_report(scenario_id)
            if scenario is None:
                raise NotFoundError(f"情景预测报告不存在: {scenario_id}")
            categories = sorted({l["category"] for l in scenario.lines})
            visible = set(AccessPolicy(store).granted_categories(
                principal, scenario.project_id, "view", categories))
        lines = [l for l in scenario.lines if l["category"] in visible]
        redacted = sorted(c for c in categories if c not in visible)
        return self._dict(scenario, lines, redacted)

    def list_reports(self, principal: Principal, project_id: str) -> list[dict]:
        with self.db.read() as conn:
            store = Store(conn)
            scenarios = store.list_scenario_reports(project_id)
            policy = AccessPolicy(store)
            result = []
            for scenario in scenarios:
                categories = sorted({l["category"] for l in scenario.lines})
                visible = set(policy.granted_categories(
                    principal, project_id, "view", categories))
                lines = [l for l in scenario.lines if l["category"] in visible]
                redacted = sorted(c for c in categories if c not in visible)
                result.append(self._dict(scenario, lines, redacted))
        return result

    # ---- 复算 ----
    def recompute(self, principal: Principal, scenario_id: str) -> dict:
        """按固化的假设与基线版本重放投影，核对输入/结果指纹。

        指标新版本、迟到数据均不影响复算：输入全部取自库内固化的情景假设与
        基线报告行。指纹一致即“重算复现了旧输入”。
        """
        with self.db.read() as conn:
            store = Store(conn)
            scenario = store.get_scenario_report(scenario_id)
            if scenario is None:
                raise NotFoundError(f"情景预测报告不存在: {scenario_id}")
            AccessPolicy(store).require(
                principal, scenario.project_id, "*", "view"
            )
            baseline = store.get_report(scenario.baseline_report_id)

        if baseline is None:
            # 外键保证不应发生；保留显式状态错误以防历史脏数据。
            from ..domain.errors import StateError

            raise StateError("复算失败：固化的基线报告已不存在")

        projection = project_forecast(
            scenario.model, scenario.assumptions,
            scenario.horizon_start, scenario.horizon_end,
            baseline.lines, (baseline.window_start, baseline.window_end),
        )
        lines = [self._line_dict(l) for l in projection.lines]
        input_payload = {
            "baseline_report_id": scenario.baseline_report_id,
            "model": scenario.model,
            "assumptions": scenario.assumptions,
            "horizon": [scenario.horizon_start, scenario.horizon_end],
            "pins": scenario.pins,
            "baseline_input_fingerprint": scenario.baseline_input_fingerprint,
        }
        recomputed_input_fp = fingerprint(input_payload)
        recomputed_result_fp = fingerprint(lines)
        return {
            "scenario_id": scenario_id,
            "calculation_version": scenario.calculation_version,
            "engine_version": ENGINE_VERSION,
            "input_match": recomputed_input_fp == scenario.input_fingerprint,
            "result_match": recomputed_result_fp == scenario.result_fingerprint,
            "pins_match": scenario.pins == baseline.pins,
            "baseline_input_match":
                scenario.baseline_input_fingerprint == baseline.input_fingerprint,
            "stored_result_fingerprint": scenario.result_fingerprint,
            "recomputed_result_fingerprint": recomputed_result_fp,
            "assumptions": scenario.assumptions,
            "pins": scenario.pins,
        }

    # ---- 序列化 ----
    @staticmethod
    def _line_dict(line) -> dict:
        return {
            "code": line.code,
            "name": line.name,
            "category": line.category,
            "unit": line.unit,
            "indicator_version_no": line.indicator_version_no,
            "baseline_value": line.baseline_value,
            "value": line.value,
            "covered_periods": list(line.covered_periods),
            "missing_periods": list(line.missing_periods),
            "notes": list(line.notes),
            "evidence_ids": list(line.evidence_ids),
        }

    @staticmethod
    def _dict(scenario: ScenarioReport, lines: list[dict],
              redacted: list[str]) -> dict:
        return {
            "scenario_id": scenario.id,
            "project_id": scenario.project_id,
            "scenario_name": scenario.scenario_name,
            "model": scenario.model,
            "assumptions": scenario.assumptions,
            "horizon": [scenario.horizon_start, scenario.horizon_end],
            "baseline_report_id": scenario.baseline_report_id,
            "calculation_version": scenario.calculation_version,
            "pins": scenario.pins,
            "baseline_input_fingerprint": scenario.baseline_input_fingerprint,
            "lines": lines,
            "input_fingerprint": scenario.input_fingerprint,
            "result_fingerprint": scenario.result_fingerprint,
            "redacted_categories": redacted,
            "created_by": scenario.created_by,
            "created_at": scenario.created_at,
        }
