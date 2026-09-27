"""情景预测报告：假设参数与计算版本固化、缺参拒绝、按旧输入重算复现。"""
from __future__ import annotations

import unittest

from service_09252_010.domain.errors import (
    NotFoundError,
    PermissionDeniedError,
    ValidationError,
)
from service_09252_010.persistence.store import Store
from support import INST_A, INST_B, PROJECT, SUPERVISOR, RigTestCase, TARGET

BASELINE_WINDOW = ("2024-01", "2024-02")
HORIZON = ("2024-03", "2024-04")
MODEL = "linear_growth"


class SeededScenarioCase(RigTestCase):
    """基线报告：招生 2024-01 DE 10（×2 换算）+ 2024-02 目标口径 20 = 40。"""

    def setUp(self) -> None:
        super().setUp()
        self.seed_indicators()
        self.evidence = self.seed_evidence()
        self.seed_rule(factor=2.0, measure="enrollment_count")
        rows = [
            ("enrollment_count", "2024-01", "DE-DUAL", 10),
            ("enrollment_count", "2024-02", TARGET, 20),
        ]
        self.seed_import(
            [{"measure": m, "period": p, "caliber": c, "value": v,
              "evidence_id": self.evidence} for m, p, c, v in rows],
            self.evidence,
        )
        self.rig.grant(INST_A.institution_id, permission="calculate")
        self.rig.grant(INST_A.institution_id, permission="view")
        self.baseline_id = self.calculate(key="baseline-1",
                                          window=BASELINE_WINDOW)
        self.baseline = self.rig.calculation.get_report(
            SUPERVISOR, self.baseline_id
        )

    def create(self, principal=INST_A, **overrides) -> dict:
        params = dict(
            baseline_report_id=self.baseline_id, scenario_name="乐观情景",
            model=MODEL, assumptions={"growth_rate": 0.10},
            horizon_start=HORIZON[0], horizon_end=HORIZON[1],
        )
        params.update(overrides)
        return self.rig.scenarios.create(principal, **params)

    def line(self, report: dict, code: str) -> dict:
        return next(l for l in report["lines"] if l["code"] == code)

    def scenario_count(self) -> int:
        with self.rig.db.read() as conn:
            return len(Store(conn).list_scenario_reports(PROJECT))


class PersistenceTests(SeededScenarioCase):
    def test_assumptions_and_versions_persisted_together(self) -> None:
        created = self.create()
        self.assertEqual(created["calculation_version"], "forecast-1")
        self.assertEqual(created["assumptions"], {"growth_rate": 0.10})
        self.assertEqual(created["baseline_report_id"], self.baseline_id)
        # pins 与基线输入指纹一并固化，读取即可复现当时输入
        self.assertEqual(created["pins"], self.baseline["pins"])
        self.assertEqual(created["baseline_input_fingerprint"],
                         self.baseline["input_fingerprint"])
        self.assertTrue(created["input_fingerprint"])
        self.assertTrue(created["result_fingerprint"])

        fetched = self.rig.scenarios.get(SUPERVISOR, created["scenario_id"])
        self.assertEqual(fetched["assumptions"], {"growth_rate": 0.10})
        self.assertEqual(fetched["pins"], self.baseline["pins"])
        self.assertEqual(fetched["calculation_version"], "forecast-1")

    def test_forecast_linear_growth_value(self) -> None:
        report = self.create()
        # 基线 40 / 2 期 = 每期 20；预测 2 期 × 20 × (1+0.10) = 44
        enrollment = self.line(report, "enrollment_total")
        self.assertEqual(enrollment["baseline_value"], 40.0)
        self.assertEqual(enrollment["value"], 44.0)
        self.assertEqual(enrollment["indicator_version_no"], 1)
        self.assertEqual(enrollment["covered_periods"],
                         ["2024-03", "2024-04"])
        # 无基线数据的指标不外推，保留 None 而非臆造
        employment = self.line(report, "employment_rate")
        self.assertIsNone(employment["value"])
        self.assertIsNone(employment["baseline_value"])
        self.assertIn("基线无有效值，无法外推", employment["notes"])

    def test_list_reports_for_project(self) -> None:
        self.create(scenario_name="情景甲")
        self.create(scenario_name="情景乙")
        reports = self.rig.scenarios.list_reports(SUPERVISOR, PROJECT)
        self.assertEqual([r["scenario_name"] for r in reports],
                         ["情景甲", "情景乙"])


class MissingParameterTests(SeededScenarioCase):
    def test_missing_growth_rate_rejected_without_partial_row(self) -> None:
        before = self.scenario_count()
        with self.assertRaises(ValidationError) as ctx:
            self.create(assumptions={"other": 1})
        self.assertEqual(ctx.exception.detail, {"missing": ["growth_rate"]})
        # 拒绝生成且不写半成品
        self.assertEqual(self.scenario_count(), before)

    def test_empty_assumptions_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            self.create(assumptions={})
        self.assertEqual(self.scenario_count(), 0)

    def test_unknown_model_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            self.create(model="no-such-model")
        self.assertEqual(self.scenario_count(), 0)

    def test_non_finite_factor_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            self.create(assumptions={"growth_rate": float("nan")})
        with self.assertRaises(ValidationError):
            self.create(assumptions={"growth_rate": float("inf")})
        self.assertEqual(self.scenario_count(), 0)

    def test_unsupported_assumption_type_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            self.create(assumptions={"growth_rate": 0.1, "note": object()})

    def test_horizon_must_follow_baseline_window(self) -> None:
        with self.assertRaises(ValidationError):
            self.create(horizon_start="2024-02", horizon_end="2024-03")

    def test_missing_baseline_report_not_found(self) -> None:
        with self.assertRaises(NotFoundError):
            self.create(baseline_report_id="report-nope")
        self.assertEqual(self.scenario_count(), 0)

    def test_missing_scenario_name_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            self.create(scenario_name="  ")


class RecomputeTests(SeededScenarioCase):
    def test_recompute_reproduces_old_input(self) -> None:
        created = self.create()
        result = self.rig.scenarios.recompute(INST_A, created["scenario_id"])
        self.assertTrue(result["input_match"])
        self.assertTrue(result["result_match"])
        self.assertTrue(result["pins_match"])
        self.assertTrue(result["baseline_input_match"])
        # 重算回放出当时保存的假设参数
        self.assertEqual(result["assumptions"], {"growth_rate": 0.10})
        self.assertEqual(result["calculation_version"], "forecast-1")

    def test_recompute_stable_after_indicator_and_data_versions_move(self) -> None:
        created = self.create()
        original_fp = created["result_fingerprint"]

        # 指标定义出 v2（zero 策略）、迟到数据形成 v2
        self.rig.indicators.add_version(
            SUPERVISOR, "enrollment_total",
            formula={"type": "sum", "measure": "enrollment_count"},
            missing_policy="zero",
        )
        self.seed_import(
            [{"measure": "enrollment_count", "period": "2024-02",
              "caliber": TARGET, "value": 99, "evidence_id": self.evidence}],
            self.evidence, reason="迟到补报",
        )

        result = self.rig.scenarios.recompute(INST_A, created["scenario_id"])
        self.assertTrue(result["input_match"])
        self.assertTrue(result["result_match"])
        self.assertTrue(result["pins_match"])
        self.assertTrue(result["baseline_input_match"])

        # 固化报告本身与旧假设均未被改写
        fetched = self.rig.scenarios.get(SUPERVISOR, created["scenario_id"])
        self.assertEqual(fetched["result_fingerprint"], original_fp)
        self.assertEqual(fetched["assumptions"], {"growth_rate": 0.10})
        self.assertEqual(fetched["pins"]["indicators"]["enrollment_total"], 1)

    def test_recompute_unknown_scenario_not_found(self) -> None:
        with self.assertRaises(NotFoundError):
            self.rig.scenarios.recompute(INST_A, "scenario-nope")


class AccessTests(SeededScenarioCase):
    def test_create_requires_view_grant(self) -> None:
        with self.assertRaises(PermissionDeniedError):
            self.create(principal=INST_B)

    def test_lines_filtered_by_granted_category(self) -> None:
        created = self.create()
        self.rig.grant(INST_B.institution_id, category="招生",
                       permission="view")
        view = self.rig.scenarios.get(INST_B, created["scenario_id"])
        self.assertEqual([l["code"] for l in view["lines"]],
                         ["enrollment_total"])
        self.assertEqual(sorted(view["redacted_categories"]),
                         ["就业", "师资培养"])

        full = self.rig.scenarios.get(SUPERVISOR, created["scenario_id"])
        self.assertEqual(len(full["lines"]), 3)
        self.assertEqual(full["redacted_categories"], [])


if __name__ == "__main__":
    unittest.main()
