"""情景预测报告：假设参数与计算版本落库、缺参拒绝、重算复现旧输入。"""
from __future__ import annotations

import unittest

from service_09252_010.domain.errors import (
    PermissionDeniedError,
    ValidationError,
)
from support import INST_A, INST_B, PROJECT, SUPERVISOR, RigTestCase, TARGET

HISTORY = ("2024-01", "2024-03")

# 假设参数：招生按 10% 复利增长，就业分子固定 50，分母零增长，师资 25% 增长
ASSUMPTIONS = {
    "measures": {
        "enrollment_count": {"growth": 0.1},
        "employed_count": {"override": 50},
        "graduate_count": {"growth": 0.0},
        "trained_teacher_count": {"growth": 0.25},
    }
}


class SeededCase(RigTestCase):
    """预置三类指标、一条换算规则与历史窗口数据。"""

    def setUp(self) -> None:
        super().setUp()
        self.seed_indicators()
        self.evidence = self.seed_evidence()
        self.seed_rule(factor=2.0, measure="enrollment_count")
        rows = [
            ("enrollment_count", "2024-01", "DE-DUAL", 10),  # 换算后 20
            ("enrollment_count", "2024-02", TARGET, 20),
            ("enrollment_count", "2024-03", TARGET, 30),
            ("employed_count", "2024-01", TARGET, 45),
            ("graduate_count", "2024-01", TARGET, 60),
            ("trained_teacher_count", "2024-01", TARGET, 8),
        ]
        self.seed_import(
            [{"measure": m, "period": p, "caliber": c, "value": v,
              "evidence_id": self.evidence} for m, p, c, v in rows],
            self.evidence,
        )
        self.rig.grant(INST_A.institution_id, permission="calculate")
        self.rig.grant(INST_A.institution_id, permission="view")

    def generate(self, principal=INST_A, **overrides) -> dict:
        params = {
            "project_id": PROJECT,
            "scenario_name": "乐观情景",
            "history_start": HISTORY[0],
            "history_end": HISTORY[1],
            "horizon_months": 2,
            "target_caliber": TARGET,
            "assumptions": ASSUMPTIONS,
        }
        params.update(overrides)
        return self.rig.scenarios.generate(principal, **params)

    def scenario_count(self) -> int:
        with self.rig.db.read() as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS n FROM scenario_reports").fetchone()
        return row["n"]

    def line(self, report: dict, code: str) -> dict:
        return next(l for l in report["lines"] if l["code"] == code)


class GenerateTests(SeededCase):
    def test_generate_persists_assumptions_and_pins(self) -> None:
        report = self.generate()
        self.assertEqual(report["baseline_version_no"], 1)
        # 假设参数随结果落库，读取时可复现当时输入
        self.assertEqual(report["assumptions"], {
            "measures": {
                "enrollment_count": {"growth": 0.1},
                "employed_count": {"override": 50.0},
                "graduate_count": {"growth": 0.0},
                "trained_teacher_count": {"growth": 0.25},
            }
        })
        # 计算版本固化：数据版本、指标版本、规则版本
        self.assertEqual(report["pins"]["data_version"], 1)
        self.assertEqual(report["pins"]["indicators"], {
            "enrollment_total": 1, "employment_rate": 1,
            "trained_teachers": 1,
        })
        self.assertEqual(report["pins"]["rules"],
                         {f"enrollment_count|DE-DUAL|{TARGET}": 1})
        self.assertTrue(report["input_fingerprint"])
        self.assertTrue(report["result_fingerprint"])

        # 基线 = 历史窗口均值：招生 (20+20+30)/3 = 70/3
        enrollment = self.line(report, "enrollment_total")
        base = 70.0 / 3.0
        self.assertAlmostEqual(enrollment["basis"]["enrollment_count"]
                               ["baseline"], base)
        # 两期外推：base*1.1 + base*1.21
        self.assertAlmostEqual(enrollment["value"],
                               base * 1.1 + base * 1.21)
        self.assertEqual(enrollment["covered_periods"], ["2024-04", "2024-05"])

        employment = self.line(report, "employment_rate")
        self.assertAlmostEqual(employment["value"], 100.0 / 120.0 * 100)

        teachers = self.line(report, "trained_teachers")
        self.assertAlmostEqual(teachers["value"], 8 * 1.25 + 8 * 1.25 ** 2)

    def test_get_returns_stored_inputs(self) -> None:
        report_id = self.generate()["report_id"]
        fetched = self.rig.scenarios.get(SUPERVISOR, report_id)
        self.assertEqual(fetched["assumptions"]["measures"]
                         ["enrollment_count"], {"growth": 0.1})
        self.assertEqual(fetched["history_window"], list(HISTORY))
        self.assertEqual(fetched["horizon_months"], 2)
        self.assertEqual(fetched["pins"]["data_version"], 1)
        listed = self.rig.scenarios.list(SUPERVISOR, PROJECT)
        self.assertEqual([r["report_id"] for r in listed], [report_id])


class MissingParamTests(SeededCase):
    def test_missing_required_params_rejected_and_nothing_persisted(self) -> None:
        with self.assertRaises(ValidationError) as ctx:
            self.generate(horizon_months=None, assumptions=None)
        self.assertEqual(ctx.exception.detail["missing"],
                         ["assumptions", "horizon_months"])
        self.assertEqual(self.scenario_count(), 0)
        self.assertEqual(self.rig.scenarios.list(SUPERVISOR, PROJECT), [])

    def test_missing_assumption_measure_rejected(self) -> None:
        broken = {"measures": {
            k: v for k, v in ASSUMPTIONS["measures"].items()
            if k != "trained_teacher_count"
        }}
        with self.assertRaises(ValidationError) as ctx:
            self.generate(assumptions=broken)
        self.assertEqual(ctx.exception.detail["missing_measures"],
                         ["trained_teacher_count"])
        self.assertEqual(self.scenario_count(), 0)

    def test_malformed_assumption_rejected(self) -> None:
        bad = {"measures": {**ASSUMPTIONS["measures"],
                            "enrollment_count": {"growth": 0.1,
                                                 "override": 5}}}
        with self.assertRaises(ValidationError):
            self.generate(assumptions=bad)
        unknown = {"measures": {**ASSUMPTIONS["measures"],
                                "ghost_measure": {"growth": 0.1}}}
        with self.assertRaises(ValidationError) as ctx:
            self.generate(assumptions=unknown)
        self.assertEqual(ctx.exception.detail["unknown_measures"],
                         ["ghost_measure"])
        self.assertEqual(self.scenario_count(), 0)

    def test_invalid_horizon_and_window_rejected(self) -> None:
        for bad in (0, -1, 2.5, True, "2"):
            with self.assertRaises(ValidationError):
                self.generate(horizon_months=bad)
        with self.assertRaises(ValidationError):
            self.generate(history_start="2024-03", history_end="2024-01")
        self.assertEqual(self.scenario_count(), 0)

    def test_no_baseline_data_rejected(self) -> None:
        # trained_teacher_count 在窗口外才有数据 → 无有效基线
        with self.assertRaises(ValidationError) as ctx:
            self.generate(history_start="2024-02", history_end="2024-03")
        self.assertIn("trained_teacher_count",
                      ctx.exception.detail["measures_without_baseline"])
        self.assertEqual(self.scenario_count(), 0)

    def test_generate_requires_calculate_grant(self) -> None:
        with self.assertRaises(PermissionDeniedError):
            self.generate(principal=INST_B)
        self.assertEqual(self.scenario_count(), 0)


class RecomputeTests(SeededCase):
    def test_recompute_reproduces_old_inputs_after_world_changes(self) -> None:
        report = self.generate()
        report_id = report["report_id"]

        # 世界继续演进：指标 v2、规则 v2（factor=3）、迟到数据（数据版本 2）
        self.rig.indicators.add_version(
            SUPERVISOR, "enrollment_total",
            formula={"type": "latest", "measure": "enrollment_count"},
            missing_policy="skip",
        )
        self.seed_rule(factor=3.0, measure="enrollment_count")
        self.seed_import(
            [{"measure": "enrollment_count", "period": "2024-03",
              "caliber": TARGET, "value": 99,
              "evidence_id": self.evidence}],
            self.evidence, reason="迟到更正",
        )

        # 重算旧报告：严格按固化的假设与版本，复现当时输入
        check = self.rig.scenarios.recompute(SUPERVISOR, report_id)
        self.assertTrue(check["input_match"])
        self.assertTrue(check["result_match"])
        reproduced = check["reproduced_inputs"]
        self.assertEqual(reproduced["assumptions"], report["assumptions"])
        self.assertEqual(reproduced["pins"], report["pins"])
        self.assertEqual(reproduced["baseline_version_no"], 1)

        # 旧报告内容不被改写
        fetched = self.rig.scenarios.get(SUPERVISOR, report_id)
        self.assertEqual(fetched["lines"], report["lines"])
        self.assertEqual(fetched["result_fingerprint"],
                         report["result_fingerprint"])

        # 新情景使用最新版本与数据，结果不同
        newer = self.generate(scenario_name="最新口径情景")
        self.assertEqual(newer["baseline_version_no"], 2)
        self.assertEqual(newer["pins"]["indicators"]["enrollment_total"], 2)
        self.assertNotEqual(newer["result_fingerprint"],
                            report["result_fingerprint"])

    def test_recompute_matches_clean_regeneration(self) -> None:
        first = self.generate()
        check = self.rig.scenarios.recompute(SUPERVISOR, first["report_id"])
        self.assertEqual(check["recomputed_result_fingerprint"],
                         first["result_fingerprint"])


class AccessTests(SeededCase):
    def test_lines_filtered_by_granted_category(self) -> None:
        report_id = self.generate()["report_id"]
        self.rig.grant(INST_B.institution_id, category="招生",
                       permission="view")
        view = self.rig.scenarios.get(INST_B, report_id)
        self.assertEqual([l["code"] for l in view["lines"]],
                         ["enrollment_total"])
        self.assertEqual(sorted(view["redacted_categories"]),
                         ["就业", "师资培养"])
        # 假设与版本对可见方同样返回，便于复现
        self.assertEqual(view["assumptions"]["measures"]
                         ["enrollment_count"], {"growth": 0.1})


class HttpScenarioTests(SeededCase):
    """HTTP 边界：缺参 422 且不落半成品。"""

    def setUp(self) -> None:
        super().setUp()
        import io
        import json as jsonlib

        from service_09252_010.interfaces.wsgi_app import make_app

        self._io = io
        self._json = jsonlib
        self.app = make_app(self.rig.db.path)

    def call(self, method: str, path: str, body: dict | None = None,
             headers: dict | None = None, query: str = ""):
        io, jsonlib = self._io, self._json
        payload = jsonlib.dumps(body).encode("utf-8") if body is not None else b""
        env = {
            "REQUEST_METHOD": method,
            "PATH_INFO": path,
            "QUERY_STRING": query,
            "CONTENT_LENGTH": str(len(payload)),
            "wsgi.input": io.BytesIO(payload),
        }
        for key, value in (headers or {}).items():
            env["HTTP_" + key.upper().replace("-", "_")] = value
        captured: dict = {}

        def start_response(status, response_headers):
            captured["status"] = int(status.split()[0])

        chunks = self.app(env, start_response)
        return captured["status"], jsonlib.loads(b"".join(chunks).decode("utf-8"))

    def test_http_missing_params_422_and_nothing_persisted(self) -> None:
        headers = {"X-Institution-Id": "机构A"}
        status, body = self.call("POST", "/scenarios", {
            "project_id": PROJECT,
            "scenario_name": "缺参情景",
        }, headers)
        self.assertEqual(status, 422)
        self.assertEqual(body["error"], "validation_error")
        self.assertIn("assumptions", body["detail"]["missing"])
        self.assertEqual(self.scenario_count(), 0)

    def test_http_generate_get_recompute_flow(self) -> None:
        headers = {"X-Institution-Id": "机构A"}
        status, created = self.call("POST", "/scenarios", {
            "project_id": PROJECT,
            "scenario_name": "乐观情景",
            "history_start": HISTORY[0],
            "history_end": HISTORY[1],
            "horizon_months": 2,
            "target_caliber": TARGET,
            "assumptions": ASSUMPTIONS,
        }, headers)
        self.assertEqual(status, 201, created)
        report_id = created["report_id"]

        status, fetched = self.call("GET", f"/scenarios/{report_id}",
                                    headers=headers)
        self.assertEqual(status, 200)
        self.assertEqual(fetched["assumptions"]["measures"]
                         ["enrollment_count"], {"growth": 0.1})
        self.assertEqual(fetched["pins"]["data_version"], 1)

        status, check = self.call(
            "POST", f"/scenarios/{report_id}/recompute", {}, headers)
        self.assertEqual(status, 200)
        self.assertTrue(check["input_match"])
        self.assertTrue(check["result_match"])

        status, listed = self.call("GET", "/scenarios", headers=headers,
                                   query=f"project_id={PROJECT}")
        self.assertEqual(status, 200)
        self.assertEqual(len(listed["reports"]), 1)


if __name__ == "__main__":
    unittest.main()
