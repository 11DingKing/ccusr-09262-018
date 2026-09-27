"""情景预测：在已固化的基线报告之上套用假设参数做前向推算。

基线输入（指标行、口径版本、数据版本）固化在历史报告中；情景只额外保存一份
假设参数。预测值 = 基线指标值 × 增长系数 × 期间数，系数缺省 1.0。

假设是“参数”而非可执行代码：仅允许白名单标量与数值列表，杜绝注入，且规范化
JSON 可稳定指纹。参数缺失（必填项缺失或系数非有限数）时拒绝生成。
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from .errors import ValidationError
from .periods import iter_periods, period_key

# 允许的预测模型；每种模型所需的必填假设键。
_GROWTH = "linear_growth"
SCENARIO_MODELS: dict[str, tuple[str, ...]] = {
    # 每期按 growth_rate（相对基线期均值）线性增长
    _GROWTH: ("growth_rate",),
}

# 计算引擎版本：随投影算法/假设语义变更而递增，随报告固化。
ENGINE_VERSION = "forecast-1"

# 假设值允许的标量类型（bool 是 int 子类，需单独排除以免歧义）。
_SCALAR_TYPES = (int, float, str)


def validate_assumptions(model: str, assumptions: dict) -> None:
    """校验假设参数完整且取值合法；缺失或非法时抛 ValidationError。

    服务层在写库前调用——校验不通过绝不产生半成品记录。
    """
    if not isinstance(assumptions, dict) or not assumptions:
        raise ValidationError("情景预测必须提供非空假设参数 assumptions")
    required = SCENARIO_MODELS.get(model)
    if required is None:
        raise ValidationError(
            f"未知预测模型: {model!r}，支持 {sorted(SCENARIO_MODELS)}"
        )
    missing = [key for key in required if key not in assumptions]
    if missing:
        raise ValidationError(
            "假设参数缺失，拒绝生成", detail={"missing": sorted(missing)}
        )
    _check_value(assumptions, path="assumptions")


def _check_value(value, *, path: str) -> None:
    """递归保证假设树只含白名单标量/列表/对象，数值均有限。"""
    if isinstance(value, bool) or not isinstance(value, _SCALAR_TYPES):
        if isinstance(value, (dict, list)):
            iterator = value.values() if isinstance(value, dict) else value
            for item in iterator:
                _check_value(item, path=path)
            return
        raise ValidationError(f"假设参数 {path} 含不支持的类型: {type(value).__name__}")
    if isinstance(value, float) and not math.isfinite(value):
        raise ValidationError(f"假设参数 {path} 必须为有限数值")


@dataclass(frozen=True)
class ForecastLine:
    code: str
    name: str
    category: str
    unit: str
    indicator_version_no: int  # 取自基线报告 pins，预测不改指标定义
    baseline_value: float | None
    value: float | None
    covered_periods: tuple[str, ...]
    missing_periods: tuple[str, ...]
    notes: tuple[str, ...]
    evidence_ids: tuple[str, ...]


@dataclass(frozen=True)
class Projection:
    lines: tuple[ForecastLine, ...]
    horizon_periods: tuple[str, ...]


def project_forecast(
    model: str,
    assumptions: dict,
    horizon_start: str,
    horizon_end: str,
    baseline_lines: list[dict],
    baseline_window: tuple[str, str],
) -> Projection:
    """按假设把基线报告的每个指标行前向推算到预测窗口。

    基线期均值 = baseline_value / 基线窗口期间数；无基线值（None）的指标无法
    外推，预测值保留 None 并在 notes 说明，不臆造数字。
    """
    validate_assumptions(model, assumptions)
    horizon = iter_periods(horizon_start, horizon_end)
    if period_key(horizon_start) <= period_key(baseline_window[1]):
        raise ValidationError("预测窗口起点必须晚于基线观察期终点")

    base_periods = iter_periods(*baseline_window)
    base_count = len(base_periods)
    growth_rate = float(assumptions["growth_rate"])

    lines: list[ForecastLine] = []
    for line in baseline_lines:
        baseline_value = line["value"]
        notes = list(line.get("notes", ()))
        if baseline_value is None:
            value = None
            notes.append("基线无有效值，无法外推")
        elif model == _GROWTH:
            per_period = baseline_value / base_count
            value = per_period * (1.0 + growth_rate) * len(horizon)
        else:  # pragma: no cover - validate_assumptions 已拦截
            raise ValidationError(f"未知预测模型: {model!r}")
        lines.append(ForecastLine(
            code=line["code"],
            name=line["name"],
            category=line["category"],
            unit=line["unit"],
            indicator_version_no=line["version_no"],
            baseline_value=baseline_value,
            value=value,
            covered_periods=tuple(horizon) if value is not None else (),
            missing_periods=() if value is not None else list(horizon),
            notes=tuple(notes),
            evidence_ids=tuple(line.get("evidence_ids", ())),
        ))
    return Projection(lines=tuple(lines), horizon_periods=tuple(horizon))
