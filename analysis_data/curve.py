"""有界饱和曲线拟合、轮廓可识别性诊断与联合预测包络。

模型为 ``f(n) = a - b*exp(-k*n/scale)``。k 位于资料给定的有限有序网格，
(a,b) 位于三角形 ``0 <= b <= a <= 1``。目标按每一行的已知方差加权：

    Q(k,a,b) = sum((y - a + b*exp(-k*n/scale))^2 / variance)

本模块只使用历史汇总行，不训练任何学习模型，也不产生新的观测。所有数值
都是合成资料。
"""
import math
from fractions import Fraction

from .io import finite, unique

MODEL_FORMULA = "a - b*exp(-k*n/scale)"
DEFAULT_TIE_TOLERANCE = 1e-10
DEFAULT_SOLVER_TOLERANCE = 1e-9
DEFAULT_MAX_ITERATIONS = 10000


def _fraction(value, name, positive=False):
    if isinstance(value, bool) or value is None:
        raise ValueError("%s 必须是有限非负有理数" % name)
    if isinstance(value, (int, float)):
        if not math.isfinite(value):
            raise ValueError("%s 必须是有限非负有理数" % name)
        result = Fraction(str(value))
    elif isinstance(value, str):
        try:
            result = Fraction(value)
        except (ValueError, ZeroDivisionError):
            raise ValueError("%s 必须是有限非负有理数" % name)
    else:
        raise ValueError("%s 必须是有限非负有理数" % name)
    if result < 0 or (positive and result == 0):
        raise ValueError("%s 必须是有限%s有理数" % (name, "正" if positive else "非负"))
    return result


def parse_material(value):
    """严格校验资料并返回拟合所需的原始行、网格和配置。"""
    if not isinstance(value, dict):
        raise ValueError("资料顶层必须是对象")
    for key in ("scale", "runs", "k_grid", "cutoff", "candidate_n",
                "cost", "baseline_n"):
        if key not in value:
            raise ValueError("缺少字段 %s" % key)

    scale = finite(value["scale"])
    if scale <= 0:
        raise ValueError("规模换算必须为正")

    runs = value["runs"]
    if not isinstance(runs, list) or not runs:
        raise ValueError("runs 必须是非空数组")
    unique(runs, "run_id")
    rows = []
    for row in runs:
        if not isinstance(row, dict):
            raise ValueError("每条运行必须是对象")
        n = finite(row["n"])
        y = finite(row["y"])
        variance = finite(row["variance"])
        if n < 0 or not 0 <= y <= 1 or variance <= 0:
            raise ValueError("规模比例、运行比例或已知测量方差错误")
        rows.append({
            "run_id": row["run_id"],
            "n": n,
            "y": y,
            "variance": variance,
            "weight": 1.0 / variance,
        })

    grid_raw = value["k_grid"]
    if not isinstance(grid_raw, list) or not grid_raw:
        raise ValueError("k_grid 必须是非空数组")
    grid = []
    for item in grid_raw:
        k = finite(item)
        if k <= 0:
            raise ValueError("k_grid 必须全为正")
        if grid and not k > grid[-1]:
            raise ValueError("k_grid 必须严格升序")
        grid.append(k)

    cutoff = finite(value["cutoff"])
    if cutoff < 0:
        raise ValueError("cutoff 必须非负")
    baseline_n = finite(value["baseline_n"])
    if baseline_n < 0:
        raise ValueError("baseline_n 必须非负")
    candidates = value["candidate_n"]
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("candidate_n 必须是非空数组")
    candidate_ns = []
    for item in candidates:
        n = finite(item)
        if n < baseline_n:
            raise ValueError("候选采集规模不得小于 baseline_n")
        candidate_ns.append(n)

    cost = value["cost"]
    if not isinstance(cost, dict):
        raise ValueError("cost 必须是对象")
    fixed = _fraction(cost.get("fixed"), "cost.fixed")
    per_sample = _fraction(cost.get("per_sample"), "cost.per_sample")

    solver = value.get("solver", {})
    if not isinstance(solver, dict):
        raise ValueError("solver 必须是对象")
    tolerance = solver.get("tolerance", DEFAULT_SOLVER_TOLERANCE)
    tolerance = finite(tolerance)
    if tolerance <= 0:
        raise ValueError("solver.tolerance 必须为正")
    max_iterations = solver.get("max_iterations", DEFAULT_MAX_ITERATIONS)
    if isinstance(max_iterations, bool) or not isinstance(max_iterations, int):
        raise ValueError("solver.max_iterations 必须是正整数")
    if max_iterations <= 0:
        raise ValueError("solver.max_iterations 必须是正整数")
    tie_tolerance = value.get("tie_tolerance", DEFAULT_TIE_TOLERANCE)
    tie_tolerance = finite(tie_tolerance)
    if tie_tolerance < 0:
        raise ValueError("tie_tolerance 必须非负")

    return {
        "scale": scale,
        "rows": rows,
        "k_grid": grid,
        "cutoff": cutoff,
        "candidate_n": candidate_ns,
        "baseline_n": baseline_n,
        "cost": {"fixed": fixed, "per_sample": per_sample},
        "solver": {"tolerance": tolerance,
                   "max_iterations": max_iterations},
        "tie_tolerance": tie_tolerance,
    }


def _design(rows, k, scale):
    """固定 k 下的加权充分统计量；每一行保留自己的权重。"""
    s0 = s1 = s2 = y0 = y1 = c = 0.0
    for row in rows:
        t = math.exp(-k * row["n"] / scale)
        w = row["weight"]
        y = row["y"]
        s0 += w
        s1 += w * t
        s2 += w * t * t
        y0 += w * y
        y1 += w * t * y
        c += w * y * y
    det = s0 * s2 - s1 * s1
    trace = s0 + s2
    disc = math.sqrt(max(0.0, trace * trace - 4.0 * det))
    eig_hi = (trace + disc) / 2.0
    eig_lo = (trace - disc) / 2.0
    condition = (eig_hi / eig_lo
                 if eig_lo > 1e-12 * max(eig_hi, 1.0) else None)
    return {"s0": s0, "s1": s1, "s2": s2,
            "y0": y0, "y1": y1, "c": c,
            "determinant": det,
            "hessian_eigen_max": eig_hi,
            "hessian_eigen_min": eig_lo,
            "hessian_condition": condition}


def _q(stats, a, b):
    return (stats["s0"] * a * a + stats["s2"] * b * b
            - 2.0 * stats["s1"] * a * b
            - 2.0 * stats["y0"] * a + 2.0 * stats["y1"] * b
            + stats["c"])


def _clamp01(x):
    return min(1.0, max(0.0, x))


def _residuals(a, b):
    return {"a<=1": a - 1.0, "-b<=0": -b, "b<=a": b - a}


def _solve_fixed_k(stats, tolerance, tie_tolerance):
    """三角形上的有界凸二次子问题。

    枚举可行域顶点、三条边上的最小值以及（若正定且可行）无约束内点。
    紧凑凸集上的凸二次最小点必在这些候选中；奇异时边/顶点仍给出最小值。
    """
    s0, s1, s2 = stats["s0"], stats["s1"], stats["s2"]
    y0, y1, c = stats["y0"], stats["y1"], stats["c"]
    det = stats["determinant"]
    det_tol = 1e-12 * s0 * s2
    small = 1e-12 * max(s0, s2, 1.0)

    candidates = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)]

    # 边 b=0, a in [0,1]
    candidates.append((_clamp01(y0 / s0), 0.0))

    # 边 a=1, b in [0,1]
    if s2 > small:
        candidates.append((1.0, _clamp01((s1 - y1) / s2)))
    else:
        candidates.append((1.0, 0.0))

    # 边 b=a, a in [0,1]
    den = s0 + s2 - 2.0 * s1
    if den > small:
        a_edge = _clamp01((y0 - y1) / den)
    else:
        a_edge = 0.0
    candidates.append((a_edge, a_edge))

    # 正定二次目标的无约束最小点，若落在三角形内
    interior = None
    if det > det_tol:
        astar = (y0 * s2 - s1 * y1) / det
        bstar = (s1 * y0 - s0 * y1) / det
        if 0.0 <= bstar <= astar <= 1.0:
            candidates.append((astar, bstar))
            interior = (astar, bstar)

    values = [_q(stats, a, b) for a, b in candidates]
    best = min(values)
    chosen = None
    for (a, b), value in zip(candidates, values):
        if value <= best + tie_tolerance:
            if chosen is None or (a, b) < chosen:
                chosen = (a, b)
    a, b = chosen
    q = _q(stats, a, b)

    distinct_ties = sum(
        1 for (ca, cb), value in zip(candidates, values)
        if value <= best + tie_tolerance
        and (abs(ca - a) > 1e-8 or abs(cb - b) > 1e-8)
    )
    residuals = _residuals(a, b)
    active = []
    if a >= 1.0 - 1e-7:
        active.append("a<=1")
    if b <= 1e-7:
        active.append("-b<=0")
    if a - b <= 1e-7:
        active.append("b<=a")

    return {
        "a": a,
        "b": b,
        "objective": q,
        "design": stats,
        "rank": 2 if det > det_tol else 1,
        "collinear_design": det <= det_tol,
        "non_unique_minimizer": det <= det_tol or distinct_ties > 0,
        "interior_minimizer": interior is not None,
        "constraint_residuals": residuals,
        "max_constraint_violation": max(0.0, max(residuals.values())),
        "active_constraints": active,
        "boundary": bool(active),
        "candidate_count": len(candidates),
    }


def fit_grid(value):
    """在声明的 k 网格上逐点拟合，并给出轮廓接受集合。"""
    material = parse_material(value)
    rows, scale, grid = material["rows"], material["scale"], material["k_grid"]
    tolerance = material["solver"]["tolerance"]
    tie_tolerance = material["tie_tolerance"]

    fits = []
    for index, k in enumerate(grid):
        stats = _design(rows, k, scale)
        fit = _solve_fixed_k(stats, tolerance, tie_tolerance)
        fit["index"] = index
        fit["k"] = k
        fits.append(fit)

    best_objective = min(item["objective"] for item in fits)
    best_index = None
    for item in fits:
        if item["objective"] <= best_objective + tie_tolerance:
            best_index = item["index"]
            break
    grid_best = fits[best_index]
    grid_ties = sum(
        1 for item in fits
        if item["objective"] <= best_objective + tie_tolerance
    )

    threshold = best_objective + material["cutoff"]
    for item in fits:
        item["profile_gap"] = item["objective"] - best_objective
        item["accepted"] = item["objective"] <= threshold

    segments = []
    current = None
    for item in fits:
        if item["accepted"]:
            if current is None or item["index"] != current[-1]["index"] + 1:
                current = [item]
                segments.append(current)
            else:
                current.append(item)
        else:
            current = None

    rank_deficient = [item["k"] for item in fits if item["collinear_design"]]
    accepted = [item for item in fits if item["accepted"]]

    return {
        "model": {
            "formula": MODEL_FORMULA,
            "normalized_n": "n/scale",
            "constraints": "0 <= b <= a <= 1",
            "k_domain": "声明网格；仅报告网格最佳解，不作连续全局最优声明",
            "fixed_k_method": "三角形上的顶点/边/内点枚举（有界凸二次子问题）",
            "bound_method": ("三角形与二次次水平集交的极值候选枚举；"
                             "正定用椭圆切点/边交点，退化用平行带多边形交点"),
            "interpretation": (
                "profile 集合是给定模型与已知方差下的模型条件集合；"
                "cutoff 由调用者按高斯近似给出，不自动赋予有限样本覆盖率"
            ),
        },
        "solver": {
            "tolerance": material["solver"]["tolerance"],
            "max_iterations": material["solver"]["max_iterations"],
            "note": "极值枚举是解析候选集，不依赖迭代收敛；最大迭代数仅保留配置口径",
        },
        "source": {
            "scale": scale,
            "baseline_n": material["baseline_n"],
            "cutoff": material["cutoff"],
            "k_grid": grid,
            "tie_tolerance": tie_tolerance,
            "observations": [
                {"run_id": r["run_id"], "n": r["n"], "y": r["y"],
                 "variance": r["variance"], "weight": r["weight"]}
                for r in rows
            ],
            "row_count": len(rows),
            "scale_group_count": len({r["n"] for r in rows}),
        },
        "grid": [
            {
                "index": item["index"],
                "k": item["k"],
                "a": item["a"],
                "b": item["b"],
                "objective": item["objective"],
                "profile_gap": item["profile_gap"],
                "accepted": item["accepted"],
                "rank": item["rank"],
                "collinear_design": item["collinear_design"],
                "non_unique_minimizer": item["non_unique_minimizer"],
                "interior_minimizer": item["interior_minimizer"],
                "boundary": item["boundary"],
                "active_constraints": item["active_constraints"],
                "constraint_residuals": item["constraint_residuals"],
                "max_constraint_violation": item["max_constraint_violation"],
                "design": {
                    "sum_w": item["design"]["s0"],
                    "sum_wt": item["design"]["s1"],
                    "sum_wt2": item["design"]["s2"],
                    "sum_wy": item["design"]["y0"],
                    "sum_wty": item["design"]["y1"],
                    "sum_wy2": item["design"]["c"],
                    "determinant": item["design"]["determinant"],
                    "hessian_eigen_max": item["design"]["hessian_eigen_max"],
                    "hessian_eigen_min": item["design"]["hessian_eigen_min"],
                    "hessian_condition": item["design"]["hessian_condition"],
                },
            }
            for item in fits
        ],
        "grid_best": {
            "k": grid_best["k"],
            "a": grid_best["a"],
            "b": grid_best["b"],
            "objective": grid_best["objective"],
            "index": grid_best["index"],
            "non_unique_minimizer": grid_best["non_unique_minimizer"],
            "non_unique_within_tolerance": grid_ties > 1,
            "selection_rule": "目标同分时优先较小 k，其后 (a,b) 字典序最小",
        },
        "profile": {
            "global_min_objective": best_objective,
            "cutoff": material["cutoff"],
            "acceptance_threshold": threshold,
            "accepted_k": [item["k"] for item in accepted],
            "accepted_count": len(accepted),
            "grid_count": len(fits),
            "segments": [
                {
                    "start_index": seg[0]["index"],
                    "end_index": seg[-1]["index"],
                    "k_start": seg[0]["k"],
                    "k_end": seg[-1]["k"],
                    "k_values": [item["k"] for item in seg],
                }
                for seg in segments
            ],
            "disconnected": len(segments) > 1,
        },
        "identifiability": {
            "rank_deficient_k": rank_deficient,
            "rank_one_share": len(rank_deficient) / len(fits),
            "accepted_rank_deficient_k": [
                item["k"] for item in accepted if item["collinear_design"]
            ],
            "best_on_boundary": grid_best["boundary"],
            "best_active_constraints": grid_best["active_constraints"],
            "best_max_constraint_violation": grid_best["max_constraint_violation"],
        },
    }


def _edge_roots(v0, direction, stats, level, tolerance):
    """q(v0+s*direction)=level 在 s in [0,1] 上的根。"""
    ax, ay = v0
    dx, dy = direction
    s0, s1, s2 = stats["s0"], stats["s1"], stats["s2"]
    y0, y1 = stats["y0"], stats["y1"]

    quad = s0 * dx * dx + s2 * dy * dy - 2.0 * s1 * dx * dy
    av0 = (s0 * ax - s1 * ay, -s1 * ax + s2 * ay)
    linear = 2.0 * (av0[0] * dx + av0[1] * dy) - 2.0 * (y0 * dx - y1 * dy)
    constant = _q(stats, ax, ay) - level
    scale = max(abs(quad), abs(linear), abs(constant), 1.0)
    if abs(quad) <= tolerance * scale:
        if abs(linear) <= tolerance * scale:
            return []
        root = -constant / linear
        return [root] if -tolerance <= root <= 1.0 + tolerance else []

    discriminant = linear * linear - 4.0 * quad * constant
    disc_scale = max(1.0, linear * linear, abs(4.0 * quad * constant))
    if discriminant < -1e-12 * disc_scale:
        return []
    discriminant = max(0.0, discriminant)
    root_disc = math.sqrt(discriminant)
    roots = [(-linear - root_disc) / (2.0 * quad),
             (-linear + root_disc) / (2.0 * quad)]
    return [r for r in roots if -tolerance <= r <= 1.0 + tolerance]


def _linear_bounds_k(stats, level, c0, c1, tolerance):
    """在 {三角形 ∩ Q<=level} 上求 c0*a+c1*b 的带安全检查的上下界。

    正定目标用椭圆参数化：三角形顶点、边与椭圆交点、以及目标方向与椭圆
    相切的点构成候选集。奇异目标（所有 t 相同）退化为平行带与三角形的
    交多边形，用五条直线的两两交点枚举。两种情形都做可行性和目标检查。
    """
    s0, s1, s2 = stats["s0"], stats["s1"], stats["s2"]
    y0, y1, c = stats["y0"], stats["y1"], stats["c"]
    det = stats["determinant"]
    det_tol = 1e-12 * s0 * s2
    ftol = max(1e-7, tolerance * 100.0)
    qtol = max(1e-7, 1e-9 * max(1.0, abs(level)))

    def feasible(a, b):
        return (-ftol <= a <= 1.0 + ftol and -ftol <= b
                and b <= a + ftol)

    candidates = []
    vertices = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)]

    if det > det_tol:
        # A = [[s0,-s1],[-s1,s2]], h=(y0,-y1), Q = v'A v - 2h'v + c
        astar = (y0 * s2 - s1 * y1) / det
        bstar = (s1 * y0 - s0 * y1) / det
        qmin = c - (y0 * astar - y1 * bstar)
        delta = level - qmin
        if delta < -qtol:
            return None
        delta = max(0.0, delta)
        candidates.append((astar, bstar))

        # 目标方向 p=(c0,c1) 的切点：v* +/- sqrt(delta/(p'A^-1 p)) A^-1 p
        z0 = (s2 * c0 + s1 * c1) / det
        z1 = (s1 * c0 + s0 * c1) / det
        norm2 = c0 * z0 + c1 * z1
        if norm2 > 1e-15 * max(1.0, abs(c0), abs(c1)) ** 2:
            radius = math.sqrt(delta / norm2)
            candidates.append((astar + radius * z0, bstar + radius * z1))
            candidates.append((astar - radius * z0, bstar - radius * z1))

        edges = [((0.0, 0.0), (1.0, 0.0)),
                 ((1.0, 0.0), (0.0, 1.0)),
                 ((0.0, 0.0), (1.0, 1.0))]
        for v0, direction in edges:
            for s in _edge_roots(v0, direction, stats, level, tolerance):
                s = min(1.0, max(0.0, s))
                candidates.append((v0[0] + s * direction[0],
                                   v0[1] + s * direction[1]))
        candidates.extend(vertices)
    else:
        # 秩 1：所有 t 相同。Q = s0 (a - t b - u0)^2 + qmin。
        t = s1 / s0
        u0 = y0 / s0
        qmin = c - y0 * y0 / s0
        delta = level - qmin
        if delta < -qtol:
            return None
        radius = math.sqrt(max(0.0, delta) / s0)
        lower, upper = u0 - radius, u0 + radius

        lines = [
            (1.0, 0.0, 1.0),       # a=1
            (0.0, 1.0, 0.0),       # b=0
            (-1.0, 1.0, 0.0),      # b=a
            (1.0, -t, lower),      # a-t b=lower
            (1.0, -t, upper),      # a-t b=upper
        ]
        for i in range(len(lines)):
            for j in range(i + 1, len(lines)):
                a1, b1, g1 = lines[i]
                a2, b2, g2 = lines[j]
                determinant = a1 * b2 - a2 * b1
                if abs(determinant) <= 1e-14:
                    continue
                a = (g1 * b2 - g2 * b1) / determinant
                b = (a1 * g2 - a2 * g1) / determinant
                candidates.append((a, b))

    feasible_candidates = []
    for a, b in candidates:
        if not feasible(a, b):
            continue
        residual_q = _q(stats, a, b) - level
        if residual_q <= qtol:
            feasible_candidates.append((a, b, residual_q))

    if not feasible_candidates:
        return {
            "complete": False,
            "reason": "可行域内没有通过检查的候选点",
            "lower": None,
            "upper": None,
            "candidate_count": 0,
        }

    values = [(c0 * a + c1 * b, a, b, residual_q)
              for a, b, residual_q in feasible_candidates]
    low = min(values, key=lambda item: (item[0], item[1], item[2]))
    high = max(values, key=lambda item: (item[0], item[1], item[2]))
    low_res = _residuals(low[1], low[2])
    high_res = _residuals(high[1], high[2])
    max_violation = max(0.0, max(max(low_res.values()), max(high_res.values())))

    return {
        "complete": max_violation <= ftol and low[3] <= qtol and high[3] <= qtol,
        "lower": low[0],
        "upper": high[0],
        "arg_min": {"a": low[1], "b": low[2]},
        "arg_max": {"a": high[1], "b": high[2]},
        "candidate_count": len(feasible_candidates),
        "feasibility_check": {
            "max_constraint_violation": max_violation,
            "max_objective_excess": max(low[3], high[3]),
            "feasibility_tolerance": ftol,
            "objective_tolerance": qtol,
        },
    }


def _candidate_report(material, fits, fit_report, candidate_n):
    scale = material["scale"]
    baseline_n = material["baseline_n"]
    threshold = fit_report["profile"]["acceptance_threshold"]
    tolerance = material["solver"]["tolerance"]
    accepted = [item for item in fits if item["accepted"]]
    if not accepted:
        raise ValueError("轮廓接受集合为空")

    baseline_t = [math.exp(-item["k"] * baseline_n / scale)
                  for item in accepted]
    candidate_t = [math.exp(-item["k"] * candidate_n / scale)
                   for item in accepted]

    per_k = []
    lower_bounds = []
    upper_bounds = []
    gain_lowers = []
    gain_uppers = []
    complete = True
    for item, tb, tc in zip(accepted, baseline_t, candidate_t):
        stats = item["design"]
        pred = _linear_bounds_k(stats, threshold, 1.0, -tc, tolerance)
        base = _linear_bounds_k(stats, threshold, 1.0, -tb, tolerance)
        gain = _linear_bounds_k(stats, threshold, 0.0, tb - tc, tolerance)
        a_bound = _linear_bounds_k(stats, threshold, 1.0, 0.0, tolerance)
        b_bound = _linear_bounds_k(stats, threshold, 0.0, 1.0, tolerance)
        if (pred is None or base is None or gain is None
                or a_bound is None or b_bound is None
                or not all(x["complete"] for x in (pred, base, gain,
                                                   a_bound, b_bound))):
            complete = False
        entry = {
            "k": item["k"],
            "profile_gap": item["profile_gap"],
            "prediction": pred,
            "baseline_prediction": base,
            "gain": gain,
            "parameter_intervals": {
                "a": a_bound,
                "b": b_bound,
            },
        }
        per_k.append(entry)
        if pred and pred["lower"] is not None:
            lower_bounds.append(pred["lower"])
            upper_bounds.append(pred["upper"])
        if gain and gain["lower"] is not None:
            gain_lowers.append(gain["lower"])
            gain_uppers.append(gain["upper"])

    if candidate_n > baseline_n:
        added = candidate_n - baseline_n
        cost = (material["cost"]["fixed"]
                + material["cost"]["per_sample"] * Fraction(str(added)))
    else:
        added = 0.0
        cost = Fraction(0)
    cost_value = float(cost)

    if complete and lower_bounds and gain_lowers:
        prediction_interval = {
            "lower": min(lower_bounds),
            "upper": max(upper_bounds),
            "joint_over_full_set": True,
            "status": "certified",
        }
        gain_interval = {
            "lower": min(gain_lowers),
            "upper": max(gain_uppers),
            "joint_over_full_set": True,
            "definition": "同一 (a,b,k) 上优化 f(candidate)-f(baseline)",
            "status": "certified",
            "gain_positive_certified": min(gain_lowers) > 0.0,
            "zero_gain_not_excluded": min(gain_lowers) <= 0.0,
            "not_an_investment_recommendation": True,
        }
    else:
        # 未完成全部子问题时不得输出过窄的正式区间。约束三角形下
        # 预测值必在 [0,1]，且候选规模不小于 baseline 时增益也在 [0,1]。
        prediction_interval = {
            "lower": 0.0,
            "upper": 1.0,
            "joint_over_full_set": True,
            "status": "incomplete",
            "conservative_bracket": True,
            "note": "子问题未全部通过检查，改用三角形隐含的保守区间",
        }
        gain_interval = {
            "lower": 0.0,
            "upper": 1.0,
            "joint_over_full_set": True,
            "definition": "同一 (a,b,k) 上优化 f(candidate)-f(baseline)",
            "status": "incomplete",
            "conservative_bracket": True,
            "gain_positive_certified": False,
            "zero_gain_not_excluded": True,
            "not_an_investment_recommendation": True,
        }

    # 仅用于展示“两个独立区间相减”会给出多宽的区间；正式增益不采用它。
    independent_subtraction = {
        "lower": None,
        "upper": None,
        "note": "此列仅作反例诊断，不是联合增益界",
    }
    if complete:
        baseline_lowers = [entry["baseline_prediction"]["lower"]
                           for entry in per_k
                           if entry["baseline_prediction"]["lower"] is not None]
        baseline_uppers = [entry["baseline_prediction"]["upper"]
                           for entry in per_k
                           if entry["baseline_prediction"]["upper"] is not None]
        if baseline_lowers:
            independent_subtraction["lower"] = (prediction_interval["lower"]
                                                - max(baseline_uppers))
            independent_subtraction["upper"] = (prediction_interval["upper"]
                                                - min(baseline_lowers))

    return {
        "candidate_n": candidate_n,
        "baseline_n": baseline_n,
        "added_samples": added,
        "cost": cost_value,
        "cost_fraction": "%s/%s" % (cost.numerator, cost.denominator),
        "cost_model": "fixed + per_sample*(candidate-baseline)",
        "prediction_interval": prediction_interval,
        "gain_interval": gain_interval,
        "independent_subtraction_diagnostic": independent_subtraction,
        "per_k": per_k,
        "decision_readout": {
            "gain_lower": gain_interval["lower"],
            "gain_upper": gain_interval["upper"],
            "cannot_rule_out_no_gain": gain_interval["zero_gain_not_excluded"],
            "note": ("若下界为 0，则在给定模型与 cutoff 下无法区分"
                     "“扩大采集”与“没有可辨识预测增益”；本输出不承诺收益"),
        },
    }


def analyze(value, candidate_ns=None):
    """完整报告：网格拟合、轮廓分段、候选预测/增益联合界与成本。"""
    material = parse_material(value)
    report = fit_grid(value)
    fits = []
    for item in report["grid"]:
        # 重建内部统计量（不依赖外部可变状态）
        fits.append({
            "k": item["k"],
            "index": item["index"],
            "accepted": item["accepted"],
            "profile_gap": item["profile_gap"],
            "design": _design(material["rows"], item["k"], material["scale"]),
        })

    candidates = material["candidate_n"] if candidate_ns is None else candidate_ns
    validated = []
    for n in candidates:
        n = finite(n)
        if n < material["baseline_n"]:
            raise ValueError("候选采集规模不得小于 baseline_n")
        validated.append(n)

    rows = []
    for n in validated:
        rows.append(_candidate_report(material, fits, report, n))

    # 同一参数集合 S 上直接比较两个候选，判断谁更好仍无法区分。
    comparisons = []
    for i in range(len(validated)):
        for j in range(i + 1, len(validated)):
            ni, nj = validated[i], validated[j]
            lower, upper = None, None
            complete = True
            lows, highs = [], []
            for item in fits:
                if not item["accepted"]:
                    continue
                k = item["k"]
                ti = math.exp(-k * ni / material["scale"])
                tj = math.exp(-k * nj / material["scale"])
                direction = _linear_bounds_k(
                    item["design"],
                    report["profile"]["acceptance_threshold"],
                    0.0, tj - ti, material["solver"]["tolerance"])
                if direction is None or not direction["complete"]:
                    complete = False
                else:
                    lows.append(direction["lower"])
                    highs.append(direction["upper"])
            if lows:
                lower, upper = min(lows), max(highs)
            comparisons.append({
                "candidate_i": ni,
                "candidate_j": nj,
                "difference": "f(n_i)-f(n_j)",
                "lower": lower,
                "upper": upper,
                "complete": complete,
                "indistinguishable": (lower is not None
                                      and lower <= 0.0 <= upper),
                "note": "在同一 (a,b,k) 上联合求界，非独立区间相减",
            })

    report["candidates"] = rows
    report["pairwise_candidate_comparisons"] = comparisons
    report["candidate_table"] = [
        {
            "candidate_n": row["candidate_n"],
            "added_samples": row["added_samples"],
            "cost": row["cost"],
            "prediction_lower": row["prediction_interval"]["lower"],
            "prediction_upper": row["prediction_interval"]["upper"],
            "gain_lower": row["gain_interval"]["lower"],
            "gain_upper": row["gain_interval"]["upper"],
            "gain_positive_certified": row["gain_interval"]["gain_positive_certified"],
            "cannot_rule_out_no_gain": row["gain_interval"]["zero_gain_not_excluded"],
            "status": row["prediction_interval"]["status"],
        }
        for row in rows
    ]
    return report


def brute_force_bounds(value, candidate_n, steps=200):
    """小范围密集 (a,b) 穷举，用于交叉检查解析预测包络。

    这不是正式求界器；它只返回三角形成交网格上的观测极值，供测试/诊断
    证明解析包络不比穷举更窄。steps 是 a 与 b 方向的网格数。
    """
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
        raise ValueError("steps 必须是正整数")
    candidate_n = finite(candidate_n)
    material = parse_material(value)
    report = fit_grid(value)
    threshold = report["profile"]["acceptance_threshold"]
    accepted = [item["k"] for item in report["grid"] if item["accepted"]]
    low = math.inf
    high = -math.inf
    evaluated = 0
    for k in accepted:
        stats = _design(material["rows"], k, material["scale"])
        for ia in range(steps + 1):
            a = ia / steps
            for ib in range(steps + 1):
                b = a * ib / steps
                if b > a:
                    continue
                if _q(stats, a, b) <= threshold + 1e-7:
                    f = a - b * math.exp(-k * candidate_n / material["scale"])
                    low = min(low, f)
                    high = max(high, f)
                    evaluated += 1
    return {
        "candidate_n": candidate_n,
        "lower": low if evaluated else None,
        "upper": high if evaluated else None,
        "evaluated_points": evaluated,
        "steps": steps,
        "note": "仅覆盖三角形成交网格，不插入也未做连续求界",
    }


def best_curves_only_bounds(value, candidate_n):
    """反例诊断：只连接每个 k 的最佳 (a,b)，忽略同一 k 内不确定性。"""
    material = parse_material(value)
    report = fit_grid(value)
    values = []
    for item in report["grid"]:
        if not item["accepted"]:
            continue
        t = math.exp(-item["k"] * candidate_n / material["scale"])
        values.append(item["a"] - item["b"] * t)
    return {
        "candidate_n": candidate_n,
        "lower": min(values) if values else None,
        "upper": max(values) if values else None,
        "k_count": len(values),
        "note": ("只取每 k 最佳参数会漏掉该 k 内满足 profile 阈值的参数，"
                 "正式包络必须在整个 S 上优化"),
    }
