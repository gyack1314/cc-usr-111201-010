"""核对饱和拟合、轮廓集合、联合预测界与反例诊断。"""
import math
import unittest

from analysis_data.curve import (
    analyze,
    best_curves_only_bounds,
    brute_force_bounds,
    fit_grid,
    parse_material,
)
from analysis_data.io import load


def _objective(rows, k, a, b, scale):
    return sum(
        (row["y"] - a + b * math.exp(-k * row["n"] / scale)) ** 2
        / row["variance"]
        for row in rows
    )


class CurveTests(unittest.TestCase):
    def test_grid_objective_and_constraints_are_recomputable(self):
        for path in ("data/normal.json", "data/flat-platform.json",
                     "data/disconnected-profile.json"):
            value, _ = load(path)
            report = fit_grid(value)
            rows = parse_material(value)["rows"]
            scale = parse_material(value)["scale"]
            for item in report["grid"]:
                q = _objective(rows, item["k"], item["a"], item["b"], scale)
                self.assertAlmostEqual(q, item["objective"], places=7)
                self.assertLessEqual(item["a"], 1.0 + 1e-9)
                self.assertGreaterEqual(item["b"], -1e-9)
                self.assertLessEqual(item["b"], item["a"] + 1e-9)
                self.assertLessEqual(item["max_constraint_violation"], 1e-9)
                self.assertFalse(math.isnan(item["objective"]))
                # 网格最佳解只能来自声明的 k 网格
                self.assertIn(item["k"], report["source"]["k_grid"])

    def test_all_runs_and_sources_are_preserved(self):
        value, _ = load("data/normal.json")
        report = fit_grid(value)
        ids = [row["run_id"] for row in report["source"]["observations"]]
        self.assertEqual(len(ids), 12)
        self.assertEqual(len(set(ids)), 12)
        self.assertEqual(report["source"]["scale_group_count"], 4)
        for row in report["source"]["observations"]:
            self.assertAlmostEqual(row["weight"], 1.0 / row["variance"])

        # 少一个种子（即使它偏离均值）会改变目标，说明没有按结果挑选/删行。
        trimmed = dict(value)
        trimmed["runs"] = value["runs"][:-1]
        q_full = fit_grid(value)["grid_best"]["objective"]
        q_trim = fit_grid(trimmed)["grid_best"]["objective"]
        self.assertNotAlmostEqual(q_full, q_trim, places=6)

    def test_boundary_optimum_is_reported(self):
        value, _ = load("data/boundary-a.json")
        report = fit_grid(value)
        best = report["grid_best"]
        self.assertEqual(best["k"], 0.05)  # 同分时优先较小 k
        self.assertAlmostEqual(best["a"], 1.0, places=12)
        self.assertAlmostEqual(best["b"], 0.0, places=12)
        self.assertTrue(report["identifiability"]["best_on_boundary"])
        self.assertIn("a<=1", report["identifiability"]["best_active_constraints"])
        self.assertIn("-b<=0", report["identifiability"]["best_active_constraints"])

    def test_collinear_single_scale_is_diagnosed(self):
        value, _ = load("data/single-scale.json")
        report = fit_grid(value)
        self.assertEqual(report["identifiability"]["rank_one_share"], 1.0)
        self.assertTrue(report["identifiability"]["rank_deficient_k"])
        self.assertTrue(report["identifiability"]["accepted_rank_deficient_k"])
        for item in report["grid"]:
            self.assertEqual(item["rank"], 1)
            self.assertTrue(item["collinear_design"])
            # 共线时 a、b 不可分别识别，但目标仍可核对
            self.assertAlmostEqual(
                _objective(parse_material(value)["rows"], item["k"],
                           item["a"], item["b"], 1000.0),
                item["objective"], places=7)

    def test_near_flat_platform_keeps_profile_uncertainty(self):
        value, _ = load("data/flat-platform.json")
        report = fit_grid(value)
        self.assertEqual(report["grid_best"]["objective"], 0.0)
        self.assertEqual(report["grid_best"]["b"], 0.0)
        self.assertTrue(report["profile"]["accepted_count"] >= 2)
        best_only = best_curves_only_bounds(value, 2500)
        full = analyze(value, [2500])["candidates"][0]["prediction_interval"]
        best_only_width = best_only["upper"] - best_only["lower"]
        full_width = full["upper"] - full["lower"]
        self.assertAlmostEqual(best_only_width, 0.0, places=12)
        self.assertGreater(full_width, 0.03)
        # 只连最佳曲线会得到过窄的正式区间，密集穷举也支持完整集合更宽
        brute = brute_force_bounds(value, 2500, steps=200)
        self.assertLessEqual(full["lower"], brute["lower"] + 1e-9)
        self.assertGreaterEqual(full["upper"], brute["upper"] - 1e-9)

    def test_disconnected_acceptance_segments_are_kept(self):
        value, _ = load("data/disconnected-profile.json")
        report = fit_grid(value)
        self.assertTrue(report["profile"]["disconnected"])
        segments = report["profile"]["segments"]
        self.assertEqual(len(segments), 2)
        self.assertEqual(segments[0]["k_values"], [0.5, 1])
        self.assertEqual(segments[1]["k_values"], [4, 8])
        self.assertNotIn(2, report["profile"]["accepted_k"])
        self.assertIn(8, report["profile"]["accepted_k"])

    def test_prediction_bounds_crosschecked_by_dense_ab_enumeration(self):
        for path, candidate in (("data/flat-platform.json", 2500),
                                ("data/single-scale.json", 4000),
                                ("data/disconnected-profile.json", 10000)):
            value, _ = load(path)
            result = analyze(value, [candidate])["candidates"][0]
            analytic = result["prediction_interval"]
            brute = brute_force_bounds(value, candidate, steps=300)
            self.assertEqual(analytic["status"], "certified")
            self.assertLessEqual(analytic["lower"], brute["lower"] + 1e-9)
            self.assertGreaterEqual(analytic["upper"], brute["upper"] - 1e-9)
            # 三角形网格的 Lipschitz 包络：差不超过两方向步长之和
            self.assertLessEqual(brute["lower"] - analytic["lower"], 2.0 / 300 + 1e-9)
            self.assertLessEqual(analytic["upper"] - brute["upper"], 2.0 / 300 + 1e-9)

    def test_gain_is_optimized_jointly_and_not_by_interval_subtraction(self):
        value, _ = load("data/flat-platform.json")
        report = analyze(value)
        for candidate in report["candidates"]:
            gain = candidate["gain_interval"]
            naive = candidate["independent_subtraction_diagnostic"]
            self.assertEqual(gain["status"], "certified")
            self.assertEqual(gain["definition"],
                             "同一 (a,b,k) 上优化 f(candidate)-f(baseline)")
            self.assertGreaterEqual(gain["lower"], naive["lower"] - 1e-12)
            self.assertLessEqual(gain["upper"], naive["upper"] + 1e-12)
            self.assertLess(gain["upper"] - gain["lower"],
                            naive["upper"] - naive["lower"])  # 联合界更窄
            # 近平台资料无法排除“没有可辨识增益”
            self.assertTrue(gain["zero_gain_not_excluded"])
            self.assertFalse(gain["gain_positive_certified"])

    def test_candidate_table_cost_and_pairwise_decisions(self):
        value, _ = load("data/normal.json")
        report = analyze(value)
        self.assertEqual([row["added_samples"] for row in report["candidate_table"]],
                         [500, 2000, 8000])
        self.assertEqual([row["cost"] for row in report["candidate_table"]],
                         [850.0, 3100.0, 12100.0])
        for pair in report["pairwise_candidate_comparisons"]:
            self.assertIn("indistinguishable", pair)
            self.assertTrue(pair["complete"])
        first = report["candidates"][0]
        self.assertTrue(first["gain_interval"]["gain_positive_certified"])
        self.assertFalse(first["gain_interval"]["zero_gain_not_excluded"])

    def test_candidate_below_baseline_is_rejected(self):
        value, _ = load("data/normal.json")
        with self.assertRaises(ValueError):
            analyze(value, [1500])

    def test_equal_to_baseline_has_zero_cost_and_gain(self):
        value, _ = load("data/normal.json")
        result = analyze(value, [2000])["candidates"][0]
        self.assertEqual(result["added_samples"], 0.0)
        self.assertEqual(result["cost"], 0.0)
        self.assertEqual(result["gain_interval"]["lower"], 0.0)
        self.assertEqual(result["gain_interval"]["upper"], 0.0)
        self.assertTrue(result["gain_interval"]["zero_gain_not_excluded"])

    def test_invalid_variance_is_rejected(self):
        value, _ = load("data/invalid-variance.json")
        with self.assertRaises(ValueError):
            fit_grid(value)


if __name__ == "__main__":
    unittest.main()
