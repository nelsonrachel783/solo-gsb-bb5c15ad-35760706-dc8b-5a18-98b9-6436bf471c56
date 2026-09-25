"""备品试管替换规划测试: python -m unittest discover -s tests

核心断言: 一对一映射完全枚举的结果(总质量差最小 → 残余 |U₁| 最低 →
按待换管号顺序对应的备品编号序列裁决)必须与独立的全排列参考实现一致;
备品不重复使用、备品落在待换管原槽(角度/半径不变)、未换管保持原样;
无达标映射时必须返回残余最低的映射及超限量且不误报达标; 待换管 1~4 支、
备品 1~12 支、编号唯一且不与原方案冲突等非法请求必须拒绝; 规划过程不
修改入参(只读)。
"""
import copy
import itertools
import math
import random
import unittest

from app.spare import plan_spare_replacement


def _plan(pid, placements, radius=100.0, slots_n=12, enabled=None, name=None,
          request_tubes=None):
    """placements: {槽号: (试管编号, 质量, locked)}; 角度按槽号均布。"""
    if enabled is None:
        enabled = set(range(slots_n))

    def occ(i):
        for s, item in placements.items():
            if s == i:
                return item[0]
        return None

    slot_state = [
        {
            "slot": i,
            "angle": 360.0 * i / slots_n,
            "enabled": i in enabled,
            "tube_id": occ(i),
            "locked": False,
        }
        for i in range(slots_n)
    ]
    pls = []
    for s, item in placements.items():
        t, m = item[0], item[1]
        locked = item[2] if len(item) > 2 else False
        pls.append({
            "tube_id": t, "mass": m, "slot": s,
            "angle": 360.0 * s / slots_n, "locked": locked,
        })
    return {
        "id": pid,
        "name": name or f"方案{pid}",
        "created_at": "2026-09-25T00:00:00+00:00",
        "request": {
            "radius_mm": radius,
            "tubes": (
                [{"id": t, "mass": m} for t, m, *_ in placements.values()]
                if request_tubes is None else request_tubes
            ),
        },
        "result": {
            "config": {"radius_mm": radius},
            "slot_state": slot_state,
            "placements": pls,
            "metrics": {"u1_mag_g_mm": 0.0},
        },
    }


def _brute(plan, replace_ids, spares, limit):
    """独立全排列参考(与被测模块同选择规则)。

    返回 {"feasible": bool, "mag": 最终|U₁|, "diff": 总质量差,
          "key": 备品编号序列(待换管号升序), "count": 最优映射条数}。
    """
    res = plan["result"]
    R = res["config"]["radius_mm"]
    pmap = {p["tube_id"]: p for p in res["placements"] if p["slot"] is not None}
    targets = sorted(replace_ids)
    sp = sorted(spares, key=lambda x: x["tube_id"])
    smass = {x["tube_id"]: x["mass_g"] for x in sp}
    fixed = sum(
        (R * pmap[t]["mass"]
         * complex(math.cos(math.radians(pmap[t]["angle"])),
                   math.sin(math.radians(pmap[t]["angle"])))
         for t in pmap if t not in set(targets)),
        0j,
    )

    leaves = []
    for perm in itertools.permutations(range(len(sp)), len(targets)):
        u = fixed
        diff = 0.0
        key = []
        for j, si in enumerate(perm):
            tp = pmap[targets[j]]
            m = sp[si]["mass_g"]
            u += R * m * complex(
                math.cos(math.radians(tp["angle"])),
                math.sin(math.radians(tp["angle"])))
            diff += abs(m - tp["mass"])
            key.append(sp[si]["tube_id"])
        leaves.append((abs(u), diff, tuple(key)))

    tol = 1e-9 * max(1.0, limit)
    feas = [leaf for leaf in leaves if leaf[0] <= limit + tol]

    def choose_diff_first(pool):
        bd = min(d for _, d, _ in pool)
        dt = 1e-9 * max(1.0, bd)
        tied = [leaf for leaf in pool if leaf[1] <= bd + dt]
        bm = min(m for m, _, _ in tied)
        mt = 1e-9 * max(1.0, bm)
        opt = [leaf for leaf in tied if leaf[0] <= bm + mt]
        key = min(k for _, _, k in opt)
        m, d = next((m, d) for m, d, k in opt if k == key)
        return m, d, key, len(opt)

    def choose_mag_first(pool):
        bm = min(m for m, _, _ in pool)
        mt = 1e-9 * max(1.0, bm)
        tied = [leaf for leaf in pool if leaf[0] <= bm + mt]
        bd = min(d for _, d, _ in tied)
        dt = 1e-9 * max(1.0, bd)
        opt = [leaf for leaf in tied if leaf[1] <= bd + dt]
        key = min(k for _, _, k in opt)
        m, d = next((m, d) for m, d, k in opt if k == key)
        return m, d, key, len(opt)

    if feas:
        m, d, key, cnt = choose_diff_first(feas)
        return {"feasible": True, "mag": m, "diff": d, "key": key,
                "count": cnt, "evaluated": len(leaves)}
    m, d, key, cnt = choose_mag_first(leaves)
    return {"feasible": False, "mag": m, "diff": d, "key": key,
            "count": cnt, "evaluated": len(leaves)}


class SpareReplacementTests(unittest.TestCase):
    def test_single_tube_exact_match(self):
        """单管方案: 同质量备品放对径? 不——备品落原槽, 同质量则 |U₁| 不变。"""
        plan = _plan(1, {0: (1, 50.0)})
        out = plan_spare_replacement(
            plan, [1], [{"tube_id": 101, "mass_g": 50.0}], 6000.0)
        self.assertTrue(out["feasible"])
        self.assertAlmostEqual(out["final"]["u1_mag_g_mm"], 5000.0, places=6)
        self.assertEqual(out["total_mass_diff_g"], 0.0)
        self.assertEqual(out["replacements"][0]["spare_tube_id"], 101)
        self.assertEqual(out["replacements"][0]["slot"], 0)
        self.assertEqual(out["evaluated_mappings"], 1)

    def test_choose_smaller_mass_meets_limit(self):
        """单管方案: 限值 4000, 备品 40g(4000) 与 60g(6000) 都给出,
        达标映射中总质量差最小(均为 10)→ 残余最低 → 40g 胜出。"""
        plan = _plan(1, {0: (1, 50.0)})
        out = plan_spare_replacement(
            plan, [1],
            [{"tube_id": 101, "mass_g": 40.0},
             {"tube_id": 102, "mass_g": 60.0}],
            4500.0)
        self.assertTrue(out["feasible"])
        # 40g 达标(4000 ≤ 4500), 60g 不达标; 只有一个达标映射
        self.assertEqual(out["replacements"][0]["spare_tube_id"], 101)
        self.assertAlmostEqual(out["final"]["u1_mag_g_mm"], 4000.0, places=6)

    def test_total_mass_diff_primary_over_residual(self):
        """总质量差优先于残余 |U₁|: 两支待换管, 限值很宽松时取总质量差最小
        的映射, 即使其残余 |U₁| 略大。"""
        # 管 1=50g@0°, 管 2=50g@180° → 基线 |U₁|=0
        plan = _plan(1, {0: (1, 50.0), 6: (2, 50.0)})
        # 备品: 49/51 给两管各差 1, 总差 2; 50/50 总差 0
        out = plan_spare_replacement(
            plan, [1, 2],
            [{"tube_id": 101, "mass_g": 50.0},
             {"tube_id": 102, "mass_g": 50.0},
             {"tube_id": 103, "mass_g": 49.0},
             {"tube_id": 104, "mass_g": 51.0}],
            1e9)
        self.assertTrue(out["feasible"])
        # 总质量差 0 的映射(两个 50g 备品)胜出; 并列时按备品编号序列裁决
        self.assertEqual(out["total_mass_diff_g"], 0.0)
        self.assertEqual(
            [r["spare_tube_id"]
             for r in sorted(out["replacements"],
                             key=lambda r: r["original_tube_id"])],
            [101, 102])

    def test_tie_break_by_spare_id_sequence(self):
        """总质量差与残余 |U₁| 都并列时, 按待换管号顺序的备品编号序列裁决。"""
        # 对称两槽 0/6: 备品等质量时两种指派残余相同
        plan = _plan(1, {0: (1, 50.0), 6: (2, 50.0)})
        out = plan_spare_replacement(
            plan, [1, 2],
            [{"tube_id": 101, "mass_g": 50.0},
             {"tube_id": 102, "mass_g": 50.0}],
            10.0)
        self.assertTrue(out["feasible"])
        self.assertEqual(
            [r["spare_tube_id"]
             for r in sorted(out["replacements"],
                             key=lambda r: r["original_tube_id"])],
            [101, 102])
        self.assertEqual(out["optimal_mapping_count"], 2)
        self.assertTrue(out["tie_broken"])

    def test_no_feasible_returns_min_residual(self):
        """无达标映射: 返回残余最低的映射、超限量, 明确不达标。"""
        # 单管 50g@0°: 备品 40g/60g, 限值 1000 → 最低残余 4000 仍超限
        plan = _plan(1, {0: (1, 50.0)})
        out = plan_spare_replacement(
            plan, [1],
            [{"tube_id": 101, "mass_g": 40.0},
             {"tube_id": 102, "mass_g": 60.0}],
            1000.0)
        self.assertFalse(out["feasible"])
        self.assertFalse(out["within_limit"])
        self.assertEqual(out["replacements"][0]["spare_tube_id"], 101)
        self.assertAlmostEqual(out["final"]["u1_mag_g_mm"], 4000.0, places=6)
        self.assertAlmostEqual(out["excess_g_mm"], 3000.0, places=6)
        self.assertIn("不达标", out["reason"])

    def test_random_cross_check_against_brute(self):
        """多组随机方案与备品: 模块结果必须与独立全排列参考完全一致。"""
        rng = random.Random(20260925)
        for case in range(60):
            n_slots = rng.choice((6, 8, 12))
            n_tubes = rng.randint(1, 6)
            slots = rng.sample(range(n_slots), n_tubes)
            placements = {}
            for i, s in enumerate(slots, start=1):
                placements[s] = (i, round(rng.uniform(10.0, 80.0), 4))
            radius = rng.choice((50.0, 100.0))
            plan = _plan(case + 1, placements, radius=radius,
                         slots_n=n_slots)
            k = rng.randint(1, min(4, n_tubes))
            targets = rng.sample(list(range(1, n_tubes + 1)), k)
            n_sp = rng.randint(k, 12)
            base_id = n_tubes + 100
            spares = [
                {"tube_id": base_id + j,
                 "mass_g": round(rng.uniform(5.0, 90.0), 4)}
                for j in range(n_sp)
            ]
            limit = rng.choice((0.0, 1.0, 100.0, 1000.0, 5000.0, 1e9))
            plan_snap = copy.deepcopy(plan)
            out = plan_spare_replacement(
                plan, targets, spares, limit)
            ref = _brute(plan_snap, targets, spares, limit)
            self.assertEqual(out["feasible"], ref["feasible"], case)
            self.assertAlmostEqual(
                out["final"]["u1_mag_g_mm"], ref["mag"], places=6,
                msg=f"case {case}")
            self.assertAlmostEqual(
                out["total_mass_diff_g"], ref["diff"], places=6,
                msg=f"case {case}")
            key = tuple(
                next(r["spare_tube_id"] for r in out["replacements"]
                     if r["original_tube_id"] == t)
                for t in sorted(targets)
            )
            self.assertEqual(key, ref["key"], f"case {case}")
            self.assertEqual(out["evaluated_mappings"], ref["evaluated"])
            self.assertEqual(
                out["optimal_mapping_count"], ref["count"],
                f"case {case}")
            # 只读: 规划不改入参
            self.assertEqual(plan, plan_snap)

    def test_unreplaced_tubes_kept_and_final_tubes(self):
        """未换管保持原质量与落槽; 最终摆放逐槽完整且总质量差正确。"""
        plan = _plan(2, {0: (1, 50.0), 3: (2, 51.0), 6: (3, 52.0)})
        out = plan_spare_replacement(
            plan, [1, 3],
            [{"tube_id": 201, "mass_g": 48.0},
             {"tube_id": 202, "mass_g": 55.0}],
            1e9)
        self.assertTrue(out["feasible"])
        by_orig = {r["original_tube_id"]: r for r in out["replacements"]}
        # 备品落在待换管原槽: 管1→槽0, 管3→槽6
        self.assertEqual(by_orig[1]["slot"], 0)
        self.assertEqual(by_orig[3]["slot"], 6)
        self.assertAlmostEqual(by_orig[1]["mass_diff_g"], 2.0, places=9)
        self.assertAlmostEqual(by_orig[3]["mass_diff_g"], 3.0, places=9)
        self.assertAlmostEqual(out["total_mass_diff_g"], 5.0, places=9)
        # 逐槽最终摆放: 三槽齐全, 管 2 未换
        self.assertEqual(len(out["tubes"]), 3)
        t2 = next(t for t in out["tubes"] if t["original_tube_id"] == 2)
        self.assertFalse(t2["replaced"])
        self.assertEqual(t2["final_tube_id"], 2)
        self.assertEqual(t2["final_mass_g"], 51.0)
        # 未使用备品列表
        self.assertEqual([x["tube_id"] for x in out["unused_spares"]], [])
        # 独立复算最终 |U₁|
        ux = uy = 0.0
        for t in out["tubes"]:
            a = math.radians(t["angle_deg"])
            ux += t["final_mass_g"] * 100.0 * math.cos(a)
            uy += t["final_mass_g"] * 100.0 * math.sin(a)
        self.assertAlmostEqual(
            math.hypot(ux, uy), out["final"]["u1_mag_g_mm"], places=6)
        # 基线与方案几何一致
        self.assertAlmostEqual(out["baseline"]["u1_mag_g_mm"],
                               abs(complex(5000, 0)
                                   + 5100j
                                   + 5200 * complex(-1, 0)), places=6)

    def test_validation_errors(self):
        plan = _plan(1, {0: (1, 50.0), 6: (2, 50.0)})
        good_spares = [{"tube_id": 101, "mass_g": 49.0},
                       {"tube_id": 102, "mass_g": 51.0}]

        with self.assertRaises(ValueError):  # 待换管为空
            plan_spare_replacement(plan, [], good_spares, 1000.0)
        with self.assertRaises(ValueError):  # 待换管超过 4 支
            plan_spare_replacement(plan, [1, 2, 3, 4, 5], good_spares, 1000.0)
        with self.assertRaises(ValueError):  # 待换管重复
            plan_spare_replacement(plan, [1, 1], good_spares, 1000.0)
        with self.assertRaises(ValueError):  # 待换管不在方案中
            plan_spare_replacement(plan, [9], good_spares, 1000.0)
        with self.assertRaises(ValueError):  # 备品为空
            plan_spare_replacement(plan, [1], [], 1000.0)
        with self.assertRaises(ValueError):  # 备品超过 12 支
            plan_spare_replacement(
                plan, [1],
                [{"tube_id": 200 + i, "mass_g": 50.0} for i in range(13)],
                1000.0)
        with self.assertRaises(ValueError):  # 备品编号重复
            plan_spare_replacement(
                plan, [1],
                [{"tube_id": 101, "mass_g": 50.0},
                 {"tube_id": 101, "mass_g": 51.0}], 1000.0)
        with self.assertRaises(ValueError):  # 备品编号与原方案冲突
            plan_spare_replacement(
                plan, [1],
                [{"tube_id": 1, "mass_g": 50.0},
                 {"tube_id": 2, "mass_g": 50.0}], 1000.0)
        with self.assertRaises(ValueError):  # 备品质量非正
            plan_spare_replacement(
                plan, [1], [{"tube_id": 101, "mass_g": 0.0}], 1000.0)
        with self.assertRaises(ValueError):  # 备品质量非有限
            plan_spare_replacement(
                plan, [1], [{"tube_id": 101, "mass_g": float("inf")}],
                1000.0)
        with self.assertRaises(ValueError):  # 备品少于待换管
            plan_spare_replacement(
                plan, [1, 2], [{"tube_id": 101, "mass_g": 50.0}], 1000.0)
        with self.assertRaises(ValueError):  # 限值为负
            plan_spare_replacement(plan, [1], good_spares, -1.0)
        with self.assertRaises(ValueError):  # 限值非有限
            plan_spare_replacement(
                plan, [1], good_spares, float("nan"))

    def test_no_placed_tubes(self):
        plan = _plan(1, {})
        with self.assertRaises(ValueError):
            plan_spare_replacement(
                plan, [1], [{"tube_id": 101, "mass_g": 50.0}], 1000.0)

    def test_max_enumeration_size(self):
        """4 支待换管 × 12 支备品: 叶子数恰为 P(12,4)=11880。"""
        placements = {s: (i + 1, 50.0)
                      for i, s in enumerate((0, 3, 6, 9))}
        plan = _plan(1, placements)
        spares = [{"tube_id": 100 + i, "mass_g": 40.0 + i}
                  for i in range(12)]
        out = plan_spare_replacement(plan, [1, 2, 3, 4], spares, 0.0)
        self.assertEqual(out["evaluated_mappings"], 11880)
        # 限值 0 大概率不达标 → 必须给出残余最低映射与超限量
        if not out["feasible"]:
            self.assertFalse(out["within_limit"])
            self.assertGreater(out["excess_g_mm"], 0.0)

    def test_locked_tube_can_be_replaced_flag_propagated(self):
        """锁定管也允许被选为待换管(替换是换管而非移槽), 锁定标记带到明细。"""
        plan = _plan(1, {0: (1, 50.0, True)})
        out = plan_spare_replacement(
            plan, [1], [{"tube_id": 101, "mass_g": 50.0}], 1e9)
        self.assertTrue(out["replacements"][0]["locked_original"])
        self.assertTrue(out["tubes"][0]["locked"])


if __name__ == "__main__":
    unittest.main()
