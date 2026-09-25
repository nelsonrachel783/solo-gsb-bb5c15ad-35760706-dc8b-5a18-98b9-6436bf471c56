"""库存配重补偿测试: python -m unittest discover -s tests

核心断言: 按放置枚数分层的完全枚举结果(最少放置枚数、残余 |U₁|、最优组合
条数与按槽号及对应质量序列的并列裁决)必须与独立的全枚举参考实现一致; 配重
只放空闲启用槽、单槽至多一枚、每种用量不超库存(同质量种类合并库存); 限额内
无法达标时必须给出残余最低的组合及超限量且不误报达标; 空闲启用槽超过 16 个、
配重超过 6 种或放置上限 0~3 之外的请求必须拒绝; 规划过程不修改入参(只读)。
"""
import copy
import itertools
import math
import random
import unittest

from app.counterweight import plan_counterweight_compensation


def _plan(pid, placements, radius=100.0, slots_n=12, enabled=None, name=None):
    """placements: {槽号: (试管编号, 质量)}; 角度按槽号均布。"""
    if enabled is None:
        enabled = set(range(slots_n))
    slot_state = [
        {
            "slot": i,
            "angle": 360.0 * i / slots_n,
            "enabled": i in enabled,
            "tube_id": next(
                (t for s, (t, m) in placements.items() if s == i), None
            ),
            "locked": False,
        }
        for i in range(slots_n)
    ]
    pls = [
        {
            "tube_id": t,
            "mass": m,
            "slot": s,
            "angle": 360.0 * s / slots_n,
            "locked": False,
        }
        for s, (t, m) in placements.items()
    ]
    return {
        "id": pid,
        "name": name or f"方案{pid}",
        "created_at": "2026-09-25T00:00:00+00:00",
        "request": {"radius_mm": radius},
        "result": {
            "config": {"radius_mm": radius},
            "slot_state": slot_state,
            "placements": pls,
            "metrics": {"u1_mag_g_mm": 0.0},
        },
    }


def _single_plan(pid=1, mass=50.0, slot=0, slots_n=12, enabled=None):
    """单管方案: 一支试管压在 slot 上, 补偿前 |U₁| = mass·R。"""
    return _plan(pid, {slot: (1, mass)}, slots_n=slots_n, enabled=enabled)


def _brute(plan, weights, limit, max_weights):
    """独立全枚举参考(itertools 组合 × 笛卡尔积指派, 选择规则与被测模块一致)。

    返回 ("ok", 最少枚数, 残余|U₁|, (槽位序列, 质量序列), 最优组合条数)
    或 ("no", (残余|U₁|, (槽位序列, 质量序列), 枚数, 并列条数))。
    """
    res = plan["result"]
    radius = res["config"]["radius_mm"]
    # 同质量合并库存并截断到放置上限
    stock = {}
    for w in weights:
        m = float(w["mass_g"])
        stock[m] = min(max_weights, stock.get(m, 0) + min(int(w["stock"]), max_weights))
    masses = sorted(m for m, s in stock.items() if s > 0)
    free = sorted(
        (s["slot"], s["angle"]) for s in res["slot_state"]
        if s["enabled"] and s["tube_id"] is None
    )
    ux = uy = 0.0
    for p in res["placements"]:
        if p["slot"] is None:
            continue
        a = math.radians(p["angle"])
        ux += p["mass"] * radius * math.cos(a)
        uy += p["mass"] * radius * math.sin(a)
    base = complex(ux, uy)
    lim_tol = 1e-9 * max(1.0, limit)

    best_effort = None
    for k in range(0, max_weights + 1):
        leaves = []
        if k == 0:
            leaves.append((abs(base), ((), ())))
        else:
            for slots in itertools.combinations(range(len(free)), k):
                vecs = []
                for si in slots:
                    a = math.radians(free[si][1])
                    vecs.append(radius * complex(math.cos(a), math.sin(a)))
                for assign in itertools.product(range(len(masses)), repeat=k):
                    if any(assign.count(t) > stock[masses[t]]
                           for t in range(len(masses))):
                        continue
                    u = base + sum(masses[t] * v for t, v in zip(assign, vecs))
                    key = (tuple(free[si][0] for si in slots),
                           tuple(masses[t] for t in assign))
                    leaves.append((abs(u), key))
        if not leaves:
            continue
        bm = min(m for m, _ in leaves)
        btol = 1e-9 * max(1.0, bm)
        btied = [(m, key) for m, key in leaves if m <= bm + btol]
        bkey = min(key for _, key in btied)
        bmag = next(m for m, key in btied if key == bkey)
        if best_effort is None or bmag < best_effort[0] - 1e-9:
            best_effort = (bmag, bkey, k, len(btied))
        feas = [(m, key) for m, key in leaves if m <= limit + lim_tol]
        if feas:
            fm = min(m for m, _ in feas)
            ftol = 1e-9 * max(1.0, fm)
            ftied = [(m, key) for m, key in feas if m <= fm + ftol]
            fkey = min(key for _, key in ftied)
            fmag = next(m for m, key in ftied if key == fkey)
            return ("ok", k, fmag, fkey, len(ftied))
    return ("no", best_effort)


class TestCounterweightCompensation(unittest.TestCase):
    def test_baseline_already_within_limit(self):
        # 十字对称四管, 标称 |U₁|=0 ≤ 限值: 无需放置(0 枚)
        p = _plan(1, {0: (1, 50.0), 3: (2, 50.0), 6: (3, 50.0), 9: (4, 50.0)})
        r = plan_counterweight_compensation(
            p, [{"mass_g": 5.0, "stock": 3}],
            u1_limit_g_mm=100.0, max_weights=3)
        self.assertTrue(r["feasible"])
        self.assertTrue(r["within_limit"])
        self.assertEqual(r["weights_used"], 0)
        self.assertEqual(r["placements"], [])
        self.assertEqual(r["optimal_combination_count"], 1)
        self.assertAlmostEqual(r["final"]["u1_mag_g_mm"], 0.0, places=9)
        self.assertAlmostEqual(r["improvement_g_mm"], 0.0, places=9)

    def test_single_weight_exact_cancel(self):
        # 单管 50 g 在 0°(U=5000∠0°), 50 g 配重放对径槽 6 后残余为 0
        p = _single_plan()
        r = plan_counterweight_compensation(
            p, [{"mass_g": 50.0, "stock": 1}],
            u1_limit_g_mm=100.0, max_weights=3)
        self.assertTrue(r["feasible"])
        self.assertEqual(r["weights_used"], 1)
        self.assertEqual(len(r["placements"]), 1)
        self.assertEqual(r["placements"][0]["slot"], 6)
        self.assertEqual(r["placements"][0]["mass_g"], 50.0)
        self.assertAlmostEqual(r["final"]["u1_mag_g_mm"], 0.0, places=6)
        self.assertAlmostEqual(r["baseline"]["u1_mag_g_mm"], 5000.0, places=6)
        self.assertAlmostEqual(r["improvement_g_mm"], 5000.0, places=6)

    def test_min_count_two_weights(self):
        # 单管 50 g; 只有 25 g 配重(库存 2): 1 枚最优残余 2500,
        # 2 枚放槽 5/7(150°/210°)残余 5000−2500√3 ≈ 669.87 ≤ 1000
        p = _single_plan()
        r = plan_counterweight_compensation(
            p, [{"mass_g": 25.0, "stock": 2}],
            u1_limit_g_mm=1000.0, max_weights=3)
        self.assertTrue(r["feasible"])
        self.assertEqual(r["weights_used"], 2)
        self.assertEqual([x["slot"] for x in r["placements"]], [5, 7])
        self.assertTrue(all(x["mass_g"] == 25.0 for x in r["placements"]))
        expect = 5000.0 - 2500.0 * math.sqrt(3.0)
        self.assertAlmostEqual(r["final"]["u1_mag_g_mm"], expect, places=6)
        self.assertAlmostEqual(r["margin_g_mm"], 1000.0 - expect, places=6)

    def test_stock_limit_makes_infeasible(self):
        # 同上但 25 g 配重仅 1 件库存: 放 2 枚的组合不存在, 无法达标;
        # 限额内最优为 1 枚放槽 6(残余 2500), 必须给出超限量且不误报达标
        p = _single_plan()
        r = plan_counterweight_compensation(
            p, [{"mass_g": 25.0, "stock": 1}],
            u1_limit_g_mm=1000.0, max_weights=3)
        self.assertFalse(r["feasible"])
        self.assertFalse(r["within_limit"])
        self.assertEqual(r["weights_used"], 1)
        self.assertEqual([x["slot"] for x in r["placements"]], [6])
        self.assertAlmostEqual(r["final"]["u1_mag_g_mm"], 2500.0, places=6)
        self.assertAlmostEqual(r["excess_g_mm"], 1500.0, places=6)
        self.assertIn("不达标", r["reason"])

    def test_same_mass_types_merge_stock(self):
        # 两种 25 g 配重各 1 件 = 25 g 库存 2: 与 test_min_count_two_weights 同解
        p = _single_plan()
        r = plan_counterweight_compensation(
            p, [{"mass_g": 25.0, "stock": 1}, {"mass_g": 25.0, "stock": 1}],
            u1_limit_g_mm=1000.0, max_weights=3)
        self.assertTrue(r["feasible"])
        self.assertEqual(r["weights_used"], 2)
        self.assertEqual([x["slot"] for x in r["placements"]], [5, 7])
        self.assertEqual(r["usable_weight_types"],
                         [{"mass_g": 25.0, "stock": 2}])

    def test_zero_stock_type_unusable(self):
        # 50 g 库存 0(不可用) + 25 g 库存 1: 只能放 1 枚 25 g, 无法达标
        p = _single_plan()
        r = plan_counterweight_compensation(
            p, [{"mass_g": 50.0, "stock": 0}, {"mass_g": 25.0, "stock": 1}],
            u1_limit_g_mm=1000.0, max_weights=3)
        self.assertFalse(r["feasible"])
        self.assertEqual(r["weights_used"], 1)
        self.assertEqual(r["placements"][0]["mass_g"], 25.0)
        self.assertEqual(r["usable_weight_types"],
                         [{"mass_g": 25.0, "stock": 1}])

    def test_tie_break_by_slot_sequence(self):
        # U=5000∠0°, 只有槽 5(150°)与槽 7(210°)空闲: 50 g 配重放任一槽
        # 残余都是 5000·√(2−√3) ≈ 2588, 并列时按槽号升序取槽 5
        p = _single_plan(enabled={0, 5, 7})
        r = plan_counterweight_compensation(
            p, [{"mass_g": 50.0, "stock": 2}],
            u1_limit_g_mm=3000.0, max_weights=1)
        self.assertTrue(r["feasible"])
        self.assertEqual(r["weights_used"], 1)
        self.assertEqual(r["placements"][0]["slot"], 5)
        self.assertEqual(r["optimal_combination_count"], 2)
        self.assertTrue(r["tie_broken"])
        self.assertAlmostEqual(r["final"]["u1_mag_g_mm"],
                               5000.0 * math.sqrt(2.0 - math.sqrt(3.0)),
                               places=6)

    def test_tie_break_by_mass_sequence(self):
        # U=5000∠0°, 仅槽 6(180°)空闲: 30 g 与 70 g 配重残余都是 2000
        # (m₁+m₂ = 2·5000/100), 同槽并列时按质量序列升序取 30 g
        p = _single_plan(enabled={0, 6})
        r = plan_counterweight_compensation(
            p, [{"mass_g": 70.0, "stock": 1}, {"mass_g": 30.0, "stock": 1}],
            u1_limit_g_mm=2500.0, max_weights=1)
        self.assertTrue(r["feasible"])
        self.assertEqual(r["weights_used"], 1)
        self.assertEqual(r["placements"][0]["slot"], 6)
        self.assertEqual(r["placements"][0]["mass_g"], 30.0)
        self.assertEqual(r["optimal_combination_count"], 2)
        self.assertTrue(r["tie_broken"])
        self.assertAlmostEqual(r["final"]["u1_mag_g_mm"], 2000.0, places=6)

    def test_max_weights_zero(self):
        # 放置上限 0 枚且补偿前超限: 只能给出基线对照, 判定不达标
        p = _single_plan()
        r = plan_counterweight_compensation(
            p, [{"mass_g": 50.0, "stock": 3}],
            u1_limit_g_mm=100.0, max_weights=0)
        self.assertFalse(r["feasible"])
        self.assertEqual(r["weights_used"], 0)
        self.assertEqual(r["placements"], [])
        self.assertAlmostEqual(r["final"]["u1_mag_g_mm"], 5000.0, places=6)
        self.assertAlmostEqual(r["excess_g_mm"], 4900.0, places=6)

    def test_no_free_slots(self):
        # 全部启用槽被占满: 配重无处可放, 只能评估 0 枚
        p = _plan(1, {i: (i + 1, 50.0) for i in range(12)})
        r = plan_counterweight_compensation(
            p, [{"mass_g": 5.0, "stock": 3}],
            u1_limit_g_mm=100.0, max_weights=3)
        self.assertEqual(r["free_slot_count"], 0)
        self.assertEqual(r["evaluated_combinations"], 1)
        self.assertEqual(r["weights_used"], 0)
        # 12 支等质量均布: 基线 |U₁|≈0, 已达标
        self.assertTrue(r["feasible"])

    def test_evaluated_combinations_count(self):
        # 单管占槽 0, 空槽 11 个, 配重 2 种(各库存 3), 上限 2 枚:
        # k=0: 1; k=1: 11×2=22; k=2: C(11,2)×2²=55×4=220 → 共 243
        p = _single_plan()
        r = plan_counterweight_compensation(
            p, [{"mass_g": 10.0, "stock": 3}, {"mass_g": 20.0, "stock": 3}],
            u1_limit_g_mm=1.0, max_weights=2)
        self.assertEqual(r["evaluated_combinations"], 1 + 22 + 220)

    def test_too_many_free_slots_rejected(self):
        p = _plan(1, {0: (1, 50.0)}, slots_n=20)  # 19 个空闲启用槽
        with self.assertRaises(ValueError):
            plan_counterweight_compensation(
                p, [{"mass_g": 5.0, "stock": 1}],
                u1_limit_g_mm=100.0, max_weights=3)

    def test_validation_errors(self):
        p = _single_plan()
        good = [{"mass_g": 5.0, "stock": 1}]
        # 限值非法 / 放置上限越界
        for kw in ({"u1_limit_g_mm": -1.0}, {"u1_limit_g_mm": float("nan")},
                   {"u1_limit_g_mm": float("inf")}, {"max_weights": 4},
                   {"max_weights": -1}):
            call = {"weights": good, "u1_limit_g_mm": 100.0, "max_weights": 1}
            call.update(kw)
            with self.assertRaises(ValueError, msg=str(kw)):
                plan_counterweight_compensation(p, **call)
        # 配重种类: 空 / 超过 6 种 / 质量非正 / 质量非有限 / 库存为负 / 库存非整数
        for bad in ([], [{"mass_g": 5.0, "stock": 1}] * 7,
                    [{"mass_g": 0.0, "stock": 1}],
                    [{"mass_g": -2.0, "stock": 1}],
                    [{"mass_g": float("nan"), "stock": 1}],
                    [{"mass_g": float("inf"), "stock": 1}],
                    [{"mass_g": 5.0, "stock": -1}],
                    [{"mass_g": 5.0, "stock": 1.5}],
                    [{"mass_g": 5.0, "stock": True}]):
            with self.assertRaises(ValueError, msg=str(bad)):
                plan_counterweight_compensation(
                    p, bad, u1_limit_g_mm=100.0, max_weights=1)
        # 无已放置试管
        empty = _plan(2, {})
        with self.assertRaises(ValueError):
            plan_counterweight_compensation(
                empty, good, u1_limit_g_mm=100.0, max_weights=1)

    def test_random_against_brute_force(self):
        rng = random.Random(20260925)
        for trial in range(60):
            slots_n = rng.choice([8, 10, 12])
            n_tubes = rng.randint(1, 4)
            occ = rng.sample(range(slots_n), n_tubes)
            placements = {
                s: (i + 1, round(rng.uniform(10.0, 60.0), 2))
                for i, s in enumerate(occ)
            }
            # 随机禁用一部分槽(保证空槽数 ≤ 16)
            enabled = {i for i in range(slots_n) if rng.random() > 0.25}
            enabled.update(occ)
            p = _plan(100 + trial, placements, slots_n=slots_n, enabled=enabled)
            n_types = rng.randint(1, 6)
            weights = [
                {"mass_g": round(rng.uniform(1.0, 60.0), 2),
                 "stock": rng.randint(0, 3)}
                for _ in range(n_types)
            ]
            limit = round(rng.uniform(0.0, 8000.0), 1)
            M = rng.randint(0, 3)
            before = copy.deepcopy(p)
            r = plan_counterweight_compensation(
                p, weights, u1_limit_g_mm=limit, max_weights=M)
            self.assertEqual(p, before)  # 只读: 入参不被修改
            ref = _brute(p, weights, limit, M)
            if ref[0] == "ok":
                _, k, mag, key, count = ref
                self.assertTrue(r["feasible"], (trial, r["reason"]))
                self.assertTrue(r["within_limit"])
                self.assertEqual(r["weights_used"], k, trial)
                self.assertAlmostEqual(r["final"]["u1_mag_g_mm"], mag,
                                       places=6, msg=trial)
                got_key = (tuple(x["slot"] for x in r["placements"]),
                           tuple(x["mass_g"] for x in r["placements"]))
                self.assertEqual(got_key, key, trial)
                self.assertEqual(r["optimal_combination_count"], count, trial)
            else:
                _, (mag, key, k, count) = ref
                self.assertFalse(r["feasible"], (trial, r["reason"]))
                self.assertFalse(r["within_limit"])
                self.assertEqual(r["weights_used"], k, trial)
                self.assertAlmostEqual(r["final"]["u1_mag_g_mm"], mag,
                                       places=6, msg=trial)
                self.assertAlmostEqual(
                    r["excess_g_mm"], max(0.0, mag - limit), places=6)
                got_key = (tuple(x["slot"] for x in r["placements"]),
                           tuple(x["mass_g"] for x in r["placements"]))
                self.assertEqual(got_key, key, trial)
            # 配重只放空槽、单槽至多一枚、每种用量不超库存
            free = {s["slot"] for s in p["result"]["slot_state"]
                    if s["enabled"] and s["tube_id"] is None}
            slots_used = [x["slot"] for x in r["placements"]]
            self.assertEqual(len(slots_used), len(set(slots_used)))
            self.assertTrue(all(s in free for s in slots_used))
            stock = {}
            for w in weights:
                stock[w["mass_g"]] = stock.get(w["mass_g"], 0) + w["stock"]
            for m in {x["mass_g"] for x in r["placements"]}:
                used_m = sum(1 for x in r["placements"] if x["mass_g"] == m)
                self.assertLessEqual(used_m, stock[m])
            # 残余 |U₁| 独立复算
            radius = p["result"]["config"]["radius_mm"]
            ux = uy = 0.0
            for pl in p["result"]["placements"]:
                a = math.radians(pl["angle"])
                ux += pl["mass"] * radius * math.cos(a)
                uy += pl["mass"] * radius * math.sin(a)
            for x in r["placements"]:
                a = math.radians(x["angle_deg"])
                ux += x["mass_g"] * radius * math.cos(a)
                uy += x["mass_g"] * radius * math.sin(a)
            self.assertAlmostEqual(r["final"]["u1_mag_g_mm"],
                                   math.hypot(ux, uy), places=6)


if __name__ == "__main__":
    unittest.main()
