"""备品试管替换规划测试: python -m unittest discover -s tests

核心断言: 备品不重复使用的一对一映射精确枚举(P(s,r))必须与独立的
itertools 全排列参考一致; 达标时先选总质量差最小者, 再按残余 |U₁| 及待换
管号顺序对应的备品编号序列裁决; 无达标映射时返回残余最低的映射与超限量且
不误报达标; 原槽位与未换管保留; 待换管 1~4 支、备品 1~12 支、编号唯一且不
与原方案冲突、质量为正、备品不少于待换管等非法请求必须拒绝; 规划不修改入参
(只读)。
"""
import copy
import itertools
import math
import random
import unittest

from app.spare_replacement import plan_spare_replacement


def _plan(pid, placements, radius=100.0, slots_n=12, enabled=None, name=None,
          rpm=4000.0):
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
        "request": {
            "radius_mm": radius, "rpm": rpm,
            "tubes": [{"id": t} for t, _ in placements.values()],
        },
        "result": {
            "config": {"radius_mm": radius, "rpm": rpm},
            "slot_state": slot_state,
            "placements": pls,
            "metrics": {"u1_mag_g_mm": 0.0},
        },
    }


def _balanced_plan(pid=1, masses=(50.0, 50.0)):
    """对径布管: 管 1..n 放槽 0,6,3,9,...; 等质量时 |U₁|=0。"""
    slots = [0, 6, 3, 9]
    placements = {
        slots[i]: (i + 1, masses[i]) for i in range(len(masses))
    }
    return _plan(pid, placements)


def _brute(plan, replaced, spares, limit):
    """独立全排列参考(选择规则与被测模块一致)。

    返回 ("ok", 总质量差, 残余|U₁|, 备品编号序列, 最优条数)
    或 ("no", 残余|U₁|, 备品编号序列, 并列条数)。
    """
    res = plan["result"]
    radius = res["config"]["radius_mm"]
    placed = {}
    for p in res["placements"]:
        if p["slot"] is None:
            continue
        a = math.radians(p["angle"])
        placed[int(p["tube_id"])] = {
            "m": float(p["mass"]),
            "v": radius * float(p["mass"]) * complex(math.cos(a), math.sin(a)),
        }
    rids = sorted(int(t) for t in replaced)
    sp = sorted(((int(s["tube_id"]), float(s["mass_g"])) for s in spares),
                key=lambda x: x[0])
    base = sum((t["v"] for t in placed.values()), 0j)
    unc = base - sum(placed[t]["v"] for t in rids)
    dirs = []
    for t in rids:
        a = 0.0
        for p in res["placements"]:
            if int(p["tube_id"]) == t:
                a = math.radians(p["angle"])
        dirs.append(radius * complex(math.cos(a), math.sin(a)))
    leaves = []
    for perm in itertools.permutations(range(len(sp)), len(rids)):
        u = unc
        d = 0.0
        for j, ii in enumerate(perm):
            sid, sm = sp[ii]
            u += sm * dirs[j]
            d += abs(sm - placed[rids[j]]["m"])
        key = tuple(sp[ii][0] for ii in perm)
        leaves.append((abs(u), d, key))
    lim_tol = 1e-9 * max(1.0, limit)
    feas = [(m, d, k) for m, d, k in leaves if m <= limit + lim_tol]
    if feas:
        bd = min(d for _, d, _ in feas)
        dtol = 1e-9 * max(1.0, bd)
        band = [(m, d, k) for m, d, k in feas if d <= bd + dtol]
        bm = min(m for m, _, _ in band)
        mtol = 1e-9 * max(1.0, bm)
        opt = [(m, d, k) for m, d, k in band if m <= bm + mtol]
        key = min(k for _, _, k in opt)
        mag, dd = next((m, d) for m, d, k in opt if k == key)
        return ("ok", dd, mag, key, len(opt), len(feas))
    bm = min(m for m, _, _ in leaves)
    mtol = 1e-9 * max(1.0, bm)
    tied = [(m, d, k) for m, d, k in leaves if m <= bm + mtol]
    key = min(k for _, _, k in tied)
    mag, dd = next((m, d) for m, d, k in tied if k == key)
    return ("no", dd, mag, key, len(tied), 0)


class TestSpareReplacement(unittest.TestCase):
    def test_identical_mass_keeps_balance(self):
        # 对径两支 50 g(|U₁|=0), 用同质量备品换管 1: 残余仍为 0, D=0
        p = _balanced_plan()
        r = plan_spare_replacement(
            p, [1], [{"tube_id": 10, "mass_g": 50.0}], u1_limit_g_mm=100.0)
        self.assertTrue(r["feasible"])
        self.assertEqual(r["replace_count"], 1)
        self.assertEqual(r["mappings"][0]["spare_tube_id"], 10)
        self.assertAlmostEqual(r["total_mass_diff_g"], 0.0, places=9)
        self.assertAlmostEqual(r["final"]["u1_mag_g_mm"], 0.0, places=6)
        self.assertEqual(r["evaluated_mappings"], 1)
        self.assertEqual(r["feasible_mapping_count"], 1)

    def test_mass_diff_objective_priority(self):
        # 两支对径 50 g; 备品 49 g(残余 100, D=1)与 50 g(残余 0, D=0):
        # 限值 5000 两者都达标, 必须先取 D 最小的 50 g 备品
        p = _balanced_plan()
        r = plan_spare_replacement(
            p, [1],
            [{"tube_id": 10, "mass_g": 49.0}, {"tube_id": 11, "mass_g": 50.0}],
            u1_limit_g_mm=5000.0)
        self.assertTrue(r["feasible"])
        self.assertEqual(r["mappings"][0]["spare_tube_id"], 11)
        self.assertAlmostEqual(r["total_mass_diff_g"], 0.0, places=9)
        self.assertEqual(r["feasible_mapping_count"], 2)

    def test_residual_tie_break_by_spare_id(self):
        # 两支对径 50 g; 备品 11 与 10 都是 50 g: 两个指派(只换管 1)同 D、同残余,
        # 裁决取备品编号较小者 10
        p = _balanced_plan()
        r = plan_spare_replacement(
            p, [1],
            [{"tube_id": 11, "mass_g": 50.0}, {"tube_id": 10, "mass_g": 50.0}],
            u1_limit_g_mm=100.0)
        self.assertTrue(r["feasible"])
        self.assertEqual(r["mappings"][0]["spare_tube_id"], 10)
        self.assertEqual(r["optimal_mapping_count"], 2)
        self.assertTrue(r["tie_broken"])

    def test_two_tubes_spare_sequence_tie_break(self):
        # 两支待换、三支同质量备品: P(3,2)=6, 两个 50 g 备品的指派并列,
        # 按(管1备品号, 管2备品号)序列取 (10,11)
        p = _balanced_plan(masses=(50.0, 50.0))
        r = plan_spare_replacement(
            p, [2, 1],
            [{"tube_id": 12, "mass_g": 49.0},
             {"tube_id": 11, "mass_g": 50.0},
             {"tube_id": 10, "mass_g": 50.0}],
            u1_limit_g_mm=100.0)
        self.assertTrue(r["feasible"])
        self.assertEqual(r["evaluated_mappings"], 6)
        self.assertEqual(r["replaced_tube_ids"], [1, 2])
        self.assertEqual(
            [m["spare_tube_id"] for m in r["mappings"]], [10, 11])
        self.assertEqual(r["optimal_mapping_count"], 2)
        self.assertTrue(r["tie_broken"])

    def test_no_feasible_returns_lowest_residual(self):
        # 对径 50/50; 备品 48 g(残余 200)与 49 g(残余 100); 限值 50:
        # 无达标映射, 必须返回残余最低的 49 g 备品与超限量, 不误报达标
        p = _balanced_plan()
        r = plan_spare_replacement(
            p, [1],
            [{"tube_id": 11, "mass_g": 48.0}, {"tube_id": 10, "mass_g": 49.0}],
            u1_limit_g_mm=50.0)
        self.assertFalse(r["feasible"])
        self.assertFalse(r["within_limit"])
        self.assertEqual(r["feasible_mapping_count"], 0)
        self.assertEqual(r["mappings"][0]["spare_tube_id"], 10)
        self.assertAlmostEqual(r["final"]["u1_mag_g_mm"], 100.0, places=6)
        self.assertAlmostEqual(r["excess_g_mm"], 50.0, places=6)
        self.assertIn("不达标", r["reason"])

    def test_original_slots_and_unchanged_tubes_kept(self):
        # 换管后备品必须落在原槽、角度不变; 未换管保留编号/质量/槽位
        p = _balanced_plan(masses=(50.0, 51.0))
        r = plan_spare_replacement(
            p, [1], [{"tube_id": 10, "mass_g": 49.5}], u1_limit_g_mm=1e9)
        m = r["mappings"][0]
        self.assertEqual(m["slot"], 0)
        self.assertAlmostEqual(m["angle_deg"], 0.0)
        self.assertEqual(m["replaced_tube_id"], 1)
        self.assertEqual(m["original_mass_g"], 50.0)
        self.assertEqual(m["spare_mass_g"], 49.5)
        self.assertAlmostEqual(m["mass_diff_g"], -0.5, places=9)
        # 未换管: 管 2 仍在槽 6
        self.assertEqual(len(r["unchanged_tubes"]), 1)
        self.assertEqual(r["unchanged_tubes"][0]["tube_id"], 2)
        self.assertEqual(r["unchanged_tubes"][0]["slot"], 6)
        # 全管最终摆放: 槽 0 为备品 10, 槽 6 仍为原管 2
        rows = {row["slot"]: row for row in r["tubes"]}
        self.assertTrue(rows[0]["replaced"])
        self.assertEqual(rows[0]["final_tube_id"], 10)
        self.assertEqual(rows[0]["original_tube_id"], 1)
        self.assertFalse(rows[6]["replaced"])
        self.assertEqual(rows[6]["final_tube_id"], 2)
        # 总质量
        self.assertAlmostEqual(r["baseline"]["total_mass_g"], 101.0, places=6)
        self.assertAlmostEqual(r["final"]["total_mass_g"], 100.5, places=6)
        self.assertAlmostEqual(r["mass_change_g"], -0.5, places=9)

    def test_enumeration_counts(self):
        # 4 支待换、12 支备品: P(12,4)=11880
        slots = [0, 6, 3, 9]
        p = _plan(7, {slots[i]: (i + 1, 50.0) for i in range(4)})
        r = plan_spare_replacement(
            p, [1, 2, 3, 4],
            [{"tube_id": 100 + i, "mass_g": 50.0} for i in range(12)],
            u1_limit_g_mm=1e9)
        self.assertEqual(r["evaluated_mappings"], 12 * 11 * 10 * 9)
        # 全部同质量: 每个排列都达标且并列
        self.assertEqual(r["feasible_mapping_count"], 12 * 11 * 10 * 9)

    def test_spares_not_reused(self):
        # 备品编号在结果映射中不得重复
        p = _balanced_plan(masses=(50.0, 50.0))
        r = plan_spare_replacement(
            p, [1, 2],
            [{"tube_id": 10, "mass_g": 50.0},
             {"tube_id": 11, "mass_g": 49.0},
             {"tube_id": 12, "mass_g": 51.0}],
            u1_limit_g_mm=1e9)
        used = [m["spare_tube_id"] for m in r["mappings"]]
        self.assertEqual(len(used), len(set(used)))

    def test_validation_errors(self):
        p = _balanced_plan()
        good_spares = [{"tube_id": 10, "mass_g": 50.0}]
        # 限值非法
        for bad_limit in (-1.0, float("nan"), float("inf")):
            with self.assertRaises(ValueError, msg=str(bad_limit)):
                plan_spare_replacement(
                    p, [1], good_spares, u1_limit_g_mm=bad_limit)
        # 待换管: 空 / 超过 4 / 重复 / 不在摆放中 / 布尔编号
        for bad_replaced, sp in (
            ([], good_spares),
            ([1, 2, 3, 4, 5], good_spares),
            ([1, 1], good_spares),
            ([9], good_spares),
            ([True], good_spares),
        ):
            with self.assertRaises(ValueError, msg=str(bad_replaced)):
                plan_spare_replacement(
                    p, bad_replaced, sp, u1_limit_g_mm=100.0)
        # 备品: 空 / 13 支 / 编号重复 / 与原方案冲突 / 质量非正 / 非有限 / 布尔编号
        for bad_spares in (
            [],
            [{"tube_id": 100 + i, "mass_g": 50.0} for i in range(13)],
            [{"tube_id": 10, "mass_g": 50.0},
             {"tube_id": 10, "mass_g": 51.0}],
            [{"tube_id": 2, "mass_g": 50.0}],
            [{"tube_id": 10, "mass_g": 0.0}],
            [{"tube_id": 10, "mass_g": -3.0}],
            [{"tube_id": 10, "mass_g": float("nan")}],
            [{"tube_id": 10, "mass_g": float("inf")}],
            [{"tube_id": True, "mass_g": 50.0}],
        ):
            with self.assertRaises(ValueError, msg=str(bad_spares)):
                plan_spare_replacement(
                    p, [1], bad_spares, u1_limit_g_mm=100.0)
        # 备品少于待换管
        with self.assertRaises(ValueError):
            plan_spare_replacement(
                p, [1, 2], [{"tube_id": 10, "mass_g": 50.0}],
                u1_limit_g_mm=100.0)
        # 方案没有已放置试管
        empty = _plan(3, {})
        with self.assertRaises(ValueError):
            plan_spare_replacement(
                empty, [1], good_spares, u1_limit_g_mm=100.0)

    def test_random_against_brute_force(self):
        rng = random.Random(20260925)
        for trial in range(80):
            slots_n = rng.choice([8, 10, 12])
            n_tubes = rng.randint(1, min(4, slots_n))
            occ = rng.sample(range(slots_n), n_tubes)
            placements = {
                s: (i + 1, round(rng.uniform(10.0, 60.0), 2))
                for i, s in enumerate(occ)
            }
            p = _plan(200 + trial, placements, slots_n=slots_n)
            placed_ids = [t for t, _ in placements.values()]
            r = rng.randint(1, min(4, n_tubes))
            replaced = rng.sample(placed_ids, r)
            # 备品编号避开原方案管号
            s_count = rng.randint(r, 12)
            spares = [
                {"tube_id": 1000 + 7 * i + trial,
                 "mass_g": round(rng.uniform(5.0, 70.0), 2)}
                for i in range(s_count)
            ]
            limit = round(rng.uniform(0.0, 9000.0), 1)
            before = copy.deepcopy(p)
            out = plan_spare_replacement(
                p, replaced, spares, u1_limit_g_mm=limit)
            self.assertEqual(p, before)  # 只读: 入参不被修改
            ref = _brute(p, replaced, spares, limit)
            got_key = tuple(m["spare_tube_id"] for m in out["mappings"])
            if ref[0] == "ok":
                _, dd, mag, key, count, feas_count = ref
                self.assertTrue(out["feasible"], (trial, out["reason"]))
                self.assertTrue(out["within_limit"])
                self.assertEqual(got_key, key, trial)
                self.assertAlmostEqual(
                    out["total_mass_diff_g"], dd, places=6, msg=trial)
                self.assertAlmostEqual(
                    out["final"]["u1_mag_g_mm"], mag, places=6, msg=trial)
                self.assertEqual(out["optimal_mapping_count"], count, trial)
                self.assertEqual(out["feasible_mapping_count"], feas_count,
                                 trial)
            else:
                _, dd, mag, key, count, _ = ref
                self.assertFalse(out["feasible"], (trial, out["reason"]))
                self.assertFalse(out["within_limit"])
                self.assertEqual(got_key, key, trial)
                self.assertAlmostEqual(
                    out["total_mass_diff_g"], dd, places=6, msg=trial)
                self.assertAlmostEqual(
                    out["final"]["u1_mag_g_mm"], mag, places=6, msg=trial)
                self.assertAlmostEqual(
                    out["excess_g_mm"], max(0.0, mag - limit), places=6)
                self.assertEqual(out["optimal_mapping_count"], count, trial)
            # 备品不重复使用、全部落在待换管原槽
            self.assertEqual(len(got_key), len(set(got_key)))
            slot_by_tid = {
                int(pl["tube_id"]): int(pl["slot"])
                for pl in p["result"]["placements"] if pl["slot"] is not None
            }
            for m in out["mappings"]:
                self.assertEqual(
                    m["slot"], slot_by_tid[m["replaced_tube_id"]])
            # 残余 |U₁| 独立复算
            radius = p["result"]["config"]["radius_mm"]
            ux = uy = 0.0
            mass_of = {s["tube_id"]: s["mass_g"] for s in spares}
            rep_map = {
                m["replaced_tube_id"]: m["spare_tube_id"]
                for m in out["mappings"]
            }
            for pl in p["result"]["placements"]:
                if pl["slot"] is None:
                    continue
                tid = int(pl["tube_id"])
                mass = mass_of[rep_map[tid]] if tid in rep_map else pl["mass"]
                a = math.radians(pl["angle"])
                ux += mass * radius * math.cos(a)
                uy += mass * radius * math.sin(a)
            self.assertAlmostEqual(
                out["final"]["u1_mag_g_mm"], math.hypot(ux, uy), places=6)


if __name__ == "__main__":
    unittest.main()
