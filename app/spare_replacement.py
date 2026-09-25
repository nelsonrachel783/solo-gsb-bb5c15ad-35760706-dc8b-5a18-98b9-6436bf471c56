"""已保存配平方案的备品试管替换规划(只读, 不改变求解与保存格式)。

问题模型
--------
方案已确定: n 支试管各自固定在槽位 θ_k、标称质量 m_k, 转子半径 R, 替换前
一阶不平衡量

    U₁(base) = R · Σ_k m_k · e^{iθ_k}                (g·mm)

现场选定 r 支**待换管**(1 ≤ r ≤ 4, 均为方案中已放置的试管, 不重复), 并录入
s 支**备品试管**(1 ≤ s ≤ 12, 编号互不相同, 质量为正, 且编号不与原方案的任何
试管编号冲突)。备品替换时**保留原槽位**: 每支待换管从其原槽取下, 一支备品装入
同一槽位; **备品不得重复使用**, 即待换管与备品之间构成一对一映射(单射), 其余
未换管及其槽位完全不动。一个映射确定后, 最终一阶不平衡量

    U₁ = U₁(base)
         + R · Σ_j (m'_j − m_j) · e^{iθ_j}            (g·mm)

其中 j 遍历待换管(θ_j 为该管原槽角度, m'_j 为换上的备品质量), 必须不超过给定
的一阶不平衡量限值 L(必填, 非负)。

精确搜索(备品不重复使用的一对一完全枚举)
----------------------------------------
待换管按管号升序固定为位置 1..r, 从 s 支备品中选 r 支并逐一指派(排列):

    映射数 = P(s, r) = s!/(s−r)!,  r ≤ 4、s ≤ 12 时至多 12·11·10·9 = 11880 个,
    瞬时完成。

回溯枚举(维护备品占用标记), 每个叶子由增量向量和精确求得最终 |U₁|, 并累计
总质量差 D = Σ_j |m'_j − m_j|(g)。

目标排序(字典序, 确定性)
------------------------
1. **先选总质量差 D 最小的达标映射**(最终 |U₁| ≤ L);
2. D 在数值容差(1e-9 相对容差, 与应急迁移/配重补偿等精确模块一致)内并列时,
   取最终残余 **|U₁| 最小**者;
3. 仍并列时, 取**待换管号(升序)顺序对应的备品编号序列**字典序最小者,
   并统计达到该最优结果的映射条数。

若没有任何达标映射, 返回**残余 |U₁| 最低**的映射(容差内并列时按上述备品编号
序列裁决)、超限量与**不达标**结论, 不误报达标。

拒绝条件(ValueError → HTTP 400)
-------------------------------
- 限值不是有限非负数;
- 待换管数量不在 1~4、编号重复、不是整数, 或有编号不在方案的已放置试管中;
- 备品数量为 0 或超过 12、备品编号重复/非整数、质量非正或非有限;
- 备品编号与原方案试管编号(含未放置试管)冲突;
- 备品数量少于待换管数(无法构成一对一不重复映射);
- 方案没有已放置试管; 摆放明细中试管编号/槽位重复、质量非正、缺少槽位角度;
- 半径非法。
"""
from __future__ import annotations

import math
from typing import Any, Optional

# 精确枚举规模上限: 待换管 4 支、备品 12 支(排列数 P(12,4)=11880)
MAX_REPLACE = 4
MAX_SPARES = 12
# 与 app/evacuation.py / app/counterweight.py 一致的数值容差
_NUM_TOL = 1e-9


def _vec_block(x: float, y: float) -> dict[str, Any]:
    return {
        "u1_x_g_mm": x,
        "u1_y_g_mm": y,
        "u1_mag_g_mm": math.hypot(x, y),
        "u1_angle_deg": math.degrees(math.atan2(y, x)) % 360.0,
    }


def _as_int(raw: Any, what: str) -> int:
    if isinstance(raw, bool):
        raise ValueError(f"{what}必须为整数编号(当前 {raw})。")
    try:
        return int(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{what}必须为整数编号(当前 {raw})。") from exc


def plan_spare_replacement(plan: dict[str, Any],
                           replaced_tube_ids: list[int],
                           spares: list[dict[str, Any]],
                           u1_limit_g_mm: float) -> dict[str, Any]:
    """对已保存方案精确规划备品试管一对一替换(只读)。

    参数
    ----
    plan: GET /api/plan/{id} 返回结构的方案(含 id/name/request/result),
        仅读取, 不修改。
    replaced_tube_ids: 待换管编号列表, 1~4 支、不重复, 均须为方案中已放置试管。
    spares: 备品列表(1~12 支), 每项 {"tube_id": 编号(int), "mass_g": 质量(g)};
        编号互不相同且不与原方案管号冲突, 质量为正的有限数。
    u1_limit_g_mm: 一阶不平衡量限值(g·mm), 必填的有限非负数。

    返回
    ----
    含逐管替换映射(原管/备品、原槽角度、质量差与各自质径积贡献)、保留不动的
    未换管、替换前后 |U₁|/总质量、总质量差、达标判定(或残余最低映射与超限量)
    的字典; ``feasible=False`` 时不误报达标。
    """
    plan_id = plan.get("id")

    # ---- 限值 ----
    limit = float(u1_limit_g_mm)
    if not math.isfinite(limit):
        raise ValueError("一阶不平衡量限值必须为有限数值。")
    if limit < 0.0:
        raise ValueError("一阶不平衡量限值不能为负。")

    # ---- 半径 / 转速 ----
    result = plan.get("result") or {}
    cfg = result.get("config") or {}
    radius = cfg.get("radius_mm")
    if radius is None:
        radius = (plan.get("request") or {}).get("radius_mm", 100.0)
    radius = float(radius)
    if not math.isfinite(radius) or radius <= 0.0:
        raise ValueError("方案中的转子半径必须为正数。")
    rpm = float(cfg.get("rpm", (plan.get("request") or {}).get("rpm", 0.0)))
    omega = 2.0 * math.pi * rpm / 60.0

    # ---- 待换管入参 ----
    if not isinstance(replaced_tube_ids, (list, tuple)) or not replaced_tube_ids:
        raise ValueError("请至少选择一支待换管(1~4 支)。")
    if len(replaced_tube_ids) > MAX_REPLACE:
        raise ValueError(
            f"待换管最多选择 {MAX_REPLACE} 支(当前 {len(replaced_tube_ids)} 支)。"
        )
    replaced_ids: list[int] = []
    for raw in replaced_tube_ids:
        tid = _as_int(raw, "待换管编号")
        if tid in replaced_ids:
            raise ValueError(f"待换管 {tid} 重复选择。")
        replaced_ids.append(tid)

    # ---- 摆放明细 ----
    used = [p for p in (result.get("placements") or [])
            if p.get("slot") is not None]
    if not used:
        raise ValueError(
            f"方案 {plan_id} 没有已放置的试管, 无法进行备品试管替换。"
        )
    placed: dict[int, dict[str, Any]] = {}
    seen_slots: set[int] = set()
    base_x = base_y = 0.0
    base_total_mass = 0.0
    for p in used:
        tid = _as_int(p.get("tube_id"), "摆放明细中的试管编号")
        if tid in placed:
            raise ValueError(f"摆放明细中试管 {tid} 重复。")
        slot = _as_int(p.get("slot"), f"试管 {tid} 的槽位")
        if slot in seen_slots:
            raise ValueError(f"摆放明细中槽位 {slot} 被重复占用。")
        seen_slots.add(slot)
        mass = float(p.get("mass"))
        if not math.isfinite(mass) or mass <= 0.0:
            raise ValueError(f"试管 {tid} 的标称质量必须为正数。")
        angle = p.get("angle")
        if angle is None or not math.isfinite(float(angle)):
            raise ValueError(f"试管 {tid} 的摆放结果缺少有效槽位角度。")
        angle_deg = float(angle)
        a = math.radians(angle_deg)
        vx = mass * radius * math.cos(a)
        vy = mass * radius * math.sin(a)
        base_x += vx
        base_y += vy
        base_total_mass += mass
        placed[tid] = {
            "tube_id": tid,
            "mass": mass,
            "slot": slot,
            "angle_deg": angle_deg,
            "locked": bool(p.get("locked", False)),
            "vec": (vx, vy),
        }
    n = len(placed)

    # 原方案全部试管编号(含未放置的), 备品编号不得与之冲突
    original_ids = set(placed)
    for p in (result.get("placements") or []):
        if p.get("tube_id") is not None:
            original_ids.add(_as_int(p.get("tube_id"), "摆放明细中的试管编号"))
    for t in ((plan.get("request") or {}).get("tubes") or []):
        if isinstance(t, dict) and t.get("id") is not None:
            original_ids.add(_as_int(t.get("id"), "方案试管编号"))

    for tid in replaced_ids:
        if tid not in placed:
            raise ValueError(f"待换管 {tid} 不在方案 {plan_id} 的已放置试管中。")
    # 待换管按管号升序: 裁决序列中的备品编号即按此位置顺序
    replaced_ids.sort()

    # ---- 备品入参 ----
    if not isinstance(spares, (list, tuple)) or not spares:
        raise ValueError("请至少录入一支备品试管(编号与质量)。")
    if len(spares) > MAX_SPARES:
        raise ValueError(f"备品试管最多 {MAX_SPARES} 支(当前 {len(spares)} 支)。")
    spare_list: list[dict[str, Any]] = []
    seen_spare_ids: set[int] = set()
    for i, raw in enumerate(spares, start=1):
        if not isinstance(raw, dict):
            raise ValueError(f"第 {i} 支备品的格式必须为编号与质量的对象。")
        sid = _as_int(raw.get("tube_id"), f"第 {i} 支备品的编号")
        if sid in seen_spare_ids:
            raise ValueError(f"备品编号 {sid} 重复填写。")
        if sid in original_ids:
            raise ValueError(
                f"备品编号 {sid} 与原方案试管编号冲突, 请使用不与原方案管号"
                "重复的备品编号。"
            )
        try:
            smass = float(raw.get("mass_g"))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"备品 {sid} 的质量必须为数值。") from exc
        if not math.isfinite(smass):
            raise ValueError(f"备品 {sid} 的质量必须为有限数值。")
        if smass <= 0.0:
            raise ValueError(f"备品 {sid} 的质量必须大于零(当前 {smass:g} g)。")
        seen_spare_ids.add(sid)
        spare_list.append({"tube_id": sid, "mass_g": smass})
    spare_list.sort(key=lambda x: x["tube_id"])
    s_count = len(spare_list)
    k = len(replaced_ids)
    if s_count < k:
        raise ValueError(
            f"备品仅 {s_count} 支, 不足以一对一替换 {k} 支待换管"
            "(备品不得重复使用), 请补足备品或减少待换管。"
        )

    # ---- 未换管(保留不动)向量与逐行信息 ----
    replaced_set = set(replaced_ids)
    unc_x = base_x
    unc_y = base_y
    unchanged_rows: list[dict[str, Any]] = []
    for tid in replaced_ids:
        vx, vy = placed[tid]["vec"]
        unc_x -= vx
        unc_y -= vy
    for tid, t in placed.items():
        if tid in replaced_set:
            continue
        vx, vy = t["vec"]
        unchanged_rows.append({
            "tube_id": tid,
            "mass_g": t["mass"],
            "locked": t["locked"],
            "slot": t["slot"],
            "angle_deg": t["angle_deg"],
            "u1": _vec_block(vx, vy),
        })
    unchanged_rows.sort(key=lambda r: r["slot"])

    # 待换管原槽方向的单位质径积向量 R·e^{iθ}
    rep_dirs = []
    for tid in replaced_ids:
        a = math.radians(placed[tid]["angle_deg"])
        rep_dirs.append((radius * math.cos(a), radius * math.sin(a)))

    # ---- 备品不重复使用的一对一完全枚举(排列) ----
    # 叶子: (|U₁|, 总质量差 D, 备品编号序列(按待换管号升序位置))
    leaves: list[tuple[float, float, tuple[int, ...]]] = []
    used_flag = [False] * s_count
    key_buf: list[int] = []

    def rec(pos: int, ux: float, uy: float, mass_diff: float) -> None:
        if pos == k:
            leaves.append((math.sqrt(ux * ux + uy * uy),
                           mass_diff, tuple(key_buf)))
            return
        dx, dy = rep_dirs[pos]
        orig_mass = placed[replaced_ids[pos]]["mass"]
        for i in range(s_count):
            if used_flag[i]:
                continue
            sm = spare_list[i]["mass_g"]
            used_flag[i] = True
            key_buf.append(spare_list[i]["tube_id"])
            rec(pos + 1,
                ux + sm * dx, uy + sm * dy,
                mass_diff + abs(sm - orig_mass))
            key_buf.pop()
            used_flag[i] = False

    rec(0, unc_x, unc_y, 0.0)
    evaluated = len(leaves)

    limit_tol = _NUM_TOL * max(1.0, limit)
    feasible_leaves = [(m, d, key) for m, d, key in leaves
                       if m <= limit + limit_tol]
    if feasible_leaves:
        # ① 总质量差最小
        best_d = min(d for _, d, _ in feasible_leaves)
        d_tol = _NUM_TOL * max(1.0, best_d)
        band = [(m, d, key) for m, d, key in feasible_leaves
                if d <= best_d + d_tol]
        # ② 残余 |U₁| 最小
        best_mag = min(m for m, _, _ in band)
        mag_tol = _NUM_TOL * max(1.0, best_mag)
        optimal = [(m, d, key) for m, d, key in band
                   if m <= best_mag + mag_tol]
        # ③ 待换管号顺序对应的备品编号序列字典序
        opt_key = min(key for _, _, key in optimal)
        opt_leaf = next(leaf for leaf in optimal if leaf[2] == opt_key)
        winner = True
        chosen_mag, chosen_d, chosen_key = opt_leaf
        optimal_count = len(optimal)
    else:
        # 无达标映射: 残余 |U₁| 最低; 容差内并列按备品编号序列裁决
        best_mag = min(m for m, _, _ in leaves)
        mag_tol = _NUM_TOL * max(1.0, best_mag)
        tied = [(m, d, key) for m, d, key in leaves
                if m <= best_mag + mag_tol]
        best_key = min(key for _, _, key in tied)
        chosen_leaf = next(leaf for leaf in tied if leaf[2] == best_key)
        winner = False
        chosen_mag, chosen_d, chosen_key = chosen_leaf
        optimal_count = len(tied)

    # ---- 由裁决序列独立复算最终 U₁ 与逐管替换映射 ----
    spare_by_id = {sp["tube_id"]: sp["mass_g"] for sp in spare_list}
    mapping_rows: list[dict[str, Any]] = []
    final_x, final_y = unc_x, unc_y
    replaced_mass_before = replaced_mass_after = 0.0
    total_diff = 0.0
    for pos, sid in enumerate(chosen_key):
        tid = replaced_ids[pos]
        t = placed[tid]
        smass = spare_by_id[sid]
        a = math.radians(t["angle_deg"])
        ovx, ovy = t["vec"]
        fvx = smass * radius * math.cos(a)
        fvy = smass * radius * math.sin(a)
        final_x += fvx
        final_y += fvy
        diff = smass - t["mass"]
        total_diff += abs(diff)
        replaced_mass_before += t["mass"]
        replaced_mass_after += smass
        mapping_rows.append({
            "replaced_tube_id": tid,
            "original_mass_g": t["mass"],
            "slot": t["slot"],
            "angle_deg": t["angle_deg"],
            "locked": t["locked"],
            "spare_tube_id": sid,
            "spare_mass_g": smass,
            "mass_diff_g": diff,
            "abs_mass_diff_g": abs(diff),
            "original_u1": _vec_block(ovx, ovy),
            "final_u1": _vec_block(fvx, fvy),
            "u1_change": _vec_block(fvx - ovx, fvy - ovy),
        })

    final_total_mass = base_total_mass - replaced_mass_before + replaced_mass_after
    baseline_block = _vec_block(base_x, base_y)
    final_block = _vec_block(final_x, final_y)
    base_mag = baseline_block["u1_mag_g_mm"]
    final_mag = final_block["u1_mag_g_mm"]

    # 全部试管的最终摆放逐行(按槽位升序): 未换管保留原编号/质量, 待换槽显示备品
    final_slot_of: dict[int, dict[str, Any]] = {}
    for tid, t in placed.items():
        final_slot_of[t["slot"]] = {
            "slot": t["slot"],
            "angle_deg": t["angle_deg"],
            "replaced": tid in replaced_set,
            "original_tube_id": tid,
            "original_mass_g": t["mass"],
            "locked": t["locked"],
        }
    for row in mapping_rows:
        cell = final_slot_of[row["slot"]]
        cell["final_tube_id"] = row["spare_tube_id"]
        cell["final_mass_g"] = row["spare_mass_g"]
    for cell in final_slot_of.values():
        cell.setdefault("final_tube_id", cell["original_tube_id"])
        cell.setdefault("final_mass_g", cell["original_mass_g"])
    tube_rows = [final_slot_of[s] for s in sorted(final_slot_of)]

    feasible = winner
    within = final_mag <= limit + limit_tol
    excess = max(0.0, final_mag - limit)
    margin = max(0.0, limit - final_mag)
    map_txt = ", ".join(
        f"管 {r['replaced_tube_id']}({r['original_mass_g']:g} g, 槽 {r['slot']})"
        f"→备品 {r['spare_tube_id']}({r['spare_mass_g']:g} g)"
        for r in mapping_rows
    )

    if feasible:
        reason = (
            f"在精确枚举的 {evaluated} 种备品一对一(不重复)映射中, "
            f"{len(feasible_leaves)} 种满足残余 |U₁| ≤ 限值 {limit:g} g·mm; "
            f"取总质量差最小者: 总质量差 {total_diff:.6g} g({map_txt}), "
            f"替换后残余 |U₁|={final_mag:.6g} g·mm ≤ 限值(替换前 "
            f"{base_mag:.6g} g·mm), 判定达标; 总质量差并列时已按残余 |U₁| "
            "及待换管号顺序对应的备品编号裁决。"
        )
    else:
        reason = (
            f"共精确枚举 {evaluated} 种备品一对一(不重复)映射, "
            f"没有使残余 |U₁| ≤ 限值 {limit:g} g·mm 的映射; "
            f"残余最低的映射总质量差 {total_diff:.6g} g({map_txt}), "
            f"残余 |U₁|={final_mag:.6g} g·mm(替换前 {base_mag:.6g} g·mm), "
            f"仍超出 {excess:.6g} g·mm, 判定不达标。"
        )

    return {
        "method": (
            "备品不重复使用的一对一完全枚举(待换管 1~4 支、备品 1~12 支, "
            f"排列数 P(s,r) ≤ {MAX_SPARES}·{MAX_SPARES - 1}·{MAX_SPARES - 2}·"
            f"{MAX_SPARES - 3}): 备品装入待换管原槽、原槽位与未换管保留不动; "
            "在达标映射中先取总质量差最小者, 再按残余 |U₁| 及待换管号(升序)"
            "顺序对应的备品编号序列裁决; 无达标映射时返回残余最低的映射、"
            "超限量与不达标结论, 不误报达标"
        ),
        "feasible": feasible,
        "plan": {
            "id": plan.get("id"),
            "name": plan.get("name"),
            "created_at": plan.get("created_at"),
        },
        "radius_mm": radius,
        "rpm": rpm,
        "tube_count": n,
        "replaced_tube_ids": replaced_ids,
        "replace_count": k,
        "spares": spare_list,
        "spare_count": s_count,
        "u1_limit_g_mm": limit,
        "evaluated_mappings": evaluated,
        "feasible_mapping_count": len(feasible_leaves),
        "baseline": {
            **baseline_block,
            "total_mass_g": base_total_mass,
            "centrifugal_force_n": omega ** 2 * base_mag * 1e-6,
            "within_limit": base_mag <= limit + limit_tol,
            "excess_g_mm": max(0.0, base_mag - limit),
            "margin_g_mm": max(0.0, limit - base_mag),
        },
        "unchanged_tubes": unchanged_rows,
        "mappings": mapping_rows,
        "tubes": tube_rows,
        "total_mass_diff_g": total_diff,
        "optimal_mapping_count": optimal_count,
        "tie_broken": optimal_count > 1,
        "final": {
            **final_block,
            "total_mass_g": final_total_mass,
            "centrifugal_force_n": omega ** 2 * final_mag * 1e-6,
        },
        "within_limit": within,
        "excess_g_mm": excess,
        "margin_g_mm": margin,
        "improvement_g_mm": base_mag - final_mag,
        "mass_change_g": final_total_mass - base_total_mass,
        "reason": reason,
        "note": (
            "备品试管替换为只读规划结果, 未修改任何已保存方案; "
            "备品装入待换管原槽(保留原槽位), 备品不重复使用, 未换管保持不动。"
        ),
    }
