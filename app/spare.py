"""已保存配平方案的备品试管替换规划(只读, 不改变求解与保存格式)。

问题模型
--------
方案已确定: n 支试管各自固定在槽位 θ_k、标称质量 m_k, 转子半径 R, 替换前
一阶不平衡量

    U₁(base) = R · Σ_k m_k · e^{iθ_k}                (g·mm)

现场需要把其中 1~4 支**待换管**从转子上取下, 换成同槽位的**备品试管**:

- 待换管 k 选定后, 其**原槽位与角度保持不变**, 备品直接落在该槽
  (保留原槽位; 未换管的质量与落槽全部不动);
- 录入至多 12 支备品, 每支备品给定唯一编号与正质量, 编号不得与原方案
  任何试管编号冲突;
- 每支备品至多使用一次, 待换管与备品之间做**一对一映射**(双射)。

映射 π(待换管 j → 备品 π(j))确定后, 最终一阶不平衡量

    U₁(π) = U₁(base) − Σ_j R·m_j·e^{iθ_j}
                    + Σ_j R·m_{π(j)}·e^{iθ_j}       (g·mm)

必须不超过给定的一阶不平衡量限值 L(必填, 非负)。

精确搜索(一对一映射的完全枚举)
------------------------------
为控制搜索规模, 待换管至多 4 支、备品至多 12 支(备品数不少于待换管数,
否则不存在完整映射)。设待换管 k 支、备品 s 支, 后端对全部

    P(s, k) = s!/(s−k)!    支    (s ≤ 12, k ≤ 4 时至多 12·11·10·9 = 11880)

个备品不重复使用的有序指派做**精确完全枚举**, 瞬时完成。每个叶子的最终
|U₁| 由原向量减去待换管向量、再加备品向量精确求得。

目标排序(确定性)
----------------
在使最终 |U₁| ≤ L 的**达标映射**中依次按下列字典序择优:

1. **总质量差最小** —— Σ_j |m_{π(j)} − m_j| 最小(备品与原管的绝对质量差
   之和, 1e-9 相对容差内并列);
2. **残余 |U₁| 最低** —— 总质量差并列时取最终 |U₁| 最小者;
3. **按待换管号顺序对应的备品编号裁决** —— 待换管按管号升序排列后,
   比较其对应备品编号序列 (spare_id(管号最小者), spare_id(管号次小者), …)
   字典序最小者。

同时统计达到该最优结果的映射条数。不存在达标映射时, 在**全部映射**中取
残余 |U₁| 最低者(并列时依次按总质量差、备品编号序列裁决), 返回该映射、
超限量与**不达标**结论, 不误报达标。

拒绝条件(ValueError → HTTP 400)
-------------------------------
- 限值不是有限非负数;
- 待换管为空、超过 4 支、编号重复、含不在方案摆放明细中的试管;
- 备品为空或超过 12 支、编号重复或非整数、编号与原方案试管冲突、
  质量非正或非有限;
- 备品数量少于待换管数量(无法构成一对一映射);
- 方案没有已放置试管; 摆放明细中试管编号/槽位重复、质量非正、
  缺少槽位角度; 半径非法。
"""
from __future__ import annotations

import math
from itertools import permutations
from typing import Any, Optional

# 精确枚举规模上限: 待换管 4 支、备品 12 支
# (叶子数 P(12,4) = 11880)
MAX_REPLACE_TUBES = 4
MAX_SPARES = 12
# 与 app/counterweight.py / app/evacuation.py 一致的数值容差
_NUM_TOL = 1e-9


def _vec_block(x: float, y: float) -> dict[str, Any]:
    return {
        "u1_x_g_mm": x,
        "u1_y_g_mm": y,
        "u1_mag_g_mm": math.hypot(x, y),
        "u1_angle_deg": math.degrees(math.atan2(y, x)) % 360.0,
    }


def plan_spare_replacement(plan: dict[str, Any],
                           replace_tube_ids: list[int],
                           spares: list[dict[str, Any]],
                           u1_limit_g_mm: float) -> dict[str, Any]:
    """对已保存方案精确规划备品试管替换(只读)。

    参数
    ----
    plan: GET /api/plan/{id} 返回结构的方案(含 id/name/request/result),
        仅读取, 不修改。
    replace_tube_ids: 待换管编号列表(1~4 支, 不重复, 必须为方案中已放置
        的试管); 规划时按管号升序处理。
    spares: 备品列表(1~12 支), 每项 {"tube_id": 唯一编号(不得与原方案
        管号冲突), "mass_g": 正质量}。
    u1_limit_g_mm: 一阶不平衡量限值(g·mm), 必填的有限非负数。

    返回
    ----
    含逐管替换明细(原管/备品/原槽/前后质径积)、替换前后 |U₁|、达标判定
    (或无达标映射时残余最低的映射与超限量)的字典; ``feasible=False`` 时
    不误报达标。
    """
    plan_id = plan.get("id")

    # ---- 限值 ----
    limit = float(u1_limit_g_mm)
    if not math.isfinite(limit):
        raise ValueError("一阶不平衡量限值必须为有限数值。")
    if limit < 0.0:
        raise ValueError("一阶不平衡量限值不能为负。")

    # ---- 半径 ----
    result = plan.get("result") or {}
    cfg = result.get("config") or {}
    radius = cfg.get("radius_mm")
    if radius is None:
        radius = (plan.get("request") or {}).get("radius_mm", 100.0)
    radius = float(radius)
    if not math.isfinite(radius) or radius <= 0.0:
        raise ValueError("方案中的转子半径必须为正数。")

    # ---- 摆放明细(替换前不平衡量) ----
    used = [p for p in (result.get("placements") or [])
            if p.get("slot") is not None]
    if not used:
        raise ValueError(
            f"方案 {plan_id} 没有已放置的试管, 无法进行备品试管替换。"
        )
    place_by_tube: dict[int, dict[str, Any]] = {}
    seen_slots: set[int] = set()
    base_x = base_y = 0.0
    for p in used:
        tid = int(p["tube_id"])
        if tid in place_by_tube:
            raise ValueError(f"摆放明细中试管 {tid} 重复。")
        slot = int(p["slot"])
        if slot in seen_slots:
            raise ValueError(f"摆放明细中槽位 {slot} 被重复占用。")
        seen_slots.add(slot)
        mass = float(p["mass"])
        if not math.isfinite(mass) or mass <= 0.0:
            raise ValueError(f"试管 {tid} 的标称质量必须为正数。")
        angle = p.get("angle")
        if angle is None or not math.isfinite(float(angle)):
            raise ValueError(f"试管 {tid} 的摆放结果缺少有效槽位角度。")
        a = math.radians(float(angle))
        vx = mass * radius * math.cos(a)
        vy = mass * radius * math.sin(a)
        base_x += vx
        base_y += vy
        place_by_tube[tid] = {
            "tube_id": tid,
            "slot": slot,
            "angle_deg": float(angle),
            "mass_g": mass,
            "locked": bool(p.get("locked", False)),
            "vec": complex(vx, vy),
        }
    tube_count = len(used)
    baseline_block = _vec_block(base_x, base_y)
    base_mag = baseline_block["u1_mag_g_mm"]

    # ---- 原方案全部管号(备品不得与之冲突; 以请求试管清单为准) ----
    original_ids: set[int] = set()
    for t in ((plan.get("request") or {}).get("tubes") or []):
        try:
            original_ids.add(int(t.get("id") if isinstance(t, dict) else t))
        except (TypeError, ValueError):
            continue
    if not original_ids:
        original_ids = set(place_by_tube)

    # ---- 待换管入参 ----
    if not isinstance(replace_tube_ids, (list, tuple)) or not replace_tube_ids:
        raise ValueError("请至少选择 1 支待换管。")
    if len(replace_tube_ids) > MAX_REPLACE_TUBES:
        raise ValueError(
            f"待换管最多 {MAX_REPLACE_TUBES} 支(当前 {len(replace_tube_ids)} 支)。"
        )
    targets: list[int] = []
    for raw in replace_tube_ids:
        if isinstance(raw, bool) or not isinstance(raw, int):
            try:
                tid = int(raw)
            except (TypeError, ValueError) as exc:
                raise ValueError("待换管编号必须为整数。") from exc
            if isinstance(raw, float) and not float(raw).is_integer():
                raise ValueError("待换管编号必须为整数。")
        else:
            tid = raw
        if tid in targets:
            raise ValueError(f"待换管 {tid} 重复选择。")
        if tid not in place_by_tube:
            raise ValueError(
                f"待换管 {tid} 不在方案 {plan_id} 的摆放明细中。"
            )
        targets.append(tid)
    targets.sort()
    k = len(targets)

    # ---- 备品入参 ----
    if not isinstance(spares, (list, tuple)) or not spares:
        raise ValueError("请至少录入 1 支备品试管(编号与质量)。")
    if len(spares) > MAX_SPARES:
        raise ValueError(f"备品试管最多 {MAX_SPARES} 支(当前 {len(spares)} 支)。")
    spare_list: list[dict[str, Any]] = []
    spare_ids: set[int] = set()
    for i, raw in enumerate(spares, start=1):
        if not isinstance(raw, dict):
            raise ValueError(f"第 {i} 支备品的格式不正确。")
        sid_raw = raw.get("tube_id")
        if isinstance(sid_raw, bool) or not isinstance(sid_raw, int):
            try:
                sid = int(sid_raw)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"第 {i} 支备品的编号必须为整数。") from exc
            if isinstance(sid_raw, float) and not float(sid_raw).is_integer():
                raise ValueError(f"第 {i} 支备品的编号必须为整数。")
        else:
            sid = sid_raw
        if sid in spare_ids:
            raise ValueError(f"备品编号 {sid} 重复。")
        if sid in original_ids:
            raise ValueError(
                f"备品编号 {sid} 与原方案试管编号冲突, 请改用其他编号。"
            )
        try:
            mass = float(raw.get("mass_g"))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"备品 {sid} 的质量必须为数值。") from exc
        if not math.isfinite(mass):
            raise ValueError(f"备品 {sid} 的质量必须为有限数值。")
        if mass <= 0.0:
            raise ValueError(f"备品 {sid} 的质量必须大于零(当前 {mass:g} g)。")
        spare_ids.add(sid)
        spare_list.append({"tube_id": sid, "mass_g": mass})
    s = len(spare_list)
    if s < k:
        raise ValueError(
            f"备品数量({s} 支)少于待换管数量({k} 支), "
            "无法构成备品不重复使用的一对一映射。"
        )
    # 按备品编号升序: permutations 的枚举与裁决顺序确定
    spare_list.sort(key=lambda x: x["tube_id"])
    spare_ids_sorted = [x["tube_id"] for x in spare_list]
    spare_mass_by_id = {x["tube_id"]: x["mass_g"] for x in spare_list}

    # ---- 待换管几何(按管号升序) ----
    target_rows = [place_by_tube[t] for t in targets]
    target_set = set(targets)
    # 未换管向量和(任何映射下都不变)
    fixed_vec = sum(
        (place_by_tube[t]["vec"]
         for t in place_by_tube if t not in target_set),
        0j,
    )
    # 各待换管原槽方向的单位质径积向量 R·e^{iθ}
    unit_vecs = [
        complex(
            radius * math.cos(math.radians(row["angle_deg"])),
            radius * math.sin(math.radians(row["angle_deg"])),
        )
        for row in target_rows
    ]
    origin_vecs = [row["vec"] for row in target_rows]
    origin_masses = [row["mass_g"] for row in target_rows]

    # ---- 一对一映射的完全枚举(备品不重复使用) ----
    # 叶子: (最终 |U₁|, 总质量差, (按待换管号顺序对应的备品编号序列))
    leaves: list[tuple[float, float, tuple[int, ...]]] = []
    for perm in permutations(range(s), k):
        ux = fixed_vec.real
        uy = fixed_vec.imag
        diff = 0.0
        key = []
        for j, si in enumerate(perm):
            m_sp = spare_list[si]["mass_g"]
            v = m_sp * unit_vecs[j]
            ux += v.real
            uy += v.imag
            d = m_sp - origin_masses[j]
            diff += -d if d < 0.0 else d
            key.append(spare_ids_sorted[si])
        leaves.append((math.hypot(ux, uy), diff, tuple(key)))
    evaluated = len(leaves)

    def _choose(pool: list[tuple[float, float, tuple[int, ...]]]
                ) -> tuple[float, float, tuple[int, ...], int]:
        """从叶子池中按 (总质量差, 残余 |U₁|, 备品编号序列) 选最优者。

        返回 (残余 |U₁|, 总质量差, 备品编号序列, 容差内达到最优的叶子条数)。
        """
        best_diff = min(d for _, d, _ in pool)
        d_tol = _NUM_TOL * max(1.0, best_diff)
        diff_tied = [leaf for leaf in pool if leaf[1] <= best_diff + d_tol]
        best_mag = min(m for m, _, _ in diff_tied)
        m_tol = _NUM_TOL * max(1.0, best_mag)
        optimal = [
            (m, d, key) for m, d, key in diff_tied
            if m <= best_mag + m_tol
        ]
        opt_key = min(key for _, _, key in optimal)
        opt_mag, opt_diff = next(
            (m, d) for m, d, key in optimal if key == opt_key
        )
        return opt_mag, opt_diff, opt_key, len(optimal)

    def _choose_by_mag(pool: list[tuple[float, float, tuple[int, ...]]]
                       ) -> tuple[float, float, tuple[int, ...], int]:
        """无达标映射时的择优: 先残余 |U₁|, 再总质量差, 再备品编号序列。"""
        best_mag = min(m for m, _, _ in pool)
        m_tol = _NUM_TOL * max(1.0, best_mag)
        mag_tied = [leaf for leaf in pool if leaf[0] <= best_mag + m_tol]
        best_diff = min(d for _, d, _ in mag_tied)
        d_tol = _NUM_TOL * max(1.0, best_diff)
        optimal = [
            (m, d, key) for m, d, key in mag_tied
            if d <= best_diff + d_tol
        ]
        opt_key = min(key for _, _, key in optimal)
        opt_mag, opt_diff = next(
            (m, d) for m, d, key in optimal if key == opt_key
        )
        return opt_mag, opt_diff, opt_key, len(optimal)

    limit_tol = _NUM_TOL * max(1.0, limit)
    feasible_leaves = [leaf for leaf in leaves if leaf[0] <= limit + limit_tol]
    if feasible_leaves:
        final_mag, total_diff, sp_key, opt_count = _choose(feasible_leaves)
        feasible = True
    else:
        final_mag, total_diff, sp_key, opt_count = _choose_by_mag(leaves)
        feasible = False

    # ---- 由裁决序列独立复算最终 U₁, 并组装逐管替换明细 ----
    replacements_out: list[dict[str, Any]] = []
    final_x = fixed_vec.real
    final_y = fixed_vec.imag
    for j, sid in enumerate(sp_key):
        row = target_rows[j]
        m_sp = spare_mass_by_id[sid]
        v_new = m_sp * unit_vecs[j]
        final_x += v_new.real
        final_y += v_new.imag
        v_old = origin_vecs[j]
        signed_delta = m_sp - row["mass_g"]
        replacements_out.append({
            "original_tube_id": row["tube_id"],
            "spare_tube_id": sid,
            "slot": row["slot"],
            "angle_deg": row["angle_deg"],
            "locked_original": row["locked"],
            "original_mass_g": row["mass_g"],
            "spare_mass_g": m_sp,
            "mass_diff_g": abs(signed_delta),
            "mass_change_g": signed_delta,
            "original_u1": _vec_block(v_old.real, v_old.imag),
            "replacement_u1": _vec_block(v_new.real, v_new.imag),
            "u1_delta": _vec_block(
                v_new.real - v_old.real, v_new.imag - v_old.imag
            ),
        })

    final_block = _vec_block(final_x, final_y)
    within = final_block["u1_mag_g_mm"] <= limit + limit_tol
    excess = max(0.0, final_block["u1_mag_g_mm"] - limit)
    margin = max(0.0, limit - final_block["u1_mag_g_mm"])

    # 最终摆放: 逐槽给出最终试管编号与质量(未换管保持原样)
    sp_by_original = {r["original_tube_id"]: r for r in replacements_out}
    final_tubes: list[dict[str, Any]] = []
    for tid, row in place_by_tube.items():
        rep = sp_by_original.get(tid)
        final_tubes.append({
            "slot": row["slot"],
            "angle_deg": row["angle_deg"],
            "original_tube_id": tid,
            "final_tube_id": rep["spare_tube_id"] if rep else tid,
            "original_mass_g": row["mass_g"],
            "final_mass_g": rep["spare_mass_g"] if rep else row["mass_g"],
            "replaced": rep is not None,
            "locked": row["locked"],
        })
    final_tubes.sort(key=lambda r: (r["slot"], r["original_tube_id"]))

    used_spare_ids = set(sp_key)
    unused_spares = [
        {"tube_id": x["tube_id"], "mass_g": x["mass_g"]}
        for x in spare_list if x["tube_id"] not in used_spare_ids
    ]

    replace_txt = "、".join(
        f"管 {r['original_tube_id']}→备品 {r['spare_tube_id']}"
        f"({r['original_mass_g']:g}g→{r['spare_mass_g']:g}g, 槽 {r['slot']})"
        for r in replacements_out
    )
    if feasible:
        reason = (
            f"在 {evaluated} 种备品不重复使用的一对一映射中, 映射({replace_txt})"
            f"达标: 最终 |U₁|={final_block['u1_mag_g_mm']:.6g} g·mm ≤ 限值 "
            f"{limit:g} g·mm(替换前 {base_mag:.6g} g·mm)。它先取得最小总质量差 "
            f"{total_diff:.6g} g"
            + (
                f"; 达到该最优结果的映射共 {opt_count} 种, 已按残余 |U₁| 及"
                "待换管号顺序对应的备品编号序列裁决。"
                if opt_count > 1
                else ", 且该总质量差下最优映射唯一。"
            )
        )
    else:
        reason = (
            f"共精确枚举 {evaluated} 种备品不重复使用的一对一映射, "
            f"没有使最终 |U₁| ≤ 限值 {limit:g} g·mm 的达标映射; "
            f"残余最低的映射为({replace_txt}), 总质量差 {total_diff:.6g} g, "
            f"最终 |U₁|={final_block['u1_mag_g_mm']:.6g} g·mm"
            f"(替换前 {base_mag:.6g} g·mm), 仍超出 {excess:.6g} g·mm, "
            "判定不达标。"
        )

    return {
        "method": (
            "备品试管一对一映射的完全枚举(待换管 "
            f"1~{MAX_REPLACE_TUBES} 支、备品 ≤{MAX_SPARES} 支, "
            f"叶子数 P(s,k) ≤ {MAX_SPARES}·{MAX_SPARES - 1}·"
            f"{MAX_SPARES - 2}·{MAX_SPARES - 3}={11880}): "
            "备品编号唯一且不与原方案管号冲突、每支备品至多使用一次、"
            "备品落在待换管原槽(原角度与半径不变, 未换管不动); "
            "在达标映射中先取总质量差(绝对质量差之和)最小者, "
            "再按残余 |U₁| 及待换管号顺序对应的备品编号序列裁决; "
            "无达标映射时返回残余最低的映射、超限量与不达标结论, 不误报达标"
        ),
        "feasible": feasible,
        "plan": {
            "id": plan.get("id"),
            "name": plan.get("name"),
            "created_at": plan.get("created_at"),
        },
        "radius_mm": radius,
        "tube_count": tube_count,
        "replace_tube_ids": targets,
        "replaced_count": k,
        "spares": spare_list,
        "spare_count": s,
        "unused_spares": unused_spares,
        "u1_limit_g_mm": limit,
        "evaluated_mappings": evaluated,
        "baseline": {
            **baseline_block,
            "within_limit": base_mag <= limit + limit_tol,
            "excess_g_mm": max(0.0, base_mag - limit),
            "margin_g_mm": max(0.0, limit - base_mag),
        },
        "final": final_block,
        "within_limit": within,
        "excess_g_mm": excess,
        "margin_g_mm": margin,
        "total_mass_diff_g": total_diff,
        "improvement_g_mm": base_mag - final_block["u1_mag_g_mm"],
        "optimal_mapping_count": opt_count,
        "tie_broken": opt_count > 1,
        "replacements": replacements_out,
        "tubes": final_tubes,
        "reason": reason,
        "note": (
            "备品试管替换为只读规划结果, 未修改任何已保存方案; "
            "备品落在待换管原槽位(角度/半径不变), 未换管的质量与落槽保持原样。"
        ),
    }
