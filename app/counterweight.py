"""已保存配平方案的库存配重补偿(只读, 不改变求解与保存格式)。

问题模型
--------
方案已确定: n 支试管各自固定在槽位 θ_k、标称质量 m_k, 转子半径 R, 补偿前
一阶不平衡量

    U₁(base) = R · Σ_k m_k · e^{iθ_k}                (g·mm)

现场备有若干种**标准配重**(每种给定质量与库存件数, 至多 6 种), 允许在转子的
**空闲启用槽位**上放置配重(配重只放空槽, 单槽至多一枚, 每种配重用量不超过
其库存件数), 总放置枚数不超过 max_weights(0~3)。放置槽位集合 S 与对应
配重质量 {m(s)} 后, 残余一阶不平衡量

    U₁ = U₁(base) + R · Σ_{s∈S} m(s) · e^{iθ_s}      (g·mm)

必须不超过给定的一阶不平衡量限值 L(必填, 非负)。

精确搜索(按枚数分层的完全枚举)
------------------------------
为控制搜索规模, 仅处理至多 16 个空闲启用槽位的方案。设空闲启用槽有 q 个
(q ≤ 16)、可用配重 t 种(同质量合并库存, 至多 6 种)、放置上限 M ≤ 3,
按放置枚数 k = 0..M 逐层枚举槽位与质量的**联合组合**:

    槽位选择  C(q, k)   种(槽号升序组合);
    质量指派  ≤ t^k     种(逐槽指派一种配重, 回溯中按库存余量剪枝);
    叶子总数  ≤ Σ_k C(16,k)·6^k ≈ 12.6 万(k=0..3), 瞬时完成。

每个叶子的残余 |U₁| 由回溯中的增量向量和精确求得。

目标排序(字典序, 确定性)
------------------------
1. 能达标(残余 |U₁| ≤ L)的**最少放置枚数** k* —— 从 k=0 起逐层枚举,
   首个存在达标组合的 k 即 k*;
2. 在放置 k* 枚的达标组合中取残余 |U₁| **最低**者;
3. 数值容差(1e-9 相对容差, 与应急迁移/复称规划等精确模块一致)内并列时,
   按**槽号及对应质量序列**裁决: 先比较槽号升序的槽位序列, 再比较这些槽位
   上对应的配重质量序列(字典序最小者胜出), 并统计达到该最优结果的组合数。

若 0..M 枚内均无法达标, 给出限额内残余 |U₁| 最低的组合(并列时先取放置
枚数少者, 再按槽号及质量序列升序)及超限量, 明确判定**不达标**, 不误报
达标。

拒绝条件(ValueError → HTTP 400)
-------------------------------
- 限值不是有限非负数, 或最多放置枚数不在 0~3;
- 配重种类为空或超过 6 种; 配重质量非正/非有限; 库存件数非整数或为负;
- 方案没有已放置试管; 空闲启用槽位超过 16 个(精确枚举规模上限);
- 摆放明细中试管编号/槽位重复、质量非正、缺少槽位角度; 半径非法。
"""
from __future__ import annotations

import math
from typing import Any, Optional

# 精确枚举规模上限: 空闲启用槽 16 个、配重 6 种、放置上限 3 枚
# (叶子数 Σ_{k=0..3} C(16,k)·6^k ≈ 12.6 万)
MAX_FREE_SLOTS = 16
MAX_WEIGHT_TYPES = 6
MAX_WEIGHTS = 3
# 与 app/evacuation.py / app/reweigh.py 一致的数值容差
_NUM_TOL = 1e-9


def _vec_block(x: float, y: float) -> dict[str, Any]:
    return {
        "u1_x_g_mm": x,
        "u1_y_g_mm": y,
        "u1_mag_g_mm": math.hypot(x, y),
        "u1_angle_deg": math.degrees(math.atan2(y, x)) % 360.0,
    }


def plan_counterweight_compensation(
        plan: dict[str, Any],
        weights: list[dict[str, Any]],
        u1_limit_g_mm: float,
        max_weights: int) -> dict[str, Any]:
    """对已保存方案精确规划库存配重补偿(只读)。

    参数
    ----
    plan: GET /api/plan/{id} 返回结构的方案(含 id/name/request/result),
        仅读取, 不修改。
    weights: 标准配重种类列表(1~6 种), 每项 {"mass_g": 质量(g, >0),
        "stock": 库存件数(非负整数)}; 同质量的种类合并库存。
    u1_limit_g_mm: 一阶不平衡量限值(g·mm), 必填的有限非负数。
    max_weights: 最多放置枚数, 0~3。

    返回
    ----
    含放置清单(槽位/角度/配重质量)、补偿前后 |U₁|、达标判定(或限额内
    残余最低的组合与超限量)的字典; ``feasible=False`` 时不误报达标。
    """
    plan_id = plan.get("id")

    # ---- 限值 / 放置上限 ----
    limit = float(u1_limit_g_mm)
    if not math.isfinite(limit):
        raise ValueError("一阶不平衡量限值必须为有限数值。")
    if limit < 0.0:
        raise ValueError("一阶不平衡量限值不能为负。")
    M = int(max_weights)
    if not (0 <= M <= MAX_WEIGHTS):
        raise ValueError(f"最多放置枚数必须在 0~{MAX_WEIGHTS} 之间(当前 {M})。")

    # ---- 配重种类入参 ----
    if not isinstance(weights, (list, tuple)) or not weights:
        raise ValueError("请至少录入一种标准配重(质量与库存件数)。")
    if len(weights) > MAX_WEIGHT_TYPES:
        raise ValueError(
            f"标准配重最多 {MAX_WEIGHT_TYPES} 种(当前 {len(weights)} 种)。"
        )
    weight_types: list[dict[str, Any]] = []
    stock_by_mass: dict[float, int] = {}
    for i, raw in enumerate(weights, start=1):
        try:
            mass = float(raw.get("mass_g"))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"第 {i} 种配重的质量必须为数值。") from exc
        if not math.isfinite(mass):
            raise ValueError(f"第 {i} 种配重的质量必须为有限数值。")
        if mass <= 0.0:
            raise ValueError(f"第 {i} 种配重的质量必须大于零(当前 {mass:g} g)。")
        stock = raw.get("stock")
        try:
            stock_int = int(stock)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"第 {i} 种配重的库存件数必须为整数(当前 {stock})。"
            ) from exc
        if isinstance(stock, bool) or stock_int != stock:
            raise ValueError(
                f"第 {i} 种配重的库存件数必须为整数(当前 {stock})。"
            )
        stock = stock_int
        if stock < 0:
            raise ValueError(f"第 {i} 种配重的库存件数不能为负(当前 {stock})。")
        weight_types.append({"mass_g": mass, "stock": stock})
        # 同质量的种类对平衡效果完全等价, 合并库存(放置上限 M 枚,
        # 单种用量不可能超过 M, 库存截断到 M 不影响枚举结果)
        stock_by_mass[mass] = min(
            M, stock_by_mass.get(mass, 0) + min(stock, M)
        )
    # 可用配重: 库存 > 0 的种类, 按质量升序(枚举与裁决的确定顺序)
    usable_masses = sorted(m for m, s in stock_by_mass.items() if s > 0)
    usable_weights = [
        {"mass_g": m, "stock": stock_by_mass[m]} for m in usable_masses
    ]

    # ---- 半径 ----
    result = plan.get("result") or {}
    cfg = result.get("config") or {}
    radius = cfg.get("radius_mm")
    if radius is None:
        radius = (plan.get("request") or {}).get("radius_mm", 100.0)
    radius = float(radius)
    if not math.isfinite(radius) or radius <= 0.0:
        raise ValueError("方案中的转子半径必须为正数。")

    # ---- 摆放明细(补偿前不平衡量) ----
    used = [p for p in (result.get("placements") or [])
            if p.get("slot") is not None]
    if not used:
        raise ValueError(
            f"方案 {plan_id} 没有已放置的试管, 无法进行库存配重补偿。"
        )
    seen_ids: set[int] = set()
    seen_slots: set[int] = set()
    base_x = base_y = 0.0
    for p in used:
        tid = int(p["tube_id"])
        if tid in seen_ids:
            raise ValueError(f"摆放明细中试管 {tid} 重复。")
        seen_ids.add(tid)
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
        base_x += mass * radius * math.cos(a)
        base_y += mass * radius * math.sin(a)
    tube_count = len(used)
    baseline_block = _vec_block(base_x, base_y)

    # ---- 空闲启用槽位(以方案结果 slot_state 为准) ----
    slot_state = result.get("slot_state") or []
    if not slot_state:
        raise ValueError(
            f"方案 {plan_id} 缺少槽位状态信息, 无法进行库存配重补偿。"
        )
    free_slots: list[dict[str, Any]] = []
    for item in slot_state:
        idx = int(item["slot"])
        ang = float(item["angle"])
        if not math.isfinite(ang):
            raise ValueError(f"槽位 {idx} 的角度必须为有限数值。")
        if bool(item.get("enabled", True)) and item.get("tube_id") is None:
            free_slots.append({"slot": idx, "angle_deg": ang})
    free_slots.sort(key=lambda f: f["slot"])
    q = len(free_slots)
    if q > MAX_FREE_SLOTS:
        raise ValueError(
            f"方案有 {q} 个空闲启用槽位, 超过库存配重补偿上限 "
            f"{MAX_FREE_SLOTS} 个(精确枚举规模随空槽数增长), 明确拒绝。"
        )
    # 各空槽方向的单位质径积向量 R·e^{iθ}
    unit_vecs = [
        complex(radius * math.cos(math.radians(f["angle_deg"])),
                radius * math.sin(math.radians(f["angle_deg"])))
        for f in free_slots
    ]

    # ---- 按放置枚数分层的完全枚举 ----
    # 叶子: (残余 |U₁|, (槽号升序槽位序列, 对应配重质量序列))
    base_mag = baseline_block["u1_mag_g_mm"]

    def _enumerate_k(k: int) -> list[tuple[float, tuple]]:
        """枚举恰好放置 k 枚配重的全部合法组合。"""
        leaves: list[tuple[float, tuple]] = []
        if k == 0:
            leaves.append((base_mag, ((), ())))
            return leaves
        if k > q or not usable_masses:
            return leaves
        t = len(usable_masses)
        stock_left = [stock_by_mass[m] for m in usable_masses]
        slots_key: list[int] = []
        masses_key: list[float] = []

        def rec(start: int, depth: int, ux: float, uy: float) -> None:
            if depth == k:
                # 量级 ≤ ~1e7 g·mm, 平方和不溢出, 直接 sqrt 比 hypot 快
                mag = math.sqrt(ux * ux + uy * uy)
                leaves.append((mag, (tuple(slots_key), tuple(masses_key))))
                return
            need = k - depth
            for i in range(start, q - need + 1):
                vx = unit_vecs[i].real
                vy = unit_vecs[i].imag
                slot = free_slots[i]["slot"]
                for ti in range(t):
                    if stock_left[ti] <= 0:
                        continue
                    m = usable_masses[ti]
                    stock_left[ti] -= 1
                    slots_key.append(slot)
                    masses_key.append(m)
                    rec(i + 1, depth + 1, ux + m * vx, uy + m * vy)
                    slots_key.pop()
                    masses_key.pop()
                    stock_left[ti] += 1

        rec(0, 0, base_x, base_y)
        return leaves

    limit_tol = _NUM_TOL * max(1.0, limit)
    evaluated = 0
    winner: Optional[dict[str, Any]] = None       # 首个达标 k 上的最优组合
    best_effort: Optional[dict[str, Any]] = None  # 0..M 内残余最低的组合
    for k in range(0, M + 1):
        leaves = _enumerate_k(k)
        evaluated += len(leaves)
        if not leaves:
            continue
        # 该枚数下的全局最优(用于无法达标时的对照)
        k_best_mag = min(m for m, _ in leaves)
        k_tol = _NUM_TOL * max(1.0, k_best_mag)
        k_tied = [(m, key) for m, key in leaves if m <= k_best_mag + k_tol]
        k_best_key = min(key for _, key in k_tied)
        k_best_mag = next(m for m, key in k_tied if key == k_best_key)
        if best_effort is None or k_best_mag < best_effort["mag"] - _NUM_TOL:
            best_effort = {"k": k, "mag": k_best_mag, "key": k_best_key,
                           "count": len(k_tied)}
        # 达标组合(第一个存在达标组合的 k 即最少放置枚数)
        feasible_leaves = [(m, key) for m, key in leaves
                           if m <= limit + limit_tol]
        if feasible_leaves:
            best_mag = min(m for m, _ in feasible_leaves)
            mag_tol = _NUM_TOL * max(1.0, best_mag)
            optimal = [(m, key) for m, key in feasible_leaves
                       if m <= best_mag + mag_tol]
            opt_key = min(key for _, key in optimal)
            opt_mag = next(m for m, key in optimal if key == opt_key)
            winner = {"k": k, "mag": opt_mag, "key": opt_key,
                      "count": len(optimal)}
            break

    # k=0 的空组合必然被评估, 故 best_effort 一定存在
    assert best_effort is not None
    chosen = winner if winner is not None else best_effort
    chosen_slots, chosen_masses = chosen["key"]

    # 由裁决序列独立复算残余 U₁(避免依赖搜索时的浮点累加), 并组装放置清单
    angle_of = {f["slot"]: f["angle_deg"] for f in free_slots}
    placements_out: list[dict[str, Any]] = []
    ux, uy = base_x, base_y
    for slot, mass in zip(chosen_slots, chosen_masses):
        a = math.radians(angle_of[slot])
        vx = mass * radius * math.cos(a)
        vy = mass * radius * math.sin(a)
        ux += vx
        uy += vy
        placements_out.append({
            "slot": slot,
            "angle_deg": angle_of[slot],
            "mass_g": mass,
            "u1_x_g_mm": vx,
            "u1_y_g_mm": vy,
            "u1_mag_g_mm": math.hypot(vx, vy),
        })
    final_block = _vec_block(ux, uy)
    final_mag = final_block["u1_mag_g_mm"]

    feasible = winner is not None
    within = final_mag <= limit + limit_tol
    base_within = base_mag <= limit + limit_tol
    excess = max(0.0, final_mag - limit)
    margin = max(0.0, limit - final_mag)
    place_txt = ", ".join(
        f"槽 {p['slot']} 放 {p['mass_g']:g} g" for p in placements_out
    ) or "无"

    if feasible and chosen["k"] == 0:
        reason = (
            f"补偿前 |U₁|={base_mag:.6g} g·mm 已不超过限值 {limit:g} g·mm, "
            "无需放置配重(放置 0 枚)。"
        )
    elif feasible:
        reason = (
            f"放置 {chosen['k']} 枚配重({place_txt})即可达标: "
            f"补偿后残余 |U₁|={final_mag:.6g} g·mm ≤ 限值 {limit:g} g·mm"
            f"(补偿前 {base_mag:.6g} g·mm); 这是能达标的最少放置枚数。"
        )
    else:
        reason = (
            f"在最多放置 {M} 枚的限制内共精确评估 {evaluated} 种槽位×质量"
            f"组合, 没有使残余 |U₁| ≤ 限值 {limit:g} g·mm 的组合; "
            f"限额内残余最低的组合为放置 {chosen['k']} 枚({place_txt}), "
            f"残余 |U₁|={final_mag:.6g} g·mm, 仍超出 {excess:.6g} g·mm, "
            "判定不达标。"
        )
    if not usable_masses:
        reason += " 注意: 所有配重种类库存均为 0, 实际无配重可放。"
    elif not free_slots and M > 0:
        reason += " 注意: 方案没有空闲启用槽位, 配重无处可放。"

    return {
        "method": (
            "按放置枚数分层的完全枚举(空闲启用槽 "
            f"≤{MAX_FREE_SLOTS} 个、配重 ≤{MAX_WEIGHT_TYPES} 种、"
            f"放置上限 0~{MAX_WEIGHTS} 枚): 精确枚举槽位与配重质量的联合"
            "组合(配重只放空槽、单槽至多一枚、每种用量不超库存, 同质量"
            "种类合并库存); 先求能达标的最少放置枚数, 再在该枚数内取残余"
            " |U₁| 最低的组合, 并列按槽号及对应质量序列升序裁决; 限额内"
            "无法达标时给出残余最低的组合与超限量, 不误报达标"
        ),
        "feasible": feasible,
        "plan": {
            "id": plan.get("id"),
            "name": plan.get("name"),
            "created_at": plan.get("created_at"),
        },
        "radius_mm": radius,
        "tube_count": tube_count,
        "free_slot_count": q,
        "free_enabled_slots": free_slots,
        "weight_types": weight_types,
        "usable_weight_types": usable_weights,
        "u1_limit_g_mm": limit,
        "max_weights": M,
        "evaluated_combinations": evaluated,
        "baseline": {
            **baseline_block,
            "within_limit": base_within,
            "excess_g_mm": max(0.0, base_mag - limit),
            "margin_g_mm": max(0.0, limit - base_mag),
        },
        "weights_used": chosen["k"],
        "placements": placements_out,
        "optimal_combination_count": chosen["count"],
        "tie_broken": chosen["count"] > 1,
        "final": final_block,
        "within_limit": within,
        "excess_g_mm": excess,
        "margin_g_mm": margin,
        "improvement_g_mm": base_mag - final_mag,
        "reason": reason,
        "note": (
            "库存配重补偿为只读规划结果, 未修改任何已保存方案; "
            "配重只放入空闲启用槽位, 单槽至多一枚, 每种用量不超过库存件数。"
        ),
    }
