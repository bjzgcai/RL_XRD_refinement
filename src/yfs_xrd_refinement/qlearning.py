# SPDX-License-Identifier: MIT
# SPDX-License-Identifier: MIT
# ============================================
# QL-yfsf.py
# ============================================

import os, itertools, re, hashlib
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Optional
from concurrent.futures import ProcessPoolExecutor
from pymatgen.core import Structure, Lattice
from pymatgen.analysis.diffraction.xrd import XRDCalculator
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.core.periodic_table import Element
import argparse
import csv
from collections import defaultdict
# ============================================
# 【新增：强化学习决策模块】
# ============================================
class QLearningRefineAgent:
    def __init__(self, actions: list, learning_rate=0.1, reward_decay=0.9, e_greedy=0.2):
        self.actions = actions  # 动作索引列表
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        # Q表：使用字典存储，键为状态，值为每个动作的得分
        self.q_table = defaultdict(lambda: np.zeros(len(actions)))

    def choose_action(self, state):
        # 状态发现与动作选择 (epsilon-greedy)
        if np.random.uniform() < self.epsilon:
            # 探索：随机选一个动作
            action = np.random.choice(self.actions)
        else:
            # 开发：选择当前状态下 Q 值最高的动作
            state_action = self.q_table[state]
            action = np.argmax(state_action)
        return action

    def learn(self, s, a, r, s_):
        # 标准 Q-Learning 更新公式
        q_predict = self.q_table[s][a]
        if s_ != 'terminal':
            q_target = r + self.gamma * np.max(self.q_table[s_])
        else:
            q_target = r
        self.q_table[s][a] += self.lr * (q_target - q_predict)


##  补充读取uiso
def sync_uiso(struct):
    """
    1) 优先保留已优化的 _Uiso（避免被默认值覆盖）
    2) 若没有 _Uiso，则从 CIF 的 Uiso/Biso 读入
    3) 同步写回 Uiso/Biso，确保导出 CIF 时能带出去
    """
    for site in struct.sites:
        if "_Uiso" in site.properties:
            U = float(site.properties["_Uiso"])
        elif "Uiso" in site.properties:
            U = float(site.properties["Uiso"])
        elif "Biso" in site.properties:
            B = float(site.properties["Biso"])
            U = B / (8.0 * np.pi * np.pi)
        else:
            U = 0.01

        site.properties["_Uiso"] = float(U)
        site.properties["Uiso"] = float(U)
        site.properties["Biso"] = float(8.0 * np.pi * np.pi * U)
    return struct



# ===========================
# 全程记录日志（Rwp / frac / scale 曲线）
# ===========================
RWP_LOG = []
FRAC_LOG = []
SCALE_LOG = []
STEP_LOG = [] 

def push_log(stage_name, rwp, fr, sf):
    RWP_LOG.append(float(rwp))
    FRAC_LOG.append([float(x) for x in fr])
    SCALE_LOG.append([float(x) for x in sf])
    STEP_LOG.append(stage_name)

# =========================================
_ALT_EARLY_STOP_PATIENCE = 50   # 连续多少次无提升提前停止
_ALT_SCORE_W_DATA = 1.0         # Rwp 权重
_ALT_SCORE_W_STOICH = 0.1       # 化学计量约束权重
# =========================================

# -----------------------------
# IO
# -----------------------------
def read_xy(path: str) -> Tuple[np.ndarray, np.ndarray]:
    data = np.loadtxt(path, comments=["#", "%", ";", "!", "@", "'"])
    x, y = data[:, 0], data[:, 1]
    y = np.clip(y, 0, None)
    if y.max() > 0: y /= y.max()
    return x, y

# -----------------------------
# March–Dollase factor
# -----------------------------
def md_factor_for_hkl(struct: Structure, hkl, axis_uvw, r: float) -> float:
    """March–Dollase 强度修正；r=1 无织构；r>1 沿轴增强；r<1 垂直增强"""
    r = float(np.clip(r, 1e-3, 1e+3))
    h, k, l = (int(hkl[0]), int(hkl[1]), int(hkl[2]))
    u, v, w = (float(axis_uvw[0]), float(axis_uvw[1]), float(axis_uvw[2]))

    if abs(u)+abs(v)+abs(w) < 1e-12:
        return 1.0

    rl = struct.lattice.reciprocal_lattice
    n_cart = rl.get_cartesian_coords([h, k, l])  # 平面法向∥h a* + k b* + l c*
    d_cart = struct.lattice.get_cartesian_coords([u, v, w])  # 取向轴 ∥ u a + v b + w c

    n_norm = np.linalg.norm(n_cart); d_norm = np.linalg.norm(d_cart)
    if n_norm < 1e-12 or d_norm < 1e-12:
        return 1.0
    cos_alpha = float(np.dot(n_cart, d_cart) / (n_norm * d_norm))
    cos2 = np.clip(cos_alpha*cos_alpha, 0.0, 1.0)
    sin2 = 1.0 - cos2

    return float((r*r*cos2 + (1.0/r)*sin2) ** (-1.5))

# -----------------------------
# XRD profile with PO (TCH-like pV)
# -----------------------------
def synth_profile_po(two_theta: np.ndarray, structure: Structure,
                     wl=1.5406,
                     U=0.003, V=0.001, W=0.020, X=0.020, Y=0.010,
                     broad_base=0.08,
                     po_axis=(0,0,1), po_r: float = 1.0,
                     enable_po: bool = False) -> np.ndarray:
    """
    若 enable_po=True，则对每个峰按 March–Dollase 修正强度（对该峰贡献的多个hkl取平均因子）
    """
    U, V, W, X, Y = [max(float(v), 0.0) for v in (U, V, W, X, Y)]
    calc = XRDCalculator(wavelength=wl)
    pat = calc.get_pattern(structure, (float(two_theta.min()), float(two_theta.max())))
    y = np.zeros_like(two_theta)
    if len(pat.x) == 0:
        return y

    deg2rad = np.pi/180.0
    for t0, I0, hkls in zip(pat.x, pat.y, pat.hkls):
        # —— PO 强度修正：多个等价 hkl 贡献的平均因子（不加权）
        if enable_po and hkls:
            facs = []
            for item in hkls:
                hkl = item.get("hkl", None)
                if hkl is None: continue
                facs.append(md_factor_for_hkl(structure, hkl, po_axis, po_r))
            if facs:
                I0 = I0 * float(np.mean(facs))

        # TCH 混合峰形
        theta = 0.5 * t0 * deg2rad
        tanth, costh = np.tan(theta), max(np.cos(theta), 1e-8)

        H_G2 = U*tanth**2 + V*tanth + W
        H_L  = X/costh    + Y*tanth
        H    = (H_G2**5 + 2.69269*H_G2**4*H_L + 2.42843*H_G2**3*H_L**2 +
                4.47163*H_G2**2*H_L**3 + 0.07842*H_G2*H_L**4 + H_L**5)**(1/5)
        H    = np.sqrt(H**2 + broad_base**2)

        ratio = H_L/max(H,1e-10)
        eta = np.clip(1.36603*ratio - 0.47719*ratio**2 + 0.11116*ratio**3, 0.0, 1.0)

        z = (two_theta - t0)/max(H,1e-10)
        g = np.exp(-4*np.log(2)*z**2)
        l = 1.0/(1.0 + (2*z)**2)
        y += I0*(eta*l + (1-eta)*g)

    if y.max() > 0: y = y / y.max()
    return y

# -----------------------------
# R-factors (Rwp / χ²)
# -----------------------------
def calc_r_factors(y_obs: np.ndarray, y_calc: np.ndarray, num_params: int = 20) -> float :
    """
    计算 
    Rwp = sqrt( Σ w (yobs-ycalc)^2 / Σ w yobs^2 ) * 100
    """
    y_obs = np.clip(y_obs, 1e-10, None)
    w = 1.0 / y_obs
    num = np.sum(w * (y_obs - y_calc)**2)
    den = np.sum(w * (y_obs**2))
    Rwp = np.sqrt(num / den) * 100.0
    return Rwp

# -----------------------------
# zero-shift interpolation
# -----------------------------
def shift_spectrum(y: torch.Tensor, shift_deg: torch.Tensor, step_deg: float) -> torch.Tensor:
    n = y.numel()
    shift_pix = shift_deg / torch.clamp(step_deg, min=1e-9)
    idx = torch.arange(n, device=y.device, dtype=torch.float32) - shift_pix
    i0 = torch.clamp(torch.floor(idx).long(), 0, n-1)
    i1 = torch.clamp(i0 + 1, 0, n-1)
    frac = idx - i0.float()
    return (1-frac)*y[i0] + frac*y[i1]

# -----------------------------
# Torch model (mix + BG + zero-shift)
# -----------------------------
class TorchRietveld(torch.nn.Module):
    def __init__(self, exp_x: np.ndarray, exp_y: np.ndarray,
                 patterns: List[np.ndarray], bg_degree: int, device: torch.device,
                 freeze_scale: bool = False):
        super().__init__()
        self.x = torch.tensor(exp_x, dtype=torch.float32, device=device)
        self.y = torch.tensor(exp_y, dtype=torch.float32, device=device)
        self.patterns = [torch.tensor(p, dtype=torch.float32, device=device) for p in patterns]
        n_phase = len(self.patterns)
        self.logits = torch.nn.Parameter(torch.zeros(n_phase, device=device))
        self.zero_shift = torch.nn.Parameter(torch.tensor(0.0, device=device))
        self.bg_params = torch.nn.Parameter(torch.zeros(bg_degree + 1, device=device))
        if freeze_scale:
            self.register_buffer("scale_factors", torch.ones(n_phase, device=device))
            self._freeze_scale = True
        else:
            self.log_scale = torch.nn.Parameter(torch.zeros(n_phase, device=device))
            self._freeze_scale = False

    def forward(self):
        weights = torch.softmax(self.logits, dim=0)
        step = torch.clamp(self.x[1] - self.x[0], min=1e-9)

        # ✅ scale > 0 且限定范围 [1e-4,]
        if self._freeze_scale:
            s_pos = torch.clamp(self.scale_factors, min=1e-4, max =5.0)
        else:
            s_pos = torch.clamp(torch.exp(self.log_scale), min=1e-4, max =5.0)

        # ✅ λ_stoich 与 scale 解耦：w 控制分配，scale 控制总强度
        raw = weights * s_pos
        frac = raw / torch.clamp(raw.sum(), min=1e-9)

        # 总强度幅值自由拟合
        amp = s_pos.sum()   
    #    amp = torch.clamp(s_pos.sum().detach(), min=1e-4, max=5.0)     amp限制幅度，可以在后续试试

        mix = torch.zeros_like(self.patterns[0])
        for f, p in zip(frac, self.patterns):
            shifted = shift_spectrum(p, self.zero_shift, step)
            mix += f * shifted

        # 背景项
        x_n = (self.x - self.x.mean()) / torch.clamp(self.x.std(), min=1e-9)
        bg = torch.zeros_like(mix); xn_pow = torch.ones_like(x_n)
        for a in self.bg_params:
            bg = bg + a * xn_pow
            xn_pow = xn_pow * x_n

        # ✅ 注意：这里要乘上 amp 才是真正强度控制
        y_pred = amp * mix + bg

        # ✅ 输出 frac（分配）和 s_pos（真实scale）
        return y_pred, frac, s_pos

# -----------------------------
# Torch refine（早停 + 梯度裁剪 + LBFGS）
# -----------------------------
def torch_refine(exp_y: np.ndarray, profiles: List[np.ndarray], device: torch.device,
                 bg_degree=5, epochs=100, lr=5e-3, weight_decay=1e-4, lbfgs=True,mode="fit",
                 freeze_scale=False, early_stop=True,  patience=_ALT_EARLY_STOP_PATIENCE,min_delta=1e-4,
                 lbfgs_lr=0.3, lbfgs_max_iter=40,
                 main_bias=0.0,
                 stoich_penalty_per_phase: Optional[List[float]] = None,
                 stoich_phase_weights: Optional[List[float]] = None,
                 lambda_stoich: float = 0.0,
                 global_logits: Optional[torch.nn.Parameter] = None
                 ):
    # 这里用等间距索引作为 x，仅用于 zero-shift 插值
    model = TorchRietveld(np.arange(len(exp_y)), exp_y, profiles, bg_degree, device, freeze_scale).to(device)

    if global_logits is not None:
        with torch.no_grad():
            n_phase = len(model.patterns)
            if global_logits.numel() == n_phase:
                # 只拷贝数值，不共享 Parameter 对象
                model.logits.data.copy_(global_logits.data)
            else:
                print(f"[警告] 全局 logits 长度 {global_logits.numel()} 与相数 {n_phase} 不符，忽略复用。")
    exp_y_t = torch.tensor(exp_y, dtype=torch.float32, device=device)
    mse = torch.nn.MSELoss(reduction="none")
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    
    # ✅【新增】在相组合阶段对主相施加 softmax 初始偏置
    if main_bias != 0.0:
        with torch.no_grad():
            if model.logits.numel() > 0:
                model.logits.data[0] += float(main_bias)

    best = {"loss": 1e9, "pred": None, "w": None, "s": None}
    no_improve = 0

    for ep in range(epochs):
        opt.zero_grad()
        y_pred, frac, s = model()
        wgt = 1.0 / torch.clamp(exp_y_t, min=1e-6)
        data_loss = torch.mean(wgt * mse(y_pred, exp_y_t))

        # 化学计量惩罚项（对 logits 可微）
        
        if lambda_stoich > 0.0 and stoich_penalty_per_phase is not None:
            pen_vec = torch.tensor(stoich_penalty_per_phase, dtype=torch.float32, device=device)
            alpha_vec = torch.tensor(stoich_phase_weights or [1.0]*len(pen_vec), dtype=torch.float32, device=device)
            stoich_term = torch.sum(frac * s * alpha_vec * pen_vec)

            if mode == "fit":  # StepA
                loss = data_loss
            elif mode == "stoich":  # StepB
                # StepB 主要关注化学计量合理性，不看Rwp
                loss = lambda_stoich * stoich_term + 0.01 * data_loss
            else:
                # 精修B：兼顾两者
                loss = data_loss + lambda_stoich * stoich_term
        else:
            loss = data_loss


        # === 自定义监控指标 ===
        if mode == "fit":
            # Step A：用 Rwp 作为早停指标
            y_np = y_pred.detach().cpu().numpy()
            exp_np = exp_y_t.detach().cpu().numpy()
            cur_metric = calc_r_factors(exp_np, y_np)  # 单位：%
        else:
            # Step B：关注化学计量偏差
            if lambda_stoich > 0.0 and stoich_penalty_per_phase is not None:
                pen_vec = torch.tensor(stoich_penalty_per_phase, dtype=torch.float32, device=device)
                alpha_vec = torch.tensor(stoich_phase_weights or [1.0]*len(pen_vec), dtype=torch.float32, device=device)
                stoich_term_eval = torch.sum(frac * s * alpha_vec * pen_vec).item()
                cur_metric = float(stoich_term_eval)
            else:
                cur_metric = float(loss.item())

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)  # 梯度裁剪
        opt.step()

        # 中程衰减
        if ep in (int(epochs*0.5), int(epochs*0.75)):
            for g in opt.param_groups:
                g["lr"] *= 0.5

        cur = loss.item()

        if cur_metric + 1e-8 < best["loss"] - min_delta:
            best.update(loss=cur_metric,
                        pred=y_pred.detach().cpu().numpy(),
                        w=frac.detach().cpu().numpy(),
                        s=(s.detach().cpu().numpy() if isinstance(s, torch.Tensor)
                        else np.ones_like(frac.detach().cpu().numpy())))
            no_improve = 0
        else:
            no_improve += 1
        if early_stop and no_improve >= patience:
            break


    if lbfgs:
        opt2 = torch.optim.LBFGS(model.parameters(), lr=lbfgs_lr, max_iter=lbfgs_max_iter,
                                 tolerance_grad=1e-7, tolerance_change=1e-9)
        def closure():
            opt2.zero_grad()
            y2, frac2, s2 = model() 
            wgt = 1.0/torch.clamp(exp_y_t, min=1e-6)
            data_loss2 = torch.mean(wgt * mse(y2, exp_y_t))

            # 化学计量惩罚项（对 logits 可微）
            if lambda_stoich > 0.0 and stoich_penalty_per_phase is not None:
                pen_vec = torch.tensor(stoich_penalty_per_phase, dtype=torch.float32, device=device)
                alpha_vec = torch.tensor(stoich_phase_weights or [1.0]*len(pen_vec), dtype=torch.float32, device=device)
                stoich_term2 = torch.sum(frac2 * s2 * alpha_vec * pen_vec)
                loss2 = data_loss2 + (lambda_stoich) * stoich_term2
            else:
                loss2 = data_loss2

            loss2.backward(); 
            return loss2
        opt2.step(closure)
        with torch.no_grad():
            # ---- 重新前向计算
            y2, w2, s2 = model()

            # ---- 数据拟合项
            wgt2 = 1.0 / torch.clamp(exp_y_t, min=1e-6)
            data_loss2_eval = torch.mean(wgt2 * (y2 - exp_y_t) ** 2)

            # ---- 化学计量约束项（若启用）
            if lambda_stoich > 0.0 and stoich_penalty_per_phase is not None:
                pen_vec = torch.tensor(stoich_penalty_per_phase, dtype=torch.float32, device=device)
                alpha_vec = torch.tensor(stoich_phase_weights or [1.0]*len(pen_vec), dtype=torch.float32, device=device)
                stoich_term2_eval = torch.sum(w2 * s2 * alpha_vec * pen_vec)
                total_loss2 = (data_loss2_eval + lambda_stoich * stoich_term2_eval).item()
            else:
                total_loss2 = data_loss2_eval.item()

            # ---- 计算与训练阶段一致的“监控指标”
            if mode == "fit":
                # 与 Step A 一致：Rwp 作为指标
                y2_np = y2.detach().cpu().numpy()
                exp_np = exp_y_t.detach().cpu().numpy()
                metric2 = calc_r_factors(exp_np, y2_np)  # %
            else:
                # 与 Step B 一致：stoich_term 作为指标（若无则退化为 data_loss）
                if lambda_stoich > 0.0 and stoich_penalty_per_phase is not None:
                    pen_vec = torch.tensor(stoich_penalty_per_phase, dtype=torch.float32, device=device)
                    alpha_vec = torch.tensor(stoich_phase_weights or [1.0]*len(pen_vec), dtype=torch.float32, device=device)
                    metric2 = float(torch.sum(w2 * s2 * alpha_vec * pen_vec).item())
                else:
                    metric2 = float(data_loss2_eval.item())

            # ---- 判断是否改进（保持与训练阶段一致的“best['loss']”语义）
            if metric2 + 1e-12 < best["loss"] - min_delta:
                best.update(
                    pred=y2.detach().cpu().numpy(),
                    w=w2.detach().cpu().numpy(),
                    s=(s2.detach().cpu().numpy() if isinstance(s2, torch.Tensor)
                    else np.ones_like(w2.detach().cpu().numpy()))
                )
                best["loss"] = metric2

        # ✅ 安全保护：防止 best 未更新导致 NoneType 错误
    if best["pred"] is None:
        with torch.no_grad():
            y_last, frac_last, s_last = model()
        best["pred"] = y_last.detach().cpu().numpy()
        best["w"] = frac_last.detach().cpu().numpy()
        best["s"] = s_last.detach().cpu().numpy()
        best["loss"] = 0.0      

    # ✅ 计算 Rwp 
    Rwp = calc_r_factors(exp_y, best["pred"])
    fracs = best["w"] / np.clip(np.sum(best["w"]), 1e-12, None)
    s_final = best["s"]

        # --- 修复单相时 0维问题 ---
    fracs = np.atleast_1d(fracs)
    s_final = np.atleast_1d(s_final)
    
    return best["pred"], fracs, s_final, Rwp

    
# -----------------------------
# 并行 worker：重建结构，读取/生成缓存的峰位置与强度，再做与 v13 完全一致的 TCH 线形与 PO 修正
# -----------------------------
def _synth_profile_worker(args):
    key, st_obj, x_grid, wl, tpars, ax, r, broad_base, enable_po = args
    # 结构重建（pickle 兼容）
    if isinstance(st_obj, dict):
        st = Structure.from_dict(st_obj)
    else:
        st = st_obj
    st = sync_uiso(st)

    # --------- DW: 按元素统计 Uiso（考虑混占据），用于“按元素种类分别计算”---------
    elem_w = {}       # 元素占据权重（你原来的）
    elem_u = {}       # 元素 Uiso 加权和（你原来的）
    elem_amp_w = {}   # ✅ 新增：元素“振幅权重”（近似用 Z * occ）

    for site in st.sites:
        U_site = float(site.properties.get("_Uiso", 0.01))
        for sp, occ in site.species.items():
            occ = float(occ)
            k = str(sp)

            # 1) 占据权重（保留）
            elem_w[k] = elem_w.get(k, 0.0) + occ
            elem_u[k] = elem_u.get(k, 0.0) + occ * U_site

            # 2) ✅ 振幅权重：用 Z 近似散射振幅权重（不引入额外重计算）
            try:
                Z = float(getattr(sp, "Z", Element(str(sp)).Z))
            except Exception:
                Z = 1.0
            elem_amp_w[k] = elem_amp_w.get(k, 0.0) + occ * Z

    # 元素 Uiso 平均
    for k in list(elem_u.keys()):
        if elem_w.get(k, 0.0) > 0:
            elem_u[k] /= elem_w[k]

    # ===== TCH 限幅（防灾难区） =====
    U = float(np.clip(tpars["U"], 0.0001, 0.15))
    V = float(np.clip(tpars["V"], -0.10,  0.10))
    W = float(np.clip(tpars["W"], 0.0001, 0.15))
    X = float(np.clip(tpars["X"], 0.0001, 0.15))
    Y = float(np.clip(tpars["Y"], 0.0001, 0.25))

    # 读取/生成 XRDCalculator 结果（峰位置 t0、原始强度 I0、对应 hkls 列表）
    kmin, kmax = float(x_grid.min()), float(x_grid.max())
    calc = XRDCalculator(wavelength=wl)
    pat = calc.get_pattern(st, (kmin, kmax))
    t_list, I_list, hkls_all = np.array(pat.x), np.array(pat.y), np.array(pat.hkls, dtype=object)


    y = np.zeros_like(x_grid)
    if len(t_list) == 0:
        return key, y

    deg2rad = np.pi/180.0
    for t0, I0, hkls in zip(t_list, I_list, hkls_all):

        # ===== Debye–Waller（按元素分别计算，再按占据权重混合；让 Uiso 真正参与强度计算）=====
        theta_dw = np.deg2rad(t0 / 2.0)   # t0: 2θ (deg)
        sin2 = float(np.sin(theta_dw) ** 2)
        wl2 = float(wl) * float(wl)

        A_sum = float(sum(elem_amp_w.values())) if len(elem_amp_w) else 0.0
        if A_sum <= 0.0:
            dw_factor = 1.0
        else:
            A_dw = 0.0
            for ek, Ak in elem_amp_w.items():
                Uk = float(elem_u.get(ek, 0.01))  # 该元素的平均 Uiso
                A_dw += float(Ak) * float(np.exp(-8.0 * np.pi * np.pi * Uk * sin2 / wl2))
            # 振幅比平方 → 强度因子
            dw_factor = float((A_dw / A_sum) ** 2)

        I0 *= dw_factor
        # --- Debye–Waller 因子结束 ---
   



        # —— PO 强度修正：多个等价 hkl 贡献的平均因子（不加权）
        if enable_po and hkls is not None:
            facs = []
            for item in hkls:
                hkl = item.get("hkl", None)
                if hkl is None:
                    continue
                facs.append(md_factor_for_hkl(st, hkl, ax, r))
            if len(facs) > 0:
                I0 = I0 * float(np.mean(facs))

        theta = 0.5 * t0 * deg2rad
        tanth, costh = np.tan(theta), max(np.cos(theta), 1e-8)

        # —— 与 v13 完全一致的 TCH 线形（含 broad_base 与 eta）
        H_G2 = U*tanth**2 + V*tanth + W
        H_L  = X/costh    + Y*tanth
        H    = (H_G2**5 + 2.69269*H_G2**4*H_L + 2.42843*H_G2**3*H_L**2 +
                4.47163*H_G2**2*H_L**3 + 0.07842*H_G2*H_L**4 + H_L**5)**(1/5)
        H    = np.sqrt(H**2 + broad_base**2)

        ratio = H_L/max(H,1e-10)
        eta = np.clip(1.36603*ratio - 0.47719*ratio**2 + 0.11116*ratio**3, 0.0, 1.0)

        z = (x_grid - t0)/max(H,1e-10)
        g = np.exp(-4*np.log(2)*z**2)
        l = 1.0/(1.0 + (2*z)**2)

        y += I0*(eta*l + (1-eta)*g)

    if y.max() > 0:
        y = y / y.max()
    return key, y

# -----------------------------
# Cell helpers
# -----------------------------
def get_cell_params(struct: Structure):
    lat = struct.lattice
    return lat.a, lat.b, lat.c, lat.alpha, lat.beta, lat.gamma

def set_cell_abc(struct: Structure, a: float, b: float, c: float) -> Structure:
    al, be, ga = struct.lattice.alpha, struct.lattice.beta, struct.lattice.gamma
    new_lat = Lattice.from_parameters(a,b,c,al,be,ga)
    species = [s.species for s in struct.sites]          # ✅ 支持混占/无序
    fracs   = [s.frac_coords for s in struct.sites]
    st_new  = Structure(new_lat, species, fracs)
    st_new  = _copy_site_properties(struct, st_new)      # 保留 _Uiso/Uiso/Biso 等
    return st_new

def set_cell_abc_angles(struct: Structure, a: float, b: float, c: float, alpha: float, beta: float, gamma: float) -> Structure:
    new_lat = Lattice.from_parameters(a,b,c,alpha,beta,gamma)
    species = [s.species for s in struct.sites]          # ✅ 支持混占/无序
    fracs   = [s.frac_coords for s in struct.sites]
    st_new  = Structure(new_lat, species, fracs)
    st_new  = _copy_site_properties(struct, st_new)      # 保留 _Uiso/Uiso/Biso 等
    return st_new


def _copy_site_properties(src: Structure, dst: Structure):
    n = min(len(src.sites), len(dst.sites))
    for i in range(n):
        dst.sites[i].properties.update(src.sites[i].properties)
    return dst

# -----------------------------
# Symmetry grouping & atom shifts
# -----------------------------
def get_equivalent_groups(struct: Structure):
    sga = SpacegroupAnalyzer(struct, symprec=5e-4, angle_tolerance=3.0)  # 分组调整
    symm = sync_uiso(sga.get_symmetrized_structure())
    groups = [list(group) for group in symm.equivalent_indices]
    return groups

def structure_with_shifted_group(struct: Structure, group_indices: List[int],
                                 dx: float=0.0, dy: float=0.0, dz: float=0.0):
    lattice = struct.lattice
    species = [s.species for s in struct]
    fracs = [s.frac_coords.copy() for s in struct]
    for idx in group_indices:
        fx, fy, fz = fracs[idx]
        fracs[idx] = [(fx+dx)%1.0, (fy+dy)%1.0, (fz+dz)%1.0]
    st_new = Structure(lattice, species, fracs)
    st_new = _copy_site_properties(struct, st_new)
    return st_new

# -----------------------------
# Mixed occupancy helpers
# -----------------------------
def get_site_occ_dict(site) -> Dict[str, float]:
    return {str(el): float(frac) for el, frac in site.species.items()}

def normalize_with_vac(occ: Dict[str, float]) -> Dict[str, float]:
    total = sum(occ.values())
    if total > 1.0:
        s = total if total > 0 else 1.0
        occ = {k: v/s for k, v in occ.items()}
        vac = 0.0
    else:
        vac = 1.0 - total
    occ2 = dict(occ)
    occ2["Vac"] = vac
    return occ2

def clamp01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))

def structure_with_mixed_occupancy(struct: Structure, site_index: int, occ_mix: Dict[str, float]) -> Structure:
    lattice = struct.lattice
    fracs = [s.frac_coords.copy() for s in struct]
    new_species = []
    for i, s in enumerate(struct):
        if i != site_index:
            new_species.append(s.species); continue
        occ = {k: float(v) for k, v in occ_mix.items() if k != "Vac"}
        occ = {k: clamp01(v) for k, v in occ.items()}
        occ2 = normalize_with_vac(occ)
        d = {k: v for k, v in occ2.items() if k != "Vac" and v > 1e-12}
        new_species.append(d)
    st_new = Structure(struct.lattice, new_species, struct.frac_coords)
    st_new = _copy_site_properties(struct, st_new)
    return st_new

def list_mixed_sites(struct: Structure) -> List[int]:
    idxs = []
    for i, s in enumerate(struct.sites):
        d = get_site_occ_dict(s)
        if len(d) > 1:
            idxs.append(i)
        elif len(d) == 1 and abs(1.0 - list(d.values())[0]) > 1e-6:
            idxs.append(i)
    return idxs

# space group occ

def structure_with_mixed_occupancy_group(struct: Structure, group_indices: List[int], occ_mix: Dict[str, float]) -> Structure:
    lattice = struct.lattice
    fracs = [s.frac_coords.copy() for s in struct]
    new_species = []

    occ = {k: float(v) for k, v in occ_mix.items() if k != "Vac"}
    occ = {k: clamp01(v) for k, v in occ.items()}
    occ2 = normalize_with_vac(occ)
    d = {k: v for k, v in occ2.items() if k != "Vac" and v > 1e-12}

    group_set = set(group_indices)
    for i, s in enumerate(struct):
        if i in group_set:
            new_species.append(d)
        else:
            new_species.append(s.species)

    st_new = Structure(struct.lattice, new_species, struct.frac_coords)
    st_new = _copy_site_properties(struct, st_new)
    return st_new


def is_mixed_group(struct: Structure, group_indices: List[int]) -> bool:
    # 用组内第一个 site 判断即可（同一 Wyckoff orbit 理应一致）
    s = struct.sites[group_indices[0]]
    d = get_site_occ_dict(s)
    if len(d) > 1:
        return True
    if len(d) == 1 and abs(1.0 - list(d.values())[0]) > 1e-6:
        return True
    return False

# -----------------------------
# Stoichiometry soft-constraint
# -----------------------------
def normalized_composition_vector(struct: Structure, target_keys: List[str]) -> np.ndarray:
    cdict = struct.composition.get_el_amt_dict()
    vec = np.array([float(cdict.get(k, 0.0)) for k in target_keys], dtype=float)
    s = vec.sum()
    if s > 0:
        vec /= s
    return vec

def stoich_penalty_for_phase(struct: Structure, target: Dict[str, float]) -> float:
    keys = sorted(set(target.keys()))
    tvec = np.array([float(target[k]) for k in keys], dtype=float)
    ts = tvec.sum()
    tvec = (tvec/ts) if ts > 0 else tvec
    cvec = normalized_composition_vector(struct, keys)
    diff = cvec - tvec
    return float(np.dot(diff, diff))  # L2^2

# -----------------------------
# 外层：三阶段（粗→微→精）；加入 PO(r) 优化
# -----------------------------
def outer_refine_all(x_grid: np.ndarray, y_obs: np.ndarray,
                     phase_structs: Dict[str, Structure],
                     tch_params: Dict[str, float],
                     wavelength: float,
                     broad_base=0.08,
                     bg_degree=5,
                     # Cell/Angle
                     init_cell_step_frac=0.0025, min_cell_step_frac=0.0002,
                     init_angle_step_deg=0.3,   min_angle_step_deg=0.05,
                     # TCH
                     init_tch_step_frac=0.20,   min_tch_step_frac=0.05,
                     # Atoms xyz
                     init_pos_step=0.005, min_pos_step=0.002,
                     # Mixed occupancy
                     init_mix_step=0.08, min_mix_step=0.004,
                     # Preferred Orientation (PO)
                     enable_po=False,
                     po_axes: Optional[Dict[str, Tuple[int,int,int]]] = None,
                     po_r_init: Optional[Dict[str, float]] = None,
                     init_po_step_frac=0.25, min_po_step_frac=0.05,
                     po_r_bounds=(0.2, 5.0),
                     # Stoichiometry
                     lambda_stoich=0.0,
                     stoich_phase_key: Optional[str]=None,
                     stoich_target: Optional[Dict[str, float]]=None,
                     # stage loops
                     stage_loops=(60, 100, 150),
                     device=None,
                     num_workers: int = os.cpu_count()):
    

    import numpy as np  
    # --- 强制主相文件名标准化，防止路径不匹配（非常关键）
    if stoich_phase_key is not None:
        stoich_phase_key = os.path.basename(stoich_phase_key)
    phase_structs = {os.path.basename(k): v for k, v in phase_structs.items()}
    # === 冻结 Wyckoff 分组：始终使用初始 CIF 的对称性 ===
    fixed_groups_map: Dict[str, List[List[int]]] = {}
    for key, st in phase_structs.items():
        sga0 = SpacegroupAnalyzer(st, symprec=5e-4, angle_tolerance=3.0)
        symm0 = sga0.get_symmetrized_structure()
        fixed_groups_map[key] = [list(g) for g in symm0.equivalent_indices]

    print("📌 已冻结 Wyckoff 分组：")
    for key, groups in fixed_groups_map.items():
        print(f"   {os.path.basename(key)}: {len(groups)} 组")

    # ===【新增】为所有原子初始化 Uiso（如果缺失则给默认值 0.01）===
    for key, st in phase_structs.items():
        for site in st.sites:
            # 1. CIF 自带 Uiso
            if "Uiso" in site.properties:
                U = float(site.properties["Uiso"])

            # 2. CIF 自带 Biso → 转换成 Uiso
            elif "Biso" in site.properties:
                Biso = float(site.properties["Biso"])
                U = Biso / (8 * np.pi * np.pi)

            # 3. 都没有 → 使用默认值
            else:
                U = 0.01

            # 统一存入内部字段，后续 DW 和 B-loop 优化都用它
            site.properties["_Uiso"] = U
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- PO params bookkeeping
    if po_axes is None:
        po_axes = {k: (0,0,1) for k in phase_structs.keys()}
    if po_r_init is None:
        po_r_init = {k: 1.0 for k in phase_structs.keys()}
    po_r = {k: float(np.clip(po_r_init.get(k, 1.0), po_r_bounds[0], po_r_bounds[1])) for k in phase_structs.keys()}


    from collections import OrderedDict

    # =========================
    # Profile 缓存（按相粒度）
    # =========================
    _PROFILE_CACHE = OrderedDict()
    _PROFILE_CACHE_MAX = 256  # 够用；太大占内存
    _prof_hit = 0
    _prof_miss = 0

    def reset_profile_cache(tag=""):
        nonlocal _prof_hit, _prof_miss
        _PROFILE_CACHE.clear()
        _prof_hit = 0
        _prof_miss = 0
        if tag:
            print(f"🔄 [ProfileCache] reset: {tag}", flush=True)
    reset_profile_cache("outer_refine_all-start")
    def _tpars_key(tpars: dict):
        # 只取真正影响 profile 的峰形参数；round 是为了避免浮点噪声导致“假失效”
        return (round(float(tpars["U"]), 8),
                round(float(tpars["V"]), 8),
                round(float(tpars["W"]), 8),
                round(float(tpars["X"]), 8),
                round(float(tpars["Y"]), 8))

    def _phase_profile_key(phase_key: str, st: Structure, tpars: dict, po_r_dict: dict):
        # ✅ 快速：用 id(st) 当结构指纹（你代码里结构基本都是“新建对象”而不是原地改，所以安全且很快）
        # 如果你将来有“原地修改结构”的写法，再换成 hash(st.as_dict()) 那类慢但稳的方案。
        st_id = id(st)

        # 影响 profile 的参数（必须进 key）
        tkey = _tpars_key(tpars)
        pr = float(po_r_dict.get(phase_key, 1.0))
        ax = tuple(po_axes.get(phase_key, (0, 0, 1)))
        return (phase_key, st_id, tkey, round(float(pr), 8), ax,
                round(float(broad_base), 8), bool(enable_po), round(float(wavelength), 8),
                # x_grid 一般不变，但把边界/长度带上，防止你换数据时“误复用”
                int(len(x_grid)), round(float(x_grid[0]), 8), round(float(x_grid[-1]), 8))

    # =========================
    # 常驻进程池（强烈建议）
    # =========================
    _profile_pool = ProcessPoolExecutor(max_workers=num_workers)
    try:
            def make_profiles(structs_dict, tpars, po_r_dict):
                nonlocal _prof_hit, _prof_miss

                # 1) 先查缓存：能复用的直接拿
                profs = [None] * len(structs_dict)
                keys = list(structs_dict.keys())

                items_to_compute = []
                need_idx = []
                cache_keys = []

                for i, key in enumerate(keys):
                    st = structs_dict[key]
                    ck = _phase_profile_key(key, st, tpars, po_r_dict)

                    cache_keys.append(ck)
                    if ck in _PROFILE_CACHE:
                        profs[i] = _PROFILE_CACHE[ck]
                        _PROFILE_CACHE.move_to_end(ck)  # LRU
                        _prof_hit += 1
                    else:
                        # 需要计算的任务（只发缺失的相）
                        items_to_compute.append(
                            (key, st.as_dict(), x_grid, wavelength, tpars,
                            po_axes.get(key, (0, 0, 1)),
                            po_r_dict.get(key, 1.0),
                            broad_base, enable_po)
                        )
                        need_idx.append(i)
                        _prof_miss += 1

                # 2) 只并行计算缺失的相
                if items_to_compute:
                    results = list(_profile_pool.map(_synth_profile_worker, items_to_compute))
                    # results: [(phase_key, y_profile), ...]
                    got = {k: y for (k, y) in results}

                    for i in need_idx:
                        key = keys[i]
                        ck = cache_keys[i]
                        y = got[key]
                        profs[i] = y

                        # 写缓存 + LRU 淘汰
                        _PROFILE_CACHE[ck] = y
                        _PROFILE_CACHE.move_to_end(ck)
                        while len(_PROFILE_CACHE) > _PROFILE_CACHE_MAX:
                            _PROFILE_CACHE.popitem(last=False)

                return profs

            # =====================================================
            # 【RL 植入】1. 初始化智能体与状态记录
            # =====================================================
            # 动作定义：0:Cell, 1:Angle, 2:TCH, 3:PO, 4:Atoms, 5:Uiso, 6:Occ
            rl_actions = list(range(7))
            agent = QLearningRefineAgent(actions=rl_actions)
            
            # 用于记录 Rwp 的历史，辅助 RL 判断“状态”
            rwp_trend_history = []
            # rwp_trend_history.append(best_score) # 记录初始分数，让状态判定有据可依
            def get_refine_state():
                """根据 Rwp 的下降斜率判断当前处于什么状态"""
                if len(rwp_trend_history) < 2:
                    return "Initial"
                
                # 计算最近两次的差值
                diff = rwp_trend_history[-2] - rwp_trend_history[-1]
                
                if diff > 0.5:
                    return "Fast_Drop"   # 降得很快
                elif diff > 0.02:
                    return "Slow_Drop"   # 降得慢了
                elif diff <= 0:
                    return "Stagnant"    # 几乎不动或反弹了
                else:
                    return "Normal"

            # =====================================================

            def inner_once(structs_dict, tpars, po_r_dict, freeze_scale=False, stage_name="Stage", lambda_stoich=0.0):

                # ✅ 修改2：Uiso 试探优先，用更小的 epochs
                if "Uiso" in stage_name:
                    e, l2 = 100, 0.20
                elif "粗调" in stage_name:
                    e, l2 = 100, 0.30
                elif "微调" in stage_name:
                    e, l2 = 100, 0.30
                else:
                    e, l2 = 100, 0.25

                profiles = make_profiles(structs_dict, tpars, po_r_dict)

                # 计算每相的 stoichiometric 偏差和权重
                pen_list, alpha_list = None, None
                if lambda_stoich > 0.0 and stoich_target:
                    pen_list, alpha_list = [], []
                    for k, st in structs_dict.items():
                        pen = stoich_penalty_for_phase(st, stoich_target)
                        is_main = (
                    stoich_phase_key and (
                            os.path.basename(k) == stoich_phase_key or
                            os.path.splitext(os.path.basename(k))[0] == os.path.splitext(stoich_phase_key)[0]
                        )
                    )
                        alpha = 0.1 if is_main else 0.8
                        pen_list.append(pen)
                        alpha_list.append(alpha)

                # ===== 单次 refine 调用（外层控制 StepA / StepB） =====
                yfit, fr, sf, Rwp = torch_refine(
                    y_obs, profiles, device=device, bg_degree=bg_degree,
                    freeze_scale=freeze_scale, epochs=e, lbfgs_lr=l2, lbfgs_max_iter=20,
                    lr=5e-3, weight_decay=1e-4,
                    main_bias=0.0,
                    stoich_penalty_per_phase=pen_list,
                    stoich_phase_weights=alpha_list,
                    lambda_stoich=lambda_stoich,
                    mode="fit" if lambda_stoich == 0.0 else "stoich",
                )

                # 综合评分（越小越好）
                stoich_term = 0.0
                if lambda_stoich > 0.0 and pen_list is not None:
                    p = np.asarray(pen_list, dtype=float)
                    a = np.asarray(alpha_list, dtype=float)
                    fr_np = np.asarray(fr, dtype=float)
                    sf_np = np.asarray(sf, dtype=float)
                    stoich_term = float(np.sum(p * a * fr_np * sf_np))

                score = _ALT_SCORE_W_DATA * Rwp + _ALT_SCORE_W_STOICH * stoich_term

                # ====================================================

                # ✅ 仅第一次打印主相识别情况
                if lambda_stoich > 0.0 and stoich_target and not hasattr(inner_once, "_stoich_printed"):
                    print(f"\n📘 [StoichPenalty] λ_stoich = {lambda_stoich:.3f}")
                    print(f"👉 已识别主相为：{stoich_phase_key}")
                    print("🧩 各相的化学计量约束权重：")
                    for k in structs_dict.keys():
                        is_main = (stoich_phase_key and os.path.basename(k) == os.path.basename(stoich_phase_key))
                        w = 1.0 if is_main else 0.5
                        tag = "主相" if is_main else "杂相"
                        print(f"   ├─ {os.path.basename(k):<25s} | 类型: {tag:<3s} | λ_phase = {w*lambda_stoich:.3f}")
                    print("-------------------------------------------------------")
                    inner_once._stoich_printed = True  # 防止重复打印

                RWP_LOG.append(float(score))
                FRAC_LOG.append([float(x) for x in fr])
                SCALE_LOG.append([float(x) for x in sf])
                STEP_LOG.append(stage_name)
                return score, yfit, fr, sf, Rwp, profiles

            # init
            tpars = dict(tch_params)
            best_score, yfit, fr, sf, r_best,  _ = inner_once(phase_structs, tpars, po_r, freeze_scale=False, stage_name="初始化")
            print(f"\n🔁 外层精修启动：Rwp={r_best:.2f}% |  score={best_score:.2f}")

            ANG_MIN, ANG_MAX = 20.0, 160.0                # 限制角度范围

            # 阶段设置（步长比例 + 循环次数 ）
            stage_settings = [
                {"name": "粗调 (Stage 1)",
                    "cell_scale": 1.5, "angle_scale": 1.5, "tch_scale": 1.8,
                    "pos_scale": 4.0, "mix_scale": 2.5, "po_scale": 1.6,
                    "loops": stage_loops[0], },
                {"name": "微调 (Stage 2)",
                    "cell_scale": 1.0, "angle_scale": 1.2, "tch_scale": 1.2,
                    "pos_scale": 3.0, "mix_scale": 2.0, "po_scale": 1.2,
                    "loops": stage_loops[1], },
                {"name": "精调 (Stage 3)",
                    "cell_scale": 0.7, "angle_scale": 1.0, "tch_scale": 0.9,
                    "pos_scale": 2.5, "mix_scale": 1.8, "po_scale": 1.0,
                    "loops": stage_loops[2], },
            ]

            for stg in stage_settings:
                print(f"\n🚀 进入阶段：{stg['name']} | loops={stg['loops']} ")

                # ✅ 阶段保温读取
                if 'stage_state' in locals():
                    phase_structs = stage_state["phase_structs"]
                    tpars = stage_state["tpars"]
                    po_r = stage_state["po_r"]
                    yfit = stage_state["yfit"]
                    fr = stage_state["fr"]
                    sf = stage_state["sf"]
                    print(f"♻️ 已载入上一阶段的结构与比例，作为 {stg['name']} 初始状态。")

                # =====================================================
                # StepA + StepB 串行执行
                # =====================================================
                print(f"\n▶️ {stg['name']} StepA：结构拟合阶段（λ_stoich=0.0）")
                best_score, yfit, fr, sf, r_best, _ = inner_once(
                    phase_structs, tpars, po_r,
                    freeze_scale=False,
                    stage_name=f"{stg['name']}-StepA",
                    lambda_stoich=0.0
                )

                print(f"✅ StepA 完成：Rwp={r_best:.2f}% | score={best_score:.2f}")
                current_step = "StepA"

                # ✅ 新增：保存 StepA 的 logits 向量（取 log(frac)）
                global_logits = torch.tensor(np.log(fr + 1e-9), dtype=torch.float32).to(device)
                if stg["name"] == "粗调 (Stage 1)":
                    print(f"{stg['name']} StepB：化学计量修正（λ_stoich={lambda_stoich})")
                    best_score, yfit, fr, sf, r_best, _ = inner_once(
                        phase_structs, tpars, po_r,
                        freeze_scale=False,
                        stage_name=f"{stg['name']}-StepB",
                        lambda_stoich=lambda_stoich
                    )
                    print(f"✅ StepB 完成：Rwp={r_best:.2f}% | score={best_score:.2f}")
                else:
                    print(f"⛔ 跳过 {stg['name']} 的 StepB（避免破坏结构）")

                current_step = "StepB"
                # 阶段步长
                cell_step  = init_cell_step_frac * stg["cell_scale"]
                angle_step = init_angle_step_deg  * stg["angle_scale"]
                tch_step   = init_tch_step_frac   * stg["tch_scale"]
                pos_step   = init_pos_step        * stg["pos_scale"]
                mix_step   = init_mix_step        * stg["mix_scale"]
                po_step    = init_po_step_frac    * stg["po_scale"]

                # =========================
                # A) 先优化 Cell + Angles + TCH + PO(r)
                # =========================
                for loop in range(1, stg["loops"] + 1):
                    loop_label = current_step if 'current_step' in locals() else ''
                    improved = False

                    # --- Cell (a,b,c)
                    if cell_step >= min_cell_step_frac:
                        for key in list(phase_structs.keys()):
                            st0 = phase_structs[key]
                            a0,b0,c0,al0,be0,ga0 = get_cell_params(st0)
                            for p_name in ("a","b","c"):
                                best_local = (best_score, 0.0); best_state=None; best_metrics=None
                                for sgn in (-1.0, +1.0):
                                    delta = sgn*cell_step
                                    a,b,c = a0,b0,c0
                                    if p_name=="a": a=a0*(1+delta)
                                    if p_name=="b": b=b0*(1+delta)
                                    if p_name=="c": c=c0*(1+delta)
                                    st_try = sync_uiso(set_cell_abc(st0, a,b,c))
                                    structs_try = dict(phase_structs); structs_try[key]=st_try
                                    score_try, yfit_try, fr_try, sf_try, r_try,  _ = inner_once(structs_try, tpars, po_r,
                                                                                                freeze_scale= False,
                                                                                                stage_name=stg["name"])
                                    if score_try + 1e-3 < best_local[0]:
                                        best_local=(score_try,delta); best_state=(structs_try,yfit_try,fr_try,sf_try,r_try); 
                                if best_state and best_local[0] + 1e-6 < best_score:
                                    phase_structs, yfit, fr, sf, r_best = best_state
                                    best_score = best_local[0]; improved=True
                                    print(f"[{stg['name']} | {loop_label} | A-Loop {loop:03d}] ✅ cell {os.path.basename(key)}.{p_name} "
                                            f"{best_local[1]*100:+.3f}% → Rwp={r_best:.2f}% | score={best_score:.2f}")

                    # --- Angles (α,β,γ)
                    if angle_step >= min_angle_step_deg:
                        for key in list(phase_structs.keys()):
                            st0 = phase_structs[key]
                            a0,b0,c0,al0,be0,ga0 = get_cell_params(st0)
                            for p_name in ("alpha","beta","gamma"):
                                best_local=(best_score,0.0); best_state=None; best_metrics=None
                                for sgn in (-1.0,+1.0):
                                    delta = sgn*angle_step
                                    al,be,ga = al0,be0,ga0
                                    if p_name=="alpha": al=float(np.clip(al0+delta, ANG_MIN, ANG_MAX))
                                    if p_name=="beta":  be=float(np.clip(be0+delta, ANG_MIN, ANG_MAX))
                                    if p_name=="gamma": ga=float(np.clip(ga0+delta, ANG_MIN, ANG_MAX))
                                    st_try = sync_uiso(set_cell_abc_angles(st0, a0,b0,c0, al,be,ga))
                                    structs_try = dict(phase_structs); structs_try[key]=st_try
                                    score_try, yfit_try, fr_try, sf_try, r_try, _ = inner_once(structs_try, tpars, po_r,
                                                                                                freeze_scale= False,
                                                                                                stage_name=stg["name"])
                                    if score_try + 1e-4 < best_local[0]:
                                        best_local=(score_try,delta); best_state=(structs_try,yfit_try,fr_try,sf_try,r_try); 
                                if best_state and best_local[0] + 1e-6 < best_score:
                                    phase_structs, yfit, fr, sf, r_best = best_state
                                    best_score = best_local[0]; improved=True
                                    print(f"[{stg['name']} | A-Loop {loop:03d}| {loop_label}] ✅ angle {os.path.basename(key)}.{p_name} "
                                            f"{best_local[1]:+.3f}° → Rwp={r_best:.2f}% | score={best_score:.2f}")

                    # --- TCH (U,V,W,X,Y)
                    if tch_step >= min_tch_step_frac:
                        for name in ("U","V","W","X","Y"):
                            base = tpars[name]
                            best_local=(best_score,0.0); best_state=None; best_metrics=None
                            for sgn in (-1.0,+1.0):
                                delta = sgn*tch_step
                                val = base*(1.0+delta)
                                # ===== TCH 限幅（尝试值也必须限制） =====
                                if name == "U": val = float(np.clip(val, 0.0001, 0.15))
                                elif name == "V": val = float(np.clip(val, -0.10,  0.10))
                                elif name == "W": val = float(np.clip(val, 0.0001, 0.15))
                                elif name == "X": val = float(np.clip(val, 0.0001, 0.15))
                                elif name == "Y": val = float(np.clip(val, 0.0001, 0.25))
                                t_try = dict(tpars); t_try[name]=val
                                score_try, yfit_try, fr_try, sf_try, r_try,  _ = inner_once(phase_structs, t_try, po_r,
                                                                                            freeze_scale= False,
                                                                                            stage_name=stg["name"])

                                if score_try + 1e-3 < best_local[0]:
                                    best_local = (score_try, delta)
                                    best_state = (t_try, yfit_try, fr_try, sf_try, r_try)
                            if best_state is not None and best_local[0] + 1e-6 < best_score:
                                    tpars, yfit, fr, sf, r_best = best_state
                                    best_score = best_local[0]
                                    improved = True
                                    print(f"[{stg['name']} | A-Loop {loop:03d}] ✅ TCH {name} {best_local[1]*100:+.1f}% "
                                        f"→ Rwp={r_best:.2f}% | score={best_score:.2f}")


                    # --- PO (r) 每相独立（乘性步长，边界 [po_r_bounds]）
                    if enable_po and po_step >= min_po_step_frac:
                        for key in list(phase_structs.keys()):
                            r0 = po_r[key]
                            best_local=(best_score,0.0); best_state=None; best_metrics=None
                            for sgn in (-1.0, +1.0):
                                delta = sgn*po_step
                                r_try_val = float(np.clip(r0*(1.0+delta), po_r_bounds[0], po_r_bounds[1]))
                                po_r_try = dict(po_r); po_r_try[key] = r_try_val
                                score_try, yfit_try, fr_try, sf_try, r_try, _ = inner_once(phase_structs, tpars, po_r_try,
                                                                                            freeze_scale= False,
                                                                                            stage_name=stg["name"])
                                if score_try + 1e-3 < best_local[0]:
                                    best_local=(score_try, r_try_val); best_state=(po_r_try, yfit_try, fr_try, sf_try, r_try); 
                            if best_state and best_local[0] + 1e-6 < best_score:
                                po_r, yfit, fr, sf, r_best = best_state
                                best_score = best_local[0]; improved=True
                                print(f"[{stg['name']} | A-Loop {loop:03d}] ✅ PO r {os.path.basename(key)} "
                                        f"→ r={best_local[1]:.3f} | Rwp={r_best:.2f}% | score={best_score:.2f}")

                    # 若本轮无提升，则衰减步长
                    if not improved:
                        did_decay=False
                        prev = (cell_step, angle_step, tch_step, po_step)
                        if cell_step  >= min_cell_step_frac: cell_step  *= 0.7; did_decay=True
                        if angle_step >= min_angle_step_deg: angle_step *= 0.7; did_decay=True
                        if tch_step   >= min_tch_step_frac:  tch_step   *= 0.7; did_decay=True
                        if po_step    >= min_po_step_frac:   po_step    *= 0.7; did_decay=True
                        if did_decay:
                            print(f"[{stg['name']} | A-Loop {loop:03d}] ↘️ 步长衰减："
                                    f"cell {prev[0]:.5f}->{cell_step:.5f} | angle {prev[1]:.3f}->{angle_step:.3f} | "
                                    f"tch {prev[2]:.3f}->{tch_step:.3f} | po {prev[3]:.3f}->{po_step:.3f} "
                                    f"| Rwp={r_best:.2f}% | score={best_score:.2f}")
                        else:
                            print(f"[{stg['name']} | A-Loop {loop:03d}] ⛳ 已到最小步长阈值，结束 A 段。")
                            break

                # =========================
                # B) 再优化 Atoms + Mixed Occupancy
                # =========================
                for loop in range(1, stg["loops"] + 1):
                    improved = False

                    # --- Atoms xyz（按等价位组微调）
                    if pos_step >= min_pos_step:
                        for key in list(phase_structs.keys()):
                            st0 = phase_structs[key]
                            # 粗调：使用初始化时冻结的 Wyckoff 分组，严格跟随原空间群
                            if stg["name"] == "粗调 (Stage 1)":
                                groups = fixed_groups_map[key]
                            # 精调：打开硬分组，按当前结构重新分组（甚至可以改成每个原子一组）
                            elif stg["name"] == "精调 (Stage 3)":
                                # 方案 A：按当前结构重新用 SpacegroupAnalyzer 分组（可能接近 P1）

                                # 👉 如果你想“完全每个原子单独调”，可以改成下面这一行：
                                groups = [[i] for i in range(len(st0.sites))]
                            # 微调（Stage 2）：保持固定分组
                            else:
                                groups = groups = get_equivalent_groups(st0)
                            for gidx, group in enumerate(groups):
                                for axis, vec in zip(("x","y","z"), [(1,0,0),(0,1,0),(0,0,1)]):
                                    best_local=(best_score,0.0); best_state=None; best_metrics=None
                                    for sgn in (-1.0,+1.0):
                                        delta = sgn*pos_step
                                        dx,dy,dz = delta*vec[0], delta*vec[1], delta*vec[2]
                                        st_try = sync_uiso(structure_with_shifted_group(st0, group, dx,dy,dz))
                                        structs_try = dict(phase_structs); structs_try[key]=st_try
                                        score_try, yfit_try, fr_try, sf_try, r_try,  _ = inner_once(structs_try, tpars, po_r,
                                                                                                    freeze_scale= False,
                                                                                                    stage_name=stg["name"])
                                        if score_try + 5e-4 < best_local[0]:
                                            best_local=(score_try,delta); best_state=(structs_try,yfit_try,fr_try,sf_try,r_try); 
                                    if best_state and best_local[0] + 1e-6 < best_score:
                                        phase_structs, yfit, fr, sf, r_best = best_state
                                        best_score = best_local[0]; improved=True
                                        print(f"[{stg['name']} | B-Loop {loop:03d}] ✅ atoms {os.path.basename(key)} "
                                                f"group#{gidx} {axis} {best_local[1]:+.4f} → Rwp={r_best:.2f}%  "
                                                f"pos_step={pos_step:.4f} mix_step={mix_step:.3f}")
                    # --- Uiso（原子热振动）微调模块 ---
                    u_step = pos_step * 0.5
                    min_u = 0.0005
                    max_u = 0.05

                    # 为整个 Uiso 搜索准备一个全局 best
                    best_local_uiso = (best_score, None)

                    # 遍历所有相
                    for key in list(phase_structs.keys()):
                        st0 = phase_structs[key]

                        # 遍历所有 site
                        for sidx, site in enumerate(st0.sites):

                            U0 = float(site.properties.get("_Uiso", 0.01))

                            # 对某一个 site 局部搜索 +u/-u
                            for sgn in (-1.0, +1.0):

                                U_try = float(np.clip(U0 + sgn * u_step, min_u, max_u))

                                st_try = st0.copy()
                                st_try.sites[sidx].properties["_Uiso"] = U_try

                                structs_try = dict(phase_structs)
                                structs_try[key] = st_try

                                score_try, yfit_try, fr_try, sf_try, r_try, _ = inner_once(
                                    structs_try, tpars, po_r,
                                    freeze_scale=False,
                                    stage_name=f"{stg['name']}-Uiso"
                                )

                                # 更新 Uiso 全局最优
                                if score_try + 1e-4 < best_local_uiso[0]:
                                    best_local_uiso = (score_try, (structs_try, yfit_try, fr_try, sf_try, r_try))

                    # ←←← 注意：这里已经跳出 for key 和 for site 循环（左移一级缩进）
                    # 在本 B-loop 内立即 apply 全局最佳 Uiso
                    if best_local_uiso[1] is not None:
                        structs_try, yfit, fr, sf, r_best = best_local_uiso[1]
                        phase_structs = structs_try
                        best_score = best_local_uiso[0]
                        improved = True
                        print(f"[{stg['name']} | B-Loop {loop:03d}]  Uiso optimized → Rwp={r_best:.2f}%")
   

                    # --- Mixed occupancies（含 Vac）
                    if mix_step >= min_mix_step:
                        for key in list(phase_structs.keys()):
                            st0 = phase_structs[key]
                            site_indices = list_mixed_sites(st0)
                            for sidx in site_indices:
                                site = st0.sites[sidx]
                                occ0 = get_site_occ_dict(site)
                                occ0 = normalize_with_vac(occ0)
                                elems = list(occ0.keys())  # 含“Vac”
                                for e in elems:
                                    cands = [("Vac", +mix_step), ("Vac", -mix_step)] if e=="Vac" else [(e, +mix_step), (e, -mix_step)]
                                    best_local=(best_score, None); best_state=None; best_metrics=None
                                    for (elem, stepval) in cands:
                                        occ_try = dict(occ0)
                                        occ_try[elem] = clamp01(occ_try.get(elem, 0.0) + stepval)
                                        occ_try = normalize_with_vac({k:v for k,v in occ_try.items() if k!="Vac"})
                                        st_try = sync_uiso(structure_with_mixed_occupancy(st0, sidx, occ_try))
                                        structs_try = dict(phase_structs); structs_try[key]=st_try
                                        # 混占阶段建议冻结 scale，避免与占据率耦合
                                        score_try, yfit_try, fr_try, sf_try, r_try, _ = inner_once(structs_try, tpars, po_r,
                                                                                                    freeze_scale=False ,
                                                                                                    stage_name=stg["name"])
                                        if score_try + 1e-4 < best_local[0]:
                                            best_local=(score_try, (elem, stepval, occ_try)); best_state=(structs_try,yfit_try,fr_try,sf_try,r_try); 
                                    if best_state and best_local[0] + 1e-6 < best_score:
                                        phase_structs, yfit, fr, sf, r_best = best_state

                                        best_score = best_local[0]; improved=True
                                        elem, stepval, occ_final = best_local[1]
                                        occ_str = ", ".join([f"{k}:{v:.3f}" for k,v in occ_final.items()])
                                        print(f"[{stg['name']} | B-Loop {loop:03d}] ✅ mix {os.path.basename(key)} site#{sidx} "
                                                f"{elem} {stepval:+.3f} → Rwp={r_best:.2f}% "
                                                f"occ={{ {occ_str} }} | pos_step={pos_step:.4f} mix_step={mix_step:.3f}")

                    # 若本轮无提升，则衰减步长
                    if not improved:
                        did_decay=False
                        prev = (pos_step, mix_step)
                        if pos_step >= min_pos_step:  pos_step  *= 0.7; did_decay=True
                        if mix_step >= min_mix_step:  mix_step  *= 0.7; did_decay=True
                        if did_decay:
                            print(f"[{stg['name']} | B-Loop {loop:03d}] ↘️ 步长衰减："
                                    f"pos {prev[0]:.4f}->{pos_step:.4f} | mix {prev[1]:.3f}->{mix_step:.3f} "
                                    f"| Rwp={r_best:.2f}% ")
                        else:
                            print(f"[{stg['name']} | B-Loop {loop:03d}] ⛳ 已到最小步长阈值，结束 B 段。")
                            break

                print(f"✅ 阶段完成：{stg['name']} | 当前 Rwp={r_best:.2f}% | score={best_score:.2f}")

                # === 【新增】粗调 (Stage 1) 结果完整保存 ===
                if stg["name"] == "粗调 (Stage 1)":
                    out_prefix = "Stage1_Refined"
                    os.makedirs("stage1_output", exist_ok=True)

                    # 图像
                    plt.figure(figsize=(10,6))
                    plt.plot(x_grid, y_obs, lw=1.0, label="Experiment")
                    plt.plot(x_grid, yfit, lw=1.0, label="Stage1 Fit")
                    plt.xlabel("2θ (deg)")
                    plt.ylabel("Normalized Intensity")
                    plt.title("Stage 1  Refinement Result")

                    text_lines = []
                    text_lines.append("Composition:")
                    for f, w in zip(phase_structs.keys(), fr):
                        text_lines.append(f"  {os.path.basename(f)}: {w*100:.2f}%")
                    text_lines.append("")
                    text_lines.append(f"Rwp = {r_best:.2f}%")
                    text_str = "\n".join(text_lines)

                    plt.text(1.02, 0.98, text_str, transform=plt.gca().transAxes,
                             fontsize=10, va="top", ha="left",
                             bbox=dict(facecolor="white", alpha=0.8, edgecolor="gray"))
                    plt.legend()
                    plt.tight_layout(rect=[0, 0, 0.8, 1])
                    plt.savefig(f"stage1_output/{out_prefix}.png", dpi=300)
                    plt.close()
                    print(f"🖼️ 已保存粗调阶段图像：stage1_output/{out_prefix}.png")

                    # xy 拟合曲线
                    xy_out = np.column_stack([x_grid, y_obs, yfit])
                    np.savetxt(f"stage1_output/{out_prefix}.xy", xy_out, fmt="%.6f",
                               header="2Theta  Intensity_Obs  Intensity_Fit")
                    print(f"💾 已保存粗调阶段谱线：stage1_output/{out_prefix}.xy")

                    # 文本报告
                    with open(f"stage1_output/{out_prefix}.txt", "w", encoding="utf-8") as fw:
                        fw.write("=== Stage 1 Refinement Result ===\n")
                        fw.write(f"Rwp : {r_best:.3f}%\n")
                        fw.write("TCH params: " + ", ".join([f"{k}={v:.6g}" for k,v in tpars.items()]) + "\n")
                        fw.write("\nPhases:\n")
                        for f, w, s in zip(phase_structs.keys(), fr, sf):
                            fw.write(f"  {os.path.basename(f):<28s} frac={w*100:6.2f}% | scale={float(s):.4f}\n")
                    print(f"🧾 已保存粗调阶段文本报告：stage1_output/{out_prefix}.txt")

                    # 导出 CIF
                    cif_dir = "stage1_output/cifs"
                    os.makedirs(cif_dir, exist_ok=True)
                    for fpath, struct in phase_structs.items():
                        base = os.path.basename(fpath)
                        name, _ = os.path.splitext(base)
                        out_path = os.path.join(cif_dir, f"{name}_Stage1.cif")
                        struct.to(filename=out_path)
                    print(f"💾 已导出粗调阶段 CIF 文件至 {cif_dir}/")

                # === 【新增】微调 (Stage 2) 结果完整保存 ===
                if stg["name"] == "微调 (Stage 2)":
                    out_prefix = "Stage2_Refined"
                    os.makedirs("stage2_output", exist_ok=True)

                    # ---------- 图像 ----------
                    plt.figure(figsize=(10,6))
                    plt.plot(x_grid, y_obs, lw=1.0, label="Experiment")
                    plt.plot(x_grid, yfit, lw=1.0, label="Stage2 Fit")
                    plt.xlabel("2θ (deg)")
                    plt.ylabel("Normalized Intensity")
                    plt.title("Stage 2  Refinement Result")

                    text_lines = []
                    text_lines.append("Composition:")
                    for f, w in zip(phase_structs.keys(), fr):
                        text_lines.append(f"  {os.path.basename(f)}: {w*100:.2f}%")
                    text_lines.append("")
                    text_lines.append(f"Rwp = {r_best:.2f}%")
                    text_str = "\n".join(text_lines)

                    plt.text(1.02, 0.98, text_str, transform=plt.gca().transAxes,
                            fontsize=10, va="top", ha="left",
                            bbox=dict(facecolor="white", alpha=0.8, edgecolor="gray"))
                    plt.legend()
                    plt.tight_layout(rect=[0, 0, 0.8, 1])
                    plt.savefig(f"stage2_output/{out_prefix}.png", dpi=300)
                    plt.close()
                    print(f"🖼️ 已保存微调阶段图像：stage2_output/{out_prefix}.png")

                    # ---------- XY 拟合曲线 ----------
                    xy_out = np.column_stack([x_grid, y_obs, yfit])
                    np.savetxt(f"stage2_output/{out_prefix}.xy", xy_out, fmt="%.6f",
                            header="2Theta  Intensity_Obs  Intensity_Fit")
                    print(f"💾 已保存微调阶段谱线：stage2_output/{out_prefix}.xy")

                    # ---------- 文本报告 ----------
                    with open(f"stage2_output/{out_prefix}.txt", "w", encoding="utf-8") as fw:
                        fw.write("=== Stage 2 Refinement Result ===\n")
                        fw.write(f"Rwp : {r_best:.3f}%\n")
                        fw.write("TCH params: " + ", ".join([f"{k}={v:.6g}" for k,v in tpars.items()]) + "\n")
                        fw.write("\nPhases:\n")
                        for f, w, s in zip(phase_structs.keys(), fr, sf):
                            fw.write(f"  {os.path.basename(f):<28s} frac={w*100:6.2f}% | scale={float(s):.4f}\n")
                    print(f"🧾 已保存微调阶段文本报告：stage2_output/{out_prefix}.txt")

                    # ---------- 导出 CIF ----------
                    cif_dir = "stage2_output/cifs"
                    os.makedirs(cif_dir, exist_ok=True)
                    for fpath, struct in phase_structs.items():
                        base = os.path.basename(fpath)
                        name, _ = os.path.splitext(base)
                        out_path = os.path.join(cif_dir, f"{name}_Stage2.cif")
                        struct.to(filename=out_path)
                    print(f"💾 已导出微调阶段 CIF 文件至 {cif_dir}/")



                # ✅ 阶段保温存档
                stage_state = {
                    "phase_structs": phase_structs,
                    "tpars": tpars,
                    "po_r": po_r,
                    "yfit": yfit,
                    "fr": fr,
                    "sf": sf
                }
                print(f"🧊 阶段保温：{stg['name']} 结果已保存，将传入下一阶段。")

    
            # ==========================================================
            # 生成 refine 日志曲线
            # ==========================================================
            import csv
            import numpy as np

            # ----- 导出 CSV -----
            with open("refine_log.csv", "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["index","Rwp","step","frac","scale"])
                for i,(r,st,frv,scv) in enumerate(zip(RWP_LOG,STEP_LOG,FRAC_LOG,SCALE_LOG)):
                    writer.writerow([i,r,st,frv,scv])
            print("📁 已生成 refine_log.csv")

            # ----- Rwp 曲线 -----
            plt.figure(figsize=(10,5))
            plt.plot(RWP_LOG, marker="o")
            plt.xlabel("step index")
            plt.ylabel("Rwp (%)")
            plt.title("Rwp Evolution")
            plt.grid(True)
            plt.savefig("Rwp_curve.png", dpi=200)
            print("📈 已保存 Rwp_curve.png")

            # ----- 相分数曲线 -----
            FR = np.array(FRAC_LOG)
            plt.figure(figsize=(10,5))
            for i in range(FR.shape[1]):
                plt.plot(FR[:,i], label=f"phase {i}")
            plt.legend()
            plt.grid(True)
            plt.title("Phase Fraction Evolution")
            plt.savefig("phase_fraction_curve.png", dpi=200)
            print("📈 已保存 phase_fraction_curve.png")

            # ----- scale 曲线 -----
            SC = np.array(SCALE_LOG)
            plt.figure(figsize=(10,5))
            for i in range(SC.shape[1]):
                plt.plot(SC[:,i], label=f"scale {i}")
            plt.legend()
            plt.grid(True)
            plt.title("Scale Evolution")
            plt.savefig("scale_curve.png", dpi=200)
            print("📈 已保存 scale_curve.png")

            # ✅ 最终结果直接使用最后一次循环得到的结果（不再重新计算 Rwp）
            Rwp_final = r_best
            yfit_final = yfit
            fr_final = fr
            sf_final = sf

            return phase_structs, tpars, None, yfit_final, fr_final, sf_final, Rwp_final, po_r, po_axes
    finally:
        _profile_pool.shutdown(wait=True, cancel_futures=True)
# -----------------------------
# 主流程
# -----------------------------
def main(
    xy_file=None,
    main_cif="Li6PS5Cl.cif",
    imp_dir="impure_phase",
    wavelength=1.5406,
    tch_init=(0.003, 0.001, 0.020, 0.020, 0.010),
    broad_base=0.08,
    bg_degree=5,
    max_candidates=6,
    max_phases_in_mix=4,
    # 三阶段循环次数
    stage_loops=(60, 100, 150),
    single_phase=False,
    # atoms refine params
    init_pos_step=0.005, min_pos_step=0.002,
    # mixed occupancy
    init_mix_step=0.08,  min_mix_step=0.004,
    # angle/cell/TCH
    init_angle_step_deg=0.3, min_angle_step_deg=0.05,
    init_cell_step_frac=0.0025, min_cell_step_frac=0.0002,
    init_tch_step_frac=0.20,   min_tch_step_frac=0.05,
    # Preferred Orientation
    enable_po=True,                 #  择优取向
    init_po_step_frac=0.25, min_po_step_frac=0.05,
    po_r_bounds=(0.3, 3.0),
    po_axes_user: Optional[Dict[str, Tuple[int,int,int]]] = None,  # 如 {"LTP.cif": (0,0,1)}
    po_r_init_user: Optional[Dict[str, float]] = None,
    # stoichiometry 
    lambda_stoich=0.5,
    stoich_phase="Li6PS5Cl.cif",
    stoich_target: Optional[Dict[str,float]] = None,
    # 并行进程数
    num_workers: int = os.cpu_count(),
    #组合过程中的主相偏置
    main_bias: float=1.0,
):
    # 0) 读实验谱
    if xy_file is None:
        xy_files = [f for f in os.listdir(".") if f.lower().endswith(".xy")]
        if not xy_files: raise FileNotFoundError("未发现 .xy 文件")
        xy_file = sorted(xy_files)[0]
    print(f"📄 实验数据：{xy_file}")
    x, y = read_xy(xy_file)

    # 1) 主相
    if not os.path.exists(main_cif): raise FileNotFoundError(f"主相 {main_cif} 不存在")
    st_main = Structure.from_file(main_cif)
    # === 强制补齐 Uiso 字段 ===
    for site in st_main.sites:
        U = site.properties.get("Uiso", 0.01)
        site.properties["_Uiso"] = float(U)

    # 2) 候选杂质
    imp_files = []
    if (not single_phase) and imp_dir and isinstance(imp_dir, str) and os.path.isdir(imp_dir):
        imp_files = [os.path.join(imp_dir, f) for f in os.listdir(imp_dir) if f.lower().endswith(".cif")]
        imp_files.sort()
    print(f"🔍 候选杂质相：{len(imp_files)} 个")

    # 3) 相关性预筛
    U0,V0,W0,X0,Y0 = tch_init
    cands=[]
    for f in imp_files:
        st = Structure.from_file(f)
        # === 强制补齐 Uiso 字段 ===
        for site in st.sites:
            U = site.properties.get("Uiso", 0.01)
            site.properties["_Uiso"] = float(U)

        # 初筛无需PO以避免提前偏置
        prof = synth_profile_po(x, st, wl=wavelength, U=U0,V=V0,W=W0,X=X0,Y=Y0, broad_base=broad_base, enable_po=False)
        c = np.corrcoef(y, prof)[0,1]
        cands.append((f, prof, c))
    cands.sort(key=lambda z:z[2], reverse=True)
    top = cands[:max_candidates]
    if top:
        print("\n🏷️ Top 候选：")
        for f,_,c in top:
            print(f"   {os.path.basename(f):<28s} corr={c:.3f}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🖥️ 设备：{device}")

    # 4) 组合搜索（初选）
    best = {"files":[main_cif], "yfit": None, "rwp": 1e9, "fr": None, "sf": None}
    pools = [[]] if single_phase else [[]] + [list(c) for k in range(1, max_phases_in_mix)
                                                for c in itertools.combinations([f for f,_,_ in top], k)]
    for combo in pools:
        files = [main_cif] + list(combo)
        structs = [Structure.from_file(f) for f in files]
        profiles = [synth_profile_po(x, st, wl=wavelength, U=U0,V=V0,W=W0,X=X0,Y=Y0, broad_base=broad_base, enable_po=False)
                    for st in structs]
        
        #加入偏置，在相筛选时，提高主相权重，防止主相占比被忽视
        yfit, fr, sf, Rwp0_tmp, = torch_refine(
                    y, profiles, device=device, bg_degree=bg_degree,
                    epochs=200, lbfgs_lr=0.3, lbfgs_max_iter=60,
                    main_bias=main_bias if len(files) > 1 else 0.0,   # 多相组合时启用偏置
                )

        print(f"• 组合 {' + '.join(os.path.basename(ff) for ff in files):<60s} → "
                f"Rwp={Rwp0_tmp:6.2f}% ")
        if Rwp0_tmp < best['rwp']:
            best = {
                "files": files, "yfit": yfit,
                "rwp": Rwp0_tmp, "fr": fr, "sf": sf
            }


    print("\n✅ 初选最佳组合：", " + ".join(os.path.basename(f) for f in best["files"]))
    print(f"   初选 Rwp = {best['rwp']:.2f}% ")

    # 5) 外层联合（含混占/化学计量/PO）——三阶段
    phase_structs = {f: Structure.from_file(f) for f in best["files"]}
    tch_dict = {"U":U0, "V":V0, "W":W0, "X":X0, "Y":Y0}

    # PO 初值
    po_axes = {k: (0,0,1) for k in phase_structs.keys()}
    if po_axes_user:
        for k,v in po_axes_user.items():
            if k in po_axes: po_axes[k] = tuple(v)
    po_r_init = {k: 1.0 for k in phase_structs.keys()}
    if po_r_init_user:
        for k,v in po_r_init_user.items():
            if k in po_r_init: po_r_init[k] = float(v)

    # 若未指定目标计量，默认以主相初始 CIF 的配比为目标
    if stoich_target is None:
        comp = phase_structs.get(stoich_phase, st_main).composition.get_el_amt_dict()
        stoich_target = {k: float(v) for k, v in comp.items() if v > 1e-6}
        # ✅ 统一主相识别为文件名，防止路径不同导致匹配错误
    stoich_phase = os.path.basename(main_cif)
    refined_structs, tch_final, profiles, yfit_final, fr_final, sf_final, \
        rwp_final, po_r_final, po_axes_final = outer_refine_all(
        x_grid=x, y_obs=y, phase_structs=phase_structs, tch_params=tch_dict,
        wavelength=wavelength, broad_base=broad_base, bg_degree=bg_degree,
        init_cell_step_frac=init_cell_step_frac, min_cell_step_frac=min_cell_step_frac,
        init_angle_step_deg=init_angle_step_deg, min_angle_step_deg=min_angle_step_deg,
        init_tch_step_frac=init_tch_step_frac,   min_tch_step_frac=min_tch_step_frac,
        init_pos_step=init_pos_step, min_pos_step=min_pos_step,
        init_mix_step=init_mix_step, min_mix_step=min_mix_step,
        enable_po=enable_po, po_axes=po_axes, po_r_init=po_r_init,
        init_po_step_frac=init_po_step_frac, min_po_step_frac=min_po_step_frac,
        po_r_bounds=po_r_bounds,
        lambda_stoich=lambda_stoich, stoich_phase_key=stoich_phase, stoich_target=stoich_target,
        stage_loops=stage_loops, device=device,
        num_workers=num_workers,
    )

    # 6) 绘图（最终）
    plt.figure(figsize=(10,6))
    plt.plot(x, y, lw=1.0, label="Experiment")
    plt.plot(x, yfit_final, lw=1.0, label="v13 Final Fit (PO)")
    plt.xlabel("2θ (deg)")
    plt.ylabel("Normalized Intensity")
    plt.title("yfsf  Refinement Result")

    # ✅ 在图右侧添加文字说明（主相、杂相、R因子等）
    text_lines = []
    text_lines.append("Composition:")
    for f, w in zip(refined_structs.keys(), fr_final):
        text_lines.append(f"  {os.path.basename(f)}: {w*100:.2f}%")
    text_lines.append("")
    text_lines.append(f"Rwp = {rwp_final:.2f}%")
    text_str = "\n".join(text_lines)

    plt.text(1.02, 0.98, text_str, transform=plt.gca().transAxes,
                fontsize=10, va="top", ha="left",
                bbox=dict(facecolor="white", alpha=0.8, edgecolor="gray"))

    plt.legend()
    plt.tight_layout(rect=[0, 0, 0.8, 1])  # 留出右侧空间
    plt.savefig("yfsf_Refined.png", dpi=300)
    print("🖼️ 拟合图已保存：yfsf_Refined.png")

    # ✅ 生成拟合曲线的 .xy 文件
    xy_out = np.column_stack([x, y, yfit_final])
    np.savetxt("yfsf_Refined.xy", xy_out, fmt="%.6f", header="2Theta  Intensity_Obs  Intensity_Fit")
    print("💾 已保存拟合曲线数据：yfsf_Refined.xy")

    # 7) 报告 & 导出 CIF
    os.makedirs("yfsf_refined_cifs", exist_ok=True)
    with open("yfsf_Refined.txt", "w", encoding="utf-8") as fw:
        fw.write("=== yfsf_Refined ===\n")
        fw.write(f"XY file   : {xy_file}\n")
        fw.write(f"Final Rwp : {rwp_final:.3f}%\n")
        fw.write("TCH params (final): " + ", ".join([f"{k}={v:.6g}" for k,v in tch_final.items()]) + "\n")
        fw.write("\nPhases (fractions & per-phase scales):\n")
        for f, w, s in zip(refined_structs.keys(), fr_final, sf_final):
            fw.write(f"  {os.path.basename(f):<28s} frac={w*100:6.2f}% | scale={float(s):.4f}\n")
        fw.write("\nPreferred Orientation (March–Dollase):\n")
        fw.write("  bounds: r in [{:.2f}, {:.2f}]\n".format(*po_r_bounds))
        for f in refined_structs.keys():
            axis = po_axes_final.get(f, (0,0,1))
            rfin = po_r_final.get(f, 1.0)
            fw.write(f"  {os.path.basename(f):<28s} axis=[{axis[0]},{axis[1]},{axis[2]}] | r={rfin:.4f}\n")
        fw.write("\nStoichiometry target (phase={}): {}\n".format(stoich_phase, stoich_target))
        fw.write("\nExported CIFs:\n")

    for fpath, struct in refined_structs.items():
        base = os.path.basename(fpath)
        name, _ = os.path.splitext(base)
        out_path = os.path.join("yfsf_refined_cifs", f"{name}_refined.cif")
        struct = sync_uiso(struct)        # ★关键：把 _Uiso 同步到 Uiso/Biso



        def write_cif_with_uiso(struct: Structure, out_path: str):
            """
            终极修复版：
            1. 严格过滤重复标签，确保 Uiso 仅出现在 Occupancy 之后。
            2. 强制 Occupancy 精度为小数点后 4 位，解决换行问题。
            3. 增加索引合法性检查，防止 IndexError。
            """
            # 1. 准备热参数数据
            u_vals = [float(s.properties.get("_Uiso", s.properties.get("Uiso", 0.01))) for s in struct.sites]
            
            # 2. 生成基础文件
            struct.to(filename=out_path)

            with open(out_path, "r", encoding="utf-8") as f:
                lines = f.readlines()

            final_lines = []
            i = 0
            while i < len(lines):
                line = lines[i]
                # 定位到原子坐标循环块
                if line.strip() == "loop_" and i+1 < len(lines) and "_atom_site_" in lines[i+1]:
                    raw_headers = []
                    i += 1
                    # A. 提取原始 Header，同时过滤掉任何已存在的 Uiso/Biso 标签
                    while i < len(lines) and lines[i].strip().startswith("_atom_site_"):
                        tag = lines[i].strip()
                        if tag not in ["_atom_site_U_iso_or_equiv", "_atom_site_B_iso_or_equiv"]:
                            raw_headers.append(tag)
                        i += 1
                    
                    # B. 确定 Occupancy 的插入位置
                    try:
                        occ_idx = raw_headers.index("_atom_site_occupancy")
                    except ValueError:
                        # 如果没找到 occupancy，则放在末尾
                        occ_idx = len(raw_headers) - 1
                    
                    # 重新构建 Header：在 occupancy 后插入 uiso
                    new_headers = list(raw_headers)
                    new_headers.insert(occ_idx + 1, "_atom_site_U_iso_or_equiv")

                    # C. 处理数据行
                    data_rows = []
                    atom_count = 0
                    while i < len(lines) and lines[i].strip() and not lines[i].strip().startswith(("_", "loop_", "data_")):
                        tokens = lines[i].split()
                        
                        # 安全检查：确保当前行的列数至少能覆盖到 occupancy
                        if len(tokens) >= len(raw_headers) and atom_count < len(u_vals):
                            # 1. 整理原始数据（移除可能多出来的旧 uiso 列）
                            # 这一步是为了防止原始文件里已经有 uiso 导致列数对不上
                            current_tokens = tokens[:len(raw_headers)]
                            
                            # 2. 修正 Occupancy 精度（防止数值过长导致换行）
                            try:
                                occ_val = float(current_tokens[occ_idx])
                                current_tokens[occ_idx] = f"{occ_val:.4f}"
                            except (ValueError, IndexError):
                                pass
                            
                            # 3. 在 Occupancy 对应位置插入新的 Uiso 值
                            current_tokens.insert(occ_idx + 1, f"{u_vals[atom_count]:.6f}")
                            
                            # 4. 格式化组装行（固定列宽 16，彻底解决换行对齐问题）
                            formatted_row = "".join([f"{t:<16}" for t in current_tokens]).rstrip() + "\n"
                            data_rows.append(formatted_row)
                            atom_count += 1
                        i += 1
                    
                    # D. 回填整个 Loop 块
                    final_lines.append("loop_\n")
                    for h in new_headers:
                        final_lines.append(f" {h}\n")
                    final_lines.extend(data_rows)
                    
                    # 继续处理后续行
                    if i < len(lines):
                        final_lines.append(lines[i])
                        i += 1
                else:
                    final_lines.append(line)
                    i += 1

            # 3. 最终写入
            with open(out_path, "w", encoding="utf-8") as f:
                f.writelines(final_lines)

        write_cif_with_uiso(struct, out_path) 
        print(f"💾 已成功导出修复后的 CIF: {out_path}")

    print("\n📈 最终指标：Rwp = {:.2f}% ".format(rwp_final))
    print("📊 相分数（softmax）与每相独立 scale：")
    for f, w, s in zip(refined_structs.keys(), fr_final, sf_final):
        print(f"   {os.path.basename(f):<28s} frac={w*100:6.2f}% | scale={float(s):.3f}")
    if enable_po:
        print("📌 择优取向（March–Dollase）参数：")
        for f in refined_structs.keys():
            axis = po_axes_final.get(f, (0,0,1))
            rfin = po_r_final.get(f, 1.0)
            print(f"   {os.path.basename(f):<28s} axis=[{axis[0]},{axis[1]},{axis[2]}] | r={rfin:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--xy", type=str, help="实验谱文件 (.xy)")
    parser.add_argument("--main", type=str, help="主相 CIF 文件")
    parser.add_argument("--imp", type=str, help="杂相目录")
    # --- 新增波长输入框 ---
    parser.add_argument("--wl", type=float, default=1.5406, help="X-ray 波长 (Angstrom)")
    parser.add_argument("--num-workers", type=int, default=os.cpu_count(), help="并行进程数")
    parser.add_argument("--main-bias", type=float, default=1.0,
                    help="主相偏置系数 (仅影响相组合筛选阶段)")
    parser.add_argument("--stoich-phase", type=str, default=None,
                        help="指定化学计量约束参考相 (通常与主相相同)")
    parser.add_argument("--stoich", type=str, default=None,
                        help='目标化学计量比，例如 "Li:6,S:5,P:1,Cl:1"')
    parser.add_argument("--lambda-stoich", type=float, default=0.5,
                        help="化学计量约束强度 λ_stoich (默认 0.5)")
    args = parser.parse_args()
    main(
        xy_file=args.xy, 
        main_cif=args.main, 
        imp_dir=args.imp, 
        main_bias=args.main_bias,
        # --- 将解析到的波长传给 main 函数 ---
        wavelength=args.wl,
        num_workers=args.num_workers,
        stoich_phase=args.stoich_phase,
        stoich_target=None if args.stoich is None else {
            k: float(v) for k, v in (pair.split(":") for pair in args.stoich.split(","))
        },
        lambda_stoich=args.lambda_stoich
        )
