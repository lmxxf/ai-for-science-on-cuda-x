#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate 16 clean, white-background, mobile-friendly single-column SVGs for WeChat Issue 382
from authentic raw experimental outputs in /home/lmxxf/work/ai-for-science-on-cuda-x/results/.
"""
import base64
import io
import json
import math
from pathlib import Path
from xml.sax.saxutils import escape

import cairosvg
import matplotlib
matplotlib.use("Agg")
import matplotlib.cm as cm
from PIL import Image
import numpy as np

CUDAX_ROOT = Path("/home/lmxxf/work/ai-for-science-on-cuda-x")
RES = CUDAX_ROOT / "results"
OUT_DIR = CUDAX_ROOT / "visuals"

OUT_DIR.mkdir(parents=True, exist_ok=True)


FONT = "'WenQuanYi Micro Hei','Noto Sans CJK SC',sans-serif"


def svg_header(w: int, h: int) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}" font-family="{FONT}">',
        f'<rect width="{w}" height="{h}" fill="#ffffff"/>',
    ]


def save_svg(name: str, lines: list[str]) -> None:
    lines.append("</svg>")
    svg_path = OUT_DIR / name
    svg_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"OK: {name} ({svg_path.stat().st_size} bytes)")


# ---------------------------------------------------------
# 1. fig4a & fig4b: cuSpa Grid + Ray-Casting & Benchmark
# ---------------------------------------------------------
def build_fig4():
    r = json.loads((RES / "module3_results.json").read_text())
    amd = json.loads((RES / "module5_amd9070_results.json").read_text())
    d = np.load(RES / "module3_preview.npz")
    polys = d["polys"]
    pts = d["points"]
    genes = d["genes"]

    # fig4a: Geometry mechanism (640 x 455)
    s = svg_header(640, 455)
    s.append('<text x="24" y="34" font-size="19" font-weight="bold" fill="#0f172a">cuSpa 几何原理：二维网格包围盒过滤 ＋ 水平射线奇偶法</text>')
    s.append('<text x="24" y="57" font-size="12.5" fill="#475569">全库仅 2,378 行 C++/CUDA，0 处 Tensor Core；先按网格缩窄候选细胞，再向右发水平射线数交点奇偶</text>')

    ox, oy, sc = 48, 76, 11.2
    gw, gh = 40 * sc, 28 * sc  # 448 x 313.6
    # Center the 4x3 grid: 0..40 in x, 0..30 in y -> sc=11.0 => 440 x 330
    ox, oy, sc = 40, 74, 11.0
    s.append(f'<rect x="{ox}" y="{oy}" width="{40*sc:.1f}" height="{30*sc:.1f}" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1.2"/>')
    for x in range(10, 40, 10):
        s.append(f'<line x1="{ox+x*sc:.1f}" y1="{oy}" x2="{ox+x*sc:.1f}" y2="{oy+30*sc:.1f}" stroke="#cbd5e1" stroke-width="1" stroke-dasharray="4 3"/>')
    for y in range(10, 30, 10):
        s.append(f'<line x1="{ox}" y1="{oy+y*sc:.1f}" x2="{ox+40*sc:.1f}" y2="{oy+y*sc:.1f}" stroke="#cbd5e1" stroke-width="1" stroke-dasharray="4 3"/>')

    shown_polys = []
    for poly in polys:
        if poly[:, 0].mean() <= 40 and poly[:, 1].mean() <= 30:
            shown_polys.append(poly)
            xy = " ".join(f"{ox+x*sc:.2f},{oy+y*sc:.2f}" for x, y in poly)
            s.append(f'<polygon points="{xy}" fill="#e6f6f4" stroke="#0f766e" stroke-width="1.7"/>')

    for xy, g in zip(pts[:115], genes[:115]):
        if xy[0] <= 40 and xy[1] <= 30:
            col = "#ef4444" if g % 2 == 0 else "#3b82f6"
            s.append(f'<circle cx="{ox+xy[0]*sc:.2f}" cy="{oy+xy[1]*sc:.2f}" r="2.4" fill="{col}" fill-opacity="0.78"/>')

    # Compute exact ray intersections with first top-left polygon (around center 5,5)
    p0 = min(shown_polys, key=lambda pl: (pl[:, 0].mean() - 5)**2 + (pl[:, 1].mean() - 5)**2)
    def ray_intersections(px_val, py_val, poly):
        xs = []
        n = len(poly) - 1 if np.allclose(poly[0], poly[-1]) else len(poly)
        for i in range(n):
            x1, y1 = poly[i]
            x2, y2 = poly[(i + 1) % n]
            if (y1 > py_val) != (y2 > py_val):
                xin = x1 + (py_val - y1) * (x2 - x1) / (y2 - y1)
                if xin > px_val:
                    xs.append(xin)
        return sorted(xs)

    # Inside point
    in_x, in_y = float(p0[:, 0].mean()), float(p0[:, 1].mean())
    in_hits = ray_intersections(in_x, in_y, p0)
    in_end = (in_hits[0] + 0.85) if in_hits else 9.2
    s.append(f'<line x1="{ox+in_x*sc:.2f}" y1="{oy+in_y*sc:.2f}" x2="{ox+in_end*sc:.2f}" y2="{oy+in_y*sc:.2f}" stroke="#d97706" stroke-width="2.2" stroke-dasharray="4 3"/>')
    s.append(f'<circle cx="{ox+in_x*sc:.2f}" cy="{oy+in_y*sc:.2f}" r="4.8" fill="#d97706" stroke="#ffffff" stroke-width="1.2"/>')
    for hx in in_hits:
        s.append(f'<circle cx="{ox+hx*sc:.2f}" cy="{oy+in_y*sc:.2f}" r="3.8" fill="#ffffff" stroke="#d97706" stroke-width="2.2"/>')

    # Outside point (to the left of second polygon on row 0, around x=10.6, y=p1_mean_y)
    p1 = min(shown_polys, key=lambda pl: (pl[:, 0].mean() - 15)**2 + (pl[:, 1].mean() - 5)**2)
    out_y = float(p1[:, 1].mean())
    out_x = float(p1[:, 0].min() - 1.3)
    out_hits = ray_intersections(out_x, out_y, p1)
    s.append(f'<line x1="{ox+out_x*sc:.2f}" y1="{oy+out_y*sc:.2f}" x2="{ox+19.8*sc:.2f}" y2="{oy+out_y*sc:.2f}" stroke="#7c3aed" stroke-width="2.2" stroke-dasharray="4 3"/>')
    s.append(f'<circle cx="{ox+out_x*sc:.2f}" cy="{oy+out_y*sc:.2f}" r="4.8" fill="#7c3aed" stroke="#ffffff" stroke-width="1.2"/>')
    for hx in out_hits:
        s.append(f'<circle cx="{ox+hx*sc:.2f}" cy="{oy+out_y*sc:.2f}" r="3.8" fill="#ffffff" stroke="#7c3aed" stroke-width="2.2"/>')

    # Right-side clean legend callout (x=494..624)
    lx = 494
    s.append(f'<circle cx="{lx+8}" cy="112" r="5" fill="#d97706"/>')
    s.append(f'<text x="{lx+20}" y="116" font-size="12.5" font-weight="bold" fill="#92400e">细胞内基因点</text>')
    s.append(f'<text x="{lx+20}" y="135" font-size="11.5" fill="#475569">向右穿边界 1 次</text>')
    s.append(f'<text x="{lx+20}" y="152" font-size="11.5" fill="#475569">（奇数 ＝ 命中）</text>')

    s.append(f'<circle cx="{lx+8}" cy="190" r="5" fill="#7c3aed"/>')
    s.append(f'<text x="{lx+20}" y="194" font-size="12.5" font-weight="bold" fill="#5b21b6">细胞外基因点</text>')
    s.append(f'<text x="{lx+20}" y="213" font-size="11.5" fill="#475569">向右穿边界 2 次</text>')
    s.append(f'<text x="{lx+20}" y="230" font-size="11.5" fill="#475569">（偶数 ＝ 排除）</text>')

    s.append(f'<rect x="{lx+2}" y="264" width="12" height="12" fill="#e6f6f4" stroke="#0f766e" stroke-width="1.5"/>')
    s.append(f'<text x="{lx+20}" y="275" font-size="12" fill="#334155">细胞多边形边界</text>')
    s.append(f'<line x1="{lx+2}" y1="302" x2="{lx+15}" y2="302" stroke="#cbd5e1" stroke-width="1.5" stroke-dasharray="3 2"/>')
    s.append(f'<text x="{lx+20}" y="306" font-size="12" fill="#334155">224×224 索引格</text>')
    s.append(f'<text x="{lx+2}" y="344" font-size="11.5" fill="#475569">第3步：64位整数</text>')
    s.append(f'<text x="{lx+2}" y="362" font-size="11.5" fill="#475569">(细胞ID, 基因ID)</text>')
    s.append(f'<text x="{lx+2}" y="380" font-size="11.5" fill="#475569">基数排序转 CSR</text>')

    s.append('<text x="24" y="434" font-size="11.5" fill="#64748b">数据源：NVIDIA/cuspa (commit 2df9fa78) 源码审计与局部网格切片预览；排版经 Gemini 润色</text>')
    save_svg("fig4a_cuspa_grid_raycasting.svg", s)

    # fig4b: Benchmark (640 x 335)
    aa = amd["amd"]
    nv = amd["gb10_same_harness"]
    cpu_tot = r["cpu_pipeline_median_s"] * 1000.0
    cpu_ray = r.get("cpu_query_median_s", 0.236064) * 1000.0
    nv_tot = nv["pipeline_median_wall_ms"]
    aa_tot = aa["pipeline_median_wall_ms"]
    nv_ray = nv["query_median_wall_ms"]
    aa_ray = aa["query_median_wall_ms"]

    s = svg_header(640, 335)
    s.append('<text x="24" y="34" font-size="19" font-weight="bold" fill="#0f172a">cuSpa 跨架构实测：50,000 细胞 × 2,000,000 点 × 500 基因</text>')
    s.append('<text x="24" y="56" font-size="12.5" fill="#047857" font-weight="bold">200 万点归属与四个 CSR int32 数组 SHA256 逐字节完全一致（0 误差）</text>')

    s.append('<text x="24" y="88" font-size="14.5" font-weight="bold" fill="#0f172a">① 同源 C++ 全流程耗时（网格索引 ＋ 射线查询 ＋ 基数排序转 CSR，越短越快）</text>')
    rows1 = [
        ("CPU 单核编译基线", cpu_tot, "#94a3b8", f"{cpu_tot:.2f} ms", False),
        ("NVIDIA GB10 (CUDA)", nv_tot, "#2563eb", f"{nv_tot:.2f} ms  (快 47.3×)", False),
        ("AMD RX 9070 XT (HIP)", aa_tot, "#dc2626", f"{aa_tot:.2f} ms  (快 45.0×)", True),
    ]
    for i, (lbl, val, col, note, bold) in enumerate(rows1):
        y = 116 + i * 30
        w = max(5.0, val / cpu_tot * 270.0)
        fw = "bold" if bold else "normal"
        fc = "#0f172a" if bold else "#334155"
        s.append(f'<text x="24" y="{y}" font-size="13" font-weight="{fw}" fill="{fc}">{lbl}</text>')
        s.append(f'<rect x="192" y="{y-13}" width="{w:.1f}" height="17" rx="3.5" fill="{col}"/>')
        s.append(f'<text x="{192+w+10:.1f}" y="{y}" font-size="12.5" font-weight="{fw}" fill="{fc}">{note}</text>')

    s.append('<text x="24" y="216" font-size="14.5" font-weight="bold" fill="#0f172a">② 核心算力阶段：200 万点水平射线几何查询耗时（古典流处理器分支＋算术）</text>')
    # Scale GPU bars visibly on secondary zoom or log-friendly bar; let's show GPU zoom bar + CPU reference!
    rows2 = [
        ("CPU 单核编译射线查询", cpu_ray, 260.0, "#94a3b8", f"{cpu_ray:.2f} ms", False),
        ("NVIDIA GB10 (CUDA)", nv_ray, nv_ray / 2.0 * 140.0, "#2563eb", f"{nv_ray:.3f} ms  (快 155.3×)", False),
        ("AMD RX 9070 XT (HIP)", aa_ray, aa_ray / 2.0 * 140.0, "#dc2626", f"{aa_ray:.3f} ms  (快 256.3×，反超 GB10 达 39%)", True),
    ]
    for i, (lbl, _, bar_w, col, note, bold) in enumerate(rows2):
        y = 244 + i * 30
        fw = "bold" if bold else "normal"
        fc = "#0f172a" if bold else "#334155"
        s.append(f'<text x="24" y="{y}" font-size="13" font-weight="{fw}" fill="{fc}">{lbl}</text>')
        if i == 0:
            s.append(f'<rect x="192" y="{y-13}" width="250" height="17" rx="3.5" fill="{col}"/>')
            s.append(f'<line x1="432" y1="{y-15}" x2="426" y2="{y+6}" stroke="#ffffff" stroke-width="2.2"/>')
            w_end = 250.0
        else:
            s.append(f'<rect x="192" y="{y-13}" width="{bar_w:.1f}" height="17" rx="3.5" fill="{col}"/>')
            w_end = bar_w
        s.append(f'<text x="{192+w_end+10:.1f}" y="{y}" font-size="12.5" font-weight="{fw}" fill="{fc}">{note}</text>')

    s.append('<text x="24" y="321" font-size="11.5" fill="#64748b">数据源：DGX Spark (GB10) 与 AMD RX 9070 XT 同源二进制输入稳态中位数；排版经 Gemini 润色</text>')
    save_svg("fig4b_cuspa_benchmark.svg", s)


# ---------------------------------------------------------
# 2. fig11a & fig11b: GPU-HMMER 100k Protein Viterbi
# ---------------------------------------------------------
def build_fig11():
    r = json.loads((RES / "module12_results.json").read_text())
    z = np.load(RES / "module12_all_scores.npz")
    score = z["hip"]
    labels = z["labels"]

    # fig11a: Score bimodal distribution (640 x 345)
    s = svg_header(640, 345)
    s.append('<text x="24" y="34" font-size="19" font-weight="bold" fill="#0f172a">10 万条蛋白序列 Profile-HMM 维特比扫描：得分双峰分离</text>')
    s.append('<text x="24" y="56" font-size="12.5" fill="#475569">56 位点真实 GB1 的 M/I/D 三状态轮廓扫描 1,920 万残基，500 条同源域检出 ROC-AUC = 1.000000</text>')

    x0, y0, pw, ph = 64, 76, 544, 198
    s.append(f'<rect x="{x0}" y="{y0}" width="{pw}" height="{ph}" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1.2"/>')
    lo, hi = -130.0, 240.0
    for tick in [-100, -50, 0, 50, 100, 150, 200]:
        tx = x0 + (tick - lo) / (hi - lo) * pw
        s.append(f'<line x1="{tx:.1f}" y1="{y0}" x2="{tx:.1f}" y2="{y0+ph}" stroke="#e2e8f0" stroke-width="1"/>')
        s.append(f'<text x="{tx:.1f}" y="{y0+ph+18}" font-size="11.5" fill="#475569" text-anchor="middle">{tick}</text>')
    for frac, ylbl in [(0.0, "0.0"), (0.5, "0.5"), (1.0, "1.0")]:
        ty = y0 + ph - frac * (ph - 18)
        s.append(f'<line x1="{x0}" y1="{ty:.1f}" x2="{x0+pw}" y2="{ty:.1f}" stroke="#e2e8f0" stroke-width="1"/>')
        s.append(f'<text x="{x0-8}" y="{ty+4:.1f}" font-size="11" fill="#475569" text-anchor="end">{ylbl}</text>')

    bins = np.linspace(lo, hi, 75)
    hn, _ = np.histogram(score[labels == 0], bins=bins)
    hp, _ = np.histogram(score[labels == 1], bins=bins)
    bw = pw / (len(bins) - 1)
    for hist, col in [(hn, "#2563eb"), (hp, "#dc2626")]:
        mx = max(int(hist.max()), 1)
        for j, v in enumerate(hist):
            if v == 0:
                continue
            bh = (ph - 18) * (v / mx)
            bx = x0 + j * bw
            by = y0 + ph - bh
            s.append(f'<rect x="{bx+0.4:.2f}" y="{by:.2f}" width="{max(bw-0.8, 1.0):.2f}" height="{bh:.2f}" fill="{col}" fill-opacity="0.82"/>')

    # Legend box inside empty middle gap
    s.append('<rect x="198" y="88" width="248" height="58" rx="6" fill="#ffffff" stroke="#cbd5e1" stroke-width="1"/>')
    s.append('<rect x="210" y="101" width="14" height="11" rx="2" fill="#2563eb"/>')
    s.append('<text x="230" y="111" font-size="12" font-weight="bold" fill="#1e40af">99,500 条随机背景序列（左峰）</text>')
    s.append('<rect x="210" y="125" width="14" height="11" rx="2" fill="#dc2626"/>')
    s.append('<text x="230" y="135" font-size="12" font-weight="bold" fill="#991b1b">500 条植入 GB1 同源域序列（右簇）</text>')

    s.append(f'<text x="{x0+pw/2:.1f}" y="311" font-size="12.5" fill="#334155" text-anchor="middle">横轴：维特比对数几率得分（bit-score） │ 纵轴：各类别峰值归一化频数</text>')
    s.append('<text x="24" y="333" font-size="11.5" fill="#64748b">数据源：真实 GB1 发射矩阵扫描 100,000 条蛋白序列（module12）；排版经 Gemini 润色</text>')
    save_svg("fig11a_gpuhmmer_score_dist.svg", s)

    # fig11b: Timing & throughput benchmark (640 x 265)
    cpu_t = r["backends"]["cpu"]
    nv_t = r["backends"]["cuda"]
    amd_t = r["backends"]["hip"]
    s = svg_header(640, 265)
    s.append('<text x="24" y="34" font-size="19" font-weight="bold" fill="#0f172a">10 万条蛋白序列动态规划耗时：AMD RX 9070 XT 快 GB10 3.86 倍</text>')
    s.append('<text x="24" y="57" font-size="12.5" fill="#047857" font-weight="bold">纯标量 add / max 滚动数组无矩阵乘法；100,000 个得分三端 SHA256 逐字节一致</text>')

    max_ms = cpu_t["median_ms"]
    rows = [
        ("CPU 单核编译 C++", cpu_t["median_ms"], cpu_t["residues_per_s"] / 1e6, "#94a3b8", "", False),
        ("NVIDIA GB10 (CUDA)", nv_t["median_ms"], nv_t["residues_per_s"] / 1e6, "#2563eb", "快 CPU 4.8×", False),
        ("AMD RX 9070 XT (HIP)", amd_t["median_ms"], amd_t["residues_per_s"] / 1e6, "#dc2626", "快 GB10 3.86× / 快 CPU 18.7×", True),
    ]
    for i, (lbl, ms, mres, col, tag, bold) in enumerate(rows):
        y = 98 + i * 48
        w = max(8.0, ms / max_ms * 195.0)
        fw = "bold" if bold else "normal"
        fc = "#0f172a" if bold else "#334155"
        s.append(f'<text x="24" y="{y}" font-size="13.5" font-weight="{fw}" fill="{fc}">{lbl}</text>')
        s.append(f'<rect x="192" y="{y-14}" width="{w:.1f}" height="20" rx="4" fill="{col}"/>')
        extra = f" ({tag})" if tag else ""
        s.append(f'<text x="{192+w+10:.1f}" y="{y+1}" font-size="12.2" font-weight="{fw}" fill="{fc}">{ms:.2f} ms │ {mres:.2f} 百万残基/秒{extra}</text>')

    s.append('<text x="24" y="249" font-size="11.5" fill="#64748b">数据源：DGX Spark (GB10) 与 AMD RX 9070 XT 扫描 1,920 万氨基酸残基稳态中位数；排版经 Gemini 润色</text>')
    save_svg("fig11b_gpuhmmer_benchmark.svg", s)


# ---------------------------------------------------------
# 3. fig6a & fig6b: Open-cuPotential Nonbonded Forcefield
# ---------------------------------------------------------
def build_fig6():
    r = json.loads((RES / "module7_results.json").read_text())

    # fig6a: 27-cell neighbor list + LJ & Reaction-Field curves (640 x 355)
    s = svg_header(640, 355)
    s.append('<text x="24" y="34" font-size="19" font-weight="bold" fill="#0f172a">三维分子非键力场原理：27 邻格空间分桶 ＋ 范德华与库仑势</text>')
    s.append('<text x="24" y="56" font-size="12.5" fill="#475569">截断半径 r_cut = 10 埃：每个原子只查自身与周围 26 个邻格，融合计算势能与解析负梯度力 F = -∇E</text>')

    # Left panel: 3 slices of 3x3 cells (x=24..234, y=78..304)
    s.append('<text x="28" y="84" font-size="13" font-weight="bold" fill="#0f172a">① 3×3×3 = 27 邻格分桶</text>')
    for z in range(3):
        ox = 28 + z * 42
        oy = 100 + z * 46
        for yy in range(3):
            for xx in range(3):
                cx = ox + xx * 36
                cy = oy + yy * 28
                is_center = (z == 1 and xx == 1 and yy == 1)
                fill = "#fde68a" if is_center else ["#eff6ff", "#ecfdf5", "#f1f5f9"][z]
                stroke = "#d97706" if is_center else "#94a3b8"
                sw = "1.8" if is_center else "1.0"
                s.append(f'<rect x="{cx}" y="{cy}" width="35" height="27" fill="{fill}" fill-opacity="0.92" stroke="{stroke}" stroke-width="{sw}"/>')
                if (xx + yy + z) % 2 == 0 or is_center:
                    pcol = "#dc2626" if is_center else "#2563eb"
                    pr = "3.6" if is_center else "2.6"
                    s.append(f'<circle cx="{cx+17}" cy="{cy+13}" r="{pr}" fill="{pcol}"/>')
    s.append('<text x="28" y="294" font-size="11.5" fill="#475569">黄格中心红点仅与这 27 格</text>')
    s.append('<text x="28" y="311" font-size="11.5" fill="#475569">内的蓝点算成对作用力</text>')

    # Right panel: Potential curve (x=276..614, y=94..284)
    s.append('<text x="276" y="84" font-size="13" font-weight="bold" fill="#0f172a">② 范德华 LJ 12-6 与反应场库仑势曲线</text>')
    x0, y0, pw, ph = 286, 98, 326, 184
    s.append(f'<rect x="{x0}" y="{y0}" width="{pw}" height="{ph}" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1.2"/>')
    def px(rv):
        return x0 + (rv - 2.7) / (10.0 - 2.7) * pw
    def py(ev):
        return y0 + (1.5 - ev) / 2.5 * ph

    for ev in [-1.0, -0.5, 0.0, 0.5, 1.0]:
        yy = py(ev)
        dash = "" if abs(ev) < 1e-9 else 'stroke-dasharray="3 3"'
        col = "#94a3b8" if abs(ev) < 1e-9 else "#e2e8f0"
        s.append(f'<line x1="{x0}" y1="{yy:.1f}" x2="{x0+pw}" y2="{yy:.1f}" stroke="{col}" stroke-width="1" {dash}/>')
        if ev in (-1.0, 0.0, 1.0):
            s.append(f'<text x="{x0-6}" y="{yy+4:.1f}" font-size="11" fill="#475569" text-anchor="end">{ev:.0f}</text>')
    for rv in [3, 5, 7, 10]:
        xx = px(rv)
        s.append(f'<line x1="{xx:.1f}" y1="{y0}" x2="{xx:.1f}" y2="{y0+ph}" stroke="#e2e8f0" stroke-width="1"/>')
        s.append(f'<text x="{xx:.1f}" y="{y0+ph+16}" font-size="11" fill="#475569" text-anchor="middle">{rv} Å</text>')

    for mode, col in [("lj", "#ea580c"), ("rf", "#2563eb")]:
        pts = []
        for i in range(220):
            rv = 2.72 + (10.0 - 2.72) * i / 219.0
            if mode == "lj":
                v = 4.0 * 0.2 * ((3.0 / rv) ** 12 - (3.0 / rv) ** 6)
            else:
                v = 1389.354576 * (-0.05 * 0.05) * (1.0 / rv + (78.3 - 1.0) / (2 * 78.3 + 1.0) / 1000.0 * rv * rv - 3 * 78.3 / (2 * 78.3 + 1.0) / 10.0)
            v = max(min(v, 1.48), -0.98)
            pts.append(f"{px(rv):.2f},{py(v):.2f}")
        s.append(f'<polyline points="{" ".join(pts)}" fill="none" stroke="{col}" stroke-width="2.4"/>')

    s.append('<rect x="404" y="108" width="198" height="48" rx="5" fill="#ffffff" stroke="#cbd5e1" stroke-width="1"/>')
    s.append('<line x1="414" y1="123" x2="436" y2="123" stroke="#ea580c" stroke-width="2.5"/>')
    s.append('<text x="442" y="127" font-size="11.5" font-weight="bold" fill="#9a3412">LJ 12-6 势（近斥远吸）</text>')
    s.append('<line x1="414" y1="143" x2="436" y2="143" stroke="#2563eb" stroke-width="2.5"/>')
    s.append('<text x="442" y="147" font-size="11.5" font-weight="bold" fill="#1e40af">异号电荷反应场库仑势</text>')
    s.append(f'<text x="{x0+pw/2:.1f}" y="314" font-size="11.5" fill="#475569" text-anchor="middle">原子间距 r（埃） │ 纵轴：成对势能（kJ/mol）</text>')

    s.append('<text x="24" y="341" font-size="11.5" fill="#64748b">数据源：open-cupotential 同源 C++/CUDA/HIP 参考核（module7）；排版经 Gemini 润色</text>')
    save_svg("fig6a_nonbonded_potential.svg", s)

    # fig6b: 32K & 128K benchmark (640 x 330)
    s = svg_header(640, 330)
    s.append('<text x="24" y="34" font-size="19" font-weight="bold" fill="#0f172a">三维分子非键力场真机对决：12.8 万原子 RX 9070 XT 反超 17.5%</text>')
    s.append('<text x="24" y="56" font-size="12.5" fill="#047857" font-weight="bold">双精度 float64：全部 128,000 个能量与 384,000 个受力分量三端逐值一致（相对 L2 差 = 0）</text>')

    for idx, sys_entry in enumerate(r["systems"]):
        n_atoms = sys_entry["n"]
        y_base = 88 + idx * 116
        cpu_ms = sys_entry["backends"]["cpu"]["pipeline_median_ms"]
        nv_ms = sys_entry["backends"]["cuda"]["pipeline_median_ms"]
        amd_ms = sys_entry["backends"]["hip"]["pipeline_median_ms"]
        title = f"{'①' if idx == 0 else '②'} {n_atoms:,} 原子体系（{n_atoms*3:,} 个三维受力分量，含分桶＋能量＋解析力）"
        s.append(f'<text x="24" y="{y_base}" font-size="14" font-weight="bold" fill="#0f172a">{title}</text>')
        items = [
            ("CPU 单核编译 C++", cpu_ms, "#94a3b8", f"{cpu_ms:.2f} ms", False),
            ("NVIDIA GB10 (CUDA)", nv_ms, "#2563eb", f"{nv_ms:.2f} ms  (快 {cpu_ms/nv_ms:.1f}×)", False),
            ("AMD RX 9070 XT (HIP)", amd_ms, "#dc2626",
             f"{amd_ms:.2f} ms  (快 {cpu_ms/amd_ms:.1f}×" + ("，反超 GB10 17.5%)" if idx == 1 else "，追平 GB10)"), True),
        ]
        for j, (lbl, ms, col, note, bold) in enumerate(items):
            yy = y_base + 26 + j * 27
            w = max(8.0, ms / cpu_ms * 255.0)
            fw = "bold" if bold else "normal"
            fc = "#0f172a" if bold else "#334155"
            s.append(f'<text x="24" y="{yy}" font-size="12.8" font-weight="{fw}" fill="{fc}">{lbl}</text>')
            s.append(f'<rect x="192" y="{yy-13}" width="{w:.1f}" height="17" rx="3.5" fill="{col}"/>')
            s.append(f'<text x="{192+w+10:.1f}" y="{yy}" font-size="12.5" font-weight="{fw}" fill="{fc}">{note}</text>')

    s.append('<text x="24" y="316" font-size="11.5" fill="#64748b">数据源：DGX Spark (GB10) 与 AMD RX 9070 XT 同源二进制实测（module7）；排版经 Gemini 润色</text>')
    save_svg("fig6b_nonbonded_benchmark.svg", s)


# ---------------------------------------------------------
# 4. fig8a & fig8b: PME 3D-FFT + 1,000-Step MD
# ---------------------------------------------------------
def build_fig8():
    r = json.loads((RES / "module9_results.json").read_text())
    h = np.fromfile(RES / "module9_cuda/nve_history.bin", dtype=np.float64).reshape(-1, 7)
    vals = h[:, 1:4] - h[0, 1:4]  # PE, KE, Total relative to t=0

    # fig8a: NVE energy conservation curve (640 x 365)
    s = svg_header(640, 365)
    s.append('<text x="24" y="34" font-size="18.5" font-weight="bold" fill="#0f172a">PME 3D-FFT 连续 1,000 步：动能/势能镜像交换与能量守恒</text>')
    s.append('<text x="24" y="56" font-size="12.2" fill="#475569">每步重算 4 阶 B 样条与 64³ FFT 长程静电；总机械能最大相对漂移仅 1.09e-4，步长减半改善 4.0004×</text>')

    x0, y0, pw, ph = 64, 76, 528, 212
    s.append(f'<rect x="{x0}" y="{y0}" width="{pw}" height="{ph}" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1.2"/>')
    lim = 95.0
    for ev in [-80, -40, 0, 40, 80]:
        yy = y0 + ph / 2.0 - (ev / lim) * (ph / 2.0)
        col = "#94a3b8" if ev == 0 else "#e2e8f0"
        s.append(f'<line x1="{x0}" y1="{yy:.1f}" x2="{x0+pw}" y2="{yy:.1f}" stroke="{col}" stroke-width="1"/>')
        lbl = f"{ev:+d}" if ev != 0 else "0"
        s.append(f'<text x="{x0-8}" y="{yy+4:.1f}" font-size="11" fill="#475569" text-anchor="end">{lbl}</text>')
    for step_t, ps_lbl in [(0, "0.0 ps (0步)"), (250, "0.125 ps"), (500, "0.25 ps (500步)"), (750, "0.375 ps"), (1000, "0.50 ps (1k步)")]:
        xx = x0 + step_t / 1000.0 * pw
        s.append(f'<line x1="{xx:.1f}" y1="{y0}" x2="{xx:.1f}" y2="{y0+ph}" stroke="#e2e8f0" stroke-width="1"/>')
        s.append(f'<text x="{xx:.1f}" y="{y0+ph+17}" font-size="11" fill="#475569" text-anchor="middle">{ps_lbl}</text>')

    for j, (col, sw) in enumerate([("#ea580c", "1.8"), ("#2563eb", "1.8"), ("#059669", "3.0")]):
        coords = " ".join(
            f"{x0 + i / (len(h) - 1) * pw:.2f},{y0 + ph / 2.0 - vals[i, j] / lim * (ph / 2.0):.2f}"
            for i in range(0, len(h), 2)
        )
        s.append(f'<polyline points="{coords}" fill="none" stroke="{col}" stroke-width="{sw}"/>')

    # Legend bar below x-axis
    s.append('<line x1="92" y1="325" x2="118" y2="325" stroke="#ea580c" stroke-width="2.4"/>')
    s.append('<text x="124" y="329" font-size="12" font-weight="bold" fill="#9a3412">势能变化 ΔE_pot (kJ/mol)</text>')
    s.append('<line x1="286" y1="325" x2="312" y2="325" stroke="#2563eb" stroke-width="2.4"/>')
    s.append('<text x="318" y="329" font-size="12" font-weight="bold" fill="#1e40af">动能变化 ΔE_kin</text>')
    s.append('<line x1="432" y1="325" x2="458" y2="325" stroke="#059669" stroke-width="3.2"/>')
    s.append('<text x="464" y="329" font-size="12" font-weight="bold" fill="#065f46">总机械能 ΔE_tot（守恒直线）</text>')

    s.append('<text x="24" y="352" font-size="11.5" fill="#64748b">数据源：Velocity-Verlet NVE 1,000 步驻留轨迹实测（module9）；图表排版经 Gemini 润色</text>')
    save_svg("fig8a_pme_nve_conservation.svg", s)

    # fig8b: NVE & NVT PME MD benchmark (640 x 320)
    s = svg_header(640, 320)
    s.append('<text x="24" y="34" font-size="19" font-weight="bold" fill="#0f172a">1,000 步含 PME 3D-FFT 分子动力学耗时：hipFFT 双杀反超 cuFFT</text>')
    s.append('<text x="24" y="56" font-size="12.3" fill="#047857" font-weight="bold">CUDA 与 HIP 初始全受力相对 L2 差异仅 4.66e-16；NVT 300K 热浴稳态均温三端均为 297.35 K</text>')

    modes = [
        ("nve", "① NVE 微正则自由演化（1,000 步，每步完整 64³ PME 3D-FFT）", "25.0%"),
        ("nvt", "② NVT 300K BAOAB 朗之万热浴（1,000 步，每步完整 64³ PME 3D-FFT）", "38.3%"),
    ]
    for idx, (mkey, mtitle, beat_pct) in enumerate(modes):
        yb = 86 + idx * 112
        cpu_s = r["backends"]["cpu"]["trajectories"][mkey]["seconds"]
        nv_s = r["backends"]["cuda"]["trajectories"][mkey]["seconds"]
        amd_s = r["backends"]["hip"]["trajectories"][mkey]["seconds"]
        s.append(f'<text x="24" y="{yb}" font-size="14" font-weight="bold" fill="#0f172a">{mtitle}</text>')
        for j, (lbl, sec, col, note, bold) in enumerate([
            ("CPU (NumPy/SciPy FFT)", cpu_s, "#94a3b8", f"{sec:.3f} s" if False else f"{cpu_s:.3f} s", False),
            ("NVIDIA GB10 (cuFFT)", nv_s, "#2563eb", f"{nv_s:.3f} s  (快 CPU {cpu_s/nv_s:.2f}×)", False),
            ("AMD RX 9070 XT (hipFFT)", amd_s, "#dc2626", f"{amd_s:.3f} s  (反超 GB10 达 {beat_pct})", True),
        ]):
            yy = yb + 25 + j * 26
            w = max(8.0, sec / cpu_s * 255.0)
            fw = "bold" if bold else "normal"
            fc = "#0f172a" if bold else "#334155"
            s.append(f'<text x="24" y="{yy}" font-size="12.8" font-weight="{fw}" fill="{fc}">{lbl}</text>')
            s.append(f'<rect x="196" y="{yy-13}" width="{w:.1f}" height="17" rx="3.5" fill="{col}"/>')
            s.append(f'<text x="{196+w+10:.1f}" y="{yy}" font-size="12.5" font-weight="{fw}" fill="{fc}">{note}</text>')

    s.append('<text x="24" y="306" font-size="11.5" fill="#64748b">数据源：DGX Spark (GB10) 与 AMD RX 9070 XT 设备驻留 1,000 步实测（module9）；排版经 Gemini 润色</text>')
    save_svg("fig8b_pme_md_benchmark.svg", s)


# ---------------------------------------------------------
# 5. fig13a & fig13b: 10,000 Conformers Batched Relaxation
# ---------------------------------------------------------
def build_fig13():
    r = json.loads((RES / "module14_results.json").read_text())
    h = np.fromfile(RES / "module14_cuda/history.bin", dtype=np.float64).reshape(10000, 201)[480]  # Nifedipine #1

    # fig13a: Relaxation curve + collision zeroing summary (640 x 345)
    s = svg_header(640, 345)
    s.append('<text x="24" y="34" font-size="19" font-weight="bold" fill="#0f172a">10,000 个三维分子构象 × 200 步共享内存批量力场松弛轨迹</text>')
    s.append('<text x="24" y="56" font-size="12.5" fill="#475569">每构象 1 个 GPU Block 常驻共享内存无 CPU 往返；修正范德华排斥后空间碰撞从 47 处彻底清零</text>')

    x0, y0, pw, ph = 70, 76, 540, 196
    s.append(f'<rect x="{x0}" y="{y0}" width="{pw}" height="{ph}" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1.2"/>')
    lo, hi = -225.0, -125.0
    for ev in [-220, -200, -180, -160, -140]:
        yy = y0 + ph - (ev - lo) / (hi - lo) * ph
        s.append(f'<line x1="{x0}" y1="{yy:.1f}" x2="{x0+pw}" y2="{yy:.1f}" stroke="#e2e8f0" stroke-width="1"/>')
        s.append(f'<text x="{x0-8}" y="{yy+4:.1f}" font-size="11" fill="#475569" text-anchor="end">{ev}</text>')
    for st in [0, 50, 100, 150, 200]:
        xx = x0 + st / 200.0 * pw
        s.append(f'<line x1="{xx:.1f}" y1="{y0}" x2="{xx:.1f}" y2="{y0+ph}" stroke="#e2e8f0" stroke-width="1"/>')
        s.append(f'<text x="{xx:.1f}" y="{y0+ph+17}" font-size="11" fill="#475569" text-anchor="middle">第 {st} 步</text>')

    pts = " ".join(f"{x0 + i / 200.0 * pw:.2f},{y0 + ph - (v - lo) / (hi - lo) * ph:.2f}" for i, v in enumerate(h))
    s.append(f'<polyline points="{pts}" fill="none" stroke="#0284c7" stroke-width="2.6"/>')

    # Clean data callout inside upper-right whitespace of plot
    c = r["backends"]["cpu"]
    s.append('<rect x="274" y="88" width="322" height="86" rx="6" fill="#ffffff" stroke="#cbd5e1" stroke-width="1"/>')
    s.append('<text x="286" y="109" font-size="12.5" font-weight="bold" fill="#0f172a">ESOL 全体 10,000 构象（133,450 重原子）统计</text>')
    s.append(f'<text x="286" y="129" font-size="12" fill="#334155">• 平均势能：+{c["energy_initial_mean"]:.2f} → {c["energy_final_mean"]:.2f} kJ/mol（9,990 严格下降）</text>')
    s.append(f'<text x="286" y="148" font-size="12" fill="#334155">• 平均键长偏差：{c["bond_abs_deviation_initial_mean_A"]:.4f} → {c["bond_abs_deviation_final_mean_A"]:.4f} 埃</text>')
    s.append(f'<text x="286" y="166" font-size="12" font-weight="bold" fill="#047857">• 非键近距空间碰撞：{c["nonbonded_collision_pairs_initial"]} 处 → {c["nonbonded_collision_pairs_final"]} 处（清零）</text>')

    s.append(f'<text x="{x0+pw/2:.1f}" y="311" font-size="12" fill="#334155" text-anchor="middle">蓝线：降压药硝苯地平（第 1 构象）200 步回溯下降能量曲线（单位 kJ/mol）</text>')
    s.append('<text x="24" y="333" font-size="11.5" fill="#64748b">数据源：1,000 真实分子 × 10 构象共享内存松弛轨迹（module14）；排版经 Gemini 润色</text>')
    save_svg("fig13a_conformer_relaxation.svg", s)

    # fig13b: Timing comparison (640 x 250)
    cpu_s = r["backends"]["cpu"]["seconds"]
    nv_s = r["backends"]["cuda"]["seconds"]
    amd_s = r["backends"]["hip"]["seconds"]
    s = svg_header(640, 250)
    s.append('<text x="24" y="34" font-size="19" font-weight="bold" fill="#0f172a">10,000 构象 × 200 步批量松弛耗时：GPU 3 秒级扫完</text>')
    s.append('<text x="24" y="56" font-size="12.5" fill="#047857" font-weight="bold">全部 400,350 个三维坐标与 10,000×201 步轨迹在 CPU/CUDA/HIP 三端逐位相同（L2 = 0）</text>')

    for i, (lbl, sec, col, note, bold) in enumerate([
        ("同源编译 C++ CPU 单核", cpu_s, "#94a3b8", f"{cpu_s:.3f} s", False),
        ("NVIDIA GB10 (CUDA)", nv_s, "#2563eb", f"{nv_s:.3f} s  (快同源 CPU {cpu_s/nv_s:.2f}×)", False),
        ("AMD RX 9070 XT (HIP)", amd_s, "#dc2626", f"{amd_s:.3f} s  (快同源 CPU {cpu_s/amd_s:.2f}×)", True),
    ]):
        y = 95 + i * 42
        w = max(8.0, sec / cpu_s * 270.0)
        fw = "bold" if bold else "normal"
        fc = "#0f172a" if bold else "#334155"
        s.append(f'<text x="24" y="{y}" font-size="13.2" font-weight="{fw}" fill="{fc}">{lbl}</text>')
        s.append(f'<rect x="196" y="{y-14}" width="{w:.1f}" height="19" rx="4" fill="{col}"/>')
        s.append(f'<text x="{196+w+10:.1f}" y="{y}" font-size="12.5" font-weight="{fw}" fill="{fc}">{note}</text>')

    s.append('<text x="24" y="233" font-size="11.5" fill="#64748b">数据源：同源重原子力场双精度实测（module14，不含显式氢优化）；排版经 Gemini 润色</text>')
    save_svg("fig13b_conformer_benchmark.svg", s)


# ---------------------------------------------------------
# 6. fig9a & fig9b: Resident Block-Davidson Quantum Eigensolver
# ---------------------------------------------------------
def build_fig9():
    r = json.loads((RES / "module15_results.json").read_text())
    p_cuda = np.genfromtxt(RES / "module15_cuda_256_batch_history.csv", delimiter=",", names=True)
    p_hip = np.genfromtxt(RES / "module15_hip_256_batch_history.csv", delimiter=",", names=True)

    # fig9a: Residual convergence over 16 iterations (640 x 345)
    s = svg_header(640, 345)
    s.append('<text x="24" y="34" font-size="19" font-weight="bold" fill="#0f172a">全 GPU 驻留 Block-Davidson 本征值求解：16 轮残差指数收敛</text>')
    s.append('<text x="24" y="56" font-size="12.2" fill="#475569">32×32 共享内存 Jacobi 与两遍 MGS 全进持久 Block 核，零主机往返；4,096 个本征值误差 ≤ 7.22e-15</text>')

    x0, y0, pw, ph = 74, 76, 536, 198
    s.append(f'<rect x="{x0}" y="{y0}" width="{pw}" height="{ph}" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1.2"/>')
    log_lo, log_hi = -11.0, -1.0
    for exp_v in [-10, -8, -6, -4, -2]:
        yy = y0 + ph - (exp_v - log_lo) / (log_hi - log_lo) * ph
        col = "#dc2626" if exp_v == -10 else "#e2e8f0"
        dash = 'stroke-dasharray="4 3"' if exp_v == -10 else ""
        s.append(f'<line x1="{x0}" y1="{yy:.1f}" x2="{x0+pw}" y2="{yy:.1f}" stroke="{col}" stroke-width="1.1" {dash}/>')
        s.append(f'<text x="{x0-8}" y="{yy+4:.1f}" font-size="11" fill="#475569" text-anchor="end">10^{exp_v}</text>')
    n_it = int(max(p_cuda["iteration"].max(), p_hip["iteration"].max()))
    for it in range(2, n_it + 1, 2):
        xx = x0 + (it - 1) / (n_it - 1) * pw
        s.append(f'<line x1="{xx:.1f}" y1="{y0}" x2="{xx:.1f}" y2="{y0+ph}" stroke="#e2e8f0" stroke-width="1"/>')
        s.append(f'<text x="{xx:.1f}" y="{y0+ph+17}" font-size="11" fill="#475569" text-anchor="middle">{it}</text>')

    for arr, col, dash, r_dot in [(p_cuda, "#2563eb", "", 3.6), (p_hip, "#ea580c", "5 3", 2.4)]:
        pts = []
        for it, rv in zip(arr["iteration"], arr["max_residual"]):
            lv = max(math.log10(max(float(rv), 1e-12)), log_lo)
            xx = x0 + (float(it) - 1.0) / (n_it - 1.0) * pw
            yy = y0 + ph - (lv - log_lo) / (log_hi - log_lo) * ph
            pts.append((xx, yy))
        pstr = " ".join(f"{x:.2f},{y:.2f}" for x, y in pts)
        d_attr = f'stroke-dasharray="{dash}"' if dash else ""
        s.append(f'<polyline points="{pstr}" fill="none" stroke="{col}" stroke-width="2.4" {d_attr}/>')
        for x, y in pts:
            s.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{r_dot}" fill="{col}"/>')

    s.append('<rect x="364" y="88" width="234" height="66" rx="6" fill="#ffffff" stroke="#cbd5e1" stroke-width="1"/>')
    s.append('<line x1="376" y1="106" x2="402" y2="106" stroke="#2563eb" stroke-width="2.5"/>')
    s.append('<text x="410" y="110" font-size="12" font-weight="bold" fill="#1e40af">NVIDIA GB10 (CUDA 256组)</text>')
    s.append('<line x1="376" y1="126" x2="402" y2="126" stroke="#ea580c" stroke-width="2.5" stroke-dasharray="4 3"/>')
    s.append('<text x="410" y="130" font-size="12" font-weight="bold" fill="#9a3412">AMD RX 9070 XT (HIP 重合)</text>')
    s.append('<line x1="376" y1="145" x2="402" y2="145" stroke="#dc2626" stroke-width="1.3" stroke-dasharray="4 3"/>')
    s.append('<text x="410" y="149" font-size="11.5" fill="#b91c1c">设备内部收敛阈值 10^-10</text>')

    s.append(f'<text x="{x0+pw/2:.1f}" y="311" font-size="12" fill="#334155" text-anchor="middle">横轴：持久核内部迭代轮数（32 维厚重启子空间） │ 纵轴：256 个系统最大残差</text>')
    s.append('<text x="24" y="333" font-size="11.5" fill="#64748b">数据源：全 GPU 驻留 Block-Davidson 求解 256×384 哈密顿量（module15）；排版经 Gemini 润色</text>')
    save_svg("fig9a_davidson_convergence.svg", s)

    # fig9b: Sequential vs Batching Speedup on 256 systems (640 x 355)
    sys64, sys256 = r["systems"][0], r["systems"][1]
    b256 = sys256["backends"]
    np256 = sys256["metadata"]["CPU_numpy_full_eigvalsh_seconds"]
    b64 = sys64["backends"]

    s = svg_header(640, 355)
    s.append('<text x="24" y="34" font-size="19" font-weight="bold" fill="#0f172a">量子本征值串行 vs 全驻留批处理：256 个 k 点提速几十倍</text>')
    s.append(f'<text x="24" y="56" font-size="11.8" fill="#475569">单晶胞串行吃不饱 GPU 反输 CPU；全驻留批处理 RX 9070 XT 仅 0.698s（64 组亦以 {b64["hip_batch"]["total_seconds"]:.3f}s 快于 GB10 {b64["cuda_batch"]["total_seconds"]:.3f}s）</text>')

    # Log10 bar chart from 0.1s to 60s so both 39.2s and 0.698s are crystal clear!
    x0, w_max = 212, 260.0
    lmin, lmax = math.log10(0.2), math.log10(50.0)
    def lbar(sec):
        return max(12.0, (math.log10(sec) - lmin) / (lmax - lmin) * w_max)

    items = [
        ("NVIDIA GB10 串行单发 (seq)", b256["cuda_seq"]["total_seconds"], "#93c5fd", "39.193 s（小核串行吃不饱）", False),
        ("AMD RX 9070 XT 串行单发", b256["hip_seq"]["total_seconds"], "#fca5a5", "20.495 s", False),
        ("CPU 单线程 Davidson", b256["cpu"]["total_seconds"], "#94a3b8", "11.655 s", False),
        ("CPU NumPy 全矩阵 eigh", np256, "#64748b", f"{np256:.3f} s（高度优化全对角化）", False),
        ("NVIDIA GB10 全驻留批处理", b256["cuda_batch"]["total_seconds"], "#2563eb", "0.791 s（批处理提速 49.6×）", False),
        ("AMD RX 9070 XT 全驻留批处理", b256["hip_batch"]["total_seconds"], "#dc2626", "0.698 s（提速 29.4×，反超 GB10 与 NumPy）", True),
    ]
    for i, (lbl, sec, col, note, bold) in enumerate(items):
        y = 95 + i * 38
        bw = lbar(sec)
        fw = "bold" if bold else "normal"
        fc = "#0f172a" if bold else "#334155"
        s.append(f'<text x="24" y="{y}" font-size="12.8" font-weight="{fw}" fill="{fc}">{lbl}</text>')
        s.append(f'<rect x="{x0}" y="{y-14}" width="{bw:.1f}" height="19" rx="4" fill="{col}"/>')
        s.append(f'<text x="{x0+bw+9:.1f}" y="{y}" font-size="12.2" font-weight="{fw}" fill="{fc}">{note}</text>')

    s.append('<text x="24" y="337" font-size="11.5" fill="#64748b">数据源：256 系统 × 384 平面波求 4,096 个本征值总耗时（对数条长，module15）；排版经 Gemini 润色</text>')
    save_svg("fig9b_davidson_batch_benchmark.svg", s)


# ---------------------------------------------------------
# 7. fig10a & fig10b: cuPhoton Blackhole FX Correlator
# ---------------------------------------------------------
def arr_to_png_b64(arr: np.ndarray, cmap_name: str, out_px: int = 160) -> str:
    a = np.flipud(np.asarray(arr, dtype=np.float64))  # origin='lower'
    lo, hi = float(a.min()), float(a.max())
    norm = (a - lo) / max(hi - lo, 1e-12)
    if cmap_name == "gray":
        rgb = np.stack([norm * 255] * 3, axis=-1).astype(np.uint8)
    else:
        rgba = cm.get_cmap(cmap_name)(norm)
        rgb = (rgba[:, :, :3] * 255).astype(np.uint8)
    img = Image.fromarray(rgb).resize((out_px, out_px), resample=Image.NEAREST)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def build_fig10():
    r = json.loads((RES / "module11_results.json").read_text())
    sky = np.load(RES / "module11_data/sky.npy")
    im = np.load(RES / "module11_hip_images.npz")

    # fig10a: 2x2 Blackhole Imaging Pipeline (640 x 455)
    s = svg_header(640, 455)
    s.append('<text x="24" y="34" font-size="18.5" font-weight="bold" fill="#0f172a">6,710 万射电复采样干涉成像：从天线交叉相关重建黑洞光子环</text>')
    s.append('<text x="24" y="56" font-size="12.0" fill="#475569">16 天线 × 120 基线 × 64 时段：GPU 可见度跨卡相对 L2 = 2.84e-16，恢复图像相对 L2 = 7.51e-16</text>')

    panels = [
        (0, 0, sky, "inferno", "① 合成黑洞光子环输入天空"),
        (0, 1, im["uv_mask"], "gray", "② 16 天线 120 基线 (u,v) 频域覆盖"),
        (1, 0, im["dirty"], "inferno", "③ GPU FX 可见度逆 FFT 脏图"),
        (1, 1, im["clean"], "inferno", "④ Högbom CLEAN 去卷积恢复环 (15.23 dB)"),
    ]
    for row, col, data, cmap_n, title in panels:
        px_x = 76 + col * 276
        px_y = 92 + row * 172
        b64 = arr_to_png_b64(data, cmap_n, 136)
        s.append(f'<text x="{px_x+68}" y="{px_y-8}" font-size="12.5" font-weight="bold" fill="#0f172a" text-anchor="middle">{title}</text>')
        s.append(f'<rect x="{px_x-1}" y="{px_y-1}" width="138" height="138" fill="none" stroke="#94a3b8" stroke-width="1.2"/>')
        s.append(f'<image x="{px_x}" y="{px_y}" width="136" height="136" href="data:image/png;base64,{b64}"/>')

    # Arrows between stages
    s.append('<text x="282" y="165" font-size="18" font-weight="bold" fill="#64748b" text-anchor="middle">→</text>')
    s.append('<text x="282" y="337" font-size="18" font-weight="bold" fill="#64748b" text-anchor="middle">→</text>')

    s.append('<text x="24" y="439" font-size="11.5" fill="#64748b">数据源：AMD RX 9070 XT 实测 67,108,864 个 complex128 采样重建图像（module11）；排版经 Gemini 润色</text>')
    save_svg("fig10a_blackhole_imaging.svg", s)

    # fig10b: FX Correlator timing benchmark (640 x 305)
    cpu_b, nv_b, amd_b = r["backends"]["cpu"], r["backends"]["cuda"], r["backends"]["hip"]
    cpu_tot = cpu_b["FFT_seconds"] + cpu_b["X_seconds"]
    nv_tot = nv_b["FFT_seconds"] + nv_b["X_seconds"]
    amd_tot = amd_b["FFT_seconds"] + amd_b["X_seconds"]
    cpu_x = cpu_b["X_seconds"] * 1000.0
    nv_x = nv_b["X_seconds"] * 1000.0
    amd_x = amd_b["X_seconds"] * 1000.0

    s = svg_header(640, 305)
    s.append('<text x="24" y="34" font-size="19" font-weight="bold" fill="#0f172a">射电望远镜 FX 关联器耗时：6,710 万个双精度复采样</text>')

    s.append('<text x="24" y="68" font-size="14" font-weight="bold" fill="#0f172a">① 核心算力大头：X 阶段 120 基线复数共轭交叉相关累加耗时</text>')
    for i, (lbl, ms, col, note, bold) in enumerate([
        ("CPU 参考基线", cpu_x, "#94a3b8", f"{cpu_x:.2f} ms", False),
        ("NVIDIA GB10 (CUDA)", nv_x, "#2563eb", f"{nv_x:.2f} ms  (快 CPU {cpu_x/nv_x:.1f}×)", False),
        ("AMD RX 9070 XT (HIP)", amd_x, "#dc2626", f"{amd_x:.2f} ms  (快 CPU {cpu_x/amd_x:.1f}×)", True),
    ]):
        y = 95 + i * 28
        w = max(8.0, ms / cpu_x * 265.0)
        fw = "bold" if bold else "normal"
        fc = "#0f172a" if bold else "#334155"
        s.append(f'<text x="24" y="{y}" font-size="12.8" font-weight="{fw}" fill="{fc}">{lbl}</text>')
        s.append(f'<rect x="192" y="{y-13}" width="{w:.1f}" height="17" rx="3.5" fill="{col}"/>')
        s.append(f'<text x="{192+w+10:.1f}" y="{y}" font-size="12.5" font-weight="{fw}" fill="{fc}">{note}</text>')

    s.append('<text x="24" y="188" font-size="14" font-weight="bold" fill="#0f172a">② F 阶段分信道 FFT ＋ X 阶段复数相关总耗时</text>')
    for i, (lbl, sec, col, note, bold) in enumerate([
        ("CPU 参考基线", cpu_tot, "#94a3b8", f"{cpu_tot:.3f} s", False),
        ("NVIDIA GB10 (CUDA)", nv_tot, "#2563eb", f"{nv_tot:.3f} s  (快 CPU {cpu_tot/nv_tot:.1f}×)", False),
        ("AMD RX 9070 XT (HIP)", amd_tot, "#dc2626", f"{amd_tot:.3f} s  (快 CPU {cpu_tot/amd_tot:.1f}×)", True),
    ]):
        y = 215 + i * 28
        w = max(8.0, sec / cpu_tot * 265.0)
        fw = "bold" if bold else "normal"
        fc = "#0f172a" if bold else "#334155"
        s.append(f'<text x="24" y="{y}" font-size="12.8" font-weight="{fw}" fill="{fc}">{lbl}</text>')
        s.append(f'<rect x="192" y="{y-13}" width="{w:.1f}" height="17" rx="3.5" fill="{col}"/>')
        s.append(f'<text x="{192+w+10:.1f}" y="{y}" font-size="12.5" font-weight="{fw}" fill="{fc}">{note}</text>')

    s.append('<text x="24" y="291" font-size="11.5" fill="#64748b">数据源：DGX Spark (GB10) 与 AMD RX 9070 XT 原生 FP64 复数干涉实测（module11）；排版经 Gemini 润色</text>')
    save_svg("fig10b_cuphoton_fx_benchmark.svg", s)


# ---------------------------------------------------------
# 8. fig12a & fig12b: Full 15-Layer Uni-Mol Native HIP Inference
# ---------------------------------------------------------
def build_fig12():
    r = json.loads((RES / "module13_results.json").read_text())
    xs = np.array(r["predictions_GB10_FP64"])
    ys = np.array(r["predictions_AMD_FP64"])

    # fig12a: 112 Test Molecules Parity Scatter (640 x 375)
    s = svg_header(640, 375)
    s.append('<text x="24" y="34" font-size="18.5" font-weight="bold" fill="#0f172a">深势 Uni-Mol 全 15 层原生 HIP 推理：112 个测试药物分子逐点对账</text>')
    s.append('<text x="24" y="56" font-size="12.3" fill="#047857" font-weight="bold">47,331,139 参数脱离 PyTorch 原生移植 AMD RX 9070 XT：预测最大差仅 7.11e-15 logS</text>')

    x0, y0, pw, ph = 74, 76, 536, 226
    s.append(f'<rect x="{x0}" y="{y0}" width="{pw}" height="{ph}" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1.2"/>')
    lo, hi = -10.0, 2.0
    for tv in [-10, -8, -6, -4, -2, 0, 2]:
        xx = x0 + (tv - lo) / (hi - lo) * pw
        yy = y0 + ph - (tv - lo) / (hi - lo) * ph
        s.append(f'<line x1="{xx:.1f}" y1="{y0}" x2="{xx:.1f}" y2="{y0+ph}" stroke="#e2e8f0" stroke-width="1"/>')
        s.append(f'<line x1="{x0}" y1="{yy:.1f}" x2="{x0+pw}" y2="{yy:.1f}" stroke="#e2e8f0" stroke-width="1"/>')
        s.append(f'<text x="{xx:.1f}" y="{y0+ph+17}" font-size="11.5" fill="#475569" text-anchor="middle">{tv}</text>')
        s.append(f'<text x="{x0-8}" y="{yy+4:.1f}" font-size="11.5" fill="#475569" text-anchor="end">{tv}</text>')

    s.append(f'<line x1="{x0}" y1="{y0+ph}" x2="{x0+pw}" y2="{y0}" stroke="#64748b" stroke-width="1.5" stroke-dasharray="5 4"/>')
    for a, b in zip(xs, ys):
        cx = x0 + (a - lo) / (hi - lo) * pw
        cy = y0 + ph - (b - lo) / (hi - lo) * ph
        s.append(f'<circle cx="{cx:.2f}" cy="{cy:.2f}" r="3.8" fill="#10b981" fill-opacity="0.82" stroke="#047857" stroke-width="0.6"/>')

    s.append('<rect x="92" y="90" width="292" height="66" rx="6" fill="#ffffff" stroke="#cbd5e1" stroke-width="1"/>')
    s.append('<text x="104" y="111" font-size="12" font-weight="bold" fill="#0f172a">ESOL 112 个留出测试药物分子指标</text>')
    s.append(f'<text x="104" y="130" font-size="11.8" fill="#1e40af">• NVIDIA GB10：RMSE = 0.573634, R² = 0.930467</text>')
    s.append(f'<text x="104" y="148" font-size="11.8" font-weight="bold" fill="#991b1b">• AMD RX 9070 XT：RMSE = 0.573634, R² = 0.930467</text>')

    s.append(f'<text x="{x0+pw/2:.1f}" y="339" font-size="12" fill="#334155" text-anchor="middle">横轴：NVIDIA GB10 预测水溶性 logS │ 纵轴：AMD RX 9070 XT 原生 HIP 预测 logS</text>')
    s.append('<text x="24" y="361" font-size="11.5" fill="#64748b">数据源：导出全部 195 个权重张量在 HIP + hipBLAS 上端到端推理（module13）；排版经 Gemini 润色</text>')
    save_svg("fig12a_unimol_hip_scatter.svg", s)

    # fig12b: Layerwise 50.5M intermediate tensor max error trace (640 x 350)
    lw = r["layerwise_all112_max_abs"]
    vals = [max(v["x_max_abs"], v["pair_max_abs"]) for v in lw]
    stage_names = [v.get("stage", f"L{i}") for i, v in enumerate(lw)]

    s = svg_header(640, 350)
    s.append('<text x="24" y="34" font-size="18.5" font-weight="bold" fill="#0f172a">全 15 层 5,050 万个中间张量元素逐层核对：误差全程低于 10^-12</text>')
    s.append('<text x="24" y="56" font-size="12.0" fill="#475569">同公式 FP64 逐层最大差 ≤ 9.09e-13；对齐高斯核 FP32 路径后与官方 PyTorch 最大差 2.27e-6 logS</text>')

    x0, y0, pw, ph = 78, 76, 532, 198
    s.append(f'<rect x="{x0}" y="{y0}" width="{pw}" height="{ph}" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1.2"/>')
    log_lo, log_hi = -15.5, -11.5
    for ev in [-15, -14, -13, -12]:
        yy = y0 + ph - (ev - log_lo) / (log_hi - log_lo) * ph
        s.append(f'<line x1="{x0}" y1="{yy:.1f}" x2="{x0+pw}" y2="{yy:.1f}" stroke="#e2e8f0" stroke-width="1"/>')
        s.append(f'<text x="{x0-8}" y="{yy+4:.1f}" font-size="11" fill="#475569" text-anchor="end">10^{ev}</text>')

    n_pts = len(vals)
    pts = []
    for i, v in enumerate(vals):
        xx = x0 + i / max(n_pts - 1, 1) * pw
        lv = max(min(math.log10(max(v, 1e-16)), log_hi), log_lo)
        yy = y0 + ph - (lv - log_lo) / (log_hi - log_lo) * ph
        pts.append((xx, yy, stage_names[i]))
        if i in (0, 3, 6, 9, 12, n_pts - 1):
            s.append(f'<line x1="{xx:.1f}" y1="{y0}" x2="{xx:.1f}" y2="{y0+ph}" stroke="#e2e8f0" stroke-width="1"/>')
            short_lbl = (
                stage_names[i]
                .replace("embedding_and_gaussian", "Emb+Gauss")
                .replace("encoder_", "")
                .replace("layer_", "L")
            )
            s.append(f'<text x="{xx:.1f}" y="{y0+ph+17}" font-size="11" fill="#475569" text-anchor="middle">{short_lbl}</text>')

    pstr = " ".join(f"{x:.2f},{y:.2f}" for x, y, _ in pts)
    s.append(f'<polyline points="{pstr}" fill="none" stroke="#2563eb" stroke-width="2.5"/>')
    for x, y, _ in pts:
        s.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="3.4" fill="#2563eb" stroke="#ffffff" stroke-width="1"/>')

    # Info box in lower-right whitespace (10^-14 .. 10^-15 area, zero overlap with curve!)
    s.append('<rect x="268" y="196" width="328" height="66" rx="6" fill="#ffffff" stroke="#cbd5e1" stroke-width="1"/>')
    s.append('<text x="280" y="216" font-size="12" font-weight="bold" fill="#0f172a">全网推理耗时（含 15 层注意力成对偏置更新）</text>')
    s.append('<text x="280" y="235" font-size="11.8" fill="#334155">• NVIDIA GB10 (FP64)：1.756 s │ AMD RX 9070 XT：2.291 s</text>')
    s.append('<text x="280" y="253" font-size="11.8" fill="#047857">• 混合精度对齐官方 PyTorch 全 FP32：最大差 2.27e-6 logS</text>')

    s.append(f'<text x="{x0+pw/2:.1f}" y="313" font-size="12" fill="#334155" text-anchor="middle">横轴：Uni-Mol 前向传播层序列（Emb+Gauss → L1..L15） │ 纵轴：112 分子最大绝对误差</text>')
    s.append('<text x="24" y="336" font-size="11.5" fill="#64748b">数据源：50,501,632 个节点与原子对中间张量元素跨卡比对（module13）；排版经 Gemini 润色</text>')
    save_svg("fig12b_unimol_15layer_error.svg", s)


if __name__ == "__main__":
    build_fig4()
    build_fig11()
    build_fig6()
    build_fig8()
    build_fig13()
    build_fig9()
    build_fig10()
    build_fig12()
