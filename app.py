import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import json
import os
from typing import Any

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Benchmark Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
}
.stApp {
    background: #0a0a0f;
    color: #e2e8f0;
}
.block-container {
    padding-top: 2rem;
    max-width: 1400px;
}

/* Hero header */
.hero-header {
    background: linear-gradient(135deg, #0d1117 0%, #161b22 50%, #0d1117 100%);
    border: 1px solid #30363d;
    border-radius: 16px;
    padding: 2rem 2.5rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero-header::before {
    content: '';
    position: absolute;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: radial-gradient(ellipse at 70% 30%, rgba(56,189,248,0.06) 0%, transparent 60%),
                radial-gradient(ellipse at 30% 70%, rgba(168,85,247,0.06) 0%, transparent 60%);
    pointer-events: none;
}
.hero-title {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 2.4rem;
    background: linear-gradient(90deg, #38bdf8, #a855f7, #f472b6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
    letter-spacing: -0.02em;
}
.hero-sub {
    font-family: 'Space Mono', monospace;
    font-size: 0.78rem;
    color: #6b7280;
    margin-top: 0.4rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
}

/* Metric cards */
.metric-card {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    position: relative;
    overflow: hidden;
    transition: border-color 0.2s;
}
.metric-card:hover { border-color: #30363d; }
.metric-card::after {
    content: '';
    position: absolute;
    bottom: 0; left: 0; right: 0;
    height: 2px;
}
.metric-card.cpu::after { background: linear-gradient(90deg, #38bdf8, #0ea5e9); }
.metric-card.gpu::after { background: linear-gradient(90deg, #a855f7, #ec4899); }
.metric-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.68rem;
    color: #6b7280;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 0.3rem;
}
.metric-value {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 2rem;
    line-height: 1;
    margin-bottom: 0.2rem;
}
.metric-value.cpu { color: #38bdf8; }
.metric-value.gpu { color: #a855f7; }
.metric-unit {
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    color: #4b5563;
    text-transform: uppercase;
}

/* Section headers */
.section-header {
    font-family: 'Space Mono', monospace;
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.15em;
    padding: 0.3rem 0.8rem;
    border-radius: 4px;
    display: inline-block;
    margin-bottom: 1rem;
    margin-top: 0.5rem;
}
.section-header.cpu {
    background: rgba(56,189,248,0.1);
    color: #38bdf8;
    border: 1px solid rgba(56,189,248,0.2);
}
.section-header.gpu {
    background: rgba(168,85,247,0.1);
    color: #a855f7;
    border: 1px solid rgba(168,85,247,0.2);
}
.section-header.compare {
    background: rgba(244,114,182,0.1);
    color: #f472b6;
    border: 1px solid rgba(244,114,182,0.2);
}

/* Info box */
.info-box {
    background: #161b22;
    border: 1px solid #21262d;
    border-left: 3px solid #38bdf8;
    border-radius: 8px;
    padding: 1rem 1.2rem;
    font-family: 'Space Mono', monospace;
    font-size: 0.78rem;
    color: #9ca3af;
    line-height: 1.7;
}
.info-box.gpu { border-left-color: #a855f7; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: #0d1117 !important;
    border-right: 1px solid #21262d;
}
[data-testid="stSidebar"] .stMarkdown { color: #9ca3af; }

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: #161b22;
    border-radius: 8px;
    padding: 4px;
    gap: 4px;
    border: 1px solid #21262d;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'Space Mono', monospace;
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #6b7280 !important;
    border-radius: 6px;
}
.stTabs [aria-selected="true"] {
    background: #21262d !important;
    color: #e2e8f0 !important;
}

/* Divider */
.divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, #21262d, transparent);
    margin: 1.5rem 0;
}
</style>
""", unsafe_allow_html=True)

# ─── Color constants ───────────────────────────────────────────────────────────
PLOT_BG   = "#0d1117"
PAPER_BG  = "#0d1117"
GRID_COL  = "#21262d"
TEXT_COL  = "#9ca3af"
CPU_COL   = "#38bdf8"
GPU_FP16  = "#a855f7"
GPU_FP32  = "#f472b6"
GPU_FP64  = "#fb923c"
DT_COLORS = {"torch.float16": GPU_FP16, "torch.float32": GPU_FP32, "torch.float64": GPU_FP64}
DT_LABELS = {"torch.float16": "FP16", "torch.float32": "FP32", "torch.float64": "FP64"}

BASE_LAYOUT: dict[str, Any] = dict(
    paper_bgcolor=PAPER_BG,
    plot_bgcolor=PLOT_BG,
    font=dict(family="Space Mono, monospace", color=TEXT_COL, size=11),
    margin=dict(l=50, r=30, t=40, b=50),
    xaxis=dict(gridcolor=GRID_COL, zeroline=False, tickcolor=GRID_COL, linecolor=GRID_COL),
    yaxis=dict(gridcolor=GRID_COL, zeroline=False, tickcolor=GRID_COL, linecolor=GRID_COL),
    legend=dict(bgcolor="rgba(22,27,34,0.8)", bordercolor=GRID_COL, borderwidth=1),
)

# ─── Data loading ──────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def load_json(fname):
    path = os.path.join(SCRIPT_DIR, fname)
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None

cpu_data = load_json("data/cpu_benchmark_results.json")
gpu_data = load_json("data/gpu_benchmark_results.json")

# ─── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚡ Benchmark Data")
    st.divider()

    st.markdown("**Data sources**")
    cpu_ok = cpu_data is not None
    gpu_ok = gpu_data is not None

    if cpu_ok:
        st.success("✅ cpu_benchmark_results.json")
    else:
        st.error("❌ cpu_benchmark_results.json not found")

    if gpu_ok:
        st.success("✅ gpu_benchmark_results.json")
    else:
        st.error("❌ gpu_benchmark_results.json not found")

    st.divider()
    st.markdown("""
    <div style='font-family:Space Mono,monospace;font-size:0.68rem;color:#374151;line-height:1.7'>
    Run the notebooks to generate<br>
    the JSON result files.<br><br>
    Place <b>cpu_benchmark_results.json</b><br>
    and <b>gpu_benchmark_results.json</b><br>
    in the same folder as this script.
    </div>
    """, unsafe_allow_html=True)

    if cpu_ok:
        st.divider()
        st.markdown("**GPU Theoretical Peaks (TFLOPS)**")
        peak_fp16 = st.number_input("FP16 peak", value=641.0, step=10.0)
        peak_fp32 = st.number_input("FP32 peak (TF32)", value=641.0, step=10.0)
        peak_fp64 = st.number_input("FP64 peak", value=1.2, step=0.1, format="%.1f")
        THEORETICAL_PEAKS = {
            "torch.float16": peak_fp16,
            "torch.float32": peak_fp32,
            "torch.float64": peak_fp64,
        }
    else:
        THEORETICAL_PEAKS = {"torch.float16": 641.0, "torch.float32": 641.0, "torch.float64": 1.2}

# ─── Header ────────────────────────────────────────────────────────────────────
cpu_name_str  = cpu_data["cpu_name"]  if cpu_ok else "—"
gpu_name_str  = gpu_data["gpu_name"]  if gpu_ok else "—"

st.markdown(f"""
<div class="hero-header">
  <p class="hero-title">⚡ Hardware Benchmark Dashboard</p>
  <p class="hero-sub">CPU: {cpu_name_str}  ·  GPU: {gpu_name_str}</p>
</div>
""", unsafe_allow_html=True)

if not cpu_ok and not gpu_ok:
    st.warning("No benchmark data found. Run the notebooks first to generate the JSON result files.")
    st.stop()

# ─── KPI row ───────────────────────────────────────────────────────────────────
def kpi(col, label, val, unit, cls):
    col.markdown(f"""
    <div class="metric-card {cls}">
      <div class="metric-label">{label}</div>
      <div class="metric-value {cls}">{val:.1f}</div>
      <div class="metric-unit">{unit}</div>
    </div>
    """, unsafe_allow_html=True)

k1, k2, k3, k4, k5, k6 = st.columns(6)

cpu_peak: float = 0.0
cpu_stable: float = 0.0
gflops_hist: list[float] = []

if cpu_ok:
    gflops_hist = cpu_data["gflops_history"]
    cpu_peak    = max(gflops_hist)
    cpu_stable  = sum(gflops_hist[1:]) / max(len(gflops_hist) - 1, 1)
    kpi(k1, "CPU Peak",   cpu_peak,   "GFLOPS", "cpu")
    kpi(k2, "CPU Stable", cpu_stable, "GFLOPS", "cpu")

if gpu_ok:
    results = gpu_data["results"]
    first_dt = list(results.keys())[0]
    gd = results[first_dt]
    kpi(k3, f"GPU Peak ({DT_LABELS.get(first_dt, first_dt)})",   max(gd["tflops_history"]), "TFLOPS", "gpu")
    kpi(k4, f"GPU Stable ({DT_LABELS.get(first_dt, first_dt)})", gd["stable_mean"],         "TFLOPS", "gpu")

    # FP16/FP32 ratio
    fp16_key = "torch.float16"
    fp32_key = "torch.float32"
    if fp16_key in results and fp32_key in results:
        r16 = np.mean(results[fp16_key]["tflops_history"])
        r32 = np.mean(results[fp32_key]["tflops_history"])
        kpi(k5, "FP16/FP32 ratio", r16 / max(r32, 1e-9), "×", "gpu")

    # GPU vs CPU
    if cpu_ok:
        gpu_gflops = np.mean(gd["tflops_history"]) * 1000
        kpi(k6, "GPU/CPU (×)", gpu_gflops / max(cpu_stable, 1e-9), "speedup", "gpu")

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# ─── Tabs ──────────────────────────────────────────────────────────────────────
tab_cpu, tab_gpu, tab_compare = st.tabs(["🖥️  CPU Benchmark", "🚀  GPU Benchmark", "⚖️  Comparison"])

# ══════════════════════════════════════════════════════════════════════════════
# TAB: CPU
# ══════════════════════════════════════════════════════════════════════════════
with tab_cpu:
    if not cpu_ok:
        st.info("Run CPU_Benchmark.ipynb to generate data.")
    else:
        st.markdown(f'<span class="section-header cpu">CPU · {cpu_data["cpu_name"]} · Matmul {cpu_data["dtype"].upper()}</span>',
                    unsafe_allow_html=True)

        gflops_hist  = cpu_data["gflops_history"]
        elapsed_hist = cpu_data["elapsed_history"]
        bench_iters  = cpu_data["bench_iters"]
        matrix_size  = cpu_data["matrix_size"]
        dtype_str    = cpu_data["dtype"]
        mean_g       = np.mean(gflops_hist)
        stable_g     = np.mean(gflops_hist[1:]) if len(gflops_hist) > 1 else mean_g
        peak_g       = max(gflops_hist)
        iters        = list(range(1, len(gflops_hist) + 1))

        c1, c2 = st.columns([3, 2])

        # Chart 1 — GFLOPS per iteration
        with c1:
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=iters, y=gflops_hist,
                marker=dict(
                    color=gflops_hist,
                    colorscale=[[0, "#0f4c75"], [0.5, "#1b6ca8"], [1, CPU_COL]],
                    line=dict(width=0),
                ),
                name="GFLOPS / iter",
                hovertemplate="Iter %{x}: <b>%{y:.2f} GFLOPS</b><extra></extra>",
            ))
            for val, col, dash, lbl in [
                (mean_g,   "#ef4444", "dash",    f"Mean: {mean_g:.1f}"),
                (stable_g, "#22c55e", "dot",     f"Stable: {stable_g:.1f}"),
                (peak_g,   "#fbbf24", "dashdot", f"Peak: {peak_g:.1f}"),
            ]:
                fig.add_hline(y=val, line_color=col, line_dash=dash, line_width=1.5,
                              annotation_text=lbl, annotation_position="right",
                              annotation_font=dict(color=col, size=10))
            fig.update_layout(**BASE_LAYOUT,
                title=dict(text=f"GFLOPS / Iteration — {dtype_str.upper()} | {matrix_size}×{matrix_size}", x=0),
                xaxis_title="Iteration", yaxis_title="GFLOPS",
                showlegend=False, height=320,
            )
            st.plotly_chart(fig, width='stretch')

        # Chart 2 — Thread scaling
        with c2:
            thread_results = cpu_data.get("thread_results", [])
            if thread_results:
                t_x = [r[0] for r in thread_results]
                t_y = [r[1] for r in thread_results]
                ideal = [t_y[0] * (t / t_x[0]) for t in t_x]

                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(
                    x=t_x, y=ideal, mode="lines",
                    line=dict(color=GRID_COL, dash="dash", width=1.5),
                    name="Ideal (linear)", hoverinfo="skip",
                ))
                fig2.add_trace(go.Scatter(
                    x=t_x, y=t_y, mode="lines+markers",
                    line=dict(color=CPU_COL, width=2.5),
                    marker=dict(size=8, color=CPU_COL, line=dict(color="#0a0a0f", width=2)),
                    name="Measured",
                    fill="tonexty", fillcolor="rgba(56,189,248,0.07)",
                    hovertemplate="%{x} threads: <b>%{y:.1f} GFLOPS</b><extra></extra>",
                ))
                layout_cfg = BASE_LAYOUT.copy()
                layout_cfg["legend"] = {
                    **BASE_LAYOUT.get("legend", {}),
                    "orientation": "h",
                    "y": -0.2,
                }

                fig2.update_layout(**layout_cfg,
                    title=dict(text="Multi-thread Scaling", x=0),
                    xaxis_title="Threads", yaxis_title="GFLOPS",
                    height=320,
                )
                st.plotly_chart(fig2, width='stretch')
            else:
                st.info("Thread scaling data not available.")

        # Bottom row
        c3, c4 = st.columns(2)
        with c3:
            fig3 = go.Figure(go.Scatter(
                x=iters, y=elapsed_hist,
                mode="lines+markers",
                line=dict(color=CPU_COL, width=1.5),
                fill="tozeroy", fillcolor="rgba(56,189,248,0.07)",
                hovertemplate="Iter %{x}: <b>%{y:.1f} ms</b><extra></extra>",
            ))
            fig3.update_layout(**BASE_LAYOUT,
                title=dict(text="Time per Iteration (ms)", x=0),
                xaxis_title="Iteration", yaxis_title="ms",
                height=260,
            )
            st.plotly_chart(fig3, width='stretch')

        with c4:
            st.markdown('<span class="section-header cpu">System Info</span>', unsafe_allow_html=True)
            st.markdown(f"""
            <div class="info-box">
            CPU: {cpu_data['cpu_name']}<br>
            Cores: {cpu_data['cpu_cores']}<br>
            Frequency: {cpu_data['freq_str']}<br>
            RAM: {cpu_data['ram_total']:.2f} GB<br>
            PyTorch threads: {cpu_data['torch_threads']}<br>
            Matrix: {matrix_size}×{matrix_size} {dtype_str.upper()}<br>
            Iterations: {bench_iters}<br>
            Total time: {cpu_data['total_elapsed']:.2f}s
            </div>
            """, unsafe_allow_html=True)

        # Stats table
        st.markdown('<span class="section-header cpu">Statistics</span>', unsafe_allow_html=True)
        stats = pd.DataFrame({
            "Metric": ["Peak GFLOPS", "Mean GFLOPS", "Stable Mean (iter 2+)", "Mean Time/iter (ms)"],
            "Value":  [f"{peak_g:.2f}", f"{mean_g:.2f}", f"{stable_g:.2f}",
                       f"{np.mean(elapsed_hist):.1f}"],
        })
        st.dataframe(stats, width='stretch', hide_index=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB: GPU
# ══════════════════════════════════════════════════════════════════════════════
with tab_gpu:
    if not gpu_ok:
        st.info("Run GPU_Benchmark.ipynb to generate data.")
    else:
        results   = gpu_data["results"]
        gpu_name  = gpu_data["gpu_name"]
        matrix_size = gpu_data["matrix_size"]
        bench_iters = gpu_data["bench_iters"]

        st.markdown(f'<span class="section-header gpu">GPU · {gpu_name} · Matmul FP16/FP32/FP64</span>',
                    unsafe_allow_html=True)

        iters_gpu = list(range(1, bench_iters + 1))

        # Main TFLOPS chart
        fig_gpu = go.Figure()
        for dt, gd in results.items():
            col = DT_COLORS.get(dt, "#fff")
            label = DT_LABELS.get(dt, dt)
            fig_gpu.add_trace(go.Scatter(
                x=iters_gpu, y=gd["tflops_history"],
                mode="lines+markers", name=label,
                line=dict(color=col, width=2),
                marker=dict(size=6, color=col),
                hovertemplate=f"{label} — Iter %{{x}}: <b>%{{y:.2f}} TFLOPS</b><extra></extra>",
            ))
            fig_gpu.add_hline(
                y=gd["stable_mean"], line_color=col, line_dash="dot",
                line_width=1, opacity=0.5,
                annotation_text=f"{label} stable: {gd['stable_mean']:.1f}",
                annotation_font=dict(color=col, size=9),
                annotation_position="right",
            )

        gpu_layout_cfg = BASE_LAYOUT.copy()
        base_legend = BASE_LAYOUT.get("legend", {})
        if not isinstance(base_legend, dict):
            base_legend = {}
        gpu_layout_cfg["legend"] = {
            **base_legend,
            "orientation": "h",
            "y": -0.18,
        }

        fig_gpu.update_layout(**gpu_layout_cfg,
            title=dict(text=f"TFLOPS / Iteration — {matrix_size}×{matrix_size}", x=0),
            xaxis_title="Iteration", yaxis_title="TFLOPS",
            height=360,
        )
        st.plotly_chart(fig_gpu, width='stretch')

        # Efficiency gauges
        st.markdown('<span class="section-header gpu">Efficiency vs Theoretical Peak</span>',
                    unsafe_allow_html=True)

        eff_cols = st.columns(len(results))
        for idx, (dt, gd) in enumerate(results.items()):
            peak = THEORETICAL_PEAKS.get(dt, 0)
            eff  = (gd["stable_mean"] / peak * 100) if peak > 0 else 0
            col  = DT_COLORS.get(dt, "#fff")
            label = DT_LABELS.get(dt, dt)
            fig_eff = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=eff,
                number=dict(suffix="%", font=dict(size=28, color=col)),
                delta=dict(reference=80, increasing_color=col),
                title=dict(text=f"{label} Efficiency<br><span style='font-size:0.75em;color:#6b7280'>"
                                f"Peak: {peak} TFLOPS</span>"),
                gauge=dict(
                    axis=dict(range=[0, 100], tickwidth=1, tickcolor=GRID_COL),
                    bar=dict(color=col, thickness=0.25),
                    bgcolor="#161b22",
                    borderwidth=0,
                    steps=[dict(range=[0, 100], color="#1c2128")],
                    threshold=dict(line=dict(color="#4b5563", width=2), value=80),
                ),
            ))
            fig_eff.update_layout(
                paper_bgcolor=PAPER_BG, font=dict(color=TEXT_COL, family="Space Mono"),
                height=240, margin=dict(l=30, r=30, t=50, b=10),
            )
            eff_cols[idx].plotly_chart(fig_eff, width='stretch')

        # System info
        g1, g2 = st.columns(2)
        with g1:
            st.markdown('<span class="section-header gpu">System Info</span>', unsafe_allow_html=True)
            st.markdown(f"""
            <div class="info-box gpu">
            GPU: {gpu_data['gpu_name']}<br>
            VRAM: {gpu_data['total_vram']:.2f} GB<br>
            SM count: {gpu_data['sm_count']}<br>
            Matrix: {matrix_size}×{matrix_size}<br>
            Iterations: {bench_iters}
            </div>
            """, unsafe_allow_html=True)

        with g2:
            # GPU stats table
            rows = []
            for dt, gd in results.items():
                peak = THEORETICAL_PEAKS.get(dt, 0)
                eff  = (gd["stable_mean"] / peak * 100) if peak > 0 else 0
                rows.append({
                    "Dtype": DT_LABELS.get(dt, dt),
                    "Peak (TFLOPS)": f"{max(gd['tflops_history']):.2f}",
                    "Mean (TFLOPS)": f"{np.mean(gd['tflops_history']):.2f}",
                    "Stable Mean":   f"{gd['stable_mean']:.2f}",
                    "Theoretical":   f"{peak:.1f}",
                    "Efficiency":    f"{eff:.1f}%",
                })
            st.dataframe(pd.DataFrame(rows), width='stretch', hide_index=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB: COMPARISON
# ══════════════════════════════════
    st.markdown('<span class="section-header compare">CPU vs GPU · Performance Breakdown</span>',
                unsafe_allow_html=True)

    if not cpu_ok or not gpu_ok:
        st.info("Both CPU and GPU data are required for comparison.")
    else:
        results = gpu_data["results"]

        # Bar comparison
        labels = ["CPU FP32"]
        values = [cpu_stable]
        colors = [CPU_COL]
        for dt, gd in results.items():
            labels.append(f"GPU {DT_LABELS.get(dt, dt)}")
            values.append(np.mean(gd["tflops_history"]) * 1000)
            colors.append(DT_COLORS.get(dt, "#fff"))

        fig_cmp = go.Figure(go.Bar(
            x=labels, y=values,
            marker=dict(color=colors, opacity=0.85, line=dict(color=colors, width=1)),
            text=[f"{v:.0f}" for v in values],
            textposition="outside",
            textfont=dict(size=11, color=TEXT_COL),
            hovertemplate="%{x}: <b>%{y:.1f} GFLOPS</b><extra></extra>",
        ))
        fig_cmp.update_layout(**BASE_LAYOUT,
            title=dict(text="Performance Comparison (GFLOPS — normalized)", x=0),
            yaxis_title="GFLOPS",
            height=380, showlegend=False,
        )
        st.plotly_chart(fig_cmp, width='stretch')

        # Radar + speedup table
        r1, r2 = st.columns(2)

        with r1:
            cats = ["Peak Perf", "Efficiency", "Stability", "Throughput"]
            cpu_norm_vals = [
                min(cpu_peak / 500 * 100, 100),
                90,
                (1 - np.std(gflops_hist) / max(np.mean(gflops_hist), 1e-9)) * 100,
                cpu_stable / max(cpu_peak, 1e-9) * 100,
            ]
            fig_r = go.Figure()
            fig_r.add_trace(go.Scatterpolar(
                r=cpu_norm_vals + [cpu_norm_vals[0]],
                theta=cats + [cats[0]],
                fill="toself", fillcolor="rgba(56,189,248,0.12)",
                line=dict(color=CPU_COL, width=2),
                name="CPU",
            ))

            fp16_key = "torch.float16"
            if fp16_key in results:
                gd16  = results[fp16_key]
                peak16 = THEORETICAL_PEAKS.get(fp16_key, 641)
                eff16  = gd16["stable_mean"] / max(peak16, 1e-9) * 100
                gpu_norm_vals = [
                    min(max(gd16["tflops_history"]) / peak16 * 100, 100),
                    eff16,
                    (1 - np.std(gd16["tflops_history"]) / max(np.mean(gd16["tflops_history"]), 1e-9)) * 100,
                    gd16["stable_mean"] / max(max(gd16["tflops_history"]), 1e-9) * 100,
                ]
                fig_r.add_trace(go.Scatterpolar(
                    r=gpu_norm_vals + [gpu_norm_vals[0]],
                    theta=cats + [cats[0]],
                    fill="toself", fillcolor="rgba(168,85,247,0.12)",
                    line=dict(color=GPU_FP16, width=2),
                    name="GPU FP16",
                ))

            fig_r.update_layout(
                polar=dict(
                    radialaxis=dict(range=[0, 100], gridcolor=GRID_COL, tickcolor=GRID_COL,
                                    showticklabels=False),
                    angularaxis=dict(gridcolor=GRID_COL),
                    bgcolor=PLOT_BG,
                ),
                paper_bgcolor=PAPER_BG, font=dict(color=TEXT_COL, family="Space Mono", size=10),
                legend=dict(bgcolor="rgba(22,27,34,0.8)", bordercolor=GRID_COL, borderwidth=1),
                title=dict(text="CPU vs GPU (normalized scores)", x=0.5),
                height=360, margin=dict(l=60, r=60, t=60, b=40),
            )
            st.plotly_chart(fig_r, width='stretch')

        with r2:
            speedups = []
            for dt, gd in results.items():
                gpu_gf = np.mean(gd["tflops_history"]) * 1000
                speedups.append({
                    "Config":        f"GPU {DT_LABELS.get(dt, dt)} vs CPU FP32",
                    "GPU (GFLOPS)":  f"{gpu_gf:.0f}",
                    "CPU (GFLOPS)":  f"{cpu_stable:.1f}",
                    "Speedup":       f"{gpu_gf / max(cpu_stable, 1e-9):.0f}×",
                })
            st.markdown('<span class="section-header compare">Speedup Summary</span>',
                        unsafe_allow_html=True)
            st.dataframe(pd.DataFrame(speedups), width='stretch', hide_index=True)

            st.markdown("""
            <div style="background:#161b22;border:1px solid #21262d;border-radius:8px;
            padding:1rem 1.2rem;margin-top:1rem;">
            <div style="font-family:Space Mono,monospace;font-size:0.68rem;
            color:#6b7280;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:0.8rem">
            Architecture Notes</div>
            <ul style="font-size:0.82rem;color:#9ca3af;line-height:1.9;margin:0;padding-left:1.2rem">
              <li>CPU FP16 not natively accelerated (no Tensor Cores)</li>
              <li>GPU FP32 uses TF32 via Tensor Cores when enabled</li>
              <li>GPU FP64 severely limited on consumer cards</li>
              <li>Stable mean excludes 1st iteration (cold start)</li>
              <li>Real TFLOPS &lt; theoretical due to memory bandwidth</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)

# ─── Footer ────────────────────────────────────────────────────────────────────
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
st.markdown("""
<div style="text-align:center;font-family:Space Mono,monospace;font-size:0.65rem;
color:#374151;padding:0.5rem 0">
⚡ HW BENCHMARK DASHBOARD · Real hardware data from Jupyter notebooks
</div>
""", unsafe_allow_html=True)
