import streamlit as st
import cv2
import tempfile
import os
import time
from pathlib import Path
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

from detector import CattleDetector, MODELS, COCO_COW_CLASS_ID

# ─────────────────────────────────────────────────────────────────────────────
# Page configuration
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="BovSmart — Contador de Bovinos",
    page_icon="🐄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Global ── */
[data-testid="stAppViewContainer"] {
    background: #0f1a0f;
}
[data-testid="stSidebar"] {
    background: #0d160d;
    border-right: 1px solid #2d5016;
}

/* ── Header banner ── */
.bov-header {
    background: linear-gradient(135deg, #1a3a08 0%, #2d5916 50%, #1a3a08 100%);
    border: 1px solid #4a8c23;
    border-radius: 12px;
    padding: 1.6rem 2rem;
    text-align: center;
    margin-bottom: 1.5rem;
    box-shadow: 0 4px 20px rgba(74,140,35,0.3);
}
.bov-header h1 { color: #7fff5a; font-size: 2.4rem; margin: 0; letter-spacing: 3px; }
.bov-header p  { color: #a8d890; margin: 0.3rem 0 0; font-size: 1rem; }

/* ── Stat cards ── */
.stat-grid { display: flex; gap: 1rem; margin: 1rem 0; }
.stat-card {
    flex: 1;
    background: linear-gradient(135deg, #152b08, #1e3d0d);
    border: 1px solid #3a6b1a;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    text-align: center;
}
.stat-value { font-size: 2.6rem; font-weight: 800; color: #7fff5a; line-height: 1.1; }
.stat-label { font-size: 0.78rem; color: #8fb870; text-transform: uppercase; letter-spacing: 1px; margin-top: 4px; }

/* ── Section headers ── */
.section-title {
    color: #7fff5a;
    font-size: 1.15rem;
    font-weight: 700;
    border-left: 4px solid #4a8c23;
    padding-left: 10px;
    margin: 1.5rem 0 0.8rem;
}

/* ── Upload zone ── */
[data-testid="stFileUploader"] {
    border: 2px dashed #3a6b1a !important;
    border-radius: 10px;
    background: #0d1a0d !important;
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #2d6b10, #4a9c1e);
    color: white;
    border: none;
    border-radius: 8px;
    font-weight: 700;
    letter-spacing: 0.5px;
    padding: 0.5rem 1.5rem;
    transition: all 0.2s;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #3a8c14, #5db828);
    box-shadow: 0 4px 12px rgba(74,156,30,0.4);
}

/* ── Metrics ── */
[data-testid="metric-container"] {
    background: #152b08;
    border: 1px solid #2d5016;
    border-radius: 8px;
    padding: 0.5rem 1rem;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="bov-header">
  <h1>🐄 BOVSMART</h1>
  <p>Sistema Inteligente de Contagem e Identificação de Bovinos com Inteligência Artificial</p>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar — settings
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Configurações")
    st.markdown("---")

    # ── Modo de câmera ──────────────────────────────────────────────────────
    st.markdown("#### 📷 Tipo de Filmagem")
    drone_mode = st.toggle(
        "Modo Drone (vista aérea)",
        value=True,
        help=(
            "Ativa o SAHI — divide cada frame em tiles sobrepostos e "
            "roda o YOLO em cada um. Essencial para detectar bovinos "
            "vistos de cima em alta resolução."
        ),
    )

    if drone_mode:
        st.success("🚁 SAHI ativado — detecção por tiles")
        imgsz = 1280  # not used in SAHI mode but kept for compatibility
        with st.expander("Configurações SAHI", expanded=False):
            tile_size = st.select_slider(
                "Tamanho do tile (px)",
                options=[320, 512, 640, 768, 1024],
                value=640,
                help="Tile menor = detecta animais menores, mas é mais lento.",
            )
            tile_overlap = st.slider(
                "Overlap entre tiles",
                min_value=0.10,
                max_value=0.40,
                value=0.20,
                step=0.05,
                format="%.2f",
                help="Overlap maior evita perder animais na borda dos tiles.",
            )
    else:
        st.info("📹 Modo solo — inferência padrão")
        tile_size = 640
        tile_overlap = 0.20
        imgsz = st.select_slider(
            "Resolução de inferência (imgsz)",
            options=[640, 1280, 1920],
            value=1280,
            help="Maior resolução detecta objetos menores, mas é mais lento.",
        )

    st.markdown("---")

    # ── Modelo ──────────────────────────────────────────────────────────────
    st.markdown("#### 🧠 Modelo de IA")
    use_custom = st.checkbox("Usar modelo customizado (.pt)", value=False)

    if use_custom:
        custom_model_path = st.text_input(
            "Caminho do modelo (.pt)",
            placeholder="C:/modelos/meu_modelo_drone.pt",
            help="Modelo treinado especificamente para vista aérea de bovinos.",
        )
        cow_class_id = st.number_input(
            "ID da classe boi no modelo",
            min_value=0,
            max_value=99,
            value=0,
            help="0 se o modelo foi treinado só com bovinos. 19 para COCO.",
        )
        model_key = None
    else:
        custom_model_path = None
        cow_class_id = COCO_COW_CLASS_ID
        model_key = st.selectbox(
            "Modelo YOLOv8",
            options=list(MODELS.keys()),
            index=2,
            help="Modelos maiores são mais precisos mas mais lentos.",
        )

    st.markdown("---")

    # ── Thresholds ──────────────────────────────────────────────────────────
    st.markdown("#### 🎯 Thresholds")
    confidence = st.slider(
        "Confiança mínima",
        min_value=0.10,
        max_value=0.90,
        value=0.25 if drone_mode else 0.40,
        step=0.05,
        format="%.2f",
        help="Drone: use 0.20–0.30. Solo: use 0.35–0.50.",
    )
    iou = st.slider(
        "IOU (supressão de duplicatas)",
        min_value=0.10,
        max_value=0.70,
        value=0.30 if drone_mode else 0.45,
        step=0.05,
        format="%.2f",
        help="Menor = remove mais caixas sobrepostas. Recomendado baixo para drone.",
    )

    st.markdown("---")

    preview_every = st.slider(
        "Preview a cada N frames",
        min_value=1,
        max_value=60,
        value=10,
        step=1,
    )

    st.markdown("---")
    st.markdown(
        "<small style='color:#5a8a3a'>BovSmart v2.0 · YOLOv8 + SAHI + ByteTrack</small>",
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Helper: save uploaded file to temp path
# ─────────────────────────────────────────────────────────────────────────────
def save_upload(uploaded_file) -> str:
    suffix = Path(uploaded_file.name).suffix
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(uploaded_file.read())
    tmp.flush()
    return tmp.name


# ─────────────────────────────────────────────────────────────────────────────
# Helper: process video and return output path + stats
# ─────────────────────────────────────────────────────────────────────────────
def process_video(
    input_path: str,
    detector: CattleDetector,
    preview_every: int,
    preview_placeholder,
    progress_bar,
    stats_placeholder,
) -> str:
    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out_path = tempfile.mktemp(suffix=".mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    frame_idx = 0
    t0 = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        annotated, count = detector.process_frame(frame)
        writer.write(annotated)

        # ── Live preview ────────────────────────────────────────────────────
        if frame_idx % preview_every == 0:
            rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            preview_placeholder.image(rgb, use_container_width=True, caption=f"Frame {frame_idx}")

            # Live stats
            elapsed = time.time() - t0
            fps_proc = frame_idx / elapsed if elapsed > 0 else 0
            eta = (total_frames - frame_idx) / fps_proc if fps_proc > 0 else 0
            with stats_placeholder.container():
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("🐄 Na tela", count)
                c2.metric("🔢 Total únicos", detector.stats.total_unique)
                c3.metric("📈 Máx simultâneos", detector.stats.max_simultaneous)
                c4.metric("⏱️ ETA", f"{eta:.0f}s")

        # ── Progress bar ─────────────────────────────────────────────────────
        if total_frames > 0:
            progress_bar.progress(min(frame_idx / total_frames, 1.0))

        frame_idx += 1

    cap.release()
    writer.release()
    progress_bar.progress(1.0)
    return out_path


# ─────────────────────────────────────────────────────────────────────────────
# Helper: build results charts
# ─────────────────────────────────────────────────────────────────────────────
def build_charts(frame_counts: list, fps: float):
    times = [i / fps for i in range(len(frame_counts))]

    # Animals over time line chart
    fig_line = go.Figure()
    fig_line.add_trace(go.Scatter(
        x=times,
        y=frame_counts,
        mode="lines",
        fill="tozeroy",
        line=dict(color="#4CAF50", width=2),
        fillcolor="rgba(76,175,80,0.2)",
        name="Bois visíveis",
    ))
    fig_line.update_layout(
        title="Bois visíveis ao longo do tempo",
        xaxis_title="Tempo (segundos)",
        yaxis_title="Quantidade",
        paper_bgcolor="#0f1a0f",
        plot_bgcolor="#0d160d",
        font=dict(color="#a8d890"),
        title_font=dict(color="#7fff5a"),
        xaxis=dict(gridcolor="#1e3a10"),
        yaxis=dict(gridcolor="#1e3a10"),
        margin=dict(l=40, r=20, t=50, b=40),
    )

    # Histogram of counts
    fig_hist = px.histogram(
        x=frame_counts,
        nbins=20,
        labels={"x": "Quantidade de bois", "y": "Frequência (frames)"},
        title="Distribuição — quantos bois aparecem por frame",
        color_discrete_sequence=["#4CAF50"],
    )
    fig_hist.update_layout(
        paper_bgcolor="#0f1a0f",
        plot_bgcolor="#0d160d",
        font=dict(color="#a8d890"),
        title_font=dict(color="#7fff5a"),
        xaxis=dict(gridcolor="#1e3a10"),
        yaxis=dict(gridcolor="#1e3a10"),
        margin=dict(l=40, r=20, t=50, b=40),
    )

    return fig_line, fig_hist


# ─────────────────────────────────────────────────────────────────────────────
# Main UI
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-title">📹 Upload do Vídeo</div>', unsafe_allow_html=True)

uploaded = st.file_uploader(
    "Arraste ou selecione um vídeo da fazenda",
    type=["mp4", "avi", "mov", "mkv"],
    help="Formatos suportados: MP4, AVI, MOV, MKV",
)

if uploaded:
    st.markdown("---")

    # Show original video
    col_orig, col_info = st.columns([2, 1])
    with col_orig:
        st.markdown('<div class="section-title">🎬 Vídeo Original</div>', unsafe_allow_html=True)
        st.video(uploaded)

    with col_info:
        st.markdown('<div class="section-title">📋 Arquivo</div>', unsafe_allow_html=True)
        size_mb = uploaded.size / (1024 * 1024)
        st.info(
            f"**Nome:** {uploaded.name}\n\n"
            f"**Tamanho:** {size_mb:.1f} MB\n\n"
            f"**Modelo:** {model_key}\n\n"
            f"**Confiança:** {confidence:.0%}"
        )
        st.markdown("")
        start_btn = st.button("🚀 Iniciar Análise com IA", use_container_width=True)

    # ── Processing ──────────────────────────────────────────────────────────
    if start_btn:
        st.markdown("---")
        st.markdown('<div class="section-title">🔄 Processando com IA...</div>', unsafe_allow_html=True)
        if drone_mode:
            st.caption(
                "Modo Drone ativo: SAHI divide cada frame em tiles sobrepostos, "
                "roda YOLOv8 em cada tile e mergeia os resultados. "
                "ByteTrack atribui IDs únicos persistentes a cada animal."
            )
        else:
            st.caption(
                "YOLOv8 detectando bovinos frame a frame. "
                "ByteTrack atribui IDs únicos a cada animal."
            )

        # Placeholders
        progress_bar = st.progress(0.0, text="Inicializando modelo...")
        preview_placeholder = st.empty()
        stats_placeholder = st.empty()

        # Save upload & init detector
        input_path = save_upload(uploaded)

        with st.spinner("Carregando modelo de IA (primeira vez faz download automático)..."):
            detector = CattleDetector(
                model_key=model_key,
                custom_model_path=custom_model_path or None,
                confidence=confidence,
                iou=iou,
                drone_mode=drone_mode,
                tile_size=tile_size,
                tile_overlap=tile_overlap,
                imgsz=imgsz if not drone_mode else 1280,
                cow_class_id=int(cow_class_id),
            )

        # Get FPS for charts
        cap_tmp = cv2.VideoCapture(input_path)
        fps = cap_tmp.get(cv2.CAP_PROP_FPS) or 25
        cap_tmp.release()

        progress_bar.progress(0.0, text="Analisando vídeo...")
        t_start = time.time()

        output_path = process_video(
            input_path=input_path,
            detector=detector,
            preview_every=preview_every,
            preview_placeholder=preview_placeholder,
            progress_bar=progress_bar,
            stats_placeholder=stats_placeholder,
        )

        elapsed = time.time() - t_start
        stats_placeholder.empty()
        preview_placeholder.empty()

        # ── Results ─────────────────────────────────────────────────────────
        st.markdown("---")
        st.markdown('<div class="section-title">✅ Resultado da Análise</div>', unsafe_allow_html=True)

        # Summary metrics
        st.markdown("""
        <div class="stat-grid">
        """, unsafe_allow_html=True)

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("🐄 Total Únicos", detector.stats.total_unique)
        m2.metric("📈 Máx. Simultâneos", detector.stats.max_simultaneous)
        m3.metric("📊 Média por Frame", f"{detector.stats.avg_per_frame:.1f}")
        m4.metric("🎞️ Frames Analisados", detector.stats.total_frames)
        m5.metric("⏱️ Tempo de Processamento", f"{elapsed:.1f}s")

        st.markdown("---")

        # Processed video
        col_vid, col_dl = st.columns([3, 1])
        with col_vid:
            st.markdown('<div class="section-title">🎬 Vídeo Processado</div>', unsafe_allow_html=True)
            with open(output_path, "rb") as f:
                video_bytes = f.read()
            st.video(video_bytes)

        with col_dl:
            st.markdown('<div class="section-title">💾 Download</div>', unsafe_allow_html=True)
            st.markdown("")
            st.markdown("")
            output_name = Path(uploaded.name).stem + "_bovsmart.mp4"
            st.download_button(
                label="⬇️ Baixar Vídeo Anotado",
                data=video_bytes,
                file_name=output_name,
                mime="video/mp4",
                use_container_width=True,
            )

            st.markdown("")
            st.markdown("**Legenda:**")
            st.markdown("- Cada cor = 1 animal único")
            st.markdown("- `Boi #N` = ID único do animal")
            st.markdown("- Linha tracejada = trajetória")
            st.markdown("- % = confiança da detecção")

        # Charts
        st.markdown("---")
        st.markdown('<div class="section-title">📊 Análise Temporal</div>', unsafe_allow_html=True)

        fig_line, fig_hist = build_charts(detector.stats.frame_counts, fps)

        ch1, ch2 = st.columns(2)
        with ch1:
            st.plotly_chart(fig_line, use_container_width=True)
        with ch2:
            st.plotly_chart(fig_hist, use_container_width=True)

        # Summary report
        st.markdown("---")
        st.markdown('<div class="section-title">📋 Relatório Final</div>', unsafe_allow_html=True)

        mode_str = f"Drone (SAHI tiles={tile_size}px, overlap={tile_overlap:.0%})" if drone_mode else "Solo (padrão)"
        report_lines = [
            f"## Relatório BovSmart — {uploaded.name}",
            f"",
            f"| Métrica | Valor |",
            f"|---|---|",
            f"| Total de bovinos únicos identificados | **{detector.stats.total_unique}** |",
            f"| Máximo simultâneo no quadro | **{detector.stats.max_simultaneous}** |",
            f"| Média de bovinos por frame | **{detector.stats.avg_per_frame:.1f}** |",
            f"| Frames analisados | **{detector.stats.total_frames}** |",
            f"| Modo de filmagem | **{mode_str}** |",
            f"| Modelo de IA utilizado | **{model_key or custom_model_path}** |",
            f"| Confiança mínima | **{confidence:.0%}** |",
            f"| IOU threshold | **{iou:.0%}** |",
            f"| Tempo de processamento | **{elapsed:.1f} segundos** |",
        ]
        st.markdown("\n".join(report_lines))

        # Cleanup temp files
        try:
            os.unlink(input_path)
        except Exception:
            pass

else:
    # Empty state
    st.markdown("")
    st.markdown("""
    <div style="text-align:center; padding: 3rem; color: #5a8a3a; border: 1px dashed #2d5016; border-radius: 12px; background: #0d1a0d;">
        <div style="font-size: 4rem;">🐄</div>
        <h3 style="color: #7fff5a;">Faça upload de um vídeo para começar</h3>
        <p>Suporte a MP4, AVI, MOV e MKV</p>
        <p style="font-size: 0.85rem; margin-top: 1rem; color: #3a6a20;">
            A IA irá detectar, identificar individualmente e contar todos os bovinos presentes no vídeo.<br>
            Cada animal recebe um ID único que persiste ao longo de todo o vídeo.
        </p>
    </div>
    """, unsafe_allow_html=True)
