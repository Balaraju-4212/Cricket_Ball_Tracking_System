import os
import sys
import tempfile
import cv2
import pandas as pd
import numpy as np
import streamlit as st
from tracker import (
    process_cricket_video,
    generate_sample_video,
    generate_3d_hawkeye_plot,
    generate_2d_pitch_map_plot,
    generate_2d_height_profile_plot
)


@st.cache_data(show_spinner=False)
def cached_process_video(video_path: str, conf_thresh: float, pitch_pixel_len: float) -> dict:
    """Cached wrapper around process_cricket_video for instant dashboard reloads."""
    import hashlib
    h = hashlib.md5(f"{video_path}_{conf_thresh}_{pitch_pixel_len}".encode()).hexdigest()[:8]
    out_file = f"outputs/tracked_delivery_{h}.mp4"
    return process_cricket_video(
        video_path=video_path,
        output_path=out_file,
        conf_thresh=conf_thresh,
        pitch_pixel_len=pitch_pixel_len
    )


def run_app():
    # Page Configuration & Custom CSS Styling
    st.set_page_config(page_title="Cricket AI Tracker & DRS Review", page_icon="🏏", layout="wide")

    st.markdown("""
    <style>
    .main-header { font-size: 2.2rem; font-weight: 700; color: #00D7FF; margin-bottom: 0px; }
    .sub-header { font-size: 1.0rem; color: #A0AAB5; margin-bottom: 20px; }
    .metric-card { background: #161B22; padding: 12px; border-radius: 8px; border: 1px solid #30363D; text-align: center; }
    .stVideo { border-radius: 10px; overflow: hidden; border: 2px solid #30363D; }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("<h1 class='main-header'>🏏 Cricket Ball AI Tracker & DRS Decision System</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-header'>Real-Time Computer Vision • Pitch Line Tracking • DRS Trajectory Extrapolation • 2D & 3D Hawkeye Graphs</p>", unsafe_allow_html=True)

    # Sidebar Navigation & Settings
    st.sidebar.header("📌 Navigation")
    has_metrics = 'metrics' in st.session_state

    if has_metrics:
        nav_page = st.sidebar.radio("Select View", ["📺 Video & DRS Review", "📊 Hawkeye Graphical Visualizations", "📑 Telemetry Dataset"])
    else:
        nav_page = "📺 Video Setup"
        st.sidebar.info("💡 Graph Visualizations unlock after analyzing a cricket delivery video.")

    st.sidebar.header("⚙️ Delivery Settings")
    video_option = st.sidebar.selectbox("Video Source", ["Default Sample", "Paste Video URL", "Upload Video", "Generate Clip"])
    conf_thresh = st.sidebar.slider("Confidence Threshold", 0.1, 0.9, 0.25, 0.05)
    pitch_pixels = 500.0  # Automatic AI pitch length estimation enabled

    # Input Video Handling
    video_path = "data/raw/cricket_video.mp4"

    if video_option == "Paste Video URL":
        url_input = st.sidebar.text_input("Enter Video URL (YouTube, MP4, WebM)", "")
        if url_input:
            video_path = url_input.strip()

    elif video_option == "Upload Video":
        uploaded = st.sidebar.file_uploader("Upload MP4/MOV/AVI Video", type=["mp4", "mov", "avi"])
        if uploaded:
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            tfile.write(uploaded.read())
            video_path = tfile.name

    elif video_option == "Generate Clip":
        video_path = generate_sample_video()
        st.sidebar.success("Generated sample video!")

    else:
        if not os.path.exists(video_path):
            video_path = generate_sample_video(video_path)

    # Action Bar for Manual Delivery Analysis Only
    st.markdown("<div style='background-color:#161B22; padding:15px; border-radius:10px; border:1px solid #30363D; margin-bottom:20px;'>", unsafe_allow_html=True)
    ctrl_col1, ctrl_col2 = st.columns([3, 1])
    with ctrl_col1:
        st.write(f"**Selected Input Source**: `{video_path}`")
    with ctrl_col2:
        run_btn = st.button("🚀 Analyze Delivery", type="primary", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # Process Video ONLY when the user clicks the Analyze button
    if run_btn:
        st.cache_data.clear()
        if 'metrics' in st.session_state:
            del st.session_state['metrics']

        progress_bar = st.progress(0.0)
        status_text = st.empty()

        def update_progress(pct):
            try:
                progress_bar.progress(float(pct))
                status_text.text(f"Tracking ball motion & rendering pitch lines... {int(pct * 100)}%")
            except Exception:
                pass

        try:
            import time as _time
            status_text.text("Analyzing video stream...")
            metrics = process_cricket_video(
                video_path=video_path,
                output_path=f"outputs/tracked_{int(_time.time())}.mp4",
                conf_thresh=conf_thresh,
                pitch_pixel_len=pitch_pixels,
                progress_callback=update_progress
            )
            status_text.success("Tracking & Analytics Complete! Hawkeye pitch graphs are now unlocked in the sidebar.")
            st.session_state['metrics'] = metrics
            st.rerun()
        except Exception as err:
            status_text.empty()
            if 'metrics' in st.session_state:
                del st.session_state['metrics']
            st.cache_data.clear()
            st.error(f"❌ Non-Cricket Video Rejected: {err}")
            st.warning("⚠️ No output video or telemetry dataset will be generated or stored for non-cricket videos.")

    # Display Page Views
    if 'metrics' in st.session_state:
        metrics = st.session_state['metrics']

        # ==========================================
        # PAGE 1: 📺 Video & DRS Review
        # ==========================================
        if nav_page == "📺 Video & DRS Review":
            st.subheader("🏆 DECISION REVIEW SYSTEM (DRS)")
            drs_col1, drs_col2, drs_col3, drs_col4 = st.columns(4)
            drs_col1.metric("1️⃣ PITCHING", metrics.get("drs_pitching", "IN-LINE"))
            drs_col2.metric("2️⃣ IMPACT", metrics.get("drs_impact", "IN-LINE"))
            drs_col3.metric("3️⃣ WICKETS", metrics.get("drs_wickets", "HITTING"))
            
            dec = metrics.get("drs_decision", "OUT")
            dec_color = "#8B0000" if "OUT" in dec and "NOT" not in dec else "#006400"
            drs_col4.markdown(f"<div style='background-color:{dec_color}; color:white; padding:12px; border-radius:8px; text-align:center; font-weight:bold; font-size:18px;'>{dec}<br><span style='font-size:11px;'>{metrics.get('drs_hit_detail', '')}</span></div>", unsafe_allow_html=True)

            st.markdown("---")
            col_v, col_m = st.columns([3, 2])
            with col_v:
                st.subheader("📺 Tracked Delivery Video Analysis")
                if os.path.exists(metrics["output_path"]):
                    with open(metrics["output_path"], "rb") as vf:
                        video_bytes = vf.read()
                    st.video(video_bytes, format="video/mp4")
                else:
                    st.error("Output video file not found.")

            with col_m:
                st.subheader("🏏 Telemetry & Flight Metrics")
                st.metric("Predicted Batsman Shot", metrics.get("predicted_shot", "Straight Drive"))
                st.metric("Pitch Bounce Length Zone", metrics.get("pitch_length_zone", "Good Length"))
                st.metric("Peak Ball Delivery Speed", f"{metrics['max_speed_kmh']} km/h")
                st.metric("Average Speed", f"{metrics['avg_speed_kmh']} km/h")

        # ==========================================
        # PAGE 2: 📊 Hawkeye Graphical Visualizations
        # ==========================================
        elif nav_page == "📊 Hawkeye Graphical Visualizations":
            st.subheader("📊 Hawkeye Analytics & Graphical Visualizations")
            
            if os.path.exists(metrics.get("csv_path", "")):
                try:
                    df_traj = pd.read_csv(metrics["csv_path"])
                    if "is_projected" in df_traj:
                        df_actual = df_traj[df_traj["is_projected"].astype(str).str.lower() == "false"]
                        df_proj = df_traj[df_traj["is_projected"].astype(str).str.lower() == "true"]
                    else:
                        df_actual = df_traj
                        df_proj = pd.DataFrame()

                    pts = list(zip(df_actual["x_pixel"], df_actual["y_pixel"])) if "x_pixel" in df_actual and "y_pixel" in df_actual else []
                    drs_dict = {
                        "projected_points": list(zip(df_proj["x_pixel"], df_proj["y_pixel"])) if "x_pixel" in df_proj and "y_pixel" in df_proj else []
                    }
                    
                    tab1, tab2, tab3 = st.tabs([
                        "🏏 2D Pitch Top-Down Map",
                        "📈 Ball Height Profile Arc",
                        "📐 Hawkeye 3D Parabola"
                    ])

                    with tab1:
                        fig_2d = generate_2d_pitch_map_plot(pts, metrics.get("bounce_point"), metrics.get("impact_point"), drs_dict)
                        if fig_2d:
                            st.plotly_chart(fig_2d, use_container_width=True)
                        else:
                            st.info("2D Pitch Map graph unavailable.")

                    with tab2:
                        fig_height = generate_2d_height_profile_plot(pts, metrics.get("bounce_point"), drs_dict)
                        if fig_height:
                            st.plotly_chart(fig_height, use_container_width=True)
                        else:
                            st.info("Ball Height Profile graph unavailable.")

                    with tab3:
                        fig_3d = generate_3d_hawkeye_plot(pts, metrics.get("bounce_point"), drs_dict)
                        if fig_3d:
                            st.plotly_chart(fig_3d, use_container_width=True)
                        else:
                            st.info("3D Hawkeye visualization unavailable.")

                except Exception as err:
                    st.error(f"Error generating Hawkeye graphs: {err}")
            else:
                st.info("No telemetry CSV dataset available.")

        # ==========================================
        # PAGE 3: 📑 Telemetry Dataset
        # ==========================================
        elif nav_page == "📑 Telemetry Dataset":
            st.subheader("📑 Telemetry Dataset (.csv)")
            if os.path.exists(metrics.get("csv_path", "")):
                df = pd.read_csv(metrics["csv_path"])
                csv_bytes = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Telemetry CSV Report",
                    data=csv_bytes,
                    file_name="ball_tracking_telemetry.csv",
                    mime="text/csv",
                    type="primary"
                )
                st.dataframe(df, use_container_width=True, height=450)
            else:
                st.info("No telemetry dataset available.")
    else:
        st.markdown("---")
        st.markdown("### 🏏 Cricket AI Verification & Usage Guide")
        
        guide_col1, guide_col2 = st.columns(2)
        with guide_col1:
            st.markdown("""
            <div style='background-color:#161B22; padding:18px; border-radius:10px; border:1px solid #30363D;'>
                <h4 style='color:#00D7FF; margin-top:0;'>1️⃣ How to Run Cricket Ball Tracking</h4>
                <ol style='color:#D0D7DE; line-height:1.7;'>
                    <li>Select a video source in the sidebar (<b>Default Sample</b>, <b>Upload Video</b>, or <b>Paste Video URL</b>).</li>
                    <li>Click the red <b>🚀 Analyze Delivery</b> button to run deep computer vision tracking.</li>
                    <li>Once analyzed, full Hawkeye pitch graphs (2D Pitch Map, Height Profile Arc, and 3D Hawkeye Parabola) will unlock in the sidebar navigation.</li>
                </ol>
            </div>
            """, unsafe_allow_html=True)

        with guide_col2:
            st.markdown("""
            <div style='background-color:#161B22; padding:18px; border-radius:10px; border:1px solid #30363D;'>
                <h4 style='color:#00D7FF; margin-top:0;'>🛡️ Automatic Non-Cricket Video Rejection</h4>
                <ul style='color:#D0D7DE; line-height:1.7;'>
                    <li>❌ <b>Rejected Immediately</b>: PyTorch Neural Network detects that there is no cricket pitch or bowling motion.</li>
                    <li>🚫 <b>No Output Files Stored</b>: System cancels processing before writing any video or storing any metrics.</li>
                    <li>⚠️ <b>Clear Alert</b>: Displays <code>❌ Non-Cricket Video Rejected: This video does not contain a cricket pitch or match field. Ball tracking and DRS analysis are disabled for non-cricket videos.</code></li>
                </ul>
            </div>
            """, unsafe_allow_html=True)



if __name__ == "__main__":
    import os
    import sys
    import subprocess

    # Detect if we are already running inside a streamlit process
    is_streamlit = os.environ.get("STREAMLIT_RUNNING") == "1" or "streamlit" in sys.argv[0]

    if is_streamlit:
        os.environ["STREAMLIT_RUNNING"] = "1"
        run_app()
    else:
        os.environ["STREAMLIT_RUNNING"] = "1"
        print("[INFO] Launching Streamlit Web App in browser...")
        try:
            subprocess.run(["streamlit", "run", "app.py"])
        except KeyboardInterrupt:
            print("\n[INFO] Stopped Streamlit App.")
