import os
import subprocess

import streamlit as st

from analyzer import analyze_transcript, generate_social_pack
from downloader import download_video
from transcriber import transcribe_audio
from video_engine import create_short


def _create_preview(input_path, output_path):
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        input_path,
        "-vf",
        "scale=-2:960",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "28",
        "-movflags",
        "+faststart",
        "-c:a",
        "aac",
        "-b:a",
        "128k",
        output_path,
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def _init_state():
    defaults = {
        "video_path": None,
        "transcript": None,
        "all_words": None,
        "clip_editor": [],
        "social_pack": {},
        "processed_clips": [],
        "analysis_done": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _video_duration_from_transcript(transcript):
    if not transcript:
        return 60.0
    return float(max(seg.get("end", 0.0) for seg in transcript))


st.set_page_config(page_title="AI Reel Short Maker", page_icon="🎬", layout="wide")
_init_state()

st.markdown(
    """
<style>
    .stApp { background: linear-gradient(135deg, #111827, #1f2937, #0f172a); }
    h1 { color: #f8fafc; }
    .stTextInput > div > div > input { background-color: #111827; color: #f8fafc; border: 1px solid #2563eb; border-radius: 10px; }
    .stButton > button { background: linear-gradient(90deg, #0ea5e9, #2563eb); color: white; border: none; border-radius: 10px; padding: 0.55rem 1.8rem; font-weight: 600; }
    .stDownloadButton > button { background: linear-gradient(90deg, #10b981, #059669); color: white; border: none; border-radius: 8px; }
</style>
""",
    unsafe_allow_html=True,
)

st.title("AI Reel and Short Maker")
st.caption("Analyze first, edit clips like Vizard, then render.")

with st.sidebar:
    st.header("Settings")
    existing_key = os.getenv("GOOGLE_API_KEY", "").strip()
    if existing_key:
        st.success("Gemini key loaded from server secret.")
        use_manual_key = st.checkbox("Override Gemini key manually", value=False)
    else:
        use_manual_key = True

    if use_manual_key:
        gemini_key = st.text_input(
            "Gemini API Key",
            type="password",
            help="Get one from https://aistudio.google.com",
        )
        if gemini_key:
            os.environ["GOOGLE_API_KEY"] = gemini_key

    runtime_mode = st.selectbox(
        "Runtime",
        ["Local PC", "Google Colab / Cloud"],
        index=1,
    )

    quality_label = st.selectbox(
        "Output Quality",
        ["720p (Fast)", "1080p HD (Best)"],
        index=1 if runtime_mode == "Google Colab / Cloud" else 0,
    )

    speed_mode_label = st.selectbox(
        "Render Speed Mode",
        ["Turbo (Fastest)", "Balanced", "Best Quality"],
        index=1 if runtime_mode == "Google Colab / Cloud" else 1,
    )

    caption_density = st.selectbox(
        "Caption Style",
        ["Live Word-by-Word", "Phrase (Faster)"],
        index=0,
    )

    whisper_model = st.selectbox(
        "Whisper Model",
        ["small", "medium", "large-v3"],
        index=1 if runtime_mode == "Google Colab / Cloud" else 0,
    )

    max_clips = st.slider("Initial AI Clips", min_value=1, max_value=8, value=4)

youtube_url = st.text_input("YouTube Video URL", placeholder="https://www.youtube.com/watch?v=...")

analyze_col, clear_col = st.columns([3, 1])
with analyze_col:
    analyze_clicked = st.button("1) Analyze Video", use_container_width=True)
with clear_col:
    clear_clicked = st.button("Reset", use_container_width=True)

if clear_clicked:
    for key in ["video_path", "transcript", "all_words", "clip_editor", "social_pack", "processed_clips", "analysis_done"]:
        st.session_state[key] = [] if key in ["clip_editor", "processed_clips"] else ({} if key == "social_pack" else None)
    st.session_state["analysis_done"] = False
    st.rerun()

if analyze_clicked:
    if not youtube_url.strip():
        st.error("Please enter a YouTube URL.")
    else:
        try:
            is_hd = quality_label.startswith("1080p")
            global_progress = st.progress(0, text="Starting analysis...")

            with st.status("Analyzing source video...", expanded=True):
                st.write("Downloading source video...")
                video_path = download_video(youtube_url, prefer_1080=is_hd)
                global_progress.progress(20, text="Video downloaded")

                st.write("Transcribing with faster-whisper...")
                transcript = transcribe_audio(
                    video_path=video_path,
                    model_size=whisper_model,
                    language="hi",
                    device="auto",
                )
                global_progress.progress(45, text="Transcription complete")

                st.write("Selecting AI clips...")
                viral_clips = analyze_transcript(transcript)[:max_clips]
                global_progress.progress(65, text="AI clip selection complete")

                st.write("Generating social pack...")
                social_pack = generate_social_pack(transcript, viral_clips)
                global_progress.progress(80, text="Social pack complete")

                all_words = []
                for seg in transcript:
                    all_words.extend(seg.get("words", []))

                clip_editor = []
                for clip in viral_clips:
                    clip_editor.append(
                        {
                            "enabled": True,
                            "start": float(clip.get("start", 0.0)),
                            "end": float(clip.get("end", 30.0)),
                            "hook": str(clip.get("hook", "")),
                            "reason": str(clip.get("reason", "")),
                            "score": int(clip.get("score", 50)),
                            "format": str(clip.get("format", "both")),
                            "type": str(clip.get("type", "general")),
                        }
                    )

                st.session_state.video_path = video_path
                st.session_state.transcript = transcript
                st.session_state.all_words = all_words
                st.session_state.clip_editor = clip_editor
                st.session_state.social_pack = social_pack
                st.session_state.processed_clips = []
                st.session_state.analysis_done = True

                global_progress.progress(100, text="Analysis done. Edit clips below.")

        except Exception as err:
            st.error(f"Analysis failed: {err}")
            st.exception(err)

if st.session_state.analysis_done:
    st.header("2) Edit Clips (Like Vizard)")
    duration = _video_duration_from_transcript(st.session_state.transcript)

    if st.button("Add Custom Clip"):
        st.session_state.clip_editor.append(
            {
                "enabled": True,
                "start": 0.0,
                "end": min(30.0, duration),
                "hook": "Custom clip",
                "reason": "Manual selection",
                "score": 60,
                "format": "both",
                "type": "general",
            }
        )
        st.rerun()

    for i, clip in enumerate(st.session_state.clip_editor):
        with st.expander(f"Clip {i+1}: {clip.get('hook', 'Clip')}", expanded=(i < 2)):
            c1, c2, c3 = st.columns([1, 1, 1])
            with c1:
                clip["enabled"] = st.checkbox("Include", value=clip.get("enabled", True), key=f"en_{i}")
            with c2:
                clip["start"] = st.number_input(
                    "Start (sec)",
                    min_value=0.0,
                    max_value=float(duration),
                    value=float(clip.get("start", 0.0)),
                    step=0.5,
                    key=f"st_{i}",
                )
            with c3:
                clip["end"] = st.number_input(
                    "End (sec)",
                    min_value=0.5,
                    max_value=float(duration),
                    value=float(clip.get("end", min(30.0, duration))),
                    step=0.5,
                    key=f"ed_{i}",
                )

            clip["hook"] = st.text_input("Hook", value=clip.get("hook", ""), key=f"hk_{i}")
            clip["reason"] = st.text_area("Reason", value=clip.get("reason", ""), key=f"rs_{i}", height=70)

            if clip["end"] <= clip["start"]:
                st.warning("End must be greater than start. This clip will be skipped if not fixed.")

    st.header("3) Render Edited Clips")
    render_clicked = st.button("Render Selected Clips", use_container_width=True)

    if render_clicked:
        try:
            is_hd = quality_label.startswith("1080p")
            target_resolution = (1080, 1920) if is_hd else (720, 1280)
            speed_map = {
                "Turbo (Fastest)": "turbo",
                "Balanced": "balanced",
                "Best Quality": "quality",
            }
            render_profile = speed_map[speed_mode_label]
            caption_mode = "word" if caption_density == "Live Word-by-Word" else "phrase"

            selected = [c for c in st.session_state.clip_editor if c.get("enabled", True) and c["end"] > c["start"]]
            if not selected:
                st.error("No valid clips selected.")
            else:
                os.makedirs("downloads", exist_ok=True)
                processed = []
                overall = st.progress(0, text="Starting render...")

                for i, clip in enumerate(selected, start=1):
                    st.write(f"Rendering clip {i}: {clip.get('hook', '')}")
                    clip_progress = st.progress(0, text=f"Clip {i}: starting")

                    def on_clip_progress(stage, pct):
                        clip_progress.progress(int(pct), text=f"Clip {i}: {stage}")

                    output_name = f"downloads/clip_{i}.mp4"
                    final_path = create_short(
                        video_path=st.session_state.video_path,
                        start_time=float(clip["start"]),
                        end_time=float(clip["end"]),
                        output_path=output_name,
                        transcript_words=st.session_state.all_words,
                        target_resolution=target_resolution,
                        render_profile=render_profile,
                        caption_mode=caption_mode,
                        progress_callback=on_clip_progress,
                    )

                    preview_path = f"downloads/clip_{i}_preview.mp4"
                    try:
                        _create_preview(final_path, preview_path)
                    except Exception:
                        preview_path = final_path

                    processed.append((clip, final_path, preview_path))
                    overall.progress(int((i / len(selected)) * 100), text=f"Rendered {i}/{len(selected)} clips")

                st.session_state.processed_clips = processed
                st.success(f"Done: {len(processed)} clips generated.")

        except Exception as err:
            st.error(f"Render failed: {err}")
            st.exception(err)

if st.session_state.processed_clips:
    st.header("Generated Clips")
    for i, (clip_info, clip_path, preview_path) in enumerate(st.session_state.processed_clips, start=1):
        col1, col2 = st.columns([2, 1])
        with col1:
            st.video(preview_path)
        with col2:
            st.subheader(f"Clip {i}")
            st.write(f"Hook: {clip_info.get('hook', 'N/A')}")
            st.write(f"Reason: {clip_info.get('reason', 'N/A')}")
            st.write(f"Viral score: {clip_info.get('score', 'N/A')}/100")
            st.write(f"Best for: {clip_info.get('format', 'both')}")
            st.write(f"Type: {clip_info.get('type', 'general')}")
            with open(clip_path, "rb") as file_obj:
                st.download_button(
                    label=f"Download Clip {i} (HQ)",
                    data=file_obj,
                    file_name=f"viral_clip_{i}.mp4",
                    mime="video/mp4",
                    key=f"dl_{i}",
                )
        st.divider()

if st.session_state.social_pack:
    st.header("AI Social Pack")
    yt = st.session_state.social_pack.get("youtube_shorts", {})
    ig = st.session_state.social_pack.get("instagram_reels", {})
    ideas = st.session_state.social_pack.get("content_ideas", {})

    st.subheader("YouTube Shorts")
    st.text_area("Title", value=yt.get("title", ""), height=70)
    st.text_area("Description", value=yt.get("description", ""), height=120)
    st.code(" ".join(yt.get("hashtags", [])) or "#shorts #viral")

    st.subheader("Instagram Reels")
    st.text_area("Caption", value=ig.get("caption", ""), height=120)
    st.code(" ".join(ig.get("hashtags", [])) or "#reels #viral")

    st.subheader("Hook, CTA, Background and Music Ideas")
    st.write("Hooks:")
    for item in ideas.get("hook_options", []):
        st.write(f"- {item}")

    st.write("CTA options:")
    for item in ideas.get("cta_options", []):
        st.write(f"- {item}")

    st.write("Background visual ideas:")
    for item in ideas.get("background_visual_ideas", []):
        st.write(f"- {item}")

    st.write("Music mood ideas:")
    for item in ideas.get("music_mood_ideas", []):
        st.write(f"- {item}")
