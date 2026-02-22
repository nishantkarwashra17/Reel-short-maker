import os

import streamlit as st

from analyzer import analyze_transcript
from downloader import download_video
from transcriber import transcribe_audio
from video_engine import create_short

st.set_page_config(page_title="AI Reel Short Maker", page_icon="🎬", layout="wide")

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
st.caption("Paste a YouTube link and generate vertical clips with AI captions.")

with st.sidebar:
    st.header("Settings")
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
        index=0,
        help="Cloud mode uses heavier AI defaults and is better for low-power PCs.",
    )

    quality_label = st.selectbox(
        "Output Quality",
        ["720p (Fast)", "1080p HD (Best)"],
        index=1 if runtime_mode == "Google Colab / Cloud" else 0,
    )

    whisper_model = st.selectbox(
        "Whisper Model",
        ["small", "medium", "large-v3"],
        index=1 if runtime_mode == "Google Colab / Cloud" else 0,
        help="Use medium/large-v3 in Colab GPU mode for better accuracy.",
    )

youtube_url = st.text_input(
    "YouTube Video URL", placeholder="https://www.youtube.com/watch?v=..."
)

if st.button("Generate Viral Clips", use_container_width=True):
    if not youtube_url.strip():
        st.error("Please enter a YouTube URL.")
    else:
        try:
            is_hd = quality_label.startswith("1080p")
            target_resolution = (1080, 1920) if is_hd else (720, 1280)
            render_profile = "quality" if is_hd else "fast"

            with st.status("Processing video...", expanded=True) as status:
                st.write("Downloading source video...")
                video_path = download_video(youtube_url, prefer_1080=is_hd)
                st.write(f"Downloaded: `{video_path}`")

                st.write("Transcribing with faster-whisper...")
                transcript = transcribe_audio(
                    video_path=video_path,
                    model_size=whisper_model,
                    language="hi",
                    device="auto",
                )
                st.write(f"Transcribed {len(transcript)} segments.")

                st.write("Selecting viral segments with Gemini...")
                viral_clips = analyze_transcript(transcript)
                st.write(f"Found {len(viral_clips)} candidate clips.")

                all_words = []
                for seg in transcript:
                    all_words.extend(seg.get("words", []))

                os.makedirs("downloads", exist_ok=True)
                processed_clips = []
                for i, clip in enumerate(viral_clips, start=1):
                    hook_text = clip.get("hook", f"Clip {i}")
                    st.write(f"Rendering clip {i}: {hook_text}")
                    output_name = f"downloads/clip_{i}.mp4"
                    try:
                        final_path = create_short(
                            video_path=video_path,
                            start_time=float(clip["start"]),
                            end_time=float(clip["end"]),
                            output_path=output_name,
                            transcript_words=all_words,
                            target_resolution=target_resolution,
                            render_profile=render_profile,
                        )
                        processed_clips.append((clip, final_path))
                    except Exception as clip_error:
                        st.warning(f"Clip {i} failed: {clip_error}")

                status.update(
                    label=f"Completed: {len(processed_clips)} clips generated.",
                    state="complete",
                )

            if processed_clips:
                st.header("Generated Clips")
                for i, (clip_info, clip_path) in enumerate(processed_clips, start=1):
                    col1, col2 = st.columns([2, 1])
                    with col1:
                        st.video(clip_path)
                    with col2:
                        st.subheader(f"Clip {i}")
                        st.write(f"Hook: {clip_info.get('hook', 'N/A')}")
                        st.write(f"Reason: {clip_info.get('reason', 'N/A')}")
                        with open(clip_path, "rb") as file_obj:
                            st.download_button(
                                label=f"Download Clip {i}",
                                data=file_obj,
                                file_name=f"viral_clip_{i}.mp4",
                                mime="video/mp4",
                                key=f"dl_{i}",
                            )
                    st.divider()
            else:
                st.error("No clips were generated. Check warnings above.")

        except Exception as err:
            st.error(f"Pipeline failed: {err}")
            st.exception(err)
