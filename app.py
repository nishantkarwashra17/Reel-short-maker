import os

import streamlit as st

from analyzer import analyze_transcript, generate_social_pack
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
        index=1,
        help="Cloud mode is recommended for low-power PCs.",
    )

    quality_label = st.selectbox(
        "Output Quality",
        ["720p (Fast)", "1080p HD (Best)"],
        index=0,
    )

    speed_mode_label = st.selectbox(
        "Render Speed Mode",
        ["Turbo (Fastest)", "Balanced", "Best Quality"],
        index=0 if runtime_mode == "Google Colab / Cloud" else 1,
        help="Turbo is much faster. Best Quality is slower.",
    )

    whisper_model = st.selectbox(
        "Whisper Model",
        ["small", "medium", "large-v3"],
        index=1 if runtime_mode == "Google Colab / Cloud" else 0,
        help="medium is usually the best speed/quality mix on Colab GPU.",
    )

    max_clips = st.slider("Max Clips", min_value=1, max_value=5, value=3)

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

            speed_map = {
                "Turbo (Fastest)": "turbo",
                "Balanced": "balanced",
                "Best Quality": "quality",
            }
            render_profile = speed_map[speed_mode_label]
            caption_mode = "phrase" if render_profile == "turbo" else "word"

            global_progress = st.progress(0, text="Starting pipeline...")
            completed_steps = 0

            with st.status("Processing video...", expanded=True) as status:
                st.write("Downloading source video...")
                video_path = download_video(youtube_url, prefer_1080=is_hd)
                completed_steps += 1
                global_progress.progress(10, text="Video downloaded")
                st.write(f"Downloaded: `{video_path}`")

                st.write("Transcribing with faster-whisper...")
                transcript = transcribe_audio(
                    video_path=video_path,
                    model_size=whisper_model,
                    language="hi",
                    device="auto",
                )
                completed_steps += 1
                global_progress.progress(25, text="Transcription complete")
                st.write(f"Transcribed {len(transcript)} segments.")

                st.write("Selecting viral segments with Gemini...")
                viral_clips = analyze_transcript(transcript)
                viral_clips = viral_clips[:max_clips]
                completed_steps += 1
                global_progress.progress(35, text="Clip selection complete")
                st.write(f"Selected {len(viral_clips)} clips.")

                st.write("Generating social content pack...")
                social_pack = generate_social_pack(transcript, viral_clips)
                completed_steps += 1
                global_progress.progress(45, text="Social content pack ready")

                all_words = []
                for seg in transcript:
                    all_words.extend(seg.get("words", []))

                os.makedirs("downloads", exist_ok=True)
                processed_clips = []

                total_steps = max(1, 4 + len(viral_clips))
                for i, clip in enumerate(viral_clips, start=1):
                    hook_text = clip.get("hook", f"Clip {i}")
                    st.write(f"Rendering clip {i}: {hook_text}")

                    clip_progress = st.progress(0, text=f"Clip {i}: starting")

                    def on_clip_progress(stage, pct):
                        clip_progress.progress(int(pct), text=f"Clip {i}: {stage}")

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
                            caption_mode=caption_mode,
                            progress_callback=on_clip_progress,
                        )
                        processed_clips.append((clip, final_path))
                        clip_progress.progress(100, text=f"Clip {i}: done")
                    except Exception as clip_error:
                        st.warning(f"Clip {i} failed: {clip_error}")

                    completed_steps += 1
                    pct = int((completed_steps / total_steps) * 100)
                    global_progress.progress(min(pct, 99), text=f"Overall progress: {min(pct, 99)}%")

                global_progress.progress(100, text="All done")
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

                st.header("AI Social Pack")
                yt = social_pack.get("youtube_shorts", {})
                ig = social_pack.get("instagram_reels", {})
                ideas = social_pack.get("content_ideas", {})

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
            else:
                st.error("No clips were generated. Check warnings above.")

        except Exception as err:
            st.error(f"Pipeline failed: {err}")
            st.exception(err)
