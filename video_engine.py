import cv2
import numpy as np
from moviepy.editor import CompositeVideoClip, TextClip, VideoFileClip

try:
    import mediapipe as mp
except Exception:
    mp = None


def _get_render_params(render_profile):
    if render_profile == "quality":
        return {
            "preset": "medium",
            "bitrate": "8000k",
            "crf": "20",
            "sample_rate": 2.5,
            "threads": 4,
        }
    if render_profile == "balanced":
        return {
            "preset": "veryfast",
            "bitrate": "6000k",
            "crf": "22",
            "sample_rate": 1.8,
            "threads": 4,
        }
    return {
        "preset": "ultrafast",
        "bitrate": "4200k",
        "crf": "25",
        "sample_rate": 1.0,
        "threads": 2,
    }


def _notify(progress_callback, stage, pct):
    if progress_callback:
        progress_callback(stage, max(0, min(100, int(pct))))


def _build_phrase_captions(words, group_size=3):
    """Merge words into short phrases to reduce TextClip count and speed up rendering."""
    if not words:
        return []

    phrases = []
    bucket = []
    for wd in words:
        bucket.append(wd)
        if len(bucket) >= group_size:
            phrases.append(bucket)
            bucket = []

    if bucket:
        phrases.append(bucket)

    merged = []
    for chunk in phrases:
        start = chunk[0]["start"]
        end = chunk[-1]["end"]
        text = " ".join(item["word"] for item in chunk).strip()
        if text:
            merged.append({"start": start, "end": end, "word": text})
    return merged


def create_short(
    video_path,
    start_time,
    end_time,
    output_path,
    transcript_words,
    target_resolution=(720, 1280),
    render_profile="turbo",
    caption_mode="word",
    progress_callback=None,
):
    """Create vertical short with face-aware crop and captions."""
    render = _get_render_params(render_profile)

    _notify(progress_callback, "Opening clip", 2)

    detector = None
    mediapipe_ready = bool(mp and getattr(mp, "solutions", None))
    if mediapipe_ready:
        try:
            mp_face = mp.solutions.face_detection
            detector = mp_face.FaceDetection(model_selection=1, min_detection_confidence=0.5)
        except Exception:
            detector = None
            _notify(progress_callback, "MediaPipe init failed, using center crop", 5)
    else:
        _notify(progress_callback, "MediaPipe unavailable, using center crop", 5)

    source = VideoFileClip(video_path).subclip(start_time, end_time)
    src_w, src_h = source.size
    out_w, out_h = target_resolution

    crop_w = int(src_h * 9 / 16)
    crop_w = min(crop_w, src_w)

    duration = max(source.duration, 0.001)
    num_samples = max(int(duration * render["sample_rate"]), 1)
    sample_times = np.linspace(0, max(duration - 0.01, 0), num_samples)

    centers = []
    for idx, t in enumerate(sample_times):
        result = None
        if detector is not None:
            frame = source.get_frame(float(t))
            bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            result = detector.process(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

        if result is not None and result.detections:
            try:
                bbox = result.detections[0].location_data.relative_bounding_box
                cx = bbox.xmin + bbox.width / 2
                centers.append(float(np.clip(cx, 0.0, 1.0)))
            except Exception:
                centers.append(centers[-1] if centers else 0.5)
        else:
            centers.append(centers[-1] if centers else 0.5)

        _notify(progress_callback, "Tracking face", 5 + (idx + 1) * 30 / len(sample_times))

    if detector is not None:
        detector.close()

    if len(centers) > 2:
        kernel = np.ones(min(7, len(centers)), dtype=np.float32)
        kernel /= kernel.sum()
        centers = np.convolve(np.array(centers), kernel, mode="same")

    def crop_at_time(get_frame, t):
        frame = get_frame(t)
        idx = min(int((t / duration) * (len(centers) - 1)), len(centers) - 1)
        cx_px = int(float(centers[idx]) * src_w)
        x1 = max(0, min(src_w - crop_w, cx_px - crop_w // 2))
        crop = frame[:, x1 : x1 + crop_w]
        return cv2.resize(crop, (out_w, out_h), interpolation=cv2.INTER_LINEAR)

    _notify(progress_callback, "Cropping vertical frame", 38)
    cropped = source.fl(crop_at_time)

    caption_words = []
    for wd in transcript_words:
        if wd.get("start") is None or wd.get("end") is None:
            continue
        if wd["end"] <= start_time or wd["start"] >= end_time:
            continue
        caption_words.append(
            {
                "start": max(float(wd["start"]) - start_time, 0.0),
                "end": min(float(wd["end"]) - start_time, duration),
                "word": str(wd.get("word", "")).strip(),
            }
        )

    if caption_mode == "phrase":
        caption_words = _build_phrase_captions(caption_words, group_size=3)

    caption_layers = []
    total_caps = max(len(caption_words), 1)
    caption_render_failed = False
    for idx, wd in enumerate(caption_words):
        word_duration = wd["end"] - wd["start"]
        if word_duration <= 0.03 or not wd["word"]:
            continue

        try:
            txt = TextClip(
                wd["word"].upper(),
                fontsize=68 if out_w >= 1080 else 52,
                color="yellow",
                font="Liberation-Sans-Bold",
                stroke_color="black",
                stroke_width=3,
                method="caption",
                size=(out_w - 80, None),
            )
            txt = txt.set_start(wd["start"]).set_duration(word_duration)
            txt = txt.set_position(("center", int(out_h * 0.72)))
            caption_layers.append(txt)
        except Exception:
            # If ImageMagick policy blocks text rendering, continue without captions.
            caption_render_failed = True
            caption_layers = []
            break

        if idx % max(total_caps // 10, 1) == 0:
            _notify(progress_callback, "Building captions", 40 + (idx + 1) * 25 / total_caps)

    if caption_render_failed:
        _notify(progress_callback, "Caption engine unavailable, exporting without captions", 68)

    final = CompositeVideoClip([cropped] + caption_layers, size=(out_w, out_h))

    _notify(progress_callback, "Encoding video", 70)
    fps = source.fps or 24
    final.write_videofile(
        output_path,
        codec="libx264",
        audio_codec="aac",
        fps=fps,
        preset=render["preset"],
        bitrate=render["bitrate"],
        threads=render["threads"],
        logger=None,
        ffmpeg_params=["-crf", render["crf"]],
    )

    source.close()
    final.close()
    _notify(progress_callback, "Completed", 100)
    return output_path


if __name__ == "__main__":
    pass
