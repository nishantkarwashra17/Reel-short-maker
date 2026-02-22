import cv2
import mediapipe as mp
import numpy as np
from moviepy.editor import CompositeVideoClip, TextClip, VideoFileClip


def _get_render_params(render_profile):
    if render_profile == "quality":
        return {
            "preset": "medium",
            "bitrate": "8000k",
            "crf": "20",
        }
    return {
        "preset": "veryfast",
        "bitrate": "4500k",
        "crf": "24",
    }


def create_short(
    video_path,
    start_time,
    end_time,
    output_path,
    transcript_words,
    target_resolution=(720, 1280),
    render_profile="fast",
):
    """Create vertical short with face-aware crop and word-level captions."""
    mp_face = mp.solutions.face_detection
    detector = mp_face.FaceDetection(model_selection=1, min_detection_confidence=0.5)

    source = VideoFileClip(video_path).subclip(start_time, end_time)
    src_w, src_h = source.size
    out_w, out_h = target_resolution

    # 9:16 crop width derived from source height.
    crop_w = int(src_h * 9 / 16)
    crop_w = min(crop_w, src_w)

    duration = max(source.duration, 0.001)
    num_samples = max(int(duration * 3), 1)
    sample_times = np.linspace(0, max(duration - 0.01, 0), num_samples)

    centers = []
    for t in sample_times:
        frame = source.get_frame(float(t))
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        result = detector.process(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

        if result.detections:
            bbox = result.detections[0].location_data.relative_bounding_box
            cx = bbox.xmin + bbox.width / 2
            centers.append(float(np.clip(cx, 0.0, 1.0)))
        else:
            centers.append(centers[-1] if centers else 0.5)

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

    cropped = source.fl(crop_at_time)

    # Word timestamps are absolute timeline values, convert to local clip offsets.
    caption_layers = []
    for wd in transcript_words:
        if wd.get("start") is None or wd.get("end") is None:
            continue
        if wd["end"] <= start_time or wd["start"] >= end_time:
            continue

        local_start = max(float(wd["start"]) - start_time, 0.0)
        local_end = min(float(wd["end"]) - start_time, duration)
        word_duration = local_end - local_start
        if word_duration <= 0.03:
            continue

        word_text = str(wd.get("word", "")).strip()
        if not word_text:
            continue

        txt = TextClip(
            word_text.upper(),
            fontsize=68 if out_w >= 1080 else 52,
            color="yellow",
            font="Liberation-Sans-Bold",
            stroke_color="black",
            stroke_width=3,
            method="caption",
            size=(out_w - 80, None),
        )
        txt = txt.set_start(local_start).set_duration(word_duration)
        txt = txt.set_position(("center", int(out_h * 0.72)))
        caption_layers.append(txt)

    final = CompositeVideoClip([cropped] + caption_layers, size=(out_w, out_h))

    render = _get_render_params(render_profile)
    fps = source.fps or 24
    final.write_videofile(
        output_path,
        codec="libx264",
        audio_codec="aac",
        fps=fps,
        preset=render["preset"],
        bitrate=render["bitrate"],
        threads=0,
        logger=None,
        ffmpeg_params=["-crf", render["crf"]],
    )

    source.close()
    final.close()
    return output_path


if __name__ == "__main__":
    pass
