import os

from faster_whisper import WhisperModel


def _load_model(model_size, device="auto", compute_type=None):
    """Load model and prefer CUDA when device=auto."""
    if device != "auto":
        chosen_compute = compute_type or ("float16" if device == "cuda" else "int8")
        model = WhisperModel(model_size, device=device, compute_type=chosen_compute)
        return model, device, chosen_compute

    try:
        chosen_compute = compute_type or "float16"
        model = WhisperModel(model_size, device="cuda", compute_type=chosen_compute)
        return model, "cuda", chosen_compute
    except Exception:
        chosen_compute = compute_type or "int8"
        model = WhisperModel(model_size, device="cpu", compute_type=chosen_compute)
        return model, "cpu", chosen_compute


def transcribe_audio(video_path, model_size=None, language="hi", device="auto"):
    """Transcribe with faster-whisper and include word timestamps for captioning."""
    selected_model = model_size or os.getenv("WHISPER_MODEL", "small")
    model, used_device, used_compute = _load_model(selected_model, device=device)

    print(
        f"Loading faster-whisper model='{selected_model}' on {used_device} ({used_compute})."
    )

    segments_iter, _ = model.transcribe(
        video_path,
        language=language,
        task="transcribe",
        word_timestamps=True,
        vad_filter=True,
        beam_size=5,
    )

    segments = []
    for seg in segments_iter:
        words = []
        if seg.words:
            for wd in seg.words:
                if wd.start is None or wd.end is None:
                    continue
                words.append(
                    {
                        "start": float(wd.start),
                        "end": float(wd.end),
                        "word": wd.word.strip(),
                    }
                )

        segments.append(
            {
                "start": float(seg.start),
                "end": float(seg.end),
                "text": seg.text.strip(),
                "words": words,
            }
        )

    print(f"Done. {len(segments)} segments transcribed.")
    return segments


if __name__ == "__main__":
    pass
