import json
import os
import re

import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()


def _strip_json_block(text):
    return text.strip().replace("```json", "").replace("```", "").strip()


def _get_model():
    api_key = os.getenv("GOOGLE_API_KEY", "")
    if not api_key:
        return None
    genai.configure(api_key=api_key)
    return genai.GenerativeModel("gemini-2.5-flash")


def _build_transcript_text(transcript_data, max_chars=30000):
    lines = []
    for seg in transcript_data:
        lines.append(f"[{seg['start']:.1f}s - {seg['end']:.1f}s]: {seg['text']}")
    text = "\n".join(lines)
    return text[:max_chars]


def _extract_clip_text(transcript_data, start, end):
    parts = []
    for seg in transcript_data:
        if seg["end"] <= start or seg["start"] >= end:
            continue
        text = str(seg.get("text", "")).strip()
        if text:
            parts.append(text)
    return " ".join(parts).strip()


def _safe_clip(clip, default_start=0.0, default_end=30.0):
    try:
        start = float(clip.get("start", default_start))
        end = float(clip.get("end", default_end))
    except Exception:
        start, end = default_start, default_end

    if end <= start:
        end = start + 20.0

    duration = end - start
    if duration < 15:
        end = start + 15
    elif duration > 70:
        end = start + 70

    score_raw = clip.get("score", clip.get("viral_score", 50))
    try:
        score = int(round(float(score_raw)))
    except Exception:
        score = 50
    score = max(1, min(100, score))

    return {
        "start": round(start, 2),
        "end": round(end, 2),
        "reason": str(clip.get("reason", "Strong segment")),
        "hook": str(clip.get("hook", "Viral moment"))[:120],
        "score": score,
        "format": str(clip.get("format", "both")).lower(),
        "type": str(clip.get("type", "general")).lower(),
    }


def _overlap_seconds(a, b):
    return max(0.0, min(a["end"], b["end"]) - max(a["start"], b["start"]))


def _dedupe_clips(clips, target_count=3, min_gap_sec=8.0):
    """
    Keep clips that are not near-duplicates in timeline.
    """
    normalized = [_safe_clip(c) for c in clips]
    normalized.sort(key=lambda c: c["score"], reverse=True)

    kept = []
    for clip in normalized:
        is_duplicate = False
        for existing in kept:
            overlap = _overlap_seconds(clip, existing)
            min_len = max(1.0, min(clip["end"] - clip["start"], existing["end"] - existing["start"]))
            if overlap / min_len > 0.45:
                is_duplicate = True
                break
            if abs(clip["start"] - existing["start"]) < min_gap_sec:
                is_duplicate = True
                break
        if not is_duplicate:
            kept.append(clip)
        if len(kept) >= target_count:
            break
    kept.sort(key=lambda c: c["start"])
    return kept


def _local_viral_score(text, duration_sec):
    """
    Lightweight heuristic score (1-100) to stabilize Gemini quality.
    """
    if not text:
        return 35

    t = text.lower()
    score = 40

    # Hook patterns (english + hindi cues)
    hook_terms = [
        "why",
        "how",
        "secret",
        "mistake",
        "truth",
        "shocking",
        "wait",
        "stop",
        "क्यों",
        "कैसे",
        "गलती",
        "सच",
        "राज",
        "रुको",
    ]
    for term in hook_terms:
        if term in t:
            score += 4

    # Curiosity/retention punctuation
    score += min(t.count("?") * 3, 12)
    score += min(t.count("!") * 2, 8)

    # Numbers/lists often perform better in short-form
    if re.search(r"\b\d+\b", t):
        score += 6

    # Avoid flat/too short text
    words = t.split()
    if len(words) < 12:
        score -= 8
    elif len(words) > 50:
        score += 4

    # Duration sweet spot for shorts
    if 22 <= duration_sec <= 50:
        score += 10
    elif duration_sec > 65:
        score -= 8

    return max(1, min(100, int(score)))


def _rerank_candidates(clips, transcript_data):
    reranked = []
    for clip in clips:
        start, end = clip["start"], clip["end"]
        duration = max(1.0, end - start)
        clip_text = _extract_clip_text(transcript_data, start, end)
        local_score = _local_viral_score(clip_text, duration)
        blended = int(round((0.7 * clip["score"]) + (0.3 * local_score)))
        clip["score"] = max(1, min(100, blended))
        reranked.append(clip)
    reranked.sort(key=lambda c: c["score"], reverse=True)
    return reranked


def analyze_transcript(transcript_data):
    """
    Uses Gemini AI to identify viral clips from the transcript.
    """
    model = _get_model()
    if model is None:
        print("Warning: GOOGLE_API_KEY not found. Returning first 60 seconds as default.")
        max_end = min(60, transcript_data[-1]["end"]) if transcript_data else 60
        return [
            {
                "start": 0,
                "end": max_end,
                "reason": "No API key",
                "hook": "Default clip (no API key)",
            }
        ]

    text_with_times = _build_transcript_text(transcript_data)
    prompt = f"""You are an expert YouTube and Instagram growth strategist.
Analyze this transcript and identify the top 10 most viral, DISTINCT segments for Reels/Shorts.

Selection criteria:
1. Strong hook
2. Self-contained segment
3. High retention potential
4. Length between 20 and 60 seconds
5. Segments must be from different moments, not overlapping repeats
6. Include a score from 1-100 (100 = strongest viral potential)
7. Prefer moments with clear tension, payoff, conflict, transformation, or highly actionable tips

Transcript:
{text_with_times}

Return ONLY valid JSON array with keys: start,end,reason,hook,score,format,type
format must be one of: "shorts", "reels", "both"
type must be one of: "story", "tip", "controversy", "emotion", "general"
[
  {{"start": 10.5, "end": 45.2, "reason": "why this is viral", "hook": "opening hook", "score": 88, "format": "both", "type": "story"}}
]"""

    try:
        response = model.generate_content(prompt)
        clips = json.loads(_strip_json_block(response.text))
        for clip in clips:
            if "hook" not in clip:
                clip["hook"] = clip.get("reason", "Viral moment")[:60]
        normalized = [_safe_clip(c) for c in clips]
        reranked = _rerank_candidates(normalized, transcript_data)
        return _dedupe_clips(reranked, target_count=5)
    except Exception as err:
        print(f"Gemini error: {err}")
        max_end = min(60, transcript_data[-1]["end"]) if transcript_data else 60
        return [
            {
                "start": 0,
                "end": max_end,
                "reason": str(err),
                "hook": "Auto-selected clip",
            }
        ]


def generate_social_pack(transcript_data, viral_clips):
    """
    Generate captions/hashtags/description/background ideas for Shorts and Reels.
    """
    model = _get_model()
    fallback = {
        "youtube_shorts": {
            "title": "Must watch moment",
            "description": "Watch till end for the main takeaway.",
            "hashtags": ["#shorts", "#viral", "#trending"],
        },
        "instagram_reels": {
            "caption": "This part changed everything.",
            "hashtags": ["#reels", "#explore", "#viralreels"],
        },
        "content_ideas": {
            "hook_options": ["Wait for the ending."],
            "cta_options": ["Follow for more daily clips."],
            "background_visual_ideas": ["High-contrast kinetic text overlay."],
            "music_mood_ideas": ["High-energy cinematic beat."],
        },
    }

    if model is None:
        return fallback

    transcript_text = _build_transcript_text(transcript_data, max_chars=18000)
    clips_text = json.dumps(viral_clips[:5], ensure_ascii=True)

    prompt = f"""You are a social media growth assistant.
Generate high-conversion copy for YouTube Shorts and Instagram Reels.

Transcript:
{transcript_text}

Selected clips:
{clips_text}

Rules:
- Keep language simple and punchy.
- Give hashtags that are broad + niche mix.
- No fake claims.
- Return ONLY valid JSON.

Schema:
{{
  "youtube_shorts": {{
    "title": "max 80 chars",
    "description": "2-4 lines with CTA",
    "hashtags": ["#shorts", "#topic"]
  }},
  "instagram_reels": {{
    "caption": "2-4 lines with CTA",
    "hashtags": ["#reels", "#topic"]
  }},
  "content_ideas": {{
    "hook_options": ["hook 1", "hook 2", "hook 3"],
    "cta_options": ["cta 1", "cta 2", "cta 3"],
    "background_visual_ideas": ["idea 1", "idea 2", "idea 3"],
    "music_mood_ideas": ["mood 1", "mood 2", "mood 3"]
  }}
}}"""

    try:
        response = model.generate_content(prompt)
        pack = json.loads(_strip_json_block(response.text))
        return pack
    except Exception as err:
        print(f"Gemini social pack error: {err}")
        return fallback


if __name__ == "__main__":
    pass
