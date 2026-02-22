import json
import os

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
Analyze this transcript and identify the top 3-5 most viral segments for Reels/Shorts.

Selection criteria:
1. Strong hook
2. Self-contained segment
3. High retention potential
4. Length between 20 and 60 seconds

Transcript:
{text_with_times}

Return ONLY valid JSON array:
[
  {{"start": 10.5, "end": 45.2, "reason": "why this is viral", "hook": "opening hook"}}
]"""

    try:
        response = model.generate_content(prompt)
        clips = json.loads(_strip_json_block(response.text))
        for clip in clips:
            if "hook" not in clip:
                clip["hook"] = clip.get("reason", "Viral moment")[:60]
        return clips
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
