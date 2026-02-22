import google.generativeai as genai
import os
import json
from dotenv import load_dotenv

load_dotenv()

def analyze_transcript(transcript_data):
    """
    Uses Gemini AI to identify viral clips from the transcript.
    """
    api_key = os.getenv("GOOGLE_API_KEY", "")
    
    if not api_key:
        print("Warning: GOOGLE_API_KEY not found. Returning first 60 seconds as default.")
        max_end = min(60, transcript_data[-1]['end']) if transcript_data else 60
        return [{"start": 0, "end": max_end, "reason": "No API key", "hook": "Default clip (no API key)"}]

    genai.configure(api_key=api_key)

    # Build transcript text
    text_with_times = ""
    for seg in transcript_data:
        text_with_times += f"[{seg['start']:.1f}s - {seg['end']:.1f}s]: {seg['text']}\n"

    prompt = f"""You are an expert YouTube and Instagram growth strategist. 
Analyze this transcript and identify the top 3-5 most viral segments for Reels/Shorts.

Selection criteria:
1. Strong Hook - starts with curiosity gap or high-energy statement
2. Self-contained - makes sense on its own
3. High Retention - emotional moments, controversial takes, or high-value tips
4. Length - between 30 to 60 seconds

Transcript:
{text_with_times}

Return ONLY a valid JSON array, no extra text:
[
    {{"start": 10.5, "end": 45.2, "reason": "why this is viral", "hook": "the opening hook text"}}
]"""

    try:
        model = genai.GenerativeModel('gemini-2.5-flash')
        response = model.generate_content(prompt)
        json_text = response.text.strip()
        # Remove markdown code fences if present
        json_text = json_text.replace('```json', '').replace('```', '').strip()
        clips = json.loads(json_text)
        
        # Ensure every clip has a "hook" key
        for clip in clips:
            if 'hook' not in clip:
                clip['hook'] = clip.get('reason', 'Viral moment')[:50]
        
        return clips
    except Exception as e:
        print(f"Gemini error: {e}")
        max_end = min(60, transcript_data[-1]['end']) if transcript_data else 60
        return [{"start": 0, "end": max_end, "reason": str(e), "hook": "Auto-selected clip"}]


if __name__ == "__main__":
    pass
