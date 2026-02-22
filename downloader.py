import os

import yt_dlp


def download_video(url, output_path="downloads", prefer_1080=False):
    """Download a YouTube video as mp4 (optionally preferring 1080p inputs)."""
    os.makedirs(output_path, exist_ok=True)

    if prefer_1080:
        fmt = (
            "bestvideo[ext=mp4][height<=1080]+bestaudio[ext=m4a]/"
            "best[ext=mp4][height<=1080]/best"
        )
    else:
        fmt = "best[ext=mp4][height<=720]/best[ext=mp4]/best"

    ydl_opts = {
        "format": fmt,
        "outtmpl": f"{output_path}/%(id)s.%(ext)s",
        "merge_output_format": "mp4",
        "quiet": True,
        "no_warnings": True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        video_id = info.get("id")
        final_path = os.path.join(output_path, f"{video_id}.mp4")

        if os.path.exists(final_path):
            return final_path

        for file_name in os.listdir(output_path):
            if video_id in file_name:
                return os.path.join(output_path, file_name)

    raise FileNotFoundError(f"Could not find downloaded file for {video_id}")


if __name__ == "__main__":
    pass
