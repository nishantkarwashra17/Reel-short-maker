# AI Reel and Short Maker

Transform any YouTube video into vertical clips with AI clip selection and captions.

## Local Run

1. Install requirements:

```bash
pip install -r requirements.txt
```

2. Start app:

```bash
streamlit run app.py
```

3. In the app sidebar, paste your Gemini API key and YouTube URL.

## Google Colab (Recommended for low-power PCs)

Use Colab when your PC is slow and you want faster processing with better quality.

1. Open a new notebook in Colab.
2. Set runtime to GPU: `Runtime > Change runtime type > T4/A100 GPU`.
3. Run these cells:

```bash
!git clone <YOUR_GITHUB_REPO_URL> reel-short-maker
%cd reel-short-maker
!pip install -U pip
!pip install -r requirements.txt
!apt-get update -y && apt-get install -y ffmpeg imagemagick fonts-liberation
```

```bash
!python -m streamlit run app.py --server.port 7860 --server.address 0.0.0.0 --server.headless true
```

4. Expose the Streamlit URL from Colab using your preferred tunnel (Cloudflare/ngrok/localtunnel).

Cloud defaults to use in sidebar:
- Runtime: `Google Colab / Cloud`
- Output Quality: `1080p HD (Best)`
- Whisper Model: `medium` or `large-v3`

## GitHub Connection (this folder is not connected yet)

A helper script is included to initialize git and connect this project to GitHub.

```powershell
./github_connect.ps1 -RemoteUrl "https://github.com/<username>/<repo>.git"
```

The script will:
- Initialize git if missing
- Create first commit (or update commit)
- Set `origin`
- Push to `main`

If git asks for name/email, set once:

```powershell
git config --global user.name "Your Name"
git config --global user.email "you@example.com"
```
