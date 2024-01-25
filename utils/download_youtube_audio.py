import yt_dlp

def download_audio(video_url, output_format='mp3'):
    options = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': output_format,
        }],
        'outtmpl': '%(title)s.%(ext)s',
    }

    with yt_dlp.YoutubeDL(options) as ydl:
        ydl.download([video_url])

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Download audio from a YouTube video.')
    parser.add_argument('--url', type=str, required=True, help='YouTube video URL')
    parser.add_argument('--format', type=str, default='mp3', help='Output audio format (default: mp3)')

    args = parser.parse_args()
    download_audio(args.url, args.format)

