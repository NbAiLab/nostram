def remove_without_subtitles(mp4_folder, vtt_folder):
    # Remove any .mp4 files from mp4_folder that does not have a corresponding .vtt file in vtt_folder
    mp4_files = os.listdir(mp4_folder)
    vtt_files = set(os.listdir(vtt_folder))
    for mp4_file in mp4_files:
        if mp4_file.endswith('.mp4'):
            if mp4_file.replace('.mp4', '.vtt') not in vtt_files:
                os.remove(os.path.join(mp4_folder, mp4_file))
