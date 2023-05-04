import argparse
import json
from concurrent.futures import ThreadPoolExecutor

import requests
import os

from tqdm import tqdm

HEADERS = {
    'User-Agent': 'National Library of Norway - AiLab - NoSTraM Project - User Agent v 1.0',
    'From': 'ai-lab@nb.no'
}


def main(tv_json, vtt_folder, manifest_folder):
    with requests.Session() as session:
        session.headers.update(HEADERS)
        with open(tv_json) as tv:

            # Helper function for threading
            def handle(episode_id):
                manifest_path = os.path.join(manifest_folder, f"{episode_id}_manifest.json")

                exists = os.path.isfile(manifest_path)
                if exists:  # No need to redo the request, just load the file
                    with open(manifest_path, "r") as f:
                        try:
                            manifest = json.load(f)
                        except json.JSONDecodeError:
                            manifest = {}
                else:
                    resp = session.get(f"https://psapi.nrk.no/playback/manifest/program/{episode_id}")
                    manifest = resp.json()
                    if isinstance(manifest, dict) and manifest_folder is not None:
                        with open(manifest_path, "w") as out:
                            json.dump(manifest, out)

                # Sometimes subtitles may be missing, producing different errors depending on the format
                try:
                    subtitles = manifest["playable"]["subtitles"]
                except (AttributeError, KeyError, TypeError) as error:
                    if exists:  # Existed before
                        os.remove(manifest_path)
                        handle(episode_id)
                    else:
                        it.set_postfix_str(f"{episode_id=}, {error=}")
                    return

                for subtitle_info in subtitles:
                    # Display programs of interest if they occur
                    if subtitle_info["type"] not in ("ttv", "nor") \
                            or subtitle_info["language"] not in ("nb",) \
                            or subtitle_info["label"] not in ("Norsk", "Norsk - på all tale", "Norsk – på all tale"):
                        print(episode_id, subtitle_info)

                    subtitle_code = subtitle_info["type"]
                    web_vtt = subtitle_info["webVtt"]

                    out_path = os.path.join(vtt_folder, f"{episode_id}_{subtitle_code}.vtt")

                    # If VTT already exists there's no point doing the request again
                    if not os.path.isfile(out_path):
                        resp = session.get(web_vtt)
                        if 200 <= resp.status_code < 300:
                            with open(out_path, "w") as out:
                                out.write(resp.text)
                    it.set_postfix_str(f"{episode_id=}, {subtitle_code=}")

            episode_ids = sorted(set(json.loads(line)["episode_id"] for line in tv))
            # Makes everything way faster
            it = tqdm(episode_ids, "Episodes processed")
            for ep_id in it:
                handle(ep_id)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--tv_json", help="JSON-lines file containing metadata for programs "
                                          "(specifically it just needs an `episode_id` field)")
    parser.add_argument("--vtt_folder", help="Output folder for .vtt files")
    parser.add_argument("--manifest_folder", required=False, help="Output folder for manifest (.json) files")

    args = parser.parse_args()
    main(args.tv_json, args.vtt_folder, args.manifest_folder)
