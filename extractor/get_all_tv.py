import traceback

import requests
import json
import re
import os
import subprocess
import time
import sys
import argparse
import isodate
import csv
import dateutil.parser
import jsonlines

##################################################################################
# Get all tv episodes and series - from category interface - creates a json file #
##################################################################################

HEADERS = {
    'User-Agent': 'National Library of Norway - AiLab - NoSTraM Project - User Agent v 1.0',
    'From': 'ai-lab@nb.no'
}


def main(args):
    error = 0
    valid_manifest = 0
    invalid_manifest = 0
    tv_file = os.path.join(args.output_path, "tv.json")

    with jsonlines.open(tv_file, mode='w') as writer:
        seconds = 0
        base_url = "https://psapi.nrk.no"

        r = requests.get(base_url + "/tv/pages/")
        if r.status_code != 200:
            raise Exception("Failed to load metadata from '%s'" % murl)

        categories_json = json.loads(r.text)

        other_categories = ["livsstil", "vitenskap", "kultur", "underholdning",
                            "unknown", "tegnspraak", "synstolk", "kvensk"]

        all_categories = [{"id": cat, "_links": {"self": {"href": f"/tv/pages/{cat}"}}} for cat in other_categories] \
                         + categories_json['pageListItems']

        processed_categories = set()
        processed_series = set()
        processed_programs = set()
        for category in all_categories:
            if category['id'] in processed_categories:
                continue
            print(f"\nProcessing Category {category['id']}")

            all_url = base_url + category['_links']['self']['href']
            all_json = get_json(all_url)

            if not all_json:
                continue

            for section in all_json['sections']:
                if "included" not in section:
                    continue
                print(f"\n-Processing Section {section['included']['title']}")

                for n, item in enumerate(section['included']['plugs']):
                    itemtype = item['targetType']

                    if 'series' == itemtype:
                        if item["series"]["seriesId"] in processed_series:
                            continue
                        print(
                            f"\n--#{n} Processing {itemtype} {item['displayContractContent']['contentTitle'].strip()}")

                        serie_json = get_json(base_url + item[item['targetType']]['_links']['self']['href'])
                        if not serie_json:
                            print("\n\n***ERROR Serie Json\n")
                            continue

                        if 'seasons' in serie_json:
                            for season in serie_json['seasons']:
                                print(f"\n--season - {season['name']}")
                                season_json = get_json(
                                    base_url + '/tv/catalog/series/' + serie_json['id'] + '/seasons/' + season['name'])

                                if not season_json:
                                    print("\n\n***ERROR Season Json\n")
                                    continue

                                if not 'episodes' in season_json['_embedded']:
                                    try:
                                        season_json['_embedded']['episodes'] = season_json['_embedded']['instalments']
                                    except:
                                        print("\n\n***ERROR Season Json Episodes\n")
                                        continue

                                for episode in season_json['_embedded']['episodes']:
                                    try:
                                        episode_seconds = write_episode(episode, writer, serie_json['category']['id'],
                                                                        serie_json['title'],
                                                                        serie_json['image']['webImages'][0]['imageUrl'])
                                    except:
                                        print("Failed because of unknown error")
                                        episode_seconds = 0
                                    # breakpoint()
                                    # episode_seconds = write_episode(episode,writer,season_json['image'][0]['url'])
                                    seconds += episode_seconds
                                    if episode_seconds:
                                        valid_manifest += 1
                                    else:
                                        invalid_manifest += 1

                        else:
                            print(
                                "This happens for radio, but I have not found it in TV. It is therefore untested. If it crashes here, uncomment, and restart")
                            breakpoint()

                            # for episode in serie_json['_embedded']['episodes']:
                            #    episode_seconds = write_episode(episode, writer, serie_json['series']['image'][0]['url'])
                            #    seconds += episode_seconds
                            #    if episode_seconds:
                            #        valid_manifest += 1
                            #    else:
                            #        invalid_manifest +=1
                        processed_series.add(item["series"]["seriesId"])


                    elif 'episode' == itemtype or 'standaloneProgram' == itemtype:
                        if item[itemtype]["programId"] in processed_programs:
                            continue
                        print(f"\n--{itemtype} - {item['displayContractContent']['contentTitle']}")
                        episode = get_json(base_url + item[itemtype]['_links']['self']['href'])
                        episode_seconds = write_episode(episode, writer)
                        seconds += episode_seconds
                        if episode_seconds:
                            valid_manifest += 1
                        else:
                            invalid_manifest += 1

                        processed_programs.add(item[itemtype]["programId"])

                    elif 'channel' == itemtype:
                        print("\n\n***ERROR Channel\n")
                        continue

                    else:
                        print("This should not happen!")
                        print(item)
                        # breakpoint()
            processed_categories.add(category['id'])

        print(f"\nTotal time: {round(seconds / 3600)} hours.")
        print(
            f"\nThere were a total of {valid_manifest} episodes with valid manifest files, and {invalid_manifest} episodes with an invalid one.")
        print(f"\nFinished writing json output file to {(tv_file)}")


def write_episode(episode, writer, category="Undefined", serie_title="Undefined", serie_image_url="None"):
    base_url = "https://psapi.nrk.no"
    is_standalone = "prfId" in episode
    episode_id = episode['prfId'] if is_standalone else episode["id"]

    medium = episode['_links']['self']['href'].split("/")[1]
    program_image_url = episode['image'][0]['url'] if is_standalone \
        else episode['image']['webImages'][0]['imageUrl']

    title = episode['titles']['title'] if is_standalone else episode["title"]
    subtitle = episode['titles']['subtitle'] if is_standalone else episode.get("subTitle", "")
    year = episode['productionYear']

    # Availability - Take this from the manifest-file since it is more accurate
    # Get the playback file
    url_path = episode['_links']['playbackmetadata']['href'] if is_standalone \
        else f"/playback/metadata/program/{episode_id}"
    playback_json = get_json(base_url + url_path)

    # Get the manifest-file
    # try:
    try:
        try:
            url_path = playback_json['_links']['manifests'][0]['href']
        except (KeyError, TypeError):
            url_path + f"/playback/manifest/program/{episode_id}"
        manifest_json = get_json(base_url + url_path)

        if not manifest_json['availability']['onDemand']['hasRightsNow']:
            raise ValueError("No rights")
        elif manifest_json['playability'] == 'nonPlayable':
            raise ValueError("Not playable")

        availability_information = manifest_json['availability']['information']
        is_geoblocked = manifest_json['availability']['isGeoBlocked']
        external_embedding_allowed = manifest_json['availability']['externalEmbeddingAllowed']
        duration = round(isodate.parse_duration(manifest_json['playable']['duration']).total_seconds() * 1000)
        audio_file = manifest_json['playable']['assets'][0]['url']
        audio_format = manifest_json['playable']['assets'][0]['format']
        audio_mime_type = manifest_json['playable']['assets'][0]['mimeType']
    except ValueError as e:
        # No manifest-file exists
        print('M', end='', flush=True)
        return 0
    except Exception as e:
        traceback.print_exc()
        print('M', end='', flush=True)
        return 0

    try:
        on_demand_from = manifest_json['availability']['onDemand']['from']
        on_demand_to = manifest_json['availability']['onDemand']['to']
    except:
        on_demand_from = "undefined"
        on_demand_to = "undefined"

    row = {'episode_id': episode_id,
           'medium': medium,
           'program_image_url': program_image_url,
           'serie_image_url': serie_image_url,
           'title': title,
           'subtitle': subtitle,
           'category': category,
           'serie_title': serie_title,
           'year': year,
           'duration': duration,
           'availability_information': availability_information,
           'is_geoblocked': is_geoblocked,
           'external_embedding_allowed': external_embedding_allowed,
           'on_demand_from': on_demand_from,
           'on_demand_to': on_demand_to,
           'audio_file': audio_file,
           'audio_format': audio_format,
           'audio_mime_type': audio_mime_type}

    writer.write(row)
    print('.', end='', flush=True)

    return duration


def get_json(url):
    r = requests.get(url, headers=HEADERS)
    if r.status_code != 200:
        return False

    return json.loads(r.text)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_path',
                        help="Complete path to json output file. The exact name will be given in the program.",
                        required=True)

    args = parser.parse_args()
    main(args)
