[<img align="right" width="150px" src="../images/nblogo.png">](https://ai.nb.no)
# NRK Extractor
The software here is a fork of code created by Njaal Borch for the Media Future Project. The code is released under an Apache 2.0-licence.

The code extracts both audio-files and sub-titles from NRK.

# JSON Lines Format
All keys are lowercase and used undercasing for space. The values should be in typical printing format, including utf-8 characters.

```bash
#Automatically calculated
"id": "DNPR63700111_130664_131324" # The format is "program_id" + "start_time" + "end_time".
"start_time": 130664 # The time in ms from the start of the episode file.
"end_time": 131324 # The time in ms from the start of the episode file.
"duration": 660 # Convenience field made from "end_time - start_time" in ms. 

#Calculated from meta-data
"program_id": "DNPR63700111" # Available as "id" in the episode meta-file.
"category_id": "barn" # Available from the season meta-file as "category"->"id".
"source": "NRK TV" # Set manually for each source. 
"title": "Kråkeklubben" # Available from "preplay"->"titles"->"title"
"subtitle": "1. havet" # Available from "preplay"->"titles"->"title"
"availability_information": "Usually empty" # Available from "availability"->"information". If empty, the key should be dropdded
"is_geoblocked": false # Available from "availability"->"isGeoBlocked". Boolean true or false.
"on_demand_from": "2012-03-21T18:21:00+01:00" # Available from "availability"->"onDemand"->"from".
"on_demand_to": "9999-12-31T00:00:00+01:00" # Available from "availability"->"onDemand"->"to".
"external_embedding_allowed": true # Available from "availability"->"externalEmbeddingAllowed". Boolean true or false.

#For subtitled version only
"subtitle_text": "the actual subtitle text" # Subtitle text if this is available. If empty, the key should be dropdded.

```


## To Do List
* Implement a generator class for getting all episodes in the same serie, as well as all related meta-data.
*  Add the extra meta data to the output json
*  Split the output in two different json files
*  Compare the voice detection with the software from Lasse Hansen.
