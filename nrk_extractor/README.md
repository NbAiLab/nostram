[<img align="right" width="150px" src="../images/nblogo.png">](https://ai.nb.no)
# NRK Extractor
The software here is a fork of code created by Njaal Borch for the Media Future Project. The code is released under an Apache 2.0-licence.

The code extracts both audio-files and sub-titles from NRK.

# JSON Lines Format
All keys are lowercase and used undercasing for space. The values should be in typical printing format, including utf-8 characters.

```bash

"id": "DNPR63700111_130664_131324" # The format is the pragramid_starttime(ms)_stoptime(ms).
"programid": "DNPR63700111" # Available as "id" in the episode meta-file
"start_time": 130664 # The time in ms from the start of the episode file.
"start_time": 131324 # The time in ms from the start of the episode file.
"source": "NRK TV" # Set manually for each source. 
"title": "Kråkeklubben" # Available from "preplay"->"titles"->"title"
"subtitle": "1. havet" # Available from "preplay"->"titles"->"title"
"availability_information": "Usually empty" # Available from "availability"->"information". If empty, key should be dropdded
"availability_isgeoblocked": "false" # Available from "availability"->"isGeoBlocked". Encoded as string here, "true" or "false"
"ondemand_from": "2012-03-21T18:21:00+01:00" # Available from "availability"->"onDemand"->"from".
"ondemand_to": "9999-12-31T00:00:00+01:00" # Available from "availability"->"onDemand"->"to".
"externalembeddingallowed": "true" # # Available from "availability"->"externalEmbeddingAllowed". Encoded as string here, "true" or "false"

"doc_type": "book" # The type of material. Newspaper or book
"ocr_date": "20191224" #Date for scanning in the format yyyymmdd. Set to N/A if not in mods post.
"publish_date": "20190101" # Date for publication. For books this is set to 0101 for the publication year. Set to N/A if not in mods post.
"language_reported": "nob" #3-letter language code. nob for Bokmål and nno for Nynorsk. Only reported for books in METS/ALTO. 
"language_detected": "nob" #3-letter language code
"tesseract_version": "4.1.1" #If Tesseract is used for scanning
"docworks_version": "6.5-1.28" #Text reported in METS/ALTO
"abbyy_version": "8.1" #Text reported in METS/ALTO
"document_word_confidence": 0.9 #Float 0-1. Average calculated while processing. 
"page": 1 #Page number - From in METS/ALTO - If documents are divided into one document per page
"paragraphs":   "paragraph_id": 1 #Integer. Starting on 0. Counted during processing.
                "page": 1 #Page number - From in METS/ALTO - if entire book is one document
                "block": 1 #Block number on current page - From in METS/ALTO
                "confidence": 0.36 #Float 0-1. From METS/ALTO
                "text": "text goes here" #utf8-encoded-text
```


## To Do List
