## Structure

Please note that for some of the source corpora, specified test and validation splits exist. In these cases, they are treated as separate source corpora.

### Source_1

The raw dump of incoming data.

### Json_2

Structured format. The main transformation here involves converting everything into a semi-structured JSON format.

### Clean_3

Since the cleaning process varies significantly across different corpora, this stage is divided into separate directories for each source corpus. For example, the NRK corpus requires an additional `clean.json` file. All comparisons that do not require external processing (i.e., inference) should be performed here.

- **Subfolders**: Source-corpus (e.g., `nrk_tv_transcribe`, `nrk_tv_translate`, `nrk_tv_veryshort`, `nrk_tv_silence`, `stortinget`, `fleurs`, `nst`, etc.)
- **Level**: If needed, specify levels (e.g., `3a`, `3b`, `3c`)

> **Run complete validation tests on all JSON files.**

### Inference_4 (formerly `transcribe_json_4`)

- **Inference_dataset**: Directly copied from `clean_json_3`. Note that we will also ensure that the necessary splits are performed here. Can be used directly for building Hugging Face datasets.
- **Inference_results**: All output from inference on `inference_dataset`.
- **Processed**: All files cleaned after obtaining inference results. The actions depend on individual corpus differences.

> **Run complete validation tests on both JSON and MP3 files.**

### Translations_5

- **Translation files**: Files containing translations.
- **Processed**: All files where translations have been added.

> **Run complete validation tests on both JSON and MP3 files.**
