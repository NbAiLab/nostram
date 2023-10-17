import argparse
from google.cloud import translate

def batch_translate_text(input_bucket_file: str, output_bucket_folder: str, source_language_code: str = "no", target_language_codes: list = ["en", "es"], timeout: int = 3600) -> None:
    client = translate.TranslationServiceClient()

    location = "us-central1"
    gcs_source = {"input_uri": input_bucket_file}
    input_configs_element = {
        "gcs_source": gcs_source,
        "mime_type": "text/html",  # Can be "text/plain" or "text/html".
    }
    gcs_destination = {"output_uri_prefix": output_bucket_folder}
    output_config = {"gcs_destination": gcs_destination}
    project_id = "north-390910"  # This should be parameterized if it changes often
    parent = f"projects/{project_id}/locations/{location}"

    operation = client.batch_translate_text(
        request={
            "parent": parent,
            "source_language_code": source_language_code,
            "target_language_codes": target_language_codes,
            "input_configs": [input_configs_element],
            "output_config": output_config,
        }
    )

    print("Waiting for operation to complete...")
    response = operation.result(timeout)

    print(f"Total Characters: {response.total_characters}")
    print(f"Translated Characters: {response.translated_characters}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_bucket_file", required=True, help="The input URI of the texts to be translated.")
    parser.add_argument("--output_bucket_folder", required=True, help="The output URI folder for the translated texts.")
    parser.add_argument("--source_language_code", default="no", help="The source language code.")
    parser.add_argument("--target_language_codes", nargs='+', default=["en"], help="The target language codes.")
    parser.add_argument("--timeout", type=int, default=7200, help="The timeout for this batch translation operation.")
    args = parser.parse_args()

    batch_translate_text(args.input_bucket_file, args.output_bucket_folder, args.source_language_code, args.target_language_codes, args.timeout)
