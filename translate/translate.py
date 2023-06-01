from google.cloud import translate


def batch_translate_text(
    input_uri: str = "gs://nostram/translate/test.tsv",
    output_uri: str = "gs://nostram/translate/output3/",
    project_id: str = "vacma-250010",
    timeout: int = 180
) -> translate.TranslateTextResponse:
    """Translates a batch of texts on GCS and stores the result in a GCS location.

    Args:
        input_uri: The input URI of the texts to be translated.
        output_uri: The output URI of the translated texts.
        project_id: The ID of the project that owns the destination bucket.
        timeout: The timeout for this batch translation operation.

    Returns:
        The translated texts.
    """

    client = translate.TranslationServiceClient()

    location = "us-central1"
    # Supported file types: https://cloud.google.com/translate/docs/supported-formats
    gcs_source = {"input_uri": input_uri}

    input_configs_element = {
        "gcs_source": gcs_source,
        "mime_type": "text/html",  # Can be "text/plain" or "text/html".
    }
    gcs_destination = {"output_uri_prefix": output_uri}
    output_config = {"gcs_destination": gcs_destination}
    parent = f"projects/{project_id}/locations/{location}"

    # Supported language codes: https://cloud.google.com/translate/docs/languages
    operation = client.batch_translate_text(
        request={
            "parent": parent,
            "source_language_code": "no",
            "target_language_codes": ["en","es"],  # Up to 10 language codes here.
            "input_configs": [input_configs_element],
            "output_config": output_config,
        }
    )

    print("Waiting for operation to complete...")
    response = operation.result(timeout)

    print(f"Total Characters: {response.total_characters}")
    print(f"Translated Characters: {response.translated_characters}")

    return response

batch_translate_text()
