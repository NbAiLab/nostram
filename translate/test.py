from google.cloud import translate_v3 as translate

client = translate.TranslationServiceClient.from_service_account_json('/home/perk/service-account-file.json')

parent = client.location_path('vacma-250010', 'global')
response = client.translate_text(
    parent=parent,
    contents=["test"],
    mime_type='text/plain',
    source_language_code='no',
    target_language_code='en'
)

for translation in response.translations:
    print(f'Translated text: {translation.translated_text}')
