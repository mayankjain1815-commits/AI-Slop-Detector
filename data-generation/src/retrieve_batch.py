import openai


def retrieve_results(
    batch_data: openai.types.Batch,
    client: openai.OpenAI,
) -> str:
    output_file_id = batch_data.output_file_id
    assert isinstance(output_file_id, str)

    file_response = client.files.content(output_file_id)
    return file_response.text
