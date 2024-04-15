"""A simple CLI tool for transcribing, translating, and summarising videos.
"""

import json
import os
import tempfile

import magic
import moviepy.editor
import openai
import rich.console
import rich.markdown
import spacy
import spacy.cli.download
import tiktoken
import typer
import typing_extensions
from icecream import ic

CHUNK_SPACER = "\n"
PREVIOUS_CHUNK_ORIGINAL_HEADER = (
    "<!-- Previous Chunk's Original Ending For Context -->\n"
)
PREVIOUS_CHUNK_TRANSLATION_HEADER = (
    "<!-- Previous Chunk's Translated Ending For Context -->\n"
)
NEXT_CHUNK_HEADER = "<!-- Next Chunk's Beginning for Context -->\n"
CURRENT_CHUNK_HEADER = (
    """<!-- Current Chunk to be Translated: ONLY TRANSLATE"""
    """THE BELOW, using the above for context -->\n"""
)

openai_config_path = os.path.expanduser("~/.aiology/openai.json")
with open(openai_config_path, "r") as openai_config_str:
    openai_config = json.loads(openai_config_str.read())
openai_client = openai.OpenAI(api_key=openai_config["api_key"])
app = typer.Typer()
console = rich.console.Console()

gpt4_tokenizer = tiktoken.encoding_for_model("gpt-4")


@app.command()
def summarize(
    filename: str,
    model: str = "gpt-3.5-turbo-16k",
    prompt: typing_extensions.Annotated[
        str,
        typer.Option(
            help="Instructions to AI - sway the model how to format its output.",
            prompt=True,
        ),
    ] = """
        ABOUT YOU
        You are a linguist. You will be given a piece of text, e.g.
        a transcript of an interview between a Mandarin student and future
        teachers."
        ===
        ENGLISH
        You speak standard British English and enthusiastically advise
        Received Pronunciation through the International Phonetic Alphabet.

        Your responses are comprehensive and concise, always prioritising
        completeness and clarity. You observe the User’s level of English
        and match your level of English to make your responses easy
        to understand. If in doubt, speak simple English.

        Yes: favourite, emphasise
        No: favorite, emphasize
        --
        When discussing vocabulary, offer an assortment of formal, neutral,
        and informal registers. When discussing or introducing new
        or difficult words, include IPA pronunciation.

        Example: … word /IPA/ …
        ===
        CHINESE
        You speak standard Mandarin Chinese and - similar to above - advise
        pinyin (or other corresponding romanisation) when discussing vocabulary.

        High-Level Examples
        “The Chinese word for apple - <苹果 píngguǒ> - is also …”
        ===
        YOUR TASK
        Please summarise the content given, offer any insights, possible
        follow-up questions, any pertaining existing literature and research
        (particularly titles and links), and so on.

        Be creative and take moonshots while making clear what's a fact
        and what's a guess or conjecture. State any existing facts first,
        and then add your opinion and suggestions.

        Priotise completeness over brevity while being reasonably concise.
        """,
):
    """Produce concise, human-readable summary of the text given."""
    text = open(filename, "r").read()
    response = openai_client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": text},
        ],
    )
    assert response.choices[0].message.content

    summary = response.choices[0].message.content
    formatted_summary = rich.markdown.Markdown(summary)
    console.print(formatted_summary)

    outfilename = filename + ".summary.txt"
    with open(outfilename, "w") as file:
        file.write(summary)


@app.command()
def transcribe(
    filename: str,
    model: str = "whisper-1",
    prompt: typing_extensions.Annotated[
        str,
        typer.Option(
            help=(
                "Example to AI - drive the style of output and cue spelling"
                " for whimsical lexicon oddities."
            ),
            prompt=True,
        ),
    ] = """
        Hello! Here is an interview between a Mandarin student and future
        teachers. Included is a mixture of English and Mandarin,
        such as the phrase 手足无措食醋漂流.
        """,
):
    """Turn audio (or video's audio) into subtitles."""
    if is_video_file(filename):
        filename = extract_audio_from_video(filename)

    audio_file = open(filename, "rb")
    transcript = openai_client.audio.transcriptions.create(
        model=model,
        file=audio_file,
        response_format="srt",
    )
    assert isinstance(transcript, str)

    print(transcript)

    outfilename = filename + ".srt.txt"
    with open(outfilename, "w") as file:
        file.write(transcript)


def extract_audio_from_video(video_path):
    audio_output_path = video_path + ".mp3"
    with moviepy.editor.VideoFileClip(video_path) as video:
        video.audio.write_audiofile(audio_output_path)
    return audio_output_path


def is_video_file(filepath):
    mime = magic.Magic(mime=True)
    mimetype = mime.from_file(filepath)
    return mimetype.startswith("video")


@app.command()
def translate(
    filename: str,
    model: str = "gpt-3.5-turbo-16k",
    input_language_code: typing_extensions.Annotated[
        str,
        typer.Option(
            help="Main language of the document.",
            prompt=False,
        ),
    ] = "zh",
    output_language_code: str = "en",
    prompt: typing_extensions.Annotated[
        str,
        typer.Option(
            help="Instructions to AI - tell the model what to do and sway its output.",
            prompt=True,
        ),
    ] = """
        You are a professional Mandarin <-> English translator.
        When given content in Mandarin, produce equivalent in English
        (and vice versa), keeping the style, tone, etc, of the original
        message.
        Due to technical limitations, you'll be translating some content
        in chunks. When this is the case, you'll be presented with the last
        few sentences of the preceding chunk, translation for the (roughly)
        corresponding chunk, and the first few sentences of following chunk.
        Use those for context to allow for smooth merging of the chunks.
        Output only the chunk to be translated without any comments or markers.
    """,
):
    """Translate the given text into the given language."""
    outnlp = load_spacy_model(output_language_code)
    outnlp.add_pipe("sentencizer")

    text = open(filename, "r").read()

    translations = []
    chunk_generator = generate_chunks_for_translation(text, input_language_code)
    i = 0
    try:
        chunk = next(chunk_generator)  # Get the first chunk
        while chunk:
            response = openai_client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": prompt},
                    {
                        "role": "system",
                        "content": f"Output language = {output_language_code}",
                    },
                    {"role": "user", "content": chunk},
                ],
            )
            translation = response.choices[0].message.content
            assert isinstance(translation, str)

            translations.append(translation)

            md_translation = rich.markdown.Markdown(translation)
            console.print(md_translation)

            # Pass the translation and get the next chunk
            doc = outnlp(translation)
            chunk = chunk_generator.send(
                ic("".join([sentence.text for sentence in doc.sents][-8:]))
            )
    except StopIteration as e:
        ic(e)

    translated_document = "".join(translations)

    outfilename = filename + "." + output_language_code + ".txt"
    with open(outfilename, "w") as file:
        file.write(translated_document)


def load_spacy_model(language_code):
    try:
        return spacy.load(f"{language_code}_core_web_sm")
    except OSError:
        print(f"Downloading language model for the '{language_code}' language...")
        spacy.cli.download(f"{language_code}_core_web_sm")
        return spacy.load(f"{language_code}_core_web_sm")


def generate_chunks_for_translation(
    text, input_language_code, max_tokens=5000, context_sentences=8
):
    nlp = load_spacy_model(input_language_code)
    nlp.add_pipe("sentencizer")
    doc = nlp(text)

    sentences = [sentence.text for sentence in doc.sents]
    num_sentences = len(sentences)

    current_chunk = [CURRENT_CHUNK_HEADER]
    current_length = 0
    current_chunk_beginning = 0
    previous_chunk = None
    previous_chunk_translation = None

    for i, sentence in enumerate(sentences):
        sentence_length = len((gpt4_tokenizer.encode(sentence)))

        if (current_length) + (sentence_length) > (max_tokens):
            # Finalize the current chunk
            if i < num_sentences - 1:
                end_context = min(i + context_sentences, num_sentences)
                current_chunk = (
                    [CHUNK_SPACER, NEXT_CHUNK_HEADER]
                    + sentences[i:end_context]
                    + current_chunk
                )
            if previous_chunk_translation is not None:
                current_chunk = (
                    [CHUNK_SPACER, PREVIOUS_CHUNK_TRANSLATION_HEADER]
                    + [previous_chunk_translation]
                    + current_chunk
                )
            if previous_chunk:
                current_chunk = (
                    [CHUNK_SPACER, PREVIOUS_CHUNK_ORIGINAL_HEADER]
                    + previous_chunk
                    + current_chunk
                )

            # Yield the current chunk and receive the translation for the next one
            chunk = ic("".join(current_chunk))
            previous_chunk_translation = yield chunk

            # Prepare the next chunk
            previous_chunk = current_chunk[-context_sentences:]
            current_chunk = [CHUNK_SPACER, CURRENT_CHUNK_HEADER, sentence]
            current_length = sentence_length
            current_chunk_beginning = i
        else:
            current_chunk.append(sentence)
            current_length += sentence_length

    # Yield the last chunk
    if current_chunk:
        yield "".join(current_chunk)


if __name__ == "__main__":
    app()
