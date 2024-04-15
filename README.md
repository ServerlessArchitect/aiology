# AIology

AIology is a CLI tool designed to transcribe, translate, and summarize videos, making it easier to handle multimedia content in multiple languages. It uses OpenAI's GPT models, Whisper for transcription, and Spacy for language processing tasks.

## Features

- **Transcribe** audio and video files to text using OpenAI's Whisper model.
- **Translate** text between English and Mandarin using OpenAI's GPT models.
- **Summarize** video content in a concise, human-readable format.

## Installation

Before installing AIology, ensure you have Python installed on your machine. Then follow these steps:

```bash
# Clone the repository
git clone https://your-repository-url.git
cd aiology

# Install the required packages
pip install -r requirements.txt
```

## Configuration

1. Obtain an API key from [OpenAI](https://platform.openai.com/api-keys).
2. Create a configuration file for the API key:
   
   ```bash
   mkdir -p ~/.aiology
   echo '{"api_key":"your-openai-api-key"}' > ~/.aiology/openai.json
   ```
   
   Replace `your-openai-api-key` with the key you obtained from OpenAI.

## Usage

### General Help

To see a list of available commands and their descriptions, you can run:

```bash
python aiology.py --help
```

### Transcription

To transcribe audio from a video file:

```bash
python aiology.py transcribe path/to/your/video.mp4
```

### Translation

To translate text from Mandarin to English:

```bash
python aiology.py translate path/to/your/file.txt
```

### Summarization

To summarize a video file:

```bash
python aiology.py summarize path/to/your/video.mp4
```

## Contributing

Contributions are what make the open-source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
   **NB** Please follow [Conventional Commits](https://www.conventionalcommits.org/).
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

Distributed under the MIT License. See `LICENSE` for more information.

## Acknowledgements

- [OpenAI](https://www.openai.com/)
- [Spacy](https://spacy.io/)
- [Typer](https://typer.tiangolo.com/)
