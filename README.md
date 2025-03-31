# WhisperLive 🎙️

Real-time audio transcription powered by OpenAI's Whisper model.

## Features ✨

- Real-time audio capture and transcription
- Support for multiple languages (auto-detection)
- Continuous streaming transcription
- Automatic sentence detection and formatting
- Save transcriptions to text files
- Hallucination prevention mechanisms
- Clean, formatted output with timestamps

## Requirements 📋

- Python 3.8+
- BlackHole 2ch (or similar virtual audio device)
- Required Python packages:
  - sounddevice
  - numpy
  - whisper
  - wave

## Installation 🚀

1. Clone the repository:
```bash
git clone https://github.com/diegodscamara/whisperlive.git
cd whisperlive
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Install BlackHole 2ch (macOS):
```bash
brew install blackhole-2ch
```

For other operating systems, use an equivalent virtual audio device.

## Models 🤖

WhisperLive uses OpenAI's Whisper models for transcription. The models are automatically downloaded when you first run the application. By default, it uses the "base" model, which offers a good balance between accuracy and performance.

The models are stored in the `models` directory but are not included in the repository due to their size. They will be downloaded automatically when needed.

Available models:
- `tiny` (74MB) - Fastest, least accurate
- `base` (142MB) - Good balance for most uses
- `small` (466MB) - More accurate but slower
- `medium` (1.5GB) - Even more accurate
- `large` (2.9GB) - Most accurate, slowest

To change the model, modify the `model_name` parameter in `main.py`:
```python
transcriber = WhisperTranscriber(model_name="base")  # Change "base" to your preferred model
```

## Usage 💡

1. Set up your virtual audio device (BlackHole 2ch) as your system's audio output.

2. Run the transcription:
```bash
python main.py
```

3. Start speaking or playing audio. The transcription will appear in real-time.

4. Press `Ctrl+C` to stop the transcription.

## Output Files 📁

The app creates two types of files in the `recordings` directory:
- `recording_YYYYMMDD_HHMMSS.wav` - Audio recording
- `transcript_YYYYMMDD_HHMMSS.txt` - Text transcription

## Configuration ⚙️

Default settings in `main.py`:
- Sample rate: 16000 Hz
- Buffer size: 5 seconds
- Minimum process size: 2 seconds
- Model: "base" (can be changed to other Whisper models)

## Contributing 🤝

Contributions are welcome! Please feel free to submit a Pull Request.

## License 📄

MIT License - feel free to use this project for any purpose.

## Acknowledgments 🙏

- [OpenAI Whisper](https://github.com/openai/whisper) for the amazing speech recognition model
- [sounddevice](https://python-sounddevice.readthedocs.io/) for audio handling
- [BlackHole](https://github.com/ExistentialAudio/BlackHole) for virtual audio routing 