# Voice-Enabled Loan Assistant

A conversational AI loan assistant built in an afternoon, showcasing modern speech recognition with OpenAI's Whisper and natural language processing with GPT-3.5.

## 🎯 Project Overview

This proof-of-concept demonstrates how quickly we can build sophisticated voice-enabled applications using cutting-edge AI tools. The assistant conducts a natural conversation to collect loan application information, using:

- **OpenAI Whisper** for offline speech recognition (no cloud STT required!)
- **Voice Activity Detection** for natural conversation flow
- **GPT-3.5** for intelligent parsing of user responses
- **Cross-platform compatibility** (works on M1 Macs, Intel Macs, and Windows)

## ✨ Features

- **Offline Speech Recognition**: Uses Whisper models locally - no internet required for voice processing
- **Model Comparison**: Runs three Whisper models (tiny, base, small) to showcase speed/accuracy tradeoffs
- **Natural Conversation Flow**: Voice activity detection automatically stops recording after silence
- **Intelligent Response Parsing**: GPT-3.5 extracts structured data from natural speech
- **Conversational UI**: Text-to-speech responses create a natural dialogue experience
- **Structured Output**: Generates a formatted loan application summary table

## 📋 Requirements

- Python 3.8+
- macOS (Intel or Apple Silicon) or Windows
- Microphone access
- OpenAI API key (for GPT-3.5 response parsing)
- ~500MB disk space for Whisper models

## 🚀 Installation

1. Clone this repository:
```bash
git clone <your-repo-url>
cd voicendm
```

2. Create and activate a virtual environment:
```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

4. Create a `.env` file with your OpenAI API key:
```bash
echo "OPENAI_API_KEY=your-api-key-here" > .env
```

## 💬 Usage

Run the assistant:
```bash
python speechDemo.py
```

The assistant will:
1. Download Whisper models on first run (tiny: 39MB, base: 140MB, small: 483MB)
2. List available audio devices and let you select one
3. Guide you through a loan application conversation
4. Display transcription performance metrics for each Whisper model
5. Generate a structured loan application summary

### Conversation Flow

1. **Name Collection**: "Hi, what's your name?"
2. **Loan Interest**: "Are you looking for a loan?"
3. **State Information**: "What state do you live in?" (includes fun sports team banter)
4. **House Value**: "What is the value of your house?"
5. **Rate Negotiation**: Offers initial rate and potential discount

## 🔧 How It Works

### Speech Recognition Pipeline
1. **Audio Capture**: Uses `sounddevice` for cross-platform microphone access
2. **Voice Activity Detection**: Monitors audio levels to detect speech and silence
3. **Model Comparison**: Processes audio through three Whisper models:
   - Small (244M params): Most accurate, slowest
   - Base (74M params): Balanced performance ✓ **Used for production**
   - Tiny (39M params): Fastest, least accurate
4. **Result Selection**: Uses base model output for optimal speed/accuracy balance

### Natural Language Processing
- GPT-3.5 interprets conversational responses
- Extracts structured data (names, yes/no responses, dollar amounts, states)
- Handles various input formats ("1.5 million", "$1.5M", "one point five mil")

## 🎨 Innovation Highlights

Built in a single afternoon, this project demonstrates:
- **Modern AI Integration**: Combines Whisper + GPT-3.5 for a complete voice interface
- **Practical Problem Solving**: Moved from PyAudio (M1 incompatible) to sounddevice
- **Performance Transparency**: Shows real-time model performance comparisons
- **Production Considerations**: Balances accuracy, speed, and user experience

## 🐛 Troubleshooting

### SSL Certificate Errors (macOS)
The code includes an SSL workaround for macOS certificate issues. For production, properly configure certificates.

### Audio Device Issues
- The app lists all available input devices on startup
- Select the appropriate microphone when prompted
- Default device usually works fine

### Model Download
- First run downloads ~662MB of models
- Models are cached in `./whisper_model/`
- Add this directory to `.gitignore`

### Performance
- Base model typically processes in 0.5-2 seconds
- Small model takes 2-5 seconds
- Tiny model processes in 0.2-0.8 seconds

## 📝 License

This is a demonstration project showcasing rapid prototyping with modern AI tools.

## 🏗️ Built With

- [OpenAI Whisper](https://github.com/openai/whisper) - Speech recognition
- [sounddevice](https://python-sounddevice.readthedocs.io/) - Audio I/O
- [OpenAI GPT-3.5](https://platform.openai.com/) - Natural language understanding
- [pyttsx3](https://pyttsx3.readthedocs.io/) - Text-to-speech

---

*An afternoon experiment in voice-enabled AI applications*