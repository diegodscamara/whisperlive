import sounddevice as sd
import numpy as np
import wave
import threading
import queue
import whisper
import time
import logging
import sys
from datetime import datetime
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("transcriber.log")
    ]
)
logger = logging.getLogger("whisperlive")

class AudioCapture:
    def __init__(self, device_name="BlackHole 2ch", sample_rate=16000, chunk_size=1024):
        self.setup_audio_device(device_name, sample_rate, chunk_size)
        self.setup_output_files()

    def setup_audio_device(self, device_name, sample_rate, chunk_size):
        """Initialize audio device settings"""
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.audio_queue = queue.Queue()
        self.is_running = False
        
        # Find audio device
        devices = sd.query_devices()
        self.device_id = next(
            (i for i, d in enumerate(devices) if device_name in d["name"]),
            None
        )
        
        if self.device_id is None:
            raise ValueError(f"Could not find audio device: {device_name}")
        
        logger.info(f"Using audio device: {devices[self.device_id]['name']}")

    def setup_output_files(self):
        """Setup output directories and files"""
        self.output_dir = Path("recordings")
        self.output_dir.mkdir(exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Initialize WAV file
        self.audio_file = self.output_dir / f"recording_{self.timestamp}.wav"
        self.wav_file = wave.open(str(self.audio_file), 'wb')
        self.wav_file.setnchannels(1)
        self.wav_file.setsampwidth(2)
        self.wav_file.setframerate(self.sample_rate)

    def audio_callback(self, indata, frames, time, status):
        """Process incoming audio data"""
        if status:
            logger.warning(f"Audio callback status: {status}")
        
        # Convert to mono and save
        audio_data = np.mean(indata, axis=1).astype(np.float32)
        audio_data_int16 = (audio_data * 32767).astype(np.int16)
        self.wav_file.writeframes(audio_data_int16.tobytes())
        self.audio_queue.put(audio_data)

    def start(self):
        """Start audio capture"""
        self.is_running = True
        self.stream = sd.InputStream(
            device=self.device_id,
            channels=2,
            samplerate=self.sample_rate,
            blocksize=self.chunk_size,
            callback=self.audio_callback
        )
        self.stream.start()
        logger.info("Started audio capture")

    def stop(self):
        """Stop audio capture"""
        self.is_running = False
        if hasattr(self, 'stream'):
            self.stream.stop()
            self.stream.close()
        self.wav_file.close()
        logger.info("Stopped audio capture")

class WhisperTranscriber:
    def __init__(self, model_name="base"):
        self.setup_model(model_name)
        self.setup_transcription_state()
        self.setup_output_file()

    def setup_model(self, model_name):
        """Initialize Whisper model"""
        logger.info(f"Loading Whisper model: {model_name}")
        self.model = whisper.load_model(model_name)
        self.buffer_size = 16000 * 5  # 5 seconds buffer
        self.min_process_size = 16000 * 2  # 2 seconds minimum

    def setup_transcription_state(self):
        """Initialize transcription state"""
        self.is_running = False
        self.current_sentence = ""
        self.full_transcript = []

    def setup_output_file(self):
        """Setup transcript file"""
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.transcript_file = Path("recordings") / f"transcript_{self.timestamp}.txt"
        self.transcript_file.parent.mkdir(exist_ok=True)
        
        with open(self.transcript_file, "w", encoding="utf-8") as f:
            f.write(f"Transcript - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n\n")

    def process_audio(self, audio_data):
        """Process audio chunk and return transcribed text"""
        try:
            audio_data = whisper.pad_or_trim(audio_data)
            mel = whisper.log_mel_spectrogram(audio_data).to(self.model.device)
            
            # Detect language and decode
            _, probs = self.model.detect_language(mel)
            detected_language = max(probs, key=probs.get)
            
            result = self.model.decode(
                mel,
                whisper.DecodingOptions(
                    language=detected_language,
                    without_timestamps=True,
                    fp16=False,
                    beam_size=5
                )
            )
            
            text = result.text.strip()
            
            # Validate transcription
            if not self.is_valid_transcription(text):
                return "", False
                
            is_complete_sentence = text[-1] in ".!?" if text else False
            return text, is_complete_sentence
            
        except Exception as e:
            logger.error(f"Error in process_audio: {e}")
            return "", False

    def is_valid_transcription(self, text):
        """Validate transcription output"""
        if not text or len(text) < 2:
            return False
            
        words = text.split()
        
        # Check for repetitive patterns
        if len(words) >= 4:
            for i in range(len(words)-3):
                if words[i:i+2] == words[i+2:i+4]:
                    return False
        
        # Check for extremely long words
        if any(len(word) > 20 for word in words):
            return False
            
        return True

    def update_transcript(self, text, is_complete_sentence):
        """Update transcript with new text"""
        if text.strip():
            self.current_sentence += " " + text.strip()
            self.current_sentence = self.current_sentence.strip()
            
            if is_complete_sentence:
                self.full_transcript.append(self.current_sentence)
                self.save_to_file(self.current_sentence + "\n", True)
                self.current_sentence = ""
            else:
                self.save_to_file(text + " ")
            
            self.display_transcript()

    def display_transcript(self):
        """Display formatted transcript"""
        print("\033[2J\033[H", end="")  # Clear screen
        print(self.format_transcript())
        sys.stdout.flush()

    def format_transcript(self):
        """Format transcript for display"""
        output = [
            "\n" + "=" * 80,
            "LIVE TRANSCRIPT",
            "=" * 80 + "\n",
            *self.full_transcript
        ]
        
        if self.current_sentence:
            output.append("\nCurrent: " + self.current_sentence)
        
        return "\n".join(output)

    def save_to_file(self, text, is_complete_sentence=False):
        """Save transcript to file"""
        with open(self.transcript_file, "a", encoding="utf-8") as f:
            f.write(text if is_complete_sentence else text)

    def process_audio_queue(self, audio_queue):
        """Process audio from queue"""
        accumulated_audio = np.array([], dtype=np.float32)
        last_process_size = 0
        
        while self.is_running:
            try:
                audio_chunk = audio_queue.get(timeout=0.1)
                accumulated_audio = np.concatenate([accumulated_audio, audio_chunk])
                
                if len(accumulated_audio) - last_process_size >= self.min_process_size:
                    accumulated_audio = accumulated_audio[-self.buffer_size:]
                    text, is_complete = self.process_audio(accumulated_audio)
                    self.update_transcript(text, is_complete)
                    
                    last_process_size = len(accumulated_audio)
                    if len(accumulated_audio) > self.min_process_size:
                        accumulated_audio = accumulated_audio[-self.min_process_size:]
                        last_process_size = len(accumulated_audio)
                    
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error processing audio: {e}")

    def start(self, audio_queue):
        """Start transcription processing"""
        self.is_running = True
        threading.Thread(
            target=self.process_audio_queue,
            args=(audio_queue,),
            daemon=True
        ).start()

    def stop(self):
        """Stop transcription"""
        self.is_running = False
        if self.current_sentence:
            self.save_to_file(self.current_sentence + "\n", True)
        self.save_to_file(f"\nEnd of Transcript - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n", True)

def main():
    try:
        audio_capture = AudioCapture()
        transcriber = WhisperTranscriber()

        audio_capture.start()
        transcriber.start(audio_capture.audio_queue)

        print("WhisperLive started. Press Ctrl+C to stop.")
        while True:
            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\nStopping WhisperLive...")
    finally:
        transcriber.stop()
        audio_capture.stop()
        print("\nWhisperLive stopped.")

if __name__ == "__main__":
    main() 