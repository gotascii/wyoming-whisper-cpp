"""Event handler for clients of the server."""
import argparse
import asyncio
import io
import json
import logging
import os
import wave
import time
import numpy as np
from datetime import datetime
from scipy import signal
from scipy.io import wavfile
from asyncio.subprocess import Process

from wyoming.asr import Transcribe, Transcript
from wyoming.audio import AudioChunk, AudioChunkConverter, AudioStop
from wyoming.event import Event
from wyoming.info import Describe, Info
from wyoming.server import AsyncEventHandler

_LOGGER = logging.getLogger(__name__)


def apply_compression(audio: np.ndarray, threshold: float = -20, ratio: float = 4.0, attack_ms: float = 5.0, release_ms: float = 50.0, sample_rate: int = 16000) -> np.ndarray:
    """Apply dynamic range compression to the audio signal."""
    # Convert threshold from dB to linear
    threshold_linear = 10.0 ** (threshold / 20.0)
    
    # Calculate attack and release in samples
    attack_samples = int(sample_rate * attack_ms / 1000.0)
    release_samples = int(sample_rate * release_ms / 1000.0)
    
    # Initialize gain memory
    gain_memory = 1.0
    compressed = np.zeros_like(audio)
    
    # Apply compression sample by sample
    for i in range(len(audio)):
        # Calculate instantaneous level
        level = abs(audio[i])
        
        # Calculate target gain
        if level > threshold_linear:
            target_gain = threshold_linear + (level - threshold_linear) / ratio
            target_gain = target_gain / level
        else:
            target_gain = 1.0
        
        # Apply attack/release
        if target_gain < gain_memory:
            gain_memory = target_gain + (gain_memory - target_gain) * np.exp(-1.0 / attack_samples)
        else:
            gain_memory = target_gain + (gain_memory - target_gain) * np.exp(-1.0 / release_samples)
        
        # Apply gain
        compressed[i] = audio[i] * gain_memory
    
    return compressed


def preprocess_audio(audio_data: bytes, sample_rate: int = 16000) -> bytes:
    """Apply DSP preprocessing to improve audio quality."""
    # Convert bytes to numpy array
    audio_array = np.frombuffer(audio_data, dtype=np.int16)
    
    # Skip the first 400ms of audio (6400 samples at 16kHz)
    start_trim = int(0.4 * sample_rate)  # 400ms worth of samples
    end_trim = sample_rate  # 1 second from the end
    
    # Only trim if we have enough audio
    if len(audio_array) > (start_trim + end_trim + sample_rate):  # Ensure we have at least 1s of audio left
        audio_array = audio_array[start_trim:-end_trim]
    elif len(audio_array) > start_trim:  # If not enough for end trim, just do start trim
        audio_array = audio_array[start_trim:]
    
    # Convert to float32 for processing
    audio_float = audio_array.astype(np.float32) / 32768.0

    # 1. High-pass filter to remove low frequency noise (below 80Hz)
    nyquist = sample_rate / 2
    cutoff = 80 / nyquist
    b, a = signal.butter(4, cutoff, btype='high')
    audio_float = signal.filtfilt(b, a, audio_float)

    # 2. Band-pass filter for voice frequencies (300Hz-3.4kHz)
    low_cutoff = 300 / nyquist
    high_cutoff = 3400 / nyquist
    b, a = signal.butter(4, [low_cutoff, high_cutoff], btype='band')
    audio_float = signal.filtfilt(b, a, audio_float)

    # 3. Pre-emphasis filter (boost high frequencies)
    pre_emphasis = 0.97
    audio_float = np.append(audio_float[0], audio_float[1:] - pre_emphasis * audio_float[:-1])

    # 4. Apply compression
    audio_float = apply_compression(
        audio_float,
        threshold=-20,      # Start compressing at -20 dB
        ratio=4.0,         # 4:1 compression ratio
        attack_ms=5.0,     # 5ms attack time
        release_ms=50.0    # 50ms release time
    )

    # 5. Final normalization to full range
    max_val = np.max(np.abs(audio_float))
    if max_val > 0:
        audio_float = audio_float / max_val  # Use full dynamic range

    # Convert back to int16
    audio_processed = (audio_float * 32768.0).astype(np.int16)
    
    return audio_processed.tobytes()


def cleanup_old_files(directory: str, prefix: str = "whisper_", keep_count: int = 10):
    """Keep only the most recent files with given prefix in directory."""
    try:
        # Get all files with our prefix
        files = [f for f in os.listdir(directory) if f.startswith(prefix) and f.endswith(".wav")]
        # Sort by name (which includes timestamp, so this is chronological)
        files.sort(reverse=True)
        
        # Remove all but the most recent keep_count files
        for old_file in files[keep_count:]:
            try:
                os.remove(os.path.join(directory, old_file))
                _LOGGER.debug("Removed old file: %s", old_file)
            except OSError as e:
                _LOGGER.warning("Failed to remove old file %s: %s", old_file, str(e))
    except Exception as e:
        _LOGGER.warning("Error during file cleanup: %s", str(e))


class WhisperCppEventHandler(AsyncEventHandler):
    """Event handler for clients."""

    def __init__(
        self,
        wyoming_info: Info,
        cli_args: argparse.Namespace,
        model_proc: Process,
        model_proc_lock: asyncio.Lock,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.cli_args = cli_args
        self.wyoming_info_event = wyoming_info.event()
        self.model_proc = model_proc
        self.model_proc_lock = model_proc_lock
        self.audio = bytes()
        self.audio_converter = AudioChunkConverter(
            rate=16000,
            width=2,
            channels=1,
        )
        self._language = self.cli_args.language
        # Ensure data_dir is a proper string path
        if isinstance(self.cli_args.data_dir, list):
            self.data_dir = self.cli_args.data_dir[0].strip('[]')
        else:
            self.data_dir = str(self.cli_args.data_dir)

    async def handle_event(self, event: Event) -> bool:
        if AudioChunk.is_type(event.type):
            if not self.audio:
                _LOGGER.debug("Receiving audio")

            chunk = AudioChunk.from_event(event)
            chunk = self.audio_converter.convert(chunk)
            self.audio += chunk.audio

            return True

        if AudioStop.is_type(event.type):
            _LOGGER.debug("Audio stopped")
            text = ""
            
            # Apply preprocessing to the audio
            processed_audio = preprocess_audio(self.audio)
            
            with io.BytesIO() as wav_io:
                wav_file: wave.Wave_write = wave.open(wav_io, "wb")
                with wav_file:
                    wav_file.setframerate(16000)
                    wav_file.setsampwidth(2)
                    wav_file.setnchannels(1)
                    wav_file.writeframes(processed_audio)

                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"whisper_{timestamp}.wav"
                output_path = os.path.join(self.data_dir, filename)
                _LOGGER.debug("Saving audio to %s", output_path)
                
                # Ensure the directory exists
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                
                with open(output_path, "wb") as f:
                    f.write(wav_io.getvalue())
                
                # Clean up old files, keeping only the 10 most recent
                cleanup_old_files(self.data_dir)

                wav_io.seek(0)
                wav_bytes = wav_io.getvalue()

                assert self.model_proc.stdin is not None
                assert self.model_proc.stdout is not None

                async with self.model_proc_lock:
                    request_str = json.dumps(
                        {"size": len(wav_bytes), "language": self._language}
                    )
                    request_line = f"{request_str}\n".encode("utf-8")
                    self.model_proc.stdin.write(request_line)
                    self.model_proc.stdin.write(wav_bytes)
                    await self.model_proc.stdin.drain()

                    lines = []
                    line = (await self.model_proc.stdout.readline()).decode().strip()
                    while line != "<|endoftext|>":
                        if line:
                            lines.append(line)
                        line = (
                            (await self.model_proc.stdout.readline()).decode().strip()
                        )

                text = " ".join(lines)
                text = text.replace("[BLANK_AUDIO]", "").strip()

            _LOGGER.info(text)

            await self.write_event(Transcript(text=text).event())
            _LOGGER.debug("Completed request")

            # Reset
            self.audio = bytes()
            self._language = self.cli_args.language

            return False

        if Transcribe.is_type(event.type):
            transcribe = Transcribe.from_event(event)
            if transcribe.language:
                self._language = transcribe.language
                _LOGGER.debug("Language set to %s", transcribe.language)
            return True

        if Describe.is_type(event.type):
            await self.write_event(self.wyoming_info_event)
            _LOGGER.debug("Sent info")
            return True

        return True
