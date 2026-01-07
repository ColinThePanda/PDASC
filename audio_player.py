import sounddevice as sd
import threading
import numpy as np
from typing import Generator

class AudioPlayer:
    def __init__(self, audio_gen: Generator[np.ndarray, None, None],
                 samplerate=44100, channels=2, blocksize=1024):
        self.audio_gen = audio_gen
        self.samplerate = samplerate
        self.channels = channels
        self.blocksize = blocksize
        self.buffer = np.empty((0, channels), dtype=np.float32)
        self.lock = threading.Lock()
        self.done = False

    def callback(self, outdata, frames, time_info, status):
        with self.lock:
            # Take only the amount we need
            if len(self.buffer) >= frames:
                outdata[:] = self.buffer[:frames]
                self.buffer = self.buffer[frames:]
            else:
                # Fill what we have, rest with zeros
                outdata[:len(self.buffer)] = self.buffer
                outdata[len(self.buffer):] = 0
                self.buffer = np.empty((0, self.channels), dtype=np.float32)

    def start(self):
        # Start stream
        self.stream = sd.OutputStream(
            samplerate=self.samplerate,
            channels=self.channels,
            callback=self.callback,
            blocksize=self.blocksize
        )
        self.stream.start()

        # Feeding thread
        def feed():
            for chunk in self.audio_gen:
                # Make sure chunk is float32 and correct shape
                if chunk.dtype != np.float32:
                    chunk = chunk.astype(np.float32) / 32768.0  # int16 → float32
                if chunk.ndim == 1 and self.channels == 2:
                    chunk = np.stack([chunk, chunk], axis=-1)
                elif chunk.ndim == 2 and chunk.shape[1] != self.channels:
                    if chunk.shape[1] == 1:
                        chunk = np.repeat(chunk, self.channels, axis=1)
                    else:
                        raise ValueError("Audio chunk channels mismatch")
                with self.lock:
                    self.buffer = np.concatenate([self.buffer, chunk])
            self.done = True

        threading.Thread(target=feed, daemon=True).start()
