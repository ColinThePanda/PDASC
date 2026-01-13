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
        self.stream = None
        self.feed_thread = None
        self.stopped = False

    def callback(self, outdata, frames, time_info, status):
        with self.lock:
            if len(self.buffer) >= frames:
                outdata[:] = self.buffer[:frames]
                self.buffer = self.buffer[frames:]
            else:
                outdata[:len(self.buffer)] = self.buffer
                outdata[len(self.buffer):] = 0
                self.buffer = np.empty((0, self.channels), dtype=np.float32)

    def start(self):
        self.stream = sd.OutputStream(
            samplerate=self.samplerate,
            channels=self.channels,
            callback=self.callback,
            blocksize=self.blocksize
        )
        self.stream.start()

        def feed():
            try:
                for chunk in self.audio_gen:
                    if self.stopped:
                        break
                    # Make sure chunk is float32 and correct shape
                    if chunk.dtype != np.float32:
                        chunk = chunk.astype(np.float32) / 32768.0  # int16 -> float32
                    if chunk.ndim == 1 and self.channels == 2:
                        chunk = np.stack([chunk, chunk], axis=-1)
                    elif chunk.ndim == 2 and chunk.shape[1] != self.channels:
                        if chunk.shape[1] == 1:
                            chunk = np.repeat(chunk, self.channels, axis=1)
                        else:
                            raise ValueError("Audio chunk channels mismatch")
                    with self.lock:
                        self.buffer = np.concatenate([self.buffer, chunk])
            except:
                pass
            finally:
                self.done = True

        self.feed_thread = threading.Thread(target=feed, daemon=True)
        self.feed_thread.start()
    
    def stop(self):
        """Properly stop and cleanup audio player"""
        self.stopped = True
        
        # Wait for feed thread to finish first (with timeout)
        if self.feed_thread and self.feed_thread.is_alive():
            self.feed_thread.join(timeout=0.5)
        
        if self.stream:
            try:
                # Give stream time to finish current callback
                import time
                time.sleep(0.1)
                self.stream.stop()
                time.sleep(0.05)
                self.stream.close()
            except:
                pass
            self.stream = None