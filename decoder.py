import numpy as np
import pyaudio
import scipy.signal as signal
from scipy.fftpack import fft
from collections import deque

# Step 1: Define Note Frequencies & Signals
NOTE_FREQS = {
    'C4': 261.63, 'D4': 293.66, 'E4': 329.63, 'F4': 349.23,
    'G4': 392.00, 'A4': 440.00, 'B4': 493.88, 'C5': 523.25,
    'D5': 587.33, 'E5': 659.25, 'F5': 698.46, 'G5': 783.99,
    'A5': 880.00, 'B5': 987.77, 'C6': 1046.50, 'D6': 1174.66
}
FREQ_TO_BIT = {freq: str(i) for i, freq in enumerate(NOTE_FREQS.values())}
ROGER_SIGNAL = 1000.0  # A5 = End Signal
START_SIGNAL_FREQ = 600.0  # Start signal (600 Hz)

# Step 2: Set Up Microphone
SAMPLE_RATE = 44100
CHUNK_SIZE = int(0.2 * SAMPLE_RATE)  # 200ms per chunk (faster)
RECORD_SECONDS = 30  # Increase slightly to ensure we capture the signals

p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paInt16, channels=1, rate=SAMPLE_RATE,
                input=True, frames_per_buffer=CHUNK_SIZE)

print("Listening for start signal...")

# Step 3: Listen for Start Signal Before Decoding
start_signal_detected = False
frames = []
binary_sequence = ""

for _ in range(int(SAMPLE_RATE / CHUNK_SIZE * RECORD_SECONDS)):
    data = np.frombuffer(stream.read(CHUNK_SIZE), dtype=np.int16)

    # Step 4: Perform FFT
    fft_spectrum = np.abs(fft(data)[:len(data) // 2])
    freqs = np.fft.rfftfreq(len(data), d=1 / SAMPLE_RATE)

    # Detect peaks
    peaks, _ = signal.find_peaks(fft_spectrum, height=np.max(fft_spectrum) * 0.2)

    # Step 5: Detect Start Signal
    detected_freqs = [freqs[p] for p in peaks]
    if any(abs(f - START_SIGNAL_FREQ) < 10 for f in detected_freqs) and not start_signal_detected:
        print("Start signal detected. Beginning decoding...")
        start_signal_detected = True

    # Step 6: Decode Data After Start Signal is Detected
    if start_signal_detected:
        # Step 7: Detect Roger Signal (Exit if Found)
        if any(abs(f - ROGER_SIGNAL) < 10 for f in detected_freqs):
            print("Roger signal detected. Stopping decoding.")
            break

        # Step 8: Convert Peaks to Binary
        bits = deque(['0'] * 16)  # More bits per symbol (using 16 frequencies now)
        for peak in peaks:
            closest_freq = min(NOTE_FREQS.values(), key=lambda f: abs(f - freqs[peak]))
            if closest_freq in FREQ_TO_BIT:
                bits[int(FREQ_TO_BIT[closest_freq])] = '1'

        binary_sequence += ''.join(bits)

stream.stop_stream()
stream.close()
p.terminate()

# Step 9: Convert Binary to Text
decoded_text = ''.join(chr(int(binary_sequence[i:i + 8], 2)) for i in range(0, len(binary_sequence), 8))
print(f"Decoded Text: {decoded_text}")
