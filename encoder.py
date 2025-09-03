import numpy as np
import soundfile as sf
import sounddevice as sd

# Step 1: Define Notes & the Roger Signal
NOTE_FREQS = {
    'C4': 261.63, 'D4': 293.66, 'E4': 329.63, 'F4': 349.23,
    'G4': 392.00, 'A4': 440.00, 'B4': 493.88, 'C5': 523.25,
    'D5': 587.33, 'E5': 659.25, 'F5': 698.46, 'G5': 783.99,
    'A5': 880.00, 'B5': 987.77, 'C6': 1046.50, 'D6': 1174.66
}
ROGER_SIGNAL = 1000.0  # A5 as the Roger signal
START_SIGNAL_FREQ = 600.0  # Starting signal (600 Hz)

# Step 2: Convert Text to Binary
text = "if i were a rich girl"
binary_data = ''.join(format(ord(c), '08b') for c in text)

# Step 3: Encode Binary as Chords (the shorter the duration for faster transmission, however, low accuracy)
SAMPLE_RATE = 44100
DURATION = 0.2  # 200ms per chord (faster transmission)
t = np.linspace(0, DURATION, int(SAMPLE_RATE * DURATION), endpoint=False)

audio_signal = []
for i in range(0, len(binary_data), 16):  # uses 16 bits per symbol
    bits = binary_data[i:i+16].ljust(16, '0')  # Ensure we have 16 bits per symbol
    chord = sum(0.2 * np.sin(2 * np.pi * list(NOTE_FREQS.values())[j] * t) for j in range(16) if bits[j] == '1')  # More frequencies
    audio_signal.append(chord)

# Step 4: Add Start and Roger Signals (with lower volume %20-ish)
start_signal = 0.2 * np.sin(2 * np.pi * START_SIGNAL_FREQ * t)
roger_wave = 0.2 * np.sin(2 * np.pi * ROGER_SIGNAL * t)

audio_signal.insert(0, start_signal)
audio_signal.append(roger_wave)

# Step 5: Concatenate & Normalize
# Fuzzy?
audio_output = np.concatenate(audio_signal)
audio_output = audio_output / np.max(np.abs(audio_output)) * 0.8  # Normalize to 80% max volume

# Step 6: Play sound before Saving
sd.play(audio_output, SAMPLE_RATE)
sd.wait()

# Step 7: Save File
sf.write("x.mp3", audio_output, SAMPLE_RATE)
print("Encoded file saved as 'x.wav'")
