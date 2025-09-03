import numpy as np
import soundfile as sf
import sounddevice as sd

# Step 1: Define Notes & the Roger Signal
NOTE_FREQS = {
    'C4': 261.63, 'D4': 293.66, 'E4': 329.63, 'F4': 349.23,
    'G4': 392.00, 'A4': 440.00, 'B4': 493.88, 'C5': 523.25
}
ROGER_SIGNAL = 880.0  # A5 as Roger signal

# Step 2: Convert Text to Binary
text = "roger that captain!"
binary_data = ''.join(format(ord(c), '08b') for c in text)

# Step 3: Encode Binary as Chords
SAMPLE_RATE = 44100
DURATION = 0.3  # 300ms per chord
t = np.linspace(0, DURATION, int(SAMPLE_RATE * DURATION), endpoint=False)

audio_signal = []
for i in range(0, len(binary_data), 8):
    bits = binary_data[i:i+8]
    chord = sum(0.2 * np.sin(2 * np.pi * list(NOTE_FREQS.values())[j] * t) for j in range(8) if bits[j] == '1')  # Lower volume
    audio_signal.append(chord)

# Step 4: Add Roger Signal at 20% volume at the end
roger_wave = 0.2 * np.sin(2 * np.pi * ROGER_SIGNAL * t)
audio_signal.append(roger_wave)

# Step 5: Concatenate & Normalize
# Maybe I can do some fuzzy stuff here
audio_output = np.concatenate(audio_signal)
audio_output = audio_output / np.max(np.abs(audio_output)) * 0.8  # Normalize to 80% max volume

# Step 6: Play Sound Before Saving
sd.play(audio_output, SAMPLE_RATE)
sd.wait()

# Step 7: Save File
sf.write("encoded_hi_chord.wav", audio_output, SAMPLE_RATE)
print("Encoded file saved as 'encoded_hi_chord.wav'")
