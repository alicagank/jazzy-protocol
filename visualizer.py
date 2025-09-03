import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
import librosa
import librosa.display
from scipy.fft import fft

# Load the audio file (change this to 'x.wav' in the same directory)
file_path = 'encoded_hi.mp3'
y, sr = librosa.load(file_path, sr=None)

# Set up the plot to display multiple visualizations
fig, axs = plt.subplots(2, 2, figsize=(12, 10))

# 1. Plot the waveform of the audio file
axs[0, 0].plot(np.linspace(0, len(y)/sr, len(y)), y, color='b')
axs[0, 0].set_title('Waveform')
axs[0, 0].set_xlabel('Time (s)')
axs[0, 0].set_ylabel('Amplitude')

# 2. Plot the Spectrogram (using librosa)
D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
librosa.display.specshow(D, ax=axs[0, 1], y_axis='log', x_axis='time', sr=sr)
axs[0, 1].set_title('Spectrogram')
axs[0, 1].set_xlabel('Time (s)')
axs[0, 1].set_ylabel('Frequency (Hz)')
fig.colorbar(librosa.display.specshow(D, ax=axs[0, 1], y_axis='log', x_axis='time', sr=sr))

# 3. Plot the Frequency Domain (FFT)
n = len(y)
frequencies = np.fft.fftfreq(n, 1/sr)
fft_vals = np.abs(fft(y))
axs[1, 0].plot(frequencies[:n//2], fft_vals[:n//2])
axs[1, 0].set_title('Frequency Domain (FFT)')
axs[1, 0].set_xlabel('Frequency (Hz)')
axs[1, 0].set_ylabel('Magnitude')

# 4. Plot the Mel Spectrogram
S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
S_dB = librosa.power_to_db(S, ref=np.max)
librosa.display.specshow(S_dB, ax=axs[1, 1], x_axis='time', y_axis='mel', sr=sr)
axs[1, 1].set_title('Mel Spectrogram')
axs[1, 1].set_xlabel('Time (s)')
axs[1, 1].set_ylabel('Mel Frequency')

# Adjust layout and display the plots
plt.tight_layout()
plt.show()
