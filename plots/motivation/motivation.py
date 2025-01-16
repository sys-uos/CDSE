import os

import numpy as np
from pydub import AudioSegment
import matplotlib.pyplot as plt
from scipy.io import wavfile

def plot_motivational_figures(audio_paths, output_image_path, font_size=12, multispecies=False):
    freq_limit = 8000
    xticks_labels = ['04:18:05', '04:18:10', '04:18:15', '04:18:20', '04:18:25', '04:18:30']
    num_files = len(audio_paths)

    fig, axes = plt.subplots(num_files, 1, figsize=(27, 6 * num_files), sharex=True)
    for idx, (audio_path, ax) in enumerate(zip(audio_paths, axes)):
        # Load audio file
        audio = AudioSegment.from_file(audio_path)

        # Export as wav to read with scipy
        audio.export("temp.wav", format="wav")
        sample_rate, samples = wavfile.read("./temp.wav")

        # If stereo, convert to mono by averaging the channels
        if samples.ndim > 1:
            samples = samples.mean(axis=1)

        Pxx, freqs, bins, im = ax.specgram(samples, NFFT=1024, Fs=sample_rate, noverlap=512, scale='dB')
        # Apply frequency limit
        freq_limit_idx = np.where(freqs <= freq_limit)[0][-1]
        Pxx = Pxx[:freq_limit_idx + 1, :]
        freqs = freqs[:freq_limit_idx + 1]
        # Convert the power spectrogram (Pxx) to dB scale
        Pxx_dB = 10 * np.log10(Pxx)

        # Define a custom colormap similar to Audacity's
        colors = [
            (0, 0, 1),  # Blue
            (0, 1, 1),  # Cyan
            (0, 1, 0),  # Green
            (1, 1, 0),  # Yellow
            (1, 0, 0),  # Red
            (1, 0, 1)   # Pink
        ]
        ax.imshow(Pxx_dB, aspect='auto', extent=[bins.min(), bins.max(), freqs.min(), freqs.max() / 1000],
                  origin='lower', cmap="gray_r")
        ax.set_ylim(0, freq_limit / 1000)  # Convert y-axis to kHz
        duration = len(samples) / sample_rate
        xticks_positions = np.linspace(0, duration, len(xticks_labels))
        ax.set_xticks(xticks_positions)
        ax.set_xticklabels(xticks_labels, fontsize=font_size)
        ax.tick_params(axis='y', labelsize=font_size)

        ax.set_ylabel('Frequency [kHz]', fontsize=font_size)

        if idx == num_files - 1:
            ax.set_xlabel('Time [HH:MM:SS]', fontsize=font_size)

        # Add vertical lines at each x-tick
        for xtick in xticks_positions:
            ax.axvline(x=xtick, color='red', linestyle='--', linewidth=5)

        # Add an arrow at 04:18:15 and 6kHz frequency
        ax.annotate('', xy=(10.5, 5.9), xytext=(12, 6.5),
                    arrowprops=dict(arrowstyle='->', color='red', lw=3))

    plt.tight_layout()
    plt.savefig(output_image_path)
    print(f"Plot successfully saved at: {os.path.relpath(output_image_path)}")

    #plt.show()