import os
import numpy as np
from pydub import AudioSegment
import matplotlib.pyplot as plt
from scipy.io import wavfile
from matplotlib.colors import LinearSegmentedColormap

plt.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
plt.rc('text', usetex=True)  # use installed latex version to render labels

def plot_spectrogram(audio_path, output_image_path, font_size=14, multispecies=False, xlim = [0, 11]):
    freq_limit = 8000

    # Load audio file
    audio = AudioSegment.from_file(audio_path)

    # Export as wav to read with scipy
    audio.export("temp.wav", format="wav")
    sample_rate, samples = wavfile.read("temp.wav")

    # If stereo, convert to mono by averaging the channels
    if samples.ndim > 1:
        samples = samples.mean(axis=1)

    # Create the spectrogram using specgram
    plt.figure(figsize=(40, 10))
    Pxx, freqs, bins, im = plt.specgram(samples, NFFT=1024, Fs=sample_rate, noverlap=512, scale='dB')

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
    cmap = LinearSegmentedColormap.from_list("gray_r", colors, N=256)

    # Plot the spectrogram with the custom colormap
    plt.imshow(Pxx_dB, aspect='auto', extent=[bins.min(), bins.max(), freqs.min(), freqs.max() / 1000], origin='lower',
               cmap="gray_r")
    # plt.colorbar(label='Intensity [dB]')
    plt.ylim(0, freq_limit / 1000)  # Convert y-axis to kHz

    # Set ticks at each second on the x-axis
    plt.xticks(fontsize=font_size)
    plt.yticks(range(0, 9, 2), fontsize=font_size)
    if multispecies == False:
        plt.axvspan(3.148688, 4.661709, color='red', alpha=0.25, label='Common Redstart')

    if multispecies:
        plt.axvspan(3.148688, 4.661709, color='red', alpha=0.25, label='Common Redstart')
        plt.axvspan(4.148688, 6.356251, color='blue', alpha=0.25, label='Common Chaffinch')
        plt.axvspan(5.148688, 7.685151, color='green', alpha=0.25, label='Great Tit')

    # Add an annotation with an arrow to the interval
    plt.annotate('', xy=(3, freq_limit / 1000 + 0.2), xytext=(4.5, freq_limit / 1000 + 0.2),
                 arrowprops=dict(arrowstyle='<->', color='black'))

    plt.ylabel('Frequency [kHz]', fontsize=font_size)
    plt.xlabel('Time [s]', fontsize=font_size)
    if multispecies != None:
        plt.legend(fontsize=font_size)

    plt.tight_layout()
    plt.xlim(xlim)
    plt.savefig(output_image_path)
    print(f"Plot successfully saved at: {os.path.relpath(output_image_path)}")
    # plt.show()
