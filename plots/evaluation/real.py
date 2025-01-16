import os

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import pickle
from matplotlib.colors import LinearSegmentedColormap

def plot_realworld_scenario(audio_file,
                            opath="./plots/final/Figure_7",
                            font_size=20):
    sample_rate, audio_data = wavfile.read(audio_file)
    if audio_data.ndim > 1:
        audio_data = audio_data[:, 0]

    # Set up figure with shared x-axis for all subplots
    fig, axs = plt.subplots(6, 1, figsize=(27, 6 * 6), sharex=True)

    Pxx, freqs, bins, im = axs[0].specgram(audio_data, NFFT=1024, Fs=sample_rate, noverlap=512, scale='dB')
    freq_limit = 12000
    freq_limit_idx = np.where(freqs <= freq_limit)[0][-1]
    Pxx = Pxx[:freq_limit_idx + 1, :]
    freqs = freqs[:freq_limit_idx + 1]
    Pxx_dB = 10 * np.log10(Pxx)

    colors = [
        (0, 0, 1),  # Blue
        (0, 1, 1),  # Cyan
        (0, 1, 0),  # Green
        (1, 1, 0),  # Yellow
        (1, 0, 0),  # Red
        (1, 0, 1)   # Pink
    ]
    cmap = LinearSegmentedColormap.from_list("gray_r", colors, N=256)

    axs[0].imshow(Pxx_dB, aspect='auto', extent=[bins.min(), bins.max(), freqs.min(), freqs.max() / 1000],
                  origin='lower', cmap="gray_r")
    axs[0].set_ylim(0, freq_limit / 1000)
    axs[0].set_ylabel('Frequency [Hz]', fontsize=font_size)
    axs[0].tick_params(axis='y', labelsize=font_size)

    # chaffinch groundtruth
    axs[0].axvspan(2.013, 4.432,   color='#377eb8', alpha=0.5, label='Common Chaffinch')
    axs[0].axvspan(8.947, 12.316,  color='#377eb8', alpha=0.5)
    axs[0].axvspan(18.802, 21.584, color='#377eb8', alpha=0.5)
    axs[0].axvspan(27.637, 31.286, color='#377eb8', alpha=0.5)
    axs[0].axvspan(41.058, 43.980, color='#377eb8', alpha=0.5)
    axs[0].axvspan(49.571, 52.171, color='#377eb8', alpha=0.5)
    axs[0].axvspan(58.155, 60.0,   color='#377eb8', alpha=0.5)
    axs[1].axvspan(2.013, 4.432,   color='#377eb8', alpha=0.5, label='Ground Truth')
    axs[1].axvspan(8.947, 12.316,  color='#377eb8', alpha=0.5)
    axs[1].axvspan(18.802, 21.584, color='#377eb8', alpha=0.5)
    axs[1].axvspan(27.637, 31.286, color='#377eb8', alpha=0.5)
    axs[1].axvspan(41.058, 43.980, color='#377eb8', alpha=0.5)
    axs[1].axvspan(49.571, 52.171, color='#377eb8', alpha=0.5)
    axs[1].axvspan(58.155, 60.0,   color='#377eb8', alpha=0.5)
    # song thrush
    axs[0].axvspan(4.942, 7.57,      color='#4daf4a', alpha=0.5, label='Song Trush')
    # axs[0].axvspan(16.583, 18.726, color='#4daf4a', alpha=0.5)
    axs[0].axvspan(21.731, 24.212,   color='#4daf4a', alpha=0.5)
    axs[0].axvspan(35.774, 40.191,   color='#4daf4a', alpha=0.5)
    axs[0].axvspan(47.614, 49.594,   color='#4daf4a', alpha=0.5)
    axs[2].axvspan(4.942, 7.57,      color='#4daf4a', alpha=0.5, label='Ground Truth')
    # axs[3].axvspan(16.583, 18.726, color='#4daf4a', alpha=0.5)
    axs[2].axvspan(21.731, 24.212,   color='#4daf4a', alpha=0.5)
    axs[2].axvspan(35.774, 40.191,   color='#4daf4a', alpha=0.5)
    axs[2].axvspan(47.614, 49.594,   color='#4daf4a', alpha=0.5)
    # robin
    axs[0].axvspan(5.926, 9.135,  color='#984ea3', alpha=0.5, label='European Robin')
    axs[0].axvspan(14.343, 18.802, color='#984ea3', alpha=0.5)
    axs[0].axvspan(22.226, 27.651, color='#984ea3', alpha=0.5)
    axs[0].axvspan(34.991, 39.338, color='#984ea3', alpha=0.5)
    axs[3].axvspan(5.926, 9.135,  color='#984ea3', alpha=0.5, label='Ground Truth')
    axs[3].axvspan(14.343, 18.802, color='#984ea3', alpha=0.5)
    axs[3].axvspan(24.226, 27.651, color='#984ea3', alpha=0.5)
    axs[3].axvspan(34.991, 39.338, color='#984ea3', alpha=0.5)
    # Tree Pipit
    axs[0].axvspan(4.459, 10.117, color='#ff7f00', alpha=0.3, label='Tree Pipit')
    axs[4].axvspan(4.459, 10.117, color='#ff7f00', alpha=0.3, label='Ground Truth')
    # redstart ground truth
    axs[0].axvspan(31.153, 33.621, color='#e41a1c', alpha=0.5, label='Common Redstart')
    axs[0].axvspan(52.989, 54.681, color='#e41a1c', alpha=0.5)
    axs[5].axvspan(31.153, 33.621, color='#e41a1c', alpha=0.5, label='Ground Truth')
    axs[5].axvspan(52.989, 54.681, color='#e41a1c', alpha=0.5)

    axs[0].legend(fontsize=font_size)

    def load_pickle(filename1):
        with open(filename1, 'rb') as fd:
            y = pickle.load(fd)
        y = y[60*48000:120*48000]
        if len(y) < 60*48000:
            for _ in range(len(y), 60*48000, 1):
                y.append(0.0)
        return y

    y_list = [
        load_pickle('./data/processed/real/cdse/3249-3423_0.1_Common_Chaffinch.pkl'),
        load_pickle('./data/processed/real/cdse/3249-3423_0.1_Song_Thrush.pkl'),
        load_pickle('./data/processed/real/cdse/3249-3423_0.1_European_Robin.pkl'),
        load_pickle('./data/processed/real/cdse/3249-3423_0.1_Tree_Pipit.pkl'),
        load_pickle('./data/processed/real/cdse/3249-3423_0.1_Common_Redstart.pkl')
    ]
    labels = ['Common Chaffinch', 'Song Thrush', 'European Robin', 'Tree Pipit', 'Common Redstart']
    colors = ["#377eb8", "#4daf4a", "#984ea3", "#ff7f00", "#e41a1c"]

    x = np.linspace(0, len(audio_data) / sample_rate, len(audio_data))
    for i in range(1, 6):
        axs[i].plot(x, y_list[i-1], color=colors[i-1], label=labels[i-1], linewidth=4)
        axs[i].set_ylabel('Confidence', fontsize=font_size)
        axs[i].set_ylim([0.0, 0.6])
        axs[i].legend(fontsize=font_size, loc="upper right")
        axs[i].tick_params(axis='y', labelsize=font_size)
        axs[i].grid(axis='y')

    axs[-1].set_xlabel('Time [s]', fontsize=font_size)
    plt.xticks(fontsize=font_size)
    plt.xlim([0, 60])
    plt.tight_layout()

    plt.savefig(opath)
    print(f"Plot successfully saved at: {os.path.relpath(opath)}")

    # plt.show()
