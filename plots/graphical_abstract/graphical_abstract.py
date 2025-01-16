import pickle
import numpy as np
from pydub import AudioSegment
import matplotlib.pyplot as plt
from scipy.io import wavfile
from matplotlib.colors import LinearSegmentedColormap
import os


# plt.rc("font", family='serif')  # select a fitting font type
# plt.rc('text', usetex=True)  # use installed latex version to render labels

# plt.rcParams.update({
#     "text.usetex": True,
#     "font.family": "sans-serif",
#     "font.sans-serif": "Helvetica",
# })


def graphical_abstract(audio_paths, output_image_path, font_size=12):
    freq_limit = 8000
    xticks_labels = ['0', '2', '4', '6', '8', '11']
    num_files = len(audio_paths)

    fig, axes = plt.subplots(num_files, 1, figsize=(27, 6 * num_files), sharex=True)
    axes = [axes]
    for idx, (audio_path, ax) in enumerate(zip(audio_paths, axes)):
        audio = AudioSegment.from_file(audio_path)
        audio.export("temp.wav", format="wav")
        sample_rate, samples = wavfile.read("./temp.wav")

        # If stereo, convert to mono by averaging the channels
        if samples.ndim > 1:
            samples = samples.mean(axis=1)

        # Create the spectrogram using specgram
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
        cmap = LinearSegmentedColormap.from_list("gray_r", colors, N=256)

        # Plot the spectrogram with the custom colormap
        ax.imshow(Pxx_dB, aspect='auto', extent=[bins.min(), bins.max(), freqs.min(), freqs.max() / 1000],
                  origin='lower', cmap="gray_r")
        # ax.colorbar(label='Intensity [dB]')
        ax.set_ylim(0, freq_limit / 1000)  # Convert y-axis to kHz

        # Set custom x-ticks labels
        duration = len(samples) / sample_rate
        xticks_positions = np.linspace(0, duration, len(xticks_labels))
        xticks_positions = [int(val) for val in xticks_positions]
        ax.set_xticks(xticks_positions)
        ax.set_xticklabels(xticks_labels, fontsize=font_size)
        ax.tick_params(axis='y', labelsize=font_size)

        # Add the shaded regions between specific times
        ax.axvspan(3.148688, 4.661709, color='red', alpha=0.3, label='Common Redstart')
        ax.axvspan(4.148688, 6.356251, color='blue', alpha=0.3, label='Common Chaffinch')
        ax.axvspan(5.148688, 7.685151, color='green', alpha=0.3, label='Great Tit')

        ax.set_ylabel('Frequency [kHz]', fontsize=font_size)

        if idx == num_files - 1:
            ax.set_xlabel('Time [s]', fontsize=font_size)

        # Add legend with title and opacity
        legend = ax.legend(title="Acoustic Signals of Bird Species", loc='upper right', fontsize=font_size)
        legend.get_frame().set_alpha(0.5)  # Set legend box opacity
        legend.set_title("Acoustic Signals Intervals", prop={'size': font_size + 2})  # Increase title font size

    plt.tight_layout()

    directory = os.path.dirname(output_image_path)
    os.makedirs(directory, exist_ok=True)
    plt.savefig(output_image_path)
    print(f"Plot successfully saved at: {os.path.relpath(output_image_path)}")
    # plt.show()


def graphical_abstract_2(output_image_path, font_size=16, redstart_paths=None):
    if redstart_paths is None:
        redstart_paths = {
            "15m": './data/processed/graphical_abstract/redstart_15m.pickle',
            "20m": './data/processed/graphical_abstract/redstart_20m.pickle',
            "35m": './data/processed/graphical_abstract/redstart_35m.pickle',
            "25m": './data/processed/graphical_abstract/redstart_25m.pickle'
        }

    redstarts = {}
    for key, path in redstart_paths.items():
        with open(path, 'rb') as pkl_file:
            redstarts[key] = pickle.load(pkl_file)

    # Creating the plot with the required settings
    plt.figure(figsize=(24, 6))
    plt.plot(redstarts["15m"], c='red', linestyle='-', linewidth=5.0, label="15m")
    plt.plot(redstarts["20m"], c='red', linestyle=':', linewidth=5.0, label="20m")
    plt.plot(redstarts["25m"], c='red', linewidth=5.0, linestyle='--', label="25m")
    plt.plot(redstarts["35m"], c='red', linewidth=5.0, linestyle='-.', label="35m")

    plt.xticks(
        [0 * 48000, 2 * 48000, 4 * 48000, 6 * 48000, 8 * 48000, 11 * 48000],
        [0, 2, 4, 6, 8, 11],
        fontsize=24
    )

    # Set x-axis limits from 0 to 10
    plt.xlim(0, 10 * 48000)

    plt.xlabel("Time [s]", fontsize=24)
    plt.ylabel("Deduced Probability", fontsize=24)

    plt.legend(fontsize=font_size)
    plt.tight_layout()

    # Ensure the output directory exists
    directory = os.path.dirname(output_image_path)
    os.makedirs(directory, exist_ok=True)
    plt.savefig(output_image_path)
    print(f"Plot successfully saved at: {os.path.relpath(output_image_path)}")
    # plt.show()


if __name__ == "__main__":
    graphical_abstract(audio_paths=["../../data/processed/graphical_abstract/node50_exemplary.wav"],
                       output_image_path ="../final/graphical_abstract/Graphical_Abstract_Figure_1.png", font_size=20)

    custom_paths = {
        "15m": '../../data/processed/graphical_abstract/redstart_15m.pickle',
        "20m": '../../data/processed/graphical_abstract/redstart_20m.pickle',
        "25m": '../../data/processed/graphical_abstract/redstart_25m.pickle',
        "35m": '../../data/processed/graphical_abstract/redstart_35m.pickle'
    }
    graphical_abstract_2(redstart_paths=custom_paths,
                         output_image_path='../final/graphical_abstract/Graphical_Abstract_Figure_2.png', font_size=20)
