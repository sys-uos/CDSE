import os.path
import pickle
import numpy as np
from scipy.io import wavfile

from plots.dataset.dataset import plot_spectrogram
from plots.evaluation.real import plot_realworld_scenario
from plots.evaluation.simulation import lineplot_confidences_over_time_and_distance, plot_time_differences, \
    plot_multisource_lineplot_confidences_over_time_and_distance, plot_impact_of_threshold_singlespecies, \
    plot_impact_of_threshold_multispecies, plot_evaluation_of_timeDifferences_SingleSource, \
    plot_evaluation_of_timdeDifferences_MultiSource
from plots.graphical_abstract.graphical_abstract import graphical_abstract, graphical_abstract_2
from plots.motivation.motivation import plot_motivational_figures
from scripts.cdse import CDSE
from scripts.parsing.parse_data import Parser
from scripts.tdoa import gcc_phat


def main_paper_plots():
    # print("# --- Graphical Abstract --- #")
    # graphical_abstract(audio_paths=["./data/raw/simulation/multi_sources/audio/node[50]/dummy.wav"],
    #                      output_image_path = "./plots/final/graphical_abstract/Graphical_Abstract_Figure_1.png", font_size=20)
    # graphical_abstract_2(output_image_path='./plots/final/graphical_abstract/Graphical_Abstract_Figure_2.png', font_size=20)
    #
    # print("# --- Motivation --- #")
    # plot_motivational_figures(audio_paths=["./data/processed/real/audio/26_20230603_040000_subset.wav",
    #                                "./data/processed/real/audio/27_20230603_040000_subset.wav",
    #                                "./data/processed/real/audio/28_20230603_040000_subset.wav"],
    #                           output_image_path="./plots/final/Figure_1.pdf", font_size=36, multispecies=True)
    #
    # print("# --- Evaluation Dataset --- #")
    # plot_spectrogram(audio_path = "./data/raw/simulation/single_source/audio/node[50]/microphones[0].wav",
    #                      output_image_path = "./plots/final/Figure_2_2.pdf", font_size=36, multispecies=False)
    # plot_spectrogram(audio_path="./data/processed/real/audio/real_soundscape_with_low_interference.wav",
    #                      output_image_path="./plots/final/Figure_2_3.pdf", font_size=36, multispecies=None)
    # plot_spectrogram(audio_path = "./data/raw/simulation/multi_sources/audio/node[50]/microphones[0].wav",
    #                      output_image_path = "./plots/final/Figure_2_4.pdf", font_size=36, multispecies=True)
    # plot_spectrogram(audio_path = "./data/processed/real/audio/node29_20230603_3306-3366.wav",
    #                      output_image_path = "./plots/final/Figure_2_5.pdf", font_size=36, multispecies=None, xlim=[0, 60])
    #
    # print("# --- Evaluation Single_Source --- #")
    # lineplot_confidences_over_time_and_distance(path='./data/processed/simulation/single_source/cdse/0.5_0.0.pkl',
    #                                             ofile='./plots/final/Figure_3_1.pdf', font_size=26, multi_source=False)
    # lineplot_confidences_over_time_and_distance(path='./data/processed/simulation/single_source/cdse/1.0_0.0.pkl',
    #                                             ofile='./plots/final/Figure_3_2.pdf', font_size=26, multi_source=False)
    # lineplot_confidences_over_time_and_distance(path='./data/processed/simulation/single_source/cdse/1.5_0.0.pkl',
    #                                             ofile='./plots/final/Figure_3_3.pdf', font_size=26, multi_source=False)

    # print("# --- Evaluation Single_Source Time Difference Estimation --- #")
    # for i, sensitivity in enumerate([0.5, 0.75, 1.0, 1.25, 1.5]):
    #     for threshold in [0.0]:
    #         filepath = f"./data/processed/simulation/single_source/tdoa/{sensitivity}_{threshold}.pkl"
    #         plot_time_differences(filepath, opath=f"./plots/final/Figure_6_{i}.pdf",
    #                               threshold_ms=10.0, font_size=20)

    print("# --- Evaluation Multi_Source Time Difference Estimation --- #")
    # plot_multisource_lineplot_confidences_over_time_and_distance(
    #     ipaths=[
    #         './data/processed/simulation/multi_source/cdse/Common_Redstart/0.5_0.0.pkl',
    #         './data/processed/simulation/multi_source/cdse/Common_Chaffinch/0.5_0.0.pkl',
    #         './data/processed/simulation/multi_source/cdse/Great_Tit/0.5_0.0.pkl'],
    #     opath='./plots/final/Figure_4_1.pdf', font_size=16)
    # plot_multisource_lineplot_confidences_over_time_and_distance(
    #     ipaths=[
    #         './data/processed/simulation/multi_source/cdse/Common_Redstart/1.0_0.0.pkl',
    #         './data/processed/simulation/multi_source/cdse/Common_Chaffinch/1.0_0.0.pkl',
    #         './data/processed/simulation/multi_source/cdse/Great_Tit/1.0_0.0.pkl'],
    #     opath='./plots/final/Figure_4_2.pdf', font_size=16)
    # plot_multisource_lineplot_confidences_over_time_and_distance(
    #     ipaths=[
    #         './data/processed/simulation/multi_source/cdse/Common_Redstart/1.5_0.0.pkl',
    #         './data/processed/simulation/multi_source/cdse/Common_Chaffinch/1.5_0.0.pkl',
    #         './data/processed/simulation/multi_source/cdse/Great_Tit/1.5_0.0.pkl'],
    #     opath='./plots/final/Figure_4_3.pdf', font_size=16)
    #
    # print("# --- Evaluation Threshold's Impact --- #")
    # plot_impact_of_threshold_singlespecies(ipaths_redstart = [
    #         './data/processed/simulation/single_source/cdse/1.5_0.0.pkl',
    #         './data/processed/simulation/single_source/cdse/1.5_0.1.pkl',
    #         './data/processed/simulation/single_source/cdse/1.5_0.2.pkl',
    #         './data/processed/simulation/single_source/cdse/1.5_0.3.pkl',
    #         './data/processed/simulation/single_source/cdse/1.5_0.4.pkl',
    #         './data/processed/simulation/single_source/cdse/1.5_0.5.pkl'
    #     ], opath="./plots/final/Figure_7.pdf")
    # plot_impact_of_threshold_multispecies(opath="./plots/final/Figure_8")  # ".NUMBER_pdf" is appended automatically
    #
    # print("# --- Evaluation TimeDifference Accuracy --- #")
    # plot_evaluation_of_timeDifferences_SingleSource(base_path = './data/processed/simulation/single_source/tdoa/',
    #                                                 opath="./plots/final/Figure_9_1.pdf")
    # plot_evaluation_of_timdeDifferences_MultiSource(base_path = './data/processed/simulation/multi_source/tdoa/',
    #                                                 opath="./plots/final/Figure_9_2.pdf")
    #
    # print("# --- Evaluation real-world Data --- #")
    plot_realworld_scenario(audio_file ='./data/processed/real/audio/node29_20230603_3306-3366.wav',
                            opath="./plots/final/Figure_5.pdf",
                            font_size=20)

def main_paper_process_data():
    # --- Parse the classification results of the simulated setup, calculate CDSE and calculate the time differences --- #
    # --- Run scenario_single_species() for the single_species scenario
    # --- Run scenario_multi_species() for the multi_species scenario
    # --- Run scenario_real_world() for the real-world scenario

    def scenario_single_species():
        simulation_dir = "./data/processed/simulation/single_source/classifications/"
        dirs = sorted([d for d in os.listdir(simulation_dir) if os.path.isdir(os.path.join(simulation_dir, d))])
        for dir_name in dirs[0:]:
            dir_path = os.path.join(simulation_dir, dir_name)
            print(f"Parsing directory: {dir_name}")

            # -- Create a new Parser instance and parse the directory -- #
            parser = Parser()
            data_dict = parser.parse_simulated_directory(
                dir_path,
                rows_per_chunk=1440000,
                use_columns=None,
                column_names=["Start (s)", "End (s)", "Confidence"]
            )

            # -- Classifications might contain missing values for some timestamp, find and fill them -- #
            for key in data_dict.keys():
                print(f"Check and Fill missing values for {key}")
                data_dict[key] = Parser.check_and_fill_missing_values(df=data_dict[key], min_start=0, max_end=(10-3)*48000,
                                                                      chunk_size=144000, start_col='start', end_col='end',
                                                                      step=1, confidence_col='confidence', default_confidence=0.0)

            # -- Apply a confidence theshold when calculating cdse -- #
            for confidence_threshold in np.arange(0.0, 0.6, 0.1):
                cdse_results = {}
                for key, dataframe in data_dict.items():
                    print(f"Applying CDSE for node {key}")
                    cdse = CDSE()
                    cdse.set_data_from_parser(dataframe)
                    series = cdse.cdse_from_dataframe(
                        outpath=None,
                        end_col='end',
                        audio_sampling_frequency=48000,
                        audio_max_duration=None,
                        window_size=144000,
                        confidence_threshold=confidence_threshold,
                        progress_updates=True
                    )
                    cdse_results[key] = series

                # -- Save the data to disc -- #
                dest_dir = "./data/processed/simulation/single_source/cdse/"
                os.makedirs(dest_dir, exist_ok=True)
                sensitivity = dir_name.split('_')[0]
                dest_fname = f"{sensitivity}_{confidence_threshold:.1f}.pkl"
                fpath = os.path.join(dest_dir, dest_fname)
                with open(fpath, 'wb') as fd:
                    pickle.dump(cdse_results, fd)

        # -- Calculate time differences between all nodes  -- #
        sensitivities = [0.5, 0.75, 1.0, 1.25, 1.5]
        for sensitivity in sensitivities:
            for threshold in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]:
                pkl_file = open(f'./data/processed/simulation/single_source/cdse/{sensitivity}_{threshold}.pkl','rb')
                data = pickle.load(pkl_file)
                td = {}
                for source in data.keys():
                    print(f"Calculate TDOA for {source} and all others")
                    td[source] = []
                    for dest in data.keys():
                        tau, cc = gcc_phat(data[source], data[dest], fs=1, interp=1,
                                           max_tau=140 * (abs(source - dest) + 10))
                        td[source].append(tau)

                dest_dir = "./data/processed/simulation/single_source/tdoa_new/"
                os.makedirs(dest_dir, exist_ok=True)
                fpath = os.path.join(dest_dir, f"{sensitivity}_{threshold}.pkl")
                with open(fpath, 'wb') as handle:
                    pickle.dump(td, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def scenario_multi_species():
        bird_species = [
            "Common_Redstart",
            "Common_Chaffinch",
            "Great_Tit",
        ]
        # Base directory path
        base_dir_path = "data/processed/simulation/multi_source/classifications/"
        # Iterate over each bird species and run the parser
        for species in bird_species[2:3]:
            print(f"Parsing directory for species: {species}")
            simulation_dir = os.path.join(base_dir_path, species)
            dirs = sorted([d for d in os.listdir(simulation_dir) if os.path.isdir(os.path.join(simulation_dir, d))])
            for dir_name in dirs:
                dir_path = os.path.join(simulation_dir, dir_name)
                # -- Create a new Parser instance and parse the directory -- #
                parser = Parser()
                data_dict = parser.parse_simulated_directory(
                    dir_path,
                    rows_per_chunk=1440000,
                    use_columns=None,
                    column_names=["Start (s)", "End (s)", "Confidence"]
                )

                # -- Classifications might contain missing values for some timestamp, find and fill them -- #
                for key in data_dict.keys():
                    print(f"Check and Fill missing values for {key}")
                    data_dict[key] = Parser.check_and_fill_missing_values(df=data_dict[key], min_start=0,
                                                                          max_end=(11 - 3) * 48000,
                                                                          chunk_size=144000, start_col='start',
                                                                          end_col='end',
                                                                          step=1, confidence_col='confidence', default_confidence=0.0)

                print(f"Finished parsing: {species}")

                # -- Apply a confidence theshold when calculating cdse -- #
                for confidence_threshold in np.arange(0.0, 0.6, 0.1):
                    dest_dir = f"./data/processed/simulation/multi_source/cdse/{species}"
                    os.makedirs(dest_dir, exist_ok=True)
                    sensitivity = dir_name.split('_')[0]
                    dest_fname = f"{sensitivity}_{confidence_threshold:.1f}.pkl"
                    fpath = os.path.join(dest_dir, dest_fname)
                    # if os.path.isfile(fpath):
                    #     continue

                    cdse_results = {}
                    for key, dataframe in data_dict.items():
                        print(f"Applying CDSE for node {key}")
                        cdse = CDSE()
                        cdse.set_data_from_parser(dataframe)
                        series = cdse.cdse_from_dataframe(
                            outpath=None,
                            end_col='end',
                            audio_sampling_frequency=48000,
                            audio_max_duration=None,
                            window_size=144000,
                            confidence_threshold=confidence_threshold,
                            progress_updates=True
                        )
                        cdse_results[key+1] = series  # the key now refers to the distance

                    # -- Save the data to disc -- #
                    dest_dir = f"./data/processed/simulation/multi_source/cdse/{species}"
                    os.makedirs(dest_dir, exist_ok=True)
                    sensitivity = dir_name.split('_')[0]
                    dest_fname = f"{sensitivity}_{confidence_threshold:.1f}.pkl"
                    fpath = os.path.join(dest_dir, dest_fname)
                    with open(fpath, 'wb') as fd:
                        pickle.dump(cdse_results, fd)

        # -- Calculate time differences between all nodes  -- #
        for species in bird_species:
            dir_path = f"./data/processed/simulation/multi_source/cdse/{species}"
            sensitivities = [0.5, 0.75, 1.0, 1.25, 1.5]
            for sensitivity in sensitivities:
                for threshold in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]:
                    pkl_file = open(os.path.join(dir_path, f"{sensitivity}_{threshold}.pkl"), 'rb')
                    data = pickle.load(pkl_file)
                    td = {}
                    for source in data.keys():
                        print(f"Calculate TDOA for {species}: Distance {source} and others")
                        td[source] = []
                        for dest in data.keys():
                            tau, cc = gcc_phat(data[source], data[dest], fs=1, interp=1,
                                               max_tau=140 * (abs(source - dest) + 10))
                            td[source].append(tau)

                    dest_dir = "./data/processed/simulation/single_source/tdoa/"
                    os.makedirs(dest_dir, exist_ok=True)
                    fpath = os.path.join(dest_dir, f"{sensitivity}_{threshold}.pkl")
                    with open(fpath, 'wb') as handle:
                        pickle.dump(td, handle, protocol=pickle.HIGHEST_PROTOCOL)


    def scenario_real_world():
        bird_species = [
            "Common_Chaffinch",
            "Common_Redstart",
            "European_Robin",
            "Song_Thrush",
            "Tree_Pipit"
        ]
        # Base directory path
        base_dir_path = "data/processed/real/classifications/species_selective/"
        # Iterate over each bird species and run the parser
        for species in bird_species:
            dir_path = f"{base_dir_path}{species}/sensitivity-1.5/"
            print(f"Parsing directory for species: {species}")
            parser = Parser()
            df = parser.parse_directory_concat_data(
                dir_path,
                rows_per_chunk=1440000,
                use_columns=None,
                column_names=["Start (s)", "End (s)", "Confidence"]
            )
            print(f"Check and Fill missing values and fill them")
            df = Parser.check_and_fill_missing_values(df=df, min_start=0, max_end=(180-3) * 48000,
                                                    chunk_size=144000, start_col='start', end_col='end',
                                                    step=1, confidence_col='confidence',
                                                    default_confidence=0.0)
            print(f"Finished parsing: {species}")

            confidence_threshold = 0.0
            cdse = CDSE()
            cdse.set_data_from_parser(df)
            series = cdse.cdse_from_dataframe(
                outpath=None,
                end_col='end',
                audio_sampling_frequency=48000,
                audio_max_duration=None,
                window_size=144000,
                confidence_threshold=confidence_threshold,
                progress_updates=True
            )

            # -- Save the data to disc -- #
            dest_dir = "./data/processed/real/cdse/"
            os.makedirs(dest_dir, exist_ok=True)
            dest_fname = f"3249-3423_{confidence_threshold:.1f}_{species}.pkl"
            fpath = os.path.join(dest_dir, dest_fname)
            with open(fpath, 'wb') as fd:
                pickle.dump(series, fd)

    # scenario_single_species()
    # scenario_multi_species()
    scenario_real_world()

def main_minimal_usage_example():
    # --- A simple example how to apply CDSE on BirdNET classifications --- #
    parser = Parser()
    parser.parse_textfile("./example_data/data/processed/classifications/dummy.BirdNET.results.txt",
                           rows_per_chunk=1440000, use_columns=None, column_names=None)
    df = parser.check_and_fill_missing_values(df=parser.data, chunk_size=144000, start_col='start', end_col='end', step=1,
                                         confidence_col='confidence', default_confidence=0.0)

    # Important: CDSE expects no gaps in the classification data!
    cdse = CDSE()
    cdse.set_data_from_parser(df)
    cdse_data = cdse.cdse_from_dataframe(outpath=None,
                            end_col='end',
                            audio_sampling_frequency=48000,
                            audio_max_duration=None,
                            window_size=144000,
                            confidence_threshold=0.1,
                            progress_updates=True)


    # Plot spectrogram for audio and CDSE data
    import matplotlib.pyplot as plt
    sample_rate, audio_data = wavfile.read("./example_data/data/raw/dummy.wav")
    fig, axs = plt.subplots(2, 1, sharex=True)

    Pxx, freqs, bins, im = axs[0].specgram(audio_data, NFFT=1024, Fs=sample_rate, noverlap=512, scale='dB')
    freq_limit = 12000
    freq_limit_idx = np.where(freqs <= freq_limit)[0][-1]
    Pxx = Pxx[:freq_limit_idx + 1, :]
    freqs = freqs[:freq_limit_idx + 1]
    Pxx_dB = 10 * np.log10(Pxx)
    axs[0].imshow(Pxx_dB, aspect='auto', extent=[bins.min(), bins.max(), freqs.min(), freqs.max() / 1000],
                  origin='lower', cmap="gray_r")
    axs[0].set_ylim(0, freq_limit / 1000)
    axs[0].set_ylabel('Frequency [Hz]', fontsize=12)
    axs[0].tick_params(axis='y', labelsize=12)
    axs[0].axvspan(3.148688, 4.661709, color='red', alpha=0.3, label='Bird Signal of Common Redstart')

    x = np.linspace(0, len(audio_data)  / sample_rate, len(audio_data)-sample_rate)
    axs[1].plot(x, cdse_data, color='r', label="CDSE of Common Redstart", linewidth=2)
    axs[1].set_ylabel('Confidence', fontsize=12)
    axs[1].set_ylim([0.0, 1.0])
    axs[1].legend(fontsize=12, loc="upper right")
    axs[1].tick_params(axis='y', labelsize=12)
    axs[1].grid(axis='y')
    axs[1].axvspan(3.148688, 4.661709, color='red', alpha=0.3, label='CDSE of Common Redstart')
    axs[1].annotate('Impact of sliding window',
                 xy=(1.81, 0.8),  # Arrow tip at the start of axvspan
                 xytext=(3.148688, 0.8),  # Text position
                 arrowprops=dict(facecolor='black', shrink=0.05),
                 horizontalalignment='left',
                 verticalalignment='bottom')
    axs[1].annotate('',
                 xy=(6, 0.8),  # Arrow tip at the end of axvspan
                 xytext=(4.661709, 0.8),  # Text position
                 arrowprops=dict(facecolor='black', shrink=0.05),
                 horizontalalignment='right',
                 verticalalignment='bottom')

    axs[-1].set_xlabel('Time [s]', fontsize=12)
    plt.xticks(fontsize=12)
    plt.xlim([0, 10])
    plt.tight_layout()
    plt.savefig("pics/dummy_cdse.png")


if __name__ == '__main__':
    # To reproduce plots
    # main_paper_process_data()
    main_paper_plots()

    # Checkout how CDSE is applied
    # main_minimal_usage_example()