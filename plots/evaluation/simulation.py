import os
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Patch
from matplotlib.gridspec import GridSpec
import pickle
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable


# plt.rc("font", family='serif')  # select a fitting font type
plt.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
plt.rc('text', usetex=True)  # use installed latex version to render labels


def lineplot_confidences_over_time_and_distance(path, multi_source=True, ofile='unnamed_plot.pdf', font_size=10):
    pkl_file = open(path, 'rb')
    data = pickle.load(pkl_file)
    print("Read combined_data")

    # Set up the color map
    num_lines = len(data)
    colors = plt.cm.Reds_r(np.linspace(0, 1, num_lines))  # Using 'viridis'

    # Create a figure and GridSpec layout
    fig = plt.figure(figsize=(24, 6))
    gs = GridSpec(1, 3, width_ratios=[1, 3, 1])

    ax_main = fig.add_subplot(gs[1])  # Main plot
    ax_sub1 = fig.add_subplot(gs[0])  # Subplot to the left of the main plot
    ax_sub2 = fig.add_subplot(gs[2])  # Subplot to the right of the main plot

    # Create a scalar mappable for colorbar creation
    sm = plt.cm.ScalarMappable(cmap='Reds_r', norm=plt.Normalize(vmin=min(data.keys()), vmax=max(data.keys())))
    sm.set_array([])  # You need this line because you're plotting manually

    # Plot each series in the dictionary on the main plot
    for (key, values), color in zip(data.items(), colors):
        ax_main.plot(values, color=color)

    # Customize the main plot
    ax_main.set_xlabel('time [s]', fontsize=font_size)
    cbar = fig.colorbar(sm, ax=ax_sub2, label='source-to-node distance',
                        pad=0.22)  # Add a colorbar with a label, associate with ax_main
    cbar.ax.tick_params(labelsize=font_size)
    cbar.set_label('source-to-node distance', fontsize=font_size)
    ax_main.grid(True)  # Enable grid for better readability

    # Fill the area between the vertical lines on the main plot
    # Create a custom legend patch for the shaded area
    if multi_source:
        ax_main.axvspan(3 * 48000, 4.53 * 48000, color='red', alpha=0.25, label='Phoenicurus phoenicurus')
        bird_sound_patch = Patch(color='red', alpha=0.25, label="Phoenicurus phoenicurus")
        ax_main.axvspan(5 * 48000, 7.5365 * 48000, color='green', alpha=0.25, label='Parus major')
        bird_sound_patch3 = Patch(color='blue', alpha=0.25, label="Parus major")
        ax_main.axvspan(4 * 48000, 6.207563 * 48000, color='blue', alpha=0.25, label='Fringilla coelebs')
        bird_sound_patch2 = Patch(color='green', alpha=0.5, label="Fringilla coelebs")
        ax_main.legend(title="timing of species sound (at source)", handles=[bird_sound_patch, bird_sound_patch2, bird_sound_patch3], fontsize=font_size, title_fontsize=font_size)
    else:
        ax_main.axvspan(3 * 48000, 4.53 * 48000, color='red', alpha=0.25, label='Bird sound')
        bird_sound_patch = Patch(color='red', alpha=0.25, label="Common Redstart")
        # Add the legend with the custom patch
        ax_main.legend(title="emission time of species sound ", handles=[bird_sound_patch], fontsize=font_size, title_fontsize=font_size)

    # Convert x-axis from samples to seconds for the main plot
    sampling_frequency = 48000  # 48 kHz
    max_sample_number = max(len(values) for values in data.values())  # Find the longest array in data
    time_ticks = np.arange(0, max_sample_number + sampling_frequency,
                           sampling_frequency)  # Generate time ticks every 1 second
    ax_main.set_xticks(time_ticks)  # Set custom ticks
    ax_main.set_xticklabels([f"{x / sampling_frequency:.1f}" for x in time_ticks], fontsize=font_size)  # Label ticks in seconds

    ax_main.set_yticks([round(val,1) for val in np.arange(0, 1.1, 0.1)])  # Set y-ticks from min to max with a step of 0.1
    ax_main.set_yticklabels([round(val,1) for val in np.arange(0, 1.1, 0.1)], fontsize=font_size)  # Set y-tick labels

    ax_sub1.set_yticks([round(val,1) for val in np.arange(0, 1.1, 0.1)])  # Set y-ticks from min to max with a step of 0.1
    ax_sub1.set_yticklabels([round(val,1) for val in np.arange(0, 1.1, 0.1)], fontsize=font_size)  # Set y-tick labels
    ax_sub2.set_yticks([round(val,1) for val in np.arange(0, 1.1, 0.1)])  # Set y-ticks from min to max with a step of 0.1
    ax_sub2.set_yticklabels([round(val,1) for val in np.arange(0, 1.1, 0.1)], fontsize=font_size)  # Set y-tick labels

    ax_main.set_xlim([0, 10 * sampling_frequency])  # Adjust xlim to match the 10-second window
    ax_main.set_ylim([0.0, 1.0])

    # Plot the zoomed-in subplots
    for (key, values), color in zip(data.items(), colors):
        ax_sub1.plot(values, color=color)
        ax_sub2.plot(values, color=color)

    # Customize the first subplot
    ax_sub1.set_xlim([0.0*sampling_frequency, 1.0 * sampling_frequency])  # Focus on the 0 to 2-second window
    ax_sub1.set_ylim([0.0, 0.50])
    ax_sub1.grid(True)
    ax_sub1.set_ylabel('confidence', fontsize=font_size)

    # Move y-ticks and labels to the right side for the first subplot
    ax_sub1.yaxis.tick_left()
    ax_sub1.yaxis.set_label_position("left")

    # Customize the second subplot
    ax_sub2.set_xlim([5.5 * sampling_frequency, 6.5 * sampling_frequency])  # Focus on the 6 to 7-second window
    ax_sub2.set_ylim([0.0, 0.50])
    ax_sub2.grid(True)
    # ax_sub2.set_xlabel('time [s]', fontsize=font_size)

    # Move y-ticks and labels to the right side for the second subplot
    ax_sub2.yaxis.tick_right()
    ax_sub2.yaxis.set_label_position("right")

    # Convert x-axis from samples to seconds for the subplots
    time_ticks_sub1 = np.arange(0.0, 1.0  * sampling_frequency + sampling_frequency,
                                sampling_frequency / 1)  # Generate time ticks every 1 second
    ax_sub1.set_xticks(time_ticks_sub1)  # Set custom ticks
    ax_sub1.set_xticklabels([f"{x / sampling_frequency:.1f}" for x in time_ticks_sub1], fontsize=font_size)  # Label ticks in seconds

    time_ticks_sub2 = np.arange(5.5 * sampling_frequency, 6.5 * sampling_frequency + sampling_frequency,
                                sampling_frequency / 1)  # Generate time ticks every 1 second
    ax_sub2.set_xticks(time_ticks_sub2)  # Set custom ticks
    ax_sub2.set_xticklabels([f"{x / sampling_frequency:.1f}" for x in time_ticks_sub2], fontsize=font_size)  # Label ticks in seconds

    # Add label with arrows below the x-axis in the middle of sub1
    x_middle = (0.6 * sampling_frequency)
    # Adding the arrow
    ax_sub1.annotate('', xy=(0.2, -0.02), xycoords='axes fraction', xytext=(0.9, -0.02),
            arrowprops=dict(arrowstyle="<-", color='black'))
    ax_sub1.text(x_middle, -0.03, 'time shift', ha='center', fontsize=font_size, va='center',  fontstyle='italic')

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.1)  # Adjust the width space between subplots

    plt.savefig(ofile)
    print(f"Plot successfully saved at: {os.path.relpath(ofile)}")

    # plt.show()  # Display the plot


def plot_time_differences(filepath, opath, threshold_ms=50.0, font_size=16):
    with open(filepath, 'rb') as file:
        td = pickle.load(file)

    td_difference_to_expected = {}
    for key in td:
        td_difference_to_expected[key] = []
        for i, data in enumerate(td[key]):
            td_difference_to_expected[key].append(float((1000/48000) * (abs(data) - abs((key-(i+1))*140))))
    data_list = list(td_difference_to_expected.values())

    # Convert list of lists into a 2D numpy array
    data_array = np.array(data_list)

    # Ensure all values are positive by taking the absolute values
    data_array = np.abs(data_array)

    # Define a custom colormap from green to red
    colors = ["green", "yellow", "red"]
    cmap = LinearSegmentedColormap.from_list("Custom_RdYlGn_r", colors, N=256)

    # Define the threshold and the normalization
    norm = Normalize(vmin=data_array.min(), vmax=threshold_ms)

    # Plotting
    fig, ax = plt.subplots()
    fig.set_tight_layout(True)
    cax = ax.matshow(data_array, cmap=cmap, norm=norm)

    # Adjustments to the plot
    ax.xaxis.set_ticks_position('bottom')
    ticks = np.array([0] + list(range(10, 91, 10)) + [99])
    tick_labels = np.array([1] + list(range(10, 110, 10)))
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(tick_labels, fontsize=font_size)
    ax.set_yticklabels(tick_labels, fontsize=font_size)

    # Rotate tick labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
    plt.setp(ax.get_yticklabels(), rotation=45, ha='right', rotation_mode='anchor')

    # Colorbar
    cbar = fig.colorbar(cax, extend='max')
    cbar.set_label('time difference to ground truth [ms]', fontsize=font_size)
    cbar.ax.tick_params(labelsize=font_size)

    # Labels
    ax.set_xlabel("node's distance to source [m]", fontsize=font_size)
    ax.set_ylabel("node's distance to source [m]", fontsize=font_size)

    plt.xlim([0, 99])
    plt.ylim([0, 99])

    plt.tight_layout()
    fig.savefig(opath, bbox_inches="tight")
    print(f"Plot successfully saved at: {os.path.relpath(opath)}")

    # plt.show()


def plot_multisource_lineplot_confidences_over_time_and_distance(ipaths=['./0.5_143999_multisource_redstart_combined_data_mean_of_samples_above_0.0.pkl',
                                                                         './0.5_143999_multisource_chaffinch_combined_data_mean_of_samples_above_0.0.pkl',
                                                                         './0.5_143999_multisource_tit_combined_data_mean_of_samples_above_0.0.pkl'],
                                                                 opath='0.5_143999_combined_data_mean_of_samples_above_0.0_time_distance_plot.pdf', font_size=16):
    # Function to plot the data for each bird type
    def plot_data(ax_main, ax_sub1, data, color_map, bird_label):
        num_lines = len(data)
        cmap = plt.get_cmap(color_map)
        colors = cmap(np.linspace(0.3, 1, num_lines))
        colors = colors[::-1]

        for (key, values), color in zip(data.items(), colors):
            ax_main.plot(values, color=color)
            ax_sub1.plot(values, color=color)

        return plt.cm.ScalarMappable(cmap=color_map + '_r',
                                     norm=plt.Normalize(vmin=min(data.keys()), vmax=max(data.keys())))

    pkl_file = open(
        ipaths[0],
        'rb')
    data_redstart = pickle.load(pkl_file)
    a = data_redstart[35]

    # Create a figure and GridSpec layout
    fig = plt.figure(figsize=(24, 6))  # Adjusted figure size to fit two subplots and colorbars
    gs = GridSpec(1, 2, width_ratios=[1, 3])  # Added space for three colorbars

    ax_main = fig.add_subplot(gs[1])  # Main plot
    ax_sub1 = fig.add_subplot(gs[0])  # Subplot to the left of the main plot
    # Plot redstart data
    sm_red = plot_data(ax_main, ax_sub1, data_redstart, 'Reds', 'redstart')
    del data_redstart

    pkl_file = open(
        ipaths[1],
        'rb')
    data_chaffinch = pickle.load(pkl_file)

    # Plot chaffinch data
    sm_blue = plot_data(ax_main, ax_sub1, data_chaffinch, 'Blues', 'chaffinch')
    del data_chaffinch

    pkl_file = open(
        ipaths[2],
        'rb')
    data_tit = pickle.load(pkl_file)

    # Plot tit data
    sm_green = plot_data(ax_main, ax_sub1, data_tit, 'Greens', 'tit')
    del data_tit
    del pkl_file

    # Customize the main plot
    ax_main.set_xlabel('time [s]', fontsize=font_size)
    ax_main.grid(True)

    # Create dividers for colorbars
    divider_main = make_axes_locatable(ax_main)

    # Add colorbars with reduced gap
    cax_red = divider_main.append_axes("right", size="1.5%", pad=1.0)
    cax_blue = divider_main.append_axes("right", size="1.5%", pad=1.0)
    cax_green = divider_main.append_axes("right", size="1.5%", pad=1.0)

    cbar_red = fig.colorbar(sm_red, cax=cax_red, orientation='vertical',
                            label='source-to-node distance (Common Redstart)')
    cbar_blue = fig.colorbar(sm_blue, cax=cax_blue, orientation='vertical',
                             label='source-to-node distance (Common Chaffinch)')
    cbar_green = fig.colorbar(sm_green, cax=cax_green, orientation='vertical',
                              label='source-to-node distance (Great Tit)')

    # Set colorbar label fontsizes
    cbar_red.ax.yaxis.label.set_size(font_size)
    cbar_blue.ax.yaxis.label.set_size(font_size)
    cbar_green.ax.yaxis.label.set_size(font_size)

    # Set colorbar tick labels fontsize
    cbar_red.ax.tick_params(labelsize=font_size)
    cbar_blue.ax.tick_params(labelsize=font_size)
    cbar_green.ax.tick_params(labelsize=font_size)

    # Fill the area between the vertical lines on the main plot
    ax_main.axvspan(3 * 48000, 4.53 * 48000, color='red', alpha=0.25, label='Common Redstart')
    bird_sound_patch = Patch(color='red', alpha=0.25, label="Common Redstart")
    ax_main.axvspan(5 * 48000, 7.5365 * 48000, color='green', alpha=0.25, label='Great Tit')
    bird_sound_patch3 = Patch(color='green', alpha=0.25, label="Great Tit")
    ax_main.axvspan(4 * 48000, 6.207563 * 48000, color='blue', alpha=0.25, label='Common Chaffinch')
    bird_sound_patch2 = Patch(color='blue', alpha=0.5, label="Common Chaffinch")
    ax_main.legend(title="time of species sound (at source)",
                   handles=[bird_sound_patch, bird_sound_patch2, bird_sound_patch3], fontsize=font_size, title_fontsize=font_size)

    # Convert x-axis from samples to seconds for the main plot
    sampling_frequency = 48000  # 48 kHz
    max_sample_number = 11 * 48000  # Find the longest array in data
    time_ticks = np.arange(0, max_sample_number + sampling_frequency,
                           sampling_frequency)  # Generate time ticks every 1 second
    ax_main.set_xticks(time_ticks)  # Set custom ticks
    ax_main.set_xticklabels([f"{x / sampling_frequency:.1f}" for x in time_ticks], fontsize=font_size)  # Label ticks in seconds

    ax_main.set_yticks([round(val, 1) for val in np.arange(0, 1.1, 0.1)])  # Set y-ticks from min to max with a step of 0.1
    ax_main.set_yticklabels([round(val, 1) for val in np.arange(0, 1.1, 0.1)], fontsize=font_size)
    ax_sub1.set_yticks([round(val, 1) for val in np.arange(0, 1.1, 0.1)])  # Set y-ticks from min to max with a step of 0.1
    ax_sub1.set_yticklabels([round(val, 1) for val in np.arange(0, 1.1, 0.1)], fontsize=font_size)

    ax_main.set_xlim([0, 11 * sampling_frequency])  # Adjust xlim to match the 10-second window
    ax_main.set_ylim([0.0, 1.0])
    ax_main.yaxis.set_label_position("right")

    # Customize the first subplot
    ax_sub1.set_xlim([0, 4.5 * sampling_frequency])  # Focus on the 0 to 2-second window
    ax_sub1.set_ylim([0.0, 0.55])
    ax_sub1.grid(True)
    ax_sub1.set_ylabel('confidence', fontsize=font_size)
    ax_sub1.yaxis.tick_left()
    ax_sub1.yaxis.set_label_position("left")

    # Convert x-axis from samples to seconds for the subplots
    time_ticks_sub1 = np.arange(0, 4 * sampling_frequency + sampling_frequency,
                                sampling_frequency / 1)  # Generate time ticks every 1 second
    ax_sub1.set_xticks(time_ticks_sub1)  # Set custom ticks
    ax_sub1.set_xticklabels([f"{x / sampling_frequency:.1f}" for x in time_ticks_sub1], fontsize=font_size)  # Label ticks in seconds

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.05)  # Adjust the width space between subplots

    plt.savefig(opath)
    print(f"Plot successfully saved at: {os.path.relpath(opath)}")

    # plt.show()  # Display the plot


def plot_impact_of_threshold_singlespecies(ipaths_redstart=None, opath='singlespecies_impact_of_threshold', font_size=20):

    if ipaths_redstart == None:
        ipaths_redstart = [
            './data/processed/simulation/single_source/cdse/1.5_0.0.pkl',
            './data/processed/simulation/single_source/cdse/1.5_0.1.pkl',
            './data/processed/simulation/single_source/cdse/1.5_0.2.pkl',
            './data/processed/simulation/single_source/cdse/1.5_0.3.pkl',
            './data/processed/simulation/single_source/cdse/1.5_0.4.pkl',
            './data/processed/simulation/single_source/cdse/1.5_0.5.pkl'
        ]

    for spec_ctr, ipaths_spec in enumerate([ipaths_redstart]):
        fig = plt.figure(figsize=(24, 6))  # Adjusted figure size to fit two subplots and colorbars

        linestyles = ['-', '--', '-.', ':', (0, (5, 2)), (0, (3, 1, 1, 1))]
        colors = ['red', 'blue', 'green']
        labels = ['0.0', '0.1', '0.2', '0.3', '0.4', '0.5']
        distance = 50  # meter

        for i, ipath in enumerate(ipaths_spec):
            with open(ipath, 'rb') as pkl_file:
                data = pickle.load(pkl_file)
                time_series = data[distance]
            plt.plot(time_series, linestyle=linestyles[i], color=colors[spec_ctr], label=labels[i])

        sampling_frequency = 48000  # 48 kHz
        max_sample_number = len(time_series)  # Find the longest array in data
        time_ticks = np.arange(0, max_sample_number + sampling_frequency,
                               sampling_frequency)  # Generate time ticks every 1 second

        # Set custom ticks with font size
        plt.xticks(time_ticks, np.arange(0, len(time_ticks), 1), fontsize=font_size)
        plt.yticks(fontsize=font_size)
        plt.xlim([0, (len(time_ticks) -1 ) * sampling_frequency])
        plt.ylim([0,0.6])

        # Set labels with font size
        plt.xlabel("time [s]", fontsize=font_size)
        plt.ylabel("confidence", fontsize=font_size)

        # Add legend with title and font size
        plt.legend(title='Confidence Threshold', title_fontsize=font_size, fontsize=font_size)

        plt.tight_layout()

        # Save the figure
        fig.savefig(opath, bbox_inches="tight")
        print(f"Plot successfully saved at: {os.path.relpath(opath)}")

        # plt.show()


def plot_impact_of_threshold_multispecies(ipaths_redstart = None, ipaths_chaffinch=None, ipaths_tit=None, opath='impact_of_threshold', font_size=20):
    if ipaths_redstart == None:
        ipaths_redstart = [
            './data/processed/simulation/multi_source/cdse/Common_Redstart/1.5_0.0.pkl',
            './data/processed/simulation/multi_source/cdse/Common_Redstart/1.5_0.1.pkl',
            './data/processed/simulation/multi_source/cdse/Common_Redstart/1.5_0.2.pkl',
            './data/processed/simulation/multi_source/cdse/Common_Redstart/1.5_0.3.pkl',
            './data/processed/simulation/multi_source/cdse/Common_Redstart/1.5_0.4.pkl',
            './data/processed/simulation/multi_source/cdse/Common_Redstart/1.5_0.5.pkl'
        ]
    if ipaths_chaffinch == None:
        ipaths_chaffinch = [
            './data/processed/simulation//multi_source/cdse/Common_Chaffinch/1.5_0.0.pkl',
            './data/processed/simulation//multi_source/cdse/Common_Chaffinch/1.5_0.1.pkl',
            './data/processed/simulation//multi_source/cdse/Common_Chaffinch/1.5_0.2.pkl',
            './data/processed/simulation//multi_source/cdse/Common_Chaffinch/1.5_0.3.pkl',
            './data/processed/simulation//multi_source/cdse/Common_Chaffinch/1.5_0.4.pkl',
            './data/processed/simulation//multi_source/cdse/Common_Chaffinch/1.5_0.5.pkl'
        ]
    if ipaths_tit== None:
        ipaths_tit = [
            './data/processed/simulation//multi_source/cdse/Great_Tit/1.5_0.0.pkl',
            './data/processed/simulation//multi_source/cdse/Great_Tit/1.5_0.1.pkl',
            './data/processed/simulation//multi_source/cdse/Great_Tit/1.5_0.2.pkl',
            './data/processed/simulation//multi_source/cdse/Great_Tit/1.5_0.3.pkl',
            './data/processed/simulation//multi_source/cdse/Great_Tit/1.5_0.4.pkl',
            './data/processed/simulation//multi_source/cdse/Great_Tit/1.5_0.5.pkl'
        ]

    for spec_ctr, ipaths_spec in enumerate([ipaths_redstart, ipaths_chaffinch, ipaths_tit]):

        fig = plt.figure(figsize=(24, 6))  # Adjusted figure size to fit two subplots and colorbars

        linestyles = ['-', '--', '-.', ':', (0, (5, 2)), (0, (3, 1, 1, 1))]
        colors = ['red', 'blue', 'green']
        labels = ['0.0', '0.1', '0.2', '0.3', '0.4', '0.5']
        distance = 50  # meter

        for i, ipath in enumerate(ipaths_spec):
            with open(ipath, 'rb') as pkl_file:
                data = pickle.load(pkl_file)
                time_series = data[distance]
            plt.plot(time_series, linestyle=linestyles[i], color=colors[spec_ctr], label=labels[i])

        sampling_frequency = 48000  # 48 kHz
        max_sample_number = len(time_series)
        time_ticks = np.arange(0, max_sample_number + sampling_frequency, sampling_frequency)

        # Set custom ticks with font size
        plt.xticks(time_ticks, np.arange(0, len(time_ticks), 1), fontsize=font_size)
        plt.yticks(fontsize=font_size)

        plt.xlim([0, (11 - 1) * sampling_frequency])
        plt.ylim([0, 0.6])

        # Set labels with font size
        plt.xlabel("time [s]", fontsize=font_size)
        plt.ylabel("confidence", fontsize=font_size)

        # Add legend with title and font size
        plt.legend(title='Confidence Threshold', title_fontsize=font_size, fontsize=font_size)

        plt.tight_layout()
        species = ["redstart", "chaffinch", "tit"]
        output_path = f"{opath}_{spec_ctr+1}.pdf"
        print(f"Plot successfully saved at: {os.path.relpath(output_path)}")

        # Save the figure
        fig.savefig(output_path, bbox_inches="tight")

        # plt.show()


def plot_evaluation_of_timeDifferences_SingleSource(base_path = './data/processed/simulation/single_source/tdoa/',
                                                    opath="opath.pdf"):

    def calculate_time_differences(filepath, max_deviation_to_ground_truth=3.0):
        """Calculate precise time estimations from pickle file."""
        with open(filepath, 'rb') as pkl_file:
            td = pickle.load(pkl_file)

        precise_estimations_ctr = 0
        for key, values in td.items():
            for i, data in enumerate(values):
                val = abs(round((1000 / 48000) * (abs(data) - abs((key - (i + 1)) * 140)), 1))
                if val <= max_deviation_to_ground_truth:
                    precise_estimations_ctr += 1

        return round(precise_estimations_ctr / 99 ** 2, 2)

    def gather_data(base_path, sensitivities, thresholds):
        """Gather precise estimation data for sensitivities and thresholds."""
        sens_to_threshold = {}
        for sensitivity in sensitivities:
            ctr_estimations = []
            for threshold in thresholds:
                filepath = os.path.join(base_path, f"{sensitivity}_{threshold}.pkl")
                estimation = calculate_time_differences(filepath)
                ctr_estimations.append(estimation)
            sens_to_threshold[sensitivity] = ctr_estimations
        return sens_to_threshold

    def plot_data(data, bird, thresholds, linestyles, colors, fontsize=24):
        """Plot data for given bird and thresholds."""
        plt.figure(figsize=(14, 8))
        for sensitivity, values in data[bird].items():
            plt.plot(thresholds, values, linestyle=linestyles[sensitivity], color=colors[bird],
                     label=f'sensitivity: {sensitivity}')
        plt.xlabel('confidence threshold', fontsize=fontsize)
        plt.ylabel('number of estimations that match to ground truth (\%)', fontsize=fontsize)
        plt.legend(fontsize=fontsize)
        plt.xlim(0, max(thresholds))
        plt.ylim(0, 1)
        plt.yticks([i / 10.0 for i in range(11)], fontsize=fontsize)
        plt.xticks(fontsize=fontsize)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(opath)
        # plt.show()

    sensitivities = [0.5, 0.75, 1.0, 1.25, 1.5]
    thresholds = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]

    spec_to_sens_2_threshold = {
        'redstart': gather_data(base_path, sensitivities, thresholds)
    }

    linestyles = {
        0.5: (0, (5, 10)),
        0.75: '-.',
        1.0: ':',
        1.25: '--',
        1.5: '-'
    }

    colors = {
        'redstart': 'red'
    }

    for bird in spec_to_sens_2_threshold.keys():
        plot_data(spec_to_sens_2_threshold, bird, thresholds, linestyles, colors, fontsize=24)



def plot_evaluation_of_timdeDifferences_MultiSource(base_path = './data/processed/simulation/multi_source/tdoa/',
                                                    opath="opath.pdf"):
    def calculate_time_differences(filepath, max_deviation_to_ground_truth=3.0):
        """Calculate precise time estimations from pickle file."""
        with open(filepath, 'rb') as pkl_file:
            td = pickle.load(pkl_file)

        precise_estimations_ctr = 0
        for key, values in td.items():
            for i, data in enumerate(values):
                val = abs(round((1000 / 48000) * (abs(data) - abs((key - (i + 1)) * 140)), 1))
                if val <= max_deviation_to_ground_truth:
                    precise_estimations_ctr += 1

        return round(precise_estimations_ctr / 99 ** 2, 2)

    def gather_data(base_path, species, sensitivities, thresholds):
        """Gather precise estimation data for multiple species."""
        spec_to_sens_2_threshold = {}
        for bird in species:
            print("Species:", bird)
            sens_to_threshold = {}
            for sensitivity in sensitivities:
                print("Sensitivity:", sensitivity, end='  |  ')
                ctr_estimations = []
                for threshold in thresholds:
                    filepath = os.path.join(base_path,bird,
                                            f"{sensitivity}_{threshold}.pkl")
                    estimation = calculate_time_differences(filepath)
                    ctr_estimations.append(estimation)
                    print(estimation, end=' & ')
                sens_to_threshold[sensitivity] = ctr_estimations
                print()
            spec_to_sens_2_threshold[bird] = sens_to_threshold
        return spec_to_sens_2_threshold

    def plot_data(data, bird, thresholds, linestyles, colors, opath, fontsize=20):
        """Plot data for a single bird."""
        plt.figure(figsize=(12, 6))
        for sensitivity, values in data[bird].items():
            plt.plot(thresholds, values, linestyle=linestyles[sensitivity], color=colors[bird],
                     label=f'sensitivity: {sensitivity}')
        plt.xlabel('Confidence Threshold', fontsize=fontsize)
        plt.ylabel('Number of Estimations Matching Ground-Truth (\%)', fontsize=fontsize)
        plt.legend(fontsize=fontsize)
        plt.xlim(0, max(thresholds))
        plt.ylim(0, 1)
        plt.yticks([i / 10.0 for i in range(11)], fontsize=fontsize)
        plt.xticks(fontsize=fontsize)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(opath)
        plt.show()

    def plot_all_data(data, thresholds, linestyles, colors, opath, fontsize=24):
        """Plot data for all species in one plot."""
        plt.figure(figsize=(14, 8))
        for bird, bird_data in data.items():
            for sensitivity, values in bird_data.items():
                plt.plot(thresholds, values, linestyle=linestyles[sensitivity], color=colors[bird],
                         label=f'{bird.replace("_", " ")} - sensitivity {sensitivity}')
        plt.xlabel('Confidence Threshold', fontsize=fontsize)
        plt.ylabel('Number of Estimations Matching Ground-Truth (\%)', fontsize=fontsize)
        plt.legend(fontsize=fontsize, ncol=3)
        plt.xlim(0, max(thresholds))
        plt.ylim(0, 1)
        plt.yticks([i / 10.0 for i in range(11)], fontsize=fontsize)
        plt.xticks(fontsize=fontsize)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(opath)
        # plt.show()

    species = ['Common_Redstart', 'Common_Chaffinch', 'Great_Tit']
    sensitivities = [0.5, 0.75, 1.0, 1.25, 1.5]
    thresholds = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]

    linestyles = {
        0.5: (0, (5, 10)),
        0.75: '-.',
        1.0: ':',
        1.25: '--',
        1.5: '-'
    }

    colors = {
        'Common_Redstart': 'red',
        'Common_Chaffinch': 'blue',
        'Great_Tit': 'green'
    }

    spec_to_sens_2_threshold = gather_data(base_path, species, sensitivities, thresholds)

    plot_all_data(spec_to_sens_2_threshold, thresholds, linestyles, colors,
                  opath=opath)

