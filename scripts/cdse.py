import pickle
import sys
import pandas as pd
import numpy as np

class CDSE:

    def __init__(self):
        self.file_path = ""
        self.classifications = None
        self.data = None

    def set_data_from_parser(self, df: pd.DataFrame):
        """
        Sets the 'data' attribute with the provided DataFrame.

        Args:
            df (pd.DataFrame): The DataFrame to assign to the 'data' attribute.
        """
        if isinstance(df, pd.DataFrame):
            self.data = df
        else:
            raise ValueError("The provided input is not a pandas DataFrame.")

    def load_from_pkl(self, pkl_path):
        """
        Load a DataFrame from a pickle file.

        Parameters:
        ----------
        pkl_path : str
            Path to the pickle (.pkl) file containing the saved DataFrame.

        Returns:
        -------
        pd.DataFrame
            A DataFrame with the following format:

            ===================  =========  =========  ============
               Column Name          Type      Example      Description
            ===================  =========  =========  ============
            start                int        0           Start index or timestamp
            end                  int        144000      End index or timestamp
            confidence           float      0.0063      Confidence score for the interval
            ===================  =========  =========  ============

            Example:

            >>> load_from_pkl('data.pkl')
                   start     end  confidence
            0            0  144000      0.0063
            1            1  144001      0.0067
            2            2  144002      0.0053
            3            3  144003      0.0041
            4            4  144004      0.0049
            ...        ...     ...         ...
            335996  335996  479996      0.0057
            335997  335997  479997      0.0039
            335998  335998  479998      0.0036
            335999  335999  479999      0.0035
            336000  336000  480000      0.0030

        Notes:
        ------
        - The confidence values are represented as floating-point numbers.
        - Ensure that the file at `file_path` is accessible and contains a valid pickle file with the expected DataFrame structure.
        """

        try:
            with open(pkl_path, 'rb') as file:
                self.data = pickle.load(file)
            print(f"Data loaded from {pkl_path}")
        except FileNotFoundError:
            print(f"Error: The file at {pkl_path} was not found.")
            raise
        except Exception as e:
            print(f"An error occurred while loading the file: {e}")
            raise

        self.file_path = pkl_path

    def has_holes(self) -> bool:
        """
        Check if the DataFrame contains any holes in the sequence of the 'start' and 'end' columns.

        A hole is defined as a missing or non-sequential range in the indices between
        the 'start' and 'end' columns.

        Parameters:
        ----------
        df : pd.DataFrame
            A DataFrame with 'start', 'end', and 'confidence' columns.

        Returns:
        -------
        bool
            True if the DataFrame contains holes, False otherwise.

        Example:
        --------
        >>> df = pd.DataFrame({'start': [0, 1, 2, 4], 'end': [144000, 144001, 144002, 144004], 'confidence': [0.0063, 0.0067, 0.0053, 0.0049]})
        >>> has_holes(df)
        True  # There is a missing index between 2 and 4
        """
        # Check if DataFrame has the necessary columns
        if not {'start', 'end', 'confidence'}.issubset(self.data.columns):
            raise ValueError("DataFrame must contain 'start', 'end', and 'confidence' columns.")

        # Check if the 'start' and 'end' columns are integers and sorted
        if not pd.api.types.is_integer_dtype(self.data['start']) or not pd.api.types.is_integer_dtype(self.data['end']):
            raise ValueError("Columns 'start' and 'end' must be of integer type.")

        # Check if the DataFrame is sorted correctly
        if not self.data['start'].is_monotonic_increasing or not self.data['end'].is_monotonic_increasing:
            raise ValueError("The DataFrame must be sorted in ascending order by 'start' and 'end'.")

        # Check for holes in the 'start' and 'end' sequence
        start_diff = self.data['start'].diff().fillna(1)  # Difference between consecutive 'start' values
        end_diff = self.data['end'].diff().fillna(1)  # Difference between consecutive 'end' values

        # Check for gaps: expecting consecutive numbers where diff == 1
        has_hole = (start_diff != 1).any() or (end_diff != 1).any()

        return has_hole

    def cdse_from_dataframe(self, outpath="testpkl.pkl",
                            end_col='end',
                            audio_sampling_frequency=48000,
                            audio_max_duration=None,
                            window_size=144000,
                            confidence_threshold=0.1,
                            progress_updates=True,
                            progress_interval=48000) -> np.ndarray:
        """
        Calculate the Classifier-Deduced-Signal-Extraction (CDSE) from a DataFrame and optionally save the results.

        This function processes a DataFrame containing confidence values for BirdNET classifications, computes a sliding
        window average of confidence values, and optionally saves the result to a pickle file.

        Args:
            outpath (str): Path to save the resulting CDSE array. Defaults to "testpkl.pkl".
            end_col (str): Name of the column indicating the end of audio segments. Defaults to 'end'.
            audio_sampling_frequency (int): Sampling frequency of the audio data in Hz. Defaults to 48000, used by BirdNET v2.4.
            audio_max_duration (int, optional): Maximum duration of the audio in samples. If None, it's derived from the
                maximum value in the `end_col`. Defaults to None.
            window_size (int): Size of the sliding window in samples for calculating CDSE. Defaults to 144000, referring to 3-second-interval of BirdNET v2.4.
            confidence_threshold (float): Minimum confidence level to consider in the calculations. Values below this
                threshold are set to 0. Defaults to 0.1.
            progress_updates (bool): Whether to display progress updates during calculation. Defaults to True.
            progress_interval (int): Interval (in samples) at which progress updates are printed. Defaults to 48000.

        Returns:
            np.ndarray: Array containing the calculated CDSE values for each sample.

        Raises:
            SystemExit: If the DataFrame contains "holes" (gaps) in the sequence of 'start' and 'end' columns.

        Notes:
            - This method expects the DataFrame to contain a 'confidence' column for processing.
            - A "hole" in the sequence is identified if there are gaps in the `start` and `end` columns.

        Example:
            ```
            # Assuming `self.data` is a properly formatted DataFrame:
            confs = obj.cdse_from_dataframe(outpath="output.pkl", window_size=144000)
            ```
        """
        if self.has_holes():
            print("Warning: The DataFrame contains holes in the sequence of 'start' and 'end' columns.")
            sys.exit(1)
        else:
            print("No holes detected. Continuing process.")

        # Set maximum audio duration if not provided
        if not audio_max_duration:
            audio_max_duration = self.data[end_col].max()

        print(f"Applying confidence threshold of {confidence_threshold}...")

        # Apply confidence threshold
        if confidence_threshold != 0.0:
            self.data.loc[self.data['confidence'] < confidence_threshold, 'confidence'] = 0

        print("Start performing CDSE. This may take a while...")

        # Convert confidence column to a NumPy array for fast computation
        confidence_array = self.data['confidence'].to_numpy()

        # Perform confidence calculations using a sliding window
        confs = np.zeros(audio_max_duration)

        for sample in range(0, audio_max_duration):
            # start_idx and eng_idx refers to start of classification interval, meaning
            # Classification-Example:   line 0: 0,144000, VAL1
            #                           line 2: 1,144001, VAL1
            #                           line 3: 1,144001, VAL1
            #                           ...
            # Calculation CDSE for sample 0: Refers only to line 0 (because sample 0 is only part of line 0).
            #                      sample 1: Refers only to line 0 and line 1 (because sample 1 is part of line 0 & 1), etc.
            start_idx = max(0, sample - window_size)
            end_idx = sample + 1
            confs[sample] = confidence_array[start_idx:end_idx].mean()

            # Display progress updates
            if progress_updates and (sample % progress_interval == 0):
                print(f"Calculated CDSE for {sample / audio_sampling_frequency:.2f} seconds")

        # Save results to a pickle file
        if outpath:
            with open(outpath, 'wb') as fd:
                pickle.dump(confs, fd)
                print(f"CDSE calculation completed and saved to {outpath}.")

        return confs
