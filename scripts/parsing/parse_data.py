import pandas as pd
import numpy as np
import pickle
import os

class Parser:
    def __init__(self):
        self.file_path = ""
        self.data = None

    def parse_textfile(self, file_path, rows_per_chunk=1440000, use_columns=None, column_names=None):
        """
        Parses a large text file in chunks and concatenates the results into a single DataFrame.

        Parameters:
        file_path (str): Path to the text file to be parsed.
        rows_per_chunk (int, optional): Number of rows per chunk to be processed. Defaults to 1440000.
        use_columns (list of int, optional): List of column indices to read from the file. Defaults to [0, 1, 4].
        column_names (list of str, optional): List of column names to assign to the DataFrame. Defaults to ['start', 'end', 'confidence'].

        Returns:
        pandas.DataFrame: A concatenated DataFrame containing all the data from the specified columns of the file.

        Raises:
        FileNotFoundError: If the specified file path does not exist.
        ValueError: If the column names and use_columns lengths do not match.
        """

        def remove_header_row_if_present(df):
            # Check if the first row contains non-numeric values
            if any(isinstance(x, str) for x in df.iloc[0]):
                print("Header row detected. Removing it...")
                df = df.iloc[1:].reset_index(drop=True)
                # Optionally, convert data types after removing the header
                df = df.astype({'start': 'int64', 'end': 'int64', 'confidence': 'float64'})
            else:
                print("No header row detected.")
            return df

        if use_columns is None:
            use_columns = [0, 1, 4]

        if column_names is None:
            column_names = ['start', 'end', 'confidence']

        if len(use_columns) != len(column_names):
            raise ValueError("The number of column names must match the number of columns specified in use_columns.")

        chunk_list = []

        try:
            for chunk in pd.read_csv(file_path, dtype=str, header=None, usecols=use_columns, names=column_names, chunksize=rows_per_chunk):
                chunk_list.append(chunk)

            self.data = pd.concat(chunk_list, ignore_index=True)
            self.data = remove_header_row_if_present(self.data)

            self.data['start'] = self.data['start'].astype(int)
            self.data['end'] = self.data['end'].astype(int)
            self.data['confidence'] = self.data['confidence'].astype(float)

            print("Parsed file successfully.")
        except FileNotFoundError as e:
            print(f"Error: The file at {self.file_path} was not found.")
            raise
        except Exception as e:
            print(f"An error occurred while parsing the file: {e}")
            raise
        return self.data

    def parse_directory_concat_data(self, dir_path, rows_per_chunk=1440000, use_columns=None, column_names=None):
        """
        Parses all text files in the specified directory and concatenates the results into a single DataFrame.

        This method reads text files from the given directory path, processes them in chunks,
        and concatenates them into a single DataFrame. The text files are expected to be structured
        in tabular format, with values corresponding to specific columns.

        Parameters:
        ----------
        dir_path : str
            Path to the directory containing text files to be parsed.
        rows_per_chunk : int, optional
            Number of rows per chunk to be processed at a time. Defaults to 1,440,000 rows.
        use_columns : list of int, optional
            List of column indices to read from the text files. Defaults to [0, 1, 4], corresponding
            to 'start', 'end', and 'confidence' respectively.
        column_names : list of str, optional
            List of column names to assign to the parsed DataFrame. Defaults to ['start', 'end', 'confidence'].

        Returns:
        -------
        list of pd.DataFrame
            A list containing DataFrame chunks read from the text files.

        Notes:
        ------
        - The method reads each file line-by-line in chunks using `pandas.read_csv` to handle large files.
        - The final concatenated DataFrame is stored in `self.data`.
        - The files in the directory must be structured similarly and readable as tabular data.

        Raises:
        ------
        FileNotFoundError
            If the specified directory path does not exist or cannot be accessed.
        ValueError
            If no valid files are found in the directory.
        Exception
            For any other errors encountered during file parsing.

        Example:
        --------
        >>> parser = DataParser()
        >>> parser.parse_directory_concat_data("/path/to/directory")
        Parsing file: /path/to/directory/file1.txt
        Parsing file: /path/to/directory/file2.txt
        Parsed all files in the directory successfully.

        The concatenated DataFrame is stored in `parser.data`.
        """
        if use_columns is None:
            use_columns = [0, 1, 4]

        if column_names is None:
            column_names = ['start', 'end', 'confidence']

        all_chunks = []

        try:
            last_end_value = 0  # Initialize last_end_value

            for filename in os.listdir(dir_path):
                file_path = os.path.join(dir_path, filename)
                if os.path.isfile(file_path):
                    print(f"Parsing file: {file_path}")

                    # Only update `last_end_value` after the first file
                    if all_chunks:
                        last_end_value = all_chunks[-1]["end"].iloc[-1]  # Get the last end time from the last chunk

                    try:
                        for chunk in pd.read_csv(file_path, header=None, usecols=use_columns, names=column_names,
                                                 chunksize=rows_per_chunk):
                            # Ensure the columns are correct
                            if list(chunk.columns) != ['start', 'end', 'confidence']:
                                chunk.columns = ['start', 'end', 'confidence']

                            # Increment 'start' and 'end' by the last end value
                            chunk["start"] += last_end_value
                            chunk["end"] += last_end_value

                            all_chunks.append(chunk)
                    except Exception as e:
                        print(f"An error occurred while parsing {file_path}: {e}")

            if not all_chunks:
                raise ValueError(f"No valid files found in the directory: {dir_path}")

            # Concatenate all chunks into a single DataFrame
            concatenated_df = pd.concat(all_chunks, ignore_index=True).reset_index(drop=True)
            concatenated_df.columns = ['start', 'end', 'confidence']

            print("Parsed all files in the directory successfully.")
        except FileNotFoundError as e:
            print(f"Error: The directory at {dir_path} was not found.")
            raise
        except Exception as e:
            print(f"An error occurred while parsing the directory: {e}")
            raise

        return concatenated_df

    def parse_simulated_directory(self, dir_path, rows_per_chunk=1440000, use_columns=None, column_names=None):
        """
        Parses a directory containing subdirectories with CSV files. The function reads the data from CSV files,
        processes it in chunks, and stores the parsed data in a dictionary where the keys are derived from the
        subdirectory names and the values are DataFrames containing the parsed data.

        Args:
            dir_path (str): The path to the root directory containing subdirectories with CSV files to be parsed.
            rows_per_chunk (int, optional): The number of rows to read per chunk from each CSV file. Defaults to 1440000.
            use_columns (list of int, optional): A list of column indices to use from each CSV file. Defaults to [0, 1, 4].
            column_names (list of str, optional): A list of column names for the CSV files. Defaults to ['start', 'end', 'confidence'].

        Returns:
            dict: A dictionary where the keys are integers derived from subdirectory names, and the values are pandas DataFrames
                  containing the concatenated data from all the CSV files in the respective subdirectory.

        Raises:
            FileNotFoundError: If the specified directory path does not exist.
            Exception: For any other errors that occur during the parsing process.

        Notes:
            - The function assumes that each CSV file contains the same structure (i.e., the same columns).
            - The directory structure is expected to be such that each subdirectory contains CSV files to be processed.
            - It reads CSV files in chunks to handle large datasets efficiently.
            - The function automatically determines whether the CSV files contain a header row or not and adjusts accordingly.
        """
        if use_columns is None:
            use_columns = [0, 1, 4]

        if column_names is None:
            column_names = ['start', 'end', 'confidence']

        data_dict = {}

        try:
            dirs = [d for d in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path, d))]
            sorted_dirs = sorted(dirs)
            for dirname in sorted_dirs:
                dir_full_path = os.path.join(dir_path, dirname)
                if os.path.isdir(dir_full_path):
                    all_chunks = []
                    print(f"Parsing directory: {dir_full_path}")
                    for filename in os.listdir(dir_full_path):
                        file_path = os.path.join(dir_full_path, filename)
                        if os.path.isfile(file_path):
                            # print(f"Parsing file: {file_path}")
                            try:
                                # Read a small sample (first 5 rows) to check for a header
                                sample_df = pd.read_csv(file_path, nrows=1, usecols=use_columns, names=column_names,
                                                        header=None)

                                # Check if the length matches before comparing
                                if len(sample_df.columns) == len(column_names):
                                    has_header = (sample_df.iloc[0].values == column_names).all()
                                else:
                                    has_header = False
                                # Set header accordingly
                                header_option = 0 if has_header else None
                                # Read the full file in chunks
                                for chunk in pd.read_csv(file_path, header=header_option, usecols=use_columns,
                                                         names=column_names if header_option is None else None,
                                                         chunksize=rows_per_chunk):
                                    all_chunks.append(chunk)
                            except Exception as e:
                                print(f"An error occurred while parsing {file_path}: {e}")

                    # Concatenate chunks and store in the dictionary
                    if all_chunks:
                        concatenated_df = pd.concat(all_chunks, ignore_index=True).reset_index(drop=True)
                        concatenated_df.columns = ['start', 'end', 'confidence']
                        data_dict[int(dirname.split(']')[0].split('[')[1])] = concatenated_df
                        # print(f"Parsed all files in directory {dirname} successfully.")
        except FileNotFoundError as e:
            print(f"Error: The directory at {dir_path} was not found.")
            raise
        except Exception as e:
            print(f"An error occurred while processing the directories: {e}")
            raise
        return dict(sorted(data_dict.items()))

    @staticmethod
    def check_and_fill_missing_values(df, min_start=None, max_end=None, chunk_size=144000, start_col='start',
                                      end_col='end', step=1, confidence_col='confidence', default_confidence=0.0):
        """
        Checks for missing chunks of data within a given range and fills in the missing chunks with default values.

        This method compares the existing data's start and end times against the expected time chunks (defined by `min_start`, `max_end`, and `chunk_size`).
        It identifies any missing chunks and adds them to the DataFrame with default values for confidence, sorting the data afterward.

        Args:
            df (pandas.DataFrame): The input DataFrame containing the data with at least two columns: `start` and `end` (representing time intervals),
                                   and optionally a `confidence` column.
            min_start (int, optional): The minimum start time to begin creating time chunks. Defaults to 0 if not provided.
            max_end (int, optional): The maximum end time for the chunks. Defaults to the maximum value in the `start_col` column if not provided.
            chunk_size (int, optional): The size of each time chunk. Defaults to 144000 (1 hour, assuming time units are in seconds).
            start_col (str, optional): The column name for the start time of each interval. Defaults to 'start'.
            end_col (str, optional): The column name for the end time of each interval. Defaults to 'end'.
            step (int, optional): The step size between each time chunk. Defaults to 1, which means chunks will be generated sequentially with no gaps.
            confidence_col (str, optional): The column name for the confidence values. Defaults to 'confidence'.
            default_confidence (float, optional): The default confidence value to assign to missing chunks. Defaults to 0.0.

        Returns:
            pandas.DataFrame: A DataFrame that includes the original data with any missing time chunks filled in. The DataFrame will be sorted by the `start`
                               and `end` columns.

        Raises:
            ValueError: If the DataFrame `df` is None or if data has not been loaded.

        Notes:
            - This function assumes that the `start_col` and `end_col` represent time intervals and that the data is sorted by these columns.
            - Missing time chunks are identified by checking all possible combinations of `start` and `end` times in the range from `min_start` to `max_end`.
            - New entries with default confidence values are added for missing chunks.
            - The resulting DataFrame will be sorted by the `start` and `end` columns to ensure chronological order.
        """
        if df is None:
            raise ValueError("Data not loaded. Please parse a file first.")

        if min_start is None:
            min_start = 0
        if max_end is None:
            max_end = df[start_col].max()

        starts = np.arange(min_start, max_end + step, step)
        ends = starts + chunk_size
        all_combinations = np.column_stack((starts, ends))

        existing_pairs = set(zip(df[start_col].values, df[end_col].values))

        missing_indices = [
            i for i, pair in enumerate(all_combinations) if tuple(pair) not in existing_pairs
        ]

        missing_data = pd.DataFrame(all_combinations[missing_indices], columns=[start_col, end_col])
        missing_data[confidence_col] = default_confidence

        df = pd.concat([df, missing_data], ignore_index=True)
        df = df.sort_values(by=[start_col, end_col], kind="mergesort").reset_index(drop=True)
        return df

    def save_as_pkl(self, output_path):
        """
        Saves the current DataFrame to a .pkl file.
        """
        if self.data is None:
            raise ValueError("No data to save. Please parse a file first.")

        with open(output_path, 'wb') as file:
            pickle.dump(self.data, file)
        print(f"Data saved to {output_path}")

    def load_from_pkl(self, pkl_path):
        """
        Loads a DataFrame from a .pkl file.
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