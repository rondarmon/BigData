import logging
import os
import shutil
import pandas as pd
import numpy as np
import sqlite3
import threading
import binpacking
from datetime import datetime


# UTILS
def remove_file_from_path(path, logging):
    if os.path.exists(path):
        os.remove(path)
    else:
        logging.info("The file does not exist")


def merge_small_files(input_data, min_chunk_size=32, max_chunk_size=64, time_limit=10):
    """
    Description:
    ------------
    Simple algorithm of merging small files in evenly sized chunks, utilizing bin-packing strategy.
    The algorithm works on csv files with the same properties.

    Parameters:
    -----------
    input_data: list of csv files paths.

    chunk_size: int, default=64
        The maximal size (in MegaBytes) of each chunk.

    time_limit: int, default=10
        The maximal time a file will have to wait in order to be sent in a small chunk size.

    """

    run_time = datetime.now()

    remaining_data = {}
    for file in input_data:
        creation_time = os.stat(file).st_ctime
        file_size = os.path.getsize(file) / (1024 ** 2)
        remaining_data[file] = {'metadata': {'creation_time': creation_time, 'file_size': file_size}}

    # Form chunks with bin-packing
    file_sizes = {key: remaining_data[key]['metadata']['file_size'] for key in remaining_data}
    # Consider file as small if its size is below 32MB.
    chunks = binpacking.to_constant_volume(d=file_sizes, V_max=max_chunk_size, lower_bound=0, upper_bound=32)

    merged_data = []
    for i, chunk in enumerate(chunks, 1):
        chunk_files, chunk_size = list(chunk.keys()), sum(chunk.values())

        string = f"\nFiles in chunk {i}: {', '.join(chunk_files)} - Total size of chunk: {chunk_size}"
        logging.info(string)

        lowest_creation_time = min([remaining_data[file]['metadata']['creation_time'] for file in chunk_files])

        old_file_in_chunk = (run_time - datetime.fromtimestamp(lowest_creation_time)).seconds
        # We deliver chunks with size greater than minimum threshold to the mappers.
        if (chunk_size >= min_chunk_size) or (old_file_in_chunk > (60 * time_limit)):

            # Merge small files together
            chunk_data = None
            for file in chunk_files:
                df = pd.read_csv(file)
                chunk_data = pd.concat([chunk_data, df])

                # Remove files from data source
                remaining_data.pop(file)

            merged_data.append(chunk_data)

    return remaining_data, merged_data


def inverted_map(data, save_path):
    """
    Description:
    ------------
    The Map function receives input path and output path,
    returns a dataframe in the form of (key, value).

    """

    inverted_list = []

    for idx in range(len(data)):
        for column in data.columns:
            inverted_list.append(tuple([column + '_' + data[column].iloc[idx], data]))

    dataframe = pd.DataFrame(columns=['key', 'value'], data=inverted_list)  # [(key, value), (key, value)...]
    dataframe.to_csv(save_path, index=False)


def inverted_reduce(value, documents, save_path):
    """
    Description:
    ------------
    The Reduce function which recevies a value, a list of values and an output path,
    Creates a set from the list of values and returns a dataframe in the form of (key, value).

    """

    inverted_reduce = [value, list(set(documents))]

    dataframe = pd.DataFrame(columns=['key', 'value'], data=[inverted_reduce])  # [(key, value), (key, value)...]
    dataframe.to_csv(save_path, index=False)


class MapReduceEngine:
    """
    Description:
    ------------
    Class which performs map-reduce operations on parallel.
    """

    def __init__(self, destination):
        self.destination = destination
        self.map_paths = []
        self.reduce_paths = []
        self.success = None

    def execute(self, input_data, map_function, reduce_function):
        """
        Description:
        ------------
        The execute function takes input data, then performs map-reduce procedure using
        the map and the reduce functions provided.

        Parameters:
        -----------
        input_data: list, an array of csv paths.

        map_function: mapping function which receives input path and output path.
            The map function returns a dataframe in the form of (key, value).

        reduce_function: reducing function which recevies a value, a list of values and an output path.
            The reduce functions returns a dataframe in the form of (key, value).

        """

        tasks_map, tasks_reduce = [], []

        try:
            for idx, data in enumerate(input_data, 1):
                prefix = idx
                save_path = f'{self.destination}/mapreducetemp/part-tmp-{prefix}.csv'
                self.map_paths.append(save_path)

                task = threading.Thread(target=map_function, args=[data, save_path])
                task.start()
                tasks_map.append(task)

            for task in tasks_map:
                task.join(timeout=300.0)

            # Connecting and writing CSV's to a local database
            con = sqlite3.connect(f'{self.destination}/mydb.db')

            for path in self.map_paths:  # Collect
                data = pd.read_csv(path)
                data.to_sql('temp_results', con=con, if_exists='append', index=False)

            # Querying the local database (Shuffle step)
            query = """
SELECT key
    ,  GROUP_CONCAT(value) AS all_values
FROM temp_results
GROUP BY key
ORDER BY key
"""

            dataframe = pd.read_sql(sql=query, con=con)

            for idx, row in dataframe.iterrows():
                value = row['key']
                documents = row['all_values'].split(',')

                prefix = idx + 1
                save_path = f'{self.destination}/mapreducefinal/part-{prefix}-final.csv'
                self.reduce_paths.append(save_path)

                task = threading.Thread(target=reduce_function, args=[value, documents, save_path])
                task.start()
                tasks_reduce.append(task)

            for task in tasks_reduce:
                task.join(timeout=300.0)

            # Save an empty csv to mark the completion of the reduce phase.
            self.success = True
            success_path = f'{self.destination}/mapreducefinal/_SUCCESS.csv'
            self.reduce_paths.append(success_path)
            pd.DataFrame().to_csv(success_path)

            return str("MapReduce Completed")

        except Exception:
            return str("MapReduce Failed")


if __name__ == '__main__':

    # Create log file
    logging.basicConfig(filename='merge_small_files_mapreduce.log', encoding='utf-8', level=logging.DEBUG)

    # Define a folder path to store the data
    cur_dir = os.getcwd()
    folder_path = cur_dir + '/mapreduce'

    # Create 2 sub-folders for mappers and reducers
    if os.path.exists(f'{folder_path}/mapreducetemp'):
        shutil.rmtree(f'{folder_path}/mapreducetemp')
    os.makedirs(f'{folder_path}/mapreducetemp')

    if os.path.exists(f'{folder_path}/mapreducefinal'):
        shutil.rmtree(f'{folder_path}/mapreducefinal')
    os.makedirs(f'{folder_path}/mapreducefinal')

    # Establish connection to the database (sqlite)
    con = sqlite3.connect(f'{folder_path}/mydb.db')
    cur = con.cursor()

    # Create table with metadata
    cur.execute("""
CREATE TABLE temp_results
    (
    key VARCHAR(20),
    value VARCHAR(20)
    )
            """)

    # Creating randomized datasets
    firstname = ['John', 'Dana', 'Scott', 'Marc', 'Steven', 'Michael', 'Albert', 'Johanna']
    city = ['NewYork', 'Haifa', 'Munchen', 'London', 'PaloAlto', 'TelAviv', 'Kiel', 'Hamburg']

    for i in range(1, 21):
        names = np.random.choice(a=firstname, size=10)
        cities = np.random.choice(a=city, size=10)
        second_names = np.random.choice(a=firstname, size=10)

        df = pd.DataFrame(data={'firstname': names, 'secondname': second_names, 'city': cities})
        df.to_csv(f'{folder_path}/myCSV[{i}].csv', index=False)

    # Input data
    input_data = [f'{folder_path}/myCSV[{i}].csv' for i in range(1, 21)]

    # Run Merge Small Files algorithm
    remaining_data, merged_data = merge_small_files(input_data=input_data,
                                                    min_chunk_size=32,
                                                    max_chunk_size=64,
                                                    time_limit=10)

    # Run MapReduce on the merged data
    mapreduce = MapReduceEngine(destination=folder_path)
    status = mapreduce.execute(input_data=merged_data, map_function=inverted_map, reduce_function=inverted_reduce)

    logging.info(status)

    # If mapreduce succeeded then delete all the mapping phase temporary files.
    if mapreduce.success:
        for path in mapreduce.map_paths:
            remove_file_from_path(path, logging)

    # Close connections of sqlite
    try:
        cur.close()
        con.close()

    # Exception of 'ProgrammingError' determine that the connection is closed
    except sqlite3.ProgrammingError:
        logging.warning('Connection is already closed!')

    # Delete the temporary sqlite database.
    if os.path.exists(f'{folder_path}/mydb.db'):
        os.remove(f'{folder_path}/mydb.db')
