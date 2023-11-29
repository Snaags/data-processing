import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
import os



def parse_experiment_folder_names(folder_name : str) -> tuple[str, str, int]:
    """
    splits the experiment folder from experimentName-index-dataset
    """

    # Find Index
    split_index = None
    two_digit_flag = 0
    for idx,i in enumerate(folder_name):
        if i.isdigit():
            split_index = idx
            if folder_name[idx+1].isdigit():
                two_digit_flat = 1
            print(split_index)
    
    experiment_name = folder_name[:split_index-1]
    index = int(folder_name[split_index])
    dataset_name = folder_name[split_index+2+two_digit_flag:]

    return experiment_name, dataset_name, index



def generate_directory_dict(PATH : str) -> dict:
    directory = os.listdir(PATH)
    mapping_dict = {}

    for file in directory:
        mapping_dict[file] = parse_experiment_folder_names(file)

    return mapping_dict
        

def get_scores(path):
    path += "/test_results.csv"
    data = pd.read_csv(path)
    ensemble_1 = float(data["ensemble_1"].iloc[0])
    ensemble_2 = float(data["ensemble_10"].iloc[0])
    print(ensemble_1,ensemble_2)
    return ensemble_1, ensemble_2

def init_data_struct(mapping : dict,path) -> pd.DataFrame:

    """ DF """
    #[experiment, index, dataset, ensemble, score]
    #    name       2     LSST       1      0.98  
    #    name2      3     LSST       2      0.37
    #    name       2     SCP2       1      0.87
    #    name2      2     LSST       2      0.78

    df = pd.DataFrame(columns = ["experiment","experiment_index","dataset","ensemble","score"])

    # Loop through data creating vectors for df (added once for each ensemble size)
    experiments = []
    indices = []
    datasets = []
    ensemble = []
    scores = []
    for inst in mapping:
        score_1, score_2 = get_scores(path+inst)
        experiments.append(mapping[inst][0])
        indices.append(mapping[inst][1])
        datasets.append(mapping[inst][2])
        ensemble.append(1)
        scores.append(score_1)

        experiments.append(mapping[inst][0])
        indices.append(mapping[inst][1])
        datasets.append(mapping[inst][2])
        ensemble.append(2)
        scores.append(score_2)
    
    df["experiment"] = experiments
    df["experiment_index"] = indices
    df["dataset"] = datasets
    df["ensemble"] = ensemble
    df["score"] = score

    return df

def load_benchmarks(paths : [str]) -> pd.DataFrame:
    for path in paths:
        df_temp = pd.DataFrame()
        df_temp.read_csv(path)
        if len(paths) == 1:
            return df_temp

    return df

def average_dataframe(df : pd.DataFrame, target_column : str) -> pd.DataFrame:
    cols = list(df.columns).remove(target_column)
    grouped_df = df.groupby(cols).mean()
    return grouped_df.reset_index()

def convert_to_methods(df):
    """
    Switch the dataFrame into wide form to make the specific ensemble experiment pairs into specific methods
    """
    wide_df = df.pivot(index = "dataset" , columns = ["experiment","ensemble"],values = "score")
    wide_df.columns = ["_".join(str(i) for i in col) for col in wide_df.columns]
    wide_df = wide_df.reset_index()
    return wide_df

def combine_dataframes(df_1,df_2):
    """
    Both dataframes will have a column with the name of the datasets but with different names
    """
    dataset_id_1 = None
    dataset_id_2 = None
    cols_1 = df_1.columns 
    cols_2 = df_2.columns
    # Find 2 columns that are identical and the values are strings (just in case)
    for i_1,i_2 in zip(cols_1,cols_2):
        if type(i_1) == float:
            cols_1.remove(i_1)
        if type(i_2) == float:
            cols_2.remove(i_2)
    for i_1 in cols_1:
        for i_2 in cols_2:
            if df_1[i_1] == df_2[i_2]:
                df_2[i_1] = df_2[i_2]
                df_2.drop(columns = [i_2])
                df = pd.merge(df_1,df_2, on = i_1)
                return df
    raise ValueError("Dataset columns are not matched!!")
    


if __name__ == "__main__":
    PATH = "HPO/experiments/"
    SERVER_PATH = "/home/snaags/cava/"
    PATH = SERVER_PATH+PATH
    UCR_RESULTS_PATH = [None]
    dir_dict = generate_directory_dict(PATH)
    full_df = init_data_struct(dir_dict,PATH)
    averaged_df = average_dataframe(full_df,"experiment_index")
    benchmark = load_benchmark_csv(UCR_RESULTS_PATH)
    processed_df = convert_to_methods(averaged_df)
    combined_df = combined_dataframes(processed_df,benchmark)
    print(combined_df)
