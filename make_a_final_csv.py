import os
import pandas as pd
import glob
from pathlib import Path
import re
import argparse


def preprocess_df(df, add_test_results, report_all=False):
    def check_format(df):
        assert df["Unnamed: 1"][2] == "model"
        breakpoint()
        assert "optimize/" in df["all.4"][0]
        assert df["all.4"][1] == "mean"
        assert df["all.8"][0] == "train/e2e_time"
        assert df["all.8"][1] == "mean"
        assert df["all.12"][0] == "train/iter_time"
        assert df["all.12"][1] == "mean"
        assert df["all.2"][0] == "test(train)/e2e_time"
        assert df["all.2"][1] == "mean"
        assert df["all.6"][0] == "test/iter_time"
        assert df["all.6"][1] == "mean"
        assert df["all.10"][0] == "train/epoch"
        assert df["all.10"][1] == "mean"

    # check_format(df)
    results_to_report = [col for col in df.columns if re.search(r'^all\.(2|4|6|10|12|14|16|18|20|22|24)$', col)]
    results_to_report.insert(0, "all") # export
    possible_score_naming = ["image_F1Score", "Dice", "f1-score", "PCK"]
    columns_to_keep = ['data_group', 'device', 'Unnamed: 1'] + results_to_report
    df = df[columns_to_keep]
    new_columns = []
    # rename to more readable names
    for col in df.columns:
        if col.startswith('all'):
            appended = False
            for acc_name in possible_score_naming:
                if acc_name in df[col][0]:
                    type_name = df[col][0].split("/")[0]
                    new_columns.append(type_name + "/" + "accuracy")
                    appended = True
                    break
            if not appended:
                new_columns.append(df[col][0])
        elif col == 'Unnamed: 1':
            new_columns.append('model_name')
        elif col == 'data_group':
            new_columns.append('task')
        else:
            new_columns.append(col)
    df.columns = new_columns
    # drop redundant first 3 rows from OTX benchmark
    df = df.drop(index=[0,1,2])
    df.reset_index(drop=True, inplace=True)
    # split tasks if necessery.
    # It may happen that there are multiple tasks in one file (classification)
    tasks = df['task'].unique()
    if "anomaly" in tasks or "anomaly_classification" in tasks:
        df["task"] = "anomaly"
    if len(tasks) > 1:
        task_dfs = []
        for task in tasks:
            task_df = df[df['task'] == task].copy()
            task_df.dropna(axis=1, inplace=True)
            task_dfs.append(task_df)
        return task_dfs
    else:
        return [df]


def load_csv_files(folder_path, device_name, add_test_results, report_all, tasks_to_drop=[]):
    dataframes = []
    for file_path in glob.glob(os.path.join(folder_path, '**', 'benchmark-summary.csv'), recursive=True):
        df = pd.read_csv(file_path)
        if df["data_group"][3] in tasks_to_drop:
            continue
        df['device'] = device_name
        df = preprocess_df(df, add_test_results, report_all)
        dataframes.extend(df)
    return dataframes


def create_pivot(arc770_dfs, gtx3090_dfs, output_folder):
    all_dfs = arc770_dfs + gtx3090_dfs
    combined_df = pd.concat(all_dfs)
    for col in combined_df.columns:
        # convert to float if possible
        if combined_df[col].dtype == 'object':
            try:
                combined_df[col] = combined_df[col].astype(float)
            except ValueError:
                pass
    combined_df = combined_df.round(4)
    combined_df = combined_df.sort_values(by=['task', 'model_name'])
    tasks = combined_df['task'].unique()
    diff_data = []
    task_dfs = []

    for task in tasks:
        task_df = combined_df[combined_df['task'] == task]
        arc770_rows = task_df[task_df['device'] == 'releases/2.3']
        gtx3090_rows = task_df[task_df['device'] == 'releases/2.2']
        if len(arc770_rows) != len(gtx3090_rows):
            arc770_rows = arc770_rows[arc770_rows['model_name'].isin(gtx3090_rows['model_name'])]
            gtx3090_rows = gtx3090_rows[gtx3090_rows['model_name'].isin(arc770_rows['model_name'])]
            task_df = task_df[task_df['model_name'].isin(arc770_rows['model_name'])]
        if not arc770_rows.empty and not gtx3090_rows.empty:
            diff_row_rel = arc770_rows.copy()
            for column in combined_df.columns:
                if column not in ['task', 'device', "model_name"]:
                    if "time" in column:
                        times = gtx3090_rows[column].values / arc770_rows[column].values
                    else:
                        times = arc770_rows[column].values / (gtx3090_rows[column].values + 1e-6)
                    procents = times * 100 - 100
                    diff_row_rel[column] = [f"{t:.4f} ({p:.2f}%)" for t, p in zip(times, procents)]
            diff_row_rel['device'] = 'OTX2.3/OTX2.2_diff'
            diff_data.append(diff_row_rel)
            task_dfs.append(task_df)

    diff_df = pd.concat(diff_data)
    task_dfs_combined = pd.concat(task_dfs)
    result_df = pd.concat([task_dfs_combined, diff_df])
    result_df = result_df.sort_values(by=['task', 'model_name'])
    # save to csv
    output_name = Path(output_folder) / 'benchmark_comparison_result.xlsx'
    with pd.ExcelWriter(output_name) as writer:
        result_df.to_excel(writer, index=False)
    return output_name


def create_per_task_summary(arc770_dfs, gtx3090_dfs, output_folder):
    arc770_dfs = sorted(arc770_dfs, key=lambda df: df['task'].iloc[0])
    gtx3090_dfs = sorted(gtx3090_dfs, key=lambda df: df['task'].iloc[0])
    assert len(arc770_dfs) == len(gtx3090_dfs), "Different number of tasks in ARC770 and GTX3090 benchmarks"
    all_task_dfs = []
    for arcdf, gtxdf in zip(arc770_dfs, gtx3090_dfs):
        combined_df = pd.concat([arcdf, gtxdf])
        for col in combined_df.columns:
            # convert to float if possible
            if combined_df[col].dtype == 'object':
                try:
                    combined_df[col] = combined_df[col].astype(float)
                except ValueError:
                    pass
        combined_df = combined_df.round(4)
        combined_df = combined_df.sort_values(by=['task', 'model_name'])
        arc770_rows = combined_df[combined_df['device'] == 'releases/2.3']
        gtx3090_rows = combined_df[combined_df['device'] == 'releases/2.2']
        if len(arc770_rows) != len(gtx3090_rows):
            arc770_rows = arc770_rows[arc770_rows['model_name'].isin(gtx3090_rows['model_name'])]
            gtx3090_rows = gtx3090_rows[gtx3090_rows['model_name'].isin(arc770_rows['model_name'])]
            combined_df = combined_df[combined_df['model_name'].isin(arc770_rows['model_name'])]
        if not arc770_rows.empty and not gtx3090_rows.empty:
            diff_row_rel = arc770_rows.copy()
            for column in combined_df.columns:
                if column not in ['task', 'device', "model_name"]:
                    if "time" in column:
                        times = gtx3090_rows[column].values / arc770_rows[column].values
                    else:
                        times = arc770_rows[column].values / (gtx3090_rows[column].values + 1e-6)
                    procents = times * 100 - 100
                    diff_row_rel[column] = [f"{t:.4f} ({p:.2f}%)" for t, p in zip(times, procents)]
            diff_row_rel['device'] = 'OTX2.3/OTX2.2_diff'
        result_df = pd.concat([combined_df, diff_row_rel])
        result_df = result_df.sort_values(by=['task', 'model_name'])
        all_task_dfs.append(result_df)
    # save to csv
    output_name = Path(output_folder) / 'benchmark_comparison_per_task.xlsx'
    with pd.ExcelWriter(output_name) as writer:
        for i, task_df in enumerate(all_task_dfs):
            task_name = task_df['task'].iloc[0]
            if "/" in task_name:
                task_name = task_name.replace("/", "_")
            task_df.to_excel(writer, sheet_name=task_name, index=False)
    return output_name


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process benchmark CSV files.')

    parser.add_argument('--add_test_results', action='store_true', help='Include test statistics (e2e, test iter time, epochs) in the report')
    parser.add_argument('--per_task', action='store_true', help='Generate per task detailed summary instead of 1 pivot table')
    parser.add_argument('--arc770_folder', type=str, required=True, help='Path to the ARC770 benchmark folder')
    parser.add_argument('--gtx3090_folder', type=str, required=True, help='Path to the GTX3090 benchmark folder')
    parser.add_argument('--output_folder', type=str, required=False, help='Path to the output folder', default='./')
    parser.add_argument('--tasks_to_drop', type=str, action="append", help='Tasks to drop from the report')

    args = parser.parse_args()
    add_test_results = args.add_test_results
    per_task = args.per_task
    arc770_folder = args.arc770_folder
    gtx3090_folder = args.gtx3090_folder
    output_folder = args.output_folder
    tasks_to_drop = args.tasks_to_drop if args.tasks_to_drop else []

    arc770_dfs = load_csv_files(arc770_folder, 'releases/2.3', add_test_results, report_all = per_task, tasks_to_drop=tasks_to_drop)
    gtx3090_dfs = load_csv_files(gtx3090_folder, 'releases/2.2', add_test_results, report_all = per_task, tasks_to_drop=tasks_to_drop)
    if per_task:
        output_name = create_per_task_summary(arc770_dfs.copy(), gtx3090_dfs.copy(), output_folder)
        print("Done! Path to per task summary table: ", output_name)
    else:
        output_name = create_pivot(arc770_dfs.copy(), gtx3090_dfs.copy(), output_folder)
        print("Done! Path to pivot table: ", output_name)
