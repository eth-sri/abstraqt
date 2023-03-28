import os
import pandas as pd

from abstraqt.experiments.run_experiments import results_directory, circuits_csv_file, tools_file


def combine():
    combined = pd.read_csv(circuits_csv_file)
    for column in ['cx', 'h', 's', 'sdg', 'cz', 'x', 'z']:
        combined[column] = combined[column].fillna(0)
    combined['clifford'] = \
        combined['cx'] + \
        combined['h'] + \
        combined['s'] + \
        combined['sdg'] + \
        combined['cz'] + \
        combined['x'] + \
        combined['z']
    tools = pd.read_csv(tools_file)
    for index, row in tools.iterrows():
        tool_name = row['tool_name']

        results_file = os.path.join(results_directory, f'results_{tool_name}.csv')
        results = pd.read_csv(results_file, dtype={
            'label': str,
            'q_file_success': bool,
            'time': float,
            'error': str,
            'precise': float
        })

        results = results.add_suffix('_' + tool_name)
        results = results.rename(columns={'label_' + tool_name: 'label'})

        combined = combined.merge(results, on='label', how='outer')

    assert isinstance(combined, pd.DataFrame)
    return combined


def main():
    combined = combine()
    combined_file = os.path.join(results_directory, 'results_combined.csv')
    combined.to_csv(combined_file, index=False)


if __name__ == '__main__':
    main()
