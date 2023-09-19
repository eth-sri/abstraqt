
import pandas as pd
from typing import List


def move_to_front(df: pd.DataFrame, columns: List[str]):
    moved_columns = [df.pop(col) for col in columns]

    for i, col in enumerate(columns):
        df.insert(i, col, moved_columns[i])
    
    return df


def value_counts_str(series, **kwargs):
    value_counts = series.value_counts(**kwargs)
    ret = ','.join([f'{index}:{count}' for index, count in value_counts.items()])
    return ret
