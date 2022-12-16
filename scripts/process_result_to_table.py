from src.utils.files import read_file, write_file


def aggregate_f(col_data):
    s = col_data[col_data.notnull()]
    s = s.mean()
    return s


name = '4_model_test'
df = read_file(f'experiments/outputs/results/{name}.csv')
df['downstream_task'] = df['downstream_task'].fillna('')
df['downstream_task'] = df['downstream_task'].apply(lambda x: 'glue' if x in {'sst2', 'mrpc', 'rte', 'wsc'} else x)
df = df.groupby(['model_name', 'model_type', 'downstream_task']).agg(aggregate_f).reset_index()
df = df.sort_values(by=['downstream_task', 'model_type', 'model_name'])





write_file(f'experiments/outputs/results/{name}_aggr.csv', df)
print(df.to_string())
