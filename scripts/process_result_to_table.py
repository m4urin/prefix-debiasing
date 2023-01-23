from src.utils.files import read_file, write_file


def aggregate_f(col_data):
    s = col_data[col_data.notnull()]
    s = s.mean()
    return s


def seat_results(name):
    df = read_file(f'experiments/outputs/results/{name}.csv')
    df = df[df['downstream_task'].isnull()]
    df = df.sort_values(by=['model_name', 'model_type']).reset_index(drop=True)
    # df = df.dropna(axis=1, how='all')  # remove columns if all is NaN
    df = df.dropna(axis=1)  # remove columns if any is NaN

    def f(x):
        x = str(round(x, 2))
        if len(x.split('.')[-1]) == 1:
            x += '0'
        return x

    cols = ['model_name', 'model_type']
    for i in [6, 7, 8, 'stereo']:
        e, p, c = f'seat {i} effect_size', f'seat {i} p_val', f'SEAT-{i}'
        df[e] = df[e].apply(f)
        df[p] = df[p].apply(lambda x: '†' if x < 0.05 else '')
        df[c] = df[[p, e]].agg(''.join, axis=1)
        cols.append(c)
    df = df[cols]
    print(df.to_string())
    write_file(f'experiments/outputs/results/processed/{name}_seat.csv', df)


def lpbs_results(name):
    df = read_file(f'experiments/outputs/results/{name}.csv')
    df = df[df['downstream_task'].isnull()]
    df = df.sort_values(by=['model_name', 'model_type']).reset_index(drop=True)
    # df = df.dropna(axis=1, how='all')  # remove columns if all is NaN
    df = df.dropna(axis=1)  # remove columns if any is NaN

    def f(x):
        x = str(round(x, 2))
        if len(x.split('.')[-1]) == 1:
            x += '0'
        return x
    cols = ['model_name', 'model_type']
    for i, c in [('adjectives', 'adjectives'), ('occupations', 'occupations'), ('kaneko_stereotypes', 'stereotypes')]:
        e, p, c = f'lpbs {i} bias_score', f'lpbs {i} bias_score_std', f'LPBS {c}'
        df[e] = df[e].apply(f)
        df[p] = df[p].apply(f)
        df[c] = df[[e, p]].agg(' ± '.join, axis=1)
        cols.append(c)
    df = df[cols]
    print(df.to_string())
    write_file(f'experiments/outputs/results/processed/{name}_lpbs.csv', df)


def downstream_results(name):
    df = read_file(f'experiments/outputs/results/{name}.csv')
    df['downstream_task'] = df['downstream_task'].apply(lambda x: 'glue' if x in {'sst2', 'mrpc', 'rte', 'wsc'} else 'other')
    df = df[df['downstream_task'] == 'glue']

    # df = df.dropna(axis=1, how='all')  # remove columns if all is NaN
    #df = df.dropna(axis=1)  # remove columns if any is NaN
    df = df[['model_name', 'model_type', 'debias_method',
             'sst2', 'mrpc', 'rte', 'wsc']]
    df['debias_method'] = df['debias_method'].fillna('')

    df = df.groupby(by=['model_name', 'model_type', 'debias_method']).agg(aggregate_f).reset_index()

    df = df.sort_values(by=['model_name', 'model_type', 'debias_method']).reset_index(drop=True)
    df = df[['model_name', 'model_type',
             'sst2', 'mrpc', 'rte', 'wsc']]
    #df['training_time (minutes)'] = df['training_time (minutes)'].apply(lambda x: round(x, 1))
    print(df.to_string())
    print()
    write_file(f'experiments/outputs/results/processed/{name}_downstream.csv', df)


def probe_results(name):
    df = read_file(f'experiments/outputs/results/{name}.csv')
    df = df[df['downstream_task'] == 'probe']
    df = df[['model_name', 'model_type', 'probe gender_acc',
             'probe stereotype_acc', 'probe stereotype_conf', 'probe p_value']]
    print(df.to_string())
    print()
    write_file(f'experiments/outputs/results/processed/{name}_probe.csv', df)


name = '4_model_test'
seat_results(name)
lpbs_results(name)
downstream_results(name)
probe_results(name)

#df['downstream_task'] = df['downstream_task'].fillna('')
#df['debias_method'] = df['debias_method'].fillna('')

#df = df[df['downstream_task'] == '']
#df = df.sort_values(by=['model_name', 'model_type']).reset_index(drop=True)
#df = df.dropna(axis=1, how='all')  # remove columns if all is NaN
#df = df.dropna(axis=1)  # remove columns if any is NaN
#df = df[['model_name', 'model_type']]

#df['downstream_task'] = df['downstream_task'].apply(lambda x: 'glue' if x in {'sst2', 'mrpc', 'rte', 'wsc'} else x)
#df = df.groupby(['model_name', 'debias_method', 'model_type', 'downstream_task']).agg(aggregate_f).reset_index()
#df = df.sort_values(by=['debias_method', 'downstream_task', 'model_type', 'model_name']).reset_index()

#write_file(f'experiments/outputs/results/{name}_aggr.csv', df)

