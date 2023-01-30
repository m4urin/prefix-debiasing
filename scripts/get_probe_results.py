import os

import numpy as np
import pandas as pd

from src.utils.files import get_all_files, read_file

data = {'name': [], 'gender_acc': [], 'stereo_acc': [], 'conf': [], 'stereo_acc_star': [], 'conf_star': []}
for f in get_all_files('experiments/outputs/gender_probe/', '.json'):
    e = f.read()
    data['name'].append(' '.join(f.name.split(', ')[:2]))
    data['gender_acc'].append(round(100*e['test_acc'][-1], 2))
    data['stereo_acc'].append(round(100*e['stereotype_acc'][-1], 2))
    data['conf'].append(round(e['conf'][-1], 3))
    e['stereotype_acc'] = e['stereotype_acc'][:10] + e['stereotype_acc'][11:21]
    i = np.array(e['stereotype_acc']).argmax()
    data['stereo_acc_star'].append(round(100 * e['stereotype_acc'][i], 2))
    data['conf_star'].append(round(e['conf'][i], 3))

    #eval = read_file(os.path.join('experiments/outputs/evaluations', f.name))["evaluations"]["probe"]
    #data['stereo_acc'] = eval["stereotype_acc"]
    #data['stereotype_conf'] = eval["stereotype_conf"]
    #data['p_value'] = eval["p_value"]

df = pd.DataFrame(data)
print(df.to_string())
