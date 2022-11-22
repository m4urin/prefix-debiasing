from src.MLM import MLM, ModelConfig
from src.evaluation_metrics import run_metrics
from src.trainers import Trainer
from src.utils import dataframe_from_dicts, flatten_dict, print_title, stack_dicts
from src.utils.io import DATA_DIR
from src.utils.pytorch import DEVICE
from src.utils.user_dicts import TASKS, ExperimentResult
from copy import deepcopy


def configs_from_hyper_parameters(hp: dict) -> list[ModelConfig]:
    all_configs = []
    for model_name in hp['model_name']:
        for model_type in hp['model_type']:
            base_config = {'model_name': model_name, 'model_type': model_type}
            if model_type == 'base':
                all_configs.append(base_config)
            else:
                train_config_1 = {k: hp[k] for k in ['batch_size', 'epochs', 'num_warmup_steps', 'seed']}
                for loss_function in hp['loss_function']:
                    train_config_2 = {'loss_function': loss_function, 'lr': hp['lr'][model_type]}
                    if model_type == 'finetune':
                        all_configs.append({**base_config, **train_config_1, **train_config_2})
                    elif model_type == 'prefix':
                        for prefix_mode in hp['prefix_mode']:
                            for prefix_layers in hp['prefix_layers']:
                                for n_prefix_tokens in hp['n_prefix_tokens']:
                                    prefix_config = {'prefix_mode': prefix_mode, 'prefix_layers': prefix_layers,
                                                     'n_prefix_tokens': n_prefix_tokens}
                                    all_configs.append({**base_config, **train_config_1,
                                                        **train_config_2, **prefix_config})
                    else:
                        raise ValueError(f"Modeltype '{model_type}' not supported.")
    return [ModelConfig(**c) for c in all_configs]


class ExperimentTracker:
    def __init__(self, name: str):
        self.name = name
        self.cache = DATA_DIR.get_folder('.cache', create_if_not_exist=True)
        self.work_dir = DATA_DIR \
            .get_folder('experiments', create_if_not_exist=True) \
            .get_folder(self.name, create_if_not_exist=True)
        config_file = self.work_dir.read_file('config.json')
        self.configs = configs_from_hyper_parameters(config_file)
        self.tasks = config_file['tasks']

    def run_experiment(self, redo_training=None, redo_eval=None):
        # for tasks
        redo_training = set(redo_training) if redo_training is not None else set()
        redo_eval = set(redo_eval) if redo_eval is not None else set()

        print_title(self.name)
        all_evaluations = {task_name: [] for task_name in self.tasks}
        for config in self.configs:
            file_name = config.get_hash() + '.pt'
            if self.cache.file_exists(file_name):
                data = self.cache.read_file(file_name)
            else:
                data = {'config': config}
            result = ExperimentResult.from_dict(data)

            for task_name in self.tasks:
                task_result = result.task[task_name]
                print(f"\n{task_name}: {config}")
                can_train = config.model_type not in TASKS[task_name]
                force_training = task_name in redo_training
                force_evaluation = task_name in redo_eval
                do_training = can_train and (force_training or not task_result.train_info.training_completed)
                do_evaluation = do_training or force_evaluation or len(task_result.evaluations) == 0

                model = None

                if do_training:
                    # must be trained
                    if force_training:
                        print(f'   Force re-training of model..')
                    else:
                        print(f'   No parameters found, train model..')
                    parameters = None
                    if task_name == 'coreference-resolution':
                        parameters = deepcopy(result.task['default'].parameters)
                        if parameters is None:
                            raise ValueError('Default parameters not present..')
                        else:
                            print('Using parameters from default.')
                    model = MLM.from_config(config, task_name, parameters).to(DEVICE)
                    task_result.train_info = Trainer.run(task_name, model)
                    task_result.parameters = model.parameters_dict
                    self.cache.write_file(file_name, result.to_dict())
                else:
                    print(f'   Parameters found, skip training..')

                if do_evaluation:
                    if force_evaluation:
                        print(f'   Force re-evaluation of model..')
                    else:
                        print(f'   No evaluations found, evaluate model..')
                    if not do_training:
                        model = MLM.from_config(config, task_name, task_result.parameters).to(DEVICE)
                    task_result.evaluations = run_metrics(task_name, model)
                    self.cache.write_file(file_name, result.to_dict())
                else:
                    print(f'   Evaluations found, skip evaluation..')

                train_time = round(task_result.train_info.total_training_time / 60, 1)
                n_parameters = model.n_parameters if model is not None else 0
                all_evaluations[task_name].append({'task': task_name,
                                                   **result.config.to_dict(),
                                                   **flatten_dict(task_result.evaluations),
                                                   'training_time (minutes)': train_time,
                                                   'n_parameters': n_parameters})

        #print(all_evaluations)
        df = sum(all_evaluations.values(), [])
        df = dataframe_from_dicts(df).fillna('')
        self.work_dir.write_file(f'result.csv', df)
        print(df.to_string())

        #for task_name in self.tasks:
        #    df = dataframe_from_dicts(all_evaluations[task_name]).fillna('')
        #    self.work_dir.write_file(f'{task_name}.csv', df)
        #    print(df.to_string())
