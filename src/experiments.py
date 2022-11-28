from typing import Union

from src.MLM import MLM
from src.evaluation_metrics import run_metrics
from src.trainers import train_model
from src.utils.config import ModelConfig, ModelResult
from src.utils.files import get_folder
from src.utils.functions import dataframe_from_dicts
from src.utils.printing import print_title


class Experiment:
    def __init__(self, name: str,
                 force_training=False,
                 force_evaluation=False,
                 exclude_train: set[ModelConfig] = None,
                 exclude_eval: set[ModelConfig] = None):
        self.name = name
        self.cache = get_folder('.cache', create_if_not_exists=True)
        self.work_dir = get_folder(f'experiments/{self.name}', create_if_not_exists=True)
        hyper_parameters = self.work_dir.read_file('config.json')
        self.configs = ModelConfig.from_hyper_params(hyper_parameters)

        self.force_training = force_training
        self.force_evaluation = force_evaluation
        self.exclude_train = set() if exclude_train is None else exclude_train
        self.exclude_eval = set() if exclude_eval is None else exclude_eval

    def read(self, config: ModelConfig) -> ModelResult:
        file_name = config.get_filename()
        if self.cache.file_exists(file_name):
            result = self.cache.read_file(file_name)
        else:
            result = {'model_config': config}
        return ModelResult.from_dict(result)

    def write(self, result: ModelResult):
        self.cache.write_file(result.model_config.get_filename(), result.to_dict())

    def train(self, config: ModelConfig):
        result = self.read(config)
        if config not in self.exclude_train and (self.force_training or not result.training_completed):
            print('Train', config)
            parameters = None
            if not config.is_default():
                print('Load debiased parameters..')
                parameters = self.read(config.to_default()).parameters
                if parameters is None:
                    raise ValueError(f'Cannot train {config}, because the base model '
                                     f'({config.to_default()}) does not exist.')
            model = MLM.from_config(config, parameters)
            result = train_model(model)
            self.write(result)
            self.exclude_train.add(config)

    def eval(self, config: ModelConfig):
        result = self.read(config)
        if config not in self.exclude_eval and (self.force_evaluation or len(result.evaluations) == 0):
            print('Evaluate', config)
            if result.parameters is None:
                raise ValueError(f'Cannot evaluate {config}, because it was never trained.')
            result.evaluations = run_metrics(MLM.from_config(config, result.parameters))
            self.write(result)
            self.exclude_eval.add(config)

    def run(self):
        print_title(self.name)
        all_evaluations = []
        for config in self.configs:
            self.train(config)
            self.eval(config)
            result = self.read(config)
            all_evaluations.append({**config.to_dict(),
                                    'training_time (minutes)': round(result.total_training_time / 60, 1),
                                    'n_parameters': result.n_parameters,
                                    **result.evaluations})
        df = dataframe_from_dicts(all_evaluations).fillna('').sort_values(by=['objective'])
        self.work_dir.write_file(f'result.csv', df)
        print(df.to_string(), '\n')
        return self


def run_experiments(names: Union[str, list[str]],
                    *more_names: str,
                    force_training=False,
                    force_evaluation=False,
                    exclude_train: set[ModelConfig] = None,
                    exclude_eval: set[ModelConfig] = None):
    all_names = (names, *more_names) if isinstance(names, str) else (*names, *more_names)
    for name in all_names:
        experiment = Experiment(name, force_training, force_evaluation, exclude_train, exclude_eval).run()
        exclude_train = experiment.exclude_train
        exclude_eval = experiment.exclude_eval
