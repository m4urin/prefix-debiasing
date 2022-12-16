from typing import Union

from src.data.structs.model_config import ModelConfig
from src.data.structs.results import TrainResult, EvaluationResult
from src.language_models.language_model import LanguageModel
from src.evaluations.evaluation_metrics import run_metrics
from src.experiments.trainers import train_model
from src.utils.files import get_folder, read_file
from src.utils.functions import dataframe_from_dicts
from src.utils.printing import print_title


class Experiment:
    def __init__(self, name: str,
                 force_training=False,
                 force_evaluation=False,
                 exclude_train: set[ModelConfig] = None,
                 exclude_eval: set[ModelConfig] = None):
        self.name = name
        self.model_configs = ModelConfig.from_hyper_params(read_file(f'experiments/inputs/settings/{name}.json'))
        self.output_train = get_folder('experiments/outputs/trained_models', create_if_not_exists=True)
        self.output_eval = get_folder('experiments/outputs/evaluations', create_if_not_exists=True)
        self.output_result = get_folder('experiments/outputs/results', create_if_not_exists=True)

        self.force_training = force_training
        self.force_evaluation = force_evaluation
        self.exclude_train = set() if exclude_train is None else exclude_train
        self.exclude_eval = set() if exclude_eval is None else exclude_eval

    def read_train(self, config: ModelConfig) -> TrainResult:
        file_name = config.get_filename() + '.pt'
        if self.output_train.file_exists(file_name):
            return TrainResult.from_dict(self.output_train.read_file(file_name))
        else:
            return TrainResult(config)

    def read_eval(self, config: ModelConfig) -> EvaluationResult:
        file_name = config.get_filename() + '.json'
        if self.output_eval.file_exists(file_name):
            return EvaluationResult.from_dict(self.output_eval.read_file(file_name))
        else:
            return EvaluationResult(config)

    def train(self, config: ModelConfig):
        result = self.read_train(config)
        if config not in self.exclude_train and (self.force_training or not result.training_completed):
            print('Train', config)
            if config.is_downstream():
                requires = config.requires_model()
                print('Load required model:', requires)
                model = self.read_train(requires).model
                if model is None:
                    raise ValueError(f'Cannot train {config}, because the required model '
                                     f'({requires}) does not exist.')
                model.add_cls_head(config)
            else:
                model = LanguageModel.from_config(config)
            result = train_model(model)
            self.output_train.write_file(result.get_filename(), result.to_dict())
            self.exclude_train.add(config)
        return result.total_training_time, result.n_parameters

    def eval(self, config: ModelConfig):
        result = self.read_eval(config)
        if config not in self.exclude_eval and (self.force_evaluation or len(result.evaluations) == 0):
            print('Evaluate', config)
            train_result = self.read_train(config)
            if not train_result.training_completed or train_result.model is None:
                raise ValueError(f'Cannot evaluate {config}, because it was never trained.')
            result.evaluations = run_metrics(train_result.model)
            self.output_eval.write_file(result.get_filename(), result.to_dict())
            self.exclude_eval.add(config)
        return result.evaluations

    def run(self):
        print_title(self.name)
        all_evaluations = []
        for config in self.model_configs:
            train_time, n_parameters = self.train(config)
            evaluations = self.eval(config)
            all_evaluations.append({**config.to_dict(),
                                    'training_time (minutes)': round(train_time / 60, 1),
                                    'n_parameters': n_parameters,
                                    **evaluations})
        df = dataframe_from_dicts(all_evaluations).fillna('').sort_values(by=['model_name', 'debias_method', 'model_type', 'n_prefix_tokens'])
        self.output_result.write_file(f'{self.name}.csv', df)
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
        exclude_train, exclude_eval = experiment.exclude_train, experiment.exclude_eval
