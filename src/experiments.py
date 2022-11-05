from typing import Union
from src.language_models import MaskedLanguageModel, Config
from src.evaluation_metrics import run_metrics
from src.trainers import Trainer, TrainResult
from src.utils import dataframe_from_dicts, flatten_dict, DEVICE, DATA_DIR, print_title


def configs_from_hyper_parameters(hp: dict) -> list[Config]:
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
    return [Config(**c) for c in all_configs]


class ExperimentResult:
    def __init__(self,
                 config: Union[dict, Config],
                 train_result: Union[dict, TrainResult] = None,
                 evaluations: dict = None):
        self.config = config if isinstance(config, Config) else Config(**config)
        train_result = {} if train_result is None else train_result
        self.train_result = train_result if isinstance(train_result, TrainResult) else TrainResult(**train_result)
        self.evaluations: dict = evaluations if evaluations is not None else {}

    def to_dict(self):
        d = {'config': self.config.to_dict()}
        if self.train_result.training_completed:
            d['train_result'] = self.train_result.to_dict()
        if len(self.evaluations) > 0:
            d['evaluations'] = self.evaluations
        return d


class ExperimentTracker:
    def __init__(self, name: str):
        self.cache = DATA_DIR.get_folder('.cache', create_if_not_exist=True)
        self.work_dir = DATA_DIR \
            .get_folder('experiments', create_if_not_exist=True) \
            .get_folder(name, create_if_not_exist=True)
        self.configs = configs_from_hyper_parameters(self.work_dir.read_file('hyper_parameters.json'))
        self.name = name

    def run_experiment(self, skip_train_if_cached=True, skip_eval_if_cached=True):
        print_title(self.name)
        all_evaluations = []
        for config in self.configs:
            print(config)
            file_name = config.get_hash() + '.pt'
            if self.cache.file_exists(file_name):
                data = self.cache.read_file(file_name)
            else:
                data = {'config': config}
            result = ExperimentResult(**data)

            do_training = config.is_trainable() and (not skip_train_if_cached or
                                                     not result.train_result.training_completed)
            do_evaluation = not skip_eval_if_cached or len(result.evaluations) == 0
            model = None

            if do_training:
                # must be trained
                model = MaskedLanguageModel.from_config(config).to(DEVICE)
                result.train_result = Trainer.from_config(config).train(model)
                self.cache.write_file(file_name, result.to_dict())
            else:
                print(f'\tSkip training..   ({file_name})')

            if do_evaluation:
                if not do_training:
                    model = MaskedLanguageModel.from_config(config, parameters=result.train_result.parameters)\
                        .to(DEVICE)
                result.evaluations = run_metrics(model)
                self.cache.write_file(file_name, result.to_dict())
            else:
                print(f'\tSkip evaluation.. ({file_name})')

            all_evaluations.append({**result.config.to_dict(),
                                    'training_time (minutes)': round(result.train_result.total_training_time/60, 1),
                                    'final_loss': result.train_result.minimal_loss,
                                    'n_parameters': result.train_result.n_parameters,
                                    **flatten_dict(result.evaluations)})
        df = dataframe_from_dicts(all_evaluations).fillna('')
        self.work_dir.write_file('results.csv', df)
        print(df.to_string())
