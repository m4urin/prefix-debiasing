from typing import Union
import hashlib

from src.data.structs.userdict import Config
from src.utils.files import read_file
from src.utils.functions import convert, nested_loop_dict


MODEL_NAMES = ['distilbert-base-uncased', 'roberta-base', 'bert-base-uncased']
DEBIAS_METHODS = ['kaneko']
MODEL_TYPES = ['base', 'finetune', 'prefix']
PREFIX_MODES = ['identity', 'linear', 'replace']
PREFIX_LAYERS = ['all', 'half']

DEBIAS_PARAMS = {}
for row_dict in read_file('experiments/inputs/parameters/debias_parameters.csv').to_dict(orient="records"):
    _method, _type = row_dict['debias_method'], row_dict['model_type']
    if _method not in DEBIAS_PARAMS:
        DEBIAS_PARAMS[_method] = {}
    DEBIAS_PARAMS[_method][_type] = row_dict

GLUE_PARAMS = {}
for row_dict in read_file('experiments/inputs/parameters/downstream_parameters.csv').to_dict(orient="records"):
    GLUE_PARAMS[row_dict['task']] = row_dict

TASKS = list(GLUE_PARAMS.keys())
SEED = 42


class ModelConfig(Config):
    def __init__(self,
                 model_name: str,
                 model_type: str = 'base',
                 debias_method: str = None,
                 downstream_task: str = None,
                 prefix_mode: str = None,
                 prefix_layers: Union[str, int] = None,
                 n_prefix_tokens: int = None,
                 epochs: int = None,
                 batch_size: int = None,
                 lr: float = None,
                 num_warmup_steps: int = None,
                 seed: int = None,
                 head_size: int = None):
        self.model_name = model_name
        self.model_type = model_type
        has_debias = debias_method is not None and debias_method != '' and model_type != 'base'
        has_task = downstream_task is not None and downstream_task != ''
        if has_debias:
            self.debias_method = debias_method
            if self.is_prefix():
                self.prefix_mode = prefix_mode
                self.prefix_layers = convert(prefix_layers, int, prefix_layers)
                self.n_prefix_tokens = n_prefix_tokens
            if not has_task:
                train_params = DEBIAS_PARAMS[debias_method][self.model_type]
                self.epochs = train_params['epochs'] if epochs is None else epochs
                self.batch_size = train_params['batch_size'] if batch_size is None else batch_size
                self.lr = train_params['lr'] if lr is None else lr
                self.num_warmup_steps = train_params['num_warmup_steps'] if num_warmup_steps is None \
                    else num_warmup_steps
                self.seed = SEED if seed is None else seed
        if has_task:
            self.downstream_task = downstream_task
            glue_params = GLUE_PARAMS[downstream_task]
            self.head_size = glue_params['head_size'] if head_size is None else head_size
            self.epochs = glue_params['epochs'] if epochs is None else epochs
            self.batch_size = glue_params['batch_size'] if batch_size is None else batch_size
            self.lr = glue_params['lr'] if lr is None else lr
            self.num_warmup_steps = glue_params['num_warmup_steps'] if num_warmup_steps is None else num_warmup_steps
            self.seed = SEED if seed is None else seed

        super().__init__()

    def verify_config(self, errors: list[str]):
        self.verify_attribute(errors, 'model_name', text=MODEL_NAMES)
        self.verify_attribute(errors, 'model_type', text=MODEL_TYPES)
        if self.is_debiased():
            self.verify_attribute(errors, 'debias_method', text=DEBIAS_METHODS)
            if self.is_prefix():
                self.verify_attribute(errors, 'prefix_mode', text=PREFIX_MODES)
                self.verify_attribute(errors, 'prefix_layers', numeric=(int, 1, 24), text=PREFIX_LAYERS)
                self.verify_attribute(errors, 'n_prefix_tokens', numeric=(int, 1, 65))
        if self.can_train():
            self.verify_attribute(errors, 'epochs', numeric=(int, 0, 100))
            self.verify_attribute(errors, 'batch_size', numeric=(int, 1, 65))
            self.verify_attribute(errors, 'lr', numeric=(float, 1e-7, 0.1))
            self.verify_attribute(errors, 'num_warmup_steps', numeric=(int, 1, 10000))
            self.verify_attribute(errors, 'seed', numeric=(int, 0, 1000000000))
        if self.is_downstream():
            self.verify_attribute(errors, 'head_size', numeric=(int, 1, 4))

    def is_debiased(self):
        return hasattr(self, 'debias_method')

    def is_downstream(self):
        return hasattr(self, 'downstream_task')

    def is_prefix(self) -> bool:
        return self.model_type == 'prefix'

    def is_finetune(self) -> bool:
        return self.model_type == 'finetune'

    def is_base(self) -> bool:
        return self.model_type == 'base'

    def can_train(self):
        return not self.is_base() or self.is_downstream()

    def requires_model(self):
        if not self.is_downstream():
            return self
        method = self.debias_method if self.is_debiased() else None
        if self.is_prefix():
            return ModelConfig(self.model_name, self.model_type, method,
                               prefix_mode=self.prefix_mode,
                               prefix_layers=self.prefix_layers,
                               n_prefix_tokens=self.n_prefix_tokens)
        return ModelConfig(self.model_name, self.model_type, method)

    def to_original_model(self):
        return ModelConfig(self.model_name, 'base')

    @staticmethod
    def from_hyper_params(file: dict):
        result, done = [], set()
        for d in nested_loop_dict(file):
            c = ModelConfig(**d)
            if c not in done:
                result.append(c)
                done.add(c)
        return result

    def _formatted(self):
        info = {'model_name': "{}",
                'model_type': "{}",
                'prefix_mode': "mode={}",
                'prefix_layers': "layers={}",
                'n_prefix_tokens': "n_tok={}",
                'debias_method': "method={}",
                'downstream_task': "task={}",
                }
        return [s.format(getattr(self, k)) for k, s in info.items() if hasattr(self, k)]

    def __str__(self):
        return f"ModelConfig({', '.join(self._formatted())})"

    def get_filename(self):
        formatted = self._formatted() + [f"train_param={hashlib.sha1(str(self.to_dict()).encode()).hexdigest()[:8]}"]
        return ', '.join(formatted)


if __name__ == '__main__':
    data = read_file('experiments/experiment_settings/test.json')
    for i, m in enumerate(ModelConfig.from_hyper_params(data)):
        print(i)
        print(m.get_filename())
        print(m)
        print(m.requires_model())
        print()
