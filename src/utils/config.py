from time import time
from typing import Union
import hashlib

from src.utils.files import read_file
from src.utils.functions import convert, nested_loop_dict
from src.utils.printing import explore_dict
from src.utils.pytorch import count_parameters

MODEL_NAMES = ['distilbert-base-uncased', 'roberta-base']
MODEL_TYPES = ['base', 'finetune', 'prefix']
PREFIX_MODES = ['identity', 'linear', 'replace']
PREFIX_LAYERS = ['all', 'half']
PREFIX_FINETUNES = ['prefix', 'model', 'all']
OBJECTIVES = ['kaneko', 'coreference-resolution']
EXTENSIONS = ['coreference-resolution']

LR = {
    'kaneko': {
        'finetune': 2e-5,
        'prefix': 1e-4
    },
    'coreference-resolution': {
        'base': 5e-5,
        'finetune': 5e-5,
        'prefix': 5e-5
    }
}
EPOCHS = {'kaneko': 3, 'coreference-resolution': 8}
BATCH_SIZE = {'kaneko': 32, 'coreference-resolution': 16}
WARMUP = {'kaneko': 100, 'coreference-resolution': 10}
SEED = 42


class UserDict:
    def __init__(self, **kwargs):
        pass

    def __eq__(self, other):
        if isinstance(other, UserDict):
            other_dict = other.to_dict()
            self_dict = self.to_dict()
            for key in set(list(other_dict.keys()) + list(self_dict.keys())):
                if key not in self_dict or key not in other_dict or self_dict[key] != other_dict[key]:
                    return False
            return True
        return False

    def __repr__(self):
        return f'{self.__class__.__name__}()'

    def __hash__(self):
        return hash(str(self.to_dict()))

    def __str__(self):
        return explore_dict({self.__class__.__name__: self.to_dict()})

    def get_filename(self):
        return hashlib.sha1(str(self.to_dict()).encode()).hexdigest() + '.pt'

    def to_dict(self):
        return {k: v.to_dict() if isinstance(v, UserDict) else v for k, v in self.__dict__.items()}

    @classmethod
    def from_dict(cls, data):
        if isinstance(data, UserDict):
            data = data.to_dict()
        return cls(**data)


class Config(UserDict):
    def __init__(self):
        super().__init__()
        errors = []
        self.verify_config(errors)
        if len(errors) > 0:
            raise ValueError(str(self) + '\n- '.join([''] + errors))

    def verify_attribute(self, errors: list[str], attr_name: str, numeric=None, text: list[str] = None):
        if not hasattr(self, attr_name):
            errors.append(f"Value for attribute '{attr_name}' is missing.")
            return errors
        attr_value = getattr(self, attr_name)
        if text is not None and attr_value in text:
            return errors
        if text is not None and attr_value not in text:
            if isinstance(attr_value, str) or numeric is None:
                txt = f"{type(attr_value).__name__} value '{attr_value}' for attribute '{attr_name}' " \
                      f"is not supported, please choose from: {text}"
                if numeric is not None:
                    numeric_type, lower_bound, upper_bound = numeric
                    txt += f", or provide a {numeric_type.__name__} in the range of [{lower_bound}, {upper_bound}]"
                errors.append(txt + '.')
                return
        if numeric is not None:
            numeric_type, lower_bound, upper_bound = numeric
            # is numeric
            if type(attr_value) != numeric_type or attr_value < lower_bound or attr_value >= upper_bound:
                txt = f"{type(attr_value).__name__} value '{attr_value}' for attribute '{attr_name}' " \
                      f"should be a {numeric_type.__name__} in the range of [{lower_bound}, {upper_bound}]"
                if text is not None:
                    txt += f', or one of the following: {text}'
                errors.append(txt + '.')

    def verify_config(self, errors: list[str]):
        raise NotImplementedError('Not implemented yet.')


class ModelConfig(Config):
    def __init__(self,
                 model_name: str,
                 model_type: str,
                 objective: str,
                 prefix_mode: str = None,
                 prefix_layers: Union[str, int] = None,
                 n_prefix_tokens: int = None,
                 prefix_finetune: str = None,
                 **kwargs):
        self.model_name = model_name
        self.model_type = model_type
        self.objective = objective
        if self.is_prefix():
            self.prefix_mode = prefix_mode
            self.prefix_layers = convert(prefix_layers, int, prefix_layers)
            self.n_prefix_tokens = n_prefix_tokens
            if not self.is_default():
                self.prefix_finetune = prefix_finetune
        if self.can_train():
            self.epochs = EPOCHS[objective]
            self.batch_size = BATCH_SIZE[objective]
            self.lr = LR[objective][model_type]
            self.num_warmup_steps = WARMUP[objective]
            self.seed = SEED
        super().__init__()

    def verify_config(self, errors: list[str]):
        self.verify_attribute(errors, 'model_name', text=MODEL_NAMES)
        self.verify_attribute(errors, 'model_type', text=MODEL_TYPES)
        self.verify_attribute(errors, 'objective', text=OBJECTIVES)
        if self.is_prefix():
            self.verify_attribute(errors, 'prefix_mode', text=PREFIX_MODES)
            self.verify_attribute(errors, 'prefix_layers', numeric=(int, 1, 24), text=PREFIX_LAYERS)
            self.verify_attribute(errors, 'n_prefix_tokens', numeric=(int, 1, 65))
            if not self.is_default():
                self.verify_attribute(errors, 'prefix_finetune', text=PREFIX_FINETUNES)
        if self.can_train():
            self.verify_attribute(errors, 'epochs', numeric=(int, 0, 100))
            self.verify_attribute(errors, 'batch_size', numeric=(int, 1, 65))
            self.verify_attribute(errors, 'lr', numeric=(float, 1e-7, 0.1))
            self.verify_attribute(errors, 'num_warmup_steps', numeric=(int, 1, 10000))
            self.verify_attribute(errors, 'seed', numeric=(int, 0, 1000000000))

    def is_prefix(self) -> bool:
        return self.model_type == 'prefix'

    def is_finetune(self) -> bool:
        return self.model_type == 'finetune'

    def is_base(self) -> bool:
        return self.model_type == 'base'

    def is_default(self):
        return self.objective == 'kaneko'

    def without_extensions(self):
        if not self.is_default():
            as_dict = self.to_dict()
            as_dict['objective'] = 'kaneko'
            return ModelConfig(**as_dict)
        return self

    def as_base(self):
        return ModelConfig(self.model_name, 'base', 'kaneko')

    def can_train(self):
        return not (self.is_base() and self.is_default())

    @staticmethod
    def from_hyper_params(file: dict):
        result, done = [], set()
        for d in nested_loop_dict(file):
            config = ModelConfig(**d)
            if config not in done:
                result.append(config)
                done.add(config)
        return result

    def __str__(self):
        param_names = {'model_name': "{}",
                       'model_type': "{}",
                       'objective': "{}",
                       'prefix_mode': "mode={}",
                       'prefix_layers': "layers={}",
                       'n_prefix_tokens': "n_tok={}",
                       'prefix_finetune': "param={}"}
        info = ', '.join([s.format(getattr(self, k)) for k, s in param_names.items() if hasattr(self, k)])
        return f"ModelConfig({info})"

    def get_filename(self):
        info = {'model_name': "{}",
                'model_type': "{}",
                'objective': "{}",
                'prefix_mode': "mode={}",
                'prefix_layers': "layers={}",
                'n_prefix_tokens': "n_tok={}",
                'prefix_finetune': "param={}"}
        info = [s.format(getattr(self, k)) for k, s in info.items() if hasattr(self, k)]
        info.append(f"hash={hashlib.sha1(str(self.to_dict()).encode()).hexdigest()[:8]}")
        info = ','.join(info)
        return info + '.pt'


class ModelResult(UserDict):
    def __init__(self,
                 model_config: ModelConfig,
                 training_completed: bool = False,
                 total_training_time: int = -1,
                 scalars: dict = None,
                 evaluations: dict = None,
                 parameters=None,
                 n_parameters: int = 0
                 ):
        super().__init__()
        self.model_config = model_config
        self.training_completed: bool = training_completed
        self.total_training_time: int = total_training_time
        self.scalars: dict = scalars if scalars is not None else {}
        self.evaluations = evaluations if evaluations is not None else {}
        self.parameters = parameters
        self.n_parameters = n_parameters

    def add_scalar(self, name: str, scalar: float):
        names = name.split('/')
        sub = self.scalars
        for n in names[:-1]:
            if n not in sub:
                sub[n] = {}
            sub = sub[n]
        if names[-1] not in sub:
            sub[names[-1]] = []
        sub[names[-1]].append(scalar)

    def finish_training(self, start_time, model_parameters):
        self.parameters = model_parameters
        self.n_parameters = count_parameters(model_parameters)
        self.total_training_time = int(round(time() - start_time))
        self.training_completed = True
        return self

    def has_parameters(self):
        return self.parameters is not None

    @classmethod
    def from_dict(cls, data: dict):
        if isinstance(data['model_config'], dict):
            data['model_config'] = ModelConfig.from_dict(data['model_config'])
        return ModelResult(**data)


if __name__ == '__main__':
    data = read_file('experiments/test/config.json')
    for m in ModelConfig.from_hyper_params(data):
        print(m.get_filename())
        print(m)
        print()

