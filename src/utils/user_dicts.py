from typing import Union

from src.utils import try_int, explore_dict
import hashlib

MODEL_NAMES = ['distilbert-base-uncased', 'roberta-base']
MODEL_TYPES = ['base', 'finetune', 'prefix']
PREFIX_MODES = ['identity', 'linear', 'replace']
PREFIX_LAYERS = ['all', 'half']
LOSS_FUNCTIONS = ['kaneko']
TASKS = {'default': {'base'}, 'coreference-resolution': set()}  # 'sentiment-analysis'


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
        return str(self)

    def __hash__(self):
        return hash(str(self.to_dict()))

    def __str__(self):
        s = explore_dict(self.to_dict()).replace('\n', '\n   ')
        return f'{self.__class__.__name__} (\n   {s}\n)'

    def get_hash(self):
        return hashlib.sha1(str(self.to_dict()).encode()).hexdigest()

    def to_dict(self):
        return self.__dict__

    @classmethod
    def from_dict(cls, data: dict):
        return cls(**data)


class Config(UserDict):
    def __init__(self):
        super().__init__()
        errors = self.verify_config()
        if len(errors) > 0:
            raise ValueError(str(self) + '\n- '.join([''] + errors))

    def verify_attribute(self, attr_name: str, numeric=None, text: list[str] = None) -> list[str]:
        errors = []
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
                return errors
        if numeric is not None:
            numeric_type, lower_bound, upper_bound = numeric
            # is numeric
            if type(attr_value) != numeric_type or attr_value < lower_bound or attr_value >= upper_bound:
                txt = f"{type(attr_value).__name__} value '{attr_value}' for attribute '{attr_name}' " \
                      f"should be a {numeric_type.__name__} in the range of [{lower_bound}, {upper_bound}]"
                if text is not None:
                    txt += f', or one of the following: {text}'
                errors.append(txt + '.')
        return errors

    def verify_config(self) -> list[str]:
        raise NotImplementedError('Not implemented yet.')


class ModelConfig(Config):
    def __init__(self,
                 model_name: str,
                 model_type: str,
                 prefix_mode: str = None,
                 prefix_layers: Union[str, int] = None,
                 n_prefix_tokens: int = None,
                 loss_function: str = None,
                 epochs: int = None,
                 batch_size: int = None,
                 lr: float = None,
                 num_warmup_steps: int = None,
                 seed: int = None,
                 task: str = 'default'):
        self.model_name = model_name
        self.model_type = model_type
        if self.is_prefix():
            self.prefix_mode = prefix_mode
            self.prefix_layers = try_int(prefix_layers)
            self.n_prefix_tokens = n_prefix_tokens
        if not self.is_base() or task == 'coreference_resolution':
            self.loss_function = loss_function
            self.epochs = epochs
            self.batch_size = batch_size
            self.lr = lr
            self.num_warmup_steps = num_warmup_steps
            self.seed = seed
        super().__init__()

    def verify_config(self) -> list[str]:
        errors = []
        errors.extend(self.verify_attribute('model_name', text=MODEL_NAMES))
        errors.extend(self.verify_attribute('model_type', text=MODEL_TYPES))
        if self.is_prefix():
            errors.extend(self.verify_attribute('prefix_mode', text=PREFIX_MODES))
            errors.extend(self.verify_attribute('prefix_layers', numeric=(int, 1, 24), text=PREFIX_LAYERS))
            errors.extend(self.verify_attribute('n_prefix_tokens', numeric=(int, 1, 65)))
        if not self.is_base():
            errors.extend(self.verify_attribute('loss_function', text=LOSS_FUNCTIONS))
            errors.extend(self.verify_attribute('epochs', numeric=(int, 1, 10)))
            errors.extend(self.verify_attribute('batch_size', numeric=(int, 1, 65)))
            errors.extend(self.verify_attribute('lr', numeric=(float, 1e-7, 0.1)))
            errors.extend(self.verify_attribute('num_warmup_steps', numeric=(int, 2, 10000)))
            errors.extend(self.verify_attribute('seed', numeric=(int, 0, 1000000000)))
        return errors

    def is_prefix(self) -> bool:
        return self.model_type == 'prefix'

    def is_finetune(self) -> bool:
        return self.model_type == 'finetune'

    def is_base(self) -> bool:
        return self.model_type == 'base'


class TrainInfo(UserDict):
    def __init__(self,
                 training_completed: bool = False,
                 total_training_time: int = 0,
                 train_loss: float = float('INF'),
                 test_loss: float = float('INF'),
                 train_metrics=None,
                 test_metrics=None,
                 scalars: dict = None
                 ):
        super().__init__()
        self.training_completed: bool = training_completed
        self.total_training_time: int = total_training_time
        self.train_loss: float = train_loss
        self.test_loss: float = test_loss
        self.train_metrics: float = train_metrics
        self.test_metrics: float = test_metrics
        self.scalars: dict = scalars if scalars is not None else {}

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


class TaskResult(UserDict):
    def __init__(self, train_info: TrainInfo = None,
                 evaluations: dict = None,
                 parameters=None):
        super().__init__()
        self.train_info = train_info if train_info is not None else TrainInfo()
        self.evaluations = evaluations if evaluations is not None else {}
        self.parameters = parameters

    @classmethod
    def from_dict(cls, data: dict):
        train_info = TrainInfo.from_dict(data['train_info']) if 'train_info' in data else TrainInfo()
        evaluations = data['evaluations'] if 'evaluations' in data else {}
        parameters = data['parameters'] if 'parameters' in data else None
        return TaskResult(train_info, evaluations, parameters)

    def to_dict(self):
        return {
            'train_info': self.train_info.to_dict(),
            'evaluations': self.evaluations,
            'parameters': self.parameters
        }


class ExperimentResult(UserDict):
    def __init__(self,
                 config: ModelConfig,
                 task: dict[str, TaskResult] = None):
        super().__init__()
        self.config = config
        self.task = {} if task is None else task

    def to_dict(self):
        return {
            'config': self.config.to_dict(),
            'task': {k: v.to_dict() for k, v in self.task.items()}
        }

    @classmethod
    def from_dict(cls, data: dict):
        config = ModelConfig.from_dict(data['config'])
        task = {}
        if 'task' in data and len(data['task']) > 0:
            task = {k: TaskResult.from_dict(v) for k, v in data['task'].items()}
        for t in TASKS.keys():
            if t not in task:
                task[t] = TaskResult()
        return ExperimentResult(config, task)


if __name__ == '__main__':
    # e1 = TaskResult()
    # print(e1)
    m = ModelConfig('roberta-base', 'base')
    print(m)
    # e2 = ExperimentResult(m, {'default': e1})
    # print(e2)
    # print(e2.to_dict())
