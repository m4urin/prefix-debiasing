from time import time

from src.data.structs.userdict import UserDict
from src.data.structs.model_config import ModelConfig
from src.utils.pytorch import count_parameters


class Result(UserDict):
    def __init__(self, model_config: ModelConfig):
        super().__init__()
        self.model_config = model_config

    @classmethod
    def from_dict(cls, _data: dict):
        if isinstance(_data['model_config'], dict):
            _data['model_config'] = ModelConfig.from_dict(_data['model_config'])
        return cls(**_data)


class TrainResult(Result):
    def __init__(self,
                 model_config: ModelConfig,
                 training_completed: bool = False,
                 total_training_time: int = -1,
                 n_parameters: int = 0,
                 model=None,
                 scalars: dict = None
                 ):
        super().__init__(model_config)
        self.training_completed: bool = training_completed
        self.total_training_time: int = total_training_time
        self.n_parameters = n_parameters
        self.model = model
        self.scalars: dict = scalars if scalars is not None else {}

    def get_filename(self):
        return f'{self.model_config.get_filename()}.pt'

    def add_scalar(self, name: str, scalar: float):
        names = name.split('/')
        sub = self.scalars
        for n in names[:-1]:
            if n not in sub:
                sub[n] = {}
            sub = sub[n]
        if names[-1] not in sub:
            sub[names[-1]] = []
        if not isinstance(sub[names[-1]], list):
            raise ValueError(f"Cannot append scalar to '{name}', '{names[-1]}' is a '{type(sub[names[-1]])}'.")
        sub[names[-1]].append(scalar)

    def finish_training(self, start_time, model):
        self.model = model
        self.n_parameters = count_parameters(list(model.get_parameters_to_train()))
        self.total_training_time = int(round(time() - start_time))
        self.training_completed = True
        return self

    def has_parameters(self):
        return self.model is not None


class EvaluationResult(Result):
    def __init__(self,
                 model_config: ModelConfig,
                 evaluations: dict = None,
                 ):
        super().__init__(model_config)
        self.evaluations = evaluations if evaluations is not None else {}

    def get_filename(self):
        return f'{self.model_config.get_filename()}.json'
