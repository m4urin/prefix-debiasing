import hashlib
from src.utils.printing import explore_dict


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
    def from_dict(cls, _data):
        if isinstance(_data, UserDict):
            _data = _data.to_dict()
        return cls(**_data)


class Config(UserDict):
    def __init__(self):
        super().__init__()
        errors = []
        self.verify_config(errors)
        if len(errors) > 0:
            print(self.__dict__)
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
