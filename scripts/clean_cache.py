
from src.utils.config import ModelConfig
from src.utils.files import get_all_files, get_folder

complete_config = {
  "model_name": ["distilbert-base-uncased", "roberta-base"],
  "model_type": ["base", "finetune", "prefix"],
  "prefix_mode": ["identity", "linear", "replace"],
  "prefix_layers": ["all", "half", 3],
  "n_prefix_tokens": [8, 16, 24, 32],
  "prefix_finetune": ["prefix", "model", "all"],
  "objective": ["kaneko", "sst2", "mrpc", "stsb", "rte", "wnli", "wsc"]
}
complete_config = {c.get_filename() for c in ModelConfig.from_hyper_params(complete_config)}

for f in get_all_files('.cache', '.pt'):
    if f.file_name not in complete_config:
        print('remove', f.file_name)
        f.delete()
