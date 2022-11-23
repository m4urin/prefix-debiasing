from src.experiments import ExperimentTracker


for experiment_name in ['1_find_n_tokens', '2_find_prefix_mode', '3_find_prefix_layers', '4_model_test']:
    ExperimentTracker(experiment_name).run_experiment()

