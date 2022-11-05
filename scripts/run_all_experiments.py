from src.experiments import ExperimentTracker

for experiment_name in ['1_find_prefix_mode', '2_find_n_tokens', '3_find_prefix_layers',
                        '4_learning_rate', '5_model_test']:
    ExperimentTracker(experiment_name).run_experiment()
