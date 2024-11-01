import argparse
from hyperparameter_tuning.experiment import Experiment


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hyperparameter Tuning")
    parser.add_argument("--projectname", type=str, default="glue_hyperparameter_tuning")
    parser.add_argument("--model_name_or_path", type=str, default="bert-base-uncased")
    parser.add_argument("--task_name", type=str, default="mrpc")
    parser.add_argument("--max_seq_length", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--scheduler", type=str, default="linear")
    parser.add_argument("--warmup_steps", type=int, default=0)
    parser.add_argument("--optimizer", type=str, default="adamw")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    args = parser.parse_args()

    experiment = Experiment(args)
    experiment.run_experiment()
