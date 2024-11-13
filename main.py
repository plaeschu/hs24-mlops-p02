import argparse
from hyperparameter_tuning.experiment import Experiment

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Hyperparameter Tuning",
        prog="main.py",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--projectname",
        type=str,
        default="glue_hyperparameter_tuning",
        help="Name of the project",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="bert-base-uncased",
        help="Model name or path",
    )
    parser.add_argument("--task_name", type=str, default="mrpc", help="Name of the task")
    parser.add_argument("--max_seq_length", type=int, default=128, help="Maximum sequence length")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay")
    parser.add_argument("--scheduler", type=str, default="linear", help="Scheduler type")
    parser.add_argument("--warmup_steps", type=int, default=0, help="Number of warmup steps")
    parser.add_argument("--optimizer", type=str, default="adamw", help="Optimizer type")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--seed", type=int, default=None, help="Seed for reproducibility")
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="checkpoints",
        help="Directory to save checkpoints",
    )
    args = parser.parse_args()

    experiment = Experiment()
    experiment.run_experiment(args)
