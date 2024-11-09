import wandb
import lightning as L
from pytorch_lightning.loggers import WandbLogger
from hyperparameter_tuning.glue_data_module import GLUEDataModule
from hyperparameter_tuning.glue_transformer import GLUETransformer


class Experiment:
    """
    Class to set up and run an experiment using PyTorch Lightning and Weights & Biases.
    """

    def __init__(self, args):
        """
        Initializes the Experiment class with the given arguments.

        Args:
            args (Namespace): The arguments for the experiment.
        """
        self.args = args
        self.wandb_logger = None
        self.dm = None
        self.model = None
        self.checkpoint_callback = None
        self.trainer = None

    def setup_wandb(self):
        """
        Sets up Weights & Biases (wandb) for logging the experiment.
        """
        run_name = f"{self.args.optimizer}_learningRate{self.args.learning_rate}_{self.args.scheduler}_warmupSteps{self.args.warmup_steps}_weightDecay{self.args.weight_decay}_batchSize{self.args.batch_size}"
        self.wandb_logger = WandbLogger(project=self.args.projectname, name=run_name, log_model=True)
        self.wandb_logger.log_hyperparams(vars(self.args))

    def setup_data_and_model(self):
        """
        Sets up the data module and model for the experiment.
        """
        self.dm = GLUEDataModule(
            model_name_or_path=self.args.model_name_or_path,
            task_name=self.args.task_name,
            max_seq_length=self.args.max_seq_length,
            train_batch_size=self.args.batch_size,
            eval_batch_size=self.args.batch_size,
        )
        self.dm.setup("fit")

        self.model = GLUETransformer(
            model_name_or_path=self.args.model_name_or_path,
            num_labels=self.dm.num_labels,
            task_name=self.dm.task_name,
            learning_rate=self.args.learning_rate,
            warmup_steps=self.args.warmup_steps,
            weight_decay=self.args.weight_decay,
            train_batch_size=self.args.batch_size,
            eval_batch_size=self.args.batch_size,
            optimizer=self.args.optimizer,
            scheduler=self.args.scheduler,
        )

    def setup_trainer(self):
        """
        Sets up the PyTorch Lightning trainer for the experiment.
        """
        self.trainer = L.Trainer(
            max_epochs=self.args.epochs,
            accelerator="auto",
            devices=1,
            logger=self.wandb_logger,
            callbacks=[self.checkpoint_callback],
        )

    def run_experiment(self):
        """
        Runs the experiment.
        """
        self.setup_wandb()
        self.setup_data_and_model()
        self.setup_trainer()
        self.trainer.fit(self.model, self.dm)
        wandb.finish()
