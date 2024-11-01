import lightning as L
from pytorch_lightning.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from hyperparameter_tuning.glue_data_module import GLUEDataModule
from hyperparameter_tuning.glue_transformer import GLUETransformer

class Experiment:
    def __init__(self, args):
        self.args = args
        self.wandb_logger = None
        self.dm = None
        self.model = None
        self.checkpoint_callback = None
        self.trainer = None

    def setup_wandb(self):
        run_name = f"{self.args.optimizer}_learningRate{self.args.learning_rate}_{self.args.scheduler}_warmupSteps{self.args.warmup_steps}_weightDecay{self.args.weight_decay}_batchSize{self.args.batch_size}"
        self.wandb_logger = WandbLogger(project=self.args.projectname, name=run_name, log_model=True)
        self.wandb_logger.log_hyperparams(vars(self.args))

    def setup_data_and_model(self):
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
        self.trainer = L.Trainer(
            max_epochs=self.args.epochs,
            accelerator="auto",
            devices=1,
            logger=self.wandb_logger,
            callbacks=[self.checkpoint_callback]
        )

    def run_experiment(self):
        self.setup_wandb()
        self.setup_data_and_model()
        self.setup_trainer()

        self.trainer.fit(self.model, self.dm)
        wandb.finish()
        