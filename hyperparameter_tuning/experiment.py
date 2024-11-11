import wandb
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from hyperparameter_tuning.glue_data_module import GLUEDataModule
from hyperparameter_tuning.glue_transformer import GLUETransformer


class Experiment:
    """
    Class to set up and run an experiment using PyTorch Lightning and Weights & Biases.
    """

    def setup_data_module(self, config) -> GLUEDataModule:
        """
        Sets up the data module and model for the experiment.
        """
        return GLUEDataModule(
            model_name_or_path=config.model_name_or_path,
            task_name=config.task_name,
            max_seq_length=config.max_seq_length,
            train_batch_size=config.batch_size,
            eval_batch_size=config.batch_size,
        )

    def setup_model(self, config, dm) -> GLUETransformer:
        """
        Sets up the model for the experiment.
        """
        return GLUETransformer(
            model_name_or_path=config.model_name_or_path,
            num_labels=dm.num_labels,
            task_name=dm.task_name,
            learning_rate=config.learning_rate,
            warmup_steps=config.warmup_steps,
            weight_decay=config.weight_decay,
            train_batch_size=config.batch_size,
            eval_batch_size=config.batch_size,
            optimizer=config.optimizer,
            scheduler=config.scheduler,
        )

    def setup_trainer(self, run_name, config, logger) -> L.Trainer:
        """
        Sets up the PyTorch Lightning trainer for the experiment.
        """
        model_checkpoint = ModelCheckpoint(
            dirpath=config.checkpoint_dir,
            filename=f"{config.model_name_or_path}_{run_name}",
            monitor="val_loss",
            mode="min",
            save_top_k=1,
            save_weights_only=True,
        )

        return L.Trainer(
            max_epochs=config.epochs,
            accelerator="auto",
            devices=1,
            logger=logger,
            callbacks=[model_checkpoint],
        )

    def run_experiment(self, config):
        """
        Runs the experiment.
        """
        run_name = (
            f"{config.optimizer}"
            + f"_learningRate{config.learning_rate}_{config.scheduler}"
            + f"_warmupSteps{config.warmup_steps}"
            + f"_weightDecay{config.weight_decay}"
            + f"_batchSize{config.batch_size}"
        )
        with wandb.init(project=config.projectname, name=run_name, config=config):
            config = wandb.config
            data_module = self.setup_data_module(config)
            data_module.setup("fit")
            model = self.setup_model(config, data_module)
            logger = WandbLogger()
            trainer = self.setup_trainer(run_name, config, logger)
            trainer.fit(model, data_module)
            wandb.log_model(path=f"{config.checkpoint_dir}/{config.model_name_or_path}_{run_name}.ckpt", name=run_name)
            wandb.finish()
