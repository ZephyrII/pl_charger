import pytorch_lightning as pl
from pytorch_lightning.loggers import NeptuneLogger
from charger_kpts import ChargerKpts
from charger_kpts_hrnet import ChargerKptsHrnet

if __name__ == '__main__':
    neptune_logger = NeptuneLogger(
    api_key="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI4NGRmMmFkNi0wMWNjLTQxY2EtYjQ1OS01YjQ0YzRkYmFlNGIifQ==",  # replace with your own
    project_name="tnowak/charger",
    experiment_name="resnet50_9l",
    # offline_mode=True
)
    ch_call = pl.callbacks.ModelCheckpoint(save_last=True, dirpath="./checkpoints/resnet50_9l", every_n_epochs=1, save_top_k=-1)
    trainer = pl.Trainer(gpus=1, checkpoint_callback=True, callbacks=[ch_call], accumulate_grad_batches=2, logger=neptune_logger)#, resume_from_checkpoint="./checkpoints/resnet50_9l/epoch=2-step=14999.ckpt") #, limit_train_batches=0.001, limit_val_batches=0.5)
    model = ChargerKpts("/root/share/tf/dataset/final_localization/tape_1.0/")
    # model = ChargerKptsHrnet("/root/share/tf/dataset/final_localization/tape_1.0/")
    trainer.fit(model)
# limit_train_batches=300, 
