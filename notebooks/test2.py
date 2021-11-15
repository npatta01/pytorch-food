import torch, flash, pandas as pd, utility, flash.image

artifact_model_path ="artifacts/model/model_mobilenet_v2.pt"
model = flash.image.ImageClassifier.load_from_checkpoint(artifact_model_path)

df_eval = pd.read_parquet("artifacts/data/df_eval.parquet")
eval_dataset = utility.create_fiftyone_dataset (df_eval,"eval")
datamodule_eval = flash.image.ImageClassificationData.from_fiftyone(predict_dataset=eval_dataset)

trainer = flash.Trainer( gpus=torch.cuda.device_count() ,strategy = "ddp")

predictions = trainer.predict(model, datamodule=datamodule_eval) # datamodule_predict



