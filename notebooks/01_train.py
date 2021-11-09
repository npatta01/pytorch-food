import datetime
import plotly.figure_factory as ff
import wandb
from matplotlib import pyplot as plt
import sklearn
import pytorch_lightning as pl
import flash
import flash.image
from flash.core.data.utils import download_data
import os
import glob
import pandas as pd
import sklearn.model_selection
import numpy as np
import torch
from torchvision import transforms as T
import flash
from flash.core.data.data_source import DefaultDataKeys
from flash.core.data.transforms import ApplyToKeys, merge_transforms
import flash.core.integrations.fiftyone
import flash.core.classification 
import fiftyone as fo
import warnings
import itertools
import timeit
import json
import torchinfo
import utility
import wandb
import joblib

def setup(seed=99):
    pl.seed_everything(seed)
    
    
    

def prepare_dataset(data_dir:str):
    
    df_train = pd.read_parquet(f"{data_dir}/df_train.parquet" )
    df_test = pd.read_parquet(f"{data_dir}/df_test.parquet")
    df_val = pd.read_parquet(f"{data_dir}/df_val.parquet")
    df_eval = pd.read_parquet(f"{data_dir}/df_eval.parquet")
    
    
    train_dataset = utility.create_fiftyone_dataset (df_train,"train")
    val_dataset = utility.create_fiftyone_dataset (df_val,"validation") 
    test_dataset = utility.create_fiftyone_dataset (df_test,"test")
    
    
    new_transforms = utility.image_transforms()
    
    datamodule = flash.image.ImageClassificationData.from_fiftyone(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        train_transform=new_transforms,
    )
    
    return datamodule , df_eval
    

    
    
def time_inference(model,sample_image:str, device='cpu'):
    model = model.eval()

    with torch.no_grad():
        model.to(device)
        predictions = model.predict(sample_image)
    
    
def train_evaluate_timing(model_name:str,epochs:int,unfreeze_epoch:int,df_eval:pd.DataFrame , data_module,num_iterations=10, artifact_dir="artifacts"):
    
    metrics = {}
    
    print (f"Model Spec: {model_name}")
    
    label_encoder = joblib.load(f'{artifact_dir}/label_encoder.joblib')
    labels = label_encoder.classes_
    
    model = utility.create_model(model_name,data_module)
    
    wandb.watch(model)

    
    
    print(model)
    
    print ("Model Summary")
    sample_input_size = (1, 3, 224, 224)
    model_summary = torchinfo.summary(model, input_size=sample_input_size, verbose=0)
    print (model_summary)

    print ("Training Model")
    trainer = flash.Trainer(max_epochs=epochs, gpus=torch.cuda.device_count() )
    model = utility.train_model(model, epochs = epochs, unfreeze_epoch = 5 ,num_iterations=10 )
    
    artifact_model_path = f"{artifact_dir}/model/model_{model_name}.pt"
    trainer.save_checkpoint(artifact_model_path)

    model = flash.image.ImageClassifier.load_from_checkpoint(artifact_model_path)
    
    
    # timing information
    sample_image = df_eval['file_path'].iloc[0]
    time_gpu = timeit.timeit(lambda: time_inference (model,sample_image, device='cuda:0') ,number=num_iterations   ) 
    time_cpu = timeit.timeit(lambda: time_inference (model,sample_image,device='cpu') ,number=num_iterations   ) 
    
    # compute validation metrics
    trainer = flash.Trainer( gpus=torch.cuda.device_count() )
    metrics_test = trainer.test(model, datamodule=datamodule)
    metrics_val =trainer.validate(model, datamodule=datamodule)
    
    eval_accuracy, (y_proba,y_pred,y_true) = utility.evaluate_model(model,df_eval)
    
    wandb.log({"pr": wandb.plot.pr_curve(y_true, y_proba,labels )})
    wandb.log({"roc": wandb.plot.roc_curve(y_true, y_proba, labels)})
    cm = wandb.plot.confusion_matrix(
            y_true=y_true,
            preds=y_proba,
            class_names=labels)
    
    wandb.log({"conf_mat": cm})
    
    df_predictions = df_eval["file_path"]
    df_predictions['y_true'] = label_encoder.inverse_transform( y_true)
    df_predictions['y_pred'] = label_encoder.inverse_transform( y_pred)
    df_predictions['image'] = df_predictions.apply( lambda x:  wandb.Image(x))
    
        
    eval_results_table = wandb.Table(dataframe=df_predictions)
    eval_results_table.add_column(name="image", data=col_data)
    
    wandb.log({"eval_results": eval_results_table})
    
    
    res = {**metrics_test[0], **metrics_val[0]}
    res['time_cpu'] = time_cpu
    res['time_gpu'] = time_gpu
    res['eval_accuracy'] = eval_accuracy
    
    res['params_total'] = model_summary.total_params
    res['params_trainable'] = model_summary.trainable_params
    
    print (res)
    
    wandb.log(res )
    
    return res
    

if __name__ == "__main__":
    seed = 99
    data_dir = "data/food-101"
    
    setup(seed=seed)
    
    data_module, df_eval = prepare_dataset("artifacts/data")
    
    
    max_epochs = 30
    unfreeze_epoch = 10

    model_metrics = {}
    models = ['vgg19','resnet50','mobilenet_v2','mobilenetv3_small_100']
    
    models = models[:1]

    
    for model_name in models:
        
        wandb.init(name=model_name, 
           project='pytorch_food',
           notes=f'model trained with {model_name};    ', 
        )
            
        res = train_evaluate_timing(model_name=model_name, epochs = max_epochs, unfreeze_epoch = unfreeze_epoch, df_eval=df_eval, data_module=data_module)
        
        
        wandb.log(res)
        
        wandb.finish()
        model_metrics[model_name] = res
        
    
    print (model_metrics)
        
        
