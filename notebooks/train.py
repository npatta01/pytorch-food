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
# import flash.core.integrations.fiftyone
import flash.core.classification 
# import fiftyone as fo
import warnings
import itertools
import timeit
import json
import torchinfo
import logging
import argparse
from pytorch_lightning.callbacks import QuantizationAwareTraining



# import wandb
import joblib
import warnings

warnings.filterwarnings("ignore", message="Default upsampling behavior when mode=bilinear")
warnings.filterwarnings("ignore", message="find_unused_parameters=True was specified in DDP constructor")


import utility



def setup(seed=99):
    pl.seed_everything(seed)
    
    
    


    

    
    
def time_inference(model,sample_image:str, device='cpu'):
    model = model.eval()

    with torch.no_grad():
        model.to(device)
        predictions = model.predict(sample_image)
    
    
def train_evaluate_timing(model_name:str,epochs:int,unfreeze_epoch:int,trainer:flash.Trainer, df_eval:pd.DataFrame , data_module:flash.core.data.data_module.DataModule, artifact_model_path:str, num_iterations=10, artifact_dir="artifacts", strategy="ddp"):
    
    metrics = {}
    
    utility.log_main (f"Model Spec: {model_name}")
    
    label_encoder = joblib.load(f'{artifact_dir}/label_encoder.joblib')
    labels = label_encoder.classes_
    
    model = utility.create_model(model_name,data_module)
    
#     wandb.watch(model)


    
    
    utility.log_main(model)
    
    logging.info ("Model Summary")
    sample_input_size = (1, 3, 224, 224)
    model_summary = torchinfo.summary(model, input_size=sample_input_size, verbose=0)
    #print (model_summary)

    utility.log_main ("Training Model")
    
    utility.train_model(model, data_module, trainer=trainer
                        , epochs = epochs, unfreeze_epoch = unfreeze_epoch , artifact_model_path=artifact_model_path )
    
    model = flash.image.ImageClassifier.load_from_checkpoint(artifact_model_path)
    
    
    utility.log_main ("Timing Cpu/GPU")
    # timing information
    sample_image = df_eval['file_path'].iloc[0]
    num_iterations = 10
    time_gpu = timeit.timeit(lambda: time_inference (model,sample_image, device='cuda:0') ,number=num_iterations   ) 
    time_cpu = timeit.timeit(lambda: time_inference (model,sample_image,device='cpu') ,number=num_iterations   ) 
    
    # compute validation metrics
    utility.log_main ("compute validation metrics")
    #trainer = flash.Trainer( gpus=torch.cuda.device_count(),strategy = "ddp") 
    #trainer = flash.Trainer( gpus=[0] )

    metrics_test = trainer.test(model, datamodule=data_module)
    metrics_val =trainer.validate(model, datamodule=data_module)
    
#     eval_accuracy, (y_proba,y_pred,y_true) = utility.evaluate_model(model,df_eval,trainer)
    
#     wandb.log({"pr": wandb.plot.pr_curve(y_true, y_proba,labels )})
#     wandb.log({"roc": wandb.plot.roc_curve(y_true, y_proba, labels)})
#     cm = wandb.plot.confusion_matrix(
#             y_true=y_true,
#             preds=y_proba,
#             class_names=labels)
    
#     wandb.log({"conf_mat": cm})
    
#     df_predictions = df_eval["file_path"]
#     df_predictions['y_true'] = label_encoder.inverse_transform( y_true)
#     df_predictions['y_pred'] = label_encoder.inverse_transform( y_pred)
#     df_predictions['image'] = df_predictions.apply( lambda x:  wandb.Image(x))
    
        
#     eval_results_table = wandb.Table(dataframe=df_predictions)
#     eval_results_table.add_column(name="image", data=col_data)
    
#     wandb.log({"eval_results": eval_results_table})
    
    
    res = {**metrics_test[0], **metrics_val[0]}
    res['inference_time_cpu'] = time_cpu
    res['inference_time_gpu'] = time_gpu
#     res['eval_accuracy'] = eval_accuracy
    
    res['params_total'] = model_summary.total_params
    res['params_trainable'] = model_summary.trainable_params
    
    utility.log_main (res)
    
    
    with open(f"{artifact_dir}/metrics/{model_name}.json","w") as f:
        content = json.dumps(res)
        f.write(content)
    
#     wandb.log(res )
    
    return res
    

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Train model.')
    parser.add_argument('--model', help='model name')
    parser.add_argument('--gpu', help='gpu id to use', default=0, type=int)
#     parser.add_argument('--quantize', help='quantize', action='store_true')


    args = parser.parse_args()
    
    
    logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)

    
    
    
    seed = 99
    data_dir = "data/food-101"
    max_epochs = 20
    unfreeze_epoch = 5
    
    artifact_dir="artifacts"
    artifact_model_path = f"{artifact_dir}/model/model_{args.model}.pt"
    
    setup(seed=seed)
    
#     callbacks=[]
#     if args.quantize:
#         callbacks = [QuantizationAwareTraining()]
#         artifact_model_path = f"{artifact_dir}/model/model_{args.model}__quantized.pt"
        
    print(artifact_model_path)
    logging.info(artifact_model_path)
    
    
    
    data_module, df_eval = utility.prepare_dataset("artifacts/data")
    
    

    
    gpus=torch.cuda.device_count()
#     gpus=[args.gpu]
    


    trainer = flash.Trainer(max_epochs=max_epochs, gpus=gpus
                         , strategy="ddp"
                        ,  num_sanity_val_steps=0
                        #, callbacks = callbacks
                       )

    model_metrics = {}
    models = ['vgg19','resnet50','mobilenet_v2','mobilenetv3_small_100',"mobilenet_v2_quant"]
    
    models = [args.model]
    
#     models = ['mobilenet_v2']
    
#     fo.delete_non_persistent_datasets()

    
    for model_name in models:
        
#         wandb.init(name=model_name, 
#            project='pytorch_food',
#            notes=f'model trained with {model_name};    ', 
#         )
            
        res = train_evaluate_timing(model_name=model_name, epochs = max_epochs, unfreeze_epoch = unfreeze_epoch, df_eval=df_eval, data_module=data_module, trainer=trainer,artifact_model_path=artifact_model_path)
        
        
#         wandb.log(res)
        
#         wandb.finish()
        model_metrics[model_name] = res
    
        
        

        
    utility.log_main(model_metrics)
    
        
        
