import pandas as pd
import flash
import torchvision
import torchinfo
import torch
import flash.image
# import fiftyone as fo
import itertools
import sklearn
import numpy as np
import sys
import flash
import os
import logging


@flash.image.ImageClassifier.backbones(name="mobilenet_v2_quant")
def fn_mobilenet_v2_quant(pretrained: bool = True):
    model = torchvision.models.quantization.mobilenet.mobilenet_v2(pretrained=True)    

    # remove the last two layers & turn it into a Sequential model
    # backbone = torch.nn.Sequential(*list(model.children())[:-2])
    
    backbone = model.features
    num_features = model.classifier[-1].in_features
    # backbones need to return the num_features to build the head
    return backbone, num_features


# https://stackoverflow.com/questions/66261729/pytorch-lightning-duplicates-main-script-in-ddp-mode
def shutdown_process(model):
    if model.global_rank != 0:
        sys.exit(0)
        
        
def log_main(msg:str):
    rank = int(os.getenv('RANK', 0))
    
    if rank:
        logger.info(msg)


def create_fiftyone_dataset(df:pd.DataFrame, name="test_df"):

    samples = []
    
    
    fo.delete_dataset(name)
    for record in df.to_dict(orient='records'):
        samples.append(
            fo.Sample(
                filepath=record['file_path'],
                ground_truth=fo.Classification(label=record['label_name']),
            )
        )
        

    dataset = fo.Dataset(name)
    dataset.add_samples(samples)
    return dataset


def image_transforms():
    default_transforms = flash.image.classification.transforms.default_transforms((224, 224))
    
    post_tensor_transform = flash.core.data.transforms.ApplyToKeys(
        flash.core.data.data_source.DefaultDataKeys.INPUT,
        torchvision.transforms.Compose([torchvision.transforms.RandomHorizontalFlip()
                                        , torchvision.transforms.ColorJitter()
                                        , torchvision.transforms.RandomAutocontrast()
                                        , torchvision.transforms.RandomPerspective()] 
                                      ),
    )

    new_transforms = flash.core.data.transforms.merge_transforms(default_transforms, {"post_tensor_transform": post_tensor_transform})

    
    return new_transforms


def prepare_dataset(data_dir:str)->flash.core.data.data_module.DataModule :
    
    new_transforms = image_transforms()

        
    df_train = pd.read_parquet(f"{data_dir}/df_train.parquet" )
    df_test = pd.read_parquet(f"{data_dir}/df_test.parquet")
    df_val = pd.read_parquet(f"{data_dir}/df_val.parquet")
    df_eval = pd.read_parquet(f"{data_dir}/df_eval.parquet")
    
    
#     train_dataset = utility.create_fiftyone_dataset (df_train,"train")
#     val_dataset = utility.create_fiftyone_dataset (df_val,"validation") 
#     test_dataset = utility.create_fiftyone_dataset (df_test,"test")
    
    
    
#     datamodule = flash.image.ImageClassificationData.from_fiftyone(
#         train_dataset=train_dataset,
#         val_dataset=val_dataset,
#         test_dataset=test_dataset,
#         train_transform=new_transforms,
#     )

    datamodule = flash.image.ImageClassificationData.from_data_frame(
        "file_path",
        "label",
        train_data_frame=df_train,
        val_data_frame=df_val,
        test_data_frame=df_test,
        train_transform=new_transforms,
        batch_size=64,
        num_workers=16
    )

    
    return datamodule , df_eval

def create_model(model_name:str, datamodule:flash.core.data.data_module.DataModule):
    model = flash.image.ImageClassifier(num_classes=datamodule.num_classes,backbone=model_name)
    return model



def train_model(model, datamodule:flash.core.data.data_module.DataModule, artifact_model_path:str, trainer:flash.Trainer, epochs = 2, unfreeze_epoch = 5 , strategy="ddp"  ):

#     if trainer is None:
#         trainer = flash.Trainer(max_epochs=epochs, gpus=torch.cuda.device_count()
#                             , strategy=strategy
#                             ,  num_sanity_val_steps=0

#                            )
        
    
    
    trainer.finetune(model, datamodule=datamodule
                 , strategy=flash.core.finetuning.FreezeUnfreeze(unfreeze_epoch=unfreeze_epoch) )
    
    
    trainer.save_checkpoint(artifact_model_path)
    
    #shutdown_process(model)

    
    
def evaluate_model(model, df_eval:pd.DataFrame,trainer:flash.Trainer):
    # eval dataset accuracy
#     eval_dataset = create_fiftyone_dataset (df_eval,"eval")
#     datamodule_eval = flash.image.ImageClassificationData.from_fiftyone(predict_dataset=eval_dataset)
    
    datamodule_eval = flash.image.ImageClassificationData.from_data_frame(
        "file_path",
        "label",
        predict_data_frame=df_eval,
        batch_size=64,
        num_workers=16
    )
    
    
    model.serializer = flash.core.classification.Probabilities()
    
    #trainer = flash.Trainer( gpus=torch.cuda.device_count(),strategy = "ddp") 
#     if trainer is None:
#         trainer = flash.Trainer( gpus=1) 


    trainer = flash.Trainer( gpus=1,  num_sanity_val_steps=0) 
    
    predictions = trainer.predict(model, datamodule=datamodule_eval) # datamodule_predict
    predictions = list(itertools.chain.from_iterable(predictions))  # flatten batches
    
    y_proba = predictions
    y_pred = np.argmax(y_proba, axis=1)
    y_true = list(df_eval['label'] )
    
    
    #shutdown_process(model)

    

    eval_accuracy = sklearn.metrics.accuracy_score(y_true, y_pred)
    
    return eval_accuracy, (y_proba,y_pred,y_true)