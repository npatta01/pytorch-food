import pandas as pd
import fiftyone as fo
import flash
import torchvision
import torchinfo
import torch


def create_fiftyone_dataset(df:pd.DataFrame, name="test_df"):
    samples = []
    
    if fo.dataset_exists(name):
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


def create_model(model_name:str, datamodule):
    model = flash.image.ImageClassifier(num_classes=datamodule.num_classes,backbone=model_name)
    return model



def train_model(model, datamodule, epochs = 2, unfreeze_epoch = 5   ):

    trainer = flash.Trainer(max_epochs=epochs, gpus=torch.cuda.device_count() )
    
    trainer.finetune(model, datamodule=datamodule
                 , strategy=flash.core.finetuning.FreezeUnfreeze(unfreeze_epoch=unfreeze_epoch) )
    
    

    return model 
    
    
def evaluate_model(model, df_eval:pd.DataFrame):
    # eval dataset accuracy
    eval_dataset = create_fiftyone_dataset (df_eval,"eval")
    datamodule_eval = flash.image.ImageClassificationData.from_fiftyone(predict_dataset=eval_dataset)
    
    trainer = flash.Trainer( gpus=torch.cuda.device_count() )
    
    predictions = trainer.predict(model, datamodule=datamodule_eval) # datamodule_predict
    predictions = list(itertools.chain.from_iterable(predictions))  # flatten batches
    
    y_proba = predictions
    y_pred = np.argmax(y_proba, axis=1)
    y_true = df_eval['label']

    eval_accuracy = sklearn.metrics.accuracy_score(y_true, y_pred)
    
    return eval_accuracy, (y_proba,y_pred,y_true)