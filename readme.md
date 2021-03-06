# README



## About

This repo is a tutorial/sample code for :
1. How to train a computer vision classifier using PyTorch
2. Serve the model using PyTorch serving
3. Deploy the model using Torch Mobile for Android
4. Deploy the model using Torch Mobile for IOS



## Dataset

This project uses the [Food-101](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/] dataset.



## Demo 


### Android 
![android](assets/android.png)


### iOS

![ios](assets/ios.png)



## References

The android and ios code was forked from the very helpful examples provided by the Pytorch Team.
[IOS](https://github.com/pytorch/ios-demo-app)
[Android](https://github.com/pytorch/android-demo-app)



## Notes

```
cd notebooks
papermill -p model resnet34 01_training.ipynb 01_training__output.ipynb -k python3

```