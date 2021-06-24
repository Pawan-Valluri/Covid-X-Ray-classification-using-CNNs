# Covid X-Ray Classification using Resnet, Alexnet, VGG16, Squeezenet, Densenet and Invception-V3 in Pytorch 
<p align="middle"><img width="276.5" src="https://media-thumbs.golden.com/OLqzmrmwAzY1P7Sl29k2T9WjJdM=/200x200/smart/golden-storage-production.s3.amazonaws.com/topic_images/e08914afa10a4179893eeb07cb5e4713.png" /></p>

In this repository we'll take a look at the implementation of some of the most used pretrained CNN models on a dataset of X-Rays. 
For the Kaggle dataset, you can follow this [link](https://www.kaggle.com/amanullahasraf/covid19-pneumonia-normal-chest-xray-pa-dataset). For the Kaggle notebook you can follow this [link](https://www.kaggle.com/pawanvalluri/covid-xray-classification-with-multiple-cnn-models).
### I'll also guide through the important parts of the implementation and share useful links here to make this repository beginner friendly 👍🏻. 

#### All the above models are pretrained on imagenet and they perform very well on their own. More informoation of these models in [pytorch documentation](https://pytorch.org/vision/stable/models.html). I'll be taking the pretrained weights on Imagenet for this implementation.

1. But to use these models, they need to be finetuned. Remember that these models are trained on imagenet, and imagenet has 1000 classes. So we need to change the model's final layer's out_features from 1000 to 3 (dataset has 3 classes). 

2. I've included finetuning and initializing of each of models in the function called `initialize_model` in the notebook. This function if given name of the model it'll return finetuned model and the size of the input image for the CNN. With this our model will be ready 🥧 for trainnig. I've searched a lot for easier ways to finetune, and finally I've included the **easiest way that I could find** in the function 👍🏻.  

3. To train these models we need data. I've taken this [dataset](https://www.kaggle.com/amanullahasraf/covid19-pneumonia-normal-chest-xray-pa-dataset). This is a huge data set contains more than 2300 X-Ray images. The dataset has three classes: Covid, Pneumonia, Normal. 

4. Here comes the image augmentation part and the dataloader part. I've used transforms from torchvision for this. There are a ton of transforms available, you can find more about them in [pytorch documentation](https://pytorch.org/vision/stable/transforms.html). Using trasforms the image is randomly rotated or sheared by a few small degrees, randomly flipped, resized to the model's preferable size, and finally converting image to a tensor for the model input. 

5. I've split the dataset for training and validation and created a dataloader with  a batch size of 32. More on the dataloader in [pytorch documentation](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader). I've visualized an output of the dataloader to check the working of the transforms.

6. Finally our data is ready and our model waiting to be trained. I've written a function called `train_model` for the training. This will help us quite a lot in the next step. I've given the info for training in the commented lines of code.

7. Finally it's time to train the models. I've set a loop which goes througth the above mentioned models. 

##### In each iteration:
- The seleted model will be initialised and finetuned.
- Dataloader will be created for the model separately (this is because some models require different image sizes).
- Trained and Validated on the dataset. I've set the epochs to 3.
#### I've trained the model many times and the results vary very little each time. Most of the times the model with highest validation accuracy is Densenet and VGG16 in the second (sometimes vgg did come first as well). The highest accuracy that I could get with these models was capping around 93%.
#### Finally to check things off I've run the models on the whole dataset. I've included instructions in the notebook if you want to run any of trained models(find the saved models/weights [here](https://www.kaggle.com/pawanvalluri/covid-xray-classification-with-multiple-cnn-models/output)).
### Hope this helps 😁 👍🏻.
