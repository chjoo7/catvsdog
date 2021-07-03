# Transfer Learning using small cats vs dogs dataset
"""
Classifying images as dogs or cats, in a dataset containing 4,000 pictures of cats and dogs (2,000 cats, 2,000 dogs). We’ll use 2,000 pictures for training—1,000 for validation, and 1,000 for testing.Start by training a small convnet on the 2,000 training samples, without any regularization, to set a baseline. This will get us to a validation classification accuracy of about 76%. By using data augmentation, we’ll improve the model to reach an
accuracy of 85%. To the next step, feature extraction with a pretrained model which will get us to an accuracy of 97.7%. If we augmented the dataset and add the new classifier, it reached an validation accuracy of 96%. When we attempted next tine-fume which was to unfreeze some portion of convolutional base, it reached to the accuracy of 96.5%.
"""

## Environment
"""
This model tested on 4 GPU Ubuntu server. Used Tensorflow Distributed Mirrored Strategy. Tensorflow 2.5
"""

### Downloading the data
"""
You can download the original dataset from www.kaggle.com/c/dogs-vs-cats/data
This dataset contains 25,000 images of dogs and cats (12,500 from each class) and is 543 MB (compressed). After downloading and uncompressing the data, we’ll create a new dataset containing three subsets: a training set with 1,000 samples of each class, a validation set with 500 samples of each class, and a test set with 1,000 samples of each class.
"""

### Data Preprocessing
"""
Calling image_dataset_from_directory(directory) will first list the subdirectories of directory and assume each one contains images from cats and dogs class. It will then index the image files in each subdirectory. Finally, it will create and return a tf.data.Dataset object configured to read these files, shuffle them, decode them to tensors, resize them to a shared size, and pack them into batches.
"""

### Model Save
"""
ModelCheckpoint callback to save the model after each epoch. save_best_only=True and monitor="val_loss" argument enabled checkpoint to only save a new file when the current value of the val_loss metric is lower than at any previous time during training.
"""

### Data Augmentation
"""
By adding a number of data augmentation layers at the start of your model, data augmentation takes the approach of generating more training data from existing training samples and so it help to avoid the overfitting problem. 
"""

### Fine-tuning
"""
Fine-tuning consists of unfreezing a few of top layers of a pretrained model and add a new classifier. In this case, unfree the top 3 layers together with newly added classifier. 
"""

