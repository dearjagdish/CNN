
# coding: utf-8

# In[1]:


from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense


# In[6]:


import os
os.chdir("D:\\Data science\\Deep_Learning_A_Z\\Deep_Learning_A_Z\\Volume 1 - Supervised Deep Learning\\Part 2 - Convolutional Neural Networks (CNN)\\Section 8 - Building a CNN\\Convolutional_Neural_Networks\\dataset")


# In[3]:


#initialising CNN
classifier = Sequential()


# In[4]:


#step -1 convolution
classifier.add(Convolution2D(32, 3, 3, input_shape = (64, 64, 3), activation = 'relu'))


# In[5]:


#step -2  pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))


# In[ ]:


#adding another conovlution layer
classifier.add(Convolution2D(32, 3, 3,  activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))


# In[7]:


#step -3 flattening
classifier.add(Flatten())


# In[9]:


#step -4 full connection
classifier.add(Dense(output_dim = 128, activation = "relu"))
classifier.add(Dense(output_dim = 1, activation = "sigmoid"))


# In[10]:


#compiling CNN
classifier.compile(optimizer = "adam", loss ='binary_crossentropy', metrics = ["accuracy"])


# In[ ]:


#part -2 fitting CNN with images
#copied code from keras documentation website image preprocessing - flow from directory function
###....caution.....takes time to execute......#####..depending on CPU/GPU.....
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
                                                'training_set',
                                                target_size=(64, 64),
                                                batch_size=32,
                                                class_mode='binary')

test_set = test_datagen.flow_from_directory(
                                            'test_set',
                                            target_size=(64, 64),
                                            batch_size=32,
                                            class_mode='binary')

classifier.fit_generator(
                        training_set,
                        steps_per_epoch=8000,
                        epochs=50,
                        validation_data=test_set,
                        validation_steps=2000)


# In[25]:


#home work solution
from keras.preprocessing import image
import numpy as np
test_image = image.load_img("single_prediction/cat_or_dog_2.jpg", target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis =0)
result = classifier.predict(test_image)
training_set.class_indices
if result[0][0] == 1:
    prediction = 'dog'
else:
    prediction = 'cat'


# In[26]:


prediction

