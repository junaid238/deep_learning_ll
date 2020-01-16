#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow
import numpy as np 
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')


# In[2]:


import keras


# In[3]:


from keras.datasets import mnist
from keras.preprocessing import image
from keras.models import Sequential
from keras.layers import Dense


# In[4]:


(X_train , y_train),(X_test , y_test) = mnist.load_data()


# In[5]:


print(X_train[0])


# In[6]:


print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


# In[7]:


plt.imshow(X_train[0],cmap="gray")


# In[8]:


X_train = X_train.reshape(60000,28*28)
X_test = X_test.reshape(10000,28*28)


# In[9]:


X_train.astype("float32")
X_test.astype("float32")
X_train = X_train/255
X_test = X_test/255
print(X_train[0])


# In[10]:


from keras.utils import to_categorical 
y_train = to_categorical(y_train,10)
y_test = to_categorical(y_test,10)
print(y_train[0])


# In[11]:


model = Sequential()
model.add(Dense(512 ,activation = "relu", input_shape=(784,) ))
model.add(Dense(512 ,activation = "relu"))
model.add(Dense(10 ,activation = "softmax"))
model.compile(optimizer="adam",loss="categorical_crossentropy",metrics=["accuracy"])


# In[ ]:


model.summary()


# In[ ]:


model_trained = model.fit(X_train,y_train,epochs=10,validation_data=(X_test,y_test))


# In[ ]:


score = model.evaluate(X_test,y_test)
print(score)


# In[ ]:


y_pred = model.predict(X_test)


# In[ ]:


y_pred


# In[ ]:


plt.imshow(X_test[500].reshape(28,28))


# In[ ]:


max(y_pred[500])


# In[ ]:


list(y_pred[500]).index(max(list(y_pred[500])))


# In[ ]:


from keras.layers import Dense,Conv2D,MaxPool2D,Flatten
cnn = Sequential()
cnn.add(Conv2D(32, kernel_size=(5,5),input_shape=(28,28,1),activation = "relu",padding ="same"))
cnn.add(MaxPool2D())
cnn.add(Conv2D(64, kernel_size=(5,5),activation = "relu",padding ="same"))
cnn.add(MaxPool2D())
cnn.add(Flatten())
cnn.add(Dense(512 ,activation = "relu"))
cnn.add(Dense(10 ,activation = "softmax"))


# In[ ]:


cnn.compile(optimizer="adam",loss="categorical_crossentropy",metrics=["accuracy"])


# In[ ]:


from keras.utils import to_categorical 

(X_train , y_train),(X_test , y_test) = mnist.load_data()

X_train = X_train[:30000].reshape(30000,28,28,1)
X_test = X_test[:5000].reshape(5000,28,28,1)

y_train = to_categorical(y_train,10)
y_test = to_categorical(y_test,10)
y_train = y_train[:30000]
y_test = y_test[:5000]


# In[ ]:


cnn.fit(X_train,y_train,epochs=10)


# In[ ]:


cnn.evaluate(X_test,y_test)


# In[ ]:


y_pred = cnn.predict(X_test)


# In[ ]:


print((y_pred))


# In[ ]:


max(y_pred[20])


# In[ ]:


max(y_pred[18])


# In[ ]:


list(y_pred[20]).index(max(list(y_pred[20])))


# In[ ]:


plt.imshow(X_test[20].reshape(28,28))


# In[ ]:


(X_train , y_train),(X_test , y_test) = mnist.load_data()

X_train = X_train.reshape(60000,28,28,1)
X_test = X_test.reshape(10000,28,28,1)

y_train = to_categorical(y_train,10)
y_test = to_categorical(y_test,10)
y_train = y_train
y_test = y_test


# In[ ]:


from keras.layers import Dense,Conv2D,MaxPool2D,Flatten
cnn = Sequential()
cnn.add(Conv2D(32, kernel_size=(5,5),input_shape=(28,28,1),activation = "relu",padding ="same"))
cnn.add(MaxPool2D())
cnn.add(Conv2D(64, kernel_size=(5,5),activation = "relu",padding ="same"))
cnn.add(MaxPool2D())
cnn.add(Flatten())
cnn.add(Dense(512 ,activation = "relu"))
cnn.add(Dense(10 ,activation = "softmax"))


# In[ ]:


cnn.compile(optimizer="adam",loss="categorical_crossentropy",metrics=["accuracy"])


# In[ ]:


# cnn.load_weights("weights/cnn-model5.h5")


# In[ ]:


cnn.fit(X_train,y_train,epochs=10)


# In[ ]:




