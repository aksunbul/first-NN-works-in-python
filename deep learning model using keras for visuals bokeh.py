
# coding: utf-8

# In[1]:


import keras
from keras.models import Sequential
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.layers.core import Dense, Activation
import numpy as np
import pandas as pd


# In[2]:


from bokeh.io import output_notebook, show
from bokeh.plotting import figure, show, output_file
output_notebook()


# In[3]:


# reading data from train and test datasets
data_train=pd.read_csv('data/ann-train_data.csv', header=None)

(x_train, y_train) = data_train[data_train.columns[:21]], pd.get_dummies(data_train[21])

data_test=pd.read_csv('data/ann-test_data.csv', header=None)
(x_test, y_test) = data_test[data_test.columns[:21]], pd.get_dummies(data_test[21])


# In[5]:


# creating keras model
model = Sequential()
model.add(Dense(20, input_dim=x_train.shape[1], activation='relu'))
model.add(Dense(10))
model.add(Dense(y_train.shape[1],activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
monitor = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=5, verbose=0, mode='auto')
checkpointer = ModelCheckpoint(filepath="best_weights.hdf5", verbose=0, save_best_only=True) # save best model
history = model.fit(x_train,y_train,validation_data=(x_test,y_test),callbacks=[monitor,checkpointer],verbose=0,epochs=1000)


# In[6]:


# create a new plot (with a title) using figure
p = figure(plot_width=400, plot_height=400, title="Comparison of Losses (in blue) and Validation Losses (in red)")

# add a line renderer
p.multi_line([list(range(len(history.history["loss"]))), list(range(len(history.history["val_loss"])))], 
             [history.history["loss"], history.history["val_loss"]], color=["firebrick", "navy"], line_width=2)

show(p) # show the results


# In[7]:


from sklearn.metrics import confusion_matrix

pred = model.predict(x_test)
pred = np.argmax(pred,axis=1)
y_test2 = np.argmax(y_test.values,axis=1)

# Compute confusion matrix
cm = confusion_matrix(y_test2, pred)
np.set_printoptions(precision=2)
print('Confusion matrix, without normalization')
#print(cm)

# confusion matrix in table
aa = pd.DataFrame(
    [cm[0], cm[1],cm[2]],
    columns=["normal ","hyperfunction","subnormal functioning"],
    index=["normal ","hyperfunction","subnormal functioning"])
aa.index.name = 'real'
aa.columns.name = 'predictions'

aa


# In[8]:


# calculating accuracy rate
accuracy = np.sum(y_test2 == pred)/y_test2.size
print("accuracy is: ","{:.2%}".format(accuracy))


# In[9]:


from bokeh.io import show, output_file
from bokeh.models import ColumnDataSource
from bokeh.palettes import Spectral6
from bokeh.plotting import figure
import matplotlib.pyplot as plt


# In[10]:


# a graph for visualising class counts
classes = ["normal","hyperfunction","subnormal functioning"]
counts = [np.sum(y_train[1]),np.sum(y_train[2]),np.sum(y_train[3])]

source = ColumnDataSource(data=dict(classes=classes, counts=counts, color=Spectral6))

p = figure(x_range=classes, y_range=(0,np.sum(y_train[3])), plot_height=250, title="Class Counts",
           toolbar_location=None, tools="")

p.vbar(x='classes', top='counts', width=0.9, color='color', legend="classes", source=source)

p.xgrid.grid_line_color = None
p.legend.orientation = "horizontal"
p.legend.location = "top_left"

show(p)


# In[12]:


# creating confusion matrix to compare predicted and real results
def plot_confusion_matrix(cm, names, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(names))
    plt.xticks(tick_marks, names, rotation=45)
    plt.yticks(tick_marks, names)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
plt.figure()
plot_confusion_matrix(aa, ["normal ","hyperfunction","subnormal functioning"], title='Confusion matrix')

plt.show()


# In[13]:


# creating a plot fora age and class examination
vect=np.arange(len(y_train))
vect[y_train[1]==1]=1
vect[y_train[2]==1]=2
vect[y_train[3]==1]=3

plot = figure(plot_width=400, plot_height=400, tools="tap", title="Age and class examination")
renderer = plot.circle(x_train[0]*100, vect, size=5)

show(plot)


# In[14]:


# creating scatter plot for examining outliers on the graph
def mscatter(p, x, y, marker):
    p.scatter(x, y, marker=marker, size=10,
              line_color="navy", fill_color="red", alpha=1)

p = figure(title="Outlier detection by comparing FTI measure and age", toolbar_location=None)
mscatter(p,x_train[0]*100, x_train[19]*100, "circle")

show(p)  

