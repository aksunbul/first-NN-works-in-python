
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data.sampler import SubsetRandomSampler
import torch.optim as optimizer
import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt

#read data from csv file
data=pd.read_csv('ctg_v2.csv')
save_data= data
#define X and Y
X = data.iloc[:, :-1].values
Y = data[[data.columns[21]]].values
save_Y = Y
save_X = X

# normalize X
X_min = X.min(axis=0)
X = (X - X_min) / (X.max(axis=0) - X_min)

# make class label in shape of 0-1
data_point_count = Y.shape[0]
b = np.zeros((data_point_count, 3))

for x in range(data_point_count):
  b[x,Y[x]-1] = 1
Y = b

#train test split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25, random_state = 0)

X_train = torch.from_numpy(X_train).float()
X_test = torch.from_numpy(X_test).float()
Y_train = torch.from_numpy(Y_train).float()
Y_test = torch.from_numpy(Y_test).float()


# In[2]:


class Model(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(21, 16)
        self.relu1 = nn.ReLU()
        self.dout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(16, 100)
        self.prelu = nn.PReLU(1)
        self.out = nn.Linear(100, 3)
        self.out_act = nn.Sigmoid()
        
    def forward(self, input_):
        a1 = self.fc1(input_)
        h1 = self.relu1(a1)
        dout = self.dout(h1)
        a2 = self.fc2(dout)
        h2 = self.prelu(a2)
        a3 = self.out(h2)
        y = self.out_act(a3)
        return y
    
model = Model()
# tuning parameters
Optimizer = optimizer.Adam(model.parameters(), lr=0.02, betas=(0.9, 0.999))
criterion = nn.BCELoss()


# In[3]:


def epoch_train(model, Optimizer, criterion, b_size=450):
    model.train()
    losses = []
    for i in range(0, X_train.size(0), b_size):
        x_batch = X_train[i:i + b_size, :]
        y_batch = Y_train[i:i + b_size, :]
        x_batch = Variable(x_batch)
        y_batch = Variable(y_batch)

        Optimizer.zero_grad()
        # Forward
        y_hat = model(x_batch)
        # Compute diff
        loss = criterion(y_hat, y_batch)
        # Compute gradients
        loss.backward()
        # update weights
        Optimizer.step()        
        losses.append(loss.data.numpy())
    return losses


# In[4]:


# creating an array for loss values
losses_arr = []
num_epochs = 25
for e in range(num_epochs):
    losses_arr += epoch_train(model, Optimizer, criterion)
plt.plot(losses_arr)


# In[5]:


# model evaluation and creating predicted arrays
x_t = Variable(X_train)
model.eval()
#print(model(x_t))
x_1_t = Variable(X_test)
Y_pred=model(x_1_t)
#print(Y_pred)


# In[6]:


# comparing predictions and actual values
total=0
for x in range(Y_pred.size()[0]):
  b=np.array_equal(Y_pred.detach().numpy().round()[x],Y_test.detach().numpy().round()[x])
  total +=b

# calculating accuracy
accuracy = total/Y_pred.size()[0]
print("accuracy is: ","{:.2%}".format(accuracy))


# In[7]:


import plotly
plotly.tools.set_credentials_file(username='aksunbul', api_key='1Ovoe7SZ8D4dVyKBy3d7')

import plotly.plotly as py
import plotly.graph_objs as go
from plotly import tools

# creating a pie chart to show the percentages of classes in the dataset
fig = {
    'data': [
        {
            'labels': ['normal','suspect','pathologic'],
            'values': [np.sum(save_Y==1),np.sum(save_Y==2),np.sum(save_Y==3)],
            'type': 'pie'
        }
    ],
    'layout': {'title': 'The Percentages of Classes in the Dataset'}
}

py.iplot(fig, filename='pie_chart_subplots')


# In[8]:


# creating boxplots for all variables in one chart

trace1 = go.Box(x=save_X[:,0], opacity=0.5, name = "LB")
trace2 = go.Box(x=save_X[:,1], opacity=0.5, name = "AC")
trace3 = go.Box(x=save_X[:,2], opacity=0.5, name = "FM")
trace4 = go.Box(x=save_X[:,3], opacity=0.5, name = "UC")
trace5 = go.Box(x=save_X[:,4], opacity=0.5, name = "ASTV")
trace6 = go.Box(x=save_X[:,5], opacity=0.5, name = "MSTV")
trace7 = go.Box(x=save_X[:,6], opacity=0.5, name = "ALTV")
trace8 = go.Box(x=save_X[:,7], opacity=0.5, name = "MLTV")
trace9 = go.Box(x=save_X[:,8], opacity=0.5, name = "DL")
trace10 = go.Box(x=save_X[:,9], opacity=0.5, name = "DS")
trace11 = go.Box(x=save_X[:,10], opacity=0.5, name = "DP")
trace12 = go.Box(x=save_X[:,11], opacity=0.5, name = "WIDTH")
trace13 = go.Box(x=save_X[:,12], opacity=0.5, name = "MIN")
trace14 = go.Box(x=save_X[:,13], opacity=0.5, name = "MAX")
trace15 = go.Box(x=save_X[:,14], opacity=0.5, name = "NMAX")
trace16 = go.Box(x=save_X[:,15], opacity=0.5, name = "NZEROS")
trace17 = go.Box(x=save_X[:,16], opacity=0.5, name = "MODE")
trace18 = go.Box(x=save_X[:,17], opacity=0.5, name = "MEAN")
trace19 = go.Box(x=save_X[:,18], opacity=0.5, name = "MEDIAN")
trace20 = go.Box(x=save_X[:,19], opacity=0.5, name = "VARIANCE")
trace21 = go.Box(x=save_X[:,20], opacity=0.5, name = "TENDENCY")

fig = tools.make_subplots(rows=7, cols=3)
fig.append_trace(trace1, 1, 1)
fig.append_trace(trace2, 1, 2)
fig.append_trace(trace3, 1, 3)
fig.append_trace(trace4, 2, 1)
fig.append_trace(trace5, 2, 2)
fig.append_trace(trace6, 2, 3)
fig.append_trace(trace7, 3, 1)
fig.append_trace(trace8, 3, 2)
fig.append_trace(trace9, 3, 3)
fig.append_trace(trace10, 4, 1)
fig.append_trace(trace11, 4, 2)
fig.append_trace(trace12, 4, 3)
fig.append_trace(trace13, 5, 1)
fig.append_trace(trace14, 5, 2)
fig.append_trace(trace15, 5, 3)
fig.append_trace(trace16, 6, 1)
fig.append_trace(trace17, 6, 2)
fig.append_trace(trace18, 6, 3)
fig.append_trace(trace19, 7, 1)
fig.append_trace(trace20, 7, 2)
fig.append_trace(trace21, 7, 3)

py.iplot(fig, filename='custom binning')


# In[9]:


# comparing two variables with their histograms in one chart
trace1 = go.Histogram(x=save_X[:,4], opacity=0.5, name = "ASTV") #ASTV
trace2 = go.Histogram(x=save_X[:,11], opacity=0.5, name = "WIDTH") #Width

data = [trace1, trace2]

layout = go.Layout(barmode='overlay')
fig = go.Figure(data=data, layout=layout)

py.iplot(fig, filename='overlaid histogram')


# In[10]:


# examining the seperation of two variables on scatter plot in terms of different classes 
save_data1= save_data[(save_data.NSP == 1) ]
save_data2= save_data[(save_data.NSP == 2) ]
save_data3= save_data[(save_data.NSP == 3) ]

# Create a trace
aaa = go.Scatter(
    x = save_data1.iloc[:,4],
    y = save_data1.iloc[:,11],
    mode = 'markers'
)

bbb = go.Scatter(
    x = save_data2.iloc[:,4],
    y = save_data2.iloc[:,11],
    mode = 'markers'
)

ccc = go.Scatter(
    x = save_data3.iloc[:,4],
    y = save_data3.iloc[:,11],
    mode = 'markers'
)

layout= go.Layout(
    title= 'Comparison of ASTV and WIDTH wrt CLASSES',
    hovermode= 'closest',
    xaxis= dict(
        title= 'ASTV',
        ticklen= 5,
        zeroline= False,
        gridwidth= 2,
    ),
    yaxis=dict(
        title= 'WIDTH',
        ticklen= 5,
        gridwidth= 2,
    ),
    showlegend= False
)

data = [aaa,bbb,ccc]

fig= go.Figure(data=data, layout=layout)

py.iplot(fig)


# In[12]:


# understanding the correlation between two variables with a heatmap
trace = go.Heatmap(z=[save_Y[:], save_X[:,20]], opacity=0.75)
data=[trace]

layout = go.Layout(
    title='Tendency - Class Correlation'
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig)

