# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 23:03:26 2019

@author: Andrei
"""


import pandas as pd

from keras.callbacks import ModelCheckpoint
import csv
#import processCSV as p
import tensorflow as tf
import numpy as np
from imp import reload
import rdkit.Chem.rdChemReactions as cR
from rdkit import Chem
import rdkit.DataStructs.cDataStructs as cS
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
import sklearn.neural_network as nn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn import datasets, linear_model
from sklearn.model_selection import cross_val_score, KFold
#from keras.models import Sequential
from sklearn.metrics import accuracy_score
#from keras.layers import Dense, Conv2D, Flatten,MaxPooling2D
#from keras.layers import Dense
#import  keras
#from keras.wrappers.scikit_learn import KerasRegressor
#from keras.layers import Input,Dropout,Embedding
#from keras.models import Model
#from keras.wrappers.scikit_learn import KerasRegressor
import pickle
from rdkit.Chem.Fingerprints import FingerprintMols
from sklearn.preprocessing import StandardScaler,MinMaxScaler
import pydot
from tensorflow.keras.layers import Input, Dropout,Embedding,Dense, Conv2D, Flatten,MaxPooling2D
from tensorflow.keras.layers import Add,Concatenate,Conv1D,Reshape,average,maximum,multiply,BatchNormalization
from tensorflow.keras.models import Model,Sequential
from tensorflow.keras.optimizers import Adam,Adagrad
from tensorflow.keras.initializers import RandomNormal

from tensorflow.keras import layers
#from tensorflow.keras.layers.normalization import BatchNormalization
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.utils import plot_model
from keras.regularizers import l1,l2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from superkeras import permutational_layer as pl
from matplotlib import pyplot
from scipy.stats import pearsonr
import utils
#from NGF.preprocessing import tensorise_smiles, tensorise_smiles_mp
#from NGF.layers import NeuralGraphHidden, NeuralGraphOutput
#from NGF.models import build_graph_conv_model
#from NGF.sparse import GraphTensor, EpochIterator

from deepchem.models.graph_models import GraphConvModel
from deepchem.feat  import ConvMolFeaturizer


#from tensorflow.keras.models import Sequential
import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'

seed = 1

#reload(p)
suffix="\\"
#file="FinalCleaned.csv"
file="WeiReactionsAndAllFps.pkl"
filename="FinalAllcsv.csv"

featurizer=ConvMolFeaturizer()

#pCSV = p.processCSV(filename)



#readOriginalDataSetAndCreatePickledSmilesOnly
#pCSV.readPickleAndTranslateToMols()


#pCSV = p.processCSV(filename)
#pCSV = p.processCSV(file)
#pCSV.readOriginalDataSetAndCreatePickledSmilesOnly(file)
#df=pickle.load(open("WeiReactionsAndAllFps.pkl",'rb'))
df=pd.read_pickle("./WeiReactionsNeuralFp.pkl")


fpSize=767
FpLen=256
FpLen1=256
#pCSV.readPickleAndTranslateToMols(fpSize,1)


maxL=3



#for i in range(len(pCSV.Ws)):
#    pCSV.Ws[i]=pCSV.Ws[i]+[0.]*(maxL-len(pCSV.Ws[i]))
##################################### FC newtork ###########################
#X_train, X_test, y_train, y_test = train_test_split(pCSV.df, test_size=0.2, random_state=0)
df_train, df_test = train_test_split(df, test_size=0.8, random_state=0)


#scaler = StandardScaler()
#scaler = MinMaxScaler()


#ff0=df_train['Morgan'].to_numpy()
#ff00=df_test['Morgan'].to_numpy()

#ff0=df_train['Seq2Seq'].to_numpy()
#ff00=df_test['Seq2Seq'].to_numpy()


ff0=df_train['NeuralFp'].to_numpy()
ff00=df_test['NeuralFp'].to_numpy()


X_train=np.asarray([list(i) for i in ff0])
X_test=np.asarray([list(i) for i in ff00])



#X_train= df_train['Morgan']
#X_test=df_test['Morgan']

#X_train=np.asarray([[j for j in i] for i in df_train['fp']])





#X_train=np.asarray([[j for j in i] for i in df_train['fp']])

#X_test=np.asarray([[j for j in i] for i in df_test['fp']])


#y1_train=np.asarray([[j for j in i] for i in df_train['Ws']])
y_train=np.asarray([i for i in df_train['Target']])
y_test=np.asarray([i for i in df_test['Target']])
#y_train=np.asarray(df_train['Target'])
#y_test=np.asarray(df_test['Target'])

#y_train=np.asarray([float(i) for i in df_train['Loss']])
#y_test=np.asarray([float(i) for i in df_test['Loss']])

#aa=df_train['Ws'].to_numpy()
#aa1=df_test['Ws'].to_numpy()

#ff=df_train['Seq2Seq'].to_numpy()
#ff1=df_test['Seq2Seq'].to_numpy()


#fp_train=df_train['Mol2Vec']
#fp_test=df_test['Mol2Vec']
#fp_train=np.asarray([list(i) for i in ff])
#fp_test=np.asarray([list(i) for i in ff1])


#ff_1=df_train['Mol2Vec'].to_numpy()
#ff1_1=df_test['Mol2Vec'].to_numpy()


#fp1_train=np.asarray([list(i) for i in ff_1])
#fp1_test=np.asarray([list(i) for i in ff1_1])




#ff1_1=df_test['Mol2Vec'].to_numpy()

#ff=df_train['fpss'].to_numpy()
#ff1=df_test['fpss'].to_numpy()


#fp_train=np.asarray([list(i)+[[0. for i in range(len(i[0]))]]*(maxL-len(i)) for i in ff])
#fp_test=np.asarray([list(i)+[[0. for i in range(len(i[0]))]]*(maxL-len(i)) for i in ff1])


#fp1_train=np.asarray([list(i)+[[0. for i in range(len(i[0]))]]*(maxL-len(i)) for i in ff_1])
#fp1_test=np.asarray([list(i)+[[0. for i in range(len(i[0]))]]*(maxL-len(i)) for i in ff1_1])


#y1_train=np.asarray([list(i)+[0.]*(maxL-len(i)) for i in aa])
#y2_train=np.asarray(df_train['pH'].to_numpy())
#y3_train=np.asarray(df_train['InCnc'].to_numpy())

#trainMask=np.asarray(y1_train!=0).astype(float)
#y1_test=np.asarray([list(i)+[0.]*(maxL-len(i)) for i in aa1])
#y2_test=np.asarray(df_test['pH'].to_numpy())
#y3_test=np.asarray(df_test['InCnc'].to_numpy())







#scaler = StandardScaler()
#scaler1 = scaler.fit(y1_train)
#y1_train = scaler.transform(y1_train)
#y1_test = scaler.transform(y1_test)
#sc = StandardScaler()
#sc = scaler.fit(y2_train)
#y2_train = sc.transform(y2_train)
#y2_test = sc.transform(y2_test)
my_init=glorot_uniform(seed=42)



def MT_model(perm):
 
    #def funx1(i,maxL):

     #if i<maxL:
         #return Input(shape=(fpSize,),name=f"MRg_{i}")
     #elif i<2*maxL-3 and i!=maxL+2:
     
    # elif maxL<=i< 2*maxL:
         #return Input(shape=(FpLen,),name=f"Ml_{i}")
     #else:   
     #    return Input(shape=(FpLen1,),name=f"Sq_{i}")
    
    def funx1(i,maxL):

     if i<maxL:
         return Input(shape=(156,),name=f"MoR_{i}")
    



    
    
    
    x2 = [funx1(i,maxL) for i in range(1*maxL)]
    
    
    x2_1=Concatenate()([x2[perm[0]],x2[perm[1]],x2[perm[2]]])
    #x2_1=Concatenate()([x2[2],x2[0],x2[1]])

    #x2_1=Concatenate()([x2[1],x2[0],x2[2]])
    #x2_1=Concatenate()([x2[1],x2[2],x2[0]])
    
    #x2_1=Concatenate()([x2[2],x2[1],x2[0]])
    #x2_1=Concatenate()([x2[2],x2[0],x2[1]])
    
    print("x2",x2)
    #output_51 = Dense(100, activation='relu',kernel_initializer=my_init)(x2_1)
    #output_51 = Dense(100, activation='relu',kernel_initializer=my_init)(x2_1)
    #x2_1 = Dense(100, activation='relu',kernel_initializer=my_init)(x2_1)
    #x2_1 = Dense(58, activation='relu',kernel_initializer=my_init)(x2_1)
    #x2_1 = Dense(58, activation='relu',kernel_initializer=my_init)(x2_1)
    #x2_1 = Dense(58, activation='relu',kernel_initializer=my_init)(x2_1)
    #x2_1 = Dense(58, activation='relu',kernel_initializer=my_init)(x2_1)
    #x2_1 = Dense(112, activation='relu',kernel_initializer=my_init)(x2_1)
    output_51 = Dense(100, activation='relu',kernel_initializer=my_init)(x2_1)
    #output_51 = Dense(50, activation='relu',kernel_initializer=my_init)(x2_1)

    
    
    output_Loss=Dense(17,name='Loss_output',activation='softmax',kernel_initializer=my_init)(output_51)
#    output_Loss=Dense(17,name='Loss_output',activation='linear',kernel_initializer=my_init)(output_51)
    #output_Loss=Dense(17,name='Loss_output',activation='linear',kernel_initializer=my_init)(output_51)


    model=Model(inputs=x2,outputs=output_Loss)
    

#    model.compile(loss={'Loss_output':'categorical_crossentropy'},
#              optimizer=Adam(lr=0.00005, beta_1=0.9, beta_2=0.999, epsilon=1e-8),  loss_weights = {'Loss_output':1.}
#              ,metrics=['accuracy']
#              )

    model.compile(loss={'Loss_output':'categorical_crossentropy'},
              #optimizer=Adam(lr=0.00005, beta_1=0.9, beta_2=0.999, epsilon=1e-8),  loss_weights = {'Loss_output':1.}
               optimizer=Adam(lr=0.00276, beta_1=0.9, beta_2=0.999, epsilon=1e-8),  loss_weights = {'Loss_output':1.}
                  
              ,metrics=['accuracy']
              )


    model.summary()
    plot_model(model, 'Config3Mod.png', show_shapes=True)
    
    return model




#model, model1=MT_model()
#model=cnv_model()    


#results = cross_val_score(estimator, np.asarray(pCSV.features), pCSV.labels, cv=kfold)
#print("Results: %.2f (%.2f) MSE" % (results.mean(), results.std()))
#model.fit(X_train, y_train, epochs=1220,batch_size=20,validation_data=(X_test, y_test))
#model.fit({'Fp':X_train['fp']},{'Loss_output':X_train['Loss'],'Ws_output':X_train['Ws']}, epochs=1220,batch_size=20,validation_data=(X_test, y_test))


#model.fit({'Fp':X_train},{'Loss_output':y_train,'Ws_output':y1_train}, epochs=1220,batch_size=16, 
#          validation_data=({'Fp':X_test},{'Loss_output':y_test,'Ws_output':y1_test}))
#model.fit({'Fp':X_train,'WsI':y1_train},{'Loss_output':y_train,'Ws_output':y1_train}, epochs=1220,batch_size=20, 
#          validation_data=({'Fp':X_test},{'Loss_output':y_test,'Ws_output':y1_test}))


#model.fit({'Fp':X_train,'WsI':y1_train},{'Loss_output':y_train}, epochs=1220,batch_size=20, 
#          validation_data=({'Fp':X_test,'WsI':y1_test},{'Loss_output':y_test}),callbacks=[es])




#

#model.fit([y1_train[:,i] if i<maxL else X_train for i in range(maxL+1)],y_train, epochs=1220,batch_size=20, 
#          validation_data=([y1_test[:,i] if i<maxL else X_test for i in range(maxL+1)],y_test),callbacks=[es])

#def funx(i,maxL,Test=False):
#   if Test: 
#    if(i<maxL):
#        #return (y1_test[:,i],fp_test[:,i])
#        return y1_test[:,i]
#    elif i==maxL:
#        return y2_test
#    elif i==maxL+1:
#        return y3_test
#    elif i==maxL+2:
#        return X_test
#    else:
#       return fp_test[:,i-maxL-3]
# 
#   else:    
#       
#    if(i<maxL):
#        #return (y1_train[:,i],fp_train[:,i])
#        return y1_train[:,i]
#    elif i==maxL :
#        return y2_train
#    elif i==maxL+1:
#        return y3_train
#    elif i==maxL+2 :
#        return X_train
#    else:
#      return  fp_train[:,i-maxL-3]
#
#
#
#def funx1(i,maxL,Test=False):
#   if Test: 
#    if(i<maxL):
#        #return (y1_test[:,i],fp_test[:,i])
#        return y1_test[:,i]
#    elif i==maxL:
#        return y2_test
#    elif i==maxL+1:
#        return y3_test
#    elif i==maxL+2:
#        return X_test
#    else:
#       return fp_test[:,i-maxL-3]
# 
#   else:    
#       
#    if(i<maxL):
#        #return (y1_train[:,i],fp_train[:,i])
#        return y1_train[:,i]
#    elif i==maxL :
#        return y2_train
#    elif i==maxL+1:
#        return y3_train
#    elif i==maxL+2 :
#        return X_train
#    else:
#      return  fp_train[:,i-maxL-3]


def funx(i,maxL,Test=False):
       if Test: 
        if(i<maxL):

            return X_test[:,i]

            
       else:    
           
        if(i<maxL):
           return X_train[:,i]
            
       
def funxPerm(i,maxL,perm,Test=False):
       if Test: 
        if(i<maxL):

            return X_test[:,perm[i]]

            
       else:    
           
        if(i<maxL):
           return X_train[:,perm[i]]
        

def accuracy(preds, targs):
    isMaxPred = [[val == max(row) for val in row] for row in preds]
    isMaxTarg = [[val == max(row) for val in row] for row in targs]
    return float(sum([isMaxPred[ii] == isMaxTarg[ii] for ii in range(len(preds))]))/len(preds)




dP=0.1

def accuracy1(preds, targs):
    isMaxPred = [[max(row)-dP<= val <= max(row)+dP for val in row] for row in preds]
    isMaxTarg = [[val == max(row) for val in row] for row in targs]
    return float(sum([isMaxPred[ii] == isMaxTarg[ii] for ii in range(len(preds))]))/len(preds)


#model.fit([funx(i,maxL) for i in range(2*maxL)],y_train, epochs=1220,batch_size=20, 
#           validation_data=([funx(i,maxL,Test=True) for i in range(2*maxL)],y_test),
#           callbacks=[es,reduce_lr])

filepath="weights.best.hdf5"


accTest=[[],[],[],[],[],[]]
accTrain=[[],[],[],[],[],[]]
accTest1=[[],[],[],[],[],[]]
accTrain1=[[],[],[],[],[],[]]

permutations=[[0,1,2],[0,2,1],[1,0,2],[1,2,0],[2,1,0],[2,0,1]]

#for counter, value in enumerate(my_list):
for cnt1, pr in enumerate(permutations):
  #accTest
         
   for i in range(1):
#    model=MT_model(pr)
    model=MT_model(permutations[0])
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1,patience=150)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.4,
                              patience=5, min_lr=0.00001,verbose=1)

    for i in range(1):

 
#    model.fit([funx(i,maxL) for i in range(maxL)],y_train, epochs=1220,batch_size=100, 
#           validation_data=([funx(i,maxL,Test=True) for i in range(maxL)],y_test),
#           callbacks=[es,reduce_lr])

        #model.fit([funx(i,maxL) for i in range(maxL)],y_train, epochs=50,batch_size=100, 
        model.fit([funxPerm(i,maxL,pr) for i in range(maxL)],y_train, epochs=50,batch_size=100, 
           #validation_data=([funx(i,maxL,Test=True) for i in range(maxL)],y_test),
           validation_data=([funxPerm(i,maxL,pr,Test=True) for i in range(maxL)],y_test),
           #callbacks=[es,reduce_lr,checkpoint])
           callbacks=[es,reduce_lr])



    #model.load_weights("weights.best.hdf5")


    #data1=model.predict([funx(i,maxL,Test=True) for i in range(maxL)])
    #data2=model.predict([funx(i,maxL,Test=False) for i in range(maxL)])
    data1=model.predict([funxPerm(i,maxL,pr,Test=True) for i in range(maxL)])
    data2=model.predict([funxPerm(i,maxL,pr,Test=False) for i in range(maxL)])


    accTest[cnt1].append(accuracy(data1,y_test))
    accTrain[cnt1].append(accuracy(data2,y_train))
    accTest1[cnt1].append(accuracy1(data1,y_test))
    accTrain1[cnt1].append(accuracy1(data2,y_train))
    print("CNT1:",cnt1)
    print("###########     !!!!!!!!!!!!!!!!!!!!!!!!! ###############")
    print("###########       Train,Test Accuracy: ###############",accTrain[cnt1][-1],accTest[cnt1][-1])
    print("###########       Corrected Train,Test Accuracy: ###############",accTrain1[cnt1][-1],accTest1[cnt1][-1])
    print("###########     !!!!!!!!!!!!!!!!!!!!!!!!! ###############")
    del model
    del es
    del reduce_lr


print()
print()

for i in range(len(permutations)):
    print(np.mean(accTrain[i]),np.mean(accTest[i]),"             ",np.mean(accTrain1[i]),np.mean(accTest1[i]))

for i in range(len(permutations)):
    print(np.std(accTrain[i]),np.std(accTest[i]),"             ",np.std(accTrain1[i]),np.std(accTest1[i]))


#for i in range(200):
#
# 
##    model.fit([funx(i,maxL) for i in range(maxL)],y_train, epochs=1220,batch_size=100, 
##           validation_data=([funx(i,maxL,Test=True) for i in range(maxL)],y_test),
##           callbacks=[es,reduce_lr])
#
#    model.fit([funx(i,maxL) for i in range(maxL)],y_train, epochs=10,batch_size=100, 
#           validation_data=([funx(i,maxL,Test=True) for i in range(maxL)],y_test),
#           callbacks=[es,reduce_lr])
#
#
#
#    data1=model.predict([funx(i,maxL,Test=True) for i in range(maxL)])
#
#    accTest=accuracy(data1,y_test)
#
#
#    print("Test Accuracy:",accTest)

