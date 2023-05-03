#!/usr/bin/env python
# coding: utf-8

# # Importing the Required Libraries.

# In[1]:


import keras 
get_ipython().run_line_magic('env', 'SM_FRAMEWORK=tf.keras')
import segmentation_models as sm
import glob
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy.ndimage import label, generate_binary_structure
import radiomics
import cv2
import SimpleITK as sitk
import six
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.metrics import roc_auc_score
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA 
from sklearn.svm import LinearSVC
from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import mutual_info_classif as MIC
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.feature_selection import SelectKBest, SelectPercentile
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_validate
from sklearn.tree import DecisionTreeClassifier 
from sklearn.linear_model import Perceptron
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
# from hyperopt import hp

import warnings
warnings.filterwarnings('ignore')
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


# # Defining the path to Model and Images

# In[2]:


IMAGE_SIZE = (256,256,3)

path_base_model = './/models//'

# # Loading the models

# In[4]:


BACKBONE = 'efficientnetb0'
model1 = sm.Unet(BACKBONE, input_shape = (IMAGE_SIZE[0],IMAGE_SIZE[1],IMAGE_SIZE[2]),classes=1, activation='sigmoid',encoder_weights='imagenet')
model2 = sm.Unet(BACKBONE, input_shape = (IMAGE_SIZE[0],IMAGE_SIZE[1],IMAGE_SIZE[2]),classes=1, activation='sigmoid',encoder_weights='imagenet')
model3 = sm.Unet(BACKBONE, input_shape = (IMAGE_SIZE[0],IMAGE_SIZE[1],IMAGE_SIZE[2]),classes=1, activation='sigmoid',encoder_weights='imagenet')

BACKBONE = 'efficientnetb7'
model4 = sm.Unet(BACKBONE, input_shape = (IMAGE_SIZE[0],IMAGE_SIZE[1],IMAGE_SIZE[2]),classes=1, activation='sigmoid',encoder_weights='imagenet')
model5 = sm.Unet(BACKBONE, input_shape = (IMAGE_SIZE[0],IMAGE_SIZE[1],IMAGE_SIZE[2]),classes=1, activation='sigmoid',encoder_weights='imagenet')

preprocess_input = sm.get_preprocessing(BACKBONE)

model1.load_weights(path_base_model + 'model1.hdf5')
model2.load_weights(path_base_model + 'model2.hdf5')
model3.load_weights(path_base_model + 'model3.hdf5')
model4.load_weights(path_base_model + 'model4.hdf5')
model5.load_weights(path_base_model + 'model5.hdf5')


# # Defining Required Functions

# In[5]:


def preprocessing_HE(img_):
    
    hist, bins = np.histogram(img_.flatten(), 256,[0,256])
    cdf = hist.cumsum()
    cdf_m = np.ma.masked_equal(cdf,0)
    cdf_m = (cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min())
    cdf = np.ma.filled(cdf_m,0).astype('uint8')

    img_2 = cdf[img_]
    
    return img_2  
        
def get_binary_mask (mask_, th_ = 0.5):
    mask_[mask_>th_]  = 1
    mask_[mask_<=th_] = 0
    return mask_
    
def ensemble_results (mask1_, mask2_, mask3_, mask4_, mask5_):
    
    mask1_ = get_binary_mask (mask1_)
    mask2_ = get_binary_mask (mask2_)
    mask3_ = get_binary_mask (mask3_)
    mask4_ = get_binary_mask (mask4_)
    mask5_ = get_binary_mask (mask5_)
    
    ensemble_mask = mask1_ + mask2_ + mask3_ + mask4_ + mask5_
    ensemble_mask[ensemble_mask<=2.0] = 0
    ensemble_mask[ensemble_mask> 2.0] = 1
    
    return ensemble_mask

def postprocessing_HoleFilling (mask_):
    
    ensemble_mask_post_temp = ndimage.binary_fill_holes(mask_).astype(int)
     
    return ensemble_mask_post_temp

def get_maximum_index (labeled_array):
    
    ind_nums = []
    for i in range (len(np.unique(labeled_array)) - 1):
        ind_nums.append ([0, i+1])
        
    for i in range (1, len(np.unique(labeled_array))):
        ind_nums[i-1][0] = len(np.where (labeled_array == np.unique(labeled_array)[i])[0])
        
    ind_nums = sorted(ind_nums)
    
    return ind_nums[len(ind_nums)-1][1], ind_nums[len(ind_nums)-2][1]
    
def postprocessing_EliminatingIsolation (ensemble_mask_post_temp):
        
    labeled_array, num_features = label(ensemble_mask_post_temp)
    
    ind_max1, ind_max2 = get_maximum_index (labeled_array)
    
    ensemble_mask_post_temp2 = np.zeros (ensemble_mask_post_temp.shape)
    ensemble_mask_post_temp2[labeled_array == ind_max1] = 1
    ensemble_mask_post_temp2[labeled_array == ind_max2] = 1    
    
    return ensemble_mask_post_temp2.astype(int)

def get_prediction(model_, img_org_):
    
    img_org_resize = cv2.resize(img_org_,(IMAGE_SIZE[0],IMAGE_SIZE[1]),cv2.INTER_AREA)
    img_org_resize_HE = preprocessing_HE (img_org_resize)    
    img_ready = preprocess_input (img_org_resize_HE)

    img_ready = np.expand_dims(img_ready, axis=0) 
    pr_mask = model_.predict(img_ready)
    pr_mask = np.squeeze(pr_mask)
    pr_mask = np.expand_dims(pr_mask, axis=-1)    
    return pr_mask[:,:,0]

# ## Create and intantiate Feature Extractor and enable the required features

# In[9]:


extractor = radiomics.featureextractor.RadiomicsFeatureExtractor(force2D=True)
extractor.enableImageTypeByName('Original')  # extract features from the original image
extractor.enableFeatureClassByName('firstorder')  # extract first-order features
extractor.enableFeatureClassByName('glcm')  # extract GLCM features
extractor.enableFeatureClassByName('gldm')  # extract GLDM features
extractor.enableFeatureClassByName('glszm')  # extract GLSZM features
extractor.enableFeatureClassByName('ngtdm')  # extract NGTDM features



features_name = ['diagnostics_Versions_PyRadiomics',
 'diagnostics_Versions_Numpy',
 'diagnostics_Versions_SimpleITK',
 'diagnostics_Versions_PyWavelet',
 'diagnostics_Versions_Python',
 'diagnostics_Configuration_Settings',
 'diagnostics_Configuration_EnabledImageTypes',
 'diagnostics_Image-original_Hash',
 'diagnostics_Image-original_Dimensionality',
 'diagnostics_Image-original_Spacing',
 'diagnostics_Image-original_Size',
 'diagnostics_Image-original_Mean',
 'diagnostics_Image-original_Minimum',
 'diagnostics_Image-original_Maximum',
 'diagnostics_Mask-original_Hash',
 'diagnostics_Mask-original_Spacing',
 'diagnostics_Mask-original_Size',
 'diagnostics_Mask-original_BoundingBox',
 'diagnostics_Mask-original_VoxelNum',
 'diagnostics_Mask-original_VolumeNum',
 'diagnostics_Mask-original_CenterOfMassIndex',
 'diagnostics_Mask-original_CenterOfMass',
 'original_firstorder_10Percentile',
 'original_firstorder_90Percentile',
 'original_firstorder_Energy',
 'original_firstorder_Entropy',
 'original_firstorder_InterquartileRange',
 'original_firstorder_Kurtosis',
 'original_firstorder_Maximum',
 'original_firstorder_MeanAbsoluteDeviation',
 'original_firstorder_Mean',
 'original_firstorder_Median',
 'original_firstorder_Minimum',
 'original_firstorder_Range',
 'original_firstorder_RobustMeanAbsoluteDeviation',
 'original_firstorder_RootMeanSquared',
 'original_firstorder_Skewness',
 'original_firstorder_TotalEnergy',
 'original_firstorder_Uniformity',
 'original_firstorder_Variance',
 'original_glcm_Autocorrelation',
 'original_glcm_ClusterProminence',
 'original_glcm_ClusterShade',
 'original_glcm_ClusterTendency',
 'original_glcm_Contrast',
 'original_glcm_Correlation',
 'original_glcm_DifferenceAverage',
 'original_glcm_DifferenceEntropy',
 'original_glcm_DifferenceVariance',
 'original_glcm_Id',
 'original_glcm_Idm',
 'original_glcm_Idmn',
 'original_glcm_Idn',
 'original_glcm_Imc1',
 'original_glcm_Imc2',
 'original_glcm_InverseVariance',
 'original_glcm_JointAverage',
 'original_glcm_JointEnergy',
 'original_glcm_JointEntropy',
 'original_glcm_MCC',
 'original_glcm_MaximumProbability',
 'original_glcm_SumAverage',
 'original_glcm_SumEntropy',
 'original_glcm_SumSquares',
 'original_gldm_DependenceEntropy',
 'original_gldm_DependenceNonUniformity',
 'original_gldm_DependenceNonUniformityNormalized',
 'original_gldm_DependenceVariance',
 'original_gldm_GrayLevelNonUniformity',
 'original_gldm_GrayLevelVariance',
 'original_gldm_HighGrayLevelEmphasis',
 'original_gldm_LargeDependenceEmphasis',
 'original_gldm_LargeDependenceHighGrayLevelEmphasis',
 'original_gldm_LargeDependenceLowGrayLevelEmphasis',
 'original_gldm_LowGrayLevelEmphasis',
 'original_gldm_SmallDependenceEmphasis',
 'original_gldm_SmallDependenceHighGrayLevelEmphasis',
 'original_gldm_SmallDependenceLowGrayLevelEmphasis',
 'original_glrlm_GrayLevelNonUniformity',
 'original_glrlm_GrayLevelNonUniformityNormalized',
 'original_glrlm_GrayLevelVariance',
 'original_glrlm_HighGrayLevelRunEmphasis',
 'original_glrlm_LongRunEmphasis',
 'original_glrlm_LongRunHighGrayLevelEmphasis',
 'original_glrlm_LongRunLowGrayLevelEmphasis',
 'original_glrlm_LowGrayLevelRunEmphasis',
 'original_glrlm_RunEntropy',
 'original_glrlm_RunLengthNonUniformity',
 'original_glrlm_RunLengthNonUniformityNormalized',
 'original_glrlm_RunPercentage',
 'original_glrlm_RunVariance',
 'original_glrlm_ShortRunEmphasis',
 'original_glrlm_ShortRunHighGrayLevelEmphasis',
 'original_glrlm_ShortRunLowGrayLevelEmphasis',
 'original_glszm_GrayLevelNonUniformity',
 'original_glszm_GrayLevelNonUniformityNormalized',
 'original_glszm_GrayLevelVariance',
 'original_glszm_HighGrayLevelZoneEmphasis',
 'original_glszm_LargeAreaEmphasis',
 'original_glszm_LargeAreaHighGrayLevelEmphasis',
 'original_glszm_LargeAreaLowGrayLevelEmphasis',
 'original_glszm_LowGrayLevelZoneEmphasis',
 'original_glszm_SizeZoneNonUniformity',
 'original_glszm_SizeZoneNonUniformityNormalized',
 'original_glszm_SmallAreaEmphasis',
 'original_glszm_SmallAreaHighGrayLevelEmphasis',
 'original_glszm_SmallAreaLowGrayLevelEmphasis',
 'original_glszm_ZoneEntropy',
 'original_glszm_ZonePercentage',
 'original_glszm_ZoneVariance',
 'original_ngtdm_Busyness',
 'original_ngtdm_Coarseness',
 'original_ngtdm_Complexity',
 'original_ngtdm_Contrast',
 'original_ngtdm_Strength']

# In[12]:


df = pd.read_csv('final_dataset.csv')
df.shape


# In[13]:


X_train, X_test, y_train, y_test = train_test_split(
    df.drop(labels=['Label'], axis=1),
    df['Label'],
    test_size=0.2,
    random_state=23)

X_train.shape, X_test.shape


# In[14]:


from sklearn.preprocessing import StandardScaler
def scale_data(dataset):
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(dataset)
    return scaled_data


# In[15]:


scaled_train_set = scale_data(X_train)

scaled_test_set = scale_data(X_test)


# In[16]:


RF = RandomForestClassifier(criterion = 'entropy',max_depth = None, max_features = 'log2', max_leaf_nodes = None,
                               min_samples_leaf = 1,min_samples_split = 4, min_weight_fraction_leaf = 0.0, n_estimators = 200)
RF.fit(X_train,y_train) 


# In[17]:


import pickle
# create an iterator object with write permission - model.pkl
with open('model_pkl', 'wb') as files:
    pickle.dump(RF, files)


# In[19]:


model = pickle.load(open('model_pkl', 'rb'))
model.score(X_test,y_test)

# In[23]:


# from sklearn.preprocessing import MinMaxScaler


# In[42]:


def pneumonia(image):
    img_ = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized_img = cv2.resize(img_,(256,256))
    original_image = 'original.jpeg'
    cv2.imwrite(original_image,resized_img)
    img = cv2.imread(original_image)
    pr_mask1 = get_prediction (model1, img);
    pr_mask2 = get_prediction (model2, img);
    pr_mask3 = get_prediction (model3, img);
    pr_mask4 = get_prediction (model4, img);
    pr_mask5 = get_prediction (model5, img);   
    
    ensemble_mask            = ensemble_results (pr_mask1, pr_mask2, pr_mask3, pr_mask4, pr_mask5)
    ensemble_mask_post_HF    = postprocessing_HoleFilling (ensemble_mask)
    ensemble_mask_post_HF_EI = postprocessing_EliminatingIsolation (ensemble_mask_post_HF)
  
    mask = 'mask.jpeg'
    cv2.imwrite(mask,ensemble_mask_post_HF_EI*255)
  
    features = {}
    df = pd.DataFrame(columns=features_name)
    
    image_ = sitk.ReadImage(original_image, sitk.sitkInt8)
    mask = sitk.ReadImage(mask, sitk.sitkInt8)
    features = extractor.execute(image_, mask)
    
    df = df.append(features, ignore_index=True)
    cols = df.columns[22:]

    # Create new dataframe with selected columns
    DataFrame = df[cols]

    prediction = model.predict(DataFrame)

    if prediction == 0: # Determine the predicted class
        Label = "Normal"

    elif prediction == 1:
        Label = "Pneumonia"

    return Label


# In[43]:


import gradio as gr
iface =gr.Interface(fn = pneumonia,
                    inputs = "image",
                    outputs = [gr.outputs.Textbox(label="Prediction")],
                    title = "Pnuemonia Detection from Chest X ray images",
                    description = "Upload a Chest X ray image")
iface.launch(share = True)





