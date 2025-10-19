#!/usr/bin/env python
# coding: utf-8

# In[1]:


from Imports import *
from Preprocessing import *


# In[ ]:


gpuAcceleration = False
if gpuAcceleration:
    backend_options = {
        'device': "GPU",
    }

    options = {
        'backend_options': backend_options
    }

    sampler = Sampler(options = options)
else:
    sampler = Sampler()


# In[16]:


feature_dimension=8
reps=2

# Data paths
TRAIN_FILE = '../Data/X_train_scaled.csv'
TEST_FILE = '../Data/X_test_scaled.csv'
LABEL_TRAIN_FILE = '../Data/2025-Quantathon-Tornado-Q-training_data-640-examples.xlsx'
LABEL_TEST_FILE = '../Data/2025-Quantum-Tornado-Q-test_data-200-examples.xlsx'
SAVE_LOCATION = "../kernels/"

df_train = pd.read_csv(TRAIN_FILE)
df_test = pd.read_csv(TEST_FILE)

df_label_train_class = pd.read_excel(LABEL_TRAIN_FILE)["ef_class"]

df_label_test_class = pd.read_excel(LABEL_TEST_FILE)["ef_class"]

df_label_train_class, df_train = removeWeakTornados(df_label_train_class, df_train)
df_label_test_class, df_test = removeWeakTornados(df_label_test_class, df_test)

# Renormalize Data
df_train = df_train.drop(df_train.columns[-1], axis = 1)
df_train = np.tanh(df_train)
df_test = df_test.drop(df_test.columns[-1], axis = 1)
df_test = np.tanh(df_test)

print(f"✓ Training data loaded: {df_train.shape[0]} rows, {df_train.shape[1]} columns")
print(f"✓ Test data loaded: {df_test.shape[0]} rows, {df_test.shape[1]} columns")


# In[ ]:


#Classify on quantum feature generated data? Probably not useful, but who knows?


# In[ ]:


#Dict of which Encodings/Kernels to compute.
encodingSettings = {
    "basisEncoding": False,
    "amplitudeEncoding": False,
    "angleEncoding": False,
    "phaseEncoding": False,
    "denseAngleEncoding": False,
    "zEncoding":True,
    "zzEncoding:":True,
}

classicalKernelSettings = {
    "rbfKernel":True,
    "linearKernel":False,
    "polynomialKernel":False,
}


GAMMA = 0.5 #Probably worth trying a spread of gammas.


# # MAKE SURE TO SAVE KERNELS TO A FILE IN ./kernels !!!
# ### Also save graphs to a file! Make them pretty and readable! Make there be an option to save or not (for graphs and kernels)!

# In[6]:


#Data Encoding - Different Methods to try. 
#Make this a dict? Then settings choose which to include in dict. Makes training/fitting/faster
feature_maps = {
#basisEncoding = ,
#amplitudeEncoding = ,
#angleEncoding = ,
#phaseEncoding = ,
#denseAngleEncoding = ,
"zEncoding": z_feature_map(feature_dimension=feature_dimension, reps=reps, entanglement="linear"),
"zzEncoding": zz_feature_map(feature_dimension=feature_dimension, reps=reps, entanglement="linear")}

#what does varying reps do? Can I show a number on how varying reps changes things?
#Print/store data like circuit depth!
#Try different feature maps! Which is best?

#Different inner products?


# In[ ]:

print("beginning to compute kernels")
for encoding in feature_maps:
    if encodingSettings[encoding]:
        print(f"computing kernel for {encoding} feature map")
        quantum_kernel = FidelityQuantumKernel(feature_map=feature_maps[encoding], fidelity=ComputeUncompute(sampler))
        quantum_kernel_train = quantum_kernel.evaluate(df_train)
        quantum_kernel_test = quantum_kernel.evaluate(df_test, df_train)
        np.save(f'./kernels/{encoding}kernel_train.npy', quantum_kernel_train)
        np.save(f'./{encoding}kernel_test.npy', quantum_kernel_test)


# In[ ]:


#RBF Kernel
if classicalKernelSettings["rbfKernel"]:
    rbf_kernel_train = rbf_kernel(df_train, df_train, gamma=GAMMA)
    rbf_kernel_test = rbf_kernel(df_test, df_train, gamma=GAMMA)
    np.save(f'./kernels/rbf_kernel_train.npy', rbf_kernel)
    np.save(f'./rbf_kernel_test.npy', rbf_kernel)


# In[ ]:


#Linear Kernel



# In[ ]:


#Polynomial Kernel




# In[ ]:


#Cosine Kernel



# In[ ]:


#Sigmoid Kernel


