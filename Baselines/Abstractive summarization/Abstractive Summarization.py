#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import torch
import random
import numpy as np
import pandas as pd
from datasets import load_dataset
import nlpaug.augmenter.sentence as nas
from torch.utils.data import TensorDataset
from sklearn.model_selection import train_test_split
from transformers import MarianMTModel, MarianTokenizer
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler, SubsetRandomSampler


# In[2]:


device = 'cuda' if torch.cuda.is_available() else 'cpu'
aug = nas.AbstSummAug(model_path='t5-large', tokenizer_path='t5-large', max_length=185, batch_size=16, device=device)


# In[3]:


def get_dataset(dataset_name):
    if dataset_name == 'multi-covid':
        # class dictionary
        class_categories = {
                                'Case Report': 0,
                                'Diagnosis': 1,
                                'Epidemic Forecasting': 2,
                                'Mechanism': 3,
                                'Prevention': 4,
                                'Transmission': 5,
                                'Treatment': 6
                            }

        # Load the dataset into a pandas dataframe.
        train_df = pd.read_csv('../../BC7-LitCovid-Train.csv', encoding='latin')
        titles, abstracts, label_categories = train_df['title'].tolist(), train_df['abstract'].tolist(), train_df['label'].tolist()

        # get the training data
        batch = []
        labels = []

        for i in range(len(label_categories)):
            if ';' not in label_categories[i] and not pd.isna(titles[i]) and not pd.isna(abstracts[i]):
                sentence = titles[i] + ' ' + str(abstracts[i])
                batch.append(sentence)
                labels.append(class_categories[label_categories[i]])

        # shuffle the data
        batch = batch[:1000]
        labels = labels[:1000]

        batch = list(batch[len(batch)//3:])
        labels = list(labels[len(labels)//3:])

    if dataset_name == 'covid':
        # Load the dataset into a pandas dataframe.
        dataset = load_dataset('llangnickel/long-covid-classification-data')
        
        # get the training data
        batch = dataset['train']['text']
        labels = dataset['train']['label']
        
        # shuffle the data
        data = list(zip(batch, labels))
        random.shuffle(data)
        batch, labels = zip(*data)

        val_batch = batch[:len(batch)//2]
        val_labels = labels[:len(labels)//2]

        batch = batch[len(batch)//2:]
        labels = labels[len(labels)//2:]

    elif dataset_name == 'cancer':
        classes = {'Thyroid_Cancer' : 0,  'Lung_Cancer' : 1,  'Colon_Cancer' : 2}
        # Load the dataset into a pandas dataframe.
        df = pd.read_csv('cancer.csv', encoding='latin')
        values = df.values

        # get the entire data
        all_batch  = list(values[:,2])
        str_labels = list(values[:,1])
        all_labels = [classes[k] for k in str_labels]

        # training and test data split
        batch, test_batch, labels, test_labels = train_test_split(all_batch, all_labels, test_size=0.5, random_state=42)

        val_batch = batch[100:1100]
        val_labels = labels[100:1100]

        batch = batch[:100] # since it is an easy dataset use just 500 data points
        labels = labels[:100]
    
    elif dataset_name == 'medical_texts':
        
        f = open('Medical texts/train.dat', 'r')
        lines = f.readlines()
        batch = list()
        labels = list()
        for line in lines:
            labels.append(int(line[0])-1) # subtract 1 to make it in the range required by the model
            batch.append(line[2:len(line)-1])
        f.close()

        # training and test data split
        batch, test_batch, labels, test_labels = train_test_split(batch, labels, test_size=0.2, random_state=42)

        val_batch = batch[500:1500]
        val_labels = labels[500:1500]
        
        batch = batch[:500] # since it is an easy dataset use just 500 data points
        labels = labels[:500]

    elif dataset_name == 'medical_trans':
        
        label_dict = {' Neurology': 0,
                     ' Discharge Summary': 1,
                     ' Psychiatry / Psychology': 2,
                     ' Pediatrics - Neonatal': 3,
                     ' Neurosurgery': 4,
                     ' Gastroenterology': 5,
                     ' Emergency Room Reports': 6,
                     ' Ophthalmology': 7,
                     ' Office Notes': 8,
                     ' Cardiovascular / Pulmonary': 9,
                     ' Nephrology': 10,
                     ' ENT - Otolaryngology': 11,
                     ' Hematology - Oncology': 12,
                     ' Obstetrics / Gynecology': 13,
                     ' Orthopedic': 14,
                     ' Urology': 15,
                     ' Radiology': 16,
                     ' General Medicine': 17,
                     ' Pain Management': 18,
                     ' SOAP / Chart / Progress Notes': 19,
                     ' Consult - History and Phy.': 20,
                     ' Surgery': 21}
        
        f = open('medical_transcriptions.txt', 'r')
        lines = f.readlines()
        # get the training data
        all_batch = []
        all_labels = []
        for line in lines:
            parts = line.split('\t')
            batch_label = label_dict[parts[0]]
            if batch_label != 20 and batch_label != 21: # if included these labels there will be huge data imbalance
                all_labels.append(batch_label)
                all_batch.append(parts[1])
        f.close()
        
        # training and test data split
        batch, test_batch, labels, test_labels = train_test_split(all_batch, all_labels, test_size=0.835, random_state=42)

        val_batch, test_batch, val_labels, test_labels = train_test_split(test_batch, test_labels, test_size=0.5, random_state=42)
        
    return batch, labels


# In[4]:


dataset = 'multi-covid'
original_texts, original_labels = get_dataset(dataset)


# In[5]:


print(len(original_texts), len(original_labels))


# In[6]:


truncated_texts = []
for i in range(len(original_texts)):
    truncated_texts.append(' '.join(original_texts[i].split()[:256]))


# In[7]:


augmented_data = []

for i in range(len(truncated_texts)):
    augmented_data.append(aug.augment(truncated_texts[i]))


# In[8]:


f = open("Data/abstractivesummarization_"+str(dataset)+".txt" , "a" )

for i in range(len(augmented_data)):
    f.write(str(original_labels[i]) + '\t' + augmented_data[i][0] + '\n')
    
f.close()


# In[9]:


print(augmented_data[10][0])


# In[10]:


print(original_labels[10])


# In[ ]:




