#!/usr/bin/env python
# coding: utf-8

# In[1]:


from transformers import MarianMTModel, MarianTokenizer
import os
import torch
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler, SubsetRandomSampler


# In[2]:


device = 'cuda' if torch.cuda.is_available() else 'cpu'


# In[3]:


print(device)


# In[4]:


# Get the name of the first model
first_model_name = 'Helsinki-NLP/opus-mt-en-fr'

# Get the tokenizer
first_model_tkn = MarianTokenizer.from_pretrained(first_model_name)

# Load the pretrained model based on the name
first_model = MarianMTModel.from_pretrained(first_model_name).to(device)


# In[5]:


# Get the name of the second model
second_model_name = 'Helsinki-NLP/opus-mt-fr-en'

# Get the tokenizer
second_model_tkn = MarianTokenizer.from_pretrained(second_model_name)

# Load the pretrained model based on the name
second_model = MarianMTModel.from_pretrained(second_model_name).to(device)


# In[6]:


def format_batch_texts(language_code, batch_texts):
  
    formated_bach = [">>{}<< {}".format(language_code, text) for text in batch_texts]

    return formated_bach


# In[7]:


def get_dataset(dataset_name, tokenizer=first_model_tkn, language='fr'):

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

        print(len(batch), len(labels))
    
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
    
    # Prepare the text data into appropriate format for the model
    original_texts = batch
    original_labels = labels
    batch = format_batch_texts(language, batch)
    
    # shuffle the data
    data = list(zip(batch, labels))
    random.shuffle(data)
    train_batch, labels = zip(*data)
    train_labels = torch.tensor(labels)
    
    # tokenizing the sentences
    seq_length = 128
    encoding = tokenizer(train_batch, return_tensors='pt', padding=True, truncation = True, max_length=seq_length)
    input_ids = encoding['input_ids']
    attention_mask = encoding['attention_mask']
    
    # print the shapes
    print("Input shape: ")
    print(input_ids.shape, attention_mask.shape,train_labels.shape,train_labels)
    
    # turn to the tensordataset
    train_data = TensorDataset(input_ids, attention_mask, train_labels)
        
    return original_texts, original_labels, train_data


# In[8]:


dataset = 'multi-covid'
original_texts, original_labels, train_data_classifier = get_dataset(dataset)


# In[9]:


train_classifier = DataLoader(train_data_classifier, batch_size=16, sampler=torch.utils.data.sampler.RandomSampler(train_data_classifier), pin_memory=True, num_workers=0)


# In[10]:


def perform_translation_forward(dataloader, model, tokenizer):
    
    translated_texts = []
    
    aug_labels = []
    
    for step, batch in enumerate(dataloader):
        
        print("At batch : " + str(step))
        
        translated = model.generate(batch[0].to(device))
        
        batch_translated_texts = tokenizer.batch_decode(translated, skip_special_tokens=True)
        
        translated_texts = translated_texts + batch_translated_texts
        
        aug_labels = aug_labels + batch[2].tolist()
        
    return translated_texts, aug_labels


# In[11]:


translated_texts, aug_labels = perform_translation_forward(train_classifier, first_model, first_model_tkn)


# In[12]:


print(len(translated_texts))


# In[13]:


print(translated_texts[0])


# In[14]:


def perform_translation_backward(translated_texts, model, tokenizer, language='en'):
    
    # Prepare the text data into appropriate format for the model
    formated_batch_texts = format_batch_texts(language, translated_texts)
    
    translated_texts = []
    
    batch_size = 16
    
    for i in range(0, len(formated_batch_texts), batch_size):
        
        print("At batch : " + str(i//batch_size))
        
        encoding = tokenizer(formated_batch_texts[i:i+batch_size], return_tensors="pt", padding=True, truncation = True, max_length=128)
    
        # Generate translation using model
        translated = model.generate(encoding['input_ids'].to(device))
        
        # Convert the generated tokens indices back into text
        batch_translated_texts = tokenizer.batch_decode(translated, skip_special_tokens=True)
        
        translated_texts = translated_texts + batch_translated_texts
    
    return translated_texts


# In[15]:


back_translated_texts = perform_translation_backward(translated_texts, second_model, second_model_tkn)


# In[16]:


print(len(back_translated_texts))


# In[17]:


f = open("Data/backtranslation_"+str(dataset)+".txt" , "a" )

for i in range(len(back_translated_texts)):
    f.write(str(aug_labels[i]) + '\t' + back_translated_texts[i] + '\n')
    
f.close()


# In[ ]:





# In[ ]:




