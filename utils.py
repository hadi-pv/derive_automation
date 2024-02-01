import torch
import json
import numpy as np
from splitwords import Splitwords
import heapq

Dict={ 
    'description' : ['desc','ddescription'],
    'maintenance' : ['maint'],
    'project' : ['prj', 'proj'],
    'production' : ['prod'],
    'equipment': ['eqp' , 'equip'],
    'company' : ['comp'],
    'corrosion' : ['corr'],
    'maximum' : ['max'],
    'system' : ['sys'],
    'group' : ['grp'],
    'document' : ['doc','docs'],
    'functional' : ['func'],
    'segment' : ['se'],
    'id' : ['idx'],
    'indicator' : ['ind']
}
Dict2={v:k for k,l in Dict.items() for v in l}

def create_custom_embedding(vocab):
    arr=[]
    with open('embeddings.json','r') as jfile:
        data_dict=json.load(jfile)
    embedding_size=len(list(data_dict.values())[0])
    for key in vocab.keys():
        if key in data_dict.keys():
            arr.append(np.array(data_dict[key]))
        elif key=="<PAD>":
            arr.append(np.zeros(embedding_size))
        else:
            arr.append(np.random.rand(embedding_size))
    return np.array(arr)

def print_words_emb(fields,attributes,encoder,vocab,k_largest, no_of_words):
    attribute_weights=[]
    field_weights=[]
    splitwords=Splitwords()
    orig_fields=[]
    orig_attributes=[]
    field_dict=dict()
    for attribute_key in attributes:
        attribute=[vocab[x] if x in vocab.keys() else vocab["<PAD>"] for x in splitwords.splitwords2(attribute_key)]
        attribute_len=len(attribute)
        if len([vocab[x] for x in splitwords.splitwords2(attribute_key) if x in vocab.keys()])>0 and attribute_len<=no_of_words:
            orig_attributes.append(attribute_key)
            attribute_vector=attribute+[vocab["<PAD>"]]*(no_of_words-attribute_len)
            attribute_weights.append(encoder(torch.from_numpy(np.array(attribute_vector))).detach().numpy())
        elif attribute_len>no_of_words:
            orig_attributes.append(attribute_key)
            i=attribute_len-1
            while len(attribute)>no_of_words:
                if attribute[i]==vocab["<PAD>"]:
                    attribute=attribute[:-1]
                i-=1
            attribute_weights.append(encoder(torch.from_numpy(np.array(attribute_vector))).detach().numpy())
    attribute_weights=np.array(attribute_weights)
    for field_key in fields:
        field=[vocab[x] if x in vocab.keys() else vocab["<PAD>"] for x in splitwords.splitwords2(field_key)]
        field_len=len(field)
        if len([vocab[x] for x in splitwords.splitwords2(field_key) if x in vocab.keys()])>0 and field_len<=no_of_words:
            orig_fields.append(field_key)
            field_vector=field+[vocab["<PAD>"]]*(no_of_words-field_len)
            field_weights.append(encoder(torch.from_numpy(np.array(field_vector))).detach().numpy())
        elif field_len>no_of_words:
            orig_fields.append(field_key)
            i=field_len-1
            while len(field)>no_of_words:
                if field[i]==vocab["<PAD>"]:
                    field=field[:i]+field[i+1:]
                i-=1
            field_weights.append(encoder(torch.from_numpy(np.array(field_vector))).detach().numpy())
        else:
            field_dict[field_key]=[]
    field_weights=np.array(field_weights)
    result=np.dot(field_weights,attribute_weights.T)/(np.linalg.norm(field_weights)*np.linalg.norm(attribute_weights))
    result=[[(i,x[i]) for i in heapq.nlargest(k_largest, range(len(x)), x.take)] for x in result]
    for i,each in enumerate(orig_fields):
        field_dict[each]=[orig_attributes[k[0]] for k in result[i]]
    return field_dict

def print_result(model_result,edit_result,fields):
    result_dict=dict()
    for field_key in fields:
        both_result=[x for x in edit_result[field_key] if x in model_result[field_key]]
        model_edit = list(set(model_result[field_key]+edit_result[field_key]))
        result_dict[field_key]={"both":both_result,"model":model_result[field_key],"edit":edit_result[field_key],"model&edit":model_edit}
    return result_dict
    