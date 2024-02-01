import torch 
import torch.nn as nn
import numpy as np

class Encoder(nn.Module):
    def __init__(self,input_size,embedding_size,dropout_p,custom_emb,pad_idx):
        super(Encoder,self).__init__()
        self.input_size=input_size
        self.dropout=nn.Dropout(dropout_p)
        self.embedding=nn.Embedding(input_size,embedding_size,padding_idx=pad_idx)
        self.embedding.weights=torch.nn.Parameter(torch.from_numpy(custom_emb))
        self.relu=nn.ReLU()
        self.conv_layer=nn.Conv1d(in_channels=5,out_channels=1,kernel_size=1)
        self.tanh=nn.Tanh()

    def forward(self,input,isBatched=False):
        embedding=self.dropout(self.embedding(input))
        weighted_embedding=self.relu(self.dropout(self.conv_layer(embedding)))
        if isBatched:
            final_embedding=torch.sum(weighted_embedding,axis=1)      
        else:
            final_embedding=torch.sum(weighted_embedding,axis=0)
        return final_embedding
    
class Seq2SeqAttention(nn.Module):
    def __init__(self,encoder,vocab):
        super(Seq2SeqAttention,self).__init__()
        self.encoder=encoder
        self.vocab=vocab
        self.cos= nn.CosineSimilarity(dim=1, eps=1e-6)
        self.relu=nn.ReLU()

    def forward(self,source,target,isBatched=False):
        if isBatched:
            source_context_vector=self.encoder(source,isBatched=True)
            target_context_vector=self.encoder(target,isBatched=True)
        else:
            source_context_vector=self.encoder(source)
            target_context_vector=self.encoder(target)
        score=self.cos(source_context_vector,target_context_vector)
        return self.relu(score)

        
