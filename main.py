import json
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException, status, logger
from typing import List

import torch
from edit_dist import run_edit
from seq2seqmodel_class import Encoder, Seq2SeqAttention

from utils import create_custom_embedding, print_result, print_words_emb

da=FastAPI()

class reqBody(BaseModel):
    fields : List[str]=None
    attributes : List[str]=None

@da.get("/")
async def test_server():
    return {"message":"Server running successfully"}

@da.post("/")
async def get_resp(reqbody:reqBody):
    fields=reqbody.fields
    attributes=reqbody.attributes
    with open("token_words.json","r") as file:
        data=json.load(file)
    vocab=data["words"]
    inverse_vocab=data["inverse_words"]
    vocab["<PAD>"]=len(vocab.keys())
    inverse_vocab[len(vocab.keys())]="<PAD>"
    custom_emb=create_custom_embedding(vocab)

    #model hyperparameters
    load_model=False
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_size_encoder=len(vocab.keys())
    input_size_decoder=len(vocab.keys())
    output_size=len(vocab.keys())
    encoder_embedding_size=custom_emb.shape[1]
    pad_idx=vocab["<PAD>"]
    
    encoder_net=Encoder(input_size=input_size_encoder,embedding_size=encoder_embedding_size,dropout_p=0,custom_emb=custom_emb,pad_idx=pad_idx).to(device)
    model=Seq2SeqAttention(encoder=encoder_net,vocab=vocab)
    model.load_state_dict(torch.load('model'))
    model.eval()
    model_result=print_words_emb(fields=fields,attributes=attributes,encoder=encoder_net,vocab=vocab,k_largest=3,no_of_words=5)
    edit_result=run_edit(fields=fields,attributes=attributes)

    result=print_result(model_result,edit_result,fields)
    return result

