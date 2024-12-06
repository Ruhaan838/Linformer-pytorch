import torch
from torch.nn import functional as F
from config import Config

from torchmetrics import BLEUScore

import pandas as pd

def calculate_belu(pred, actual):
    belu = BLEUScore()
    return belu(pred, actual)

def top_k_sampling_decode(model, encoder_input, tokenizer_tgt, max_len, device, top_k=10, temperature=1.0):
    sos_idx = tokenizer_tgt.token_to_id('[SOS]')
    eos_idx = tokenizer_tgt.token_to_id('[EOS]')

    encoder_output = model.encode(encoder_input)
    decoder_input = torch.empty(1, Config.SEQ_LEN).fill_(sos_idx).type_as(encoder_input).to(device)

    while True:
        if decoder_input.size(1) == max_len:
            break

        out = model.decode(decoder_input, encoder_output, True)

        logits = model.projection(out[:, -1])  
        logits = logits / temperature  

        if top_k > 0:
            top_k_values, _ = torch.topk(logits, top_k)
            mask = logits < top_k_values[:, -1, None]
            logits[mask] = float('-inf')  

        probabilities = torch.softmax(logits, dim=-1)

        next_word = torch.multinomial(probabilities, num_samples=1)

        decoder_input = torch.cat(
            [decoder_input, torch.empty(1, Config.SEQ_LEN).type_as(encoder_input).fill_(next_word.item()).to(device)], dim=1
        )

        if next_word == eos_idx:
            break

    return decoder_input.squeeze(0)

def store_csv(file_name, score):
    df = pd.DataFrame({
        "Score":score
        })
    df.to_csv(file_name)