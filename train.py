import torch
from torch import nn
from torch import optim

from tqdm import tqdm
from model import get_init_model
from dataset import get_dataloader
from config import Config, GPUConfig
from utils import top_k_sampling_decode,calculate_belu, store_csv

from pathlib import Path
import os

import warnings
import argparse


def validation(model, val_dataset, tokenizer_tgt, loss_fn, max_len, device, print_msg, top_k = 10, temperature=1, num_example=2):
    model.eval()
    count = 0
    
    sorce_texts = []
    expected = []
    predicted = []
    belu_scores = []
    
    console_width = 80
    
    with torch.no_grad():
        for batch in val_dataset:
            count += 1
            encoder_input = batch['encoder_input'].to(device, non_blocking=True)

            assert encoder_input.size(0) == 1, "Working with batch size 1 only for Validation"
            
            model_out = top_k_sampling_decode(model, encoder_input, tokenizer_tgt, max_len, device, top_k=top_k, temperature=1)
            label = batch['label']
            src_text = batch['src_text'][0]
            tgt_text = batch['tgt_text'][0]
            model_out_txt = tokenizer_tgt.decode(model_out.detach().cpu().numpy())
            
            sorce_texts.append(src_text)
            expected.append(tgt_text)
            predicted.append(model_out_txt)
            
            print_msg("-"*console_width)
            print_msg(f"{f'SOURCE: ':>12}{src_text}")
            print_msg(f"{f'TARGET: ':>12}{tgt_text}")
            print_msg(f"{f'PREDICTED: ':>12}{model_out_txt}")
            
            if count == num_example:
                print_msg("-"*console_width)
                break
            
        belu = calculate_belu(predicted, expected)
        
        print_msg(f"{f'BELU-SCORE: ':>12}{belu}")
        belu_scores.append(belu.item())
        store_csv(f"./{Config.RESULT_FOLDER}/val_belu_score.csv", belu_scores)

def train_model(EPOCH):
    
    device = GPUConfig.device
    
    Path(f"{Config.DATASOURCE}_{Config.MODEL_FOLDER}").mkdir(parents=True, exist_ok=True)
    Path(f"{Config.RESULT_FOLDER}").mkdir(parents=True, exist_ok=True)
    
    
    train_dataloader, val_dataloader, src_tokenizer, tgt_tokenizer = get_dataloader()
    
    model = get_init_model(src_tokenizer.get_vocab_size(), tgt_tokenizer.get_vocab_size(),
                           Config.SEQ_LEN, Config.SEQ_LEN, Config.D_MODEL, Config.D_FF, Config.N,
                           Config.HEAD, Config.K)
    
    GPUConfig.load_cuda_mps(model, device)
    
    optimizer = optim.Adam(model.parameters(), lr=Config.LR, eps=1e-9, weight_decay=1e-5)
    loss_fn = nn.CrossEntropyLoss(ignore_index=src_tokenizer.token_to_id('[PAD]'), label_smoothing=0.1).to(device)
    losses = []
    for epoch in range(EPOCH):
        model.train()
        
        for data in (pbar := tqdm(train_dataloader, desc=f"EPOCH: {epoch+1}/{EPOCH}", bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')):
            
            encoder_input = data['encoder_input'].to(device, non_blocking=True)
            decoder_input = data['decoder_input'].to(device, non_blocking=True)
            label = data['label'].to(device, non_blocking=True)
            
            encoder_output = model.encode(encoder_input)
            decoder_output = model.decode(decoder_input, encoder_output, True)
            proj_out = model.projection(decoder_output)
            
            loss = loss_fn(proj_out.view(-1, tgt_tokenizer.get_vocab_size()), label.view(-1))
            pbar.set_postfix(Loss=round(loss.item(), 3))
            losses.append(loss.item())
            
            loss.backward()
            
            optimizer.step()
            optimizer.zero_grad()
        
        validation(model, val_dataloader, tgt_tokenizer, loss_fn, Config.SEQ_LEN, device, lambda msg:pbar.write(msg), 5, 1, 1)    
        
        model_filename = Config.save_model_epoch(epoch)
        torch.save({
            'epoch':epoch,
            'model_state_dict':model.state_dict(),
            'optimizer_state_dict':optimizer.state_dict()
        }, model_filename)
        
if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    parser = argparse.ArgumentParser(description="Take the Arguments of the Model Training.")
    
    parser.add_argument('--epoch', type=int, help="Number of Epoch to Run")
    parser.add_argument('--workers', type=int, help="Number of workers in dataset loading", default=0)
    parser.add_argument('--datalen', type=int, help="Length of Dataset", default=32332)
    parser.add_argument('--srclang', type=str, help="Source Language", default="en")
    parser.add_argument('--tgtlang', type=str, help="Target Language", default="it")
    
    args = parser.parse_args()
    Config.SRC_LANG = args.srclang
    Config.TGT_LANG = args.tgtlang
    Config.NUM_WORKERS = args.workers
    Config.DATASET_LEN = args.datalen

    
    train_model(args.epoch)