import torch
from model import  get_init_model
from dataset import get_dataloader

from train import validation
from config import Config

if __name__ == "__main__":
    
    PATH = "opus_books_weigths/tmodel_29.pt"

    device = "cpu"
    _, val_dataloader, src_tokenizer, tgt_tokenizer = get_dataloader()
    
    model = get_init_model(src_tokenizer.get_vocab_size(), tgt_tokenizer.get_vocab_size(),
                           Config.SEQ_LEN, Config.SEQ_LEN, Config.D_MODEL, Config.D_FF, Config.N,
                           Config.HEAD, Config.K)
    
    state = torch.load(PATH, weights_only=True, map_location=torch.device("cpu"))
    model.load_state_dict(state['model_state_dict'])
    
    validation(model, val_dataloader, tgt_tokenizer, Config.SEQ_LEN, device, lambda msg: print(msg), top_k=5, temperature=1, num_example=5)

