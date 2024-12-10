import torch
from pathlib import Path

class Config:
    D_MODEL : int = 512
    D_FF :int = 2048
    BATCH_SIZE = 16
    LR = 10**-4
    SEQ_LEN = 350
    SRC_LANG = "en"
    TGT_LANG = "it"
    MODEL_FOLDER = "weigths"
    MODEL_BASNAME = "tmodel_"
    TOKENIZER_FILE = "tokenizer_{0}.json"
    DATASOURCE = "opus_books"
    N = 8
    K = 256
    HEAD = 8
    NUM_WORKERS = 0
    DATASET_LEN = 32332 #max 32332
    RESULT_FOLDER = "result"
    
    @staticmethod
    def save_model_epoch(epoch):
        model_folder = f"{Config.DATASOURCE}_{Config.MODEL_FOLDER}"
        model_filename = f"{Config.MODEL_BASNAME}{str(epoch)}.pt"
        return str(Path('.') / model_folder / model_filename)
    
class GPUConfig:
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    @staticmethod
    def load_cuda_mps(model, device):
        if (device == "cuda" and (count := torch.cuda.device_count() > 1)):
            print(f"Using {count} <cuda> Device !!")
            model = torch.nn.DataParallel(model)
            return model
        else:
            print(f"Using Device <{device}> !!")
            model = model.to(device)
            return model