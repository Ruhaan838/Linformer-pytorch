import torch
from torch.utils.data import random_split, DataLoader, Dataset

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

from pathlib import Path

from config import Config

from model.embeddings import InputEmbedding

class LangDataset(Dataset):
    def __init__(self, dataset, src_tokenizer, tgt_tokenizer, src_lang, tgt_lang, seq_len):
        
        self.seq_len = seq_len
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.dataset = dataset
        
        self.SOS = torch.tensor([tgt_tokenizer.token_to_id("[SOS]")], dtype=torch.int64)
        self.EOS = torch.tensor([tgt_tokenizer.token_to_id("[EOS]")], dtype=torch.int64)
        self.PAD = torch.tensor([tgt_tokenizer.token_to_id("[PAD]")], dtype=torch.int64)
        
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, indx):
        
        src_trg = self.dataset[indx]
        src_text = src_trg['translation'][self.src_lang]
        tgt_text = src_trg['translation'][self.tgt_lang]
        
        encode_src = self.src_tokenizer.encode(src_text).ids
        encode_tgt = self.tgt_tokenizer.encode(tgt_text).ids
        
        num_pad_src = self.seq_len - len(encode_src) - 2
        num_pad_tgt = self.seq_len - len(encode_tgt) - 1
        
        if num_pad_src < 0 or num_pad_tgt < 0:
            raise ValueError("Sentence is too long")
        
        encoder_input = torch.cat([
            self.SOS,
            torch.tensor(encode_src, dtype=torch.int64),
            self.EOS,
            torch.tensor([self.PAD] * num_pad_src, dtype=torch.int64)
        ], dim=0)
        
        decoder_input = torch.cat([
            self.SOS,
            torch.tensor(encode_tgt, dtype=torch.int64),
            torch.tensor([self.PAD] * num_pad_tgt, dtype=torch.int64)
        ], dim=0)
        
        label = torch.cat([
            torch.tensor(encode_tgt, dtype=torch.int64),
            self.EOS,
            torch.tensor([self.PAD] * num_pad_tgt, dtype=torch.int64)
        ], dim=0)
        
        
        return {
            "encoder_input":encoder_input,
            "decoder_input":decoder_input,
            "label":label,
            "src_text":src_text,
            "tgt_text":tgt_text
        }


def yield_sentences(data, lang):
    for item in data:
        yield item['translation'][lang]
    
def get_tokenizer(ds, lang):
    tokenizer_path = Path(Config.TOKENIZER_FILE.format(lang))
    if not Path.exists(tokenizer_path):
        
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        word_trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"],
                                        min_frequency=2)
        tokenizer.train_from_iterator(yield_sentences(ds, lang), trainer=word_trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    
    return tokenizer

def get_dataloader():
    
    ds_raw = load_dataset(f"{Config.DATASOURCE}",f"{Config.SRC_LANG}-{Config.TGT_LANG}", split='train')
    ds_raw = ds_raw.select(range(Config.DATASET_LEN))

    tokenizer_src = get_tokenizer(ds_raw, Config.SRC_LANG)
    tokenizer_tgt = get_tokenizer(ds_raw, Config.TGT_LANG)
    
    max_len_src = 0
    max_len_tgt = 0
    
    for item in ds_raw:
        src_ids = tokenizer_src.encode(item['translation'][Config.SRC_LANG]).ids
        tgt_ids = tokenizer_src.encode(item['translation'][Config.TGT_LANG]).ids
        max_len_src = max(max_len_src, len(src_ids))
        max_len_tgt = max(max_len_tgt, len(tgt_ids))
    
    print(f"Max Length of SRC: {max_len_src}")
    print(f"Max Length of TGT: {max_len_tgt}")
    
    train_size = int(0.9 * len(ds_raw))
    val_size = len(ds_raw) - train_size

    train_dataset, val_dataset = random_split(ds_raw, (train_size, val_size))
    train_dataset = LangDataset(train_dataset, tokenizer_src, tokenizer_tgt, Config.SRC_LANG, Config.TGT_LANG, Config.SEQ_LEN)
    val_dataset = LangDataset(val_dataset, tokenizer_src, tokenizer_tgt, Config.SRC_LANG, Config.TGT_LANG, Config.SEQ_LEN)
    
    train_dataloader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=Config.NUM_WORKERS, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=True, num_workers=Config.NUM_WORKERS, pin_memory=True)
    
    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt

if __name__ == "__main__":
    train_dataloader , val_dataloader, tokenizer_src, tokenizer_tgt = get_dataloader()
    
    print(len(train_dataloader))
    print(len(val_dataloader))
    
    for data in val_dataloader:
        print(data["encoder_input"].shape)
        em = InputEmbedding(512, tokenizer_src.get_vocab_size())
        print(em(data["encoder_input"]).shape)
        print(data["decoder_input"].shape)
        print(data["label"].shape)
        print(data["src_text"])
        print(data["tgt_text"])
        break