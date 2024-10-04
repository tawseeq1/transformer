import torch
import torch.nn as nn

import config
from dataset import Translator_Dataset
from torch.utils.data import Dataset, DataLoader, random_split
from config import get_config, get_weights_file_path, latest_weights_file_path

import torchmetrics
from torch.utils.tensorboard import SummaryWriter

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace
from pathlib import Path

from model import build_transformer


def get_all_sentences(ds, lang):
    for item in ds:
        yield item['translation'][lang]
def build_tokenizer(config, ds, lang):
    tokenizer_path = Path(config['tokenizer_path'].format(lang))
    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2)
        tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer = trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer
def get_dataset(config):
    ds_raw = load_dataset(f"{config['datasource']}", f"{config['lang_src']}-{config['lang_tgt']}", split='train')
    tokenizer_src = build_tokenizer(config, ds_raw, config['lang_src'])
    tokenizer_tgt = build_tokenizer(config, ds_raw, config['lang_tgt'])
    train_ds_size = int(0.9 * len(ds_raw))
    val_ds_size = len(ds_raw) - train_ds_size
    train_ds_raw, val_ds_raw = random_split(ds_raw, [train_ds_size, val_ds_size])

    train_ds = Translator_Dataset(train_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])
    val_ds = Translator_Dataset(val_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])

    max_len_src = 0
    max_len_tgt = 0

    for item in ds_raw:
        src_ids = tokenizer_src.encode(item['translation'][config['lang_src']]).ids
        tgt_ids = tokenizer_tgt.encode(item['translation'][config['lang_src']]).ids
        max_len_src = max(max_len_src, len(src_ids))
        max_len_tgt = max(max_len_tgt, len(tgt_ids))

    print(f'Max length of source sentence: {max_len_src}')
    print(f'Max length of target sentence: {max_len_tgt}')

    train_dataloader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=True)
    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt

def get_model(config, vocab_src_len, vocab_tgt_len):
    model = build_transformer(vocab_src_len, vocab_tgt_len, config["seq_len"], config['seq_len'], d_model=config['d_model'])
    return model

def train_model():
    device =  "cuda" if torch.cuda.is_available() else "mps" if torch.has_mps or torch.backends.mps.is_available() else "cpu"
    print(f'Device : {device}')
    if device == "cuda":
        print(f'Device name: {torch.cude.get_device_name(device)}')
        print(f'Device memory : {torch.cuda.get_device_properties(device.index).total_memory / 1024 ** 3} GB')
    elif (device == 'mps'):
        print(f"Device name: <mps>")
    else:
        print("Consider using GPU for training.")
    device = torch.device(device)

Path(f'{config["datasource"]}_{config["model_folder"]}').mkdir(parents=True, exist_ok=True)

train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_dataset(config)
model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)
writer = SummaryWriter(config['experiment_name'])
optimizer = torch.optim.Adam(model.parameters(), lr = config['lr'], eps=1e-9)

initial_epoch = 0
global_step = 0
preload = config["preload"]
model_file_name =  latest_weights_file_path(config) if preload == 'latest' else get_weights_file_path(config, preload) if preload else None
if model_filename:
        print(f'Preloading model {model_filename}')
        state = torch.load(model_filename)
        model.load_state_dict(state['model_state_dict'])
        initial_epoch = state['epoch'] + 1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state['global_step']
else:
        print('No model to preload, starting from scratch')
loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_src.token_to_id('[PAD]'), label_smoothing=0.1).to(device)



