import os
import numpy as np
import h5py
import json
import torch
import torch.nn as nn
import torchvision.models as models
import pandas as pd
import spacy
from PIL import Image

import torch.optim as optim
from torchvision import transforms
from torchtext.vocab import vocab
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader, BatchSampler, Sampler
from torch.utils.tensorboard import SummaryWriter
import string

#python -m spacy download en_core_web_sm
spacy_eng = spacy.load("en_core_web_sm")

class VizWizDataset(Dataset):
    """Face Landmarks dataset."""
    def __init__(self, image_path, annotation_path, transform=None, freq_threshold = 5, index_fixer = 0):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.image_path = image_path
        self.annotation_path = annotation_path
        self.transform = transform
        self.index_fixer = index_fixer
        
        #load captions
        with open(self.annotation_path, 'r') as j:
            self.annotations_file = json.load(j)
        
        self.images = []
        self.captions = []
        
        #save image link
        for img in self.annotations_file["images"]:
            self.images.append(img["file_name"])
        
        for cap in self.annotations_file["annotations"]:
            self.captions.append({'caption': cap["caption"], "image_id": cap["image_id"]})
        
        #build vocabulary to tokenize sentences
        self.vocab = Vocabulary(freq_threshold)
        self.vocab.build_vocabulary(self.captions)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):        
        
        #get image and caption by id
        caption = self.captions[idx]
        img_id = self.images[caption["image_id"]-self.index_fixer]
        img = Image.open(os.path.join(self.image_path, img_id)).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        
        #tokenize the items based on built vocabulary
        numericalized_caption = [self.vocab.stoi["<SOS>"]]
        numericalized_caption += self.vocab.numericalize(caption["caption"])
        numericalized_caption.append(self.vocab.stoi["<EOS>"])
        
        return img, torch.tensor(numericalized_caption)
    
class Vocabulary:
    def __init__(self, freq_threshold):
        #start with padding token, start of sequence token, end of sequence token and unknown token
        self.itos = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.stoi = {"<PAD>":0, "<SOS>":1, "<EOS>":2, "<UNK>":3}
        self.freq_threshold = freq_threshold
        
    def __len__(self):
        return len(self.itos)
    
    #tokenizer with spacy_eng library
    @staticmethod
    def tokenizer_eng(text):
        return [tok.text.lower() for tok in spacy_eng.tokenizer(text)]
    
    #build vocabulary based on all sentences in annotation json
    def build_vocabulary(self, sentence_list):
        frequencies = {}
        idx = 4
        for sentence in sentence_list:
            for word in self.tokenizer_eng(sentence["caption"]):
                if word not in frequencies:
                    frequencies[word] = 1
                else: 
                    frequencies[word] += 1
                
                if frequencies[word] == self.freq_threshold:
                    self.stoi[word] = idx
                    self.itos[idx] = word
                    idx += 1

    #numericalize a text based on total vocabulary, if word is unknown, use unknown token 
    def numericalize(self, text):
        tokenized_text = self.tokenizer_eng(text)
        
        return [self.stoi[token] if token in self.stoi else self.stoi["<UNK>"] for token in tokenized_text]
    
#this function is called after get_item in dataset and 
class MyCollate:
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx
        
    def __call__(self, batch):
        #unsqueeze to make img dimensions (rgb channels, height, width) into dimension (1, rgb channels, height, width)
        imgs = [item[0].unsqueeze(0) for item in batch]
        #concatenate every image into one batch of (batch size, rgb channels, height, width)
        imgs = torch.cat(imgs, dim=0)
        
        #pad sequence on captions based on current batch and pytorch rnn pad_sequence-r
        targets = [item[1] for item in batch]
        targets = pad_sequence(targets, batch_first = False, padding_value = self.pad_idx)
        
        return imgs, targets
    
def get_loader(root_folder, annotation_file, transform, batch_size = 32, num_workers = 8, shuffle = True, pin_memory = True, index_fixer = 0):

    #define dataset for dataloader from pytorch with file path and transformation composition
    dataset = VizWizDataset(root_folder, annotation_file, transform = transform, index_fixer = index_fixer)

    #define padding token for dataloader
    pad_idx = dataset.vocab.stoi["<PAD>"]

    #return dataloader with given parameters
    loader = DataLoader(
                dataset = dataset,
                batch_size = batch_size, 
                num_workers = num_workers, 
                shuffle = shuffle,
                pin_memory = pin_memory, 
                collate_fn = MyCollate(pad_idx = pad_idx),
            )
    
    return loader

def get_test_set(root_folder, annotation_file, transform, index_fixer = 0):

    #define dataset for dataloader from pytorch with file path and transformation composition
    dataset = VizWizDataset(root_folder, annotation_file, transform = transform, index_fixer = index_fixer)
    
    return dataset

def main():
    print("hello")

if __name__ == "__main__":
    main()