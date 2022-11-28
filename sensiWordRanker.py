import argparse, os, re
from os.path import isfile
import re
import math

from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from transformers import BertTokenizer, GPT2LMHeadModel
import torch
from torch.nn import CrossEntropyLoss, MSELoss

def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def sliceMatrixCrossEntropy(t_input, t_target, leftlen, rightlen, slicelen=1):
    loss_fct = CrossEntropyLoss()
    tklen = t_target.size(0)
    mask_logits = torch.cat((t_input[leftlen:leftlen+slicelen,:], t_input[tklen-rightlen-1:,:]), dim=0)
    mask_labels = torch.cat((t_target[leftlen:leftlen+slicelen], t_target[tklen-rightlen-1:]), dim=0)    
    mask_loss = loss_fct(mask_logits.view(-1, mask_logits.size(-1)), mask_labels.view(-1))
    return mask_loss.item()

parser = argparse.ArgumentParser()

parser.add_argument("--input", type=str, help="string to find sensi, need have only one * mark", required=True)
parser.add_argument("--wordfile", type=str, nargs="?", help="path to the word list", default="sensi.txt")
parser.add_argument("--modelpath", type=str, nargs="?", help="gpt model name", default="uer/gpt2-chinese-cluecorpussmall")
parser.add_argument("--device", type=str, default="cpu", help="CPU or GPU (mps for macOs/cuda/cuda:0/cuda:1/...)",)

parser.add_argument("--batchsize", type=int, default=16, help="batch size")
parser.add_argument("--contextlen", type=int, default=20, help="using how many context to do inference, large=high precision but slower")
parser.add_argument("--ppltklen", type=int, default=2, help="how many tokens to calculate ppl")

opt = parser.parse_args()

print(f"loading GPT models: {opt.modelpath}")
tokenizer = BertTokenizer.from_pretrained(opt.modelpath)
model = GPT2LMHeadModel.from_pretrained(opt.modelpath).to(opt.device)

def getBatchPPlsVariableLen(words, left_tks, right_tks, midlen, maxslicelen=2):
    word_tks = []
    for word in words:
        tks = tokenizer([word], return_tensors="pt", add_special_tokens=False)
        if len(tks.input_ids[0]) != midlen:
            print("???", word, len(tks.input_ids[0]))
        
        word_tks.append(tks.input_ids[0])
        
    batch_size = len(words)
    bm = word_tks[0:batch_size]
    tns = []
    maxtklen = max([x.size(0) for x in bm])
    endlens = []
    leftlen = left_tks.size(0)
    rightlen = right_tks.size(0)
    
    slicelen = midlen
    if slicelen > maxslicelen:
        slicelen = maxslicelen
        
    sp_1 = torch.empty(1, dtype=word_tks[0].dtype).fill_(tokenizer.cls_token_id)
    sp_2 = torch.empty(1, dtype=word_tks[0].dtype).fill_(tokenizer.sep_token_id)

    for i in range(0, batch_size):
        rightpadding = torch.tensor([], dtype=word_tks[0].dtype)

        if bm[i].size(0) < maxtklen:
            rightpadding = torch.empty(maxtklen - bm[i].size(0), dtype=word_tks[0].dtype).fill_(tokenizer.pad_token_id)
        
        bi = torch.cat((sp_1, left_tks, bm[i], right_tks, sp_2, rightpadding), dim=0).type(word_tks[0].dtype)
        tns.append(bi)
        
        endlens.append(bi.size(0) - rightpadding.size(0) - 1)
        
    inputts = torch.stack(tns)
    input_ids = inputts.to(opt.device)
    target_ids = input_ids.clone()
    target_ids[:, :-inputts.size(1)] = -100

    losses = []
    with torch.no_grad():
        loss_fct = CrossEntropyLoss()
        outputs = model(input_ids, labels=target_ids)

        lm_logits = outputs.logits
        shift_logits = lm_logits[..., :-1, :].contiguous()
        shift_labels = target_ids[..., 1:].contiguous()

        for i in range(0, batch_size):
            loss = sliceMatrixCrossEntropy(shift_logits[i,:endlens[i],:], shift_labels[i,:endlens[i]], leftlen, rightlen, slicelen=slicelen)
            losses.append((words[i], loss))
    
    return losses    

def rankWordsVariableLen(input, words, contextlen=100, maxslicelen=2):
    if '*' not in input:
        print("* mark is not in the string")
        return None
    
    parts = re.split(r'[\*]+', input.strip())
    if len(parts) != 2:
        print("only support one * mark now")
        return None

    maxcontext = contextlen
    
    left = parts[0]
    if len(left) > maxcontext:
        left = left[-maxcontext:]
    leftk = tokenizer([left], return_tensors="pt", add_special_tokens=False).input_ids[0]

    right = parts[1]
    if len(right) > maxcontext:
        right = right[:maxcontext]
    rightk = tokenizer([right], return_tensors="pt", add_special_tokens=False).input_ids[0]
    
    print(f"candidate wordset count: {len(words)}")
    print("rank word between:")
    print(f"   {left}??{right}")

    word_tks = []
    for word in words:
        tks = tokenizer([word], return_tensors="pt", add_special_tokens=False)
        word_tks.append((word, len(tks.input_ids[0])))

    word_tks = sorted(word_tks, key=lambda x: x[1], reverse=True)
    
    allppls = []
    maxtklen = word_tks[0][1]
    for iterlen in range(maxtklen, 0, -1):
        nowwords = [x[0] for x in word_tks if x[1] == iterlen]
        if len(nowwords) == 0:
            continue

        chks = list(chunks(nowwords, opt.batchsize))
        for batch in chks:
            ppls = getBatchPPlsVariableLen(list(batch), leftk, rightk, midlen=iterlen, maxslicelen=maxslicelen)
            allppls = allppls + ppls

    allppls = sorted(allppls, key=lambda x:x[1])
    return allppls 

if not isfile(opt.wordfile):
    print("word list file not exists")
    exit()

candi_words = []
with open(opt.wordfile, encoding='utf-8') as fp:
    for line in fp:
        line = line.strip().split('\t')[0].strip()
        candi_words.append(line)
            
candi_words = list(set(candi_words))
t_oo = rankWordsVariableLen(opt.input, candi_words, opt.contextlen, maxslicelen=opt.ppltklen)
if t_oo is None:
    print("error")
    exit()

for w, r in t_oo[:5]:
    print(f"{w} [{r}]")

