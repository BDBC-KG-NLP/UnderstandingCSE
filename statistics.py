import re
import os
import pdb
import json
import random
import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd

from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer
from isotropy.models import IBert, IRoberta
from typing import List, Tuple, Dict

DEFAULT_TOKENIZER = AutoTokenizer.from_pretrained(r'google-bert/bert-base-uncased')

def load_model(model_path: str, is_bert: bool = True):
    # Load transformers' model checkpoint
    if is_bert:
        model = IBert.from_pretrained(model_path)
    else:
        model.IRoberta.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model = model.train()
    return model, tokenizer, device

def load_stsb(split: str = 'validation'):
    stsb = load_dataset(r'mteb/stsbenchmark-sts', split=split)

    res = {left: [] for left in range(5)}
    
    for row in stsb:
        for score in range(4, -1, -1):
            if row['score'] >= score:
                res[score].append((
                    row['sentence1'], row['sentence2'], row['score']
                ))
                break
    
    return res

def load_batch(path: str):
    with open(path, 'r', encoding='utf8') as fi:
        lines = [json.loads(line) for line in fi]

    return [line['sentence'] for line in lines]

def split_corpus(corpus_path: str,save_dir: str = None, heldout_percent: float = 0.1):
    with open(corpus_path, 'r', encoding='utf8') as fi:
        lines = [json.loads(line) for line in fi]
    
    random.shuffle(lines)
    
    heldout = lines[:int(len(lines) * heldout_percent)]
    train = lines[int(len(lines) * heldout_percent): ]
    
    name = os.path.splitext(os.path.split(corpus_path)[-1])[0]

    with open(os.path.join(save_dir, f'{name}_heldout.jsonl'), 'w', encoding='utf8') as fo:
        fo.write('\n'.join(json.dumps(line, ensure_ascii=False) for line in heldout))
    with open(os.path.join(save_dir, f'{name}_train.jsonl'), 'w', encoding='utf8') as fo:
        fo.write('\n'.join(json.dumps(line, ensure_ascii=False) for line in train))

def batcher(sentences, model, tokenizer, device, max_length=512):
    # Tokenization
    if max_length is not None:
        batch = tokenizer.batch_encode_plus(
            sentences,
            return_tensors='pt',
            padding=True,
            max_length=max_length,
            truncation=True
        )
    else:
        batch = tokenizer.batch_encode_plus(
            sentences,
            return_tensors='pt',
            padding=True,
        )

    # Move to the correct device
    for k in batch:
        batch[k] = batch[k].to(device)
    
    # Get raw embeddings
    with torch.no_grad():
        outputs = model(**batch, output_hidden_states=True, return_dict=True, sent_emb=True)
        last_hidden = outputs.last_hidden_state
        pooler_output = outputs.pooler_output
        hidden_states = outputs.hidden_states

    return pooler_output.cpu()


def align_loss(x, y, alpha=2):
    x = F.normalize(x, dim=-1)
    y = F.normalize(y, dim=-1)
    return (x - y).norm(p=2, dim=1).pow(alpha).mean().item()

def uniform_loss(x, t=2):
    x = F.normalize(x, dim=-1)
    return torch.pdist(x, p=2).pow(2).mul(-t).exp().mean().log().item()

def cal_avg_pos_cos(embeds): # embeds of shape[bs, 2, hs]
    return F.cosine_similarity(embeds[:, 0], embeds[:, 1], dim=-1).mean().item()

def cal_avg_neg_cos(embeds): # embeds of shape[bs, 2, hs]
    sim_matrix = F.cosine_similarity(embeds[:, :1], embeds[None, :, 1], dim=-1)
    eye = torch.eye(sim_matrix.shape[0]).to(sim_matrix)
    sim_matrix = sim_matrix[eye == 0]
    return sim_matrix.mean().item()



def cal_embeddings(model, tokenizer, device, sentences, batch_size=256):
    paired = isinstance(sentences[0], tuple)

    b_sentences = []
    features = []
    for sentence in tqdm(sentences):
        if paired:
            for l_sentence in sentence:
                b_sentences.append(l_sentence)
        else:
            b_sentences.append(sentence)

        if len(b_sentences) >= batch_size:
            features.append(batcher(b_sentences, model, tokenizer, device))
            b_sentences = []
    if len(b_sentences) > 0:
        features.append(batcher(b_sentences, model, tokenizer, device))
        b_sentences = []
    
    if paired:
        for i, b_features in enumerate(features):
            features[i] = b_features.reshape(-1, len(sentences[0]), b_features.shape[-1])
    
    features = torch.cat(features, dim=0)
    return features

def cal_cos4stsb(model_path: str, stsb_split: str, path: str = None):
    model_set = load_model(model_path)
    stsb_levels = load_stsb(stsb_split)
    
    def _cal(stsb_pairs: List[Tuple]):
        sentence_pairs = []
        scores = []
        for stsb_pair in stsb_pairs:
            sentence_pairs.append((stsb_pair[0], stsb_pair[1]))
            scores.append(stsb_pair[2])
        
        embeddings = cal_embeddings(*model_set, sentence_pairs)
        cosine_similarities = F.cosine_similarity(embeddings[:, 0], embeddings[:, 1], dim=-1).tolist()

        assert len(scores) == len(cosine_similarities)

        return [(cosine_similarities[i], scores[i]) for i in range(len(scores))]
    
    res = {}
    for level, stsb_ps in stsb_levels.items():
        res[level] = _cal(stsb_ps)

    def _save(level2scores: Dict[int, List[Tuple]], path):
        fmt = {
            'level': [],
            'cosine_similarity': [],
            'semantic_similarity': []
        }
        for level, scores in level2scores.items():
            for score in scores:
                fmt['level'].append(level)
                fmt['cosine_similarity'].append(score[0])
                fmt['semantic_similarity'].append(score[1])

        pd.DataFrame(fmt).to_csv(path)
        
    if path:
        _save(res, path)
    else:    
        return res

def cal_dist4train(model_path: str, data_path: str, path: str = None, pos: bool = False):
    model_set = load_model(model_path)
    batch = load_batch(data_path)

    if pos:
        batch = [(sent, sent) for sent in batch]
        embeds = cal_embeddings(*model_set, batch).to(model_set[2])
        cos = F.cosine_similarity(embeds[:, 0], embeds[:, 1], dim=-1).cpu()
        res = {'cosine_similarity': cos.tolist()}

    else:
        embeds = cal_embeddings(*model_set, batch).to(model_set[2])
        matrix = F.cosine_similarity(embeds[:, None], embeds[None], dim=-1).cpu()
        cos_pair = matrix[torch.triu(torch.ones_like(matrix)) == 0]
    
        res = {'cosine_similarity': cos_pair.tolist()}
    
    if path:
        pd.DataFrame(res).to_csv(path, index=False)
    else:
        return res

def cal_cos_changes(ckpts_dir: str, data_path: str, 
    max_step: int = None, path: str = None, 
    zero_path: str = r'/home/LAB/limx/download/model/bert-base-uncased/'):
    ckpt_steps = [int(d[len('checkpoint-'):]) for d in os.listdir(ckpts_dir) if 'checkpoint-' in d]
    if max_step:
        ckpt_steps = [step for step in ckpt_steps if step <= max_step]
    ckpt_steps.sort()
    if zero_path:
        ckpt_steps.insert(0, 0)
    batch = load_batch(data_path)
    batch = [(sent, sent) for sent in batch]
    res = {
        'step': [],
        'avg_pos_cos': [],
        'avg_neg_cos': [],
        'align':[],
        'uniform': []
    }
    for ckpt_step in tqdm(ckpt_steps):
        if ckpt_step == 0:
            ckpt_dir = zero_path
        else:
            ckpt_dir = os.path.join(ckpts_dir, f'checkpoint-{ckpt_step}')
        model_set = load_model(ckpt_dir)
        embeds = cal_embeddings(*model_set, batch).to(model_set[2])
        apc = cal_avg_pos_cos(embeds)
        anc = cal_avg_neg_cos(embeds)
        align = align_loss(embeds[0], embeds[1])
        uniform = (uniform_loss(embeds[0]) + uniform_loss(embeds[1])) / 2
        res['step'].append(ckpt_step)
        res['avg_pos_cos'].append(apc)
        res['avg_neg_cos'].append(anc)
        res['align'].append(align)
        res['uniform'].append(uniform)
    
    if path:
        pd.DataFrame(res).to_csv(path, index=False)
    else:
        return res

def cal_hard_neg_percentage(
    model_path: str, data_path: str, 
    temps: List[float], ms: List[float], batch_size: int, save_pre: str = None
):
    model_set = load_model(model_path)
    batch = load_batch(data_path)
    batch = [(sent, sent) for sent in batch]
    embeds = cal_embeddings(*model_set, batch)

    def _cal4batch(batch_embeds, temp, m):
        batch_embeds_norm = F.normalize(batch_embeds, dim=-1)
        sim_matrix = batch_embeds_norm[:, 0] @ batch_embeds_norm[:, 1].T
        eye = torch.eye(sim_matrix.shape[0])
        mask = (torch.diag(sim_matrix) - (sim_matrix - 1e8 * eye).max(dim=-1)[0]) < m

        if mask.any():
            row_embeds_norm = batch_embeds_norm[:, 0][mask]
            col_embeds_norm = batch_embeds_norm[:, 1]
            weight = ((sim_matrix - 1e8 * eye).div(temp))[mask]
            weight = (weight - torch.logsumexp(weight, dim=-1, keepdim=True)).exp() # row x col

            optim_vector = col_embeds_norm[None] - row_embeds_norm[:, None] \
                * sim_matrix[mask][:, :, None] # row x col x hidden_size
            weighted_vector = optim_vector * weight[:, :, None] # row x col x hidden_size
            combined_dir = (weighted_vector).sum(dim=1) # row x hidden_size
            
            hard_neg_index = weight.max(dim=-1)[1]
            hard_neg_vecs = []
            for i in range(row_embeds_norm.shape[0]):
                hard_neg_vecs.append(weighted_vector[i, hard_neg_index[i].item()])
            hard_neg_vecs = torch.stack(hard_neg_vecs, dim=0) # row x hidden_size

            hard_combine_cos = F.cosine_similarity(combined_dir, hard_neg_vecs, dim=-1)
            percentage = hard_neg_vecs.norm(dim=-1) * hard_combine_cos / combined_dir.norm(dim=-1)
        else:
            percentage = torch.tensor([]).to(batch_embeds)
        return percentage # row

    res = {
        m: {
            'temp': [], 'percentage': []
        } for m in ms
    }
    for m in ms:
        for t in temps:
            percentages = []
            for batch_index in range(embeds.shape[0] // batch_size):
                local_batch = embeds[batch_index * batch_size: (batch_index + 1) * batch_size]
                percentages.append(_cal4batch(local_batch, t, m))
            percentages = torch.cat(percentages, dim=0)
            res[m]['temp'].append(t)
            res[m]['percentage'].append(percentages.mean().item())

    if save_pre:
        for m in res:
            pd.DataFrame(res[m]).to_csv(f'{save_pre}m{m:.2f}.csv', index=False)
    else:
        return res

def cal_r_requirements(
    model_path: str, data_path: str,
    lbds: List[float], ms: List[float], batch_size: int, save_pre: str = None
):
    model_set = load_model(model_path)
    batch = load_batch(data_path)
    batch = [(sent, sent) for sent in batch]
    embeds = cal_embeddings(*model_set, batch)

    def _cal4batch(batch_embeds, lbd, m):
        batch_embeds_norm = F.normalize(batch_embeds, dim=-1)
        sim_matrix = batch_embeds_norm[:, 0] @ batch_embeds_norm[:, 1].T
        eye = torch.eye(sim_matrix.shape[0])
        mask = (torch.diag(sim_matrix) - (sim_matrix - 1e8 * eye).max(dim=-1)[0]) < m

        if mask.any():
            anchor = batch_embeds_norm[mask][:, 0] # row x hidden_size
            pos = batch_embeds_norm[mask][:, 1] # row x hidden_size
            neg_index = (sim_matrix - 1e8 * eye).max(dim=-1)[1][mask] # row
            neg = []
            for i in range(neg_index.shape[0]):
                neg.append(batch_embeds_norm[:, 1][neg_index[i].item()])
            neg = torch.stack(neg, dim=0) # row x hidden_size

            cos_alpha = (anchor * pos).sum(dim=-1) # row
            cos_beta = (anchor * neg).sum(dim=-1) # row
            alpha = torch.arccos(cos_alpha) # row
            beta = torch.arccos(cos_beta) # row

            pos_opdir = pos - anchor * cos_alpha[:, None] # row x hidden_size
            neg_opdir = neg - anchor * cos_beta[:, None] # row x hidden_size
            cos_theta = F.cosine_similarity(pos_opdir, neg_opdir, dim=-1) # row
            theta = torch.arccos(cos_theta) # row
            
            alpha_mask = cos_alpha > 0
            beta_mask = cos_beta > 0
            theta_mask = cos_theta > 0
            
            func_mask = torch.sin(alpha).pow(2) - lbd ** 2 \
                * torch.sin(beta).pow(2) * torch.sin(theta).pow(2) > 0

            alpha = alpha[alpha_mask & beta_mask & theta_mask & func_mask]
            beta = beta[alpha_mask & beta_mask & theta_mask & func_mask]
            theta = theta[alpha_mask & beta_mask & theta_mask & func_mask]

            base = 1 / lbd + torch.sin(beta) / torch.sin(alpha) * torch.cos(theta)
            delta = (1 / lbd**2 - torch.sin(beta).pow(2) / \
                torch.sin(alpha).pow(2) * torch.sin(theta).pow(2)).sqrt()
            
            low = base - delta
            high = base + delta
            

            neg_base = (1 + lbd) * torch.sin(beta) * torch.cos(theta) 
            neg_delta = torch.sin(beta).pow(2) \
                - (lbd + 1)**2 * torch.sin(beta).pow(2) * torch.sin(theta).pow(2)
            neg_mask = neg_delta > 0
            neg_delta = torch.clamp_min(neg_delta, 1e-4).sqrt()
            neg_low = (neg_base - neg_delta) / torch.sin(alpha) / lbd
            neg_high = (neg_base + neg_delta) / torch.sin(alpha) / lbd
            neg_nan_res = torch.ones_like(neg_base) * 10000
            neg_low = neg_nan_res * (neg_mask == False) + neg_low * neg_mask
            neg_high = neg_nan_res * (neg_mask == False) + neg_high * neg_mask

            return low, high, neg_low, neg_high
        else:
            return torch.tensor([]).to(batch_embeds), torch.tensor([]).to(batch_embeds), \
                torch.tensor([]).to(batch_embeds), torch.tensor([]).to(batch_embeds)
        
    res = {
        lbd:{
            m: {
                'lower_bound': [], 'upper_bound': [],
                'neg_left': [], 'neg_right': []
            } for m in ms
        } for lbd in lbds
    }

    for lbd in lbds:
        for m in ms:
            lower_bound = []
            upper_bound = []
            neg_left = []
            neg_right = []
            for batch_index in range(embeds.shape[0] // batch_size):
                local_batch = embeds[batch_index * batch_size: (batch_index + 1) * batch_size]
                low, high, neg_low, neg_high = _cal4batch(local_batch, lbd, m)
                lower_bound.append(low)
                upper_bound.append(high)
                neg_left.append(neg_low)
                neg_right.append(neg_high)
            res[lbd][m]['lower_bound'] = torch.cat(lower_bound, dim=0).tolist()
            res[lbd][m]['upper_bound'] = torch.cat(upper_bound, dim=0).tolist()
            res[lbd][m]['neg_left'] = torch.cat(neg_left, dim=0).tolist()
            res[lbd][m]['neg_right'] = torch.cat(neg_right, dim=0).tolist()

    if save_pre:
        for lbd in res:
            save_dir = f'{save_pre}lbd{lbd:.2f}'
            os.makedirs(save_dir, exist_ok=True)
            for m in res[lbd]:
                pd.DataFrame(res[lbd][m]).to_csv(f'{save_dir}/m{m:.2f}.csv', index=False)
    else:
        return res

def cal_dynamic_ratio(
    ckpts_dir: str, data_path: str, ms: List[float], batch_size: int, 
    u: float, lbd: float, temp: float, max_step: int = None, 
    save_pre: str = None, zero_path: str = r'/home/LAB/limx/download/model/bert-base-uncased/'
):
    ckpt_steps = [int(d[len('checkpoint-'):]) for d in os.listdir(ckpts_dir) if 'checkpoint-' in d]
    if max_step:
        ckpt_steps = [step for step in ckpt_steps if step <= max_step]
    ckpt_steps.sort()
    if zero_path:
        ckpt_steps.insert(0, 0)
    batch = load_batch(data_path)
    batch = [(sent, sent) for sent in batch]

    def _cal4ratio(batch_embeds, m):
        batch_embeds_norm = F.normalize(batch_embeds, dim=-1)
        sim_matrix = batch_embeds_norm[:, 0] @ batch_embeds_norm[:, 1].T
        eye = torch.eye(sim_matrix.shape[0]).to(batch_embeds)
        mask = (torch.diag(sim_matrix) - (sim_matrix - 1e8 * eye).max(dim=-1)[0]) < m

        if mask.any():
            anchor = batch_embeds_norm[mask][:, 0] # row x hidden_size
            pos = batch_embeds_norm[mask][:, 1] # row x hidden_size
            neg_index = (sim_matrix - 1e8 * eye).max(dim=-1)[1][mask] # row
            neg = []
            for i in range(neg_index.shape[0]):
                neg.append(batch_embeds_norm[:, 1][neg_index[i].item()])
            neg = torch.stack(neg, dim=0) # row x hidden_size

            # mixcse
            hj_tilde = lbd * pos + (1 - lbd) * neg # row x hidden_size
            Z_j = hj_tilde.norm(dim=-1) # row
            hj_tilde = hj_tilde / Z_j[:, None]
            up = 1 + (anchor * (hj_tilde - neg)).sum(dim=-1).div(temp).exp() * (1 - lbd)
            down = Z_j + (anchor * (hj_tilde - neg)).sum(dim=-1).div(temp).exp() * (1 - lbd)
            mixcse = up / down

            # arccon
            theta = torch.arccos((anchor * pos).sum(dim=-1))
            arccon = torch.sin(theta + u) / torch.sin(theta)          

            # met
            met = (anchor - neg).norm(dim=-1) / (anchor - pos).norm(dim=-1)

            # mat
            mat = ((1 - (anchor * neg).sum(dim=-1).pow(2)) 
                 / (1 - (anchor * pos).sum(dim=-1).pow(2))).sqrt()

            return mixcse, arccon, met, mat
        else:
            return torch.tensor([]).to(batch_embeds), torch.tensor([]).to(batch_embeds), \
                torch.tensor([]).to(batch_embeds), torch.tensor([]).to(batch_embeds)

    res = {
        m: {
            'step': [],
            'mixcse': [],
            'arccon': [],
            'met':[],
            'mat': []
        } for m in ms
    }
    for ckpt_step in tqdm(ckpt_steps):
        if ckpt_step == 0:
            ckpt_dir = zero_path
        else:
            ckpt_dir = os.path.join(ckpts_dir, f'checkpoint-{ckpt_step}')
        model_set = load_model(ckpt_dir)
        embeds = cal_embeddings(*model_set, batch).to(model_set[2])

        for m in ms:
            mixcse = []
            arccon = []
            met = []
            mat = []
            for batch_index in range(embeds.shape[0] // batch_size):
                local_batch = embeds[batch_index * batch_size: (batch_index + 1) * batch_size]
                temp_mixcse, temp_arccon, temp_met, temp_mat = _cal4ratio(local_batch, m)
                mixcse.append(temp_mixcse)
                arccon.append(temp_arccon)
                met.append(temp_met)
                mat.append(temp_mat)
            res[m]['step'].append(ckpt_step)
            res[m]['mixcse'].append(torch.cat(mixcse, dim=0).mean().item())
            res[m]['arccon'].append(torch.cat(arccon, dim=0).mean().item())
            res[m]['met'].append(torch.cat(met, dim=0).mean().item())
            res[m]['mat'].append(torch.cat(mat, dim=0).mean().item())
        
    if save_pre:
        for m in res:
            pd.DataFrame(res[m]).to_csv(f'{save_pre}m{m}.csv', index=False)
    else:
        return res



if __name__ == '__main__':
    random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
