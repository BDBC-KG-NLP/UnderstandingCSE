# import logging
import re
import pdb

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from functools import reduce

from transformers.utils import logging
from transformers.models.roberta.modeling_roberta import RobertaPreTrainedModel, RobertaModel
from transformers.models.roberta.configuration_roberta import RobertaConfig
from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertModel
from transformers.models.bert.configuration_bert import BertConfig

from transformers.modeling_outputs import SequenceClassifierOutput, BaseModelOutputWithPoolingAndCrossAttentions
from typing import Union, Dict, Any, Tuple, Union

logger = logging.get_logger()

class IBertConfig(BertConfig):
    def __init__(self,
        pooler_type: str = 'mask',
        feature_size: int = 768,
        num_layers: int = 12,
        loss_fn_settings: str = 'infonce',
        loss_param_settings: str = 'temp=5e-2',
        mt_size: int = 768,
        **kwargs
    ): 
        super().__init__(**kwargs)
        self.pooler_type = pooler_type
        self.feature_size = feature_size
        self.num_layers = num_layers

        self.loss_fn_settings = loss_fn_settings
        self.loss_param_settings = loss_param_settings
        self.mt_size = mt_size

class IRobertaConfig(RobertaConfig):
    def __init__(self,
        pooler_type: str = 'mask',
        feature_size: int = 768,
        num_layers: int = 12,
        loss_fn_settings: str = 'infonce',
        loss_param_settings: str = 'temp=5e-2',
        mt_size: int = 768, 
        **kwargs
    ): 
        super().__init__(**kwargs)
        self.pooler_type = pooler_type
        self.feature_size = feature_size
        self.num_layers = num_layers

        self.loss_fn_settings = loss_fn_settings
        self.loss_param_settings = loss_param_settings
        self.mt_size = mt_size

class MLPLayer(nn.Module):
    def __init__(self, input_size: int, output_size: int, 
        activate: bool = False, normalize: bool = False, dual: bool = False):
        super().__init__()
        self.linear = nn.Linear(input_size, output_size)
        self.dual = dual
        # self.linear1 = nn.Linear(output_size, output_size)
        self.activation = nn.Tanh()
        self.activate = activate
        self.normalization = nn.BatchNorm1d(input_size)
        self.normalize = normalize

    def forward(self, features:torch.Tensor, **kwargs) -> torch.Tensor:
        if self.normalize:
            features = self.normalization(features)

        if self.activate:
            features = self.activation(features)

        output_features = self.linear(features)
        
        return output_features

class InitAndForward:
    def _sub_init(self, config: Union[IBertConfig, IRobertaConfig], activate=False, dual=False):
        self.mlp4mt = MLPLayer(config.hidden_size, config.mt_size, 
            activate=activate, dual=dual)
        self.mlp = MLPLayer(config.hidden_size, config.feature_size, 
            activate=activate, dual=dual)

        self.similarity = lambda x,y: F.cosine_similarity(x, y, dim=-1)
        
        self.loss_logs = {'loss_all': [], 'loss1': [], 'loss2': [], 'loss3': [], 'loss4': [], 'mt_percentage': []}

    def _get_embedding(
        self, attention_mask: torch.Tensor, outputs: Dict[str, torch.Tensor],
        with_mlp: bool = True, other_mlp = None
    ) -> torch.Tensor:
        last_hidden: torch.Tensor = outputs.last_hidden_state # (bs, seq_len, hidden_len)
        hidden_states: torch.Tensor = outputs.hidden_states  # Tuple of (bs, seq_len, hidden_len)

        if 'cls' in self.config.pooler_type:
            cls_embedding = last_hidden[:, 0]
            if with_mlp:
                if other_mlp is not None:
                    return other_mlp(cls_embedding)
                return self.mlp(cls_embedding) #(bs, hidden_len)
            else:
                return cls_embedding
        elif 'avg' in self.config.pooler_type:
            return ((last_hidden * attention_mask.unsqueeze(-1)).sum(1) / 
                     attention_mask.sum(-1).unsqueeze(-1)) #(bs, hidden_len)
        elif 'mask' in self.config.pooler_type:
            mask_embedding = []
            sent_len = attention_mask.sum(dim=-1) 
            for idx in range(last_hidden.shape[0]):
                #NOTE: this is relevant to prompt
                mask_embedding.append(last_hidden[idx][sent_len[idx] - 3])
            mask_embedding = torch.stack(mask_embedding, dim=0)
            if with_mlp:
                if other_mlp is not None:
                    return other_mlp(mask_embedding)
                return self.mlp(mask_embedding) #(bs, hidden_len)
            else:
                return mask_embedding
        else:
            raise NotImplementedError

    def _organize_embedding(self, embedding, aigen_batch_size, aigen_sent_num, 
        other_batch_size, other_sent_num, pca):
        def _pca(matrix, pca):
            matrix_c = matrix - matrix.mean(dim=0)
            _, _, v = torch.pca_lowrank(matrix_c, q=min(pca, *matrix_c.shape))
            return matrix_c @ v
        aigen_h_embedding = embedding[:aigen_batch_size * aigen_sent_num].reshape(
            aigen_batch_size, aigen_sent_num, embedding.shape[-1]
        )
        other_h_embedding = embedding[aigen_batch_size * aigen_sent_num:].reshape(
            other_batch_size, other_sent_num, embedding.shape[-1]
        )

        aigen_pos_idx = 2 if aigen_sent_num > 3 else 1
        
        embeds_0 = torch.cat(
            ([other_h_embedding[:, 0]] if other_sent_num else []) + 
            ([aigen_h_embedding[:, 0]] if aigen_sent_num else []),
            dim=0
        ) # (obs + abs, hs)
        embeds_1 = torch.cat(
            ([other_h_embedding[:, 1]] if other_sent_num else []) +
            ([aigen_h_embedding[:, aigen_pos_idx]] if aigen_sent_num else []) +
            ([other_h_embedding[:, -1]] if other_sent_num > 2 else []) + 
            ([aigen_h_embedding[:, -1]] if aigen_sent_num > 2 else [])
            ,
            dim=0
        ) # (obs + abs + ..., hs)
        
        if pca > 0:
            embeds_0 = _pca(embeds_0, pca)
            embeds_1 = _pca(embeds_1, pca)
        
        embeds_0_norm = F.normalize(embeds_0, dim=-1)
        embeds_1_norm = F.normalize(embeds_1, dim=-1)

        return embeds_0, embeds_1, embeds_0_norm, embeds_1_norm

    def _i_forward(self, encoder, 
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        loss_pair=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None
    ) -> SequenceClassifierOutput:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        ori_input_ids = input_ids


        aigen_batch_size = input_ids[0].size(0)
        aigen_sent_num = input_ids[0].size(1)
        other_batch_size = input_ids[1].size(0)
        other_sent_num = input_ids[1].size(1)

        
        inp_input_ids = torch.cat([input_ids[0].reshape(-1, input_ids[0].shape[-1]), 
                                       input_ids[1].reshape(-1, input_ids[1].shape[-1])], dim=0) # shape of [abs * aigen_sent_num + obs * 2, seq_len]
        inp_attention_mask = torch.cat([attention_mask[0].reshape(-1, attention_mask[0].shape[-1]),
                                        attention_mask[1].reshape(-1, attention_mask[1].shape[-1])], dim=0)
        if token_type_ids is not None and token_type_ids[1] is not None:
            inp_token_type_ids = torch.cat([token_type_ids[0].reshape(-1, token_type_ids[0].shape[-1]),
                                            token_type_ids[1].reshape(-1, token_type_ids[1].shape[-1])], dim=0)
        else:
            inp_token_type_ids = None

        # Get raw embeddings
        outputs = encoder(
            inp_input_ids,
            attention_mask=inp_attention_mask,
            token_type_ids=inp_token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=True, # if cls.config.pooler_type in ['avg_top2', 'avg_first_last'] else False,
            return_dict=True,
        )

        embedding = self._get_embedding(inp_attention_mask, outputs, with_mlp=True)

        if dist.is_initialized() and self.training:
            raise NotImplementedError

        loss_fn_settings = self.config.loss_fn_settings.replace(' ', '').split('.')
        loss_params = {
            param.split('=')[0]: float(param.split('=')[1])
            for param in self.config.loss_param_settings.replace(' ', '').split(',')
        }

        embeds_0, embeds_1, embeds_0_norm, embeds_1_norm = self._organize_embedding(embedding,
            aigen_batch_size, aigen_sent_num, other_batch_size, other_sent_num,
            pca=(int(loss_params.get('pca', aigen_batch_size + other_batch_size)) 
                    if 'pca' in loss_fn_settings else 0)
        )
        similarity = self.similarity(embeds_0[:, None], embeds_1[None])

        eye = torch.eye(embeds_0.shape[0]).to(self.device)
        eye_hd = torch.eye(embeds_0.shape[1]).to(self.device)

        if 'infonce' in loss_fn_settings:
            loss_fn = nn.CrossEntropyLoss()

            labels = torch.arange(similarity.shape[0]).to(dtype=torch.long, device=self.device)
            loss = loss_fn(similarity / loss_params.get('temp', 1.), labels)

        elif 'arccon' in loss_fn_settings:
            loss_fn = nn.CrossEntropyLoss()

            align = torch.diag(similarity)
            theta = torch.arccos(align)
            similarity = torch.diagonal_scatter(similarity,  torch.cos(theta + loss_params.get('u', 10 / 180 * np.pi)))

            labels = torch.arange(similarity.shape[0]).to(dtype=torch.long, device=self.device)
            loss = loss_fn(similarity / loss_params.get('temp', 1.), labels)

        elif 'mpt' in loss_fn_settings:
            sim_matrix = embeds_0_norm @ embeds_1_norm.T
            sim_pos = torch.diag(sim_matrix)
            sim_neg = (sim_matrix - 1e8 * eye).max(dim=-1)[0]

            loss = F.relu(sim_neg - sim_pos + loss_params.get('m', 0.23)).mean()

        elif 'met' in loss_fn_settings:
            dis_matrix = torch.clamp_min(2 - 2 * embeds_0_norm @ embeds_1_norm.T, 1e-4).sqrt()
            dis_pos = torch.diag(dis_matrix)
            dis_neg = (dis_matrix + 1e8 * eye).min(dim=-1)[0]

            loss = F.relu(dis_pos - dis_neg + loss_params.get('m', 0.45)).mean()
        
        elif 'mat' in loss_fn_settings:
            angle_matrix = torch.arccos(embeds_0_norm @ embeds_1_norm.T)
            angle_pos = torch.diag(angle_matrix)
            angle_neg = (angle_matrix + 1e8 * eye).min(dim=-1)[0]

            loss = F.relu(angle_pos - angle_neg + loss_params.get('m', 0.15) * torch.pi ).mean()

        elif 'sg' in loss_fn_settings: # simulate gradient | w_gd
            _mask = embeds_0_norm.detach() @ embeds_1_norm.detach().T
            mask0 = (torch.diag(_mask) - (_mask + eye * -1e8 ).max(dim=-1)[0]) < loss_params.get('m', 0.30)
            mask1 = (torch.diag(_mask) - (_mask + eye * -1e8 ).max(dim= 0)[0]) < loss_params.get('m', 0.30)

            align = (embeds_0_norm * embeds_1_norm).sum(dim=-1)
            def __cal_loss4sg(embeds_row, embeds_col, mask=None):
                sim_matrix = embeds_row @ embeds_col.T
                sim_matrix = torch.diagonal_scatter(sim_matrix, align)
                target = (sim_matrix - loss_params.get('ratio', 1.) * torch.diag(sim_matrix)[:, None])

                if 'power' in loss_params:
                    if loss_params['power'] == 0.:
                        weight = torch.ones_like(eye)
                    else:
                        weight = torch.clamp_min(sim_matrix.detach(), 1e-4).pow(
                            int(loss_params.get('power', 1.)))
                    weight = weight * (eye == 0)
                    weight = weight / weight.sum(dim=-1, keepdim=True)
                else:
                    weight = (sim_matrix.detach() - 1e8 * eye).div(loss_params.get('temp', 1.))
                    weight = (weight - torch.logsumexp(weight, dim=-1, keepdim=True)).exp()

                if 'w_gd' in loss_fn_settings:
                    return (weight * target)[mask].mean()
                else:
                    return (weight * target).mean()

            loss = __cal_loss4sg(embeds_0_norm, embeds_1_norm.detach(), mask0) + \
                    __cal_loss4sg(embeds_1_norm, embeds_0_norm.detach(), mask1)
        
        else: # other | w_gd
            if 'w_gd' in loss_fn_settings:
                mask0 = embeds_0_norm.detach() @ embeds_0_norm.T.detach()
                mask0 = (torch.diag(mask0) - (mask0 - 1e8 * eye).max(dim=-1)[0]) < loss_params.get('m', 0.30)
                mask1 = embeds_1_norm.detach() @ embeds_1_norm.T.detach()
                mask1 = (torch.diag(mask1) - (mask1 - 1e8 * eye).max(dim=-1)[0]) < loss_params.get('m', 0.30)
                embeds_0_norm = embeds_0_norm * mask0[:, None] + embeds_0_norm.detach() * (mask0 == False)[:, None]
                embeds_1_norm = embeds_1_norm * mask1[:, None] + embeds_1_norm.detach() * (mask1 == False)[:, None]

            # align = (embeds_0_norm - embeds_1_norm).pow(2).sum(dim=-1)
            align = 2 - 2 * (embeds_0_norm * embeds_1_norm).sum(dim=-1)

            if 'mhe' in loss_fn_settings: # w_w, w_r
                sim_matrix0 = embeds_0_norm @ embeds_0_norm.T
                sim_matrix1 = embeds_1_norm @ embeds_1_norm.T

                if 'w_w' in loss_fn_settings:
                    sim_matrix0 = sim_matrix0 / loss_params.get('temp', 1.)
                    sim_matrix1 = sim_matrix1 / loss_params.get('temp', 1.)
                uniform0 = torch.logsumexp(
                    sim_matrix0[torch.triu(torch.ones_like(sim_matrix0)) == 0],
                dim=-1)
                uniform1 = torch.logsumexp(
                    sim_matrix1[torch.triu(torch.ones_like(sim_matrix1)) == 0],
                dim=-1)
                
                if 'w_w' in loss_fn_settings:
                    weight0 = (torch.logsumexp(sim_matrix0.detach() + -1e8 * eye, dim=-1) 
                                - uniform0.detach()).exp() / loss_params.get('temp', 1.)
                    weight1 = (torch.logsumexp(sim_matrix1.detach() + -1e8 * eye, dim=-1) 
                                - uniform1.detach()).exp() / loss_params.get('temp', 1.)
                else:
                    weight0 = 1 / embeds_0_norm.shape[0]
                    weight1 = 1 / embeds_1_norm.shape[0]
                align0 = align * weight0 / 2
                align1 = align * weight1 / 2

                if 'w_r' in loss_fn_settings:
                    align0 = align0 * loss_params.get('ratio', 1)
                    align1 = align1 * loss_params.get('ratio', 1)
                
                loss = (uniform0 + uniform1) * loss_params.get('lambda', 1) \
                    + align0.sum() + align1.sum()

            elif 'mhs' in loss_fn_settings: # w_r
                dis_matrix0 = (embeds_0_norm[:, None] - embeds_1_norm[None].detach()).pow(2).sum(dim=-1).add(1e-4).sqrt()
                dis_matrix1 = (embeds_1_norm[:, None] - embeds_0_norm[None].detach()).pow(2).sum(dim=-1).add(1e-4).sqrt()

                uniform0 = -(dis_matrix0 + 1e8 * eye).min(dim=-1)[0]
                uniform1 = -(dis_matrix1 + 1e8 * eye).min(dim=-1)[0]

                if 'w_r' in loss_fn_settings:
                    weight0 = (-1 / uniform0).detach()
                    weight1 = (-1 / uniform1).detach()
                    align0 = align * weight0 * loss_params.get('ratio', 1.) / 2
                    align1 = align * weight1 * loss_params.get('ratio', 1.) / 2
                else:
                    align0 = align / 2
                    align1 = align / 2
                
                loss = (uniform0.mean() + uniform1.mean()) * loss_params.get('lambda', 1) \
                    + align0.mean() + align1.mean()

            elif 'vicreg' in loss_fn_settings: # w_w, w_r
                assert embeds_0_norm.shape[0] == embeds_1_norm.shape[0]
                
                def _vicreg_loss(u1, u2):
                    N = u1.shape[0]
                    
                    if 'w_w' in loss_fn_settings:
                        if 'temp' in loss_params:
                            weight = (u1.detach() @ u1.detach().T
                                ).div(loss_params.get('temp', 1.)).exp()
                        else:
                            weight = torch.clamp_min(u1.detach() @ u1.detach().T, 
                                1e-4).pow(loss_params.get('power', 4))
                        weight = weight / (weight * (eye == 0)).sum()
                        # weight = (weight * (eye == 0)) / (weight * (eye == 0)).sum()
                    else:
                        weight = u1.detach() @ u1.detach().T / N
                        weight = weight * 2 / N
                    sim_matrix = u1 @ u1.detach().T # N x N
                    loss_neg = (weight * sim_matrix).sum(dim=-1)

                    if 'w_r' in loss_fn_settings:
                        align = -(u1 * u2).sum(dim=-1)
                        loss_pos = loss_params.get('lambda', 1.) * loss_params.get('ratio', 1.)\
                            * (weight * (eye == 0)).sum(dim=-1) * align
                    else:
                        loss_pos = -(u1 * u2.detach()).sum(dim=-1)
                        loss_pos = loss_pos * 2 / N

                    return (loss_pos + loss_params.get('lambda', 1.) * loss_neg).sum()

                loss = _vicreg_loss(embeds_0_norm, embeds_1_norm) \
                    + _vicreg_loss(embeds_1_norm, embeds_0_norm)

            elif 'barlowtwins' in loss_fn_settings: # w_w, w_r
                assert embeds_0_norm.shape[0] == embeds_1_norm.shape[0]

                def _bt_loss(u1, u2):
                    N = u1.shape[0]

                    if 'w_w' in loss_fn_settings:
                        if 'temp' in loss_params:
                            weight = (u2.detach() @ u2.detach().T
                                ).div(loss_params.get('temp', 1.)).exp()
                        else:
                            weight = torch.clamp_min(u2.detach() @ u2.detach().T, 
                                1e-4).pow(loss_params.get('power', 4))
                        weight = weight / (weight * (eye == 0)).sum()
                        # weight = (weight * (eye == 0)) / (weight * (eye == 0)).sum()
                    else:
                        weight = u2.detach() @ u2.detach().T / N # N x N
                        weight = weight * 2 / N
                    sim_matrix = u1 @ u1.detach().T # N x N
                    loss_neg = (weight * sim_matrix).sum(dim=-1)

                    if 'w_r' in loss_fn_settings:
                        align = -(u1 * u2).sum(dim=-1)
                        loss_pos = loss_params.get('lambda', 1.) * loss_params.get('ratio', 1.)\
                            * (weight * (eye == 0)).sum(dim=-1) * align
                    else:
                        W_cor = u1.detach().T @ u2.detach() / N # D x D
                        pos = (1 - (1 - loss_params.get('lambda', 1.)) 
                            * torch.diag(W_cor)) * 2 / N # D
                        align = -(u1 * u2.detach())
                        loss_pos = (pos * align).sum(dim=-1)

                    return (loss_pos + loss_params.get('lambda', 1.) * loss_neg).sum()

                loss = _bt_loss(embeds_0_norm, embeds_1_norm) \
                    + _bt_loss(embeds_1_norm, embeds_0_norm)


        self.loss_logs['loss_all'].append(loss.item())
                    
        if not return_dict:
            output = (similarity,) + \
                (outputs.get("hidden_states", None), outputs.get("attentions", None))
            return ((loss,) + output) if loss is not None else output
        return SequenceClassifierOutput(
            loss=loss,
            logits=similarity,
            hidden_states=outputs.get("hidden_states", None),
            attentions=outputs.get("attentions", None),
        )

    def _sentemb_forward(self, encoder, 
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        loss_pair=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None
    ) -> BaseModelOutputWithPoolingAndCrossAttentions:
        return_dict: bool = return_dict if return_dict is not None else self.config.use_return_dict

        if isinstance(input_ids, list):
            input_ids = torch.cat([input_ids[0].reshape(-1, input_ids[0].shape[-1]), input_ids[1][:, 0]], dim=0)
            attention_mask = torch.cat([attention_mask[0].reshape(-1, attention_mask[0].shape[-1]), attention_mask[1][:, 0]], dim=0)
            if token_type_ids is not None:
                token_type_ids = torch.cat([token_type_ids[0].reshape(-1, token_type_ids[0].shape[-1]), token_type_ids[1][:, 0]], dim=0)
        
        paired_data: bool = len(input_ids.shape) == 3

        if paired_data:
            batch_size = input_ids.size(0)
            # Number of sentences in one instance
            # 2: pair instance; 3: pair instance with a hard negative
            num_sent = input_ids.size(1)

            # Flatten input for encoding
            input_ids = input_ids.view((-1, input_ids.size(-1))) # (bs * num_sent, len)
            attention_mask = attention_mask.view((-1, attention_mask.size(-1))) # (bs * num_sent, len)
            if token_type_ids is not None:
                token_type_ids = token_type_ids.view((-1, token_type_ids.size(-1))) # (bs * num_sent, len)
        
        # Get outputs
        with torch.no_grad():
            outputs = encoder(
                input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=True, # if cls.config.pooler_type in ['avg_top2', 'avg_first_last'] else False,
                return_dict=True,
            )
        
        # Get embeddings
        embedding: torch.Tensor = self._get_embedding(attention_mask, outputs, with_mlp=False)
        if paired_data:
            embedding = embedding.view(batch_size, num_sent, embedding.shape[-1])

        if not return_dict:
            return (outputs.last_hidden_state, embedding, outputs.hidden_states)
        
        return BaseModelOutputWithPoolingAndCrossAttentions(
            pooler_output=embedding,
            last_hidden_state=outputs.last_hidden_state,
            hidden_states=outputs.hidden_states
        )

    def custom_param_init(self):
        if not dist.is_initialized() or dist.get_rank() == 0:
            logger.info(f'{self.config.num_layers} layers of Bert/Roberta are used for trainning.')
        unfreeze_layers = ['pooler'] + [f'layer.{11 - i}' for i in range(self.config.num_layers)]
        encoder = self.bert if hasattr(self, 'bert') else self.roberta
        for name, param in encoder.named_parameters():
            param.requires_grad = False
            for ele in unfreeze_layers:
                if ele in name:
                    param.requires_grad = True
                    break
        
        nn.init.normal_(self.mlp4mt.linear.weight, 0, 6e-3)
        nn.init.normal_(self.mlp4mt.linear.bias, 0, 6e-3)
        nn.init.normal_(self.mlp.linear.weight, 0, 6e-3)
        nn.init.normal_(self.mlp.linear.bias, 0, 6e-3)

        self.named_parameters_cache = {k: v.detach().clone() for k, v in self.named_parameters() 
                                       if 'bert' in k and v.requires_grad == True}
    
    def floating_point_ops(
        self, input_dict: Dict[str, Union[torch.Tensor, Any]], exclude_embeddings: bool = True
    ) -> int:
        def _fpo(
            input_dict: Dict[str, Union[torch.Tensor, Any]], exclude_embeddings: bool = True
        ) -> int:
            return 6 * self.estimate_tokens(input_dict) * self.num_parameters(exclude_embeddings=exclude_embeddings)
        
        total_fpo = 0.
        list_len = 0
        for v in input_dict.values():
            if isinstance(v, list):
                list_len = len(v)
                break
        
        if list_len:
            for i in range(list_len):
                tmp_inp = {k: (v[i] if isinstance(v, list) else v) for k, v in input_dict.items()}
                total_fpo += _fpo(tmp_inp, exclude_embeddings)
        else:
            total_fpo = _fpo(input_dict, exclude_embeddings)
        
        return total_fpo

class IBert(InitAndForward, BertPreTrainedModel):
    _keys_to_ignore_on_load_unexpected = ['pooler', 'cls']
    config_class = IBertConfig

    def __init__(self, config, *model_args, **model_kwargs):
        super().__init__(config)
        # self.model_args = model_kwargs["model_args"]
        self.bert = BertModel(config, add_pooling_layer=False)

        # modify `_get_embedding` after modifying this
        self.prompt = [
            {
                'input_ids': [torch.tensor([2023, 6251, 1024, 1000]), torch.tensor([1000, 2965, 103, 1012])],
                'token_type_ids': [torch.tensor([0, 0, 0, 0]), torch.tensor([0, 0, 0, 0])],
                'attention_mask': [torch.tensor([1, 1, 1, 1]), torch.tensor([1, 1, 1, 1])]
            },
            {
                'input_ids': [torch.tensor([2023, 6251, 1997, 1000]), torch.tensor([1000, 2965, 103, 1012])],
                'token_type_ids': [torch.tensor([0, 0, 0, 0]), torch.tensor([0, 0, 0, 0])],
                'attention_mask': [torch.tensor([1, 1, 1, 1]), torch.tensor([1, 1, 1, 0])]
            },
        ]

        self._sub_init(config)

    def add_prompt(self, input_ids, token_type_ids, attention_mask):
        def _add_prompt(prompt, input_ids, token_type_ids, attention_mask):
            sent_len = attention_mask.sum(dim=-1)
            n_input_ids = []
            n_token_type_ids = []
            n_attention_mask = []
            for s_idx in range(input_ids.shape[0]):
                # [cls] + prompt_pre + sent + prompt_post + [sep]
                n_input_ids.append(torch.cat([
                    input_ids[s_idx][:1], 
                    prompt['input_ids'][0].to(input_ids[s_idx][:1]), 
                    input_ids[s_idx][1:sent_len[s_idx] - 1], 
                    prompt['input_ids'][1].to(input_ids[s_idx][:1]), 
                    input_ids[s_idx][sent_len[s_idx] - 1: ]
                ], dim=0))
                n_token_type_ids.append(torch.cat([
                    token_type_ids[s_idx][:1], 
                    prompt['token_type_ids'][0].to(token_type_ids[s_idx][:1]), 
                    token_type_ids[s_idx][1:sent_len[s_idx] - 1], 
                    prompt['token_type_ids'][1].to(token_type_ids[s_idx][:1]), 
                    token_type_ids[s_idx][sent_len[s_idx] - 1: ]
                ], dim=0))
                n_attention_mask.append(torch.cat([
                    attention_mask[s_idx][:1], 
                    prompt['attention_mask'][0].to(attention_mask[s_idx][:1]),
                    attention_mask[s_idx][1:sent_len[s_idx] - 1], 
                    prompt['attention_mask'][1].to(attention_mask[s_idx][:1]), 
                    attention_mask[s_idx][sent_len[s_idx] - 1: ]
                ], dim=0))

            n_input_ids = torch.stack(n_input_ids, dim=0)
            n_token_type_ids = torch.stack(n_token_type_ids, dim=0)
            n_attention_mask = torch.stack(n_attention_mask, dim=0)
            return n_input_ids, n_token_type_ids, n_attention_mask
        
        if isinstance(input_ids, list):

            if input_ids[0].shape[0] > 0:
                n_input_ids0 = []
                n_token_type_ids0 = []
                n_attention_mask0 = []
                for idx1 in range(input_ids[0].shape[1]):
                    r_input_ids, r_token_type_ids, r_attention_mask = _add_prompt(
                        self.prompt[0], input_ids[0][:, idx1], 
                        token_type_ids[0][:, idx1], attention_mask[0][:, idx1]
                    )
                    n_input_ids0.append(r_input_ids)
                    n_token_type_ids0.append(r_token_type_ids)
                    n_attention_mask0.append(r_attention_mask)
                input_ids[0] = torch.stack(n_input_ids0, dim=1)
                token_type_ids[0] = torch.stack(n_token_type_ids0, dim=1)
                attention_mask[0] = torch.stack(n_attention_mask0, dim=1)

            if input_ids[1].shape[0] > 0:
                n_input_ids1 = []
                n_token_type_ids1 = []
                n_attention_mask1 = []
                for idx1 in range(input_ids[1].shape[1]):
                    r_input_ids, r_token_type_ids, r_attention_mask = _add_prompt(
                        self.prompt[idx1 if input_ids[1].shape[1] == 2 else 0], 
                        input_ids[1][:, idx1], token_type_ids[1][:, idx1], attention_mask[1][:, idx1]
                    )
                    n_input_ids1.append(r_input_ids)
                    n_token_type_ids1.append(r_token_type_ids)
                    n_attention_mask1.append(r_attention_mask)
                input_ids[1] = torch.stack(n_input_ids1, dim=1)
                token_type_ids[1] = torch.stack(n_token_type_ids1, dim=1)
                attention_mask[1] = torch.stack(n_attention_mask1, dim=1)
            
            if input_ids[0].shape[-1] != input_ids[1].shape[-1]:
                if input_ids[0].shape[0] == 0:
                    input_ids[0] = input_ids[0].reshape(*(input_ids[0].shape[:-1] + input_ids[1].shape[-1:]))
                    token_type_ids[0] = token_type_ids[0].reshape(*(token_type_ids[0].shape[:-1] + token_type_ids[1].shape[-1:]))
                    attention_mask[0] = attention_mask[0].reshape(*(attention_mask[0].shape[:-1] + attention_mask[1].shape[-1:]))
                elif input_ids[1].shape[0] == 0:
                    input_ids[1] = input_ids[1].reshape(*(input_ids[1].shape[:-1] + input_ids[0].shape[-1:]))
                    token_type_ids[1] = token_type_ids[1].reshape(*(token_type_ids[1].shape[:-1] + token_type_ids[0].shape[-1:]))
                    attention_mask[1] = attention_mask[1].reshape(*(attention_mask[1].shape[:-1] + attention_mask[0].shape[-1:]))
                else:
                    raise Exception

        else:

            if len(input_ids.shape) == 3:
                n_input_ids = []
                n_token_type_ids = []
                n_attention_mask = []

                for idx1 in range(input_ids.shape[1]):
                    r_input_ids, r_token_type_ids, r_attention_mask = _add_prompt(
                        self.prompt[idx1 % 2], input_ids[:, idx1], 
                        token_type_ids[:, idx1], attention_mask[:, idx1]
                    )
                    n_input_ids.append(r_input_ids)
                    n_token_type_ids.append(r_token_type_ids)
                    n_attention_mask.append(r_attention_mask)
                
                input_ids = torch.stack(n_input_ids, dim=1)
                token_type_ids = torch.stack(n_token_type_ids, dim=1)
                attention_mask = torch.stack(n_attention_mask, dim=1)

            else:
                input_ids, token_type_ids, attention_mask = _add_prompt(
                    self.prompt[0], input_ids, token_type_ids, attention_mask
                )

        return input_ids, token_type_ids, attention_mask

    def forward(self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            loss_pair=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            sent_emb=False
    ) -> Union[Tuple, Dict[str, Any]]:
        if 'mask' in self.config.pooler_type:
            input_ids, token_type_ids, attention_mask = self.add_prompt(input_ids, token_type_ids, attention_mask)

        forward_fn = self._sentemb_forward if sent_emb else self._i_forward

        return forward_fn(self.bert, 
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                labels=labels,
                loss_pair=loss_pair,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
        )

class IRoberta(InitAndForward, RobertaPreTrainedModel):
    _keys_to_ignore_on_load_unexpected = ['pooler', 'cls']
    config_class = IRobertaConfig

    def __init__(self, config, *model_args, **model_kwargs):
        super().__init__(config)
        # self.model_args = model_kwargs["model_args"]
        self.roberta = RobertaModel(config, add_pooling_layer=False)
        
        # modify `_get_embedding` after modifying this
        self.prompt = [
            { # This sentence : "[X]" means <mask> .
                'input_ids': [torch.tensor([713, 3645, 4832, 22]), torch.tensor([113, 839, 50264, 479])],
                'attention_mask': [torch.tensor([1, 1, 1, 1]), torch.tensor([1, 1, 1, 1])]
            },
            { # This sentence of "X" means <mask> .
                'input_ids': [torch.tensor([713, 3645, 9, 22]), torch.tensor([113, 839, 50264, 479])],
                'attention_mask': [torch.tensor([1, 1, 1, 1]), torch.tensor([1, 1, 1, 1])]
            }
        ]

        self._sub_init(config, activate=True)
        # self.init_weights()
    
    def add_prompt(self, input_ids, token_type_ids, attention_mask):
        def _add_prompt(prompt, input_ids, attention_mask):
            sent_len = attention_mask.sum(dim=-1)
            n_input_ids = []
            n_attention_mask = []
            for s_idx in range(input_ids.shape[0]):
                # [cls] + prompt_pre + sent + prompt_post + [sep]
                n_input_ids.append(torch.cat([
                    input_ids[s_idx][:1], 
                    prompt['input_ids'][0].to(input_ids[s_idx][:1]), 
                    input_ids[s_idx][1:sent_len[s_idx] - 1], 
                    prompt['input_ids'][1].to(input_ids[s_idx][:1]), 
                    input_ids[s_idx][sent_len[s_idx] - 1: ]
                ], dim=0))
                n_attention_mask.append(torch.cat([
                    attention_mask[s_idx][:1], 
                    prompt['attention_mask'][0].to(attention_mask[s_idx][:1]),
                    attention_mask[s_idx][1:sent_len[s_idx] - 1], 
                    prompt['attention_mask'][1].to(attention_mask[s_idx][:1]), 
                    attention_mask[s_idx][sent_len[s_idx] - 1: ]
                ], dim=0))
            n_input_ids = torch.stack(n_input_ids, dim=0)
            n_attention_mask = torch.stack(n_attention_mask, dim=0)
            return n_input_ids, n_attention_mask
        
        if isinstance(input_ids, list):

            if input_ids[0].shape[0] > 0:
                n_input_ids0 = []
                n_attention_mask0 = []
                for idx1 in range(input_ids[0].shape[1]):
                    r_input_ids, r_attention_mask = _add_prompt(
                        self.prompt[0], input_ids[0][:, idx1], attention_mask[0][:, idx1]
                    )
                    n_input_ids0.append(r_input_ids)
                    n_attention_mask0.append(r_attention_mask)
                input_ids[0] = torch.stack(n_input_ids0, dim=1)
                attention_mask[0] = torch.stack(n_attention_mask0, dim=1)

            if input_ids[1].shape[0] > 0:
                n_input_ids1 = []
                n_attention_mask1 = []
                for idx1 in range(input_ids[1].shape[1]):
                    r_input_ids, r_attention_mask = _add_prompt(
                        self.prompt[idx1 if input_ids[1].shape[1] == 2 else 0], 
                        input_ids[1][:, idx1], attention_mask[1][:, idx1]
                    )
                    n_input_ids1.append(r_input_ids)
                    n_attention_mask1.append(r_attention_mask)
                input_ids[1] = torch.stack(n_input_ids1, dim=1)
                attention_mask[1] = torch.stack(n_attention_mask1, dim=1)
            
            if input_ids[0].shape[-1] != input_ids[1].shape[-1]:
                if input_ids[0].shape[0] == 0:
                    input_ids[0] = input_ids[0].reshape(*(input_ids[0].shape[:-1] + input_ids[1].shape[-1:]))
                    attention_mask[0] = attention_mask[0].reshape(*(attention_mask[0].shape[:-1] + attention_mask[1].shape[-1:]))
                elif input_ids[1].shape[0] == 0:
                    input_ids[1] = input_ids[1].reshape(*(input_ids[1].shape[:-1] + input_ids[0].shape[-1:]))
                    attention_mask[1] = attention_mask[1].reshape(*(attention_mask[1].shape[:-1] + attention_mask[0].shape[-1:]))
                else:
                    raise Exception

        else:

            if len(input_ids.shape) == 3:
                n_input_ids = []
                n_attention_mask = []

                for idx1 in range(input_ids.shape[1]):
                    r_input_ids, r_attention_mask = _add_prompt(
                        self.prompt[idx1 % 2], input_ids[:, idx1], attention_mask[:, idx1]
                    )
                    n_input_ids.append(r_input_ids)
                    n_attention_mask.append(r_attention_mask)
                input_ids = torch.stack(n_input_ids, dim=1)
                attention_mask = torch.stack(n_attention_mask, dim=1)

            else:
                input_ids, attention_mask = _add_prompt(
                    self.prompt[0], input_ids, attention_mask
                )

        return input_ids, token_type_ids, attention_mask

    def forward(self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            loss_pair=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            sent_emb=False
    ) -> Union[Tuple, Dict[str, Any]]:
        if 'mask' in self.config.pooler_type:
            input_ids, token_type_ids, attention_mask = self.add_prompt(input_ids, token_type_ids, attention_mask)

        forward_fn = self._sentemb_forward if sent_emb else self._i_forward

        return forward_fn(self.roberta, 
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                labels=labels,
                loss_pair=loss_pair,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
        )
