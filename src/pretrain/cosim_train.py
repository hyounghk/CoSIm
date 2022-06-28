# coding=utf-8
# Copyleft 2019 project LXRT.

import collections
import os, sys
import random

from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from param import args
from pretrain.lxmert_data import InputExample, LXMERTDataset, LXMERTTorchDataset, LXMERTEvaluator
from lxrt.entry import set_visual_config
from lxrt.tokenization import BertTokenizer
from lxrt.modeling import LXRTPretraining
from torch.nn import CrossEntropyLoss
from itertools import chain

import json

DataTuple = collections.namedtuple("DataTuple", 'dataset torchdset loader evaluator')


def get_tuple(splits: str, bs: int, shuffle=False, drop_last=False, topk=-1) -> DataTuple:


    # Build dataset, data loader, and evaluator.
    print(splits)
    dset = LXMERTDataset(splits)
    tset = LXMERTTorchDataset(dset, topk)
    data_loader = DataLoader(
        tset, batch_size=bs,
        shuffle=shuffle, num_workers=args.num_workers,
        collate_fn=lambda x: x,
        drop_last=drop_last, pin_memory=True
    )
    evaluator = LXMERTEvaluator(dset)
    print()

    return DataTuple(dataset=dset, torchdset=tset, loader=data_loader, evaluator=evaluator)

print(args.train)
if not args.testing:
    train_tuple = get_tuple(args.train, args.batch_size, shuffle=True, drop_last=True)
    valid_batch_size = args.batch_size 
    valid_tuple = get_tuple(args.valid, valid_batch_size, shuffle=False, drop_last=False)
else:
    test_tuple = get_tuple(args.test, args.batch_size, shuffle=False, drop_last=False)


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                 input_ids1, input_ids2, input_ids3, input_ids4,
                 input_mask1, input_mask2, input_mask3, input_mask4, 
                 segment_ids1, segment_ids2, segment_ids3, segment_ids4,
                 visual_feats, visual_feats_mask, obj_labels, ans, uid):
        self.input_ids1 = input_ids1
        self.input_ids2 = input_ids2
        self.input_ids3 = input_ids3
        self.input_ids4 = input_ids4
        self.input_mask1 = input_mask1
        self.input_mask2 = input_mask2
        self.input_mask3 = input_mask3
        self.input_mask4 = input_mask4
        self.segment_ids1 = segment_ids1
        self.segment_ids2 = segment_ids2
        self.segment_ids3 = segment_ids3
        self.segment_ids4 = segment_ids4
        self.visual_feats = visual_feats
        self.visual_feats_mask = visual_feats_mask
        self.obj_labels = obj_labels
        self.ans = ans
        self.uid = uid




def random_feat(feats):
    mask_feats = feats.copy()
    feat_mask = np.zeros(len(feats), dtype=np.float32)
    for i in range(len(feats)):
        prob = random.random()
        # mask token with probability
        if prob < args.obj_mask_rate:
            prob /= args.obj_mask_rate

            # 80% randomly change token to zero feat
            if prob < 0.8:
                mask_feats[i, :] = 0.

            # 10% randomly change token to random feat
            elif prob < 0.9:
                mask_feats[i, :] = train_tuple.torchdset.random_feat()
            # -> rest 10% randomly keep current feat

            # Need to predict this feat
            feat_mask[i] = 1.

    return mask_feats, feat_mask


def feat_masking(feat, boxes):
    new_feat = np.zeros((37,2048), dtype=np.float32)
    # new_boxes = np.ones((37,4), dtype=np.float32)
    new_boxes = np.zeros((37,4), dtype=np.float32)
    # feat_mask = np.ones(37, dtype=np.float32)
    feat_mask = np.zeros(37, dtype=np.float32)
    num_obj = feat.shape[0]
    new_feat[1:num_obj+1] = feat
    new_boxes[1:num_obj+1] = boxes
    feat_mask[:num_obj+1] = 1
    

    return new_feat, new_boxes, feat_mask


def convert_example_to_features(example: InputExample, max_seq_length, tokenizer)->InputFeatures:
    """
    Convert a raw sample (pair of sentences as tokenized strings) into a proper training sample with
    IDs, LM labels, input_mask, CLS and SEP tokens etc.
    :param example: InputExample, containing sentence input as strings and is_next label
    :param max_seq_length: int, maximum length of sequence.
    :param tokenizer: Tokenizer
    :return: InputFeatures, containing all inputs and labels of one sample as IDs (as used for model training)
    """
    tokens1 = tokenizer.tokenize(example.sent[0].strip())
    tokens2 = tokenizer.tokenize(example.sent[1].strip())
    tokens3 = tokenizer.tokenize(example.sent[2].strip())
    tokens4 = tokenizer.tokenize(example.sent[3].strip())

    # Account for [CLS] and [SEP] with "- 2"


    if len(tokens1) > max_seq_length - 2:
        tokens1 = tokens1[-(max_seq_length - 2):]
    if len(tokens2) > max_seq_length - 2:
        tokens2 = tokens2[-(max_seq_length - 2):]
    if len(tokens3) > max_seq_length - 2:
        tokens3 = tokens3[-(max_seq_length - 2):]
    if len(tokens4) > max_seq_length - 2:
        tokens4 = tokens4[-(max_seq_length - 2):]


    tokens1 = ['[CLS]'] + tokens1 + ['[SEP]']
    tokens2 = ['[CLS]'] + tokens2 + ['[SEP]']
    tokens3 = ['[CLS]'] + tokens3 + ['[SEP]']
    tokens4 = ['[CLS]'] + tokens4 + ['[SEP]']
    input_ids1 = tokenizer.convert_tokens_to_ids(tokens1)
    input_ids2 = tokenizer.convert_tokens_to_ids(tokens2)
    input_ids3 = tokenizer.convert_tokens_to_ids(tokens3)
    input_ids4 = tokenizer.convert_tokens_to_ids(tokens4)

    input_mask1 = [1] * len(input_ids1)
    input_mask2 = [1] * len(input_ids2)
    input_mask3 = [1] * len(input_ids3)
    input_mask4 = [1] * len(input_ids4)

    segment_ids1 = []
    segment_ids2 = []
    segment_ids3 = []
    segment_ids4 = []

    seg_id = 0
    for i, token_split in enumerate(input_ids1):
        if token_split == 102:
            seg_id += 1
            if seg_id == 3:
                break
        segment_ids1.append(seg_id)

    seg_id = 0
    for i, token_split in enumerate(input_ids2):
        if token_split == 102:
            seg_id += 1
            if seg_id == 3:
                break
        segment_ids2.append(seg_id)

    seg_id = 0
    for i, token_split in enumerate(input_ids3):
        if token_split == 102:
            seg_id += 1
            if seg_id == 3:
                break
        segment_ids3.append(seg_id)

    seg_id = 0
    for i, token_split in enumerate(input_ids4):
        if token_split == 102:
            seg_id += 1
            if seg_id == 3:
                break
        segment_ids4.append(seg_id)


    segment_ids1 += [3] * (len(input_ids1) - len(segment_ids1))
    segment_ids2 += [3] * (len(input_ids2) - len(segment_ids2))
    segment_ids3 += [3] * (len(input_ids3) - len(segment_ids3))
    segment_ids4 += [3] * (len(input_ids4) - len(segment_ids4))


    while len(input_ids1) < max_seq_length:
        input_ids1.append(0)
        input_mask1.append(0)
        segment_ids1.append(0)

    while len(input_ids2) < max_seq_length:
        input_ids2.append(0)
        input_mask2.append(0)
        segment_ids2.append(0)

    while len(input_ids3) < max_seq_length:
        input_ids3.append(0)
        input_mask3.append(0)
        segment_ids3.append(0)

    while len(input_ids4) < max_seq_length:
        input_ids4.append(0)
        input_mask4.append(0)
        segment_ids4.append(0)


    assert len(input_ids1) == max_seq_length
    assert len(input_mask1) == max_seq_length
    assert len(segment_ids1) == max_seq_length


    feat, boxes = example.visual_feats
    obj_labels = example.obj_labels

    # Mask Image Features:
    feat, boxes, feat_mask = feat_masking(feat, boxes)


    features = InputFeatures(
        input_ids1=input_ids1,
        input_ids2=input_ids2,
        input_ids3=input_ids3,
        input_ids4=input_ids4,
        input_mask1=input_mask1,
        input_mask2=input_mask2,
        input_mask3=input_mask3,
        input_mask4=input_mask4,
        segment_ids1=segment_ids1,
        segment_ids2=segment_ids2,
        segment_ids3=segment_ids3,
        segment_ids4=segment_ids4,
        visual_feats=(feat, boxes),
        visual_feats_mask=feat_mask,
        obj_labels={'obj':obj_labels},
        ans=example.label,
        uid=example.uid,
    )
    return features


LOSSES_NAME = ['CoSIm']
loss_fct = CrossEntropyLoss(ignore_index=-1)

class LositScale(nn.Module):
    def __init__(self):
        super().__init__()
        self.logit_scale = nn.Parameter(torch.ones([])).cuda()

    def forward(self, x):
        return self.logit_scale * x

class LXMERT:
    def __init__(self, max_seq_length):
        super().__init__()
        self.max_seq_length = max_seq_length

        self.tokenizer = BertTokenizer.from_pretrained(
            "bert-base-uncased",
            do_lower_case=True
        )

        # Build model
        set_visual_config(args)
        self.model = LXRTPretraining.from_pretrained(
            "bert-base-uncased",
            task_mask_lm=args.task_mask_lm,
            task_obj_predict=args.task_obj_predict,
            task_matched=args.task_matched,
            task_qa=args.task_qa,
            visual_losses=args.visual_losses,
        )

        # Weight initialization and loading
        if args.from_scratch:
            print("Train from Scratch: re-initialize all BERT weights.")
            self.model.apply(self.model.init_bert_weights)
        if args.load is not None:
            self.load(args.load)
        if args.load_lxmert is not None:
            # Load lxmert would not load the answer head.
            self.load_lxmert(args.load_lxmert)

        self.LositScale = LositScale()
        

        # GPU Options
        self.model = self.model.cuda()
        if args.multiGPU:
            self.model = nn.DataParallel(self.model)


        print("num of model para: " ,self.count_parameters(self.model))

    def forward(self, examples):
        train_features = [convert_example_to_features(example, self.max_seq_length, self.tokenizer)
                          for example in examples]

        # language Inputs

        input_ids1 = torch.tensor([f.input_ids1 for f in train_features], dtype=torch.long).cuda()
        input_ids2 = torch.tensor([f.input_ids2 for f in train_features], dtype=torch.long).cuda()
        input_ids3 = torch.tensor([f.input_ids3 for f in train_features], dtype=torch.long).cuda()
        input_ids4 = torch.tensor([f.input_ids4 for f in train_features], dtype=torch.long).cuda()
        input_mask1 = torch.tensor([f.input_mask1 for f in train_features], dtype=torch.long).cuda()
        input_mask2 = torch.tensor([f.input_mask2 for f in train_features], dtype=torch.long).cuda()
        input_mask3 = torch.tensor([f.input_mask3 for f in train_features], dtype=torch.long).cuda()
        input_mask4 = torch.tensor([f.input_mask4 for f in train_features], dtype=torch.long).cuda()
        segment_ids1 = torch.tensor([f.segment_ids1 for f in train_features], dtype=torch.long).cuda()
        segment_ids2 = torch.tensor([f.segment_ids2 for f in train_features], dtype=torch.long).cuda()
        segment_ids3 = torch.tensor([f.segment_ids3 for f in train_features], dtype=torch.long).cuda()
        segment_ids4 = torch.tensor([f.segment_ids4 for f in train_features], dtype=torch.long).cuda()

        # Visual Inputs
        feats = torch.from_numpy(np.stack([f.visual_feats[0] for f in train_features])).cuda()
        feats_mask = torch.from_numpy(np.stack([f.visual_feats_mask for f in train_features])).cuda()
        pos = torch.from_numpy(np.stack([f.visual_feats[1] for f in train_features])).cuda()


        ans = torch.from_numpy(np.stack([f.ans for f in train_features])).cuda()
        uid = [f.uid for f in train_features]

        input_ids = torch.cat([input_ids1.unsqueeze(1), input_ids2.unsqueeze(1), input_ids3.unsqueeze(1), input_ids4.unsqueeze(1)], dim=1)
        input_mask = torch.cat([input_mask1.unsqueeze(1), input_mask2.unsqueeze(1), input_mask3.unsqueeze(1), input_mask4.unsqueeze(1)], dim=1)
        segment_ids = torch.cat([segment_ids1.unsqueeze(1), segment_ids2.unsqueeze(1), segment_ids3.unsqueeze(1), segment_ids4.unsqueeze(1)], dim=1)

        bsz = input_ids.size(0)
        input_ids = input_ids.view(bsz * 4, -1)
        input_mask = input_mask.view(bsz * 4, -1)
        segment_ids = segment_ids.view(bsz * 4, -1)

        feats = feats.unsqueeze(1).repeat(1, 4, 1, 1).view(bsz * 4, 37, -1)
        feats_mask = feats_mask.unsqueeze(1).repeat(1, 4, 1, 1).view(bsz * 4, 37)
        pos = pos.unsqueeze(1).repeat(1, 4, 1, 1).view(bsz * 4, 37, -1)

        ans_logit, lang_output, visn_output = self.model(
            input_ids, segment_ids, input_mask, None,
            feats, feats_mask, pos, None, None, ans
        )
        return ans_logit, ans, uid, lang_output, visn_output

    def cal_score_mat(self, feat1, feat2):
        feat1 = feat1 / (feat1.norm(dim=-1, keepdim=True) + 0.0000001)
        feat2 = feat2 / (feat2.norm(dim=-1, keepdim=True) + 0.0000001)
        # logit_scale = self.logit_scale.exp()
        scors1 = self.LositScale(feat1 @ feat2.transpose(0,1))
        scors2 = self.LositScale(feat2 @ feat1.transpose(0,1))

        return scors1, scors2

    def train_batch(self, optim, batch):
        optim.zero_grad()
        ans_logit, ans, uid, lang_output, visn_output = self.forward(batch)

        loss = loss_fct(ans_logit.view(-1, 4), ans.view(-1)) 

        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), 1.)
        optim.step()

        return loss.item(), ans_logit, ans, uid

    def valid_batch(self, batch):
        with torch.no_grad():
            ans_logit, ans, uid, lang_output, visn_output = self.forward(batch)
            loss = loss_fct(ans_logit.view(-1, 4), ans.view(-1)) 
            
        return loss.item(), ans_logit, ans, uid

    def train(self, train_tuple: DataTuple, eval_tuple: DataTuple):
        train_ld = train_tuple.loader

        # Optimizer
        from lxrt.optimization import BertAdam
        batch_per_epoch = len(train_ld)
        t_total = int(batch_per_epoch * args.epochs)
        warmup_ratio = 0.05
        warmup_iters = int(t_total * warmup_ratio)
        print("Batch per epoch: %d" % batch_per_epoch)
        print("Total Iters: %d" % t_total)
        print("Warm up Iters: %d" % warmup_iters)
        optim = BertAdam(chain(self.model.parameters(),self.LositScale.parameters()), lr=0.00001, warmup=warmup_ratio, t_total=t_total)

        # Train
        best_eval_loss = 9595.
        best_eval_accu = 0.
        for epoch in range(args.epochs):
            # Train
            self.model.train()
            total_loss = 0.
            accu_cnt = 0.
            accu_cnt_total = 0.
            for batch in tqdm(train_ld, total=len(train_ld)):
                loss, logit, ans, uid = self.train_batch(optim, batch)
                total_loss += loss
                score, label = logit.max(1)
                accu_cnt_total += logit.size(0)
                accu_cnt += (label == ans).float().sum()


            print("The training loss for Epoch %d is %0.4f" % (epoch, total_loss / accu_cnt_total))
            print("Accuracy: ", accu_cnt/accu_cnt_total)

            # Eval
            avg_eval_loss, avg_eval_accu = self.evaluate_epoch(eval_tuple, iters=-1)

            # Save
            if avg_eval_loss < best_eval_loss:
                best_eval_loss = avg_eval_loss
                self.save("BEST_EVAL_LOSS")

            if avg_eval_accu > best_eval_accu:
                best_eval_accu = avg_eval_accu


            self.save("Epoch%02d" % (epoch+1) + "_" + args.savename)
            print("Best Accuracy: ", best_eval_accu)

    def test(self, eval_tuple: DataTuple):
        
        self.evaluate_epoch(eval_tuple, iters=-1)

    def count_parameters(self, model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    def evaluate_epoch(self, eval_tuple: DataTuple, iters: int=-1):
        self.model.eval()
        eval_ld = eval_tuple.loader
        total_loss = 0.
        accu_cnt = 0.
        accu_cnt_total = 0.
        answer_record = {}
        for i, batch in enumerate(eval_ld):
            loss, logit, ans, uid = self.valid_batch(batch)
            total_loss += loss
            score, label = logit.max(1)
            accu_cnt_total += logit.size(0)
            accu_cnt += (label == ans).float().sum()

        print("The valid loss is %0.4f" % (total_loss / accu_cnt_total))


        print("Accuracy: ", accu_cnt/accu_cnt_total)

        return total_loss / len(eval_ld), accu_cnt/accu_cnt_total

    def save(self, name):
        torch.save(self.model.state_dict(),
                   os.path.join(args.output, "%s.pth" % name))
    def load(self, path):
        print("Load BERT extractor from %s" % path)
        state_dict = torch.load("%s.pth" % path)
        self.model.load_state_dict(state_dict)

    def load_lxmert(self, path):
        print("Load LXMERT model from %s" % path)
        state_dict = torch.load("%s.pth" % path)

        # Do not load any answer head

        if not args.testing:
            for key in list(state_dict.keys()):
                if 'r_layers' not in key and 'x_layers' not in key:
                    state_dict.pop(key)



        # Change Multi GPU to single GPU
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith("module."):
                new_state_dict[key[len("module."):]] = value
        state_dict = new_state_dict

        load_keys = set(state_dict.keys())
        model_keys = set(self.model.state_dict().keys())
        print()
        print("Keys in loaded but not in model:")
        for key in sorted(load_keys.difference(model_keys)):
            print(key)
        print()
        print("Keys in model but not in loaded:")
        for key in sorted(model_keys.difference(load_keys)):
            print(key)
        print()

        self.model.load_state_dict(state_dict, strict=False)


if __name__ == "__main__":

    lxmert = LXMERT(max_seq_length=300)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    if args.testing:
        lxmert.test(test_tuple)
    else:
        lxmert.train(train_tuple, valid_tuple)
