# coding=utf-8
# Copyleft 2019 project LXRT.

from collections import defaultdict
import json
import random

import numpy as np
from torch.utils.data import Dataset

from param import args

TINY_IMG_NUM = 500
FAST_IMG_NUM = 5000


class InputExample(object):
    """A single training/test example for the language model."""
    def __init__(self, uid, sent, visual_feats,
                 obj_labels, label):
        self.uid = uid
        self.sent = sent
        self.visual_feats = visual_feats
        self.obj_labels = obj_labels
        self.label = label


class LXMERTDataset:
    def __init__(self, splits: str, qa_sets=None):
        """
        :param splits: The data sources to be loaded
        :param qa_sets: if None, no action
                        o.w., only takes the answers appearing in these dsets
                              and remove all unlabeled data (MSCOCO captions)
        """
        self.name = splits
        self.sources = splits.split(',')

        print(splits, self.sources)
        # Loading datasets to data
        self.data = []
        for source in self.sources:
            self.data.extend(json.load(open("data/%s.json" % source)))
            
        print("Load %d data from %s" % (len(self.data), self.name))



    def __len__(self):
        return len(self.data)


def make_uid(img_id, dset, sent_idx):
    return "%s_%s_%03d" % (img_id, dset, sent_idx),



class LXMERTTorchDataset(Dataset):
    def __init__(self, dataset: LXMERTDataset, topk=-1):
        super().__init__()
        self.raw_dataset = dataset
        self.task_matched = args.task_matched

        if args.tiny:
            topk = TINY_IMG_NUM
        elif args.fast:
            topk = FAST_IMG_NUM

        # Flatten the dataset (into one sent + one image entries)
        self.data = []
        for datum in self.raw_dataset.data:
            new_datum = {
                'uid': datum['img_fn'],
                'img_id': datum['img_fn'],
                'question': datum['question'],
                'answer_orig': datum['answer_orig'],
                'change': datum['change'],
                'answer1': datum['answer_choices'][0],
                'answer2': datum['answer_choices'][1],
                'answer3': datum['answer_choices'][2],
                'answer4': datum['answer_choices'][3],
                'answer_label': datum['answer_label']
            }
            self.data.append(new_datum)
        
        print("Use %d data in torch dataset" % (len(self.data)))

    def __len__(self):
        return len(self.data)

    def random_feat(self):
        """Get a random obj feat from the dataset."""
        datum = self.data[random.randint(0, len(self.data)-1)]
        img_id = datum['img_id']
        img_info = self.imgid2img[img_id]
        feat = img_info['features'][random.randint(0, 35)]
        return feat

    def __getitem__(self, item: int):
        datum = self.data[item]

        uid = datum['uid']
        img_id = datum['img_id'][:-4]

        # Get image info
        img_info = np.load("data/cosim_feats/"+img_id+"_feat.npy")
        feats = np.load("data/cosim_feats/"+img_id+"_feat.npy")
        boxes = np.load("data/cosim_feats/"+img_id+"_box.npy")
        obj_labels = np.load("data/cosim_feats/"+img_id+"_obj.npy")

        assert len(boxes) == len(feats)


        sent = datum['question'] + " [SEP] " + datum['answer_orig'] + " [SEP] " + datum['change'] + " [SEP] "

        sent1 = sent + datum['answer1']
        sent2 = sent + datum['answer2']
        sent3 = sent + datum['answer3']
        sent4 = sent + datum['answer4']

        label = datum['answer_label']

        # Create target
        example = InputExample(
            uid, (sent1, sent2, sent3, sent4), (feats, boxes),
            obj_labels, label
        )
        return example


class LXMERTEvaluator:
    def __init__(self, dataset: LXMERTDataset):
        self.raw_dataset = dataset

        # Create QA Eval Data
        self.data = []
        for datum in self.raw_dataset.data:
            new_datum = {
                'uid': datum['img_fn'],
                'img_id': datum['img_fn'],
                'question': datum['question'],
                'answer_orig': datum['answer_orig'],
                'change': datum['change'],
                'answer1': datum['answer_choices'][0],
                'answer2': datum['answer_choices'][1],
                'answer3': datum['answer_choices'][2],
                'answer4': datum['answer_choices'][3],
                'answer_label': datum['answer_label']
            }
            self.data.append(new_datum)
        

        # uid2datum
        self.uid2datum = {}
        for datum in self.data:
            self.uid2datum[datum['uid']] = datum

    def evaluate(self, uid2ans: dict, pprint=False):
        score = 0.
        cnt = 0

        for uid, ans in uid2ans.items():
            if uid not in self.uid2datum:   # Not a labeled data
                continue
            datum = self.uid2datum[uid]
            label = datum['answer_label']
            dset = datum['dset']
            if ans in label:
                score += label[ans]
            cnt += 1
            dset2cnt[dset] += 1
        accu = score / cnt
        dset2accu = {}
        for dset in dset2cnt:
            dset2accu[dset] = dset2score[dset] / dset2cnt[dset]

        if pprint:
            accu_str = "Overall Accu %0.4f, " % (accu)
            sorted_keys = sorted(dset2accu.keys())
            for key in sorted_keys:
                accu_str += "%s Accu %0.4f, " % (key, dset2accu[key])
            print(accu_str)

        return accu, dset2accu

    def dump_result(self, uid2ans: dict, path):
        raise NotImplemented
