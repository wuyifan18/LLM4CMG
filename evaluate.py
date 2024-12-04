#!/usr/bin/env python
# !-*-coding:utf-8 -*-
import json
import re
import sys

import numpy as np
import pandas as pd

sys.path.append("metric")
from metric.smooth_bleu import codenn_smooth_bleu
from metric.meteor.meteor import Meteor
from metric.rouge.rouge import Rouge
from metric.cider.cider import Cider

import warnings
import logging

warnings.filterwarnings('ignore')
logging.basicConfig(format='[%(asctime)s - %(levelname)s - %(name)s] %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)


def Commitbleus(refs, preds):
    r_str_list = []
    p_str_list = []
    for r, p in zip(refs, preds):
        r_str_list.append([" ".join([str(token_id) for token_id in r[0]])])
        p_str_list.append(" ".join([str(token_id) for token_id in p]))
    bleu_list, bleu_lists = codenn_smooth_bleu(r_str_list, p_str_list)
    codenn_bleu = bleu_list[0]
    B_Norm = round(codenn_bleu, 2)
    print("BLEU: ", B_Norm)
    scores = [bleu_list[0] for bleu_list in bleu_lists]
    return B_Norm, scores


def read_to_list(filename):
    with open(filename, 'r', encoding="utf-8") as f:
        data = [json.loads(d) for d in f.readlines()]
    refs, preds = [], []
    for row in data:
        ref = [x for x in re.split(r'(\W)', row["ref"].lower()) if x.strip()]
        pred = [x for x in re.split(r'(\W)', row["pred"].lower()) if x.strip()]
        refs.append(ref)
        preds.append(pred)
    return refs, preds


def metetor_rouge_cider(refs, preds):
    refs_dict = {}
    preds_dict = {}
    for i in range(len(preds)):
        preds_dict[i] = [" ".join(preds[i])]
        refs_dict[i] = [" ".join(refs[i][0])]

    score_Meteor, scores_Meteor = Meteor().compute_score(refs_dict, preds_dict)
    score_Meteor = np.mean(scores_Meteor)
    print("Meteor: ", round(score_Meteor * 100, 2))

    score_Rouge, scores_Rouge = Rouge().compute_score(refs_dict, preds_dict)
    print("Rouge-L: ", round(score_Rouge * 100, 2))

    score_Cider, scores_Cider = Cider().compute_score(refs_dict, preds_dict)
    print("Cider: ", round(score_Cider, 2))

    return round(score_Meteor * 100, 2), round(score_Rouge * 100, 2), round(score_Cider, 2), scores_Meteor, scores_Rouge, scores_Cider


def compute(result_path):
    refs, preds = read_to_list(result_path)
    refs = [[t] for t in refs]
    bleu_score, scores_bleu = Commitbleus(refs, preds)
    meteor, rouge, cider, scores_Meteor, scores_Rouge, scores_Cider = metetor_rouge_cider(refs, preds)
    print()
    return bleu_score, meteor, rouge, cider


def evaluate(RQ, dataset):
    if RQ == "RQ1":
        from RQ1 import prompt_candidate, lan_list
    else:
        from RQ2 import prompt_template, lan_list
        prompt_candidate = {RQ: prompt_template}
    results = []
    for prompt_name in prompt_candidate.keys():
        tmp = {"prompt": prompt_name}
        avg_bleu_score, avg_meteor, avg_rouge, avg_cider = 0.0, 0.0, 0.0, 0.0
        for lan in lan_list:
            print(lan, prompt_name)
            if RQ == "RQ1":
                result_path = f"result/RQ1/{dataset}/{lan}/{prompt_name}.txt"
            else:
                result_path = f"result/{RQ}/{dataset}/{lan}.txt"
            bleu_score, meteor, rouge, cider = compute(result_path)
            tmp.update({f"{lan}_bleu": bleu_score,
                        f"{lan}_meteor": meteor,
                        f"{lan}_rouge": rouge,
                        f"{lan}_cider": cider})
            avg_bleu_score += bleu_score
            avg_meteor += meteor
            avg_rouge += rouge
            avg_cider += cider
        tmp.update({"avg_bleu": round(avg_bleu_score / len(lan_list), 2),
                    "avg_meteor": round(avg_meteor / len(lan_list), 2),
                    "avg_rouge": round(avg_rouge / len(lan_list), 2),
                    "avg_cider": round(avg_cider / len(lan_list), 2)})
        results.append(tmp)

    columns = ["prompt"]
    for lan in lan_list:
        columns.extend([f"{lan}_bleu", f"{lan}_meteor", f"{lan}_rouge", f"{lan}_cider"])
    columns.extend(["avg_bleu", "avg_meteor", "avg_rouge", "avg_cider"])
    df = pd.DataFrame(results, columns=columns)
    df.to_csv(f"result/{RQ}/{dataset}/result.csv", index=False)
