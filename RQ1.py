import json
import os
import random
from concurrent.futures import ThreadPoolExecutor
from functools import partial

import chromadb
import tiktoken
from gensim.summarization import bm25
from tqdm import tqdm

from evaluate import evaluate
from utils import gpt_call, num_tokens_from_messages, filter_diff, jaccard_similarity, text_to_split, bm25_similarity

lan_list = ["java", "csharp", "cpp", "python", "javascript"]
seed = 0
select = "jaccard"
reverse = False

# zero_shot
p = "Generate a concise commit message that summarizes the content of code changes.\n" \
    "code change:\n{}\ncommit message:"
p_role = "I want you to act as a commit message generator. I will provide you with a code change, " \
         "and your task is to generate a concise commit message that summarizes the content of code changes.\n" \
         "code change:\n{}\ncommit message:"
p_constraints = "Generate a concise commit message that summarizes the content of code changes. " \
                "Do not write explanations or other words, just reply with the commit message.\n" \
                "code change:\n{}\ncommit message:"
p_constraints_role = "I want you to act as a commit message generator. I will provide you with a code change, " \
                     "and your task is to generate a concise commit message that summarizes the content of code changes. " \
                     "Do not write explanations or other words, just reply with the commit message.\n" \
                     "code change:\n{}\ncommit message:"

# few_shot
p_fs = "Generate a concise commit message that summarizes the content of code changes.\n" \
       "{}code change:\n{}\ncommit message:"
p_role_fs = "I want you to act as a commit message generator. I will provide you with a code change, " \
            "and your task is to generate a concise commit message that summarizes the the content of code changes.\n" \
            "{}code change:\n{}\ncommit message:"
p_constraints_fs = "Generate a concise commit message that summarizes the content of code changes. " \
                   "Do not write explanations or other words, just reply with the commit message.\n" \
                   "{}code change:\n{}\ncommit message:"
p_constraints_role_fs = "I want you to act as a commit message generator. I will provide you with a code change, " \
                        "and your task is to generate a concise commit message that summarizes the content of code changes. " \
                        "Do not write explanations or other words, just reply with the commit message.\n" \
                        "{}code change:\n{}\ncommit message:"

prompt_candidate = {"p_0": p, "p_role_0": p_role, "p_constraints_0": p_constraints, "p_constraints_role_0": p_constraints_role,
                    "p_1": p_fs, "p_role_1": p_role_fs, "p_constraints_1": p_constraints_fs, "p_constraints_role_1": p_constraints_role_fs,
                    "p_2": p_fs, "p_role_2": p_role_fs, "p_constraints_2": p_constraints_fs, "p_constraints_role_2": p_constraints_role_fs,
                    "p_4": p_fs, "p_role_4": p_role_fs, "p_constraints_4": p_constraints_fs, "p_constraints_role_4": p_constraints_role_fs,
                    "p_8": p_fs, "p_role_8": p_role_fs, "p_constraints_8": p_constraints_fs, "p_constraints_role_8": p_constraints_role_fs,
                    "p_16": p_fs, "p_role_16": p_role_fs, "p_constraints_16": p_constraints_fs, "p_constraints_role_16": p_constraints_role_fs,
                    "p_32": p_fs, "p_role_32": p_role_fs, "p_constraints_32": p_constraints_fs, "p_constraints_role_32": p_constraints_role_fs,
                    "p_64": p_fs, "p_role_64": p_role_fs, "p_constraints_64": p_constraints_fs, "p_constraints_role_64": p_constraints_role_fs,
                    "p_128": p_fs, "p_role_128": p_role_fs, "p_constraints_128": p_constraints_fs, "p_constraints_role_128": p_constraints_role_fs}


def get_sample(lan, prompt_name):
    random.seed(seed)
    few_shot = int(prompt_name.split("_")[-1])
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo-16k")
    prompt_template = prompt_candidate[prompt_name]
    with open(f"dataset/MCMD/processed_data/{lan}/train.jsonl", "r") as fr:
        corpus = []
        for d in fr:
            d = json.loads(d)
            d["diff"] = filter_diff(d["diff"].strip())
            d["msg"] = d['msg'].split('\n')[0]
            corpus.append(d)
    data_samples = random.sample(corpus, k=200)

    if few_shot > 0:
        if "random" in select or "jaccard" in select or "bm25" in select:
            corpus = [d for d in corpus if d not in data_samples]
            with open(f"dataset/MCMD/processed_data/{lan}/test.jsonl", "r") as fr:
                for d in fr:
                    d = json.loads(d)
                    d["diff"] = filter_diff(d["diff"].strip())
                    d["msg"] = d['msg'].split('\n')[0]
                    if all(d["repo"] + d["sha"] != sample["repo"] + sample["sha"] for sample in data_samples):
                        corpus.append(d)
            print(f"corpus: {len(corpus)}")
            if "jaccard" in select:
                corpus_sets = [set(text_to_split(doc["diff"])) for doc in corpus]
            if "bm25" in select:
                corpus_sets = [text_to_split(doc["diff"]) for doc in corpus]
                bm25_model = bm25.BM25(corpus_sets)
        elif "semantic" in select:
            emb_dict = {}
            client = chromadb.PersistentClient(path=f"dataset/MCMD")
            collection = client.get_or_create_collection(name=lan, metadata={"hnsw:space": "cosine"})
            print(f"corpus: {collection.count()}")
            with open(f"dataset/MCMD/processed_data/{lan}/valid.vector", "r") as fr:
                for d in fr:
                    key, value = d.strip().split("\t")
                    emb_dict[key] = json.loads(value)

        for d in tqdm(data_samples, total=len(data_samples)):
            d["example"] = ""
            if "random" in select or "jaccard" in select or "bm25" in select:
                if "random" in select:
                    examples = random.sample(corpus, few_shot)
                elif "jaccard" in select:
                    top_similarities = jaccard_similarity(d["diff"], corpus_sets, few_shot, reverse=reverse)
                    examples = [corpus[index] for index, _ in top_similarities]
                elif "bm25" in select:
                    top_similarities = bm25_similarity(d["diff"], bm25_model, few_shot, reverse=reverse)
                    examples = [corpus[index] for index, _ in top_similarities]
            elif "semantic" in select:
                results = collection.query(query_embeddings=emb_dict[d["repo"] + d["sha"]],
                                           n_results=few_shot,
                                           where={"$or": [
                                               {"sha": {"$ne": d["sha"]}},
                                               {"repo": {"$ne": d["repo"]}}
                                           ]})
                results_list = list(zip(results["documents"][0], results["metadatas"][0]))
                examples = []
                for r in results_list:
                    diff = filter_diff(r[0].strip())
                    msg = r[1]["msg"].split('\n')[0]
                    examples.append({"diff": diff, "msg": msg, "repo": r[1]["repo"], "sha": r[1]["sha"]})

            for example in examples:
                assert example["repo"] + example["sha"] != d["repo"] + d["sha"]

            d["example"] = "".join([f"code change:\n{example['diff']}\ncommit message: {example['msg']}\n\n" for example in examples])
            prompt = prompt_template.format(d["example"], d["diff"])
            if num_tokens_from_messages(prompt) > 16384:
                used_length = num_tokens_from_messages(
                    prompt_template + "".join([f"code change:\n\ncommit message: {example['msg']}\n\n" for example in examples]))
                available_length = (16300 - used_length) // (few_shot + 1)
                d["diff"] = encoding.decode(encoding.encode(d["diff"], disallowed_special=())[:available_length])
                for example in examples:
                    example['diff'] = encoding.decode(encoding.encode(example['diff'], disallowed_special=())[:available_length])
                d["example"] = "".join([f"code change:\n{example['diff']}\ncommit message: {example['msg']}\n\n" for example in examples])
                prompt = prompt_template.format(d["example"], d["diff"])
            assert num_tokens_from_messages(prompt) <= 16384
    else:
        used_length = num_tokens_from_messages(prompt_template)
        for d in tqdm(data_samples, total=len(data_samples)):
            d["diff"] = encoding.decode(encoding.encode(d["diff"], disallowed_special=())[:(16384 - used_length)])
    return data_samples


def gpt_predict(sample: dict, lan: str, prompt_name: str):
    prompt_template = prompt_candidate[prompt_name]
    if int(prompt_name.split("_")[-1]) > 0:
        prompt = prompt_template.format(sample["example"], sample["diff"])
    else:
        prompt = prompt_template.format(sample["diff"])
    result = gpt_call(prompt, model="gpt-3.5-turbo-16k", temperature=0)
    pred = result["choices"][0]["message"]["content"]
    result = {"repo": sample["repo"], "sha": sample["sha"], "ref": sample["msg"], "pred": pred.split("\n")[0]}
    file_path = f"result/RQ1/{select}/{lan}/{prompt_name}.txt"
    if not os.path.exists(os.path.dirname(file_path)):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "a") as fw:
        fw.write(json.dumps(result) + "\n")


if __name__ == '__main__':
    for lan in lan_list:
        for prompt_name in prompt_candidate.keys():
            print(lan, prompt_name, select)
            data_samples = get_sample(lan, prompt_name)
            with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
                list(tqdm(executor.map(partial(gpt_predict, lan=lan, prompt_name=prompt_name), data_samples), total=len(data_samples)))
    evaluate("RQ1", select)
