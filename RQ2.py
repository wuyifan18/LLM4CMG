import json
import os
from concurrent.futures import ThreadPoolExecutor
from functools import partial

import tiktoken
from tqdm import tqdm

from evaluate import evaluate
from utils import gpt_call, num_tokens_from_messages, filter_diff, text_to_split, jaccard_similarity

prompt_template = "Generate a concise commit message that summarizes the content of code changes. " \
                  "Do not write explanations or other words, just reply with the commit message.\n" \
                  "{}code change:\n{}\ncommit message:"
dataset_name = "MCMD-NL"
lan_list = ["PHP", "R", "TypeScript", "Swift", "Objective-C"]
# lan_list = ["java", "csharp", "cpp", "python", "javascript"]
root_path = f"dataset/{dataset_name}/processed_data"


def get_sample(lan):
    with open(f"{root_path}/{lan}/test.jsonl", "r") as fr:
        data_samples = []
        for d in fr:
            d = json.loads(d)
            d["diff"] = filter_diff(d["diff"].strip())
            d["msg"] = d['msg'].split('\n')[0]
            data_samples.append(d)
    corpus = []
    with open(f"{root_path}/{lan}/train.jsonl", "r") as fr:
        for d in fr:
            d = json.loads(d)
            d["diff"] = filter_diff(d["diff"].strip())
            d["msg"] = d['msg'].split('\n')[0]
            if all(d["repo"] + d["sha"] != sample["repo"] + sample["sha"] for sample in data_samples):
                corpus.append(d)
    print(f"corpus: {len(corpus)}")
    corpus_sets = [set(text_to_split(doc["diff"])) for doc in corpus]

    for d in tqdm(data_samples, total=len(data_samples)):
        few_shot = 16
        encoding = tiktoken.encoding_for_model("gpt-3.5-turbo-16k")
        d["example"] = ""
        top_similarities = jaccard_similarity(d["diff"], corpus_sets, few_shot, reverse=False)
        examples = [corpus[index] for index, _ in top_similarities]
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
    return data_samples


def gpt_predict(sample: dict, lan: str, prompt_template: str):
    prompt = prompt_template.format(sample["example"], sample["diff"])
    result = gpt_call(prompt, model="gpt-3.5-turbo-16k", temperature=0)
    pred = result["choices"][0]["message"]["content"]
    result = {"repo": sample["repo"], "sha": sample["sha"], "ref": sample["msg"], "pred": pred.split("\n")[0]}
    file_path = f"result/RQ2/{dataset_name}/{lan}.txt"
    if not os.path.exists(os.path.dirname(file_path)):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "a") as fw:
        fw.write(json.dumps(result) + "\n")


if __name__ == '__main__':
    for lan in lan_list:
        data_samples = get_sample(lan)
        with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
            list(tqdm(executor.map(partial(gpt_predict, lan=lan, prompt_template=prompt_template), data_samples), total=len(data_samples)))
    evaluate("RQ2", dataset_name)
