#!/usr/bin/env python
import openai
from datasets import load_dataset
from tqdm import tqdm
import concurrent.futures
import opencc
import json
import os
from dotenv import load_dotenv

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY", "<EMPTY>")
openai.base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1/")
model_name = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
dataset_name = os.getenv("DATASET_NAME", "Requires")
save = open("results.jsonl", "w")

# global features
converter = opencc.OpenCC('s2t.json')


def completions_by_prompt(prompt):
    # create a completion
    completion = openai.completions.create(model=model_name, prompt=prompt, max_tokens=64)
    return completion

def completions_by_chat(messages):
    # create a chat completion
    completion = openai.chat.completions.create(
        model=model_name,
        messages=messages
    )
    return completion

def process_item(item):
    # The main idea of converting item['user'] using OpenCC is to convert Simplified Chinese to traditional Chinese.
    # I need to convert the sample of Simplified Chinese to Traditional Chinese to experiment with the transfer of approximate language knowledge and enhance differentiation.
    # Test input: item['user'] = "我说繁体中文" # The example input in Simplified Chinese.
    input_data = converter.convert(item['user'])
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": input_data},
    ]
    completion = completions_by_chat(messages)
    return {
        'input': input_data,
        'output': completion.choices[0].message.content
    }

if __name__ == "__main__":
    dataset = load_dataset(dataset_name)
    num_threads = 8

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = []
        for item in dataset['train']:
            future = executor.submit(process_item, item)
            futures.append(future)

        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            result = future.result()
            raw_line = json.dumps(result, ensure_ascii=False)
            save.write(f"{raw_line}\n")
            save.flush()
