import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
import torch
import json
import argparse
import numpy as np
from tqdm import tqdm
import math
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from utils import load_model_and_tokenizer, load_conversation_template, build_openai_messages
import configs.config as config
from configs.config import MODELS, AFFIRMATIVE_PROMPT, TEMPLATE

# Only imported when using the external API path
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


# Loads data from a file at the specified path.
def get_data(path):
    if path.endswith('.json'):
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    return data

# Gets transition scores for a given dataset from LLM.
def get_scores(model, tokenizer, conv_template, data, name, device, temperature, use_instruction=True):
    results = []
    if use_instruction:
        prompts = config.AFFIRMATIVE_PROMPT
    else:
        prompts = [""]
    for d in tqdm(data):
        temp = []
        for prompt in prompts:
            system = d['system']
            inputs = d['input']
            inputs = prompt + inputs
            if name == 'cipher':
                conv_template.set_system_message(system)
            conv_template.append_message(conv_template.roles[0], inputs)
            conv_template.append_message(conv_template.roles[1], None)
            input_text = conv_template.get_prompt()
            input_ids = tokenizer.encode(input_text, return_tensors='pt').to(device)
            #print(tokenizer.decode(input_ids[0]))
            input_length = len(input_ids[0])
            attn_masks = torch.ones_like(input_ids).to(device)

            output = model.generate(input_ids, attention_mask=attn_masks, max_new_tokens=32, return_dict_in_generate=True, output_scores=True, do_sample=True, temperature=temperature, top_p=1.0)
            # output = model.generate(input_ids, attention_mask=attn_masks, max_new_tokens=128, return_dict_in_generate=True, output_scores=True)
        
            output_text = tokenizer.decode(output.sequences[0][input_length:])
            # score = tuple(x/temperature for x in output.scores)
            #print(output_text)
            transition_scores = model.compute_transition_scores(
                    output.sequences, output.scores, normalize_logits=True)

            transition_scores = np.exp(transition_scores.cpu().numpy())
            
            TEMPLATE['input'] = input_text
            TEMPLATE['output'] = output_text
            TEMPLATE['scores'] = transition_scores.tolist()
            if name == 'benign':
                TEMPLATE['label'] = 'Benign'
            else:
                TEMPLATE['label'] = 'Jailbreak'

            temp.append(TEMPLATE.copy())
            
            conv_template.set_system_message(None)
            conv_template.messages = []
            
        results.append(temp)

    return results

# Sets the number of parallel API requests. Increase if your API allows higher concurrency; decrease if you hit rate limits.
MAX_WORKERS = 5


# Makes a single API call for one (data item, prompt) pair. Returns a dict in the same shape as "TEMPLATE". 
# Runs inside a thread, uses only local variables AND no shared state.
def _call_api_single(client, model_id, prompt, d, name, temperature):

    system = d['system']
    inputs = prompt + d['input']
    messages = build_openai_messages(system, inputs)

    max_retries = 5
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model_id,
                messages=messages,
                max_tokens=32,
                temperature=temperature,
                top_p=1.0,
                logprobs=True,
                top_logprobs=1,
            )
            break  # success
        except Exception as e:
            if attempt < max_retries - 1:
                wait = 2 ** attempt  # 1, 2, 4, 8, 16 seconds
                print(f"\n[Retry {attempt + 1}/{max_retries - 1}] Error: {e}. Retrying in {wait}s...")
                time.sleep(wait)
            else:
                print(f"\n[Failed] All {max_retries} attempts exhausted. Last error: {e}")
                raise

    output_text = response.choices[0].message.content
    token_logprobs = response.choices[0].logprobs.content
    transition_scores = np.exp(np.array([[t.logprob for t in token_logprobs]]))

    return {
        'input':  json.dumps(messages),
        'output': output_text,
        'scores': transition_scores.tolist(),
        'label':  'Benign' if name == 'benign' else 'Jailbreak',
    }


# OpenAI-compatible API version of get_scores(), with parallel requests. Each data item is submitted as a separate thread 
# (up to "MAX_WORKERS" at once). Results are collected in the original order so the output file is identical to the sequential version.
def get_scores_openai(client, model_id, model_name, data, name, temperature, use_instruction=True):
    prompts = config.AFFIRMATIVE_PROMPT if use_instruction else [""]

    # Build a flat list of (index, prompt, data_item) tasks so we can restore order.
    # Each data item gets one task per prompt (currently always 1 prompt per run).
    tasks = [(i, prompt, d) for i, d in enumerate(data) for prompt in prompts]

    # results_map[i] = list of per-prompt dicts for data[i]
    results_map = {i: [] for i in range(len(data))}

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_task = {
            executor.submit(_call_api_single, client, model_id, prompt, d, name, temperature): (i, prompt)
            for i, prompt, d in tasks
        }

        for future in tqdm(as_completed(future_to_task), total=len(tasks), desc=name):
            i, prompt = future_to_task[future]
            results_map[i].append(future.result())  # raises immediately if the call failed

    # Reassemble in original data order
    return [results_map[i] for i in range(len(data))]

    
if __name__ == '__main__':
    args = argparse.ArgumentParser('FJD Get Scores')
    args.add_argument('--model', type=str, default='llama2-7b', help='model name')
    args.add_argument('--data', type=str, default='./data/ori', help='data path root')
    args.add_argument('--output', type=str, default='./data/result', help='prompt path')
    args.add_argument('--instruction', type=bool, default=True, help='use instruction or not')
    args.add_argument('--temperature', type=bool, default=True, help='use temperature or not')
    
    # New arguments for external API
    args.add_argument('--use-api', action='store_true', help='Use external API instead of local model')
    args = args.parse_args()

    # Load config
    model_config = MODELS[args.model]
    model_path = model_config['name']
    template = model_config['template']

    if args.temperature:
        if args.instruction:
            temp = model_config['tempature-fjd']
        else:
            temp = model_config['tempature-ft']
    else:
        temp = 1.0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_path = os.path.join(args.data, args.model)
    data_list = [os.path.join(data_path, f) for f in os.listdir(data_path)]

    use_external_api = args.use_api

    # ------------------------------------------------------------------ #
    # Automation: run every temperature x affirmative prompt x 5 runs    #
    # Output: <args.output>/<model>/<prompt>/Run <n>/Temperature <t>/...  #
    # ------------------------------------------------------------------ #
    NUM_RUNS = 5
    all_prompts = AFFIRMATIVE_PROMPT  # list of 7 prompts from config

    # Temperatures to sweep over for tempature-fjd
    TEMPERATURES = [0, 0.1, 0.3, 0.5, 0.7, 1.0]
    
    if use_external_api:
        # OpenAI-compatible path
        assert OpenAI is not None, "Install openai: pip install openai"
        API_token = os.environ["API_KEY"]  # ← fill in your key
        model_name = "model"

        base_url = f"[PLACEHOLDER: API url]"
        model_id = f"[PLACEHOLDER: model-id]"

        client = OpenAI(
            api_key=API_token[6:] if API_token.startswith("Token ") else API_token,
            base_url=base_url
        )

        for temperature in TEMPERATURES:
            # Override tempature-fjd in the loaded config for this sweep iteration
            model_config['tempature-fjd'] = temperature
            temp = temperature

            for prompt in all_prompts:
                # Temporarily override the module-level list to a single prompt
                config.AFFIRMATIVE_PROMPT = [prompt]

                for run in range(1, NUM_RUNS + 1):
                    # Build output directory: <o>/<model>/<prompt>/Run <n>/Temperature <t> 
                    prompt_label = prompt.strip() if prompt.strip() else "no_prompt"
                    prompt_dir = os.path.join(args.output, args.model, prompt_label, f"Run {run}", f"Temperature {temperature}")
                    
                    os.makedirs(prompt_dir, exist_ok=True)

                    for path in data_list:
                        dataset = get_data(path)
                        filename = os.path.basename(path)
                        out_path = os.path.join(prompt_dir, filename)

                        name = os.path.splitext(filename)[0].split('-')[-2]
                        print(f'Temp: {temperature} | Prompt: "{prompt.strip()}" | Run {run} | Processing {name} | Saving to {out_path}')

                        result = get_scores_openai(client, model_id, model_path, dataset, name,
                                                   temperature=temp, use_instruction=args.instruction)

                        with open(out_path, 'w', encoding='utf-8') as f:
                            json.dump(result, f, ensure_ascii=False, indent=4)

    else:
        # Original local model path
        model, tokenizer = load_model_and_tokenizer(model_path, device=device)
        conv_template = load_conversation_template(template)

        for temperature in TEMPERATURES:
            # Override tempature-fjd in the loaded config for this sweep iteration
            model_config['tempature-fjd'] = temperature
            temp = temperature

            for prompt in all_prompts:
                # Temporarily override the module-level list to a single prompt
                config.AFFIRMATIVE_PROMPT = [prompt]

                for run in range(1, NUM_RUNS + 1):
                    # Build output directory: <o>/<model>/<prompt>/Run <n>/Temperature <t>
                    prompt_label = prompt.strip() if prompt.strip() else "no_prompt"
                    prompt_dir = os.path.join(args.output, args.model, prompt_label, f"Run {run}", f"Temperature {temperature}")
                    
                    os.makedirs(prompt_dir, exist_ok=True)

                    for path in data_list:
                        dataset = get_data(path)
                        filename = os.path.basename(path)
                        out_path = os.path.join(prompt_dir, filename)

                        name = os.path.splitext(filename)[0].split('-')[-2]
                        print(f'Temp: {temperature} | Prompt: "{prompt.strip()}" | Run {run} | Processing {name} | Saving to {out_path}')

                        result = get_scores(model, tokenizer, conv_template, dataset, name,
                                            device, temperature=temp, use_instruction=args.instruction)

                        with open(out_path, 'w', encoding='utf-8') as f:
                            json.dump(result, f, ensure_ascii=False, indent=4)