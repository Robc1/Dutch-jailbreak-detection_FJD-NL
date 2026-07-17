import torch
from fastchat import model
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig


def load_conversation_template(template_name):

    """
        Loads a conversation template from the fastchat library.

        Parameters:
            - template_name (str): The name of the conversation template to load.
    """

    if template_name == 'llama2':
        template_name = 'llama-2'
    
    conv_template = model.get_conversation_template(template_name)
    if conv_template.name == 'zero_shot':
        conv_template.roles = tuple(['### ' + r for r in conv_template.roles])
        conv_template.sep = '\n'
    elif conv_template.name == 'llama-2':
        conv_template.sep2 = conv_template.sep2.strip()

    return conv_template

def load_model_and_tokenizer(model_path, tokenizer_path=None, device='cuda:0', **kwargs):

    """
        Loads a model and tokenizer from the transformers library.
        
        Parameters:
            - model_path (str): The path to the model to load.
            - tokenizer_path (str): The path to the tokenizer to load.
            - device (str): The device to load the model on.
            - kwargs (dict): Additional keyword arguments to pass to the model.
    """

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        **kwargs
    ).to(device).eval()

    tokenizer_path = model_path if tokenizer_path is None else tokenizer_path

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        trust_remote_code=True,
        use_fast=False
    )

    if 'oasst-sft-6-llama-30b' in tokenizer_path:
        tokenizer.bos_token_id = 1
        tokenizer.unk_token_id = 0
    if 'guanaco' in tokenizer_path:
        tokenizer.eos_token_id = 2
        tokenizer.unk_token_id = 0
    if 'llama-2' in tokenizer_path:
        tokenizer.pad_token = tokenizer.unk_token
        tokenizer.padding_side = 'left'
    if 'llama-3' in tokenizer_path:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = 'left'
    if 'falcon' in tokenizer_path:
        tokenizer.padding_side = 'left'
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer

# New addition for API path. Builds a messages list for an OpenAI chat completion call. 
# Mirrors what "load_conversation_template" and "conv_template.get_prompt()"" does for local models, 
# but in the format the API expects.

def build_openai_messages(system_message, user_input):
    messages = []
    if system_message:
        messages.append({"role": "system", "content": system_message})
    messages.append({"role": "user", "content": user_input})
    return messages