import json
from tqdm import tqdm
from transformers import GPTJForCausalLM,AutoModelForCausalLM, AutoTokenizer
import torch
import warnings
import torch.nn as nn
import gc
from transformers import GPTJForCausalLM, AutoTokenizer



def memory_stats():
    print(torch.cuda.memory_allocated()/1024**2)
    print(torch.cuda.memory_cached()/1024**2)


def allocate():
    x = torch.randn(1024*1024, device='cuda')
    memory_stats()


# Suppress the warning
warnings.filterwarnings("ignore", message="TypedStorage is deprecated", category=UserWarning)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    json_name = "cifar100-GPT-J.json"


    print("Loading model and tokenizer...")
    # First, we load the model from the hub. We select the "float16" revision, which means that all parameters are stored using 16 bits, rather than the default float32 ones (which require twice as much RAM memory). 
    # We also set low_cpu_mem_usage to True, in order to only load the model once into CPU memory
    model = GPTJForCausalLM.from_pretrained(
                "EleutherAI/gpt-j-6B",
                revision="float16",
                low_cpu_mem_usage=True
            )
    model.to(device)
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
    print("Tokenizer loaded successfully!")

    cifar100_classes = [
        "apple", "aquarium_fish", "baby", "bear", "beaver", "bed", "bee", "beetle", "bicycle", "bottle",
        "bowl", "boy", "bridge", "bus", "butterfly", "camel", "can", "castle", "caterpillar", "cattle",
        "chair", "chimpanzee", "clock", "cloud", "cockroach", "couch", "crab", "crocodile", "cup", "dinosaur",
        "dolphin", "elephant", "flatfish", "forest", "fox", "girl", "hamster", "house", "kangaroo", "computer_keyboard",
        "lamp", "lawn_mower", "leopard", "lion", "lizard", "lobster", "man", "maple_tree", "motorcycle", "mountain",
        "mouse", "mushroom", "oak_tree", "orange", "orchid", "otter", "palm_tree", "pear", "pickup_truck", "pine_tree",
        "plain", "plate", "poppy", "porcupine", "possum", "rabbit", "raccoon", "ray", "road", "rocket",
        "rose", "sea", "seal", "shark", "shrew", "skunk", "skyscraper", "snail", "snake", "spider",
        "squirrel", "streetcar", "sunflower", "sweet_pepper", "table", "tank", "telephone", "television", "tiger", "tractor",
        "train", "trout", "tulip", "turtle", "wardrobe", "whale", "willow_tree", "wolf", "woman", "worm"
    ]

    all_responses = {}
    vowel_list = ['A', 'E', 'I', 'O', 'U']
    # Using OpenAI's free tier and don't have access to GPT-3.5 or GPT-4, I utilize models like GPT-2
    print("Generating prompts...")
    for category in tqdm(cifar100_classes):
        if category[0] in vowel_list:
            article = "an"
        else:
            article = "a"
        print("Category:", category)

        prompts = []
        prompts.append("Describe what " + article + " " + category + " looks like")
        prompts.append("How can you identify " + article + " " + category + "?")
        prompts.append("What does " + article + " " + category + " look like?")
        prompts.append("Describe an image from the internet of " + article + " "  + category)
        prompts.append("A caption of an image of "  + article + " "  + category + ":")

        all_result = []
        print("Generating prompts...")

        for curr_prompt in prompts:
            # prepare that prompt using the tokenizer for the model (the only input required for the model are the input_ids)
            input_ids = tokenizer(curr_prompt, return_tensors="pt").input_ids.to(device) #tokenized version of the input text 
            # Set pad token id
            # Set padding token
            tokenizer.pad_token = "[PAD]" #pad shorter sequences to match the length of the longest sequence in the batc
            # Create attention mask
            attention_mask = input_ids.ne(tokenizer.pad_token_id).float() #1 for actual tokens and 0 for padding tokens.
            
            print("Generating text...")
            gen_tokens = model.generate(
                input_ids,
                do_sample=True,
                temperature=0.9,
                max_length=50,
                attention_mask=attention_mask
            )
            print("Decoding text...")
            gen_text = tokenizer.batch_decode(gen_tokens)[0]


            #Processing API Response
            all_result.append(gen_text)

        all_responses[category] = all_result

    try:
        with open(json_name, 'w') as f:
            json.dump(all_responses, f, indent=4)
            print("JSON file created successfully!")
    except Exception as e:
        print("Error:", e)