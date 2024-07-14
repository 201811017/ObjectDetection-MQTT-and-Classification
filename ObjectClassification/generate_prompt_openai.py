from openai import OpenAI
import json
from tqdm import tqdm

json_name = "cifar100-GPT-J.json"



client = OpenAI(
    # This is the default and can be omitted
    api_key="your_own_API_Key",
)



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

for category in tqdm(cifar100_classes):

	if category[0] in vowel_list:
		article = "an"
	else:
		article = "a"

	prompts = []
	prompts.append("Describe what " + article + " " + category + " looks like")
	prompts.append("How can you identify " + article + " " + category + "?")
	prompts.append("What does " + article + " " + category + " look like?")
	prompts.append("Describe an image from the internet of " + article + " "  + category)
	prompts.append("A caption of an image of "  + article + " "  + category + ":")

    #OpenAI API Call
	all_result = []
	for curr_prompt in prompts:
		response = client.chat.completions.create(
		messages=[
			{
				"role": "user",
				"content": curr_prompt,
			}
		],
		model="gpt-3.5-turbo-instruct",
		temperature=.99, #high randomness
		max_tokens = 50,
		n=10, #10 different tokens
		stop="."
		)


        #Processing API Response
	for choice in response.choices:
		result = choice.text.strip().replace("\n\n", "") + "."  # Remove extra newlines
		all_result.append(result)

	all_responses[category] = all_result

with open(json_name, 'w') as f:
	json.dump(all_responses, f, indent=4)
