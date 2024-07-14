import torch
import time
import wandb
import open_clip
import matplotlib.pyplot as plt
from utils import CIFAR100DataModule, cifar100_templates,zeroshot_classifier_gpt
import numpy as np
import base64
import io 
from io import BytesIO
from PIL import Image
import os
import base64


def read_saved_image(image_url):
    """
    Reads an image from a base64-encoded URL (generated by save_image).

    Args:
        image_url: The base64-encoded URL of the image.

    Returns:
        A PIL image object, or None if the URL is invalid or there's an error.
    """

    try:
        # Extract the base64-encoded data from the URL
        encoded_data = image_url.split(",")[1]

        # Decode the base64 data
        decoded_data = base64.b64decode(encoded_data)

        # Load the decoded data as a PIL image
        return Image.open(io.BytesIO(decoded_data))
    except Exception as e:
        print(f"Error reading image: {e}")
        return None


def infer_model():
    last_action_time = time.time()
    logs = []  # List to store image annotations for logging
    while True:
        try:
            folder_path = r"C:\Users\TFM\src\ObjectClassification\received_images" #fill with the full path where the folder of images received wants to be located
            if folder_path:
            # Process images from folder
                for image_path in [os.path.join(folder_path, filename) for filename in os.listdir(folder_path)]:
                    # Load image using PIL
                    pil_img =  Image.open(image_path).convert("RGB")
                    
                    # Perform model inference
                    with torch.no_grad():
                        preprocessed_img = preprocess(pil_img).unsqueeze(0).to(device)  # Add batch dimension and move to device

                        # predict
                        image_features = model.encode_image(preprocessed_img)
                        image_features /= image_features.norm(dim=-1, keepdim=True)

                        logits_both_openai = logits(image_features,zeroshot_weights_gpt_both_openai)
                        
                        if time.time() - last_action_time >= 5: #every 5 seconds

                            img = preprocessed_img[0].cpu().numpy().transpose(1, 2, 0)
                                                
                            pred_idx = logits_both_openai.argmax().item()
                            
                            pred = data_module.cifar_classes[pred_idx]
                            # Convert image to numpy array and transpose
                            img_np = img.clip(0, 1)

                            
                            # Plot the image
                            plt.figure()
                            plt.imshow(img_np)
                            
                            # Add annotation for prediction and label
                            plt.text(10, 20, f"Prediction: {pred}", color='white', fontsize=20, bbox=dict(facecolor='black', alpha=0.5))
                            
                            # Hide axes
                            plt.axis('off')
                            
                            # Log the image with annotations
                            logs.append(wandb.Image(plt))
                            
                            # Close the current plot to avoid memory leak
                            plt.close()
                            #one for each model so i know how it predicts
                            
                            last_action_time = time.time()
                wandb.log({"predictions": logs})
            else:
                print("No images found in the specified folder.")

        except Exception as e:
            print(f"Error: {e}")
            break
        
def logits(image_features, zeroshot_weights):
        return image_features @ zeroshot_weights

if __name__ == "__main__":  
    wandb.init(project='predicting')
    
    model, preprocess,_ = open_clip.create_model_and_transforms('ViT-L-14', 'datacomp_xl_s13b_b90k') #another model could be used
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    PATH_TO_PROMPTS_OPENAI = r"C:\Users\TFM\src\ObjectClassification\cifar100_prompts.json" #fill whith the cifar100_prompts.json full path

    #model.load_state_dict(torch.load(r'C:\Users\TFM\src\ObjectClassification\model_weights\validation\best_val_loss.pth')) #fill with the full path

    data_module = CIFAR100DataModule(train_batch_size=120, val_batch_size=100, test_batch_size=100, num_workers=11)   

    zeroshot_weights_gpt_both_openai = zeroshot_classifier_gpt(model,data_module.cifar_classes,cifar100_templates,True,PATH_TO_PROMPTS_OPENAI, line=False)

    logs = []	

    global folder_path
    folder_path = r"C:\Users\TFM\src\ObjectClassification\received_images"
    last_action_time = time.time()
    infer_model()

    






