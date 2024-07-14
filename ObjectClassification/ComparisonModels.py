import torch
import os
import time
import wandb
import open_clip
from tqdm import tqdm
from utils import CIFAR100DataModule, cifar100_templates,zeroshot_classifier_cifar,zeroshot_classifier_gpt
import random
import matplotlib.pyplot as plt



PATH_TO_PROMPTS_OPENAI = r"C:\Users\TFM\ObjectClassification\cifar100_prompts.json" #fill whith the cifar100_prompts.json full path
PATH_TO_PROMPTS_BLOOMLM= r"C:\Users\TFM\ObjectClassification\cifar100-GPT-J.json" #fill whith the cifar100-GPT-J.json full path



class CLIPModel():
    #initialize the model
    def __init__(self, models_to_test, data_module, classes, batch_size=256, num_epochs=20,num_images_to_log=5):
        self.models_to_test = models_to_test
        self.batch_size = batch_size
        self.data_module = data_module
        self.classes = classes
        self.num_epochs = num_epochs
        self.num_images_to_log = num_images_to_log
        
    def logits(self, image_features, zeroshot_weights):
        return image_features @ zeroshot_weights
        
    def accuracy(self,output, target, topk=(1,)):
        pred = output.topk(max(topk), 1, True, True)[1].t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        return [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) for k in topk]
    
    def evaluate_models(self):
        # Define column names
        columns = ["Model", "Preprocessing", "Inference Time", "Avg Top-1 Acc (Base)", "Avg Top-1 Acc Openai (Cupl)", "Avg Top-1 Acc Openai (Both)", "Avg Top-1 Acc Bloom (Cupl)", "Avg Top-1 Acc Bloom (Both)",
                   "Avg Top-5 Acc (Base)", "Avg Top-5 Acc Openai (Cupl)", "Avg Top-5 Acc Openai (Both)","Avg Top-5 Acc Bloom (Cupl)", "Avg Top-5 Acc Bloom (Both)"]
        data = []
        
        #Directory to
        os.makedirs(os.path.join(os.getenv('LOCALAPPDATA'), 'Temp', 'tmp9_vtjfg2wandb-media'), exist_ok=True)
        


        for mn, pt in self.models_to_test:
            print(mn, pt)
            model, _, preprocess = open_clip.create_model_and_transforms(mn, pretrained=pt)
            data_module.setup(preprocess)
            test_dataloader = data_module.test_dataloader()
            model.eval()
            
            model.cuda()

            zeroshot_weights_base = zeroshot_classifier_cifar(model,self.classes, cifar100_templates)
            
            zeroshot_weights_cupl_openai = zeroshot_classifier_gpt(model,self.classes,cifar100_templates,False,PATH_TO_PROMPTS_OPENAI, line = False)
            zeroshot_weights_cupl_bloom = zeroshot_classifier_gpt(model,self.classes,cifar100_templates,False,PATH_TO_PROMPTS_BLOOMLM, line = True)

            zeroshot_weights_gpt_both_openai = zeroshot_classifier_gpt(model,self.classes,cifar100_templates,True,PATH_TO_PROMPTS_OPENAI, line=False)
            zeroshot_weights_gpt_both_bloom = zeroshot_classifier_gpt(model,self.classes,cifar100_templates,True,PATH_TO_PROMPTS_BLOOMLM, line=True)


            t0 = time.time()
            with torch.no_grad():
                
                #for epoch in range(self.num_epochs):
                top1_base, top1_cupl_openai, top1_both_openai, top5_base, top5_cupl_openai, top5_both_openai,n = 0., 0., 0., 0., 0., 0.,0
                top1_cupl_bloom, top1_both_bloom, top5_cupl_bloom, top5_both_bloom= 0., 0., 0.,0.

                for i, (images, target) in enumerate(tqdm(test_dataloader)):
                    images = images.cuda()
                    target = target.cuda()

                    # predict
                    image_features = model.encode_image(images)
                    image_features /= image_features.norm(dim=-1, keepdim=True)
                    logits_base = self.logits(image_features,zeroshot_weights_base) #por que aqui no tengo el problema de que zero_shot coge todos weights?
                    
                    logits_cupl_openai = self.logits(image_features,zeroshot_weights_cupl_openai)
                    logits_cupl_bloom = self.logits(image_features,zeroshot_weights_cupl_bloom)

                    logits_both_openai = self.logits(image_features,zeroshot_weights_gpt_both_openai)
                    logits_both_bloom = self.logits(image_features,zeroshot_weights_gpt_both_bloom)


                    # measure accuracy
                    acc1_base, acc5_base = self.accuracy(logits_base, target, topk=(1, 5))
                    
                    acc1_cupl_openai, acc5_cupl_openai = self.accuracy(logits_cupl_openai, target, topk=(1, 5)) #target because supervised learning for final classification
                    acc1_cupl_bloom, acc5_cupl_bloom = self.accuracy(logits_cupl_bloom, target, topk=(1, 5))

                    acc1_both_openai, acc5_both_openai = self.accuracy(logits_both_openai, target, topk=(1, 5))
                    acc1_both_bloom, acc5_both_bloom = self.accuracy(logits_both_bloom, target, topk=(1, 5))


                    # Accumulate accuracies
                    top1_base += acc1_base
                    top1_cupl_openai += acc1_cupl_openai
                    top1_both_openai += acc1_both_openai
                    top1_cupl_bloom += acc1_cupl_bloom
                    top1_both_bloom += acc1_both_bloom

                    top5_base += acc5_base
                    top5_cupl_openai += acc5_cupl_openai
                    top5_both_openai += acc5_both_openai
                    top5_cupl_bloom += acc5_cupl_bloom
                    top5_both_bloom += acc5_both_bloom
                    
                    n += images.size(0) 
                    # Generate random indices to select images
                    # Select a limited number of images for logging
                    if len(images) > self.num_images_to_log:
                        indices_to_log = random.sample(range(len(images)), self.num_images_to_log)
                    else:
                        indices_to_log = range(len(images))
                    logs = []
                    
                    for idx in indices_to_log:
                        img = images[idx].cpu().numpy().transpose(1, 2, 0)  # Assuming CHW format
                        
                        pred_idx = logits_base[idx].argmax().item()
                        label_idx = target[idx].item()
                        
                        pred = self.classes[pred_idx]
                        label = self.classes[label_idx]
                        # Convert image to numpy array and transpose
                        img_np = img.clip(0, 1)
                        
                        # Plot the image
                        plt.figure()
                        plt.imshow(img_np)
                        
                        # Add annotation for prediction and label
                        plt.text(10, 20, f"Prediction: {pred}", color='white', fontsize=20, bbox=dict(facecolor='black', alpha=0.5))
                        plt.text(10, 40, f"Label: {label}", color='white', fontsize=20, bbox=dict(facecolor='black', alpha=0.5))
                        
                        # Hide axes
                        plt.axis('off')
                        
                        # Log the image with annotations
                        logs.append(wandb.Image(plt))
                        
                        # Close the current plot to avoid memory leak
                        plt.close()
                        #one for each model so i know how it predicts

                    wandb.log({"predictions": logs})
                # Calculate average accuracies
                avg_top1_base = (top1_base / n) * 100
                avg_top1_cupl_openai = (top1_cupl_openai / n) * 100
                avg_top1_both_openai = (top1_both_openai / n) * 100
                avg_top1_cupl_bloom = (top1_cupl_bloom / n) * 100
                avg_top1_both_bloom = (top1_both_bloom / n) * 100

                avg_top5_base = (top5_base / n) * 100
                avg_top5_cupl_openai = (top5_cupl_openai / n) * 100
                avg_top5_both_openai = (top5_both_openai / n) * 100
                avg_top5_cupl_bloom = (top5_cupl_bloom / n) * 100
                avg_top5_both_bloom = (top5_both_bloom / n) * 100
                
                t1 = time.time()

                
            data.append([mn, pt, t1-t0, avg_top1_base,avg_top1_cupl_openai,avg_top1_both_openai,avg_top1_cupl_bloom,avg_top1_both_bloom,
                        avg_top5_base,avg_top5_cupl_openai,
                        avg_top5_both_openai,
                        avg_top5_cupl_bloom,
                        avg_top5_both_bloom])
            print(data)
        wandb.log(
            {'Result Table': wandb.Table(data=data, columns=columns)}
        )


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_module = CIFAR100DataModule(train_batch_size=120, val_batch_size=120, test_batch_size=256, num_workers=11)
    
    wandb.login()#relogin=True)
    wandb.init(project='Model Comparison',
           job_type='benchmarking')
    models_to_test = [
    ('RN50', 'cc12m'), #Resnet50 model trained on 12 million captioned images
    ('ViT-B-32-quickgelu', 'openai'), #OpenAI trained on their internal dataset of 400M images
    ('ViT-B-32-quickgelu', 'laion400m_e31'), #trained on LAION 400M by OpenCLIP
    ('ViT-H-14', 'laion2b_s32b_b79k'), #trained on 2 billion images 
    ('ViT-L-14', 'datacomp_xl_s13b_b90k')]
    # ('EVA02-E-14-plus', 'laion2b_s9b_b144k'), ('ViT-bigG-14-CLIPA', 'datacomp1b')]
    #models_to_test = [('ViT-L-14', 'datacomp_xl_s13b_b90k')] #trained on 2 billion images
    #models_to_test = [('ViT-L-14', 'datacomp_xl_s13b_b90k')] #not for 120
    # models_to_test = [('ViT-H-14', 'laion2b_s32b_b79k')]

    models = CLIPModel(models_to_test, data_module, data_module.cifar_classes, batch_size=512)
    models.evaluate_models()
    wandb.alert(title="Run Finished", text="Your run has completed successfully.")

    #wandb.finish()
    
    
    

    
