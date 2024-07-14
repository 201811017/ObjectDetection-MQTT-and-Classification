import torch
from torch import nn
import open_clip
import torch.nn.functional as F
from tqdm.auto import tqdm
import wandb
from torch.optim.lr_scheduler import CosineAnnealingLR
import open_clip
import timm
import random
import json
from utils import CIFAR100DataModule, cifar100_templates,zeroshot_classifier_cifar,zeroshot_classifier_gpt
import time
import os
import random
from urllib.request import urlopen
from PIL import Image
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

class CLIPModel():
    #initialize the model
    def __init__(self, model,data_module, classes, batch_size=256):
        self.batch_size = batch_size
        self.data_module = data_module
        self.classes = classes
        self.model = model
        
    def logits(self, image_features, zeroshot_weights):
        return image_features @ zeroshot_weights
        
    def accuracy(self,output, target, topk=(1,)):
        pred = output.topk(max(topk), 1, True, True)[1].t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        return [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) for k in topk]
    
    def evaluate_models(self):
        # Define column names
        columns = ["Model", "Preprocessing", "Inference Time", "Avg Top-1 Acc (Base)", "Avg Top-1 Acc (Cupl)", "Avg Top-1 Acc (Both)", "Avg Top-5 Acc (Base)", "Avg Top-5 Acc (Cupl)", "Avg Top-5 Acc (Both)"]
        mn = "My Model"
        pt = "datacomp_xl_s13b_b90k"
        data = []
        model = self.model

        
        os.makedirs(os.path.join(os.getenv('LOCALAPPDATA'), 'Temp', 'tmp9_vtjfg2wandb-media'), exist_ok=True)
        # Log the column names only once
        wandb.log({'Result Table': wandb.Table(columns=columns)})

        test_dataloader = data_module.test_dataloader()
        model.eval()
        
        model.cuda()

        zeroshot_weights_base = zeroshot_classifier_cifar(model,self.classes, cifar100_templates)
        zeroshot_weights_cupl = zeroshot_classifier_gpt(model,self.classes,cifar100_templates,False,PATH_TO_PROMPTS)
        zeroshot_weights_gpt_both = zeroshot_classifier_gpt(model,self.classes,cifar100_templates,True,PATH_TO_PROMPTS)

        
        t0 = time.time()
        with torch.no_grad():
            
            #for epoch in range(self.num_epochs):
            top1_base, top1_cupl, top1_both, top5_base, top5_cupl, top5_both,n = 0., 0., 0., 0., 0., 0.,0

            for i, (images, target) in enumerate(tqdm(test_dataloader)):
                images = images.cuda()
                target = target.cuda()

                # predict
                image_features = model.encode_image(images)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                logits_base = self.logits(image_features,zeroshot_weights_base) #por que aqui no tengo el problema de que zero_shot coge todos weights?
                logits_cupl = self.logits(image_features,zeroshot_weights_cupl)
                logits_both = self.logits(image_features,zeroshot_weights_gpt_both)

                # measure accuracy
                acc1_base, acc5_base = self.accuracy(logits_base, target, topk=(1, 5))
                acc1_cupl, acc5_cupl = self.accuracy(logits_cupl, target, topk=(1, 5))
                acc1_both, acc5_both = self.accuracy(logits_both, target, topk=(1, 5))

                # Accumulate accuracies
                top1_base += acc1_base
                top1_cupl += acc1_cupl
                top1_both += acc1_both

                top5_base += acc5_base
                top5_cupl += acc5_cupl
                top5_both += acc5_both
                
                n += images.size(0) 
            # Calculate average accuracies
            avg_top1_base = (top1_base / n) * 100
            avg_top1_cupl = (top1_cupl / n) * 100
            avg_top1_both = (top1_both / n) * 100

            avg_top5_base = (top5_base / n) * 100
            avg_top5_cupl = (top5_cupl / n) * 100
            avg_top5_both = (top5_both / n) * 100
            
            t1 = time.time()

                
        data.append([mn, pt, t1-t0, avg_top1_base,avg_top1_cupl,avg_top1_both,avg_top5_base,avg_top5_cupl,avg_top5_both])
        print(data)
        wandb.log(
            {'Result Table': wandb.Table(data=data, columns=columns)}
        )



class ImageEncoder(nn.Module): #encodes each image to a fixed size vector with the size of the model's output channels (in case of ResNet50 the vector size will be 2048)
    """
    Encode images to a fixed size vector
    """
    def __init__(self, model_name='resnet50', pretrained=True, trainable=True): #model_name='resnet50', pretrained=True, trainable=True
        super().__init__()
        #self.model = timm.create_model(model_name, pretrained, num_classes=0, global_pool="avg") #[batch_size, 2048, 1, 1] so  reduces the spatial dimensions to 1x1 because it is global
        img = Image.open(urlopen('https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/beignets-task-guide.png'))

        self.model = timm.create_model(
            'eva02_base_patch14_224.mim_in22k',
            pretrained=True,
            num_classes=0,  # remove classifier nn.Linear
            global_pool="avg"
        )
        self.model = self.model.eval()
        # get model specific transforms (normalization, resize)
        data_config = timm.data.resolve_model_data_config(self.model)
        transforms = timm.data.create_transform(**data_config, is_training=False)
        
        #num_classes=0 argument means that the model is created without a final classification layer => model as a feature extractor, then add a classification layer later
        for p in self.model.parameters():
            p.requires_grad = trainable

    def forward(self, x):
        return self.model(x)


def accuracy(output, target, topk=(1,)):
        pred = output.topk(max(topk), 1, True, True)[1].t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        return [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) for k in topk]


def cross_entropy(preds, targets, reduction='none'):
    loss = F.cross_entropy(preds, targets, reduction=reduction)
    return loss


#text: (batch_size, 768) to (batch_size, 256)
class ProjectionHead(nn.Module):
    def __init__(
        self,
        embedding_dim,
        projection_dim=256,
        dropout=0.4
    ):
        super().__init__()
        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(projection_dim, projection_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(projection_dim)

    def forward(self, x):
        projected = self.projection(x) 
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = self.layer_norm(x)
        x = x + projected
        return x


    
if __name__ == '__main__':  

    PATH_TO_PROMPTS = r"C:\Users\TFM\ObjectClassification\cifar100_prompts.json" #fill whith the cifar100_prompts.json full path
    with open(PATH_TO_PROMPTS) as f:
        gpt3_prompts = json.load(f)
    device = "cuda" if torch.cuda.is_available() else "cpu"  

    num_epochs = 100
    
    #With resnet 50
    text_embedding = 768 #output to vision encoder => print(text_features.shape)
    image_embedding = 2048 #in the case of resnet50 or vit_base_patch32_224
    
    #With eva02_base_patch14_224.mim_in22k
    text_embedding = 768 #output to vision encoder => print(text_features.shape)
    image_embedding = 768 #in the case of resnet50 or vit_base_patch32_224
    
    
    #zero_shot_weights [768,100] => [100,256]
    #text [batch_size, 768] => [batch_size, 256]
    #image [batch_size, 2048] => [batch_size, 256]

    
    image_encoder = ImageEncoder().cuda()
    
    data_module = CIFAR100DataModule(train_batch_size=120, val_batch_size=100, test_batch_size=100, num_workers=11)   
        
    image_projection = ProjectionHead(embedding_dim=image_embedding).to(device)

    text_projection = ProjectionHead(embedding_dim=text_embedding).to(device)

    # Load the model
    model, preprocess,_ = open_clip.create_model_and_transforms('ViT-L-14', 'datacomp_xl_s13b_b90k')
    
    # Access the text transformer blocks
    text_transformer_blocks = model.transformer.resblocks

    # Get the parameters of the last linear layer in the last transformer block
    last_transformer_block = text_transformer_blocks[-1]
    last_linear_layer_params = last_transformer_block.mlp.c_proj.parameters()

    # Enable gradients for the parameters of the last linear layer
    for param in last_linear_layer_params:
        param.requires_grad = True

    # Prepare the inputs
    data_module.setup(preprocess)
    train_dataloader = data_module.train_dataloader()
    val_dataloader = data_module.val_dataloader()
    test_dataloader = data_module.test_dataloader()
    
    model.cuda()
    
    topk = (1, 5) #top-1 and top-5 accuracy
    
    
    
    optimizer = torch.optim.Adam(list(image_projection.parameters()) + list(text_projection.parameters()), lr=0.001, weight_decay=1e-5)
    
    #scheduler = CosineAnnealingLR(optimizer, T_max=10)
    scheduler = StepLR(optimizer, step_size=30, gamma=0.1)

    #run_name = "ViT-L14-CIFAR100"+str(num_epochs)+"epochs"+str(optimizer)+str(scheduler)+str(image_projection)+str(text_projection)+str(image_encoder)+ "with_gpt3_prompts"

    wandb.init(project='finetuning')
    
    #emperature of 1.0 generally represents a balanced level of confidence and diversity in the generated predictions.
# Freeze all parameters except the last layer (c_proj) in the text encoder
    num_images_to_log=5
            
    with torch.no_grad():
        print("Training the model...")
        running_loss = 0.0
        patience = 5
        increasing_val_loss_counter = 0
        best_val_loss = 99999999999999999999999999999999999999999999999999999.
        train_losses = []
        val_losses = []
        average_val_losses = []
        average_train_losses = []
        for epoch in range(num_epochs):
            print(f"Epoch {epoch + 1}/{num_epochs}") 
            average_topk_accuracy, total_train_loss, n_train = 0., 0., 0.
            total_correct_predictions_train = 0
            total_correct_predictions_val = 0
            model.train()
            total_train_loss = 0.0
            
            for i, (images,target) in enumerate(tqdm(train_dataloader)):
                #requires_grad_() on the original target tensor before moving it to the GPU                
                images = images.cuda()
                target = target.clone().detach().float().requires_grad_(True).cuda()
                #target was not a floating so can not be requires_grad=True
                

                # Zero the gradients
                optimizer.zero_grad()

                image_features = image_encoder(images) #[batch_size, 2048]
                
                
                # Getting Image and Text Embeddings (with same dimension)
                image_embeddings = image_projection(image_features) #the output tensor's requires_grad attribute is True if it's computed from input tensors that require gradients
                #[batch_size, projected_dimension]
                # Randomly select one description for each target class
                text_embeddings = []
                for class_index in target:
                    class_name = data_module.cifar_classes[int(class_index.item())]
                    descriptions_for_class = gpt3_prompts[class_name.replace('_', ' ')]
                    selected_description = random.choice(descriptions_for_class)
                    text_embedding = model.encode_text(open_clip.tokenize([selected_description]).cuda())
                    text_embeddings.append(text_embedding.squeeze()) # Squeeze the extra dimension
                text_embeddings = torch.stack(text_embeddings) #[batch, 768]

                text_embeddings = text_projection(text_embeddings)
                
                logits = (text_embeddings @ image_embeddings.T) #logits are not the class indexes
                logits.requires_grad_(True)  # Set requires_grad=True for the logits tensor


                images_similarity = image_embeddings @ image_embeddings.T
                texts_similarity = text_embeddings @ text_embeddings.T

                pred = F.softmax((images_similarity + texts_similarity) / 2, dim=-1)
                pred.requires_grad_(True)

                
                n_train += images.size(0)
                
                images_loss = (-pred.T * nn.LogSoftmax(dim=-1)(logits.T)).sum(1)
                texts_loss = (-pred * nn.LogSoftmax(dim=-1)(logits)).sum(1)
                loss = (images_loss + texts_loss) / 2.0


                loss = loss.mean() #not the sum as i want the learning rate to be relatively insensitive to the batch size
                loss.requires_grad_(True)
                total_train_loss += loss.item()
                #as the text encoder is the one of the model, i am not calculating gradients for it
                
                # Backward pass and optimize
                loss.backward()
                optimizer.step()  

            average_train_loss = (total_train_loss / len(train_dataloader)) * 100

            average_train_losses.append(average_train_loss)
            wandb.log({ "train_loss_per_epoch": average_train_loss})
            print("Training loss: ", total_train_loss)
            print("Average training loss: ", average_train_loss)

            scheduler.step()
            # VALIDATION LOOP
            print ("Validation set")
            model.eval()
            val_loss = 0
            n_val = 0
            top1_val, total_val_loss, n_val,top5_val,average_val_loss = 0., 0., 0.,0.,0.
            
            for i, (images,target) in enumerate(tqdm(val_dataloader)):
                
                images = images.cuda()
                target = target.clone().detach().float().requires_grad_(True).cuda()
                
                optimizer.zero_grad()

                image_features = image_encoder(images) 
                image_embeddings = image_projection(image_features)
                text_embeddings = []
                
                for class_index in target:
                    class_name = data_module.cifar_classes[int(class_index.item())]
                    descriptions_for_class = gpt3_prompts[class_name.replace('_', ' ')]
                    selected_description = random.choice(descriptions_for_class)
                    text_embedding = model.encode_text(open_clip.tokenize([selected_description]).cuda())
                    text_embeddings.append(text_embedding.squeeze()) 
                text_embeddings = torch.stack(text_embeddings) 
                text_embeddings = text_projection(text_embeddings)

                logits = (text_embeddings @ image_embeddings.T)
                logits.requires_grad_(True)  # Set requires_grad=True for the logits tensor


                images_similarity = image_embeddings @ image_embeddings.T
                texts_similarity = text_embeddings @ text_embeddings.T
                

                pred = F.softmax((images_similarity + texts_similarity) / 2, dim=-1)
                pred.requires_grad_(True)

                n_val += images.size(0)


                images_loss = (-pred.T * nn.LogSoftmax(dim=-1)(logits.T)).sum(1)
                texts_loss = (-pred * nn.LogSoftmax(dim=-1)(logits)).sum(1)
                loss =  (images_loss + texts_loss) / 2.0 # similar goal to negative log-likelihood
                
                loss = loss.mean() #not the sum as i want the learning rate to be relatively insensitive to the batch size
                loss.requires_grad_(True)
                total_val_loss += loss.item()
                
                loss.backward()
                optimizer.step()

            average_val_loss = (total_val_loss / len(val_dataloader)) * 100

            print("Average validation loss: ", average_val_loss)

            wandb.log({ "val_loss_per_epoch": average_val_loss})
            
            print(f"Epoch {epoch+1}, Training Loss: {total_train_loss/len(train_dataloader)}, Validation Loss: {total_val_loss/len(val_dataloader)}")
    
            # # Overfitting detection
            # if epoch > 1 and average_val_losses[epoch-1] > average_val_losses[epoch-2] and average_train_losses[epoch-1] < average_train_losses[epoch-2]:
            #     increasing_val_loss_counter += 1
            #     if increasing_val_loss_counter >= patience:
            #         print("Overfitting detected . Training loss is decreasing but validation loss is increasing.")
            #         break
            # else:
            #     increasing_val_loss_counter = 0
            # # Early stopping
            # if average_val_loss < best_val_loss:
            #     best_val_loss = average_val_loss
            #     # Specify the path to save the model weights
            #     directory = r'C:\Users\TFM\ObjectClassification\model_weights\validation' #fill with appropiate full path
            #     os.makedirs(directory, exist_ok=True)
            #     best_save_path = os.path.join(directory, 'best_val_loss.pth')
            #     torch.save(model.state_dict(), best_save_path)
            # else:
            #     print("Early stopping due to validation loss not decreasing")
            #     break
            # scheduler.step()

        
        # Save the model and specify the epoch and the path 
        # Define the directory where the model weights will be saved
        save_dir = r'C:\Users\TFM\ObjectClassification\model_weights\train' #fill with appropiate full path
        os.makedirs(save_dir, exist_ok=True)
        # File path for saving model weights for each epoch
        epoch_save_path = os.path.join(save_dir, f'model_weights_epoch{epoch}.pth')
        torch.save(model.state_dict(), epoch_save_path)
        
    #TEST LOOP
    # Load the saved model weights
    model.load_state_dict(torch.load(r'C:\Users\TFM\ObjectClassification\model_weights\validation\best_val_loss.pth')) #fill with appropiate full path

    
    models = CLIPModel(model, data_module, data_module.cifar_classes, batch_size=512)
    models.evaluate_models()

        
    wandb.finish()
    
    
    
