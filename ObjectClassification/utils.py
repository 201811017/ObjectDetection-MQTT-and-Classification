from torchvision.datasets import CIFAR100
from torchvision import transforms
from pytorch_lightning import LightningModule
from torch.utils.data import DataLoader, random_split
from fastai.vision.all import *
from tqdm import tqdm
import open_clip

#pip install git+https://github.com/openai/CLIP.git
#pip install git+https://github.com/openai/CLIP.git
#pip install ftfy regex tqdm

class CIFAR100DataModule(LightningModule): #saved as binary files
    def __init__(
        self,
        data_dir: str = './',
        train_batch_size: int = 100,
        val_batch_size: int = 100,
        test_batch_size: int = 100,
        num_workers: int = 11,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.data_dir = data_dir
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.test_batch_size = test_batch_size
        self.num_workers = num_workers
        # Define transforms
        self.train_transforms = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),  # Convert PIL Image to tensor
        ])
        self.val_transforms = transforms.Compose([
            transforms.ToTensor(),  # Convert PIL Image to tensor
        ])
        # Load CIFAR100 data to get class names
        cifar100_temp = CIFAR100(self.data_dir, train=True, download=True)
        self.cifar_classes = list(cifar100_temp.class_to_idx.keys())  
    def prepare_data(self):
        # Download the data
        CIFAR100(self.data_dir, train=True, download=True)
        CIFAR100(self.data_dir, train=False, download=True)

    def setup(self, preprocess, stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage == 'fit' or stage is None:
            cifar_full = CIFAR100(self.data_dir, train=True, transform=preprocess)
            self.cifar_train, self.cifar_val = random_split(cifar_full, [45000, 5000]) #training,validation

        # Assign test dataset for use in dataloader(s)
        if stage == 'test' or stage is None:
            self.cifar_test = CIFAR100(self.data_dir, train=False, transform=preprocess)

    def train_dataloader(self):
        return DataLoader(self.cifar_train, batch_size=self.train_batch_size, num_workers=self.num_workers, persistent_workers=True)
    #tuple (images,labels) where images is a tensor of shape (batch_size, 3, 32, 32)
    #CIFAR100 images are 32x32 color images

    def val_dataloader(self):
        return DataLoader(self.cifar_val, batch_size=self.val_batch_size, num_workers=self.num_workers, persistent_workers=True)

    def test_dataloader(self):
        return DataLoader(self.cifar_test, batch_size=self.test_batch_size, num_workers=self.num_workers, persistent_workers=True)

cifar100_templates =[
            'a photo of a {}.',
            'a blurry photo of a {}.',
            'a black and white photo of a {}.',
            'a low contrast photo of a {}.',
            'a high contrast photo of a {}.',
            'a bad photo of a {}.',
            'a good photo of a {}.',
            'a photo of a small {}.',
            'a photo of a big {}.',
            'a photo of the {}.',
            'a blurry photo of the {}.',
            'a black and white photo of the {}.',
            'a low contrast photo of the {}.',
            'a high contrast photo of the {}.',
            'a bad photo of the {}.',
            'a good photo of the {}.',
            'a photo of the small {}.',
            'a photo of the big {}.',
        ]

#Cifar dataset
def zeroshot_classifier_cifar(model,classes,templates):
    with torch.no_grad():
        zeroshot_weights = []
        for classname in tqdm(classes):
            texts = [template.format(classname) for template in templates] #format with class
            texts = open_clip.tokenize(texts).cuda() #tokenize
            class_embeddings = model.encode_text(texts) #embed with text encoder
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).cuda() #stored in first GPU

    return zeroshot_weights
#Imagenet dataset   
def zeroshot_classifier_imagenet(model,classnames, textnames, templates):
    with torch.no_grad():
        zeroshot_weights = []
        i = 0
        for classname in tqdm(classnames):
            texts = [template.format(textnames[i]) for template in templates] #format with class
            texts = clip.tokenize(texts).cuda() #tokenize
            class_embeddings = model.encode_text(texts) #embed with text encoder
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
            i += 1
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).cuda()
        return zeroshot_weights


def zeroshot_classifier_gpt(model,classnames, templates, use_both, PATH_TO_PROMPTS, line = False):
    with open(PATH_TO_PROMPTS) as f:
        gpt3_prompts = json.load(f)

    with torch.no_grad():
        zeroshot_weights = []

        for classname in tqdm(classnames):
            if use_both:
                texts = [template.format(classname) for template in templates]
            else:
                texts = []
            if not line:
                classname = classname.replace('_', ' ')
            for t in gpt3_prompts[classname]:
                texts.append(t)
            texts = open_clip.tokenize(texts).cuda() #tokenize
            class_embeddings = model.encode_text(texts) #embed with text encoder
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)

        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).cuda() #each input text gives a feature vector of 100 (n_classes)
    return zeroshot_weights

