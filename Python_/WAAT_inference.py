import os
import argparse
from tqdm import tqdm

import numpy as np
from PIL import Image

import torchvision
import torch
from torch.utils.data import Dataset

from pascal_voc_writer import Writer

import matplotlib.image as mpimg
import matplotlib.patches as patches
import matplotlib.pyplot as plt

class MyDataset(Dataset):
  def __init__(self, path, transform=None):
    self.path = path
    self.imgs = list(sorted(os.listdir(self.path)))
    self.transform = transform
        
  def __len__(self):
    return len(self.imgs)

  def __getitem__(self, idx):
    file_image = self.imgs[idx]
    img_path = os.path.join(self.path, file_image)

    img = Image.open(img_path).convert("RGB")   
    to_tensor = torchvision.transforms.ToTensor()
    img = to_tensor(img)

    return img, None
  
def collate_fn(batch):
  return tuple(zip(*batch))

def make_prediction(model, img, threshold):
  model.eval()
  preds = model(img)

  for id in range(len(preds)):
    idx_list = list()

    for idx, score in enumerate(preds[id]['scores']):
      if score > threshold:
        idx_list.append(idx)
      
    preds[id]['boxes'] = preds[id]['boxes'][idx_list]
    preds[id]['labels'] = preds[id]['labels'][idx_list]
    preds[id]['scores'] = preds[id]['scores'][idx_list]
    
  return preds


if __name__ == '__main__':

  parser = argparse.ArgumentParser()
  parser.add_argument('--image_dir', default='./images', help='Dir Inference_Img')
  parser.add_argument('--model_path', default=None, help='Model path')
  parser.add_argument('--annotation_dir', default='./annotations', help='Dir to save annotations')

  args = parser.parse_args()

  img_path = args.image_dir
  model_path = args.model_path
  annotation_path = args.annotation_dir

  labels = list()
  preds_adj_all = list()
  annot_all = list()

  dataset = MyDataset(img_path)

  device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
  data_loader = torch.utils.data.DataLoader(dataset, batch_size=4, collate_fn=collate_fn)
 

  if model_path:
    model = torch.load(model_path)
  else:
    model = torchvision.models.detection.ssd300_vgg16(pretrained=True, pretrained_backbone = True)

  model.to(device)

  idx = 0
  for im, _ in tqdm(data_loader, position = 0, leave = True):
    im = list(img.to(device) for img in im)

    with torch.no_grad():
      preds_adj = make_prediction(model, im, 0.5)
      preds_adj = [{k: v.to(torch.device('cpu')) for k, v in t.items()} for t in preds_adj]
    
    for i, image in enumerate(im):
      # create pascal voc writer (image_path, width, height)
      writer = Writer(data_loader.dataset.path+"/"+data_loader.dataset.imgs[idx], len(image[0][0]), len(image[0]))


      # add objects (class, xmin, ymin, xmax, ymax)
      for object_idx in range(len(preds_adj[i]['boxes'])):
        writer.addObject(preds_adj[i]['labels'][object_idx].item(), preds_adj[i]['boxes'][object_idx][0].item(), preds_adj[i]['boxes'][object_idx][1].item(), preds_adj[i]['boxes'][object_idx][2].item(), preds_adj[i]['boxes'][object_idx][3].item())

      writer.save(annotation_path+'/'+data_loader.dataset.imgs[idx]+'.xml')
      idx += 1
