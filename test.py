import argparse

import torch
import matplotlib.pyplot as plt
import matplotlib.cm as CM
from tqdm import tqdm

from config import Config
from model import CSRNet
from dataset import create_train_dataloader, create_test_dataloader, CrowdDataset
from utils import denormalize

parser = argparse.ArgumentParser(description="generate density map for crane")


#train datasets
parser.add_argument("--train_image_root",type=str,help="image data root")
parser.add_argument("--train_image_gt_root",type=str,help="ground truth root")
parser.add_argument("--train_image_density_root",type=str,help="density map root.")
# test datasets
parser.add_argument("--test_image_root",type=str,help="image data root")
parser.add_argument("--test_image_gt_root",type=str,help="ground truth root")
parser.add_argument("--test_image_density_root",type=str,help="density map root.")



def cal_mae(img_root,model_param_path):
    '''
    Calculate the MAE of the test data.
    img_root: the root of test image data.
    gt_dmap_root: the root of test ground truth density-map data.
    model_param_path: the path of specific mcnn parameters.
    '''
    device=torch.device("cuda")
    model=CSRNet()
    model.load_state_dict(torch.load(model_param_path))
    model.to(device)
    dataset=create_test_dataloader(img_root)
    dataloader=torch.utils.data.DataLoader(dataset,batch_size=1,shuffle=False)
    model.eval()
    mae=0
    with torch.no_grad():
        for i, data in enumerate(tqdm(dataloader)):
            image = data['image'].cuda()
            gt_densitymap = data['densitymap'].cuda()
            # forward propagation
            et_dmap=model(image)
            mae+=abs(et_dmap.data.sum()-gt_densitymap.data.sum()).item()
            del image,gt_densitymap,et_dmap
    print("model_param_path:"+model_param_path+" mae:"+str(mae/len(dataloader)))

def estimate_density_map(img_root,model_param_path,index):
    '''
    Show one estimated density-map.
    img_root: the root of test image data.
    gt_dmap_root: the root of test ground truth density-map data.
    model_param_path: the path of specific mcnn parameters.
    index: the order of the test image in test dataset.
    '''
    device=torch.device("cuda")
    model=CSRNet().to(device)
    model.load_state_dict(torch.load(model_param_path))
    dataset=CrowdDataset(img_root)
    dataloader=torch.utils.data.DataLoader(dataset,batch_size=1,shuffle=False)
    model.eval()
    for i,(img,gt_dmap) in enumerate(dataloader):
        if i==index:
            img=img.to(device)
            gt_dmap=gt_dmap.to(device)
            # forward propagation
            et_dmap=model(img).detach()
            et_dmap=et_dmap.squeeze(0).squeeze(0).cpu().numpy()
            print(et_dmap.shape)
            plt.imshow(et_dmap,cmap=CM.jet)
            break


if __name__=="__main__":
    args = parser.parse_args()

    torch.backends.cudnn.enabled=False

    test_dataset_root = [
        args.test_image_root,
        args.test_image_gt_root,
        args.test_image_density_root
    ]

    model_param_path='./checkpoints/346.pth'
    cal_mae(test_dataset_root,model_param_path)
    # estimate_density_map(img_root,gt_dmap_root,model_param_path,3) 