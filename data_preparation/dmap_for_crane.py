### this file is made by gohyojun
import argparse
import os
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.ndimage.filters import gaussian_filter
import numpy as np

from tqdm import tqdm
"""
Density map input out에 대한 입력 출력 간의 관계
input : main argument in this file
output: denstiy map 
"""

parser = argparse.ArgumentParser(description="generate density map for crane")
# image root
parser.add_argument("--image_root",type=str,help="image data root")
# ground truth root (대가리에 점찍은거 어디있는지)
parser.add_argument("--ground_truth_root",type=str,help="ground truth root")
# output root. densitiy map 어디다 저장할지 (생성해줌)
parser.add_argument("--density_map_root",type=str,help="output densitiy map root")


"""""
argument_example
--image_root /home/gohyojun/바탕화면/Anthroprocene/Dataset/crane --ground_truth_root /home/gohyojun/바탕화면/Anthroprocene/Dataset/crane_labeled --density_map_root /home/gohyojun/바탕화면/Anthroprocene/Dataset/density_map 
"""


def generate_fixed_kernel_densitymap(image,points,sigma=15):
    '''
    Use fixed size kernel to construct the ground truth density map
    for ShanghaiTech PartB.
    image: the image with type numpy.ndarray and [height,width,channel].
    points: the points corresponding to heads with order [col,row].
    sigma: the sigma of gaussian_kernel to simulate a head.
    '''
    # the height and width of the image
    image_h = image.shape[0]
    image_w = image.shape[1]

    # coordinate of heads in the image
    points_coordinate = points
    # quantity of heads in the image
    points_quantity = len(points_coordinate)

    # generate ground truth density map
    densitymap = np.zeros((image_h, image_w))
    for point in points_coordinate:
        c = min(int(round(point[0])),image_w-1)
        r = min(int(round(point[1])),image_h-1)
        # point2density = np.zeros((image_h, image_w), dtype=np.float32)
        # point2density[r,c] = 1
        densitymap[r,c] = 1
    # densitymap += gaussian_filter(point2density, sigma=sigma, mode='constant')
    densitymap = gaussian_filter(densitymap, sigma=sigma, mode='constant')

    densitymap = densitymap / densitymap.sum() * points_quantity
    return densitymap

if __name__ == '__main__':

    args = parser.parse_args()
    # TODO Training test
    # phase_list = ['train','test']

    if not os.path.exists(args.density_map_root):
        os.mkdir(args.density_map_root)
    image_file_list = os.listdir(args.image_root)

    index = 0
    for image_file in tqdm(image_file_list):
        image_path = args.image_root+"/" + image_file

        if not os.path.isfile(image_path):
            continue

        # FIXME
        # math file rule이 같은 이름에 .jpg를 .mat 로
        mat_path = args.ground_truth_root+ "/" + image_file
        mat_path = mat_path[0:-3] + "mat"
        image = plt.imread(image_path)
        # todo 이거 연관성 써놔야함
        # todo 모든 클래스에대해서 어떻게 저장할지도 생각해놔야함
        # densitymap root
        density_path = args.density_map_root + "/" + image_file
        density_path = density_path[0:-3] + "npy"
        ################
        """
        matlab file debugging.
        
        in matfile
        
        head_class0 : 두루미 성조 
        head_class1
        head_class2
        head_class3
        head_class4
        body_class5
        body_class6
        body_class7
        body_class8
        body_class9
        """
        ################


        mat = loadmat(mat_path)
        points = mat["class0"][0][0][0][0][0]







        # points = mat['head_class0'][0][0][0][0][0]
        densitymap = generate_fixed_kernel_densitymap(image, points, sigma=15)
        np.save(density_path,densitymap)

