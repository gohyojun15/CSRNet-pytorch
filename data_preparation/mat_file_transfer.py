import argparse
import os
from tqdm import tqdm
import shutil



# mat file 옮겨야하는 경우 사용하는 것임.

parser = argparse.ArgumentParser(description=".mat file transfer to other folder")

# target folder
parser.add_argument("--root",type=str,help="folder")
# output folder
parser.add_argument("--destiny",type=str,help="output folder")





if __name__ =="__main__":
    args = parser.parse_args()
    if not os.path.exists(args.destiny):
        os.mkdir(args.destiny)
    all_file_list = os.listdir(args.root)
    index = 0
    for file in tqdm(all_file_list):
        file_path = args.root + "/" +file

        # check is matfile
        if ".mat" in file_path:
            # transfer
            target_file_path = args.destiny + "/" + file
            shutil.move(file_path, target_file_path)
