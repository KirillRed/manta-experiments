import argparse
import subprocess
import os
import re
import time
import shlex

def get_target_files(data_dir, activity, p_start, p_end):
    target_files = []
    seen_p_nums = set()  
    
    try:
        all_files = os.listdir(data_dir)
    except FileNotFoundError:
        print(f"Error: Could not find directory '{data_dir}'")
        return []
    
    all_files.sort() 
    
    for filename in all_files:
        match = re.match(r'^P(\d+)_.*?_P\d+_(.*?)(?:\.\w+)?$', filename)
        
        if match:
            p_num = int(match.group(1))
            file_act = match.group(2)
            
            if (p_start <= p_num <= p_end) and (file_act == activity) and (p_num not in seen_p_nums):
                
                target_files.append(filename)
                seen_p_nums.add(p_num)  
                
    return target_files

parser = argparse.ArgumentParser()

parser.add_argument("--dataset", type=str, default="breakfast")
parser.add_argument("--activity", type=str, default="bruh")
parser.add_argument("--test_type", type=str, default="shuffling")
parser.add_argument('--shuffle_full', dest='shuffle_full', action='store_true')
parser.set_defaults(shuffle_full=False)

args = parser.parse_args()

gt_directory = f"/data1/kredensk/anticipation/manta/datasets/breakfast/data/{args.dataset}/groundTruth" 
start_number = 0
end_number = 15   

my_file_list = get_target_files(gt_directory, args.activity, start_number, end_number)

os.makedirs(f"./test/results/{args.activity}", exist_ok=True)
if args.shuffle_full:
    os.makedirs(f"./test/results/{args.activity}/full", exist_ok=True)

for file_name in my_file_list:
    test_string = f"""python /data1/kredensk/anticipation/manta/test/run_{args.test_type}.py \
    --dataset={args.dataset} \
    --file={file_name} \
    --activity={args.activity}"""

    test_args = shlex.split(test_string)
    if args.shuffle_full:
        test_args.append("--shuffle_full")
    try:
        test = subprocess.run(test_args, capture_output=True, text=True, check=True)
        print(test.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Error: The script failed with exit code {e.returncode}")
        print(f"Error message:\n{e.stderr}")

    time.sleep(1)