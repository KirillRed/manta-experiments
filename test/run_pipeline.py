import argparse
import subprocess
import shlex

parser = argparse.ArgumentParser()

parser.add_argument("--dataset", type=str, default="50salads")
parser.add_argument("--file", type=str)
parser.add_argument("--ignore_action", type=str)
parser.add_argument("--split", type=int, default=1)

args = parser.parse_args()

dataset = "50salads" if args.dataset.startswith("5") else "breakfast"
print(dataset)

with open("/data1/kredensk/anticipation/manta/test/test.bundle", "w") as f:
    f.write(f"{args.file}\n")

with open(f"/data1/kredensk/anticipation/manta/datasets/breakfast/data/{dataset}/mapping.txt", "r") as f:
    for line in f:
        tokens = line.split()
        if args.ignore_action == tokens[1]:
            ignore_action_idx = int(tokens[0])

command_string = """python /data1/kredensk/anticipation/manta/src/main.py \
    --vid_list_file_test=/data1/kredensk/anticipation/manta/test/test.bundle \
    --ds=bf \
    --conditioned_x0 \
    --use_features \
    --use_inp_ch_dropout \
    --part_obs \
    --layer_type mamba \
    --num_diff_timesteps 1000 \
    --diff_obj pred_x0 \
    --num_samples 25 \
    --action=val \
    --bz=1 \
    --model=bit-diff-pred-tcn \
    --num_epochs=100 \
    --epoch=90 \
    --num_stages=1 \
    --num_layers=15 \
    --channel_dropout_prob=0.4 \
    --sample_rate=3"""

shell_args = shlex.split(command_string)

shell_args.append(f"--split={args.split}")

shell_args.append(f'--mapping_file=/data1/kredensk/anticipation/manta/datasets/breakfast/data/{dataset}/mapping.txt')
shell_args.append(f'--vid_list_file=/data1/kredensk/anticipation/manta/datasets/breakfast/data/{dataset}/splits/train.split{args.split}.bundle')
shell_args.append(f'--gt_path=/data1/kredensk/anticipation/manta/datasets/breakfast/data/{dataset}/groundTruth/')
shell_args.append(f'--features_path=/data1/kredensk/anticipation/manta/datasets/breakfast/data/{dataset}/features/')

if dataset=='50salads':
    print("50salads Detected")
    shell_args.append('--num_infr_diff_timesteps')
    shell_args.append('10')
    shell_args.append('--lr=0.001')
    if args.split==1:
        print("Split 1")
        shell_args.append('--model_dir=/data1/kredensk/anticipation/manta/normal_train/50salads/checkpoints')
        shell_args.append('--results_dir=/data1/kredensk/anticipation/manta/normal_train/50salads/checkpoints')
    else:
        print(f"Split {args.split}")
        shell_args.append(f'--model_dir=/data1/kredensk/anticipation/manta/baseline/split{args.split}/50salads/checkpoints')
        shell_args.append(f'--results_dir=/data1/kredensk/anticipation/manta/baseline/split{args.split}/50salads/checkpoints')

else:
    print("Probably breakfast Detected")
    shell_args.append('--num_infr_diff_timesteps')
    shell_args.append('50')
    shell_args.append('--lr=0.005')
    shell_args.append('--model_dir=/data1/kredensk/anticipation/manta/normal_train/breakfast/checkpoints')
    shell_args.append('--results_dir=/data1/kredensk/anticipation/manta/normal_train/breakfast/checkpoints')

try:
    baseline = subprocess.run(shell_args, capture_output=True, text=True, check=True)
    print(baseline.stdout)
except subprocess.CalledProcessError as e:
    print(f"Error: The script failed with exit code {e.returncode}")
    print(f"Error message:\n{e.stderr}")

shell_args.append("--qualitative")
shell_args.append(f"--ignore_action={ignore_action_idx}")

try:
    qualitative = subprocess.run(shell_args, capture_output=True, text=True, check=True)
    print(qualitative.stdout)
except subprocess.CalledProcessError as e:
    print(f"Error: The script failed with exit code {e.returncode}")
    print(f"Error message:\n{e.stderr}")

visualize_string = f"""python /data1/kredensk/anticipation/manta/test/visualize.py \
    --dataset={dataset} \
    --action={args.ignore_action}"""

visualize_args = shlex.split(visualize_string)

try:
    visualize = subprocess.run(visualize_args, capture_output=True, text=True, check=True)
    print(visualize.stdout)
except subprocess.CalledProcessError as e:
    print(f"Error: The script failed with exit code {e.returncode}")
    print(f"Error message:\n{e.stderr}")
