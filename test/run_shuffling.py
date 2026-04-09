import argparse
import subprocess
import shlex

parser = argparse.ArgumentParser()

parser.add_argument("--dataset", type=str, default="breakfast")
parser.add_argument("--file", type=str)
parser.add_argument("--split", type=int, default=1)
parser.add_argument('--reverse', dest='reverse', action='store_true')
parser.add_argument("--activity", type=str, default="none")
parser.set_defaults(reverse=False)
parser.add_argument('--shuffle_full', dest='shuffle_full', action='store_true')
parser.set_defaults(shuffle_full=False)


args = parser.parse_args()

dataset = "50salads" if args.dataset.startswith("5") else "breakfast"
print(dataset)

with open("/data1/kredensk/anticipation/manta/test/test.bundle", "w") as f:
    f.write(f"{args.file}\n")

command_string = """python /data1/kredensk/anticipation/manta/src/main.py \
    --vid_list_file_test=/data1/kredensk/anticipation/manta/test/test.bundle \
    --ds=bf \
    --conditioned_x0 \
    --use_features \
    --use_inp_ch_dropout \
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
    shell_args.append('--model_dir=/data1/kredensk/anticipation/manta/train_checkpoints/fullsample/breakfast/checkpoints')
    shell_args.append('--results_dir=/data1/kredensk/anticipation/manta/train_checkpoints/fullsample/checkpoints')

try:
    baseline = subprocess.run(shell_args, capture_output=True, text=True, check=True)
    print(baseline.stdout)
except subprocess.CalledProcessError as e:
    print(f"Error: The script failed with exit code {e.returncode}")
    print(f"Error message:\n{e.stderr}")

shell_args.append("--shuffling")
if args.reverse:
    shell_args.append("--reverse")
if args.shuffle_full:
    shell_args.append("--shuffle_full")

try:
    shuffling = subprocess.run(shell_args, capture_output=True, text=True, check=True)
    print(shuffling.stdout)
except subprocess.CalledProcessError as e:
    print(f"Error: The script failed with exit code {e.returncode}")
    print(f"Error message:\n{e.stderr}")

filname = f"{args.activity}/{args.file}"
if args.shuffle_full:
    filname = f"{args.activity}/full/{args.file}"

visualize_string = f"""python /data1/kredensk/anticipation/manta/test/visualize.py \
    --dataset={dataset} \
    --test_type=1 \
    --name={filname}"""

visualize_args = shlex.split(visualize_string)

try:
    visualize = subprocess.run(visualize_args, capture_output=True, text=True, check=True)
    print(visualize.stdout)
except subprocess.CalledProcessError as e:
    print(f"Error: The script failed with exit code {e.returncode}")
    print(f"Error message:\n{e.stderr}")
