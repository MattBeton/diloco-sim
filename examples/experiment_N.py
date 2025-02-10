import asyncio
import subprocess
import sys

def run_command(command):
    """Run a shell command and let it output directly to the terminal."""
    print(f'Running command: {command}', flush=True)
    return
    try:
        # Let stdout/stderr go directly to the terminal without buffering
        subprocess.run(
            command,
            shell=True,
            check=True,
            # Add these for line-buffered output if needed (Unix only)
            # bufsize=1,
            # universal_newlines=True,
        )
        print(f"Command succeeded: {command}", flush=True)
    except subprocess.CalledProcessError as e:
        print(f"Command failed with code {e.returncode}: {command}", flush=True)
        sys.exit(1)

def main():
    devices = '0 1 2 3 4 5 6 7'

    global_command = f'python transformer.py --cosine_anneal --train --port 12355 --wandb_project owt_diloco_N --model_size base --batch_size 16 --max_local_step 30000 --corr_interval 100000 --devices {devices}'

    H = 100

    for num_nodes in [4, 16]:
        H = 100
        command1 = f' --num_nodes {num_nodes} --p_sparta 0 --diloco_interval {H} --wandb_name n{num_nodes}_dlc{H}'
        run_command(f'{global_command} {command1}')

        H = 10000
        command2 = f' --num_nodes {num_nodes} --p_sparta 0.005 --learning_rate 0.0008 --diloco_interval {H} --wandb_name n{num_nodes}_dlc{H}_p0.005_lr100%'
        run_command(f'{global_command} {command2}')


if __name__ == "__main__":
    main()