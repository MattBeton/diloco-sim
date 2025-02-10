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
    devices = '0 1 2 3'
    num_nodes = 4

    global_command = f'python transformer.py --train --port 12355 --wandb_project owt_diloco_H_correlation --model_size base --batch_size 16 --max_local_step 10000 --corr_interval 200 --num_nodes {num_nodes} --devices {devices}'

    H = 5000
    command1 = f' --p_sparta 0 --diloco_interval {H} --wandb_name dlc{H}'
    run_command(f'{global_command} {command1}')

    command2 = f'--p_sparta {1/H} --diloco_interval {H} --wandb_name dlc{H}_p{1/H}'
    run_command(f'{global_command} {command2}')


    devices = '0 1 2 3 4 5 6 7'
    num_nodes = 16

    global_command = f'python transformer.py --train --port 12355 --wandb_project owt_diloco_H_correlation --model_size base --batch_size 16 --max_local_step 10000 --corr_interval 200 --num_nodes {num_nodes} --devices {devices}'

    H = 5000
    command1 = f' --p_sparta 0 --diloco_interval {H} --wandb_name n16_dlc{H}'
    run_command(f'{global_command} {command1}')

    command2 = f'--p_sparta 0.0005 --diloco_interval {H} --wandb_name n16_dlc{H}_p0.0005'
    run_command(f'{global_command} {command2}')


if __name__ == "__main__":
    main()