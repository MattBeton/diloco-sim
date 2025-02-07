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
    # TODO: cosine annealing
    global_command = 'python transformer.py --train --port 12355 --wandb_project owt_diloco_n --batch_size 16 --max_local_step 100 --corr_inteval 100000'

    devices = '0 1 2 3'
    num_nodes = 4
    diloco_interval = 100

    for n in [2, 4, 8, 16]:
        command1 = f'--num_nodes 1 --devices {devices} --p_sparta 0 --diloco_interval {diloco_interval} --wandb_name '
        run_command(f'{global_command} {command1}')

        command2 = f'--num_nodes {num_nodes} --devices {devices} --p_sparta 0.002 --diloco_interval {diloco_interval} --wandb_name dlc{diloco_interval}'
        run_command(f'{global_command} {command2}')



if __name__ == "__main__":
    main()