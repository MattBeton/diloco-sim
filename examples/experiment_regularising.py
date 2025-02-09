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
    global_command = 'python transformer.py --cosine_anneal --train --port 12355 --wandb_project owt_diloco++_regularising_full --model_size base --batch_size 16 --max_local_step 30000 --corr_interval 100000'

    devices = '0 1 2 3'
    num_nodes = 4
    diloco_interval = 100

    baseline_command = f'--num_nodes 1 --devices {devices[0]} --p_sparta 0 --diloco_interval 100000 --learning_rate 0.0004 --wandb_name baseline_n1'
    run_command(f'{global_command} {baseline_command}')

    command1 = f'--num_nodes {num_nodes} --devices {devices} --p_sparta 0 --diloco_interval {diloco_interval} --learning_rate 0.0004 --wandb_name dlc{diloco_interval}'
    run_command(f'{global_command} {command1}')

    command2 = f'--num_nodes {num_nodes} --devices {devices} --p_sparta 0.005 --diloco_interval {diloco_interval} --learning_rate 0.0004 --wandb_name dlc{diloco_interval}_p0.005'
    run_command(f'{global_command} {command2}')

    command3 = f'--num_nodes {num_nodes} --devices {devices} --p_sparta 0.005 --diloco_interval {diloco_interval} --learning_rate 0.0005 --wandb_name dlc{diloco_interval}_p0.005_lr25%'
    run_command(f'{global_command} {command3}')

    command4 = f'--num_nodes {num_nodes} --devices {devices} --p_sparta 0.005 --diloco_interval {diloco_interval} --learning_rate 0.0006 --wandb_name dlc{diloco_interval}_p0.005_lr50%'
    run_command(f'{global_command} {command4}')

    command4 = f'--num_nodes {num_nodes} --devices {devices} --p_sparta 0.005 --diloco_interval {diloco_interval} --learning_rate 0.0007 --wandb_name dlc{diloco_interval}_p0.005_lr75%'
    run_command(f'{global_command} {command4}')

    command4 = f'--num_nodes {num_nodes} --devices {devices} --p_sparta 0.005 --diloco_interval {diloco_interval} --learning_rate 0.0009 --wandb_name dlc{diloco_interval}_p0.005_lr125%'
    run_command(f'{global_command} {command4}')


if __name__ == "__main__":
    main()