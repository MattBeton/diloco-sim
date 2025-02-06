import asyncio
import subprocess
import sys

def run_command(command):
    """Run a shell command and print the output in real-time."""
    print(f'Running command: {command}')
    
    # Start the process
    process = subprocess.Popen(
        command,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    # Read and print stdout and stderr in real-time
    while True:
        stdout_line = process.stdout.readline()
        stderr_line = process.stderr.readline()
        
        if stdout_line:
            print(stdout_line, end='')
        
        if stderr_line:
            print(stderr_line, end='', file=sys.stderr)
        
        # Check if the process has finished
        if stdout_line == '' and stderr_line == '' and process.poll() is not None:
            break
    
    # Check the return code
    if process.returncode != 0:
        print(f"Command failed: {command}")
    else:
        print(f"Command succeeded: {command}")

def main():
    N_list = [4, 2, 1]  # Example values for grid search
    for N in N_list:
        command3 = f"python transformer.py --train --cosine_anneal --num_nodes {N} --devices 0 1 --batch_size 16 --p_sparta 0.0005 --diloco_interval 100 --max_local_step 30000 --wandb_project owt_diloco_n --wandb_name n{N}_p0.0005 --port 12356"
        run_command(command3)
        break


if __name__ == "__main__":
    main()