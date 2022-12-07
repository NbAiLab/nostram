import sys
import subprocess
from joblib import Parallel, delayed
import argparse

def main(args):
    # Read the input shell script
    try:
        with open(args.input_shell_script, 'r') as f:
            commands = f.readlines()
    except FileNotFoundError:
        print(f'Error: Input shell script "{input_shell_script}" does not exist')
        sys.exit(1)
    
    # Use joblib.Parallel to run the commands in parallel
    results = Parallel(n_jobs=10)(delayed(run_command)(command) for command in commands)

    # Print the results
    for result in results:
        if result != None:
            print(result)

# Function to run a command and return the output and exit code
def run_command(command):
    print(command)
    # Use subprocess to run the command
    proc = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # Wait for the command to finish
    proc.wait()

    # Get the output and exit code of the command
    output = proc.stdout.read().decode()
    exit_code = proc

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_shell_script', required=True, help='Path to input shell script to run in parallel.')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    main(args)