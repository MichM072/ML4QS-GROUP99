import subprocess
import os
import sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--ssh", help="Only set up SSH connection", action="store_true")
parser.add_argument("-c", "--conda", help="Only set up conda environment", action="store_true")
parser.add_argument("-e", "--easyconnect", help="Only set up easy connect", action="store_true")
args = parser.parse_args()

vunetid = None
cluster = None
location = None
hostname = None
proxy_jump = None
system_ssh = None

def locate_ssh_dir():
    if sys.platform.startswith('win'):
        system_ssh = f'C:\\Users\\{os.getlogin()}\\.ssh'
        return system_ssh
    else:
        system_ssh = f'/home/{os.getlogin()}/.ssh'
        return system_ssh

def ssh_connect():
    global vunetid, cluster, location, hostname, proxy_jump, system_ssh
    print("Setting up SSH connection to remote server...")

    print("Enter your VUnetID")
    vunetid = input()

    print("Select what cluster you want to connect to: ")

    while True:
        cluster = input()
        if cluster in ['1', '2', '3']:
            break
        else:
            print("Please respond with '1', '2' or '3'")

    print("Are you at home (h) or at uni? (u)")

    while True:
        location = input().lower()
        if location in ['h', 'u']:
            break
        else:
            print("Please respond with 'h' or 'u'")

    hostname = f'{vunetid}@{cluster}.compute.vu.nl'
    proxy_jump = f'{vunetid}@ssh.data.vu.nl'

    print("Starting connection...")
    print("Because the VU does not allow automatic command line interactions you must now open a new terminal window and enter the following command:")

    if location == 'u':
        print(f"ssh {hostname}")
    else:
        print(f"ssh -J {proxy_jump} {hostname}")

    print("You will be prompted to enter your password, this is the same password as your canvas/VUnet login.")
    print("You should be greeted by a terminal with a big VU logo.")

    while True:
        wait = input("Press enter to continue after ssh login...")
        if wait == "":
            break

    print("Connection successful!")
    print(f"You can now connect to the cluster using")
    if location == 'u':
        print(f"ssh {hostname}")
    else:
        print(f"ssh -J {proxy_jump} {hostname}")


def setup_conda():
    print("Do you want to setup a conda environment? (y/n)")
    while True:
        answer = input().lower()
        if answer == "y":
            break
        elif answer == "n":
            return
        else:
            print("Please respond with 'y' or 'n'")

    print("Setting up conda environment...")
    print("Because the VU does not allow automatic command line interactions you must now open a new terminal window and enter the following command:")
    print("conda create --name ML4QS python==3.8.8")

    while True:
        wait = input("Press enter to continue after conda setup...")
        if wait == "":
            break

    print("Conda environment setup successful!")
    print("You can now activate the environment using")
    print("conda activate ML4QS")


def setup_easy_connect():
    print("Do you want to setup easy connect? (y/n)")
    while True:
        answer = input().lower()
        if answer == "y":
            break
        elif answer == "n":
            return
        else:
            print("Please respond with 'y' or 'n'")

    print("Setting up easy connect...")
    system_ssh = locate_ssh_dir()
    path = f'{system_ssh}/config'
    print("Checking if config file already exists...")
    if os.path.exists(path):
        print("Config file already exists, appending to it...")
        write_config(path, 'a')
    else:
        print("Config file does not exist, creating new file...")
        os.makedirs(system_ssh, exist_ok=False)
        write_config(path, 'w')

    print("Config file setup successful!")
    print("You can now connect to the cluster using")
    print(f"{cluster}.compute.vu.nl")
            
def write_config(path, mode):
    if location == 'u':
        with open(path, mode) as f:
            f.write(f'# Target host\n')
            f.write(f'Host {hostname}\n')
            f.write(f'    User {vunetid}\n')
        return

    with open(path, mode) as f:
        f.write(f'# Jump host configuration for ssh.data.vu.nl\n')
        f.write(f'Host ssh.data.vu.nl\n')
        f.write(f'    User {vunetid}\n')
        # f.write(f'    IdentityFile {system_ssh}/id_rsa\n')
        f.write(f'\n')
        f.write(f'# Target host (via jump host)\n')
        f.write(f'Host {hostname}\n')
        f.write(f'    User {vunetid}\n')
        # f.write(f'    IdentityFile {system_ssh}/id_rsa\n')
        f.write(f'    ProxyJump ssh.data.vu.nl\n')

def main():
    if args.ssh:
        ssh_connect()
        return
    elif args.conda:
        setup_conda()
        return
    elif args.easyconnect:
        setup_easy_connect()
        return
    else:
        ssh_connect()
        setup_conda()
        setup_easy_connect()


if __name__ == '__main__':
    main()