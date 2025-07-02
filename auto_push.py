import subprocess

def run_command(cmd):
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error running {cmd}:\n{result.stderr}")
    else:
        print(f"Success: {cmd}")

def main():
    cmds = [
        "git add .",
        "git commit -m 'Auto update DGR data and scripts'",
        "git push origin main"
    ]
    for cmd in cmds:
        run_command(cmd)

if __name__ == "__main__":
    main()
