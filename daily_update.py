import subprocess
import sys

def run_command(command, description):
    print(f"Starting: {description} ...")
    try:
        subprocess.run(command, check=True)
        print(f"Success: {description} completed.")
    except subprocess.CalledProcessError as e:
        print(f"ERROR: {description} failed!")
        print(f"Command: {' '.join(command)}")
        print(f"Return code: {e.returncode}")
        print(f"Output/Error: {e}")
        sys.exit(1)  # Stop execution on error

def main():
    print("==== Git Auto-Push Script Started ====")

    # Step 1: git add .
    run_command(['git', 'add', '.'], "Staging changes (git add)")

    # Step 2: git commit -m "message"
    run_command(['git', 'commit', '-m', "Auto update DGR data and scripts"], "Committing changes (git commit)")

    # Step 3: git push origin main
    run_command(['git', 'push', 'origin', 'main'], "Pushing changes to GitHub (git push)")

    print("==== All Git operations completed successfully! ====")

if __name__ == "__main__":
    main()
