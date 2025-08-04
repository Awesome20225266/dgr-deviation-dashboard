import os
import subprocess
import pandas as pd
import duckdb
from datetime import datetime
import time

MAPPING_FILE = "Mapping Sheet.xlsx"
DGR_FOLDER = "DGR_Backup"
DB_FILE = "dgr_data.duckdb"
TABLE_NAME = "dgr_data"
GIT_COMMIT_MSG = "Auto update DGR data and scripts (excluding DGR_Backup)"

def clean_value(v):
    if pd.isnull(v):
        return None
    vstr = str(v).replace("−", "-").replace("–", "-").strip()
    vstr = vstr.replace(",", "")
    try:
        if "%" in vstr:
            return float(vstr.replace("%", ""))
        val = float(vstr)
        if -1 < val < 1:
            return val * 100
        else:
            return val
    except:
        return None

def import_dgr_to_duckdb():
    if not os.path.exists(MAPPING_FILE):
        print(f"ERROR: Missing {MAPPING_FILE}")
        return
    if not os.path.exists(DGR_FOLDER):
        print(f"ERROR: Missing {DGR_FOLDER} folder")
        return

    # Create DB if not exists
    fresh_db = not os.path.exists(DB_FILE)
    con = duckdb.connect(DB_FILE)
    
    if fresh_db:
        con.execute(f"""
        CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
            plant VARCHAR,
            file_name VARCHAR,
            date DATE,
            input_name VARCHAR,
            value DOUBLE
        )
        """)
        print("Created fresh database and table.")
    else:
        print("Database exists. Will only add new rows.")

    mapping_df = pd.read_excel(MAPPING_FILE)
    print(f"Loaded mapping for {mapping_df.shape[0]} plants/rows.")

    total_rows = 0
    skipped_rows = 0
    for idx, row in mapping_df.iterrows():
        plant = str(row["Plant_Name"])
        fname = str(row["File_Name"])
        sheet = str(row["Sheet"])
        header_row = int(row["Header_Row"]) - 1
        date_col = str(row["Date_Col"]).strip()
        data_start_col = str(row["Data_Start_Col"]).strip()
        data_end_col = str(row["Data_End_Col"]).strip()

        found = False
        for ext in [".xlsx", ".xlsm"]:
            fpath = os.path.join(DGR_FOLDER, fname + ext)
            if os.path.exists(fpath):
                found = True
                break
        if not found:
            print(f"❌ File not found: {fname} (.xlsx/.xlsm)")
            continue

        try:
            df = pd.read_excel(fpath, sheet_name=sheet, header=header_row)
            df.columns = [str(c).strip() for c in df.columns]
        except Exception as e:
            print(f"❌ Error reading {fpath}: {e}")
            continue

        colnames = list(df.columns)
        if date_col not in colnames:
            print(f"❌ Date column '{date_col}' not found for plant {plant}. Columns: {colnames}")
            continue
        try:
            start_idx = colnames.index(data_start_col)
            end_idx = colnames.index(data_end_col)
        except Exception as e:
            print(f"❌ Data start/end column not found for plant {plant}: {e}")
            continue

        input_cols = colnames[start_idx:end_idx+1]
        count = 0
        for _, r in df.iterrows():
            thedate = pd.to_datetime(r[date_col], dayfirst=True, errors='coerce')
            if pd.isnull(thedate):
                continue
            for col in input_cols:
                v = clean_value(r[col])
                if v is None:
                    continue
                # Check if this (plant, file, date, input_name) is already in DB
                exists = con.execute(f"""
                    SELECT 1 FROM {TABLE_NAME}
                    WHERE plant = ? AND file_name = ? AND date = ? AND input_name = ?
                    LIMIT 1
                """, [plant, fname, thedate.date(), col]).fetchone()
                if exists:
                    skipped_rows += 1
                    continue
                con.execute(f"""
                    INSERT INTO {TABLE_NAME} (plant, file_name, date, input_name, value)
                    VALUES (?, ?, ?, ?, ?)
                """, [plant, fname, thedate.date(), col, v])
                count += 1
        total_rows += count
        print(f"✅ Imported {count} new rows for {plant} (skipped {skipped_rows} duplicates)")

    con.close()
    print(f"✅ All new DGR data imported into {DB_FILE} ({total_rows} new rows, {skipped_rows} skipped).")

def git_push():
    print("Running Git push script...")
    attempt = 1
    while True:
        try:
            # Stage all changes (including deletions, honoring .gitignore)
            subprocess.run(["git", "add", "-A"], check=True)

            # Check for local uncommitted changes
            diff_result = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True)
            has_local_changes = diff_result.stdout.strip() != ""

            # If local changes exist, stash them before pulling
            stashed = False
            if has_local_changes:
                print("⚠️ Local uncommitted changes detected. Stashing before pulling remote updates...")
                subprocess.run(["git", "stash"], check=True)
                stashed = True

            # Pull remote changes
            print(f"🔄 Attempt {attempt}: Pulling remote changes...")
            pull = subprocess.run(["git", "pull", "--no-edit", "origin", "main"])
            if pull.returncode != 0:
                print("❌ Git pull failed. If you see merge conflicts, resolve them and re-run the script.")
                return

            # If stashed, pop the stash, and re-add all changes
            if stashed:
                print("🔄 Applying stashed changes...")
                pop = subprocess.run(["git", "stash", "pop"])
                if pop.returncode != 0:
                    print("\n❗ Merge conflict occurred while applying your local changes after stash pop.")
                    print("👉 Please open the conflicted files, resolve manually, and then run:")
                    print("     git add <conflicted_file>")
                    print("     git commit -m 'Resolve merge conflict after stash pop'")
                    print("     git push origin main")
                    print("Aborting automated push for your safety.\n")
                    return
                # *** Key fix: stage all changes again after stash pop ***
                subprocess.run(["git", "add", "-A"], check=True)

            # Now commit if there are any changes after merging
            result = subprocess.run(["git", "diff", "--cached", "--quiet"])
            if result.returncode == 0:
                print("⚠️ No changes to commit.")
            else:
                subprocess.run(["git", "commit", "-m", GIT_COMMIT_MSG], check=True)

            print(f"🔼 Attempt {attempt}: Pushing to remote...")
            push = subprocess.run(["git", "push", "origin", "main"])
            if push.returncode == 0:
                print("✅ Git push completed successfully.")
                break
            else:
                print("⚠️ Push was not successful—repository may have changed on remote. Will pull again and retry...")
                attempt += 1
                time.sleep(2)  # Small delay to avoid hammering server

        except subprocess.CalledProcessError as e:
            print(f"❌ Git error: {e}")
            break

def main():
    print(f"=== DGR DB (incremental) + Git update started at {datetime.now()} ===")
    import_dgr_to_duckdb()
    git_push()
    print(f"=== Completed at {datetime.now()} ===")

if __name__ == "__main__":
    main()
