import os
import subprocess
import pandas as pd
import duckdb
from datetime import datetime
import time

MAPPING_FILE    = "Mapping Sheet.xlsx"
DGR_FOLDER      = "DGR_Backup"
DB_FILE         = "dgr_data.duckdb"
TABLE_NAME      = "dgr_data"
GIT_COMMIT_MSG  = "Auto update DGR data and scripts (excluding DGR_Backup)"

def clean_value(v):
    if pd.isnull(v):
        return None
    vstr = str(v).replace("−", "-").replace("–", "-").strip().replace(",", "")
    try:
        if "%" in vstr:
            return float(vstr.replace("%", ""))
        val = float(vstr)
        # convert small decimals into percentages
        if -1 < val < 1:
            return val * 100
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

    fresh_db = not os.path.exists(DB_FILE)
    con = duckdb.connect(DB_FILE)
    if fresh_db:
        con.execute(f"""
            CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
                plant       VARCHAR,
                file_name   VARCHAR,
                date        DATE,
                input_name  VARCHAR,
                value       DOUBLE
            )
        """)
        print("Created fresh database and table.")
    else:
        print("Database exists. Will only add new rows.")

    mapping_df = pd.read_excel(MAPPING_FILE)
    print(f"Loaded mapping for {mapping_df.shape[0]} plants/rows.")

    total_new = 0
    total_skipped = 0

    for _, row in mapping_df.iterrows():
        plant      = str(row["Plant_Name"])
        fname      = str(row["File_Name"])
        sheet      = str(row["Sheet"])
        header_row = int(row["Header_Row"]) - 1
        date_col   = str(row["Date_Col"]).strip()
        start_col  = str(row["Data_Start_Col"]).strip()
        end_col    = str(row["Data_End_Col"]).strip()

        # find file path
        for ext in (".xlsx", ".xlsm"):
            fpath = os.path.join(DGR_FOLDER, fname + ext)
            if os.path.exists(fpath):
                break
        else:
            print(f"❌ File not found: {fname} (.xlsx/.xlsm)")
            continue

        try:
            df = pd.read_excel(fpath, sheet_name=sheet, header=header_row)
            df.columns = [str(c).strip() for c in df.columns]
        except Exception as e:
            print(f"❌ Error reading {fpath}: {e}")
            continue

        cols = list(df.columns)
        if date_col not in cols:
            print(f"❌ Date column '{date_col}' missing for {plant}.")
            continue
        try:
            i0 = cols.index(start_col)
            i1 = cols.index(end_col)
        except ValueError as e:
            print(f"❌ Data columns not found for {plant}: {e}")
            continue

        new_count = 0
        skip_count = 0
        for _, r in df.iterrows():
            thedate = pd.to_datetime(r[date_col], dayfirst=True, errors='coerce')
            if pd.isnull(thedate):
                continue
            for col in cols[i0 : i1 + 1]:
                v = clean_value(r[col])
                if v is None:
                    continue
                # dedupe
                exists = con.execute(f"""
                    SELECT 1 FROM {TABLE_NAME}
                    WHERE plant = ? AND file_name = ? AND date = ? AND input_name = ?
                    LIMIT 1
                """, [plant, fname, thedate.date(), col]).fetchone()
                if exists:
                    skip_count += 1
                    continue
                con.execute(f"""
                    INSERT INTO {TABLE_NAME}
                    (plant, file_name, date, input_name, value)
                    VALUES (?, ?, ?, ?, ?)
                """, [plant, fname, thedate.date(), col, v])
                new_count += 1

        total_new += new_count
        total_skipped += skip_count
        print(f"✅ {plant}: imported {new_count} new rows (skipped {skip_count} duplicates)")

    con.close()
    print(f"✅ All done: {total_new} new rows, {total_skipped} skipped.")

def git_push():
    print("Running Git push script...")
    attempt = 1
    while True:
        try:
            # 1) Stage everything, then unstage DGR_Backup
            subprocess.run(["git", "add", "-A"], check=True)
            subprocess.run(["git", "reset", "--", DGR_FOLDER], check=True)

            # 2) Check for local changes to stash
            status = subprocess.run(
                ["git", "status", "--porcelain"],
                capture_output=True, text=True
            ).stdout.strip()
            if status:
                print("⚠️ Local uncommitted changes detected. Stashing...")
                subprocess.run(["git", "stash"], check=True)
                stashed = True
            else:
                stashed = False

            # 3) Pull updates
            print(f"🔄 Attempt {attempt}: Pulling remote changes...")
            pull = subprocess.run(["git", "pull", "--no-edit", "origin", "main"])
            if pull.returncode != 0:
                print("❌ Git pull failed. Resolve conflicts and re-run the script.")
                return

            # 4) Re-apply your stash if needed
            if stashed:
                print("🔄 Applying stashed changes...")
                pop = subprocess.run(["git", "stash", "pop"])
                if pop.returncode != 0:
                    print("\n❗ Merge conflict occurred after stash pop.")
                    print("👉 Please resolve conflicts, then run:")
                    print("     git add <file>")
                    print("     git commit -m 'Resolve merge conflict'")
                    print("     git push origin main\n")
                    return
                # re-stage and unstage DGR_Backup again
                subprocess.run(["git", "add", "-A"], check=True)
                subprocess.run(["git", "reset", "--", DGR_FOLDER], check=True)

            # 5) Commit if there’s anything new
            diff = subprocess.run(["git", "diff", "--cached", "--quiet"])
            if diff.returncode == 0:
                print("⚠️ No changes to commit.")
            else:
                subprocess.run(
                    ["git", "commit", "-m", GIT_COMMIT_MSG],
                    check=True
                )

            # 6) Push
            print(f"🔼 Attempt {attempt}: Pushing to remote...")
            push = subprocess.run(["git", "push", "origin", "main"])
            if push.returncode == 0:
                print("✅ Git push completed successfully.")
                break
            else:
                print("⚠️ Push failed—retrying pull & push...")
                attempt += 1
                time.sleep(2)

        except subprocess.CalledProcessError as e:
            print(f"❌ Git error: {e}")
            break

def main():
    print(f"=== DGR + Git update started at {datetime.now()} ===")
    import_dgr_to_duckdb()
    git_push()
    print(f"=== Completed at {datetime.now()} ===")

if __name__ == "__main__":
    main()
