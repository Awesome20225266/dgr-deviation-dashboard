import os
import subprocess
import pandas as pd
import duckdb
from datetime import datetime

MAPPING_FILE = "Mapping Sheet.xlsx"
DGR_FOLDER = "DGR_Backup"
DB_FILE = "dgr_data.duckdb"
TABLE_NAME = "dgr_data"
GIT_COMMIT_MSG = "Auto update DGR data and scripts"

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
    try:
        subprocess.run(["git", "add", "."], check=True)
        subprocess.run(["git", "commit", "-m", GIT_COMMIT_MSG], check=True)
        subprocess.run(["git", "push", "origin", "main"], check=True)
        print("✅ Git push completed successfully.")
    except subprocess.CalledProcessError as e:
        error_msg = str(e)
        if "nothing to commit" in error_msg:
            print("⚠️ No changes to commit.")
            try:
                subprocess.run(["git", "push", "origin", "main"], check=True)
                print("✅ Git push completed successfully.")
            except Exception as push_e:
                print(f"❌ Git push failed: {push_e}")
        else:
            print(f"❌ Git error: {e}")

def main():
    print(f"=== DGR DB (incremental) + Git update started at {datetime.now()} ===")
    import_dgr_to_duckdb()
    git_push()
    print(f"=== Completed at {datetime.now()} ===")

if __name__ == "__main__":
    main()
