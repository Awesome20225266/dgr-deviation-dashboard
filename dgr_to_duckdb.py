import pandas as pd
import duckdb
import os

MAPPING_FILE = "Mapping Sheet.xlsx"
DGR_FOLDER = "DGR_Backup"
DB_FILE = "dgr_data.duckdb"
TABLE_NAME = "dgr_data"

# Remove the old database file (optional but recommended for a fresh start)
if os.path.exists(DB_FILE):
    os.remove(DB_FILE)

# 1. Connect and create table if needed
con = duckdb.connect(DB_FILE)
con.execute(f"""
CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
    plant VARCHAR,
    file_name VARCHAR,
    date DATE,
    input_name VARCHAR,
    value DOUBLE
)
""")

# 2. Load Mapping Sheet
mapping_df = pd.read_excel(MAPPING_FILE)
print(f"Loaded mapping sheet with {mapping_df.shape[0]} plants.")

# 3. Value cleaner - robust percent handling
def clean_value(v):
    if pd.isnull(v):
        return None
    vstr = str(v).replace("−", "-").replace("–", "-").strip()
    vstr = vstr.replace(",", "")
    try:
        # If original string contains "%", treat as percent
        if "%" in str(v):
            val = float(vstr.replace("%", ""))
            return val
        val = float(vstr)
        # If value between -1 and 1, treat as decimal (fraction of 1, multiply by 100)
        if -1 < val < 1:
            return val * 100
        else:
            return val
    except:
        return None

# 4. Import loop
for idx, row in mapping_df.iterrows():
    plant = str(row["Plant_Name"])
    fname = str(row["File_Name"])
    sheet = str(row["Sheet"])
    header_row = int(row["Header_Row"]) - 1  # 0-based for pandas
    date_col = str(row["Date_Col"]).strip()
    data_start_col = str(row["Data_Start_Col"]).strip()
    data_end_col = str(row["Data_End_Col"]).strip()

    # Try xlsx and xlsm
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
        df.columns = [str(c).strip() for c in df.columns]  # strip all col headers
    except Exception as e:
        print(f"❌ Error reading {fpath}: {e}")
        continue

    # Robust header check
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
    print(f"Plant: {plant} | Date: {date_col} | Inputs: {input_cols}")

    count = 0
    for _, r in df.iterrows():
        thedate = pd.to_datetime(r[date_col], dayfirst=True, errors='coerce')
        if pd.isnull(thedate):
            continue
        for col in input_cols:
            v = clean_value(r[col])
            if v is None:
                continue
            con.execute(f"""
                INSERT INTO {TABLE_NAME} (plant, file_name, date, input_name, value)
                VALUES (?, ?, ?, ?, ?)
            """, [plant, fname, thedate.date(), col, v])
            count += 1
    print(f"✅ Imported {count} rows for {plant}")

print("✅ All relevant DGR data imported into dgr_data.duckdb")
con.close()
