import duckdb

DB_PATH = "dgr_data.duckdb"
REASON_TABLE = "deviation_reasons"

with duckdb.connect(DB_PATH) as con:
    con.execute(f"""
        CREATE TABLE IF NOT EXISTS {REASON_TABLE} (
            plant TEXT,
            date TEXT,
            input_name TEXT,
            deviation DOUBLE,
            reason TEXT,
            comment TEXT,
            timestamp TEXT
        );
    """)
print("Deviation Reasons table is ready.")
