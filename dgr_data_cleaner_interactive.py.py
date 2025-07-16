import duckdb

DB_PATH = "dgr_data.duckdb"
TABLE_NAME = "dgr_data"

def get_plants_and_dates():
    with duckdb.connect(DB_PATH) as con:
        plants = [row[0] for row in con.execute(f"SELECT DISTINCT plant FROM {TABLE_NAME} ORDER BY plant").fetchall()]
        dates = [str(row[0]) for row in con.execute(f"SELECT DISTINCT date FROM {TABLE_NAME} ORDER BY date").fetchall()]
    return plants, dates

def delete_rows(plant, date_start, date_end):
    with duckdb.connect(DB_PATH) as con:
        if plant.lower() == "all":
            sql = f"DELETE FROM {TABLE_NAME} WHERE date BETWEEN ? AND ?"
            params = [date_start, date_end]
        else:
            sql = f"DELETE FROM {TABLE_NAME} WHERE plant = ? AND date BETWEEN ? AND ?"
            params = [plant, date_start, date_end]
        count_before = con.execute(f"SELECT COUNT(*) FROM {TABLE_NAME}").fetchone()[0]
        con.execute(sql, params)
        count_after = con.execute(f"SELECT COUNT(*) FROM {TABLE_NAME}").fetchone()[0]
        print(f"\n✅ Deleted {count_before - count_after} row(s) from {TABLE_NAME}.")

def main():
    print("\n==== DGR Data Cleaner (Interactive) ====")
    plants, dates = get_plants_and_dates()
    print("\nAvailable Plants:")
    for idx, p in enumerate(plants, 1):
        print(f"{idx}. {p}")
    print(f"{len(plants)+1}. ALL PLANTS")

    # Select plant
    while True:
        plant_input = input("\nEnter plant name (or type 'ALL' for all plants): ").strip()
        if plant_input.lower() == "all" or plant_input in plants:
            break
        else:
            print("Invalid plant name. Please enter from the list.")

    # Show available dates
    print("\nAvailable Dates:")
    for i, d in enumerate(dates, 1):
        print(f"{i}. {d}")

    # Select date start
    while True:
        date_start = input("\nEnter start date (YYYY-MM-DD) from above: ").strip()
        if date_start in dates:
            break
        else:
            print("Invalid start date. Please enter from the list.")

    # Select date end
    while True:
        date_end = input("Enter end date (YYYY-MM-DD) from above: ").strip()
        if date_end in dates and date_end >= date_start:
            break
        else:
            print("Invalid end date. Must be >= start date and in the list.")

    print(f"\nYou are about to delete rows for Plant: {plant_input} | Dates: {date_start} to {date_end}")
    confirm = input("Type 'YES' to confirm deletion: ").strip()
    if confirm == "YES":
        delete_rows(plant_input, date_start, date_end)
    else:
        print("❌ Deletion cancelled.")

if __name__ == "__main__":
    main()
