import duckdb
import pandas as pd

con = duckdb.connect("dgr_data.duckdb")

query = """
SELECT date, input_name, value
FROM dgr_data
WHERE plant = 'AXPPL'
  AND value <= -99
  AND date BETWEEN '2025-07-08' AND '2025-07-14'
ORDER BY date, input_name;
"""

df = con.execute(query).fetchdf()
print(df)
con.close()
