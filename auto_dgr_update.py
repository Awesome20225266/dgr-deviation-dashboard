#!/usr/bin/env python3
"""
V-Series Normalization Script (ultra-hardened, auto column-detect)

What this does:
- Adds & populates Tcell, Inorm, Vnorm, IVnorm in DuckDB/v_series.duckdb → v_series
- Pulls Alpha/Beta/Gamma by Module_Make from design.xlsx (module_design*) with robust header detection
- Weather from wms_series.duckdb → wms_series_with_clear (header detection; GPOA treated as GTI if needed)
- Exact time join: v_series.Date = wms.Date AND v_series.Time = wms.Time_Stamp
- Plant join via 3-char prefix after stripping trailing capacity (" 380MW" etc.)
- Set-based DuckDB SQL; progress bar in terminal; debug logs → v_series_normalization.log

Place in the same folder as:
  Dalgo-1/
    v_series_normalization.py  (this file)
    DuckDB/v_series.duckdb
    DuckDB/wms_series.duckdb
    array_configuration.xlsx
    design.xlsx
"""

from __future__ import annotations

import duckdb
import pandas as pd
from pathlib import Path
import logging
import sys
from typing import Optional, List, Dict
from tqdm import tqdm
import re

# -----------------------------------------------------------------------------
# Paths & constants
# -----------------------------------------------------------------------------
BASE_DIR = Path(__file__).parent
V_DUCK = BASE_DIR / "DuckDB" / "v_series.duckdb"
WMS_DUCK = BASE_DIR / "DuckDB" / "wms_series.duckdb"
ARRAY_CONFIG_FILE = BASE_DIR / "array_configuration.xlsx"
DESIGN_FILE = BASE_DIR / "design.xlsx"

# Defaults & physics
DEFAULT_KW = 0.03
DEFAULT_NMOT = 45.0
DEFAULT_ALPHA = 0.0004
DEFAULT_BETA  = -0.0025
DEFAULT_GAMMA = -0.0034
DELTA_T_HOURS = 1.0 / 60.0  # 1-minute cadence fallback

# -----------------------------------------------------------------------------
# Logging (file-only). On fatal, print one line to console.
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("v_series_normalization.log", encoding="utf-8")]
)
logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Utility helpers
# -----------------------------------------------------------------------------
def fuzzy_pick_sheet(xlsx_path: Path, desired_names: List[str]) -> str:
    xf = pd.ExcelFile(xlsx_path)
    sheets = xf.sheet_names
    s_lower = [s.lower().strip() for s in sheets]
    for want in desired_names:
        w = want.lower().strip()
        for i, s in enumerate(s_lower):
            if s == w:
                return sheets[i]
    for want in desired_names:
        w = want.lower().strip()
        for i, s in enumerate(s_lower):
            if w in s or s in w:
                return sheets[i]
    logger.warning(f"No close sheet match in {xlsx_path.name}; using first: {sheets[0]}")
    return sheets[0]


def strip_capacity_suffix(name: str) -> str:
    if not isinstance(name, str):
        return ""
    return re.sub(r"\s+\d+\s*MW\s*$", "", name.strip(), flags=re.IGNORECASE)


def normalize_header(h: str) -> str:
    """Lowercase alnum-only, to match variants like 'Module Make', 'Module_Make', etc."""
    return re.sub(r"[^a-z0-9]", "", h.lower()) if isinstance(h, str) else ""


def map_columns_generic(cols: List[str], wanted: Dict[str, List[str]]) -> Dict[str, Optional[str]]:
    """
    Map actual columns to a standard schema.
    cols: list of actual column names
    wanted: {'StdName': ['candidate1','candidate2',...normalized tokens]}
    Returns mapping: {'StdName': actual_col_or_None}
    """
    norm_map = {normalize_header(c): c for c in cols}
    result = {}
    for std, cands in wanted.items():
        found = None
        for cand in cands:
            if cand in norm_map:
                found = norm_map[cand]
                break
        result[std] = found
    return result


# -----------------------------------------------------------------------------
# Main normalizer
# -----------------------------------------------------------------------------
class VSeriesNormalizer:
    def __init__(self):
        self.array_config_df: Optional[pd.DataFrame] = None
        self.module_design_df: Optional[pd.DataFrame] = None
        self.con: Optional[duckdb.DuckDBPyConnection] = None

    # -------------------- IO --------------------
    def load_reference(self):
        # array_configuration
        ac_sheet = fuzzy_pick_sheet(ARRAY_CONFIG_FILE, ["Sheet1", "array", "array_configuration", "config"])
        self.array_config_df = pd.read_excel(ARRAY_CONFIG_FILE, sheet_name=ac_sheet)
        logger.info(f"array_configuration.xlsx → {ac_sheet} ({len(self.array_config_df):,} rows)")

        # design.xlsx
        design_sheet = fuzzy_pick_sheet(DESIGN_FILE, ["module_design", "module design", "module", "design"])
        self.module_design_df = pd.read_excel(DESIGN_FILE, sheet_name=design_sheet)
        logger.info(f"design.xlsx → {design_sheet} ({len(self.module_design_df):,} rows)")

        logger.debug(f"array_config columns: {list(self.array_config_df.columns)}")
        logger.debug(f"module_design columns: {list(self.module_design_df.columns)}")

    def connect_dbs(self):
        if not V_DUCK.exists() or not WMS_DUCK.exists():
            missing = [p for p in [V_DUCK, WMS_DUCK] if not p.exists()]
            for m in missing:
                logger.error(f"Missing DB: {m}")
            raise FileNotFoundError("Required DuckDB files missing.")
        self.con = duckdb.connect(str(V_DUCK))
        try:
            self.con.execute(f"ATTACH '{WMS_DUCK.as_posix()}' AS wmsdb")
        except Exception:
            pass
        # Verify base tables exist
        self.con.execute("SELECT 1 FROM v_series LIMIT 1")
        self.con.execute("SELECT 1 FROM wmsdb.wms_series_with_clear LIMIT 1")

    # -------------------- Standardize sources into temp views --------------------
    def build_standard_views(self):
        # 1) v_series → v_series_std
        v_cols = self.con.execute("PRAGMA table_info(v_series)").df()["name"].tolist()
        v_wanted = {
            "timestamp": ["timestamp", "tstamp", "ts"],
            "date":      ["date"],
            "time":      ["time", "timestamptime", "timestamppart"],
            "plant":     ["plant", "plantname", "site", "sitename"],
            "icr":       ["icr", "block", "blockno", "block_no"],
            "inv":       ["inv", "inverter", "inverterno", "inverter_no"],
            "unit":      ["unit", "unitno", "unit_no"],
            "input":     ["input", "scb", "scbno", "scb_no", "string", "stringno"],
            "vdc":       ["vdc", "v", "voltage", "voltdc"],
            "idc":       ["idc", "i", "current", "idcval"]
        }
        v_map = map_columns_generic(v_cols, v_wanted)
        missing_v = [k for k, v in v_map.items() if v is None and k in ("date","time","plant","icr","inv","unit","input","vdc","idc")]
        if missing_v:
            logger.warning(f"v_series missing expected columns (will be NULL in std view): {missing_v}")

        # build SELECT with safe aliases & types
        v_select = f"""
            SELECT
              {f"{v_map['timestamp']}" if v_map['timestamp'] else 'NULL'} AS timestamp,
              TRIM(CAST({f"{v_map['date']}" if v_map['date'] else "''"} AS VARCHAR)) AS Date,
              TRIM(CAST({f"{v_map['time']}" if v_map['time'] else "''"} AS VARCHAR)) AS Time,
              TRIM(CAST({f"{v_map['plant']}" if v_map['plant'] else "''"} AS VARCHAR)) AS Plant,
              CAST({f"{v_map['icr']}" if v_map['icr'] else 'NULL'} AS INT)  AS ICR,
              CAST({f"{v_map['inv']}" if v_map['inv'] else 'NULL'} AS INT)  AS Inv,
              CAST({f"{v_map['unit']}" if v_map['unit'] else 'NULL'} AS INT) AS Unit,
              CAST({f"{v_map['input']}" if v_map['input'] else 'NULL'} AS INT) AS Input,
              CAST({f"{v_map['vdc']}" if v_map['vdc'] else 'NULL'} AS DOUBLE) AS Vdc,
              CAST({f"{v_map['idc']}" if v_map['idc'] else 'NULL'} AS DOUBLE) AS Idc,
              ROWID AS rid
            FROM v_series
        """
        self.con.execute("DROP VIEW IF EXISTS v_series_std")
        self.con.execute(f"CREATE TEMP VIEW v_series_std AS {v_select}")

        # 2) WMS → wms_std
        w_cols = self.con.execute("PRAGMA table_info(wmsdb.wms_series_with_clear)").df()["name"].tolist()
        w_wanted = {
            "global_plant_id": ["globalplantid", "global_plant_id", "plantid", "plant_id"],
            "date":            ["date"],
            "time_stamp":      ["timestamp", "time_stamp", "time", "timestamppart"],
            "gpoa":            ["gpoa", "gti", "gpoawperm2", "gtiwperm2"],
            "at":              ["at", "ambient", "ambienttemp", "ambienttemperature"],
            "mt":              ["mt", "moduletemp", "moduletemperature", "tcell", "module_temp"],
            "ws":              ["ws", "windspeed", "wind_speed"]
        }
        # Normalize available
        w_norm = {normalize_header(c): c for c in w_cols}

        # Helper to pick best
        def pick(*opts):
            for o in opts:
                if o in w_norm:
                    return w_norm[o]
            return None

        w_gid = pick(*w_wanted["global_plant_id"])
        w_date = pick(*w_wanted["date"])
        w_time = pick(*w_wanted["time_stamp"])
        w_gpoa = pick(*w_wanted["gpoa"])
        w_at = pick(*w_wanted["at"])
        w_mt = pick(*w_wanted["mt"])
        w_ws = pick(*w_wanted["ws"])

        if not (w_gid and w_date and w_time):
            raise RuntimeError("WMS table lacks essential columns for join (global_plant_id/date/time).")

        # Build COALESCE(GPOA, GTI) if both exist; if only GTI exists use it; else only GPOA.
        gpoa_expr = "CAST(COALESCE({gpoa},{gti}) AS DOUBLE)"
        if w_gpoa and normalize_header(w_gpoa) == "gpoa":
            # see if GTI also exists
            gti_col = w_norm.get("gti", None)
            if gti_col:
                gpoa_sql = gpoa_expr.format(gpoa=w_gpoa, gti=gti_col)
            else:
                gpoa_sql = f"CAST({w_gpoa} AS DOUBLE)"
        else:
            # try GTI
            gti_col = w_norm.get("gti", None)
            if gti_col:
                gpoa_sql = f"CAST({gti_col} AS DOUBLE)"
            else:
                # last resort: if some irradiance exists with another token, leave NULL
                gpoa_sql = "CAST(NULL AS DOUBLE)"
                logger.warning("Neither GPOA nor GTI found in WMS; GPOA will be NULL.")

        w_select = f"""
            SELECT DISTINCT
              CAST({w_gid} AS VARCHAR) AS global_plant_id,
              TRIM(CAST({w_date} AS VARCHAR)) AS Date,
              TRIM(CAST({w_time} AS VARCHAR)) AS Time,
              {gpoa_sql} AS GPOA,
              CAST({w_at if w_at else 'NULL'} AS DOUBLE) AS AT,
              CAST({w_mt if w_mt else 'NULL'} AS DOUBLE) AS MT,
              CAST({w_ws if w_ws else 'NULL'} AS DOUBLE) AS WS
            FROM wmsdb.wms_series_with_clear
        """
        self.con.execute("DROP VIEW IF EXISTS wms_std")
        self.con.execute(f"CREATE TEMP VIEW wms_std AS {w_select}")

        # 3) array_configuration.xlsx → array_df_std
        ac_cols = list(self.array_config_df.columns)
        ac_wanted = {
            "plant_name": ["plantname", "plant", "site", "sitename"],
            "block_no":   ["block_no", "blockno", "icr", "block"],
            "inverter_no":["inverter_no", "inverterno", "inv", "inverter"],
            "unit_no":    ["unit_no", "unitno", "unit"],
            "scb_no":     ["scb_no", "scbno", "input", "string", "stringno"],
            "capacity":   ["capacity", "pdc", "pdc_kwp", "kwp", "dc_capacity"],
            "module_make":["module_make", "modulemake", "module", "modulemodel", "make"],
            "global_plant_id": ["global_plant_id", "plantid", "plant_id"]
        }
        ac_map = map_columns_generic(ac_cols, ac_wanted)
        # build DataFrame standardized
        df = self.array_config_df.copy()
        def col(name): return ac_map[name] if ac_map[name] in df.columns else None
        array_std = pd.DataFrame({
            "plant_name": df[col("plant_name")] if col("plant_name") else "",
            "block_no":   df[col("block_no")] if col("block_no") else None,
            "inverter_no":df[col("inverter_no")] if col("inverter_no") else None,
            "unit_no":    df[col("unit_no")] if col("unit_no") else None,
            "scb_no":     df[col("scb_no")] if col("scb_no") else None,
            "Capacity":   df[col("capacity")] if col("capacity") else None,
            "Module_Make":df[col("module_make")] if col("module_make") else "",
            "global_plant_id": df[col("global_plant_id")] if col("global_plant_id") else None
        })
        self.con.register("array_df_raw", array_std)
        self.con.execute("DROP VIEW IF EXISTS array_df_std")
        self.con.execute(
            """
            CREATE TEMP VIEW array_df_std AS
            SELECT
              UPPER(SUBSTR(REGEXP_REPLACE(COALESCE(plant_name,''), '\\s+\\d+\\s*MW$', ''), 1, 3)) AS plant_prefix,
              CAST(block_no AS INT)    AS ICR,
              CAST(inverter_no AS INT) AS Inv,
              CAST(unit_no AS INT)     AS Unit,
              CAST(scb_no AS INT)      AS Input,
              CAST(Capacity AS DOUBLE) AS pdc_kwp,
              TRIM(COALESCE(Module_Make,'')) AS Module_Make_raw,
              LOWER(REGEXP_REPLACE(TRIM(COALESCE(Module_Make,'')), '[^a-zA-Z0-9]', '')) AS Module_Make_norm,
              CAST(global_plant_id AS VARCHAR) AS global_plant_id
            FROM array_df_raw
            """
        )

        # 4) design.xlsx → design_df_std
        md_cols = list(self.module_design_df.columns)
        md_wanted = {
            "module_make": ["module_make","modulemake","module","model","make"],
            "alpha":       ["alpha","tempcoeffi","tempcoeffi_i","alphai","coeffalpha"],
            "beta":        ["beta","tempcoeffv","betav","coeffbeta"],
            "gamma":       ["gamma","tempcoeffp","gammapp","coeffgamma"]
        }
        md_map = map_columns_generic(md_cols, md_wanted)
        ddf = self.module_design_df.copy()
        def mcol(name): return md_map[name] if md_map[name] in ddf.columns else None
        design_std = pd.DataFrame({
            "Module_Make": ddf[mcol("module_make")] if mcol("module_make") else "",
            "Alpha": ddf[mcol("alpha")] if mcol("alpha") else None,
            "Beta":  ddf[mcol("beta")]  if mcol("beta")  else None,
            "Gamma": ddf[mcol("gamma")] if mcol("gamma") else None
        })
        self.con.register("design_df_raw", design_std)
        self.con.execute("DROP VIEW IF EXISTS design_df_std")
        self.con.execute(
            f"""
            CREATE TEMP VIEW design_df_std AS
            SELECT
              TRIM(COALESCE(Module_Make,'')) AS Module_Make_raw,
              LOWER(REGEXP_REPLACE(TRIM(COALESCE(Module_Make,'')), '[^a-zA-Z0-9]', '')) AS Module_Make_norm,
              COALESCE(CAST(Alpha AS DOUBLE), {DEFAULT_ALPHA}) AS Alpha,
              COALESCE(CAST(Beta  AS DOUBLE), {DEFAULT_BETA})  AS Beta,
              COALESCE(CAST(Gamma AS DOUBLE), {DEFAULT_GAMMA}) AS Gamma
            FROM design_df_raw
            """
        )

    # -------------------- Core pipeline --------------------
    def ensure_target_columns(self):
        info = self.con.execute("PRAGMA table_info(v_series)").df()
        have = set(info["name"].tolist())
        for col, dtype in [("Tcell","DOUBLE"),("Inorm","DOUBLE"),("Vnorm","DOUBLE"),("IVnorm","DOUBLE")]:
            if col not in have:
                self.con.execute(f"ALTER TABLE v_series ADD COLUMN {col} {dtype}")
                logger.info(f"Added column {col} {dtype}")

    def process(self) -> bool:
        try:
            self.ensure_target_columns()

            pbar = tqdm(
                total=7,
                desc="Initializing…",
                unit="step",
                dynamic_ncols=True,
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {desc}"
            )

            # 1) Build standardized views
            pbar.set_description("Standardizing sources…")
            self.build_standard_views()
            pbar.update(1)

            # 2) Join map
            pbar.set_description("Building join map…")
            self.con.execute("DROP TABLE IF EXISTS tmp_map")
            self.con.execute(
                """
                CREATE TEMP TABLE tmp_map AS
                SELECT 
                  v.rid,
                  v.Date AS v_Date,
                  v.Time AS v_Time,
                  v.Idc, v.Vdc,
                  a.pdc_kwp,
                  a.Module_Make_raw,
                  a.Module_Make_norm,
                  d.Alpha, d.Beta, d.Gamma,
                  w.GPOA, w.AT, w.MT, w.WS
                FROM v_series_std v
                LEFT JOIN array_df_std a
                  ON UPPER(SUBSTR(REGEXP_REPLACE(COALESCE(v.Plant,''), '\\s+\\d+\\s*MW$', ''), 1, 3)) = a.plant_prefix
                 AND v.ICR  = a.ICR
                 AND v.Inv  = a.Inv
                 AND v.Unit = a.Unit
                 AND v.Input= a.Input
                LEFT JOIN design_df_std d
                  ON (
                       a.Module_Make_norm = d.Module_Make_norm
                    OR LOWER(a.Module_Make_raw) LIKE LOWER(d.Module_Make_raw || '%')
                    OR LOWER(d.Module_Make_raw) LIKE LOWER(a.Module_Make_raw || '%')
                  )
                LEFT JOIN wms_std w
                  ON a.global_plant_id = w.global_plant_id
                 AND v.Date = w.Date
                 AND v.Time = w.Time
                """
            )
            pbar.update(1)

            # 3) Compute Tcell
            pbar.set_description("Computing Tcell…")
            self.con.execute("DROP TABLE IF EXISTS tmp_tcell")
            self.con.execute(
                f"""
                CREATE TEMP TABLE tmp_tcell AS
                SELECT
                  rid,
                  CASE 
                    WHEN MT IS NOT NULL AND MT BETWEEN 0 AND 100 THEN CAST(MT AS DOUBLE)
                    ELSE CAST(AT + (({DEFAULT_NMOT} - 20.0)/800.0) * COALESCE(GPOA,0)
                              - {DEFAULT_KW} * GREATEST(0.0, COALESCE(WS,0) - 1.0) AS DOUBLE)
                  END AS tcell_raw,
                  CAST(AT - 5.0  AS DOUBLE) AS at_min,
                  CAST(AT + 60.0 AS DOUBLE) AS at_max,
                  CAST(AT + 80.0 AS DOUBLE) AS at_max_high
                FROM tmp_map
                """
            )
            self.con.execute("DROP TABLE IF EXISTS tmp_tcell_final")
            self.con.execute(
                """
                CREATE TEMP TABLE tmp_tcell_final AS
                SELECT
                  rid,
                  CASE 
                    WHEN tcell_raw > 100.0 THEN GREATEST(at_min, LEAST(at_max_high, tcell_raw))
                    ELSE GREATEST(at_min, LEAST(at_max, tcell_raw))
                  END AS Tcell
                FROM tmp_tcell
                """
            )
            pbar.update(1)

            # 4) Update Tcell
            pbar.set_description("Updating Tcell…")
            self.con.execute(
                """
                UPDATE v_series AS v
                SET Tcell = t.Tcell
                FROM tmp_tcell_final t
                WHERE v.ROWID = t.rid
                """
            )
            pbar.update(1)

            # 5) Norms
            pbar.set_description("Updating Inorm/Vnorm/IVnorm…")
            self.con.execute("DROP TABLE IF EXISTS tmp_norms")
            self.con.execute(
                f"""
                CREATE TEMP TABLE tmp_norms AS
                SELECT
                  m.rid,
                  CASE 
                    WHEN m.pdc_kwp > 0 AND COALESCE(m.GPOA,0) > 0 AND (1.0 + m.Alpha * (v.Tcell - 25.0)) > 0
                    THEN (m.Idc / m.pdc_kwp) * (1000.0 / m.GPOA) * (1.0 / (1.0 + m.Alpha * (v.Tcell - 25.0)))
                    ELSE NULL
                  END AS Inorm,
                  CASE 
                    WHEN (1.0 + m.Beta * (v.Tcell - 25.0)) != 0
                    THEN v.Vdc / (1.0 + m.Beta * (v.Tcell - 25.0))
                    ELSE NULL
                  END AS Vnorm,
                  CASE 
                    WHEN m.pdc_kwp > 0 AND COALESCE(m.GPOA,0) > 0 AND (1.0 + m.Gamma * (v.Tcell - 25.0)) > 0
                    THEN (v.Vdc * m.Idc * {DELTA_T_HOURS}) / (m.pdc_kwp * 1000.0) * (1000.0 / m.GPOA) * (1.0 / (1.0 + m.Gamma * (v.Tcell - 25.0)))
                    ELSE NULL
                  END AS IVnorm
                FROM tmp_map m
                JOIN v_series_std v ON v.rid = m.rid
                """
            )
            self.con.execute(
                """
                UPDATE v_series AS v
                SET Inorm = n.Inorm,
                    Vnorm = n.Vnorm,
                    IVnorm = n.IVnorm
                FROM tmp_norms n
                WHERE v.ROWID = n.rid
                """
            )
            pbar.update(1)

            # 6) Commit & cleanup
            pbar.set_description("Committing & cleaning…")
            self.con.commit()
            for t in ["tmp_map","tmp_tcell","tmp_tcell_final","tmp_norms"]:
                try:
                    self.con.execute(f"DROP TABLE IF EXISTS {t}")
                except Exception:
                    pass
            pbar.update(1)
            pbar.close()

            logger.info("Normalization pipeline completed successfully.")
            return True

        except Exception as e:
            logger.exception(f"Normalization failed: {e}")
            return False

    # -------------------- Validation --------------------
    def validate(self):
        try:
            stats = self.con.execute(
                """
                SELECT 
                  COUNT(*) AS total_rows,
                  COUNT(Tcell) AS tcell_count,
                  COUNT(Inorm) AS inorm_count,
                  COUNT(Vnorm) AS vnorm_count,
                  COUNT(IVnorm) AS ivnorm_count,
                  AVG(Tcell) AS avg_tcell,
                  MIN(Tcell) AS min_tcell,
                  MAX(Tcell) AS max_tcell
                FROM v_series
                """
            ).df().iloc[0].to_dict()
            for k, v in stats.items():
                logger.info(f"{k}: {v}")

            sample = self.con.execute(
                """
                SELECT timestamp, Plant, ICR, Inv, Unit, Input,
                       Idc, Vdc, Tcell, Inorm, Vnorm, IVnorm
                FROM v_series
                WHERE Inorm IS NOT NULL
                ORDER BY timestamp
                LIMIT 10
                """
            ).df()
            logger.info("Sample populated rows:\n" + sample.to_string(index=False))
        except Exception as e:
            logger.exception(f"Validation error: {e}")

    def close(self):
        try:
            if self.con:
                self.con.close()
        except Exception:
            pass


def main() -> bool:
    # Pre-flight checks
    missing = [p for p in [V_DUCK, WMS_DUCK, ARRAY_CONFIG_FILE, DESIGN_FILE] if not p.exists()]
    if missing:
        for m in missing:
            logger.error(f"Missing required input: {m}")
        print("Normalization failed. See v_series_normalization.log for details.", file=sys.stderr)
        return False

    n = VSeriesNormalizer()
    try:
        n.load_reference()
        n.connect_dbs()
        ok = n.process()
        if ok:
            n.validate()
        else:
            print("Normalization failed. See v_series_normalization.log for details.", file=sys.stderr)
        return ok
    except Exception as e:
        logger.exception(f"Fatal error: {e}")
        print("Normalization failed. See v_series_normalization.log for details.", file=sys.stderr)
        return False
    finally:
        n.close()


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
