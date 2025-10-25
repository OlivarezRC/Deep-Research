
from __future__ import annotations
import re
import json
import duckdb
import pandas as pd
from typing import Dict, Any, Optional, Tuple, List
import chainlit as cl

_SIMPLE_ID = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

def _qid(name: str) -> str:
    """Quote identifier if needed for DuckDB."""
    s = str(name)
    return s if _SIMPLE_ID.match(s) else f"\"{s}\""

class SQLAgent:
    """
    Lightweight SQL/NL SQL agent on top of DuckDB.
    - Registers arbitrary DataFrames as a DuckDB view
    - Accepts direct SQL (SELECT/CTE) or asks LLM to produce SQL from natural language
    - Provides a heuristic NL→SQL fallback when LLM isn't available
    - Ensures only read-only SELECT queries are executed
    - Exposes a 'date' alias column when a datetime-like column is detected
    - If exactly one numeric column exists, also exposes it as 'value' for convenience
    """

    _BLOCKED_TOKENS = {
        "insert","update","delete","drop","alter","truncate","create","attach",
        "export","load","copy","pragma","call","transaction","commit","rollback","grant","revoke"
    }

    CADENCE_KEYWORDS = {
        "daily": "day", "day": "day",
        "weekly": "week", "week": "week",
        "monthly": "month", "month": "month",
        "quarterly": "quarter", "quarter": "quarter",
        "annual": "year", "yearly": "year", "year": "year",
    }

    AGG_KEYWORDS = {
        "avg": "avg", "average": "avg", "mean": "avg",
        "sum": "sum",
        "median": "median",
        "min": "min", "lowest": "min",
        "max": "max", "highest": "max",
        "std": "stddev", "stdev": "stddev", "stddev": "stddev",
    }

    MONTHS = {
        "jan":1,"january":1,"feb":2,"february":2,"mar":3,"march":3,"apr":4,"april":4,"may":5,
        "jun":6,"june":6,"jul":7,"july":7,"aug":8,"august":8,"sep":9,"sept":9,"september":9,
        "oct":10,"october":10,"nov":11,"november":11,"dec":12,"december":12
    }

    def __init__(self):
        self.con = duckdb.connect(database=':memory:', read_only=False)
        self.con.execute("PRAGMA disable_progress_bar")

    # ---------------- public API ----------------

    def register_dataframe(self, name: str, df: pd.DataFrame) -> Dict[str, Any]:
        """Register/replace a pandas DataFrame as a DuckDB view named <name>."""
        df = df.copy()
        ts_col = self._infer_datetime_col(df)
        if ts_col:
            df["date"] = pd.to_datetime(df[ts_col], errors="coerce")
        # alias a single numeric to `value` for convenience
        num_cols = df.select_dtypes(include=["number"]).columns.tolist()
        alias_value = None
        if len(num_cols) == 1 and "value" not in df.columns:
            df["value"] = df[num_cols[0]]
            alias_value = num_cols[0]
        self.con.register(name, df)
        return {
            "table": name,
            "datetime_col": ts_col,
            "aliased_value_from": alias_value,
            "columns": list(df.columns),
        }

    async def query_via_llm(self, table: str, user_prompt: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Ask the LLM (Foundry agent) to produce DuckDB SQL from a natural-language prompt.
        If provider isn't 'foundry' or the call fails, the caller can fallback to `query()`.
        """
        schema = self._schema_for_prompt(table)
        sys = (
            "You write DuckDB-compatible SQL. Return a single SELECT or a WITH...SELECT.\n"
            f"Use only the table named {table}. If an identifier has spaces, hyphens, or starts with a digit, "
            "wrap it in double quotes, e.g., \"USD-PHP Rate\". Use date_trunc('month', date) for monthly.\n"
            "Do not include comments or prose—SQL only."
        )
        prompt = (
            f"User request:\n{user_prompt}\n\n"
            f"Schema:\n{json.dumps(schema, ensure_ascii=False)}\n\n"
            "Return only SQL. Use double quotes around any non-simple identifiers."
        )

        chat_settings = cl.user_session.get("chat_settings", {})
        provider = chat_settings.get("model_provider", "litellm")
        if provider != "foundry":
            # signal to caller to use fallback
            raise RuntimeError("LLM provider not available for SQL generation")

        from utils.foundry import chat_agent
        cl.user_session.set("analytics_mode", True)
        raw = await chat_agent(f"[SYSTEM]{sys}\n\n[USER]{prompt}")
        sql = raw.strip().strip("`")
        if sql.startswith("```"):
            parts = sql.split("```")
            if len(parts) >= 2:
                sql = parts[1]
        sql = sql.strip().rstrip(";")
        self._guard_sql(sql)
        df = self.con.execute(sql).fetchdf()
        return df, {"mode": "llm", "sql": sql, "detected": {"source": "foundry"}}

    def query(self, table: str, user_prompt: str, prefer_column: Optional[str]=None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        If prompt looks like SQL (starts with WITH/SELECT), validate and run it.
        Otherwise, do a best-effort NL→SQL conversion for common analytics asks.
        """
        sql_mode = self._looks_like_sql(user_prompt)
        meta: Dict[str, Any] = {"mode": "sql" if sql_mode else "nl", "applied": []}

        if sql_mode:
            sql = user_prompt.strip().rstrip(";")
            self._guard_sql(sql)
            df = self.con.execute(sql).fetchdf()
            meta["sql"] = sql
            return df, meta

        # NL → SQL fallback
        sql, nl_meta = self._nl_to_sql(table, user_prompt, prefer_column)
        self._guard_sql(sql)
        df = self.con.execute(sql).fetchdf()
        meta.update(nl_meta)
        meta["sql"] = sql
        return df, meta

    # ---------------- internals ----------------

    def _looks_like_sql(self, text: str) -> bool:
        t = text.lstrip().lower()
        return t.startswith("with ") or t.startswith("select ")

    def _guard_sql(self, sql: str) -> None:
        low = re.sub(r"\s+", " ", sql.lower())
        for tok in self._BLOCKED_TOKENS:
            if f"{tok} " in low or low.endswith(tok):
                raise ValueError(f"Blocked SQL token detected: {tok}")
        if " select " not in f" {low} " and not low.startswith("select "):
            raise ValueError("Only SELECT/CTE queries are allowed.")

    def _infer_datetime_col(self, df: pd.DataFrame) -> Optional[str]:
        for c in df.columns:
            if str(c).lower() in {"date","datetime","time","timestamp"}:
                return c
        for c in df.columns:
            try:
                parsed = pd.to_datetime(df[c], errors="coerce")
                if parsed.notna().mean() >= 0.7:
                    return c
            except Exception:
                pass
        return None

    def _schema_for_prompt(self, table: str) -> dict:
        cols = [r[0] for r in self.con.execute(f"PRAGMA table_info('{table}')").fetchall()]
        sample = self.con.execute(f"SELECT * FROM {table} LIMIT 5").fetchdf().to_dict(orient="records")
        return {"table": table, "columns": cols, "sample": sample}

    def _detect_cadence(self, prompt: str) -> Optional[str]:
        p = prompt.lower()
        for k, v in self.CADENCE_KEYWORDS.items():
            if k in p:
                return v
        return None

    def _detect_agg(self, prompt: str) -> str:
        p = prompt.lower()
        for k, v in self.AGG_KEYWORDS.items():
            if k in p:
                return v
        return "avg"

    def _extract_years(self, prompt: str) -> List[int]:
        p = prompt.lower()
        yrs = [int(y) for y in re.findall(r"\b(19|20)\d{2}\b", p)]
        m = re.search(r"\b((19|20)\d{2})\s*(?:-|to|–|—)\s*((19|20)\d{2})\b", p)
        if m:
            a, b = int(m.group(1)), int(m.group(3))
            lo, hi = min(a,b), max(a,b)
            return list(range(lo, hi+1))
        return sorted(set(yrs))

    def _extract_months(self, prompt: str) -> List[int]:
        tokens = re.findall(r"[a-zA-Z]+", prompt)
        months = [self.MONTHS[t.lower()] for t in tokens if t.lower() in self.MONTHS]
        months += [int(m) for m in re.findall(r"\bmonth\s*(\d{1,2})\b", prompt.lower()) if 1 <= int(m) <= 12]
        return sorted(set(months))

    def _find_measure_column(self, con: duckdb.DuckDBPyConnection, table: str, prompt: str, prefer: Optional[str]) -> Optional[str]:
        cols = [str(r[0]) for r in con.execute(f"PRAGMA table_info('{table}')").fetchall()]
        nums = [str(r[0]) for r in con.execute(
            f"SELECT column_name FROM information_schema.columns "
            f"WHERE table_name='{table}' AND data_type IN ('DOUBLE','BIGINT','INTEGER','DECIMAL','HUGEINT','SMALLINT','REAL')"
        ).fetchall()]

        if prefer and str(prefer) in cols:
            return str(prefer)

        m = re.search(r"\b([A-Z]{3})\s*[-/\s]\s*([A-Z]{3})\b", prompt.upper())
        if m:
            pair = f"{m.group(1)}-{m.group(2)}".lower()
            for c in cols:
                cc = str(c).replace("/", "-").replace(" ", "-").lower()
                if pair in cc:
                    return c

        if "value" in cols:
            return "value"
        if len(nums) == 1:
            return nums[0]
        for guess in ["rate","price","close","open","mid","mean","avg","amount","index","value"]:
            if guess in [c.lower() for c in cols]:
                return [c for c in cols if c.lower()==guess][0]
        return nums[0] if nums else None

    def _sql_agg(self, agg: str) -> str:
        return {"avg":"avg","sum":"sum","median":"median","min":"min","max":"max","stddev":"stddev"}.get(agg, "avg")

    def _safe_alias(self, alias: str) -> str:
        alias = str(alias).replace('"', '')
        return alias if _SIMPLE_ID.match(alias) else f"\"{alias}\""

    def _nl_to_sql(self, table: str, prompt: str, prefer_column: Optional[str]) -> Tuple[str, Dict[str, Any]]:
        cadence = self._detect_cadence(prompt)      # day/week/month/quarter/year
        agg = self._detect_agg(prompt)              # avg/sum/median/min/max/stddev
        years = self._extract_years(prompt)
        months = self._extract_months(prompt)

        measure = self._find_measure_column(self.con, table, prompt, prefer_column)

        wheres = []
        if years:
            lo, hi = min(years), max(years)
            wheres.append(f"EXTRACT(year FROM CAST(date AS TIMESTAMP)) BETWEEN {lo} AND {hi}")
        if months:
            mlist = ",".join(str(m) for m in months)
            wheres.append(f"EXTRACT(month FROM CAST(date AS TIMESTAMP)) IN ({mlist})")
        where_sql = ("WHERE " + " AND ".join(wheres)) if wheres else ""

        date_col = "date"
        if cadence and measure:
            measure_q = _qid(measure)
            sql = f"""
WITH base AS (
  SELECT *
  FROM {table}
  {where_sql}
)
SELECT
  date_trunc('{cadence}', CAST({date_col} AS TIMESTAMP)) AS period,
  {self._sql_agg(agg)}({measure_q}) AS {self._safe_alias(agg + '_' + str(measure))}
FROM base
GROUP BY 1
ORDER BY 1
""".strip()
            meta = {
                "mode": "nl",
                "applied": [f"cadence={cadence}", f"agg={agg}", f"measure={measure}"] + ([f"years={min(years)}..{max(years)}"] if years else []) + ([f"months={months}"] if months else []),
                "detected": {"cadence": cadence, "agg": agg, "measure": measure, "years": years, "months": months},
            }
            return sql, meta

        if measure:
            measure_q = _qid(measure)
            sql = f"""
SELECT CAST({date_col} AS TIMESTAMP) AS date, {measure_q} AS value
FROM {table}
{where_sql}
ORDER BY CAST({date_col} AS TIMESTAMP)
""".strip()
            meta = {
                "mode": "nl",
                "applied": ([f"years={min(years)}..{max(years)}"] if years else []) + ([f"months={months}"] if months else []),
                "detected": {"cadence": None, "agg": None, "measure": measure, "years": years, "months": months},
            }
            return sql, meta

        sql = f"""
SELECT *
FROM {table}
{where_sql}
ORDER BY CAST({date_col} AS TIMESTAMP)
""".strip()
        meta = {
            "mode": "nl",
            "applied": ([f"years={min(years)}..{max(years)}"] if years else []) + ([f"months={months}"] if months else []),
            "detected": {"cadence": None, "agg": None, "measure": None, "years": years, "months": months},
        }
        return sql, meta
