import os
from pathlib import Path

import pandas as pd
from mysql.connector import connect, Error

excel_path = Path(os.getenv("MYSQL_IMPORT_FILE", "mysql/pharma_sales_records.xlsx"))

columns = [
    "sale_id",
    "drug_id",
    "sale_date",
    "quantity_sold",
    "unit_price",
    "total_amount",
    "customer_name",
    "region",
    "sales_rep"
]

sql = """
INSERT INTO sales_records (
    sale_id,
    drug_id,
    sale_date,
    quantity_sold,
    unit_price,
    total_amount,
    customer_name,
    region,
    sales_rep
)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
"""

conn = None
cursor = None

try:
    
    df = pd.read_excel(excel_path)

    missing_columns = [
        column for column in columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(f"Excel 缺少字段：{missing_columns}")

    df["sale_date"] = pd.to_datetime(
        df["sale_date"],
        errors="raise"
    ).dt.date

    df = df.where(pd.notna(df), None)

    conn = connect(
        host=os.getenv("MYSQL_HOST", "localhost"),
        port=int(os.getenv("MYSQL_PORT", "3306")),
        user=os.environ["MYSQL_USER"],
        password=os.environ["MYSQL_PASSWORD"],
        database=os.environ["MYSQL_DATABASE"],
        charset=os.getenv("MYSQL_CHARSET", "utf8mb4"),
    )

    cursor = conn.cursor()

    for row in df[columns].itertuples(index=False, name=None):
        cursor.execute(sql, row)

    conn.commit()

    print(f"成功导入 {len(df)} 条数据")

except (Error, ValueError, FileNotFoundError) as exc:
    
    if conn is not None and conn.is_connected():
        conn.rollback()

    print(f"导入失败：{exc}")

finally:
    if cursor is not None:
        cursor.close()

    if conn is not None and conn.is_connected():
        conn.close()
