# -*- coding: utf-8 -*-
from pathlib import Path
import re
import sys

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")


def locate_dirs():
    base = next(p for p in Path(".").rglob("截取后实验数据") if p.is_dir())
    default_dir = base / "默认"
    v500_dir = base / "500"
    return default_dir, v500_dir


def extract_orders(folder: Path):
    orders = []
    files = []
    for p in folder.rglob("*"):
        if not p.is_file():
            continue
        nums = re.findall(r"\d+", p.stem)
        if nums:
            orders.append(int(nums[0]))
            files.append(p.name)
    return sorted(set(orders)), files


def find_xlsx(version: str) -> Path:
    return next(p for p in Path(".").rglob("*.xlsx") if version in p.name and "candidate" not in p.name)


def load_params(version: str) -> pd.DataFrame:
    xlsx = find_xlsx(version)
    raw = pd.read_excel(xlsx, sheet_name=0, header=2)
    raw = raw[pd.to_numeric(raw.iloc[:, 0], errors="coerce").notna()].copy()
    raw.iloc[:, 0] = raw.iloc[:, 0].astype(int)
    rows = []
    for _, r in raw.iterrows():
        if pd.isna(r.iloc[6]) or pd.isna(r.iloc[7]) or pd.isna(r.iloc[9]):
            continue
        rows.append(
            {
                "版本": version,
                "执行顺序": int(r.iloc[0]),
                "块号": int(r.iloc[2]) if not pd.isna(r.iloc[2]) else None,
                "层": int(r.iloc[4]) if not pd.isna(r.iloc[4]) else None,
                "槽号": int(r.iloc[5]) if not pd.isna(r.iloc[5]) else None,
                "转速(rpm)": int(r.iloc[6]),
                "进给(mm/z)": float(r.iloc[7]),
                "每转进给(mm/r)": float(r.iloc[8]),
                "切深(mm)": float(r.iloc[9]),
            }
        )
    return pd.DataFrame(rows)


def print_missing_rate(params, missing, col):
    total = params.groupby(["版本", col]).size().rename("总数")
    miss = missing.groupby(["版本", col]).size().rename("缺失数")
    out = pd.concat([total, miss], axis=1).fillna(0)
    out["缺失率"] = out["缺失数"] / out["总数"]
    print(f"\n按 {col} 的缺失率")
    print(out.to_string())


def main():
    default_dir, v500_dir = locate_dirs()
    v3_have, v3_files = extract_orders(default_dir)
    v4_have, v4_files = extract_orders(v500_dir)

    params = pd.concat([load_params("v3"), load_params("v4")], ignore_index=True)
    v3_expected = set(params[params["版本"] == "v3"]["执行顺序"])
    v4_expected = set(params[params["版本"] == "v4"]["执行顺序"])

    v3_missing = sorted(v3_expected - set(v3_have))
    v4_missing = sorted(v4_expected - set(v4_have))
    missing = params[
        ((params["版本"] == "v3") & (params["执行顺序"].isin(v3_missing)))
        | ((params["版本"] == "v4") & (params["执行顺序"].isin(v4_missing)))
    ].copy()

    print(f"默认文件夹有效序号数(v3): {len(v3_have)}")
    print(f"500文件夹有效序号数(v4): {len(v4_have)}")
    print(f"v3缺失 {len(v3_missing)} 条: {v3_missing}")
    print(f"v4缺失 {len(v4_missing)} 条: {v4_missing}")
    print("\n缺失明细")
    print(missing.to_string(index=False))

    for col in ["切深(mm)", "转速(rpm)", "进给(mm/z)", "每转进给(mm/r)", "块号", "层"]:
        print(f"\n缺失数量分布：{col}")
        print(pd.crosstab(missing["版本"], missing[col]).to_string())

    print("\n缺失的块号-层-切深交叉分布")
    print(pd.crosstab([missing["版本"], missing["块号"], missing["层"]], missing["切深(mm)"]).to_string())

    for col in ["切深(mm)", "转速(rpm)", "进给(mm/z)", "块号", "层"]:
        print_missing_rate(params, missing, col)


if __name__ == "__main__":
    main()
