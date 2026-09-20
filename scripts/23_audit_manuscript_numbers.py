# -*- coding: utf-8 -*-
"""23 论文数字一致性扫描

NOTE: this script cross-checks `manuscript_numbers.csv` against manuscript
draft files (docx/md) that are internal working drafts, not included in
this public repository. It will not run out of the box; supply your own
manuscript draft file(s) at the paths in DOCS below (or point DOCS at your
own files) to use it. It is included for provenance/transparency about how
the paper's reported numbers were audited against the draft text, not as a
turnkey script.
==================================================================
把 manuscript_numbers.csv 与稿件文档(英文docx、中文md)里出现的
每一个数字比对,三类归档:不一致 / 无来源 / 未使用。docx 用 pandoc 转纯文本。

匹配容差:数字+可选百分号,统一按"文档数字自己展示了几位小数"为比较基准,
把 CSV 侧的原始高精度值就近舍入到这个精度再比较,允许 ±1 个末位数的舍入差
(见 values_match())。页码、年份、条件编号(C\\d{2} 里的两位数字)等噪声,靠
NUMBER_RE 本身要求"必须带小数点,或带百分号"排除掉了绝大多数(裸整数形式的
页码/年份/样本编号不会被提取),没有再另外写一层基于上下文的 SKIP_PATTERNS
——试过之后发现这类正则容易脆(容易连带误伤真实数字),而 Section 1 本身就
注明是"含大量噪声、需人工复核"的候选表,剩余噪声交给 Step 3 的人工复核处理
更可靠。

关于匹配逻辑的两处修正(相对最初的实现,均经实测验证):
1. 最初的"跨精度取并"写法——检查 round(v,1)/round(v,2)/round(v,3)/
   round(v,4) 是否有任一命中 known_values——会把粗精度(1位小数)下偶然相同
   的两个不同数值误判为"一致"。已知的旗舰案例 D_XRD 的 R² 冻结值是
   0.4886(0.48856...,precision=4),稿件里残留的旧值是 0.5025,这两个数在
   1位小数下都四舍五入成 0.5,最初的写法会把 0.5025 判定为"能在冻结表
   里找到来源"、不出现在候选表里,直接违背了本轮审计"揪出这个已知不一致"
   的目的。第一次修复尝试改成"只在文档数字精度与 CSV 声明精度两者中较粗
   的一级上比较",实测后发现**仍然不够**:CSV 里另一条完全无关、precision=1
   的行(某个方差分解百分比)在"取两者精度中较粗的一级"下又与 0.5025 意外
   重合,0.5025 依旧被吞掉。**最终改成完全不用 CSV 的 precision 列做容差**,
   统一按 token 自身在文档里展示的小数位数为基准(理由与例子见
   values_match() docstring)。修复后 0.5025 正确出现在第1节候选表里
   (已用脚本外的独立探针脚本核实)。
2. subprocess.run(pandoc, text=True) 在 Windows 上默认按控制台 GBK codepage
   解码 pandoc 的 UTF-8 stdout,遇到中文多字节序列直接 UnicodeDecodeError
   ——必须显式传 encoding="utf-8"(见 extract_text())。
3. brief 原始正则只认 ASCII 连字符 "-" 当负号,但三份稿件的表格/正文里负数
   几乎全部用排版负号 U+2212 "−"(Word/pandoc 常见排版惯例),导致负数 token
   丢符号变成正数,冻结表里带符号的原始值(如系数、相关系数)永远比对不上。
   已把 NUMBER_RE 和 parse_float() 都改成同时接受两种负号字符。
"""
import _bootstrap  # noqa

import re
import subprocess
from pathlib import Path

import pandas as pd

DOCS = {
    "PaperA_EN(docx)": "docs/PaperA_Manuscript_EN.docx",
    "论文A_v2(md)": "docs/论文A_材料篇_初稿_v2.md",
    "论文B_v2(md)": "docs/论文B_方法篇_初稿_v2.md",
}
NUMBERS_CSV = "manuscript/numbers/manuscript_numbers.csv"
OUT_REPORT = "reports/manuscript_number_audit.md"

# 数字 token:必须带小数点,或者带百分号(排除掉绝大多数裸整数形式的
# 页码/年份/样本编号,如 "2026"、"C07" 里的 "07"、"第 12 页" 里的 "12")。
# 负号同时接受 ASCII 连字符 "-" 和排版负号 U+2212 "−"——三份稿件的表格/正文
# 里负数几乎全部用的是后者(pandoc/Word 的排版负号),原先只认 ASCII "-" 会
# 把负号丢掉、只剩下正的数值部分,导致跟冻结表里带符号的原始值比对时符号对不上
# (例如 table6.D_XRD.precursor_L_coef=-0.4642... 在文档里写作 "−0.4642",
# 丢符号后变成 token "0.4642",永远无法跟负的冻结值匹配,被错误地计入"未使用"桶)。
NUMBER_RE = re.compile(r"(?<![\w.])[-−]?\d+\.\d+%?|(?<![\w.])[-−]?\d{2,}%")


def extract_text(label, path):
    p = Path(path)
    if p.suffix == ".docx":
        # Windows 控制台默认 codepage 是 GBK,subprocess.run(text=True) 不指定
        # encoding 时会用它解码 pandoc 的 UTF-8 stdout,遇到中文多字节序列就
        # UnicodeDecodeError——必须显式指定 encoding="utf-8"。
        result = subprocess.run(
            ["pandoc", "-t", "plain", str(p)],
            capture_output=True, text=True, encoding="utf-8", check=True,
        )
        return result.stdout
    return p.read_text(encoding="utf-8")


def extract_numbers(text):
    """返回 {规范化数字字符串: [出现次数上下文,...]}。"""
    found = {}
    for m in NUMBER_RE.finditer(text):
        tok = m.group(0)
        context = text[max(0, m.start() - 20):min(len(text), m.end() + 20)]
        found.setdefault(tok, []).append(context.replace("\n", " "))
    return found


def token_precision(tok):
    """token 自身的小数位数(百分号不计入)。整数 token(理论上不会出现,因为
    NUMBER_RE 要求整数必须带 %)按 0 位处理。"""
    core = tok.rstrip("%")
    if "." in core:
        return len(core.split(".")[1])
    return 0


def parse_float(tok):
    try:
        return float(str(tok).rstrip("%").replace("−", "-"))  # 排版负号 → ASCII
    except (TypeError, ValueError):
        return None


def values_match(token_val, token_prec, csv_raw_val):
    """比较基准统一用"文档数字自己展示了几位小数"(token_prec):把 CSV 的
    原始高精度浮点值就近舍入到 token 的小数位,再与 token 比较,容差为该
    精度下的 ±1.5 个末位单位(1.5 倍是为了让"差 1 个末位数"的舍入/誊抄误差
    稳妥落在容差内)。

    **不**用 CSV 的 `precision` 列(该数字在稿件表格里应保留几位小数的展示
    规格)来放宽容差——实测过这个更早的写法:CSV 里精度声明得比较粗的行
    (如方差分解百分比只保留 1 位小数)会在"取两者精度中较粗的一级比较"下,
    把文档里几乎任何邻近的数字都撞成"一致"。已知的旗舰案例——冻结表
    `table7.D_XRD.R2`=0.4886(0.48856...,precision=4)与稿件里残留的旧值
    0.5025——用那个写法比对时,0.5025 在 1 位小数下与另一条无关的、
    precision=1 的 CSV 行意外重合(两者都四舍五入成 0.5/0.4),导致 0.5025
    被误判为"能在冻结表里找到来源",不出现在候选表里,直接违背了本轮审计
    "揪出这个已知不一致"的目的。改成统一按 token 自身精度比较后,0.5025
    正确地不再匹配任何 CSV 值,进入候选表。"""
    rounded_csv = round(csv_raw_val, token_prec)
    tol = 1.5 * (10 ** (-token_prec))
    return abs(rounded_csv - token_val) <= tol


def main():
    numbers = pd.read_csv(NUMBERS_CSV)
    doc_texts = {label: extract_text(label, path) for label, path in DOCS.items()}
    doc_numbers = {label: extract_numbers(text) for label, text in doc_texts.items()}

    total_tokens = sum(len(toks) for toks in doc_numbers.values())

    inconsistent, no_source, unused = [], [], []

    # 建立 CSV 侧可比较的数值列表:(id, value_float_or_None, value_str, section)
    # 注意:CSV 的 precision 列只用于报告里展示,不参与 values_match 的容差
    # 计算(理由见 values_match 的 docstring)。
    csv_entries = []
    for _, row in numbers.iterrows():
        if pd.isna(row["value"]):
            no_source.append(row)
            continue
        v = parse_float(row["value"])
        csv_entries.append((row["id"], v, str(row["value"]), row["section"]))

    # --- 冻结表 -> 文档:每条 CSV 记录是否在任一文档里找到对应文本 ---
    # 数值型:按 values_match 数值比较;非数值型(比例 "26/27"、置信区间
    # "[+0.400, +0.866]" 等 21 条):退化为原始字符串子串匹配,匹配前把两边的
    # 排版负号 "−" 都归一化成 ASCII "-"(CSV 里的 CI 区间字符串本身用的是
    # ASCII "-",文档里排版成 "−",不归一化会导致这类条目被错误计入"未使用"桶)。
    doc_full_text_concat = "\n".join(doc_texts.values()).replace("−", "-")
    for (cid, v, vstr, section) in csv_entries:
        vstr_norm = vstr.replace("−", "-")
        found = False
        if v is not None:
            for label, toks in doc_numbers.items():
                for tok in toks:
                    tv = parse_float(tok)
                    if tv is None:
                        continue
                    if values_match(tv, token_precision(tok), v):
                        found = True
                        break
                if found:
                    break
        else:
            found = vstr_norm in doc_full_text_concat
        if not found:
            unused.append((cid, vstr, section))

    # --- 文档 -> 冻结表:文档里的每个数字能否在 CSV 里找到匹配来源 ---
    csv_numeric = [v for (_cid, v, _vstr, _sec) in csv_entries if v is not None]
    for label, toks in doc_numbers.items():
        for tok, ctxs in toks.items():
            tv = parse_float(tok)
            if tv is None:
                continue
            tprec = token_precision(tok)
            match = any(values_match(tv, tprec, cv) for cv in csv_numeric)
            if not match:
                inconsistent.append((label, tok, ctxs[0]))

    write_report(inconsistent, no_source, unused, numbers, doc_numbers, total_tokens)


def write_report(inconsistent, no_source, unused, numbers, doc_numbers, total_tokens):
    lines = ["# 论文数字一致性扫描报告\n\n",
             "> 由 `scripts/23_audit_manuscript_numbers.py` 自动生成。"
             "本报告只做分类,不做任何数字判断——三类之间的边界(尤其"
             "'不一致' vs '噪声匹配失败')需要人工复核后才能下结论。"
             "见文末「人工复核结论」小节。\n\n",
             "## 扫描范围\n\n"]
    for label, path in DOCS.items():
        lines.append(f"- {label}: `{path}`\n")
    lines.append(
        "\n`docs/论文A_材料篇_初稿.docx`(更早的中文草稿快照)已排除:"
        "pandoc 提取后 693 行,远短于 `论文A_材料篇_初稿_v2.md` 的 1061 行;"
        "抽查其摘要关键数字(88.8%/77.1%/67.4%/61.8%/54.2%/38.1%/10.3%/"
        "0.63-3.33% 等)在 v2.md 中逐一出现(且部分已被后续修订更新覆盖),"
        "未发现独立于 v2.md 之外的内容,判定为被 v2.md 取代的旧快照,"
        "符合路线图文档(`docs/期刊投稿_写作路线图_2026-08-06.md:6-8`)"
        "\"英文 docx + v2.md 为两份现行文档\"的定位。\n"
    )
    lines.append(f"\n三份文档共提取到 {total_tokens} 个带小数点或百分号的数字 token"
                  "(裸整数如页码/年份/样本编号两位数字,因不带小数点或百分号,"
                  "已被正则本身排除,未进入统计)。\n")

    lines.append("\n## 1. 疑似不一致或无法在冻结表匹配到的文档数字"
                 f"(候选 {len(inconsistent)} 条,含大量噪声,需人工复核)\n\n")
    lines.append("| 文档 | 数字 | 上下文 |\n|---|---|---|\n")
    for label, tok, ctx in inconsistent[:500]:
        # 上下文摘自 markdown 源文档,可能本身含 "|"(表格分隔符),
        # 不转义会把报告自己的 markdown 表格撑坏。
        safe_ctx = ctx.replace("|", "\\|")
        lines.append(f"| {label} | {tok} | {safe_ctx} |\n")

    lines.append(f"\n## 2. 冻结表里无来源的条目({len(no_source)} 条)\n\n")
    lines.append("| id | section | note |\n|---|---|---|\n")
    for row in no_source:
        lines.append(f"| {row['id']} | {row['section']} | {row['note']} |\n")

    lines.append(f"\n## 3. 冻结表里存在、但未在扫描的三份文档里找到对应文本的条目"
                 f"({len(unused)} 条)\n\n")
    lines.append("| id | value | section |\n|---|---|---|\n")
    for (cid, vstr, section) in unused:
        lines.append(f"| {cid} | {vstr} | {section} |\n")

    Path(OUT_REPORT).write_text("".join(lines), encoding="utf-8")
    print(f"已写出 {OUT_REPORT}")
    print(f"  不一致候选: {len(inconsistent)}")
    print(f"  无来源: {len(no_source)}")
    print(f"  未使用: {len(unused)}")


if __name__ == "__main__":
    main()
