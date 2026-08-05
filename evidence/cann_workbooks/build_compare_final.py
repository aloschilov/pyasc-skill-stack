import openpyxl, ast, json, math, re

# ---- fresh pyasc NPU medians (us), STATIC variant, --profile --runs 10 ----
# parsed from evidence/mr417_hw_profile.txt
PROF = "/home/aloschilov/workspace/pyasc-fork-combined/evidence/mr417_hw_profile.txt"

# key: (op, shape_tuple, dtype)  -> static us
pyasc = {"reciprocal": {}, "reduce_max": {}, "addcdiv": {}}
# For reduce_max we also track the reduce axis via case grouping.
dtmap = {"f32": "float32", "f16": "float16", "bf16": "bfloat16"}


def parse_shape(s):
    return tuple(int(x) for x in re.findall(r"-?\d+", s))


op = None
for line in open(PROF):
    line = line.strip()
    if line.startswith("## reciprocal"):
        op = "reciprocal"; continue
    if line.startswith("## addcdiv"):
        op = "addcdiv"; continue
    if line.startswith("## reduce_max"):
        op = "reduce_max"; continue
    if not line or line.startswith("#") or op is None:
        continue
    m = re.search(r"\[([0-9,\s]+)\]\s+(?:axis\d+\s+)?(f32|f16|bf16)\s+.*static\s+([0-9.]+)", line)
    if not m:
        continue
    shp = parse_shape("[" + m.group(1) + "]")
    dt = dtmap[m.group(2)]
    st = float(m.group(3))
    # axis label for reduce_max middle cases
    axis = None
    ma = re.search(r"axis(\d+)", line)
    if ma:
        axis = int(ma.group(1))
    pyasc[op][(shp, dt, axis)] = st

# ---- CANN CST from merged-MR workbooks (last-axis / f32) ----
wbmap = {"reciprocal": "reciprocal_perf.xlsx", "reduce_max": "reduce_max_perf.xlsx", "addcdiv": "addcdiv_perf.xlsx"}
cann_wb = {}
for opn, fn in wbmap.items():
    wb = openpyxl.load_workbook(fn, data_only=True); ws = wb["Data"]
    rows = list(ws.iter_rows(values_only=True)); hdr = rows[0]; ci = {h: i for i, h in enumerate(hdr)}
    d = {}
    for r in rows[1:]:
        shp = tuple(ast.literal_eval(r[ci['stc_ori_inputs']])[0])
        d[shp] = r[ci['CST_PERF']]
    cann_wb[opn] = d

# ---- new on-HW CANN CST for the previously-N/A shapes ----
na = json.load(open("cann_cst_na.json"))


def cann_lookup(op, shp, dt, axis):
    # 1) NA table (keyed with dtype and, for reduce_max, axis)
    na_op = na.get(op, {})
    key_dt = "%s|%s" % (list(shp), dt)
    if op == "reduce_max":
        ax = axis if axis is not None else -1
        key = "%s|%s|%s" % (list(shp), dt, ax)
        if key in na_op:
            return na_op[key]
    if key_dt in na_op:
        return na_op[key_dt]
    # 2) workbook (f32 last-axis) keyed by input shape only
    if dt == "float32" and (axis is None or axis == len(shp) - 1):
        return cann_wb[op].get(shp)
    return None


def gm(xs):
    xs = [x for x in xs if x]
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else None


report = []
geos = {}
for op in ["reciprocal", "addcdiv", "reduce_max"]:
    report.append("\n### %s\n" % op)
    if op == "reduce_max":
        report.append("| input shape | axis | dtype | pyasc us (NPU) | CANN CST us | ratio (CANN/pyasc) |")
        report.append("|---|---|---|---|---|---|")
    else:
        report.append("| input shape | dtype | pyasc us (NPU) | CANN CST us | ratio (CANN/pyasc) |")
        report.append("|---|---|---|---|---|")
    ratios = []
    for (shp, dt, axis), pv in pyasc[op].items():
        cv = cann_lookup(op, shp, dt, axis)
        dts = "f32" if dt == "float32" else ("f16" if dt == "float16" else "bf16")
        if cv:
            ratio = cv / pv; ratios.append(ratio)
            if op == "reduce_max":
                axlbl = str(axis) if axis is not None else "-1"
                report.append("| %s | %s | %s | %.3f | %.3f | %.3f |" % (list(shp), axlbl, dts, pv, cv, ratio))
            else:
                report.append("| %s | %s | %.3f | %.3f | %.3f |" % (list(shp), dts, pv, cv, ratio))
        else:
            if op == "reduce_max":
                axlbl = str(axis) if axis is not None else "-1"
                report.append("| %s | %s | %s | %.3f | N/A | N/A |" % (list(shp), axlbl, dts, pv))
            else:
                report.append("| %s | %s | %.3f | N/A | N/A |" % (list(shp), dts, pv))
    g = gm(ratios); geos[op] = (g, len(ratios))
    report.append("\ngeomean ratio (n=%d): **%.3f**" % (len(ratios), g))

print("\n".join(report))
print("\n==== GEOMEANS ====")
for op, (g, n) in geos.items():
    print("%s: %.3f (n=%d)" % (op, g, n))
open("comparison_final.md", "w").write("\n".join(report))
