import csv

HEADER = ["network_name","testcase_name","op_name","stc_input_dtypes","stc_ori_inputs",
"stc_ori_outputs","stc_input_ori_formats","output_ori_formats","other_compilation_params",
"other_runtime_params","stc_inputs","output_dtypes","stc_outputs","stc_input_formats",
"output_formats","dyn_inputs","dyn_input_dtypes","dyn_outputs","dyn_ori_inputs","dyn_ori_outputs",
"dyn_input_formats","dyn_input_ori_formats","dyn_input_ranges","dyn_output_ranges",
"dyn_input_as_list_distribution","stc_input_as_list_distribution","input_as_variable","stc_op_name",
"const_input_indexes","precision_tolerances","input_data_ranges","strict_precision_mode",
"absolute_precision","shape_check","output_inplace_indexes","random_buff","is_enabled","bucket",
"arity","position","tiling_key","kernel"]
assert len(HEADER) == 42


def tup(t):
    if len(t) == 1:
        return "(%d,)" % t[0]
    return "(" + ", ".join(str(x) for x in t) + ")"


def fmt(n):
    return "(" + ", ".join(["'ND'"] * n) + ("," if n == 1 else "") + ")"


def bucket(total):
    if total <= 1024:
        return "small"
    if total < 5_000_000:
        return "medium"
    return "large"


def dyn_shape(rank):
    return "(" + ", ".join(["-1"] * rank) + ("," if rank == 1 else "") + ")"


def dyn_rng(rank):
    return "(" + ", ".join(["(1, None)"] * rank) + ("," if rank == 1 else "") + ")"


def write(fname, rows):
    with open(fname, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(HEADER)
        w.writerows(rows)
    print("wrote %d rows to %s" % (len(rows), fname))


# ---------- reciprocal (arity 1) ----------
# (shape, dtype)
RECIP = [
    ([1024, 6144], "float32"),
    ([8192, 1024], "float32"),
    ([2048, 8192], "float32"),
    ([128, 2, 512], "float16"),
    ([1024, 6144], "float16"),
]
rows = []
for i, (shp, dt) in enumerate(RECIP, 1):
    total = 1
    for s in shp:
        total *= s
    rank = len(shp)
    sin = "(%s,)" % tup(tuple(shp))
    sout = sin
    row = ["UNKNOWN", "reciprocal_na_%05d" % i, "reciprocal",
           "('%s',)" % dt, sin, sout, fmt(rank), fmt(rank), "{}", "{}",
           sin, "('%s',)" % dt, sout, fmt(rank), fmt(rank),
           "(%s,)" % dyn_shape(rank), "('%s',)" % dt, "(%s,)" % dyn_shape(rank),
           "(%s,)" % dyn_shape(rank), "(%s,)" % dyn_shape(rank), fmt(rank), fmt(rank),
           "(%s,)" % dyn_rng(rank), "(%s,)" % dyn_rng(rank),
           "()", "()", "()", "reciprocal", "()", "", "((None, None),)", "1",
           "", "1", "()", "", "1", bucket(total), "1", "", "", "reciprocal"]
    assert len(row) == 42, len(row)
    rows.append(row)
write("/tmp/reciprocal_na.csv", rows)


# ---------- reduce_max (arity 2: x + axes const) ----------
# (shape, axis, dtype)
REDMAX = [
    ([3072, 113, 24], -1, "float32"),
    ([4608, 115, 12], -1, "float32"),
    ([1500, 61, 61], -1, "float32"),
    ([1, 128, 144], 1, "float32"),
    ([1024, 100, 2, 1], 2, "float32"),
    ([64, 32, 48], 1, "float32"),
    ([8, 4, 2, 64], 2, "float32"),
]
rows = []
for i, (shp, axis, dt) in enumerate(REDMAX, 1):
    total = 1
    for s in shp:
        total *= s
    rank = len(shp)
    ax = axis if axis >= 0 else rank + axis
    out_shape = [s for j, s in enumerate(shp) if j != ax]
    orank = len(out_shape)
    sin = "(%s, (1,))" % tup(tuple(shp))
    sout = "(%s,)" % tup(tuple(out_shape))
    comp = "{'keep_dims': False, 'noop_with_empty_axes': True}"
    runt = "{'axes': [%d], 'keep_dims': False, 'noop_with_empty_axes': True}" % axis
    pos = "last" if ax == rank - 1 else ""
    dyn_in = "(%s, (-1,))" % dyn_shape(rank)
    dyn_out = "(%s,)" % dyn_shape(orank)
    dyn_in_rng = "(%s, ((1, None),))" % dyn_rng(rank)
    dyn_out_rng = "(%s,)" % dyn_rng(orank)
    row = ["UNKNOWN", "reduce_max_na_%05d" % i, "reduce_max",
           "('%s', 'int32')" % dt, sin, sout,
           "('ND', 'ND')", fmt(orank), comp, runt,
           sin, "('%s',)" % dt, sout, "('ND', 'ND')", fmt(orank),
           dyn_in, "('%s', 'int32')" % dt, dyn_out,
           dyn_in, dyn_out, "('ND', 'ND')", "('ND', 'ND')",
           dyn_in_rng, dyn_out_rng,
           "()", "()", "()", "reduce_max", "(1,)", "", "((None, None),)", "1",
           "", "1", "()", "", "1", bucket(total), "2", pos, "", "reduce_max"]
    assert len(row) == 42, len(row)
    rows.append(row)
write("/tmp/reduce_max_na.csv", rows)


# ---------- addcdiv (arity 4) ----------
# (shape, dtype)  inputs: input, tensor1, tensor2, value(1,)
ADDCDIV = [
    ([1024, 1024], "float16"),
    ([98166, 128], "float16"),
]
rows = []
for i, (shp, dt) in enumerate(ADDCDIV, 1):
    total = 1
    for s in shp:
        total *= s
    rank = len(shp)
    st = tup(tuple(shp))
    sin = "(%s, %s, %s, (1,))" % (st, st, st)
    sout = "(%s,)" % st
    dtypes4 = "('%s', '%s', '%s', '%s')" % (dt, dt, dt, dt)
    dyn_in = "(%s, %s, %s, (-1,))" % (dyn_shape(rank), dyn_shape(rank), dyn_shape(rank))
    dyn_out = "(%s,)" % dyn_shape(rank)
    dyn_in_rng = "(%s, %s, %s, ((1, None),))" % (dyn_rng(rank), dyn_rng(rank), dyn_rng(rank))
    dyn_out_rng = "(%s,)" % dyn_rng(rank)
    f4 = "('ND', 'ND', 'ND', 'ND')"
    idr = "((None, None), (None, None), (None, None), (None, None))"
    row = ["UNKNOWN", "addcdiv_na_%05d" % i, "addcdiv",
           dtypes4, sin, sout, f4, fmt(rank), "{}", "{}",
           sin, "('%s',)" % dt, sout, f4, fmt(rank),
           dyn_in, dtypes4, dyn_out, dyn_in, dyn_out, f4, f4,
           dyn_in_rng, dyn_out_rng,
           "()", "()", "()", "addcdiv", "()", "", idr, "1",
           "", "1", "()", "", "1", bucket(total), "4", "", "", "addcdiv"]
    assert len(row) == 42, len(row)
    rows.append(row)
write("/tmp/addcdiv_na.csv", rows)
