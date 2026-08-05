import csv

# (input_shape, in_dtype, out_dtype, depth)  -- axis forced to -1 (depth-innermost),
# matching the pyasc kernel's actual output layout. on/off values irrelevant for perf.
CASES = [
    ([1,1,593,1,1], "int32",  "int32",   31),
    ([1,997],       "int32",  "int32",   64),
    ([1,1,1,1,1,3712,1], "int32","float32",3511),
    ([1,9216],      "int32",  "int32",   2),
    ([9600],        "int32",  "int32",   2),
    ([1,1024,2,4,6],"int32",  "int32",   4),
    ([1,65536],     "int32",  "int32",   2),
    ([1,1,1,1,1,4793,28],"int32","float32",184),
    ([2328,1,1,1,1,101,1],"int32","float32",1),
    ([2,16,256,256],"int32",  "int32",   2),
    ([359,167,1,1,163],"int32","int32",  1),
    ([42767,7,16,16],"int32", "float16", 2),
    ([1259,1,192,2,127],"int32","float32",3),
    ([800,1],       "int32",  "float32", 2),
    ([1,1],         "int32",  "float32", 7),
    ([65536],       "int32",  "float32", 2),
]

HEADER = ["network_name","testcase_name","op_name","stc_input_dtypes","stc_ori_inputs",
"stc_ori_outputs","stc_input_ori_formats","output_ori_formats","other_compilation_params",
"other_runtime_params","stc_inputs","output_dtypes","stc_outputs","stc_input_formats",
"output_formats","dyn_inputs","dyn_input_dtypes","dyn_outputs","dyn_ori_inputs","dyn_ori_outputs",
"dyn_input_formats","dyn_input_ori_formats","dyn_input_ranges","dyn_output_ranges",
"dyn_input_as_list_distribution","stc_input_as_list_distribution","input_as_variable","stc_op_name",
"const_input_indexes","precision_tolerances","input_data_ranges","strict_precision_mode",
"absolute_precision","shape_check","output_inplace_indexes","random_buff","is_enabled","bucket",
"arity","position","tiling_key","kernel"]

def tup(t):
    # python-repr tuple string, e.g. (200, 10) -> "(200, 10)"; single -> "(1,)"
    if len(t)==1:
        return "(%d,)" % t[0]
    return "(" + ", ".join(str(x) for x in t) + ")"

def bucket(n):
    if n <= 1024: return "small"
    if n < 5_000_000: return "medium"
    return "large"

rows=[]
for i,(shp,ind,outd,depth) in enumerate(CASES, start=1):
    total=1
    for s in shp: total*=s
    out_shape = shp + [depth]
    rx = len(shp); ro = len(out_shape)
    # 4 inputs: x, depth(1,), on(1,), off(1,)
    stc_input_dtypes = "('%s', 'int32', '%s', '%s')" % (ind, outd, outd)
    stc_ori_inputs   = "(%s, (1,), (1,), (1,))" % tup(tuple(shp))
    stc_ori_outputs  = "(%s,)" % tup(tuple(out_shape))
    fmt4 = "('ND', 'ND', 'ND', 'ND')"; fmt1 = "('ND',)"
    comp = "{'axis': -1}"
    runt = "{'axis': -1, 'depth': [%d]}" % depth
    output_dtypes = "('%s',)" % outd
    dyn_x = "(" + ", ".join(["-1"]*rx) + ("," if rx==1 else "") + ")"
    dyn_out = "(" + ", ".join(["-1"]*ro) + ("," if ro==1 else "") + ")"
    dyn_inputs = "(%s, (-1,), (-1,), (-1,))" % dyn_x
    dyn_outputs = "(%s,)" % dyn_out
    rng_x = "(" + ", ".join(["(1, None)"]*rx) + ("," if rx==1 else "") + ")"
    rng_out = "(" + ", ".join(["(1, None)"]*ro) + ("," if ro==1 else "") + ")"
    dyn_input_ranges = "(%s, ((1, None),), ((1, None),), ((1, None),))" % rng_x
    dyn_output_ranges = "(%s,)" % rng_out
    row = [
        "UNKNOWN", "one_hot_%05d" % i, "one_hot", stc_input_dtypes, stc_ori_inputs,
        stc_ori_outputs, fmt4, fmt1, comp, runt, stc_ori_inputs, output_dtypes,
        stc_ori_outputs, fmt4, fmt1, dyn_inputs, stc_input_dtypes, dyn_outputs,
        dyn_inputs, dyn_outputs, fmt4, fmt4, dyn_input_ranges, dyn_output_ranges,
        "()", "()", "()", "one_hot", "(1,)", "", "((0, %d),)" % depth, "1",
        "", "1", "()", "", "1", bucket(total), "4", "", "1", "one_hot",
    ]
    assert len(row)==len(HEADER), (len(row), len(HEADER))
    rows.append(row)

with open("/tmp/one_hot_selected_representative.csv","w",newline="") as f:
    w=csv.writer(f)
    w.writerow(HEADER)
    w.writerows(rows)
print("wrote %d rows" % len(rows))
print(open("/tmp/one_hot_selected_representative.csv").readline().strip()[:80],"...")
# print first 2 data rows for eyeballing
import itertools
for line in itertools.islice(open("/tmp/one_hot_selected_representative.csv"),1,3):
    print(line.rstrip())
