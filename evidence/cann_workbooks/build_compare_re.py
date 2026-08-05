import openpyxl, ast, json, math

# Remeasured pyasc NPU medians (us), --backend NPU --profile --runs 10 (5h later)
onehot_pyasc = {
 (1,1,593,1,1):142.256,(1,997):187.964,(1,1,1,1,1,3712,1):461.369,(1,9216):154.783,
 (9600,):144.669,(1,1024,2,4,6):156.04,(1,65536):176.951,(1,1,1,1,1,4793,28):697.951,
 (2328,1,1,1,1,101,1):558.034,(2,16,256,256):5435.77,(359,167,1,1,163):22915.508,
 (42767,7,16,16):193145.819,(1259,1,192,2,127):162678.231,(800,1):111.415,(1,1):1.407,(65536,):177.458,
}
onehot_cann = {
 (1,1,593,1,1):3.976,(1,997):4.445,(1,1,1,1,1,3712,1):16.364,(1,9216):2.755,
 (9600,):2.697,(1,1024,2,4,6):3.982,(1,65536):3.203,(1,1,1,1,1,4793,28):38.008,
 (2328,1,1,1,1,101,1):4.7,(2,16,256,256):27.905,(359,167,1,1,163):86.958,
 (42767,7,16,16):1139.62,(1259,1,192,2,127):1332.027,(800,1):2.397,(1,1):2.212,(65536,):2.864,
}

pyasc = {
 "reciprocal": {
   (1024,):1.544,(2400,):2.516,(16,5,1,64):2.661,(16,256):2.044,(16,320):2.668,
   (16,24,768):4.832,(128,1,2304):4.829,(2500,):2.529,(1200,):1.85,(2048,):1.643,
   (1500,):1.891,(1024,1,20):5.988,(1024,1,50):13.261,(1024,1,1000):13.615,(256,1):1.342,
   (100,14,10):7.786,(2048,1):1.629,(1024,6144):71.492,(8192,1024):96.171,(2048,8192):193.98,
 },
 "reduce_max": {
   (200,10):3.102,(13,2048,32):3.999,(10,2048,64):4.998,(45,2048,4):3.527,(64,2048,8):4.452,
   (70,2048,16):8.382,(2048,83,18):12.431,(1500,1,61):2.193,(3072,113,24):22.922,
   (4608,115,12):24.937,(1500,61,61):16.945,
 },
 "addcdiv": {
   (11734,16):5.954,(152,):1.334,(152,456):3.311,(1,168):1.335,(7,10):1.353,(8,):1.339,(80,):1.351,
   (98166,16):37.219,(1024,):1.348,(1,14,1):1.335,(1024,152):4.826,(421,):1.323,(256,320):3.384,
   (8,64):1.339,(1,40):1.356,(64,121):2.901,(48,):1.335,(1024,1024):25.239,(64,225,1):4.993,
   (16,16,1):1.333,(1820039,16):708.623,(315511,16):118.9,(98166,128):307.071,
 },
}

wbmap={"reciprocal":"reciprocal_perf.xlsx","reduce_max":"reduce_max_perf.xlsx","addcdiv":"addcdiv_perf.xlsx"}
cann={}
for op,fn in wbmap.items():
    wb=openpyxl.load_workbook(fn,data_only=True); ws=wb["Data"]
    rows=list(ws.iter_rows(values_only=True)); hdr=rows[0]; ci={h:i for i,h in enumerate(hdr)}
    d={}
    for r in rows[1:]:
        d[tuple(ast.literal_eval(r[ci['stc_ori_inputs']])[0])]=r[ci['CST_PERF']]
    cann[op]=d
pyasc["one_hot"]=onehot_pyasc; cann["one_hot"]=onehot_cann

onehot_depth = {
 (1,1,593,1,1):31,(1,997):64,(1,1,1,1,1,3712,1):3511,(1,9216):2,(9600,):2,(1,1024,2,4,6):4,
 (1,65536):2,(1,1,1,1,1,4793,28):184,(2328,1,1,1,1,101,1):1,(2,16,256,256):2,(359,167,1,1,163):1,
 (42767,7,16,16):2,(1259,1,192,2,127):3,(800,1):2,(1,1):7,(65536,):2,
}

def gm(xs):
    xs=[x for x in xs if x]
    return math.exp(sum(math.log(x) for x in xs)/len(xs)) if xs else None

report=[]
geos={}
for op in ["reciprocal","reduce_max","addcdiv","one_hot"]:
    report.append(f"\n### {op}\n")
    hdr = "| input shape | depth | pyasc us (NPU) | CANN CST us | ratio (CANN/pyasc) |" if op=="one_hot" else "| input shape | pyasc us (NPU) | CANN CST us | ratio (CANN/pyasc) |"
    sep = "|---|---|---|---|---|" if op=="one_hot" else "|---|---|---|---|"
    report.append(hdr); report.append(sep)
    ratios=[]
    for shp,pv in pyasc[op].items():
        cv=cann[op].get(shp)
        depthcol = f" {shp[-1] if False else ''} "  # placeholder
        if op=="one_hot":
            # depth = cann key's associated depth; derive from out not available -> use from onehot mapping order
            pass
        dstr = str(onehot_depth.get(shp,"")) if op=="one_hot" else None
        if cv:
            ratio=cv/pv; ratios.append(ratio)
            if op=="one_hot":
                report.append(f"| {list(shp)} | {dstr} | {pv:.3f} | {cv:.3f} | {ratio:.3f} |")
            else:
                report.append(f"| {list(shp)} | {pv:.3f} | {cv:.3f} | {ratio:.3f} |")
        else:
            if op=="one_hot":
                report.append(f"| {list(shp)} | {dstr} | {pv:.3f} | N/A | N/A |")
            else:
                report.append(f"| {list(shp)} | {pv:.3f} | N/A | N/A |")
    g=gm(ratios); geos[op]=(g,len(ratios))
    report.append(f"\ngeomean ratio (n={len(ratios)}): **{g:.3f}**")

print("\n".join(report))
print("\n==== GEOMEANS ====")
for op,(g,n) in geos.items():
    print(f"{op}: {g:.3f} (n={n})")
open("comparison_remeasure.md","w").write("\n".join(report))
