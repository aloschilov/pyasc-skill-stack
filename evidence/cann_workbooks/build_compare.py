import openpyxl, ast, json, math

# one_hot: pyasc NPU static medians (us) and CANN CST (us from TTK), keyed by input shape
onehot_pyasc = {
 (1,1,593,1,1):142.489,(1,997):187.914,(1,1,1,1,1,3712,1):462.621,(1,9216):153.404,
 (9600,):144.271,(1,1024,2,4,6):155.674,(1,65536):174.25,(1,1,1,1,1,4793,28):695.171,
 (2328,1,1,1,1,101,1):556.479,(2,16,256,256):5341.452,(359,167,1,1,163):22854.275,
 (42767,7,16,16):190064.309,(1259,1,192,2,127):160749.3,(800,1):109.376,(1,1):1.387,(65536,):174.261,
}
onehot_cann = {
 (1,1,593,1,1):3.976,(1,997):4.445,(1,1,1,1,1,3712,1):16.364,(1,9216):2.755,
 (9600,):2.697,(1,1024,2,4,6):3.982,(1,65536):3.203,(1,1,1,1,1,4793,28):38.008,
 (2328,1,1,1,1,101,1):4.7,(2,16,256,256):27.905,(359,167,1,1,163):86.958,
 (42767,7,16,16):1139.62,(1259,1,192,2,127):1332.027,(800,1):2.397,(1,1):2.212,(65536,):2.864,
}

# pyasc NPU medians (us), keyed by input shape tuple, from --backend NPU --profile --runs 10
pyasc = {
 "reciprocal": {
   (1024,):1.429,(2400,):2.411,(16,5,1,64):2.477,(16,256):1.921,(16,320):2.479,
   (16,24,768):4.774,(128,1,2304):4.78,(2500,):2.433,(1200,):1.878,(2048,):1.435,
   (1500,):1.878,(1024,1,20):5.842,(1024,1,50):13.055,(1024,1,1000):12.927,(256,1):1.304,
   (100,14,10):7.698,(2048,1):1.439,(1024,6144):70.856,(8192,1024):95.612,(2048,8192):195.582,
 },
 "reduce_max": {
   (200,10):2.979,(13,2048,32):4.021,(10,2048,64):5.035,(45,2048,4):3.542,(64,2048,8):4.432,
   (70,2048,16):8.521,(2048,83,18):12.377,(1500,1,61):2.206,(3072,113,24):23.485,
   (4608,115,12):25.502,(1500,61,61):16.576,
 },
 "addcdiv": {
   (11734,16):5.68,(152,):1.349,(152,456):3.363,(1,168):1.37,(7,10):1.377,(8,):1.368,(80,):1.381,
   (98166,16):36.842,(1024,):1.355,(1,14,1):1.37,(1024,152):4.935,(421,):1.368,(256,320):3.394,
   (8,64):1.364,(1,40):1.373,(64,121):3.12,(48,):1.377,(1024,1024):25.69,(64,225,1):5.223,
   (16,16,1):1.357,(1820039,16):716.206,(315511,16):117.81,(98166,128):300.395,
 },
}

wbmap={"reciprocal":"reciprocal_perf.xlsx","reduce_max":"reduce_max_perf.xlsx","addcdiv":"addcdiv_perf.xlsx"}
cann={}
for op,fn in wbmap.items():
    wb=openpyxl.load_workbook(fn,data_only=True); ws=wb["Data"]
    rows=list(ws.iter_rows(values_only=True)); hdr=rows[0]; ci={h:i for i,h in enumerate(hdr)}
    d={}
    for r in rows[1:]:
        tup=ast.literal_eval(r[ci['stc_ori_inputs']])
        shp=tuple(tup[0])
        d[shp]=r[ci['CST_PERF']]
    cann[op]=d

def gm(xs):
    xs=[x for x in xs if x]
    return math.exp(sum(math.log(x) for x in xs)/len(xs)) if xs else None

pyasc["one_hot"]=onehot_pyasc
cann["one_hot"]=onehot_cann

report=[]
for op in ["reciprocal","reduce_max","addcdiv","one_hot"]:
    report.append(f"\n### {op}\n")
    report.append("| input shape | pyasc us (NPU) | CANN CST us | ratio (CANN/pyasc) |")
    report.append("|---|---|---|---|")
    ratios=[]
    for shp,pv in pyasc[op].items():
        cv=cann[op].get(shp)
        if cv:
            ratio=cv/pv; ratios.append(ratio)
            report.append(f"| {list(shp)} | {pv:.3f} | {cv:.3f} | {ratio:.3f} |")
        else:
            report.append(f"| {list(shp)} | {pv:.3f} | N/A | N/A |")
    g=gm(ratios)
    report.append(f"\ngeomean ratio (cases with CANN, n={len(ratios)}): **{g:.3f}**  (ratio>1 = pyasc faster than CANN)")

txt="\n".join(report)
print(txt)
open("comparison_3ops.md","w").write(txt)

# also dump machine-readable
outj={op:{str(list(s)):{"pyasc":pyasc[op][s],"cann":cann[op].get(s)} for s in pyasc[op]} for op in pyasc}
json.dump(outj, open("comparison_all.json","w"), indent=2)
