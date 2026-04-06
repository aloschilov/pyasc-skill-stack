# asc.language.basic.metrics_prof_start

### asc.language.basic.metrics_prof_start() → None

用于设置性能数据采集信号启动，和asc.metrics_prof_stop()配合使用。
使用msProf工具进行算子上板调优时，可在kernel侧代码段前后分别调用asc.metrics_prof_start()和asc.metrics_prof_stop()来指定需要调优的代码段范围。

**对应的Ascend C函数原型**

```c++
__aicore__ inline void MetricsProfStart()
```

**参数说明**

无。

**返回值说明**

无。

**调用示例**

```python
asc.metrics_prof_start()
```
