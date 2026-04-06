# asc.language.fwk.TPipe.destroy

#### TPipe.destroy() → None

释放资源。

**对应的Ascend C函数原型**

```c++
__aicore__ inline void Destroy()
```

**参数说明**

无。

**约束说明**

用于重复申请释放tpipe，创建tpipe对象后，可调用destroy手动释放资源。

**调用示例**

```python
pipe = asc.Tpipe()
que = asc.TQue(asc.TPosition.VECOUT, 2)
num = 2
len = 128
pipe.init_buffer(que=que, num=num, len=len)
pipe.destroy()
```
