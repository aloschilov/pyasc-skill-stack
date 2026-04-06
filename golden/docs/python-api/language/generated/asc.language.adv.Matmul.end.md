# asc.language.adv.Matmul.end

#### Matmul.end() → None

多个Matmul对象之间切换计算时，必须调用一次End函数，用于释放Matmul计算资源，防止多个Matmul对象的计算资源冲突。

**对应的Ascend C函数原型**

```c++
__aicore__ inline void End()
```

**参数说明**

无。

**调用示例**

```python
mm.iterate_all(gm_c)
mm.end()
```
