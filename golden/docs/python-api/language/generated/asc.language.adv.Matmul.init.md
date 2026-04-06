# asc.language.adv.Matmul.init

#### Matmul.init(tiling: TCubeTiling) → None

灵活的自定义Matmul模板参数配置。

**对应的Ascend C函数原型**

```c++
__aicore__ inline void Init(const TCubeTiling* __restrict cubeTiling, TPipe* tpipe = nullptr)
```

**参数说明**

- cube_tiling: Matmul Tiling参数.
- tpipe: Tpipe对象。

**调用示例**

```python
regist_matmul(&pipe, get_sys_workspace_ptr(), mm)
mm.init(&tiling)
```
