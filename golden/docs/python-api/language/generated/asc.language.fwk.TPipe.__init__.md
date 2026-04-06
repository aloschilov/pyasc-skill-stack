# asc.language.fwk.TPipe._\_init_\_

#### TPipe.\_\_init_\_() → None

#### TPipe.\_\_init_\_(handle: Value) → None

构造用来管理内存和同步的TPipe对象。

**对应的Ascend C函数原型**

```c++
__aicore__ inline TPipe()
```

**参数说明**

无。

**约束说明**

- 避免TPipe在对象内创建和初始化，TPipe在对象内创建时，可能会影响编译器对对象内常量的优化，引起scalar性能劣化，具体原理请参考避免TPipe在对象内创建和初始化。
- TPipe对象同一时刻全局只能存在一份，同时定义多个TPipe对象，会出现卡死等随机行为。如果需要使用多个TPipe时，请先调用destroy接口释放前一个TPipe。

**调用示例**

```python
pipe = asc.Tpipe()
```
