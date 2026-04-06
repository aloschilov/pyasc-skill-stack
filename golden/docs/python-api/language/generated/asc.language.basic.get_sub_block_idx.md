# asc.language.basic.get_sub_block_idx

### asc.language.basic.get_sub_block_idx() → int

获取 AI Core 上 Vector 核的 ID。

**对应的 Ascend C 函数原型**

```c++
__aicore__ inline int64_t GetSubBlockIdx();
```

**参数说明**

无。

**返回值说明**

返回 Vector 核 ID。

**约束说明**

无。

**调用示例**

```python
import asc
sub_block_id = asc.get_sub_block_idx()
```
