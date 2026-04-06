# asc.language.basic.data_cache_clean_and_invalid

### asc.language.basic.data_cache_clean_and_invalid(entire_type: CacheLine, dcci_dst: DcciDst, dst: [GlobalTensor](../core.md#asc.language.core.GlobalTensor)) → None

### asc.language.basic.data_cache_clean_and_invalid(entire_type: CacheLine, dst: [GlobalTensor](../core.md#asc.language.core.GlobalTensor)) → None

用来刷新Cache，保证Cache与Global Memory之间的数据一致性。

**对应的Ascend C函数原型**

```c++
template <typename T, CacheLine entireType, DcciDst dcciDst>
__aicore__ inline void DataCacheCleanAndInvalid(const GlobalTensor<T>& dst)
```

```c++
template <typename T, CacheLine entireType, DcciDst dcciDst>
__aicore__ inline void DataCacheCleanAndInvalid(const LocalTensor<T>& dst)
```

```c++
template <typename T, CacheLine entireType>
__aicore__ inline void DataCacheCleanAndInvalid(const GlobalTensor<T>& dst)
```

**参数说明**

- entire_type：指令操作模式，类型为CacheLine枚举值：
  - SINGLE_CACHE_LINE：只刷新传入地址所在的Cache Line（若非64B对齐，仅操作对齐范围内部分）。
  - ENTIRE_DATA_CACHE：刷新整个Data Cache（耗时较大，性能敏感场景慎用）。
- dcci_dst：指定Data Cache与哪种存储保持一致性，类型为DcciDst枚举类：
  - CACHELINE_ALL：与CACHELINE_OUT效果一致。
  - CACHELINE_UB：预留参数，暂未支持。
  - CACHELINE_OUT：保证Data Cache与Global Memory一致。
  - CACHELINE_ATOMIC：部分Atlas产品上为预留参数，暂未支持。
- dst：      需要刷新Cache的Tensor。

**调用示例**

- 支持通过配置dcciDst确保Data Cache与GM存储的一致性
  ```python
  asc.data_cache_clean_and_invalid(entire_type=asc.CacheLine.SINGLE_CACHE_LINE,
                                  dcci_dst=asc.DcciDst.CACHELINE_OUT, dst=dst)
  ```
- 不支持配置dcciDst，仅支持保证Data Cache与GM的一致性
  ```python
  asc.data_cache_clean_and_invalid(entire_type=asc.CacheLine.SINGLE_CACHE_LINE, dst=dst)
  ```
