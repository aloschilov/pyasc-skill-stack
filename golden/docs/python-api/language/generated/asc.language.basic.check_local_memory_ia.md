# asc.language.basic.check_local_memory_ia

### asc.language.basic.check_local_memory_ia(check_params: CheckLocalMemoryIAParam)

Check设定范围内的UB读写行为，如果有设定范围的读写行为则会出现EXCEPTION报错，无设定范围的读写行为则不会报错。

**对应的Ascend C函数原型**

```c++
__aicore__ inline void CheckLocalMemoryIA(const CheckLocalMemoryIAParam& checkParams)
```

**参数说明**

- check_params：用于配置对UB访问的检查行为，类型为CheckLocalMemoryIAParam。
  - enable_bit：配置的异常寄存器，取值范围：enable_bit∈[0,3]，默认为0。
    - 0：异常寄存器0。
    - 1：异常寄存器1。
    - 2：异常寄存器2。
    - 3：异常寄存器3。
  - start_addr：Check的起始地址，32B对齐，取值范围：start_addr∈[0, 65535]，默认值为0。比如，可通过LocalTensor.get_phy_addr()/32来获取start_addr。
  - end_addr：Check的结束地址，32B对齐，取值范围：end_addr∈[0, 65535]。默认值为0。
  - is_scalar_read：Check标量读访问。
    - false：不开启，默认为false。
    - true：开启。
  - is_scalar_write： Check标量写访问。
    - false：不开启，默认为false。
    - true：开启。
  - is_vector_read： Check矢量读访问。
    - false：不开启，默认为false。
    - true：开启。
  - is_vector_write： Check矢量写访问。
    - false：不开启，默认为false。
    - true：开启。
  - is_mte_read： Check MTE读访问。
    - false：不开启，默认为false。
    - true：开启。
  - is_mte_write： Check MTE写访问。
    - false：不开启，默认为false。
    - true：开启。
  - is_enable： 是否使能enable_bit参数配置的异常寄存器。
    - false：不使能，默认为false。
    - true：使能。

**约束说明**

- start_addr/end_addr的单位是32B，check的范围不包含start_addr，包含end_addr，即(start_addr, end_addr]。
- 每次调用完该接口需要进行复位（配置is_enable为False进行复位）。

**调用示例**

```python
params = asc.CheckLocalMemoryIAParam()
asc.check_local_memory_ia(params)
```
