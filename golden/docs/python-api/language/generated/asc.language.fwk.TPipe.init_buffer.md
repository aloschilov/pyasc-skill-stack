# asc.language.fwk.TPipe.init_buffer

#### TPipe.init_buffer(que: [TQue](../fwk.md#asc.language.fwk.TQue), num: int = 0, len: int = 0) → None

#### TPipe.init_buffer(buf: [TBuf](../fwk.md#asc.language.fwk.TBuf), num: int = 0) → None

用于为TQue等队列和TBuf分配内存。

**对应的Ascend C函数原型**

```c++
template <class T>
__aicore__ inline bool InitBuffer(T& que, uint8_t num, uint32_t len)
```

```c++
template <TPosition bufPos>
__aicore__ inline bool InitBuffer(TBuf<bufPos>& buf, uint32_t len)
```

**参数说明**

- que：需要分配内存的TQue等对象。
- num：分配内存块的个数。double buffer功能通过该参数开启：num设置为1，表示不开启double buffer；num设置为2，表示开启double buffer。
- len：每个内存块的大小，单位为字节。当传入的len不满足32字节对齐时，API内部会自动向上补齐至32字节对齐，后续的数据搬运过程会涉及非对齐处理，具体内容请参考非对齐场景。
- buf：需要分配内存的TBuf对象。
- len：为TBuf分配的内存大小，单位为字节。当传入的len不满足32字节对齐时，API内部会自动向上补齐至32字节对齐，后续的数据搬运过程会涉及非对齐处理，具体内容请参考非对齐场景。

**约束说明**

- init_buffer申请的内存会在TPipe对象销毁时通过析构函数自动释放，无需手动释放。
- 如果需要重新分配init_buffer申请的内存，可以调用reset，再调用init_buffer接口。
- 一个kernel中所有使用的Buffer数量之和不能超过64。

**调用示例**

```python
# 为TQue分配内存，分配内存块数为2，每块大小为128字节
pipe = asc.Tpipe()
que = asc.TQue(asc.TPosition.VECOUT, 2)
num = 2
len = 128
pipe.init_buffer(que=que, num=num, len=len)
# 为TBuf分配内存，分配长度为128字节
pipe = asc.Tpipe()
buf = asc.TBuf(asc.TPosition.A1)
len = 128
pipe.init_buffer(buf=buf, num=len)
```
