# pyasc的python语法支持情况说明

## 支持的语法接口列表

- 属性访问语法

  - 示例

    ```python
    @asc.jit
    def func_visit_attribute():
        nums = [1, 2, 3]
        return nums.__len__()
    ```

- while循环

  - 示例

    ```python
    @asc.jit
    def func_visit_while(i, n, ans):
        while i < n:
            ans += i
            i += 1
        return ans
    ```
  
- for循环

  - 示例

    ```python
    @asc.jit
    def func_visit_for(num, total):
        for i in range(num):
          total += i
        return total
    ```

- 二元运算符

  - 示例

    ```python
    @asc.jit
    def func_visit_binop(num1, num2, num3):
        result = num1 + num2 * num3
        return result
    ```

- 一元运算符

  - 示例

    ```python
    @asc.jit
    def func_visit_unary_op(a):
        return a + ~1
    ```

- tuple元组

  - 示例

    ```python
    @asc.jit
    def func_visit_tuple(x, y, z):
        return x, y, z
    ```

- pass语句

  - 示例

    ```python
    @asc.jit
    def func_visit_pass():
        pass
    ```

- list列表

  - 示例

    ```python
    @asc.jit
    def func_visit_list(a, b, c):
        nums = [a, b, c]
        return nums
    ```

- 条件表达式

  - 示例

    ```python
    @asc.jit
    def func_visit_if_exp(x):
        ans = x if x > 0 else 0
        return ans
    ```

- 条件判断语句

  - 示例

    ```python
    @asc.jit
    def func_visit_if(x, y, z, ans, step):
        if x + y == z:
            ans += step
            ans = ans * 1
        elif x + y > z:
            ans += step * 2
        else:
            ans += step
            ans -= 1
        return ans == 1
    ```

- 常量

  - 示例

    ```python
    @asc.jit
    def func_visit_constant():
        a = 1
        b = True
        return a, b
    ```

- 比较表达式

  - 示例

    ```python
    @asc.jit
    def func_visit_compare(a, b, c, ans):
        if a > b:
            ans += 2
        if a + b < c:
            ans += 1
        if ans >= 10:
            ans += 3
        if ans <= 5:
            ans += 1
        if ans == c:
            ans += 5
        return ans
    ```
  
- 表达式语句/函数调用

  - 示例

    ```python
    @asc.jit
    def func():
        pass
    
    @asc.jit
    def func_visit_expr():
        func() #表达式/函数调用
    ```

- 布尔逻辑表达式

  - 示例

    ```python
    @asc.jit
    def func_visit_bool_op(value, min_threshold, max_threshold, cnt, step):
        if value >= min_threshold and value <= max_threshold:
            cnt += step
        elif value < min_threshold or value > max_threshold:
            cnt += step * 2
        return cnt == step
    ```

- 增强赋值语句

  - 示例

    ```python
    @asc.jit
    def func_visit_augassign(a, b):
        result = a
        result += b
        result *= 2
        result -= 5
        result /= a
        result %= 3
        return result
    ```

- 赋值语句

  - 示例

    ```python
    @asc.jit
    def func_visit_assign(num1, num2, num3) -> float:
        num3 = num1 + num2
        return num3
    ```

- 带类型注解的赋值语句

  - 示例

    ```python
    @asc.jit
    def func_visit_ann_assign(num1, num2, num3) -> float:
        count: int = 3
        sum_result: float = 1.0
        sum_result += num1
        sum_result += num2
        sum_result += num3
        result = sum_result / count
        return result
    ```
  
- 格式化占位符和assert语句

  - 示例

    ```python
    @asc.jit
    def func_visit_joined_and_formatted_and_assert(num):
        assert 1 < 0, f"assert failed {num}"
    ```

- 下标访问语法

  - 示例

    ```python
    @asc.jit
    def func_visit_subscript(a, b):
        nums = [a, b]
        return nums[0] + nums[1]
    ```

- 名称引用语句

  - 示例

    ```python
    @asc.jit
    def func_visit_subscript(a, b, c):
        nums = [a, b, c]
        return len(nums)
    ```

- 切片表达式

  - 示例

    ```python
    @asc.jit
    def func_visit_slice():
        nums = [1, 2, 3, 4, 5]
        ans = nums[2:]
        return len(ans)
    ```

## 不支持的语法接口列表

- Nested Functions（嵌套函数）

  - 示例：
    ```python
    @asc.jit
    def outer_function(x):
        def inner_function():
            return x * 2
        return inner_function()
    
    def outer_function(x):
        @asc.jit
        def inner_function():
            return x * 2
        return inner_function()
    
    @asc.jit
    def outer_function(x):
        @asc.jit
        def inner_function():
            return x * 2
        return inner_function()
    ```
  
- global语句
  
  - 示例：
  
    ```python
    count = 0
    
    @asc.jit
    def increase():
        global count
        count += 1
    ```
  
- return语句

  - 示例
  
    以下示例的kernel函数是@asc.jit修饰的核函数，func函数是@asc.jit修饰的Device侧执行函数。核函数和Device侧执行函数的定义请参考[pyasc模块与架构-编译运行模块](https://gitcode.com/cann/pyasc-dev/blob/master/docs/architecture_introduction.md#%E7%BC%96%E8%AF%91%E5%92%8C%E8%BF%90%E8%A1%8C%E6%A8%A1%E5%9D%97)和[《Ascend C编程指南》](https://www.hiascend.com/document/redirect/CANNCommunityOpdevAscendc)中的“核函数”章节。
 
    ```python
    # 不支持从核函数直接返回
    @asc.jit
    def kernel(x):
        return x * 2 # 不支持
    
    def launch():
        kernel[...](...)
    ```
    ```python
    # 支持核函数调用其他Device侧执行函数并获取其返回值
    @asc.jit
    def func() -> int: # Device侧执行函数
        return 2 # 支持
    
    @asc.jit
    def kernel(): # 核函数
        x = func() # 支持
    
    def launch():
        kernel[...](...)
    ```
  
  - 注意：return 作为顶层语句是支持的，但不能嵌套在if、for等结构中，且不能从核函数中返回对象，但是可以从其他Device侧执行函数中返回。
  
- continue语句
  
  - 示例
  
    ```python
    @asc.jit
    def sum(n):
        for i in range(n):
            if i % 2 == 0:
                continue
    ```

- print语句

  - 示例
  
    ```python
    @asc.jit
    def output(name, age):
        print(name + age)
    ```

- with open语句

  - 示例
  
    ```python
    @asc.jit
    def print_file():
        with open('file.txt', 'r') as f:
            ...
    ```
  
- raise语句
  
  - 示例
  
    ```python
    @asc.jit
    def check(a):
        if a < 0:
            raise ValueError("a must be a positive number")
    ```

- try-except语句

  - 示例
  
    ```python
    @asc.jit
    def calc(a):
        try:
            a = 10 / 0
        except ZeroDivisionError:
            pass
        return a
    ```
  
- yield语句
  
  - 示例
  
    ```python
    @asc.jit
    def yield_example():
        x = 1
        y = 10
        while x < y:
            yield x
            x += 1
        return x
    ```
  
- lambda语句
  
  - 示例
  
    ```python
    @asc.jit
    def lambda_example(x):
        f = lambda a : a * a
        return f(x)
    ```
  
- break语句
  
  - 示例
  
    ```python
    @asc.jit
    def find_first_even(numbers, first_even):
        for num in range(numbers):
            if num % 2 == 0:
                first_even = num
                break
        return first_even
    ```

- import语句

  - 示例
  
    ```python
    @asc.jit
    def kernel():
        import math
    ```
  
- from ... import语句

  - 示例

    ```python
    @asc.jit
    def kernel():
        from random import randint
    ```
  
- async def语句
  
  - 示例
  
    ```python
    @asc.jit
    async def async_task(delay):
        await asyncio.sleep(delay)
        return 1
    ```

- yield from语句
  
  - 示例
  
    ```python
    @asc.jit
    def generator1():
        yield "a"
        yield "b"
    
        
    @asc.jit    
    def generator2():
        yield 1
        yield 2
        yield from generator1()
    ```
  
- async with语句

  - 示例
  
    ```python
    @asc.jit
    async def main():
        async with AsyncFile() as f:
            await asyncio.sleep(1)
        return 0
    ```

- nonlocal语句
  
  - 示例
  
    ```python
    @asc.jit
    def outer_function():
        count = 0
    
        def inner_function():
            nonlocal count
            count += 1
    
        return inner_function()
    ```
  
- 被@asc.jit装饰的kernel函数封装到类中
  
  - 示例
  
    ```python
    class AddKernel:
    
        def __init__(self, xxx) -> None:
            ...
    
        @asc.jit
        def kernel(self, yyy)
            ...
    ```

- if定义域变量在if定义域外使用

  - 示例

    ```python
    @asc.jit
    def if_function(x):
        if x:
            y = 1
        z = y + 2   # y未定义
    ```

    ```python
    @asc.jit
    def if_function(x):
        y = None
        if x:
            y = 1
        else:
            y = [1, 2]
        z = y + 2   # y未定义
    ```
  注：由于`x`为变量，无法在编译期确定if语句结果。可通过使用`x: asc.ConsExpr[int]`解决该问题。

- 无符号整数类型进行比较运算

  - 示例

    ```python
    import numpy as np
    @asc.jit
    def cmpi_function(mask_low: np.uint64, mask_high: np.uint64):
        if mask_low > 0 and mask_high > 0:  # 数字0也为无符号整数类型
            mask = [mask_low, mask_high]
            asc.add(z, x, y, mask=mask, repeat_time=1, repeat_params=asc.BinaryRepeatParams(1, 1, 1, 8, 8, 8))

    mask_low = np.uint64(2**64-1)
    mask_high = np.uint64(2**64-1)
    ```
  注：mask作为一个特殊的uint64场景，由框架进行处理，在进行`if mask_low > 0 and mask_high > 0`逻辑判断时可以使用 `if mask_low and mask_high`，
  如果和非0数字进行比较则只能使用np.int64代替。
  上述代码使用np.int64后代码如下：
    ```python
    import numpy as np
    @asc.jit
    def cmpi_function(mask_low: np.int64, mask_high: np.int64):
        if mask_low != 0 and mask_high != 0:
            mask = [mask_low, mask_high]
            asc.add(z, x, y, mask=mask, repeat_time=1, repeat_params=asc.BinaryRepeatParams(1, 1, 1, 8, 8, 8))

    mask_low = np.int64(-1)    # int64(-1)在强转为uint64时等于2**64-1
    mask_high = np.int64(-1)
    ```