# pyasc Supported Syntax Details

All syntax below is valid inside `@asc.jit`-decorated functions.

## Attribute access

```python
@asc.jit
def func():
    nums = [1, 2, 3]
    return nums.__len__()
```

## Loops

```python
@asc.jit
def func(n, total):
    for i in range(n):       # for with range()
        total += i
    while total < 100:        # while loop
        total += 1
    return total
```

## Operators

```python
@asc.jit
def func(a, b):
    result = a + b * 2       # binary: +, -, *, /, %, //, **, &, |, ^, <<, >>
    neg = -a                  # unary: -, ~, not
    return result
```

## Collections

```python
@asc.jit
def func(a, b, c):
    nums = [a, b, c]         # list
    pair = a, b               # tuple
    return nums[0] + pair[1]  # subscript access
```

## Conditionals

```python
@asc.jit
def func(x, y, z, ans, step):
    ans = x if x > 0 else 0  # ternary if
    if x + y == z:            # if/elif/else
        ans += step
    elif x + y > z:
        ans += step * 2
    else:
        ans -= 1
    return ans
```

## Assignments

```python
@asc.jit
def func(a, b):
    result = a + b            # simple assignment
    result += 1               # augmented: +=, -=, *=, /=, %=
    count: int = 3            # annotated assignment
    return result
```

## Boolean logic

```python
@asc.jit
def func(val, lo, hi, cnt, step):
    if val >= lo and val <= hi:
        cnt += step
    return cnt
```

## Constants and comparisons

```python
@asc.jit
def func():
    a = 1                     # int constant
    b = True                  # bool constant
    c = 3.14                  # float constant
    return a > 0              # comparison: >, <, >=, <=, ==, !=
```

## Assert with f-string

```python
@asc.jit
def func(num):
    assert num > 0, f"Expected positive, got {num}"
```

## Slicing

```python
@asc.jit
def func():
    nums = [1, 2, 3, 4, 5]
    return len(nums[2:])      # slice from index 2
```

## Function calls

```python
@asc.jit
def helper() -> int:
    return 42

@asc.jit
def kernel():
    x = helper()              # call another @asc.jit function
```

## Pass statement

```python
@asc.jit
def func():
    pass
```
