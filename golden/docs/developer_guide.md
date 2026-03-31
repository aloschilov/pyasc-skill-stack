# Ascend C Python编程接口开发指南
在浏览本文档前，建议您先根据开发需求浏览[模块与架构](architecture_introduction.md)说明文档。若新增一个Ascend C API的Python编程接口，涉及以下开发内容和交付件，开发内容细节请参考[各模块开发内容说明](#各模块开发内容说明)。
- 开发内容 
  - （必需）Python前端模块：新增对应Python接口代码；  
  - （必需）ASC-IR定义模块：新增对应OP节点定义；  
  - （非必需）AST转ASC-IR模块：实现新语法结构的支持（已支持的语法请参考[pyasc支持的语法接口列表](python_syntax_support.md#支持的语法接口列表)）；
  - （非必需）Ascend C代码生成模块：实现对应API的ASC-IR转Ascend C代码的功能。    
- 交付件
  - （必需）Python前端模块接口定义代码文件；
  - （必需）ASC-IR定义代码文件；
  - （必需）UT测试用例（包含Python前端的UT、ASC-IR的UT，详见各模块开发内容说明）；
  - （非必需）ST测试用例（单算子测试用例，参考tutorials目录下的样例）；
  - （必需）Python接口的资料，具体开发方法和要求可参考[API文档自动生成工具使用指南](./API_docstring_generation_tool_guide.md)。
  
所有新开发代码请遵守下述的[编码规范](#编码规范)章节列举的规范内容。

## 各模块开发内容说明

### Python前端模块

#### 前置说明

Python前端模块的主要功能是定义与[Ascend C API](https://hiascend.com/document/redirect/CannCommunityAscendCApi) 一一对应的Python编程接口。  
- 目录结构说明
  - 参考Ascend C API的接口分类，python侧的目录组织如下：

    ```python
    python
    ├── asc
    │   ├── ...
    │   ├── language
    │   │   ├── adv               # 高阶API
    │   │   ├── basic             # 基础API（除内存管理与同步控制部分）
    │   │   ├── core              # 核心数据结构和枚举，以及python侧新引入的语义
    │   │   └── fwk               # 内存管理与同步控制，包括TPipe、TQue等框架相关的类型和接口
    │   └── ...
    ├── ...
    └── ...
    ```

  - 每个目录中的接口均按照类别或功能分在相应的文件下。例如，在`basic`目录下，接口被细分为以下功能模块:

    - `common.py`：通用接口
    - `data_copy.py`：数据搬运接口
    - `vec_binary_scalar.py`：双目标量运算接口
    - `vec_binary.py`：双目矢量运算接口
    - `vec_unary.py`：单目矢量运算接口
    - ...

  - 根据接口的类别或功能，选择合适的文件进行添加。
    - 若现有文件无法满足需求，可考虑新增文件以保持文件的解耦和模块的独立性。  
    - 新增代码文件时，文件命名方式建议参考[Ascend C接口所在的代码头文件](https://gitcode.com/cann/asc-devkit/tree/master/impl/basic_api/dav_c220)的命名。
    - 接口名称建议保持前后端一致，且按字典序添加。

- 编码规范要求，具体可参考[Python前端模块编码规范](#python前端模块编码规范)。

#### 具体开发步骤

本部分将从新增Python函数接口、类、结构体、枚举类型四种开发场景展开介绍（若需要开发的接口仅涉及其中部分开发场景，可单独参考对应章节即可）。

##### 新增Python函数接口

完整实现请参考<a href="../python/asc/language/basic/vec_binary.py">vec_binary.py</a>。

- Step1：基于新增API的类别或功能，确定归属文件

  以Add API为例，属于双目矢量运算API，故对应接口需在`python\asc\language\basic\vec_binary.py`文件中进行定义和实现。

- Step2：完成Python接口定义

  结合[Python前端模块编码规范](#python前端模块编码规范)中描述的命名规范、接口重载规范、模板参数处理规则等内容，完成python接口定义。
  
  以Add API为例，python接口定义如下：
  ```python
  @overload
  def add(dst: LocalTensor, src0: LocalTensor, src1: LocalTensor, count: int, is_set_mask: bool = True) -> None:
      ...


  @overload
  def add(dst: LocalTensor, src0: LocalTensor, src1: LocalTensor, mask: int, repeat_times: int,
          repeat_params: BinaryRepeatParams, is_set_mask: bool = True) -> None:
      ...


  @overload
  def add(dst: LocalTensor, src0: LocalTensor, src1: LocalTensor, mask: List[int], repeat_times: int,
          repeat_params: BinaryRepeatParams, is_set_mask: bool = True) -> None:
      ...
  ```
  `@overload`修饰的函数用于提供类型和重载的提示信息，但本身不具备实际功能，需要列出所有重载函数方式。参数中避免出现`RuntimeInt`等用户不感知的运行时参数。

- Step3：完成Python接口具体实现
  
  ```python
  @require_jit
  @set_binary_docstring(cpp_name="Add", append_text="按元素求和。")
  def add(dst: LocalTensor, src0: LocalTensor, src1: LocalTensor, *args, **kwargs) -> None:
      builder = global_builder.get_ir_builder()
      op_impl("add", dst, src0, src1, args, kwargs, builder.create_asc_AddL0Op, builder.create_asc_AddL1Op,
              builder.create_asc_AddL2Op)
  ```
  - 使用`@require_jit`标记函数，表示需要JIT编译。
  - 使用`@set_binary_docstring`标记函数，用于自动生成相应的API接口文档。
  - 在具体实现中，首先获取IR构建器（`builder = global_builder.get_ir_builder()`），再调用`op_impl`函数传递相应的参数和IR构建方法。

    ```python
    def op_impl(callee: str, dst: LocalTensor, src0: LocalTensor, src1: LocalTensor, args: Tuple[Any],
                kwargs: Dict[str, Any], build_l0: Callable, build_l1: Callable, build_l2: Callable) -> None:
        builder = build_l0.__self__
        if not isinstance(builder, ir.Builder):
            raise TypeError("Input builder must be ir.Builder")
        dispatcher = OverloadDispatcher(callee)

        check_type(callee, dst, src0, src1)

        @dispatcher.register(mask=RuntimeInt, repeat_times=RuntimeInt, repeat_params=BinaryRepeatParams, 
                            is_set_mask=DefaultValued(bool, True))
        def _(mask: RuntimeInt, repeat_times: RuntimeInt, repeat_params: BinaryRepeatParams, is_set_mask: bool = True):
            build_l0(dst.to_ir(), src0.to_ir(), src1.to_ir(),
                    _mat(mask, KT.uint64).to_ir(), _mat(repeat_times, KT.int8).to_ir(), 
                    repeat_params.to_ir(), is_set_mask)

        @dispatcher.register(mask=list, repeat_times=RuntimeInt, repeat_params=BinaryRepeatParams, 
                            is_set_mask=DefaultValued(bool, True))
        def _(mask: list, repeat_times: RuntimeInt, repeat_params: BinaryRepeatParams, is_set_mask: bool = True):
            mask = [_mat(v, KT.uint64).to_ir() for v in mask]
            build_l1(dst.to_ir(), src0.to_ir(), src1.to_ir(), mask, _mat(repeat_times, KT.int8).to_ir(), 
                    repeat_params.to_ir(), is_set_mask)

        @dispatcher.register(count=RuntimeInt, is_set_mask=DefaultValued(bool, True))
        def _(count: RuntimeInt, is_set_mask: bool = True):
            build_l2(dst.to_ir(), src0.to_ir(), src1.to_ir(), _mat(count, KT.int32).to_ir())

        dispatcher(*args, **kwargs)
    ```
    - 因为Add API涉及不同入参的重载接口，这里通过使用`OverloadDispatcher`自动识别和处理不同的参数组合。
    - 通过`@dispatcher.register(paramlist)`，调用传入的IR构建方法`build_l0`、`build_l1`等完成不同重载函数接口的实现。

- Step4：添加引用

  以Add API为例，需要在`python\asc\language\basic\__init__.py`以及`python\asc\language\__init__.py`中添加引用。

##### （可选）新增类

以Add API为例，对应的Ascend C函数接口参数包含`LocalTensor`类，`LocalTensor`类的完整实现请参考<a href="../python/asc/language/core/tensor.py">tensor.py</a>。

- Step1：确定新增类所属文件

  `LocalTensor`属于核心数据结构，故需在`python\asc\language\core\tensor.py`文件中实现类的定义。

- Step2：实现新增类的初始化方法和其他类方法

  - `LocalTensor`包含多种构造函数原型，需要实现重载的`__init__`。
    ```python
    @overload
    def __init__(self) -> None:
        ...

    @overload
    def __init__(self, handle: IRHandle, dtype: DataType, shape: TensorShape) -> None:
        ...

    @overload
    def __init__(self, dtype: DataType, addr: int = 0) -> None:
        ...

    @overload
    def __init__(self, dtype: DataType, pos: Optional[TPosition] = TPosition.VECIN, \
        addr: int = 0, tile_size: int = 0):
        ...

    def __init__(self, *args, **kwargs) -> None:
        """This contructor should not be called by user"""
        dispatcher = OverloadDispatcher(__name__)

        ... # 此处省略具体代码实现
    ```
  - `LocalTensor`的类方法定义和实现，可参考上述的[新增Python函数接口](#新增Python函数接口)。

- Step3：实现`to_ir`和`from_ir`方法

  为支持IR的生成，需要实现`to_ir`和`from_ir`方法。`LocalTensor`的对应方法实现如下：
  ```python
  @classmethod
  def from_ir(cls, handle: IRHandle) -> LocalTensor:
      ir_type = handle.get_type()
      return cls(handle, DataType.from_ir(ir.get_element_type(ir_type)), ir.get_shape(ir_type))

  def to_ir(self) -> IRHandle:
      return self.handle
  ```

- Step4：添加引用

  以`LocalTensor`为例，需要在`python\asc\language\core\__init__.py`以及`python\asc\language\__init__.py`中添加引用。

##### （可选）新增结构体

Add API提供了切分计算方式，需要`BinaryRepeatParams`结构体用于控制操作数地址步长。`BinaryRepeatParams`结构体的完整实现请参考 <a href="../python/asc/language/core/types.py">types.py</a>。

- Step1：确定新增结构体所属文件

  基础API相关的结构体定义在`python\asc\language\core\types.py`下，高阶API相关的结构体定义在`python\asc\language\adv\types.py`。

- Step2：实现新增结构体的初始化方法
  ```python
  class BinaryRepeatParams(IRValue):

    @overload
    def __init__(self, dst_blk_stride: int = 1, src0_blk_stride: int = 1, src1_blk_stride: int = 1,
                 dst_rep_stride: int = 8, src0_rep_stride: int = 8, src1_rep_stride: int = 8) -> None:
        ...

    @overload
    def __init__(self, handle: IRHandle) -> None:
        """This contructor should not be called by user"""
        ...

    @require_jit
    def __init__(self, dst_blk_stride: RuntimeInt = 1, src0_blk_stride: RuntimeInt = 1, src1_blk_stride: RuntimeInt = 1,
                 dst_rep_stride: RuntimeInt = 8, src0_rep_stride: RuntimeInt = 8, src1_rep_stride: RuntimeInt = 8,
                 handle: Optional[IRHandle] = None) -> None:
        if handle is not None:
            self.handle = handle
            return
        builder = global_builder.get_ir_builder()
        ... # 此处省略具体代码实现
  ```

- Step3：实现`to_ir`和`from_ir`方法

  ```python
  @classmethod
  def from_ir(cls, handle: IRHandle) -> BinaryRepeatParams:
      return cls(handle=handle)

  def to_ir(self) -> IRHandle:
      return self.handle
  ```

- Step4：添加引用

  以`BinaryRepeatParams`为例，需要在`python\asc\language\core\__init__.py`以及`python\asc\language\__init__.py`中增加引用。


##### （可选）新增枚举类型

Add API的函数接口参数包含`LocalTensor`，需要使用`TPosition`枚举类型来确定`LocalTensor`的存储位置。`TPosition`枚举类型的完整实现请参考<a href="../python/asc/language/core/enums.py">enums.py</a>。

- Step1：确定新增枚举类型所属文件

  枚举类型定义统一在`python\asc\language\core\enums.py`下。

- Step2：实现新增枚举类型

  继承`IntEnum`, 根据Ascend C定义给枚举值赋整数值。
  ```python
  class TPosition(IntEnum):
    GM = 0
    A1 = 1
    A2 = 2
    B1 = 3
    B2 = 4
    C1 = 5
    C2 = 6
    CO1 = 7
    CO2 = 8
    VECIN = 9
    VECOUT = 10
    VECCALC = 11
  ```

- Step3：添加引用

  以`TPosition`为例，需要`在python\asc\language\core\__init__.py`以及`python\asc\language\__init__.py`中增加引用。

##### 开发UT测试用例

在完成Python前端模块功能开发后，可以添加相应的UT测试，验证功能的正确性。
本部分通过目录结构、UT测试实现框架、UT测试代码示例三部分内容，说明Python前端UT测试的开发方式。

- 目录结构

  与Python前端接口目录结构相同，UT测试目录结构如下：

  ```python
  python
  ├── ...
  ├── test
  │   ├── ...
  │   └── unit
  │       ├── ...
  │       ├── language
  │       │   ├── adv                 # 高阶API的UT测试
  │       │   ├── basic               # 基础API的UT测试
  │       │   ├── core                # 核心数据结构和枚举的UT测试
  │       │   └── fwk                 # 内存管理与同步控制的UT测试
  │       └── ...
  └── ...
  ```

  按照不同的类别或功能，在每个目录下对应的测试文件中，完成Python前端的UT测试用例开发。例如在`fwk`目录下，测试被细分为以下模块：

  - `test_tbuf.py`：TBuf类功能测试
  - `test_tbuf_pool.py`：TBufPool类功能测试
  - `test_tpipe.py`：TPipe类功能测试
  - `test_tque.py`：TQue类功能测试
  - `test_tque_bind.py`：TQueBind类功能测试
  - ...

  根据待添加UT测试的接口类别，选择合适的文件或者在合适目录下新增文件，完成UT测试用例开发。

- UT测试实现框架

  本模块基于`pytest`框架构建，整体实现过程如下：
  - 初始化环境，注入`mock_launcher_run`桩函数；
  - 定义内核函数，调用被测对象及方法；
  - 触发内核运行；
  - 通过断言验证编译以及执行流程是否按照预期触发。

- UT测试代码示例

  这里以实现add的UT测试为例展开介绍。

  add的Python接口函数原型如下：

  ```python
  @overload
  def add(dst: LocalTensor, src0: LocalTensor, src1: LocalTensor, count: int) -> None:
      ...

  @overload
  def add(dst: LocalTensor, src0: LocalTensor, src1: LocalTensor, mask: AnyMask, repeat_times: int,
          repeat_params: BinaryRepeatParams) -> None:
      ...
  ```

    - 选择合适的文件进行开发
    
      如add属于双目矢量指令，则在`test_vector_binary.py`文件中编写UT测试代码。
      
    - 编写测试函数

      ```python
      def test_add_kernel(mock_launcher_run):

          @asc.jit
          def add_kernel():
              ... # 此处省略具体实现
              asc.add(...)

          add_kernel[1]()
          assert mock_launcher_run.call_count == 1
        
      ```

    - 执行测试
    
      完成UT测试编码后，以双目矢量指令为例，在项目根目录执行如下命令进行UT单元测试。运行未报错说明UT测试通过。

      ```bash
      pytest ./python/test/unit/language/basic/test_vector_binary.py
      ```

### AST转ASC-IR模块

#### 前置说明
一般情况下，本模块不涉及新增代码开发。以下两类场景除外：

- 若在AST转ASC-IR的过程中，AST中的节点语法结构暂未支持（详见[pyasc支持的语法接口列表](python_syntax_support.md#支持的语法接口列表)），则涉及新增代码开发：需要在AST语法树遍历类内新增节点处理接口。
- 若IR创建接口无法支持pybind自动绑定，需要手动添加，则涉及手动添加代码开发。

#### 具体开发步骤

##### 新增语法处理接口

- 代码文件：`python\asc\codegen\function_visitor.py`。

- 代码开发示例：

  - 以支持双目运算符表达式（如 *、+、-）为例，定义形如visit_Xxxxx的函数，根据运算逻辑处理对应节点。

    ```python
    def visit_BinOp(self, node: ast.BinOp) -> Any:
        lhs = self.visit(node.left)
        rhs = self.visit(node.right)
        method_name = self.get_binary_method_name(type(node.op))
        return self.apply_binary_method(method_name, lhs, rhs)
    ```
##### 添加IR创建接口

优先通过[TableGen](https://llvm.org/docs/TableGen/)工具自动生成pybind绑定代码，若自动生成代码无法满足需求，则需要在[pybind11绑定定义](../python/src/OpBuilder.cpp)中手动添加对应IR创建接口。

- 暂不支持pybind自动绑定，需要手动添加的场景（需要结合ASC-IR定义模块一起分析）

  - 复杂的API Type类型（例如包含模板参数的AscendC_Queue）
    ```
    def AscendC_Queue : AscendC_BaseQueueType<"Queue", "queue"> {
      let description = "Represents AscendC::TQue";
      let parameters = (ins "TPositionAttr":$tPositionAttr, "int64_t":$depth);
      let assemblyFormat = [{
        `<` custom<PrettyTPosition>($tPositionAttr) `,` $depth `>`
      }];
      let builders = [
        TypeBuilder<(ins "TPosition":$position, "int64_t":$depth), [{
          return $_get($_ctxt, TPositionAttr::get($_ctxt, position), depth);
        }]>,
      ];
      let extraClassDeclaration = [{
        TPosition getPosition() { return getTPositionAttr().getValue(); }
      }];
    }
    ```
  - 继承AscendC_BaseTensorType的Type类型（例如AscendC_GlobalTensor），AscendC_BaseTensorType的定义如下：
    ```
    class AscendC_BaseTensorType<string name, string typeMnemonic, list<Trait> traits = []>
        : AscendC_Type<name, typeMnemonic, [AscendC_BaseTensorTypeInterface,
            DeclareTypeInterfaceMethods<ShapedTypeInterface>] # traits> {
      let parameters = (ins ArrayRefParameter<"int64_t">:$shape,
                            "Type":$elementType);

      let builders = [
        TypeBuilderWithInferredContext<(ins
          "ArrayRef<int64_t>":$shape, "Type":$elementType)>,
        TypeBuilderWithInferredContext<(ins "Type":$elementType)>,
        TypeBuilderWithInferredContext<(ins
          "::mlir::ascendc::BaseTensorType":$baseType)>,
      ];

      let hasCustomAssemblyFormat = 1;
      let skipDefaultBuilders = 1;
    }

    def AscendC_GlobalTensor : AscendC_BaseTensorType<"GlobalTensor", "global_tensor"> {
      let summary = "Global tensor from GM (OUT)";
    }
    ```
  - 包含枚举类型参数的Op（例如AscendC_FftsCrossCoreSyncOp，包含AscendC_PipeAttr枚举类型）
    ```
    def AscendC_FftsCrossCoreSyncOp : APIOp<"ffts_cross_core_sync", "FftsCrossCoreSync"> {
      let arguments = (ins AscendC_PipeAttr:$pipe, AnyType:$config);
    }
    ```

- 代码文件：

  - 自动生成：`lib\TableGen\GenPybindDefs.cpp`。
  - 手动添加：`python\src\OpBuilder.cpp`文件内的`pyasc_init_ir_builder`函数。

- 代码开发示例：

  - OpBuilder.cpp 手动添加TPipe初始化IR创建接口。

    ```python
    .def("create_asc_PipeOp", [](PyOpBuilder &self) -> Value {
    		    return self.create<ascendc::PipeOp>();
      })
    ```

### ASC-IR定义模块

#### 前置说明

ASC-IR是基于[MLIR](https://mlir.llvm.org/)定义的Dialect（方言）。由于ASC-IR基于LLVM中的TableGen工具翻译为MLIR，所以ASC-IR基于[TableGen语法](https://llvm.org/docs/TableGen/)编写。

注：ASC-IR定义模块的全部文件均位于`代码仓根目录/include/ascir/Dialect/Asc/IR`。

##### AscendC_Op类定义介绍

本模块主要基于[APIOp类](#APIOp类定义介绍)开发ASC-IR，APIOp类继承自位于[Base.td](../include/ascir/Dialect/Asc/IR/Base.td)的AscendC_Op类，AscendC_Op类的定义如下。
```
class AscendC_Op<string mnemonic, list<Trait> traits = []>
    : Op<AscendC_Dialect, mnemonic, traits> {
  let cppNamespace = "::mlir::ascendc";
  let assemblyFormat = "operands attr-dict `:` qualified(type(operands))";
  bit genEmitter = !foldl(
    0, traits, init, trait,
    !or(init,
        !eq(trait, AscConstructor),
        !eq(trait, AscMemberFunc),
        !eq(trait, AscFunc)
    )
  );
  list<int> paramTypeLists = [];
}
```
- `cppNamespace`：是用户自定义开发IR的命名空间，默认使用 `"::mlir::ascendc"`。
- `assemblyFormat`：`operands`表示操作数列表，`qualified(type(operands))`表示操作数类型。
- `genEmitter`：是将ASC-IR节点自动化生成Ascend C代码的开关（不需要设置，是基于以下任意trait自动生成的），其中：
  - `AscConstructor`表示构造函数。
  - `AscMemberFunc`表示成员函数。
  - `AscFunc`表示普通函数。

    如果traits列表中出现上述任一函数，则会自动生成对应的Ascend C代码。  
    为了自动化处理，我们规定：能直接基于参数推导的类型模板参数不作为ASC-IR Operation类的参数，普通参数直接作为函数参数传参。对于特殊的参数需要通过`paramTypeLists`标记参数类型用于辅助自动代码生成。  
- `paramTypeLists`：是一个整数列表，定义了各个`operands`对应的参数类型。  
  该参数类型列表的顺序遵循参数处理规则顺序：*运行时必选参数*， *模板必选参数*， *运行时可选参数*， *模板可选参数*，同类型参数遵循先后出现的顺序。  
  其中数值大于0表示是模板参数，具体字段的含义如下：
  - 数值-3，表示函数参数是指针类型。
  - 数值-2，表示函数参数需要将指针转为int类型。
  - 数值-1，表示函数参数是枚举类型参数。
  - 数值0，表示普通的函数参数。
  - 数值1，从该参数中提取模板类型（例如：`<typename T> from function(T arg)`，从arg参数提取模板参数T）。
  - 数值2，从模板类型中提取元素类型(例如：`LocalTensor<T> -> T`)。
  - 数值3，非类型模板参数：枚举值`enum value`。
  - 数值4，非类型模板参数：常规值`regular value`。
  - 数值5，类型模板参数（例如：`<typename T> function()`）。

注：
- 大部分API接口场景，当前支持Ascend C代码自动生成机制，即通过ASC-IR节点定义可实现Ascend C代码自动生成，无需适配修改Ascend C代码生成模块代码。

  Ascend C代码自动生成机制的由来：为支持各类Ascend C API的python编程接口，会存在大量的Ascend C代码生成工作。其主要内容是将ASC-IR节点翻译生成对应的Ascend C代码。这类代码生成工作存在大量的相似性，容易存在霰弹式修改，且在代码生成过程中会存在一些硬编码处理。故设计推出了Ascend C代码自动生成机制（对应上述的`genEmitter`）。

- 暂不支持Ascend C代码自动生成的API接口场景
  - API接口入参包含数组类型

    例如Add的L1级接口（tensor高维切分计算）mask逐bit模式，这里的mask参数是一个数组类型参数。
    ```cpp
    template <typename T, bool isSetMask = true>
    __aicore__ inline void Add(const LocalTensor<T>& dst, const LocalTensor<T>& src0, const LocalTensor<T>& src1, uint64_t mask[], const uint8_t repeatTime, const BinaryRepeatParams& repeatParams)
    ```

##### APIOp类定义介绍

APIOp类是Operation类模板，继承于AscendC_Op。用于开发Ascend C API所对应的ASC-IR，具体定义可参考[Base.td](../include/ascir/Dialect/Asc/IR/Base.td)。
```
class APIOp<string mnemonic, string apiName, list<Trait> traits = []>
    : AscendC_Op<mnemonic, [APIOpInterface] # traits> {
  let summary = "Call `AscendC::" # apiName # "` function";
  string comment = "";
  code extraClassDeclarationBase = [{
    static StringRef getAPIName() { return "}] # apiName # [{"; }
    static StringRef getComment() { return "}] # comment # [{"; }
  }];
  let extraClassDeclaration = extraClassDeclarationBase;
}
```
- `extraClassDeclarationBase`：是一个C++代码模板，用于生成静态方法`getAPIName()`和`getComment()`，分别返回API名称和注释。
- `extraClassDeclaration`：将`extraClassDeclarationBase`赋值，以便在生成的类中包含这些方法。

这里以继承于APIOp的`AscendC_GlobalTensorGetPhyAddrOp`为示例，来进一步介绍：

```
def AscendC_GlobalTensorGetPhyAddrOp : APIOp<"global_tensor.get_phy_addr", "GetPhyAddr", [AscMemberFunc]> {
  let summary = "Call `AscendC::GlobalTensor::GetPhyAddr` method";
  let arguments = (ins AscendC_GlobalTensor:$tensor, Optional<UI64>:$offset);
  let results = (outs AnyRankedOrUnrankedMemRef:$result);
  let assemblyFormat = [{
    $tensor attr-dict (`,` $offset^)? `:` qualified(type($tensor)) 
    `,` qualified(type($result)) (`,` type($offset)^)?
  }];
}
```

- ASC-IR中的Operation类命名规则：AscendC + 下划线 + Ascend C类名 + 成员函数名，示例中为`AscendC + _ + GlobalTensor + GetPhyAddr + Op`。
- 根据APIOp类的模板，在AscendC_GlobalTensorGetPhyAddrOp中：
  - `mnemonic`（必填）：这里是`global_tensor.get_phy_addr`，规则为："Ascend C类名 + . + 成员函数"。
  - `apiName`（必填）：这里是`GetPhyAddr`，建议复用Ascend C API的命名。
  - `traits`（选填）：这里是AscMemberFunc，需要填入MLIR的[traits特征](https://mlir.llvm.org/docs/Traits/)，可以为空，如果存在多个可以用","分割。

    几个常用的自定义traits如下：
    - `AscConstructor`：标记构造函数，用于Ascend C代码自动生成用途。
    - `AscMemberFunc`：标记成员函数，用于Ascend C代码自动生成用途。
    - `AscFunc`：标记普通函数，用于Ascend C代码自动生成用途。

    常用的MLIR内置traits如下：
    - `AttrSizedOperandSegments`：标记支持多个`Optional`可选入参。
  
  - `summary`（必填）：用于存储用户自定义的注释，方便调试。
  - `arguments`（必填）：入参定义，可选参数用`Optional/OptionalAttr<ParamType>`表示，`AnyType`表示根据入参自动推导的变量类型。
    由于是1:1映射Ascend C API，且为了自动化映射代码，arguments需要遵守以下映射规则：
    - 纯类型模板参数，如果无法基于已有参数进行推导，则需要表达成ASC-IR Operation类的参数；如果能直接基于已有参数推导，则不作为ASC-IR Operation类的参数。
    - 参数顺序规则：*运行时必选参数*， *模板必选参数*， *运行时可选参数*， *模板可选参数*。
    - 参数名建议和Ascend C API的保持一致，遵循小驼峰风格。
  - ` results`（选填）：表示出参定义，不支持可选参数。如果该Op无返回值则无需填写。
  - ` assemblyFormat`（选填）：表示匹配格式，代码格式为"变量名:推导类型"，其中``` (`,` $xxx^)? ``` 表示与optional对应的可选参数。

##### Type类型定义介绍

如果新增API涉及`LocalTensor/TQue/FixpipeParams`等类或结构体类型的定义，可以在[Types.td](../include/ascir/Dialect/Asc/IR/Core/Types.td)进行定义。

- 简单的Type类型定义

  对于类似`AscendC_Mask`的简单Type类型，可以直接继承`AscendC_Type`定义如下。`AscendC_Type`的具体定义可参考[AscendC_Type定义](../include/ascir/Dialect/Asc/IR/Core/Base.td)。
  ```
  def AscendC_Mask : AscendC_Type<"Mask", "mask"> {
    let summary = "Represents vector mask (bit mode)";
  }
  ```
  - `summary`（必填）：用于存储用户自定义的注释，方便调试。

- 复杂的Type类型定义

  对于类似`AscendC_Queue`的复杂Type类型，建议继承具体的API Type进行定义。这里的`AscendC_BaseQueueType`的具体定义可参考[AscendC_BaseQueueType定义](../include/ascir/Dialect/Asc/IR/Core/Base.td)。
  ```
  def AscendC_Queue : AscendC_BaseQueueType<"Queue", "queue"> {
    let description = "Represents AscendC::TQue";
    let parameters = (ins "TPositionAttr":$tPositionAttr, "int64_t":$depth);
    let assemblyFormat = [{
      `<` custom<PrettyTPosition>($tPositionAttr) `,` $depth `>`
    }];
    let builders = [
      TypeBuilder<(ins "TPosition":$position, "int64_t":$depth), [{
        return $_get($_ctxt, TPositionAttr::get($_ctxt, position), depth);
      }]>,
    ];
    let extraClassDeclaration = [{
      TPosition getPosition() { return getTPositionAttr().getValue(); }
    }];
  }
  ```
  - `description`（必填）：该Type类型的描述信息。
  - `parameters`（选填）：参数列表定义。
  - `assemblyFormat`（选填）：声明指令格式。
  - `builders`（选填）：构建器，用于创建AscendC_Queue的实例。
  - `extraClassDeclaration`（选填）：额外的类方法。

##### Attributes属性定义介绍

如果新增API涉及`Layout/Format/Event`等枚举类型属性的定义，可以在[Attributes.td](../include/ascir/Dialect/Asc/IR/Core/Attributes.td)进行定义。

- Attributes属性定义示例

  这里以继承于MLIR I32EnumAttr的`AscendC_CubeFormatAttr`为示例，来进一步介绍：
  ```
  def AscendC_CubeFormatAttr : I32EnumAttr<"CubeFormat", "matmul tensor format", [
    I32EnumAttrCase<"ND", 0, "nd">,
    I32EnumAttrCase<"NZ", 1, "nz">,
    I32EnumAttrCase<"ZN", 2, "zn">,
    I32EnumAttrCase<"ZZ", 3, "zz">,
    I32EnumAttrCase<"NN", 4, "nn">,
    I32EnumAttrCase<"ND_ALIGN", 5, "nd_align">,
    I32EnumAttrCase<"SCALAR", 6, "scalar">,
    I32EnumAttrCase<"VECTOR", 7, "vector">,
  ]> {
    let cppNamespace = "::mlir::ascendc";
    let description = "Represents AscendC::CubeFormat";
    let underlyingType = "uint8_t";
  }
  ```
- `cppNamespace`（必填）：是用户自定义开发IR的命名空间，默认使用 `"::mlir::ascendc"`。
- `description`（选填）：该Attributes属性的描述信息。
- `underlyingType`（选填）：uint8表征这些枚举值在底层使用8位无符合整数存储。

注：枚举类型的名称及枚举的各个取值，要求与Ascend C的保持一致。

#### 具体开发步骤

- Step1：确认新增API的Ascend C函数原型

  主要确认信息：是否涉及新增Type类型、Attributes属性；是否符合已有API模板。

- Step2：如果涉及新增Type类型，可参考上面前置说明的[Type类型定义介绍](#Type类型定义介绍)，完成相关Type类型定义

- Step3：如果涉及新增Attributes属性，可参考上面前置说明的[Attributes属性定义介绍](#Attributes属性定义介绍)，完成相关Attributes属性定义

- Step4：基于Step1确认的信息，完成对应API的IR定义

  Ascend C同一类API的函数原型和参数具有高度的相似性，考虑到开发代码的可复用性和易维护性，项目中已提供了部分模板。这些模板在[Base.td](../include/ascir/Dialect/Asc/IR/Base.td)中。对于符合模板的API，开发者可以直接基于模板开发。对于当前不符合模板的API，开发者可以开发新的模板或自行定制化开发。

  - 符合模板的API

    这里以Add API为例，展开介绍符合模板的API如何开发。
    
    Add属于基础API中双目矢量类计算API，Ascend C API函数原型包含以下几类：
      - L3级，即dst = src1 + src2形式；
      - L2级，即针对源操作数的连续count个数据进行计算，连续写入目的操作数；
      - L1级，即支持对每个操作数的mask，repeatTimes等的操作，逐比特模式，其中mask是数组；
      - L0级，即支持对每个操作数的mask，repeatTimes等的操作，连续模式，其中mask是单个数值。

    在API模板定义文件[Base.td](../include/ascir/Dialect/Asc/IR/Base.td)中，有双目矢量计算模板定义`BinaryOp`。通过继承`BinaryOp`又定义了`BinaryTemplateL0Op`、`BinaryTemplateL1Op`、`BinaryL2Op`、`BinaryL3Op`。

    Ascend C API中的Add函数有L0、L1、L2、L3的用法，函数原型和参数能匹配以上模板定义，可进一步定义`BinaryTemplateL0123Op`如下：
    ```
    multiclass BinaryTemplateL012Op<string baseMnemonic, string apiName, list<Trait> traits = []> {
      def L0Op : BinaryTemplateL0Op<baseMnemonic # "_l0", apiName, traits>;
      def L1Op : BinaryTemplateL1Op<baseMnemonic # "_l1", apiName, traits>;
      def L2Op : BinaryL2Op<baseMnemonic # "_l2", apiName, traits>;
    }

    class BinaryL3Op<string mnemonic, string apiName, list<Trait> traits = []>
        : BinaryOp<mnemonic, apiName, [BinaryL3OpInterface] # traits> {
      let summary = "Call `LocalTensor::" # apiName # "` method";
      let description = "`LocalTensor::" # apiName # "` performs a vector binary operation (L3 API).\n";
      let arguments = (ins AnyType:$dst, AnyType:$src0, AnyType:$src1);
    }

    multiclass BinaryTemplateL0123Op<string baseMnemonic, string apiName, string l3operator,
                        list<Trait> traits = []> {
      defm "" : BinaryTemplateL012Op<baseMnemonic, apiName, traits>;
      def L3Op : BinaryL3Op<baseMnemonic # "_l3", l3operator, traits>;
    }
    ```

    最终ASC-IR定义如下：
    ```
    defm Add : BinaryTemplateL0123Op<"add", "Add", "operator+">;
    ```

  - 不符合模板的API

    这里以TPipe类的InitBuffer为例，对应的Ascend C函数原型：
    ```cpp
    template <class T>
    __aicore__ inline bool InitBuffer(T& que, uint8_t num, uint32_t len)
    ```
    
    参考上面前置说明的[APIOp类定义介绍](#APIOp类定义介绍)，按照参数格式和顺序，完成定义如下：
    ```
    def AscendC_TPipeInitBufferOp : APIOp<"pipe.init_buffer", "InitBuffer", [AscMemberFunc]> {
      let summary = "Initialize raw buffer";
      let arguments = (ins AscendC_Pipe:$pipe, AscendC_TBuf:$buffer,
                          AnyType:$length);
      let assemblyFormat = [{
        $pipe `,` $buffer `,` $length attr-dict `:` qualified(type($buffer)) `,`
        type($length)
      }];
    }
    ```

- Step5：开发新增API的ASC-IR UT用例

  参考[开发ASC-IR UT测试用例](#开发ASC-IR的UT测试用例)。

#### 开发ASC-IR的UT测试用例

- 目录结构
  ```
  pyasc根目录
  ├── ...
  ├── python
  │   └── ...
  ├── test
  │   ├── ...
  │   ├── Target
  │   │   └── AscendC               # ASC-IR翻译生成Ascend C代码的UT测试目录
  │   │       ├── ...               # 非基础API的UT测试用例文件
  │   │       └── basic             # 基础API的UT测试文件目录
  │   └── ...
  └── ...
  ```
  可以根据新增的API类别，在已有UT测试文件中追加测试用例，或新增新的UT测试用例文件。新增文件命名方式建议参考[Ascend C接口所在的代码头文件](https://gitcode.com/cann/asc-devkit/tree/master/impl/basic_api/dav_c220)的命名。

- UT测试使用框架

  采用MLIR的lit（LLVM Integrated Tester）测试框架，具体可参考[MLIR Testing Guide](https://mlir.llvm.org/getting_started/TestingGuide/)。

- UT测试代码示例

  以Ascend C的Add API L2级接口为例，对应的Ascend C函数原型：
  ```cpp
  template <typename T>
  __aicore__ inline void Add(const LocalTensor<T>& dst, const LocalTensor<T>& src0, const LocalTensor<T>& src1, const int32_t& count)
  ```

  对应的ASC-IR转Ascend C代码UT测试用例实现如下：
  ```
  // CHECK-LABEL:void emit_vector_binary_l2_ops(AscendC::LocalTensor<float> v1, AscendC::LocalTensor<float> v2, AscendC::LocalTensor<float> v3, int32_t v4) {
  // CHECK-NEXT:   AscendC::Add(v1, v2, v3, v4);
  // CHECK-NEXT:   return;
  // CHECK-NEXT: }
  func.func @emit_vector_binary_l2_ops(%dst: !ascendc.local_tensor<1024xf32>, %src0: !ascendc.local_tensor<1024xf32>, %src1 : !ascendc.local_tensor<1024xf32>, %calCount_i32 : i32) {
    ascendc.add_l2 %dst, %src0, %src1, %calCount_i32 : !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, i32
    return
  }
  ```


### Ascend C代码生成模块

#### 前置说明

本模块的核心功能是实现 MLIR 中间表示（ASC-IR）到 Ascend C 代码的正确转换逻辑，需要实现对应代码翻译功能接口。

#### 具体开发步骤

- Step1：确认新增Op是否支持Ascend C代码自动生成，参考[AscendC_Op类定义介绍](#AscendC_Op类定义介绍)相关内容说明。

- Step2：根据能否自动生成Ascend C代码，完成相关代码适配或开发。
  - 支持Ascend C代码自动生成
    
    在ASC-IR定义模块完成Op节点的Ascend C代码自动生成配置，即可跳过本模块的开发步骤。 

    这里以`SetAtomicAdd` API为例，介绍如何配置Ascend C代码自动生成。
    
    对应的Ascend C函数原型如下：
    ```cpp
    template <typename T>
    __aicore__ inline void SetAtomicAdd() {}
    ```
    对应的ASC-IR定义如下：
    ```
    def AscendC_SetAtomicAddOp : APIOp<"set_atomic_add", "SetAtomicAdd", [AscFunc]> {
        let summary = "Enable atomic addition for subsequent data transfers using the specified data type";
        let arguments = (ins TypeAttr:$dtype);
        let paramTypeLists = [5];
    }
    ```
    - 通过标记AscFunc的traits，并配置paramTypeLists = [5]（其中5表示类型模板参数），完成Ascend C代码自动生成配置。

  - 不支持Ascend C代码自动生成

    这里以基础API的Gather L1级接口为例，进行展开介绍。  
    - 注册新增接口

      在[Translation.cpp](../lib/Target/AscendC/Translation.cpp)中的PrintableOpTypes中注册新增的接口，如将ascendc::GatherL1Op追加注册在已有接口的后面。其中，Op的命名由ASC-IR中的定义决定。
      ```cpp
      // other vector binary operators
      ..., ascendc::GatherL1Op, ...
      ```

    - 完成Ascend C代码生成函数接口声明
      
      根据新增接口的类型，在`include\ascir\Target\Asc\`目录下对应的头文件中，增加接口声明。
      
      以Gather L1级接口为例，在`include\ascir\Target\Asc\Basic\VecGather.h`文件中增加如下接口声明：
      ```cpp
      LogicalResult printOperation(CodeEmitter& emitter, ascendc::GatherL1Op op);
      ```

      注：若实现的printOperation接口包含模板参数，则声明和实现都在.h文件中完成。

    - 完成Ascend C代码生成函数接口实现

      根据新增接口的类型，在`lib\Target\AscendC\`目录下对应的cpp文件中，增加接口实现。
      
      以Gather L1级接口为例，在`lib\Target\AscendC\Basic\VecGather.cpp`中增加如下接口实现：
      ```cpp
      LogicalResult mlir::ascendc::printOperation(CodeEmitter &emitter, ascendc::GatherL1Op op) 
      {
          auto &os = emitter.ostream();
          auto maskName = printMask(emitter, op);

          os << ascNamespace << "::" << op.getAPIName() << "(" << emitter.getOrCreateName(op.getDst()) << ", "
            << emitter.getOrCreateName(op.getSrc()) << ", " << emitter.getOrCreateName(op.getDstOffset()) << ", "
            << emitter.getOrCreateName(op.getDstBase()) << ", " << maskName << ", "
            << emitter.getOrCreateName(op.getRepeatTimes()) << ", " << emitter.getOrCreateName(op.getSrcRepStride()) << ")";
          return success();
      }
      ```
      其中getDst、getSrc等接口是ASC-IR中定义的接口，用于获取对应节点中的值。

- Step3：开发UT测试用例

  具体过程与开发ASC-IR的UT测试用例相同。
  因为Ascend C代码生成模块的UT测试，主要是验证输入的ASC-IR能否正确翻译生成对应的Ascend C代码，故与ASC-IR定义模块共用测试框架和用例。

## 编码规范
### Python代码编码规范
遵守python官方代码风格pep8，具体要求参见 [pep主页](https://peps.python.org/pep-0008/)。

在代码上库前使用`ruff`和`yapf`工具包，自动规范代码。
```bash
ruff check --fix .\python\asc       # windows下路径
yapf -i --parallel -r .\setup.py
```

### C++代码编码规范
基本参考LLVM代码风格，在此基础上根据项目特点做出一些约束，具体要求见 [CodeStyle](./codestyle.rst)。

### Python前端模块编码规范
#### 命名调整
由于Python和C++遵守不同的命名规范，因而我们需要修改python侧相关接口的名称来符合相关规范，典型的修改规则如下表所示：
||Ascend C命名|pyasc命名|
|:-----------:|:-----------:|:-------------:|
|包/模块(namespace)|CapWords|lower_with_under|
|类|CapWords|CapWords|
|函数|CapWords|lower_with_under|
|变量|capWords(全局变量前面加上g_前缀)|lower_with_under|
|常量|CAPS_WITH_UNDER|CAPS_WITH_UNDER|

#### 接口重载规范
python基础语法不支持C++中的函数重载，对于需要重载的函数，pyasc使用@overload装饰所有对外API，使用@require_jit装饰实际调用Pybind创建MLIR节点的函数实现。下面为Add双目向量预算接口的C++接口和python接口对比：

***Ascend C Add 接口***
```cpp
// tensor前n个数据计算
template <typename T>
__aicore__ inline void Add(const LocalTensor<T>& dst, const LocalTensor<T>& src0, const LocalTensor<T>& src1, const int32_t& count)

// mask逐bit模式
template <typename T, bool isSetMask = true>
__aicore__ inline void Add(const LocalTensor<T>& dst, const LocalTensor<T>& src0, const LocalTensor<T>& src1, uint64_t mask[], const uint8_t repeatTime, const BinaryRepeatParams& repeatParams)

// mask连续模式
template <typename T, bool isSetMask = true>
__aicore__ inline void Add(const LocalTensor<T>& dst, const LocalTensor<T>& src0, const LocalTensor<T>& src1, uint64_t mask, const uint8_t repeatTime, const BinaryRepeatParams& repeatParams)
```

***pyasc add 接口***
```python
@overload
def add(dst: LocalTensor, src0: LocalTensor, src1: LocalTensor, count: int) -> None:
    ...

@overload
def add(dst: LocalTensor, src0: LocalTensor, src1: LocalTensor, mask: int, repeat_times: int,
        repeat_params: BinaryRepeatParams) -> None:
    ...

@overload
def add(dst: LocalTensor, src0: LocalTensor, src1: LocalTensor, mask: List[int], repeat_times: int,
        repeat_params: BinaryRepeatParams) -> None:
    ...

@require_jit
@set_binary_docstring("sum", cpp_name="Add")
def add(dst: LocalTensor, src0: LocalTensor, src1: LocalTensor, *args, **kwargs) -> None:
    builder = global_builder.get_ir_builder()
    op_impl("add", dst, src0, src1, args, kwargs, builder.create_asc_AddL0Op,
            builder.create_asc_AddL1Op, builder.create_asc_AddL2Op)
```


#### 模板参数处理规则
##### DataType模板参数
对于Tensor的数据类型，在python类型的AI编程语言中，一般将DataType作为Tensor的成员变量，调用计算接口时根据Tensor的DataType成员变量推导实际类型的计算接口，pyasc也使用这种方式处理DataType，因而Ascend C的API中涉及DataType的模板参数均通过Tensor参数进行推导，不在pyasc的接口中显示定义该参数。
##### 非DataType模板参数
对于非DataType模板参数，由于python中不存在*模板*，因而按照以下规则统一完成Ascend C模板参数到pyasc的映射：
1. 模板参数修改为运行时参数；
2. 如果模板参数是枚举、bool等常量类型，可以直接使用；如果是非常量类型，则加上asc.ConstExpr[origin_type]泛型标记，表明为参数应该传入常量；
3. 由于模板参数和运行时参数均可能存在*必选参数*和*可选参数*，而python语法要求*可选参数*必须放在*必选参数*后，因而我们按照以下顺序重新组装pyasc的参数：*运行时必选参数*， *模板必选参数*， *运行时可选参数*， *模板可选参数*。参见下面的例子：

***Ascend C接口***
```cpp
tempate <typename T, bool tplRequire, bool tplOptional = false>
__aicore__ inline void Func(const LocalTensor<T>& rtRequire, int32_t rtOptional=128);
```

***pyasc接口***
```python
def func(rt_require: LocalTensor, tpl_require: bool, rt_optional: int = 128, tpl_optional: bool = False):...
```
##### 其他模板处理规则
- 默认模板参数
Ascend C接口中存在的默认值，pyasc应该保持同样的默认值。
- 非类型模板参数
用来指定模板中一些常量值的参数，在pyasc中引入对应的枚举或者常量定义。

- 类型模板参数
模板中非常量的class参数，在C++中使用typename或class标记。
如果该class的字段在C++中全部通过模板参数指定，在pyasc中使用不可变数据类@dataclass(frozen=True)定义，该类型的实例一旦创建就不可修改。
如果该class的字段中有部分非模板参数指定，则在pyasc中通过Generic进行封装。

- 模板模板参数
与类型模板参数规则一致，递归进行处理内部模板。
- 可变模板参数
模板参数转为运行时参数，利用python的变长参数能力进行支持。

- 模板特化
在pyasc中通过运行时参数的实际类型进行分发。
