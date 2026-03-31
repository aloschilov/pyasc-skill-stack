# Python Secure Coding Guidelines

## Numeric safety

- Validate numeric inputs before arithmetic operations
- Guard against division by zero
- Be aware of floating-point precision limitations
- Use appropriate numeric types for the operation

## Input validation

- Validate all external inputs (tensor shapes, dtypes, sizes)
- Check tensor dimensions match expected values
- Validate core_num does not exceed hardware limits
- Ensure block_length divides evenly across cores

## Resource management

- Properly initialize and release streams
- Use `config.set_platform()` before kernel launch
- Handle device/host tensor placement correctly

## Error handling (host-side only)

- Use assertions with informative messages in host code
- Catch and report import errors gracefully (torch_npu)
- Validate command-line arguments

## No hardcoded secrets

- Do not hardcode file paths, credentials, or tokens
- Use environment variables or configuration for paths
- Parameterize hardware-specific values

## Safe tensor operations

- Verify tensor shapes are compatible before operations
- Check memory alignment requirements
- Ensure buffer sizes are sufficient for data
