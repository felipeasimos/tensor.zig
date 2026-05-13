# tensor.zig

A Zig tensor library with compile-time shapes and efficient operations.

Following the explicity of zig, there is no hidden copies of tensors happening at any point in this library!
> But what if this messes up too much the strides and shapes?
Then a compile error is thrown to help you manually tweak the operations. No performance / memory surprises at runtime!

## Tensor Creation

```zig
// Create tensor with compile-time shape
var tensor = Tensor(f64, .{3, 3}).init(&data);
var ref = TensorRef(f64, .{3, 3}).init(data[0..]);

// Create zero-filled tensor
var zeros = Tensor(f64, .{2, 2}).zeroes();

// Create random tensor
var rand = Tensor(f64, .{2, 2}).random(rng);
```

## Basic Operations

### Element Access
```zig
// Get scalar value
var value = tensor.scalar(.{0, 1});

// Get mutable reference
var ref = tensor.mut(.{0, 1});

// Clone subtensor
var sub = tensor.clone(.{0});
```

### References and Slicing
```zig
// Get reference of subtensor
var ref = tensor.ref(.{0});

// Slice with ranges
var slice = tensor.slice(.{.{1, 3}, .{0, 2}});

// Reshape (contiguous tensors only)
var reshaped = tensor.reshape(.{6});
```

### Element-wise Operations
```zig
// In-place operation
tensor.wise(other, &result, addFunc);

// Create new tensor
var result = tensor.wiseNew(scalar, addFunc);

// Apply function
tensor.apply(squareFunc);
```

### Matrix Operations
```zig
// Matrix multiplication
op.matmul(&a, &b, &result);

// Matrix multiplication with new result
var result = op.matmulNew(&a, &b);

// Transpose
var transposed = tensor.transpose(.{});
```

### Broadcasting
```zig
// Broadcast to target shape
var broadcasted = tensor.broadcast(.{3, 4});
```

### Iteration
```zig
// Iterate over elements
var iter = tensor.iter();
while (iter.next()) |item| {
    // item.indices, item.value
}
```

## Function Factories

```zig
const func = @import("tensor").func;

// Basic arithmetic
const add = func.addFactory(f64);
const sub = func.subFactory(f64);
const mul = func.mulFactory(f64);
const div = func.divFactory(f64);
```

## Getting column major matrices

Before initializing the data inside the matrix, you just need to tranpose the indices approprietly. One easy way to do this is calling `transpose`:

```zig
const b_transposed = b.transpose(.{});
// you can avoid data copies using a `ref`.
// if your tensor is already a ref nothing will be copied
const b_ref_for_sure_not_copied = b.ref(.{}).transpose(.{});
```

You can also just clone a transposed matrix.
