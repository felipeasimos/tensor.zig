# tensor.zig

A Zig tensor library with compile-time shapes and efficient operations.

## Tensor Creation

```zig
// Create tensor with compile-time shape
var tensor = Tensor(f64, .{3, 3}).init(&data);
var view = TensorView(f64, .{3, 3}).init(data[0..]);

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

### Views and Slicing
```zig
// Get view of subtensor
var view = tensor.view(.{0});

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

