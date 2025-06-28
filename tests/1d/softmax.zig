const std = @import("std");
const expectApproxEqAbs = std.testing.expectApproxEqAbs;
const Tensor = @import("tensor").Tensor;

test "1D softmax basic" {
    const T = Tensor(f32, .{4});
    var data = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    var tensor = T.init(&data);
    var result = T{ .data = undefined };
    
    tensor.softmax(&result);
    
    // Expected values computed manually:
    // max = 4.0
    // exp(1-4) = exp(-3) ≈ 0.0498
    // exp(2-4) = exp(-2) ≈ 0.1353
    // exp(3-4) = exp(-1) ≈ 0.3679
    // exp(4-4) = exp(0) = 1.0
    // sum ≈ 1.5530
    // normalized: [0.0321, 0.0871, 0.2369, 0.6439]
    
    try expectApproxEqAbs(@as(f32, 0.0321), result.data[0], 0.001);
    try expectApproxEqAbs(@as(f32, 0.0871), result.data[1], 0.001);
    try expectApproxEqAbs(@as(f32, 0.2369), result.data[2], 0.001);
    try expectApproxEqAbs(@as(f32, 0.6439), result.data[3], 0.001);
}

test "1D softmax with negative values" {
    const T = Tensor(f32, .{3});
    var data = [_]f32{ -1.0, 0.0, 1.0 };
    var tensor = T.init(&data);
    var result = T{ .data = undefined };
    
    tensor.softmax(&result);
    
    // max = 1.0
    // exp(-1-1) = exp(-2) ≈ 0.1353
    // exp(0-1) = exp(-1) ≈ 0.3679
    // exp(1-1) = exp(0) = 1.0
    // sum ≈ 1.5032
    // normalized: [0.0900, 0.2447, 0.6653]
    
    try expectApproxEqAbs(@as(f32, 0.0900), result.data[0], 0.001);
    try expectApproxEqAbs(@as(f32, 0.2447), result.data[1], 0.001);
    try expectApproxEqAbs(@as(f32, 0.6653), result.data[2], 0.001);
}

test "1D softmax sum equals 1" {
    const T = Tensor(f32, .{5});
    var data = [_]f32{ 0.1, 0.2, 0.3, 0.4, 0.5 };
    var tensor = T.init(&data);
    var result = T{ .data = undefined };
    
    tensor.softmax(&result);
    
    // Check that sum equals 1.0
    var sum: f32 = 0;
    for (result.data) |val| {
        sum += val;
    }
    try expectApproxEqAbs(@as(f32, 1.0), sum, 0.001);
}

test "1D softmax new" {
    const T = Tensor(f32, .{3});
    var data = [_]f32{ 1.0, 2.0, 3.0 };
    var tensor = T.init(&data);
    
    const result = tensor.softmaxNew();
    
    // Check that sum equals 1.0
    var sum: f32 = 0;
    for (result.data) |val| {
        sum += val;
    }
    try expectApproxEqAbs(@as(f32, 1.0), sum, 0.001);
    
    // Check that all values are positive
    for (result.data) |val| {
        try std.testing.expect(val > 0);
    }
}

test "1D axis-wise softmax (axis=0)" {
    const T = Tensor(f32, .{4});
    var data = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    var tensor = T.init(&data);
    var result = T{ .data = undefined };
    tensor.softmaxAxis(0, &result);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0321), result.data[0], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0871), result.data[1], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 0.2369), result.data[2], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 0.6439), result.data[3], 0.001);
}

test "1D axis-wise softmax new (axis=0)" {
    const T = Tensor(f32, .{4});
    var data = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    var tensor = T.init(&data);
    const result = tensor.softmaxAxisNew(0);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0321), result.data[0], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0871), result.data[1], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 0.2369), result.data[2], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 0.6439), result.data[3], 0.001);
} 