const std = @import("std");
const expectApproxEqAbs = std.testing.expectApproxEqAbs;
const Tensor = @import("tensor").Tensor;

test "2D softmax basic" {
    const T = Tensor(f32, .{ 2, 3 });
    var data = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 };
    var tensor = T.init(&data);
    var result = T{ .data = undefined };
    
    tensor.softmax(&result);
    
    // max = 6.0
    // exp(1-6) = exp(-5) ≈ 0.0067
    // exp(2-6) = exp(-4) ≈ 0.0183
    // exp(3-6) = exp(-3) ≈ 0.0498
    // exp(4-6) = exp(-2) ≈ 0.1353
    // exp(5-6) = exp(-1) ≈ 0.3679
    // exp(6-6) = exp(0) = 1.0
    // sum ≈ 1.5780
    // normalized: [0.0042, 0.0116, 0.0315, 0.0857, 0.2330, 0.6340]
    
    try expectApproxEqAbs(@as(f32, 0.0042), result.data[0], 0.001);
    try expectApproxEqAbs(@as(f32, 0.0116), result.data[1], 0.001);
    try expectApproxEqAbs(@as(f32, 0.0315), result.data[2], 0.001);
    try expectApproxEqAbs(@as(f32, 0.0857), result.data[3], 0.001);
    try expectApproxEqAbs(@as(f32, 0.2330), result.data[4], 0.001);
    try expectApproxEqAbs(@as(f32, 0.6340), result.data[5], 0.001);
}

test "2D softmax with mixed values" {
    const T = Tensor(f32, .{ 2, 2 });
    var data = [_]f32{ -1.0, 0.0, 1.0, 2.0 };
    var tensor = T.init(&data);
    var result = T{ .data = undefined };
    
    tensor.softmax(&result);
    
    // max = 2.0
    // exp(-1-2) = exp(-3) ≈ 0.0498
    // exp(0-2) = exp(-2) ≈ 0.1353
    // exp(1-2) = exp(-1) ≈ 0.3679
    // exp(2-2) = exp(0) = 1.0
    // sum ≈ 1.5530
    // normalized: [0.0321, 0.0871, 0.2369, 0.6439]
    
    try expectApproxEqAbs(@as(f32, 0.0321), result.data[0], 0.001);
    try expectApproxEqAbs(@as(f32, 0.0871), result.data[1], 0.001);
    try expectApproxEqAbs(@as(f32, 0.2369), result.data[2], 0.001);
    try expectApproxEqAbs(@as(f32, 0.6439), result.data[3], 0.001);
}

test "2D softmax sum equals 1" {
    const T = Tensor(f32, .{ 3, 2 });
    var data = [_]f32{ 0.1, 0.2, 0.3, 0.4, 0.5, 0.6 };
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

test "2D softmax new" {
    const T = Tensor(f32, .{ 2, 2 });
    var data = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
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
    
    // Check that highest input corresponds to highest output
    try std.testing.expect(result.data[3] > result.data[2]);
    try std.testing.expect(result.data[2] > result.data[1]);
    try std.testing.expect(result.data[1] > result.data[0]);
} 