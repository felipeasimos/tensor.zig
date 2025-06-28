const std = @import("std");
const expectApproxEqAbs = std.testing.expectApproxEqAbs;
const Tensor = @import("tensor").Tensor;

test "3D softmax basic" {
    const T = Tensor(f32, .{ 2, 2, 2 });
    var data = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 };
    var tensor = T.init(&data);
    var result = T{ .data = undefined };
    
    tensor.softmax(&result);
    
    // max = 8.0
    // exp(1-8) = exp(-7) ≈ 0.0009
    // exp(2-8) = exp(-6) ≈ 0.0025
    // exp(3-8) = exp(-5) ≈ 0.0067
    // exp(4-8) = exp(-4) ≈ 0.0183
    // exp(5-8) = exp(-3) ≈ 0.0498
    // exp(6-8) = exp(-2) ≈ 0.1353
    // exp(7-8) = exp(-1) ≈ 0.3679
    // exp(8-8) = exp(0) = 1.0
    // sum ≈ 1.5814
    // normalized: [0.0006, 0.0016, 0.0042, 0.0116, 0.0315, 0.0857, 0.2330, 0.6318]
    
    try expectApproxEqAbs(@as(f32, 0.0006), result.data[0], 0.001);
    try expectApproxEqAbs(@as(f32, 0.0016), result.data[1], 0.001);
    try expectApproxEqAbs(@as(f32, 0.0042), result.data[2], 0.001);
    try expectApproxEqAbs(@as(f32, 0.0116), result.data[3], 0.001);
    try expectApproxEqAbs(@as(f32, 0.0315), result.data[4], 0.001);
    try expectApproxEqAbs(@as(f32, 0.0857), result.data[5], 0.001);
    try expectApproxEqAbs(@as(f32, 0.2330), result.data[6], 0.001);
    try expectApproxEqAbs(@as(f32, 0.6318), result.data[7], 0.001);
}

test "3D softmax with small values" {
    const T = Tensor(f32, .{ 1, 2, 3 });
    var data = [_]f32{ 0.1, 0.2, 0.3, 0.4, 0.5, 0.6 };
    var tensor = T.init(&data);
    var result = T{ .data = undefined };
    
    tensor.softmax(&result);
    
    // max = 0.6
    // exp(0.1-0.6) = exp(-0.5) ≈ 0.6065
    // exp(0.2-0.6) = exp(-0.4) ≈ 0.6703
    // exp(0.3-0.6) = exp(-0.3) ≈ 0.7408
    // exp(0.4-0.6) = exp(-0.2) ≈ 0.8187
    // exp(0.5-0.6) = exp(-0.1) ≈ 0.9048
    // exp(0.6-0.6) = exp(0) = 1.0
    // sum ≈ 4.7411
    // normalized: [0.1279, 0.1414, 0.1562, 0.1727, 0.1909, 0.2109]
    
    try expectApproxEqAbs(@as(f32, 0.1279), result.data[0], 0.001);
    try expectApproxEqAbs(@as(f32, 0.1414), result.data[1], 0.001);
    try expectApproxEqAbs(@as(f32, 0.1562), result.data[2], 0.001);
    try expectApproxEqAbs(@as(f32, 0.1727), result.data[3], 0.001);
    try expectApproxEqAbs(@as(f32, 0.1909), result.data[4], 0.001);
    try expectApproxEqAbs(@as(f32, 0.2109), result.data[5], 0.001);
}

test "3D softmax sum equals 1" {
    const T = Tensor(f32, .{ 2, 1, 2 });
    var data = [_]f32{ -0.5, 0.5, 1.5, 2.5 };
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

test "3D softmax new" {
    const T = Tensor(f32, .{ 1, 1, 3 });
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
    
    // Check that highest input corresponds to highest output
    try std.testing.expect(result.data[2] > result.data[1]);
    try std.testing.expect(result.data[1] > result.data[0]);
}

test "3D softmax numerical stability" {
    const T = Tensor(f32, .{ 1, 1, 4 });
    var data = [_]f32{ 1000.0, 1001.0, 1002.0, 1003.0 };
    var tensor = T.init(&data);
    var result = T{ .data = undefined };
    
    tensor.softmax(&result);
    
    // This test ensures numerical stability with large values
    // Without subtracting max, exp(1000) would overflow
    
    // Check that sum equals 1.0
    var sum: f32 = 0;
    for (result.data) |val| {
        sum += val;
    }
    try expectApproxEqAbs(@as(f32, 1.0), sum, 0.001);
    
    // Check that all values are positive and finite
    for (result.data) |val| {
        try std.testing.expect(val > 0);
        try std.testing.expect(std.math.isFinite(val));
    }
} 