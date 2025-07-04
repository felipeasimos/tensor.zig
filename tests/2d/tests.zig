const std = @import("std");
const expectEqual = std.testing.expectEqual;
const expect = std.testing.expect;
const TensorView = @import("tensor").TensorView;
const Tensor = @import("tensor").Tensor;

fn createSequence(comptime dtype: type, comptime n: usize) [n]dtype {
    var seq: [n]dtype = .{1} ** n;
    inline for (0..n) |i| {
        seq[i] = i;
    }
    return seq;
}

pub const TENSOR_2D = struct {
    test "check shape" {
        var data: [9]f64 = createSequence(f64, 9);
        const tensor = TensorView(f64, .{ 3, 3 }).init(data[0..]);
        try expectEqual(.{ 3, 3 }, tensor.shape);
    }
    test "check stride" {
        var data: [9]f64 = createSequence(f64, 9);
        const tensor = TensorView(f64, .{ 3, 3 }).init(data[0..]);
        try expectEqual(.{ 3, 1 }, tensor.strides);
    }
    test "indexing scalars" {
        var data: [9]f64 = createSequence(f64, 9);
        var tensor = TensorView(f64, .{ 3, 3 }).init(data[0..]);

        try expectEqual(data[0], tensor.scalar(.{ 0, 0 }));
        try expectEqual(data[1], tensor.scalar(.{ 0, 1 }));
        try expectEqual(data[2], tensor.scalar(.{ 0, 2 }));

        try expectEqual(data[3], tensor.scalar(.{ 1, 0 }));
        try expectEqual(data[4], tensor.scalar(.{ 1, 1 }));
        try expectEqual(data[5], tensor.scalar(.{ 1, 2 }));

        try expectEqual(data[6], tensor.scalar(.{ 2, 0 }));
        try expectEqual(data[7], tensor.scalar(.{ 2, 1 }));
        try expectEqual(data[8], tensor.scalar(.{ 2, 2 }));
    }
    test "mut sub tensor content (scalar)" {
        var data: [9]f64 = createSequence(f64, 9);
        var tensor = TensorView(f64, .{ 3, 3 }).init(data[0..]);

        try expectEqual(data[0], tensor.mut(.{ 0, 0 }));
        try expectEqual(data[1], tensor.mut(.{ 0, 1 }));
        try expectEqual(data[2], tensor.mut(.{ 0, 2 }));

        try expectEqual(data[3], tensor.mut(.{ 1, 0 }));
        try expectEqual(data[4], tensor.mut(.{ 1, 1 }));
        try expectEqual(data[5], tensor.mut(.{ 1, 2 }));

        try expectEqual(data[6], tensor.mut(.{ 2, 0 }));
        try expectEqual(data[7], tensor.mut(.{ 2, 1 }));
        try expectEqual(data[8], tensor.mut(.{ 2, 2 }));
    }
    test "mut sub tensor content (vector)" {
        var data: [9]f64 = createSequence(f64, 9);
        var tensor = TensorView(f64, .{ 3, 3 }).init(data[0..]);

        try expectEqual(data[0..3], tensor.mut(.{0}).data);
        try expectEqual(data[3..6], tensor.mut(.{1}).data);
        try expectEqual(data[6..9], tensor.mut(.{2}).data);
    }
    test "reshape 4x4 to 2x8" {
        var data: [16]f64 = createSequence(f64, 16);
        var tensor = TensorView(f64, .{ 4, 4 }).init(data[0..]);

        var new_tensor = tensor.reshape(.{ 2, 8 });

        try expectEqual(data[0..8], new_tensor.mut(.{0}).data);
        try expectEqual(data[8..16], new_tensor.mut(.{1}).data);
    }
    test "slice" {
        var data: [16]f64 = createSequence(f64, 16);
        var tensor = TensorView(f64, .{ 4, 4 }).init(data[0..]);
        var subtensor = tensor.slice(.{
            .{ 1, 3 },
            .{ 1, 3 },
        });
        try expectEqual(data[5], subtensor.mut(.{ 0, 0 }));
        try expectEqual(data[6], subtensor.mut(.{ 0, 1 }));
        try expectEqual(data[9], subtensor.mut(.{ 1, 0 }));
        try expectEqual(data[10], subtensor.mut(.{ 1, 1 }));
    }
    test "matmul 3x4 4x2" {
        var data1: [12]f64 = createSequence(f64, 12);
        var tensor1 = Tensor(f64, .{ 3, 4 }).init(&data1);
        var data2: [8]f64 = createSequence(f64, 8);
        var tensor2 = Tensor(f64, .{ 4, 2 }).init(&data2);

        var data3: [6]f64 = createSequence(f64, 6);
        var result = Tensor(f64, .{ 3, 2 }).init(&data3);
        tensor1.matmul(&tensor2, &result);
        try expectEqual(result.shape, [_]usize{ 3, 2 });
    }
    test "matmulNew 3x4 4x2" {
        var data1: [12]f64 = createSequence(f64, 12);
        var tensor1 = Tensor(f64, .{ 3, 4 }).init(&data1);
        var data2: [8]f64 = createSequence(f64, 8);
        var tensor2 = Tensor(f64, .{ 4, 2 }).init(&data2);

        const result = tensor1.matmulNew(&tensor2);
        try expectEqual(result.shape, [_]usize{ 3, 2 });
    }
    test "transpose" {
        var data: [12]f64 = createSequence(f64, 12);
        var tensor = Tensor(f64, .{ 3, 4 }).init(&data);
        const transpose = tensor.transpose(.{});
        try expectEqual(.{ 4, 3 }, transpose.shape);

        try expectEqual(data[0], transpose.clone(.{ 0, 0 }));
        try expectEqual(data[4], transpose.clone(.{ 0, 1 }));
        try expectEqual(data[8], transpose.clone(.{ 0, 2 }));

        try expectEqual(data[1], transpose.clone(.{ 1, 0 }));
        try expectEqual(data[5], transpose.clone(.{ 1, 1 }));
        try expectEqual(data[9], transpose.clone(.{ 1, 2 }));

        try expectEqual(data[2], transpose.clone(.{ 2, 0 }));
        try expectEqual(data[6], transpose.clone(.{ 2, 1 }));
        try expectEqual(data[10], transpose.clone(.{ 2, 2 }));

        try expectEqual(data[3], transpose.clone(.{ 3, 0 }));
        try expectEqual(data[7], transpose.clone(.{ 3, 1 }));
        try expectEqual(data[11], transpose.clone(.{ 3, 2 }));
    }
    test "wise element-wise addition with scalar (in place)" {
        var data: [4]f64 = .{ 1, 2, 3, 4 };
        var tensor = Tensor(f64, .{ 2, 2 }).init(data[0..]);
        var result = Tensor(f64, .{ 2, 2 }).init(data[0..]);
        tensor.wise(10, &result, (struct {
            pub fn func(a: f64, b: f64) f64 {
                return a + b;
            }
        }).func);
        try expectEqual(11, result.clone(.{ 0, 0 }));
        try expectEqual(12, result.clone(.{ 0, 1 }));
        try expectEqual(13, result.clone(.{ 1, 0 }));
        try expectEqual(14, result.clone(.{ 1, 1 }));
    }
    test "wise element-wise addition with tensor (in place)" {
        var data1: [4]f64 = .{ 1, 2, 3, 4 };
        var data2: [4]f64 = .{ 10, 20, 30, 40 };
        var tensor1 = Tensor(f64, .{ 2, 2 }).init(data1[0..]);
        var tensor2 = Tensor(f64, .{ 2, 2 }).init(data2[0..]);
        var result = Tensor(f64, .{ 2, 2 }).init(data1[0..]);
        tensor1.wise(&tensor2, &result, (struct {
            pub fn func(a: f64, b: f64) f64 {
                return a + b;
            }
        }).func);
        try expectEqual(11, result.clone(.{ 0, 0 }));
        try expectEqual(22, result.clone(.{ 0, 1 }));
        try expectEqual(33, result.clone(.{ 1, 0 }));
        try expectEqual(44, result.clone(.{ 1, 1 }));
        // tensor2 should remain unchanged
        try expectEqual(10, tensor2.clone(.{ 0, 0 }));
        try expectEqual(20, tensor2.clone(.{ 0, 1 }));
        try expectEqual(30, tensor2.clone(.{ 1, 0 }));
        try expectEqual(40, tensor2.clone(.{ 1, 1 }));
    }
    test "wiseNew element-wise addition with scalar" {
        var data: [4]f64 = .{ 1, 2, 3, 4 };
        const tensor = Tensor(f64, .{ 2, 2 }).init(data[0..]);
        const result = tensor.wiseNew(10, (struct {
            pub fn func(a: f64, b: f64) f64 {
                return a + b;
            }
        }).func);
        try expectEqual(11, result.clone(.{ 0, 0 }));
        try expectEqual(12, result.clone(.{ 0, 1 }));
        try expectEqual(13, result.clone(.{ 1, 0 }));
        try expectEqual(14, result.clone(.{ 1, 1 }));
        // Original tensor should remain unchanged
        try expectEqual(1, tensor.clone(.{ 0, 0 }));
        try expectEqual(2, tensor.clone(.{ 0, 1 }));
        try expectEqual(3, tensor.clone(.{ 1, 0 }));
        try expectEqual(4, tensor.clone(.{ 1, 1 }));
    }
    test "wiseNew element-wise addition with tensor" {
        var data1: [4]f64 = .{ 1, 2, 3, 4 };
        var data2: [4]f64 = .{ 10, 20, 30, 40 };
        const tensor1 = Tensor(f64, .{ 2, 2 }).init(data1[0..]);
        const tensor2 = Tensor(f64, .{ 2, 2 }).init(data2[0..]);
        const result = tensor1.wiseNew(&tensor2, (struct {
            pub fn func(a: f64, b: f64) f64 {
                return a + b;
            }
        }).func);
        try expectEqual(11, result.clone(.{ 0, 0 }));
        try expectEqual(22, result.clone(.{ 0, 1 }));
        try expectEqual(33, result.clone(.{ 1, 0 }));
        try expectEqual(44, result.clone(.{ 1, 1 }));
        // Original tensors should remain unchanged
        try expectEqual(1, tensor1.clone(.{ 0, 0 }));
        try expectEqual(2, tensor1.clone(.{ 0, 1 }));
        try expectEqual(3, tensor1.clone(.{ 1, 0 }));
        try expectEqual(4, tensor1.clone(.{ 1, 1 }));
        try expectEqual(10, tensor2.clone(.{ 0, 0 }));
        try expectEqual(20, tensor2.clone(.{ 0, 1 }));
        try expectEqual(30, tensor2.clone(.{ 1, 0 }));
        try expectEqual(40, tensor2.clone(.{ 1, 1 }));
    }
};

pub const CONVOLUTION_2D = struct {
    test "2D convolution - simple averaging kernel" {
        var data: [9]f64 = .{ 1, 2, 3, 4, 5, 6, 7, 8, 9 };
        var kernel_data: [4]f64 = .{ 0.25, 0.25, 0.25, 0.25 };
        var tensor = Tensor(f64, .{ 3, 3 }).init(data[0..]);
        var kernel = Tensor(f64, .{ 2, 2 }).init(kernel_data[0..]);
        
        const result = tensor.convolutionNew(&kernel);
        
        // Expected: 2x2 output
        try expectEqual(.{ 2, 2 }, result.shape);
        // Top-left: (1+2+4+5)*0.25 = 3
        try expectEqual(3, result.clone(.{ 0, 0 }));
        // Top-right: (2+3+5+6)*0.25 = 4
        try expectEqual(4, result.clone(.{ 0, 1 }));
        // Bottom-left: (4+5+7+8)*0.25 = 6
        try expectEqual(6, result.clone(.{ 1, 0 }));
        // Bottom-right: (5+6+8+9)*0.25 = 7
        try expectEqual(7, result.clone(.{ 1, 1 }));
    }

    test "2D convolution - edge detection kernel" {
        var data: [9]f64 = .{ 1, 2, 3, 4, 5, 6, 7, 8, 9 };
        var kernel_data: [9]f64 = .{ -1, -1, -1, -1, 8, -1, -1, -1, -1 };
        var tensor = Tensor(f64, .{ 3, 3 }).init(data[0..]);
        var kernel = Tensor(f64, .{ 3, 3 }).init(kernel_data[0..]);
        
        const result = tensor.convolutionNew(&kernel);
        
        // Expected: 1x1 output (3x3 input with 3x3 kernel = 1x1 output)
        try expectEqual(.{ 1, 1 }, result.shape);
        // Center element: 8*5 - (1+2+3+4+6+7+8+9) = 40 - 40 = 0
        try expectEqual(0, result.clone(.{ 0, 0 }));
    }

    test "2D convolution - in-place operation" {
        var data: [9]f64 = .{ 1, 2, 3, 4, 5, 6, 7, 8, 9 };
        var kernel_data: [4]f64 = .{ 1, 1, 1, 1 };
        var tensor = Tensor(f64, .{ 3, 3 }).init(data[0..]);
        var kernel = Tensor(f64, .{ 2, 2 }).init(kernel_data[0..]);
        var result_data: [4]f64 = .{0} ** 4;
        var result = Tensor(f64, .{ 2, 2 }).init(result_data[0..]);
        
        tensor.convolution(&kernel, &result);
        
        try expectEqual(12, result.clone(.{ 0, 0 })); // 1+2+4+5
        try expectEqual(16, result.clone(.{ 0, 1 })); // 2+3+5+6
        try expectEqual(24, result.clone(.{ 1, 0 })); // 4+5+7+8
        try expectEqual(28, result.clone(.{ 1, 1 })); // 5+6+8+9
    }
};

pub const POOLING_2D = struct {
    test "2D max pooling" {
        var data: [9]f64 = .{ 1, 2, 3, 4, 9, 6, 7, 8, 5 };
        var tensor = Tensor(f64, .{ 3, 3 }).init(data[0..]);
        
        const result = tensor.poolingNew(.{ 2, 2 }, (struct {
            pub fn max_pool(acc: f64, idx: usize, val: f64) f64 {
                return if (idx == 0) val else @max(acc, val);
            }
        }).max_pool);
        
        // Expected: 2x2 output
        try expectEqual(.{ 2, 2 }, result.shape);
        // Top-left: max(1,2,4,9) = 9
        try expectEqual(9, result.clone(.{ 0, 0 }));
        // Top-right: max(2,3,9,6) = 9
        try expectEqual(9, result.clone(.{ 0, 1 }));
        // Bottom-left: max(4,9,7,8) = 9
        try expectEqual(9, result.clone(.{ 1, 0 }));
        // Bottom-right: max(9,6,8,5) = 9
        try expectEqual(9, result.clone(.{ 1, 1 }));
    }

    test "2D average pooling" {
        var data: [9]f64 = .{ 1, 2, 3, 4, 5, 6, 7, 8, 9 };
        var tensor = Tensor(f64, .{ 3, 3 }).init(data[0..]);
        
        const result = tensor.poolingNew(.{ 2, 2 }, (struct {
            pub fn avg_pool(acc: f64, idx: usize, val: f64) f64 {
                _ = idx; // Suppress unused parameter warning
                return acc + val;
            }
        }).avg_pool);
        
        // Expected: 2x2 output (sum of 2x2 windows)
        try expectEqual(.{ 2, 2 }, result.shape);
        // Top-left: 1+2+4+5 = 12
        try expectEqual(12, result.clone(.{ 0, 0 }));
        // Top-right: 2+3+5+6 = 16
        try expectEqual(16, result.clone(.{ 0, 1 }));
        // Bottom-left: 4+5+7+8 = 24
        try expectEqual(24, result.clone(.{ 1, 0 }));
        // Bottom-right: 5+6+8+9 = 28
        try expectEqual(28, result.clone(.{ 1, 1 }));
    }

    test "2D pooling - in-place operation" {
        var data: [9]f64 = .{ 1, 2, 3, 4, 9, 6, 7, 8, 5 };
        var tensor = Tensor(f64, .{ 3, 3 }).init(data[0..]);
        var result_data: [4]f64 = .{0} ** 4;
        var result = Tensor(f64, .{ 2, 2 }).init(result_data[0..]);
        
        tensor.pooling(.{ 2, 2 }, &result, (struct {
            pub fn max_pool(acc: f64, idx: usize, val: f64) f64 {
                return if (idx == 0) val else @max(acc, val);
            }
        }).max_pool);
        
        try expectEqual(9, result.clone(.{ 0, 0 }));
        try expectEqual(9, result.clone(.{ 0, 1 }));
        try expectEqual(9, result.clone(.{ 1, 0 }));
        try expectEqual(9, result.clone(.{ 1, 1 }));
    }
};

