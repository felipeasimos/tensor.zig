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

pub const TENSOR_1D = struct {
    test "check shape" {
        var data: [9]f64 = createSequence(f64, 9);
        const tensor = TensorView(f64, .{3}).init(data[0..]);
        try expectEqual(.{3}, tensor.shape);
    }
    test "check stride" {
        var data: [9]f64 = createSequence(f64, 9);
        const tensor = TensorView(f64, .{3}).init(data[0..]);
        try expectEqual(.{1}, tensor.strides);
    }
    test "indexing scalars" {
        var data: [9]f64 = createSequence(f64, 9);
        var tensor = TensorView(f64, .{3}).init(data[0..]);
        try expectEqual(data[0], tensor.scalar(.{0}));
        try expectEqual(data[1], tensor.scalar(.{1}));
        try expectEqual(data[2], tensor.scalar(.{2}));
    }
    test "mut sub tensor content (scalar)" {
        var data: [9]f64 = createSequence(f64, 9);
        var tensor = TensorView(f64, .{3}).init(data[0..]);
        try expectEqual(data[0], tensor.clone(.{0}));
        try expectEqual(data[1], tensor.clone(.{1}));
        try expectEqual(data[2], tensor.clone(.{2}));
    }
    test "slice" {
        var data: [9]f64 = createSequence(f64, 9);
        var tensor = TensorView(f64, .{3}).init(data[0..]);
        var subtensor = tensor.slice(.{.{ 1, 3 }});
        try expectEqual(data[1], subtensor.mut(.{0}));
        try expectEqual(data[2], subtensor.mut(.{1}));
    }
    test "mut tensor from view" {
        var data: [3]f64 = createSequence(f64, 3);
        var view = TensorView(f64, .{3}).init(data[0..]);
        var tensor = view.mut(.{});
        try expectEqual(tensor.data[0], tensor.clone(.{0}));
        try expectEqual(tensor.data[1], tensor.clone(.{1}));
        try expectEqual(tensor.data[2], tensor.clone(.{2}));

        try expectEqual(tensor.data[0], tensor.clone(.{0}));
        try expectEqual(tensor.data[1], tensor.clone(.{1}));
        try expectEqual(tensor.data[2], tensor.clone(.{2}));
    }
    test "mut view from tensor from view" {
        var data: [3]f64 = createSequence(f64, 3);
        var view_tmp = TensorView(f64, .{3}).init(data[0..]);
        var tensor_tmp = view_tmp.mut(.{});
        var view = tensor_tmp.view(.{});
        try expectEqual(view.data[0], view.clone(.{0}));
        try expectEqual(view.data[1], view.clone(.{1}));
        try expectEqual(view.data[2], view.clone(.{2}));
    }
    test "element wise operation with a scalar (wise - in place)" {
        var data: [3]f64 = createSequence(f64, 3);
        var tensor1 = Tensor(f64, .{3}).init(data[0..]);
        var result = Tensor(f64, .{3}).init(data[0..]);
        tensor1.wise(2, &result, (struct {
            pub fn func(a: f64, b: f64) f64 {
                return a + b;
            }
        }).func);
        try expectEqual(2, result.clone(.{0}));
        try expectEqual(3, result.clone(.{1}));
        try expectEqual(4, result.clone(.{2}));
    }
    test "element wise operation with a tensor (wise - in place)" {
        var data: [3]f64 = createSequence(f64, 3);
        var tensor1 = Tensor(f64, .{3}).init(data[0..]);
        var tensor2 = Tensor(f64, .{3}).init(data[0..]);
        var result = Tensor(f64, .{3}).init(data[0..]);
        tensor1.wise(&tensor2, &result, (struct {
            pub fn func(a: f64, b: f64) f64 {
                return a + b;
            }
        }).func);
        try expectEqual(0, result.clone(.{0}));
        try expectEqual(2, result.clone(.{1}));
        try expectEqual(4, result.clone(.{2}));

        try expectEqual(0, tensor2.clone(.{0}));
        try expectEqual(1, tensor2.clone(.{1}));
        try expectEqual(2, tensor2.clone(.{2}));
    }
    test "wiseNew element-wise addition with scalar" {
        var data: [3]f64 = createSequence(f64, 3);
        const tensor1 = Tensor(f64, .{3}).init(data[0..]);
        const result = tensor1.wiseNew(2, (struct {
            pub fn func(a: f64, b: f64) f64 {
                return a + b;
            }
        }).func);
        try expectEqual(2, result.clone(.{0}));
        try expectEqual(3, result.clone(.{1}));
        try expectEqual(4, result.clone(.{2}));
        // Original tensor should remain unchanged
        try expectEqual(0, tensor1.clone(.{0}));
        try expectEqual(1, tensor1.clone(.{1}));
        try expectEqual(2, tensor1.clone(.{2}));
    }
    test "wiseNew element-wise addition with tensor" {
        var data1: [3]f64 = createSequence(f64, 3);
        var data2: [3]f64 = .{ 10, 20, 30 };
        const tensor1 = Tensor(f64, .{3}).init(data1[0..]);
        const tensor2 = Tensor(f64, .{3}).init(data2[0..]);
        const result = tensor1.wiseNew(&tensor2, (struct {
            pub fn func(a: f64, b: f64) f64 {
                return a + b;
            }
        }).func);
        try expectEqual(10, result.clone(.{0}));
        try expectEqual(21, result.clone(.{1}));
        try expectEqual(32, result.clone(.{2}));
        // Original tensors should remain unchanged
        try expectEqual(0, tensor1.clone(.{0}));
        try expectEqual(1, tensor1.clone(.{1}));
        try expectEqual(2, tensor1.clone(.{2}));
        try expectEqual(10, tensor2.clone(.{0}));
        try expectEqual(20, tensor2.clone(.{1}));
        try expectEqual(30, tensor2.clone(.{2}));
    }
    test "1D convolution - simple case" {
        var data: [5]f64 = .{ 1, 2, 3, 4, 5 };
        var kernel_data: [3]f64 = .{ 1, 1, 1 };
        var tensor = Tensor(f64, .{5}).init(data[0..]);
        var kernel = Tensor(f64, .{3}).init(kernel_data[0..]);
        
        const result = tensor.convolutionNew(&kernel);
        
        // Expected: [6, 9, 12] (sum of 3 consecutive elements)
        try expectEqual(.{3}, result.shape);
        try expectEqual(6, result.clone(.{0})); // 1+2+3
        try expectEqual(9, result.clone(.{1})); // 2+3+4
        try expectEqual(12, result.clone(.{2})); // 3+4+5
    }

    test "1D convolution - edge detection kernel" {
        var data: [5]f64 = .{ 1, 2, 3, 4, 5 };
        var kernel_data: [3]f64 = .{ -1, 0, 1 };
        var tensor = Tensor(f64, .{5}).init(data[0..]);
        var kernel = Tensor(f64, .{3}).init(kernel_data[0..]);
        
        const result = tensor.convolutionNew(&kernel);
        
        // Expected: [2, 2, 2] (difference between consecutive elements)
        try expectEqual(.{3}, result.shape);
        try expectEqual(2, result.clone(.{0})); // -1*1 + 0*2 + 1*3 = 2
        try expectEqual(2, result.clone(.{1})); // -1*2 + 0*3 + 1*4 = 2
        try expectEqual(2, result.clone(.{2})); // -1*3 + 0*4 + 1*5 = 2
    }

    test "1D convolution - in-place operation" {
        var data: [5]f64 = .{ 1, 2, 3, 4, 5 };
        var kernel_data: [3]f64 = .{ 1, 1, 1 };
        var tensor = Tensor(f64, .{5}).init(data[0..]);
        var kernel = Tensor(f64, .{3}).init(kernel_data[0..]);
        var result_data: [3]f64 = .{0} ** 3;
        var result = Tensor(f64, .{3}).init(result_data[0..]);
        
        tensor.convolution(&kernel, &result);
        
        try expectEqual(6, result.clone(.{0}));
        try expectEqual(9, result.clone(.{1}));
        try expectEqual(12, result.clone(.{2}));
    }
};
