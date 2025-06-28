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
    test "element wise operation with a scalar" {
        var data: [3]f64 = createSequence(f64, 3);
        var tensor1 = Tensor(f64, .{3}).init(data[0..]);
        _ = tensor1.wise(2, (struct {
            pub fn func(a: f64, b: f64) f64 {
                return a + b;
            }
        }).func);
        try expectEqual(2, tensor1.clone(.{0}));
        try expectEqual(3, tensor1.clone(.{1}));
        try expectEqual(4, tensor1.clone(.{2}));
    }
    test "element wise operation with a tensor, in_place" {
        var data: [3]f64 = createSequence(f64, 3);
        var tensor1 = Tensor(f64, .{3}).init(data[0..]);
        var tensor2 = Tensor(f64, .{3}).init(data[0..]);
        _ = tensor1.wise(&tensor2, (struct {
            pub fn func(a: f64, b: f64) f64 {
                return a + b;
            }
        }).func);
        try expectEqual(0, tensor1.clone(.{0}));
        try expectEqual(2, tensor1.clone(.{1}));
        try expectEqual(4, tensor1.clone(.{2}));

        try expectEqual(0, tensor2.clone(.{0}));
        try expectEqual(1, tensor2.clone(.{1}));
        try expectEqual(2, tensor2.clone(.{2}));
    }
};

