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
    test "broadcast 1D size 1 to size 3" {
        var data: [1]f64 = .{42};
        var tensor = TensorView(f64, .{1}).init(data[0..]);
        const broadcasted = tensor.broadcast(.{3});

        try expectEqual(.{3}, broadcasted.shape);
        try expectEqual(.{0}, broadcasted.strides);
        try expectEqual(42, broadcasted.scalar(.{0}));
        try expectEqual(42, broadcasted.scalar(.{1}));
        try expectEqual(42, broadcasted.scalar(.{2}));
    }
    test "broadcast 1D to 2D" {
        var data: [1]f64 = .{5};
        var tensor = TensorView(f64, .{1}).init(data[0..]);
        const broadcasted = tensor.broadcast(.{ 2, 3 });

        try expectEqual(.{ 2, 3 }, broadcasted.shape);
        try expectEqual(.{ 0, 0 }, broadcasted.strides);
        try expectEqual(5, broadcasted.scalar(.{ 0, 0 }));
        try expectEqual(5, broadcasted.scalar(.{ 0, 1 }));
        try expectEqual(5, broadcasted.scalar(.{ 0, 2 }));
        try expectEqual(5, broadcasted.scalar(.{ 1, 0 }));
        try expectEqual(5, broadcasted.scalar(.{ 1, 1 }));
        try expectEqual(5, broadcasted.scalar(.{ 1, 2 }));
    }
    test "broadcast 1D to 3D" {
        var data: [3]f64 = .{ 1, 2, 3 };
        var tensor = TensorView(f64, .{3}).init(data[0..]);
        const broadcasted = tensor.broadcast(.{ 2, 4, 3 });

        try expectEqual(.{ 2, 4, 3 }, broadcasted.shape);
        try expectEqual(.{ 0, 0, 1 }, broadcasted.strides);
        try expectEqual(1, broadcasted.scalar(.{ 0, 0, 0 }));
        try expectEqual(2, broadcasted.scalar(.{ 0, 0, 1 }));
        try expectEqual(3, broadcasted.scalar(.{ 0, 0, 2 }));
        try expectEqual(1, broadcasted.scalar(.{ 1, 3, 0 }));
        try expectEqual(2, broadcasted.scalar(.{ 1, 3, 1 }));
        try expectEqual(3, broadcasted.scalar(.{ 1, 3, 2 }));
    }
    test "iterator basic 1D" {
        var data: [3]f64 = .{ 5, 10, 15 };
        var tensor = TensorView(f64, .{3}).init(data[0..]);
        var iter = tensor.iter();

        var count: usize = 0;
        while (iter.next()) |item| {
            count += 1;
            switch (count) {
                1 => {
                    try expectEqual(.{0}, item.indices);
                    try expectEqual(5, item.value);
                },
                2 => {
                    try expectEqual(.{1}, item.indices);
                    try expectEqual(10, item.value);
                },
                3 => {
                    try expectEqual(.{2}, item.indices);
                    try expectEqual(15, item.value);
                },
                else => unreachable,
            }
        }
        try expectEqual(3, count);
    }
};
