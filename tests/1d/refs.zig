const std = @import("std");
const expectEqual = std.testing.expectEqual;
const expect = std.testing.expect;
const TensorView = @import("tensor").TensorView;
const Tensor = @import("tensor").Tensor;
const func = @import("tensor").func;

fn createSequence(comptime dtype: type, comptime n: usize) [n]dtype {
    var seq: [n]dtype = .{1} ** n;
    inline for (0..n) |i| {
        seq[i] = i;
    }
    return seq;
}

pub const REFS_1D = struct {
    test "reference operations - mut view" {
        var data: [3]f64 = createSequence(f64, 3);
        var tensor = Tensor(f64, .{3}).init(data[0..]);
        var mut_view = tensor.mut(.{});

        // Test that we can access the data
        try expectEqual(data[0], mut_view.scalar(.{0}).*);
        try expectEqual(data[1], mut_view.scalar(.{1}).*);
        try expectEqual(data[2], mut_view.scalar(.{2}).*);

        // Test that we can modify the data
        mut_view.scalar(.{0}).* = 100;
        mut_view.scalar(.{1}).* = 200;
        mut_view.scalar(.{2}).* = 300;

        try expectEqual(100, tensor.clone(.{0}));
        try expectEqual(200, tensor.clone(.{1}));
        try expectEqual(300, tensor.clone(.{2}));
    }

    test "reference operations - pointer to tensor" {
        var data: [3]f64 = createSequence(f64, 3);
        var tensor = Tensor(f64, .{3}).init(data[0..]);
        const tensor_ref = &tensor;

        // Test operations through reference
        try expectEqual(data[0], tensor_ref.scalar(.{0}).*);
        try expectEqual(data[1], tensor_ref.scalar(.{1}).*);
        try expectEqual(data[2], tensor_ref.scalar(.{2}).*);

        // Test that we can modify through reference
        tensor_ref.scalar(.{0}).* = 100;
        tensor_ref.scalar(.{1}).* = 200;
        tensor_ref.scalar(.{2}).* = 300;

        try expectEqual(100, tensor.clone(.{0}));
        try expectEqual(200, tensor.clone(.{1}));
        try expectEqual(300, tensor.clone(.{2}));
    }

    test "reference operations - wise through reference" {
        var data1: [3]f64 = createSequence(f64, 3);
        var data2: [3]f64 = .{ 10, 20, 30 };
        var tensor1 = Tensor(f64, .{3}).init(data1[0..]);
        var tensor2 = Tensor(f64, .{3}).init(data2[0..]);
        const tensor1_ref = &tensor1;
        const tensor2_ref = &tensor2;
        var result = Tensor(f64, .{3}).init(data1[0..]);

        tensor1_ref.wise(tensor2_ref, &result, func.addFactory(f64));

        try expectEqual(10, result.clone(.{0}));
        try expectEqual(21, result.clone(.{1}));
        try expectEqual(32, result.clone(.{2}));
    }

    test "reference operations - wiseNew through reference" {
        var data1: [3]f64 = createSequence(f64, 3);
        var data2: [3]f64 = .{ 10, 20, 30 };
        var tensor1 = Tensor(f64, .{3}).init(data1[0..]);
        var tensor2 = Tensor(f64, .{3}).init(data2[0..]);
        const tensor1_ref = &tensor1;
        const tensor2_ref = &tensor2;

        const result = tensor1_ref.wiseNew(tensor2_ref, (struct {
            pub fn func(a: f64, b: f64) f64 {
                return a + b;
            }
        }).func);

        try expectEqual(10, result.clone(.{0}));
        try expectEqual(21, result.clone(.{1}));
        try expectEqual(32, result.clone(.{2}));
    }
};
