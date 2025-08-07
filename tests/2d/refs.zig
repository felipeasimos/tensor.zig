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

pub const REFS_2D = struct {
    test "reference operations - mut view" {
        var data: [4]f64 = createSequence(f64, 4);
        var tensor = Tensor(f64, .{ 2, 2 }).init(data[0..]);
        var mut_view = tensor.mut(.{});

        // Test that we can access the data
        try expectEqual(data[0], mut_view.scalar(.{ 0, 0 }).*);
        try expectEqual(data[1], mut_view.scalar(.{ 0, 1 }).*);
        try expectEqual(data[2], mut_view.scalar(.{ 1, 0 }).*);
        try expectEqual(data[3], mut_view.scalar(.{ 1, 1 }).*);

        // Test that we can modify the data
        mut_view.scalar(.{ 0, 0 }).* = 100;
        mut_view.scalar(.{ 0, 1 }).* = 200;
        mut_view.scalar(.{ 1, 0 }).* = 300;
        mut_view.scalar(.{ 1, 1 }).* = 400;

        try expectEqual(100, tensor.clone(.{ 0, 0 }));
        try expectEqual(200, tensor.clone(.{ 0, 1 }));
        try expectEqual(300, tensor.clone(.{ 1, 0 }));
        try expectEqual(400, tensor.clone(.{ 1, 1 }));
    }

    test "reference operations - pointer to tensor" {
        var data: [4]f64 = createSequence(f64, 4);
        var tensor = Tensor(f64, .{ 2, 2 }).init(data[0..]);
        const tensor_ref = &tensor;

        // Test operations through reference
        try expectEqual(data[0], tensor_ref.scalar(.{ 0, 0 }).*);
        try expectEqual(data[1], tensor_ref.scalar(.{ 0, 1 }).*);
        try expectEqual(data[2], tensor_ref.scalar(.{ 1, 0 }).*);
        try expectEqual(data[3], tensor_ref.scalar(.{ 1, 1 }).*);

        // Test that we can modify through reference
        tensor_ref.scalar(.{ 0, 0 }).* = 100;
        tensor_ref.scalar(.{ 0, 1 }).* = 200;
        tensor_ref.scalar(.{ 1, 0 }).* = 300;
        tensor_ref.scalar(.{ 1, 1 }).* = 400;

        try expectEqual(100, tensor.clone(.{ 0, 0 }));
        try expectEqual(200, tensor.clone(.{ 0, 1 }));
        try expectEqual(300, tensor.clone(.{ 1, 0 }));
        try expectEqual(400, tensor.clone(.{ 1, 1 }));
    }

    test "reference operations - wise through reference" {
        var data1: [4]f64 = .{ 1, 2, 3, 4 };
        var data2: [4]f64 = .{ 10, 20, 30, 40 };
        var tensor1 = Tensor(f64, .{ 2, 2 }).init(data1[0..]);
        var tensor2 = Tensor(f64, .{ 2, 2 }).init(data2[0..]);
        const tensor1_ref = &tensor1;
        const tensor2_ref = &tensor2;
        var result = Tensor(f64, .{ 2, 2 }).init(data1[0..]);

        tensor1_ref.wise(.{tensor2_ref}, (struct {
            pub fn func(a: f64, b: f64) f64 {
                return a + b;
            }
        }).func);

        try expectEqual(11, result.clone(.{ 0, 0 }));
        try expectEqual(22, result.clone(.{ 0, 1 }));
        try expectEqual(33, result.clone(.{ 1, 0 }));
        try expectEqual(44, result.clone(.{ 1, 1 }));
    }

    test "reference operations - wiseNew through reference" {
        var data1: [4]f64 = .{ 1, 2, 3, 4 };
        var data2: [4]f64 = .{ 10, 20, 30, 40 };
        var tensor1 = Tensor(f64, .{ 2, 2 }).init(data1[0..]);
        var tensor2 = Tensor(f64, .{ 2, 2 }).init(data2[0..]);
        const tensor1_ref = &tensor1;
        const tensor2_ref = &tensor2;

        const result = tensor1_ref.wiseNew(tensor2_ref, (struct {
            pub fn func(a: f64, b: f64) f64 {
                return a + b;
            }
        }).func);

        try expectEqual(11, result.clone(.{ 0, 0 }));
        try expectEqual(22, result.clone(.{ 0, 1 }));
        try expectEqual(33, result.clone(.{ 1, 0 }));
        try expectEqual(44, result.clone(.{ 1, 1 }));
    }
};
