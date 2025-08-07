const std = @import("std");
const expectEqual = std.testing.expectEqual;
const expect = std.testing.expect;
const TensorView = @import("tensor").TensorView;
const Tensor = @import("tensor").Tensor;
const op = @import("tensor").op;

fn createSequence(comptime dtype: type, comptime n: usize) [n]dtype {
    var seq: [n]dtype = .{1} ** n;
    inline for (0..n) |i| {
        seq[i] = i;
    }
    return seq;
}

pub const VIEWS_3D = struct {
    test "view operations - scalar access" {
        var data: [8]f64 = createSequence(f64, 8);
        var tensor = Tensor(f64, .{ 2, 2, 2 }).init(data[0..]);
        const view = tensor.view(.{});

        try expectEqual(data[0], view.scalar(.{ 0, 0, 0 }));
        try expectEqual(data[1], view.scalar(.{ 0, 0, 1 }));
        try expectEqual(data[2], view.scalar(.{ 0, 1, 0 }));
        try expectEqual(data[3], view.scalar(.{ 0, 1, 1 }));
        try expectEqual(data[4], view.scalar(.{ 1, 0, 0 }));
        try expectEqual(data[5], view.scalar(.{ 1, 0, 1 }));
        try expectEqual(data[6], view.scalar(.{ 1, 1, 0 }));
        try expectEqual(data[7], view.scalar(.{ 1, 1, 1 }));
    }

    test "view operations - clone" {
        var data: [8]f64 = createSequence(f64, 8);
        var tensor = Tensor(f64, .{ 2, 2, 2 }).init(data[0..]);
        const view = tensor.view(.{});

        try expectEqual(data[0], view.clone(.{ 0, 0, 0 }));
        try expectEqual(data[1], view.clone(.{ 0, 0, 1 }));
        try expectEqual(data[2], view.clone(.{ 0, 1, 0 }));
        try expectEqual(data[3], view.clone(.{ 0, 1, 1 }));
        try expectEqual(data[4], view.clone(.{ 1, 0, 0 }));
        try expectEqual(data[5], view.clone(.{ 1, 0, 1 }));
        try expectEqual(data[6], view.clone(.{ 1, 1, 0 }));
        try expectEqual(data[7], view.clone(.{ 1, 1, 1 }));
    }

    test "view operations - reshape" {
        var data: [12]f64 = createSequence(f64, 12);
        var tensor = Tensor(f64, .{ 2, 3, 2 }).init(data[0..]);
        const view = tensor.view(.{});
        const reshaped = view.reshape(.{ 3, 2, 2 });

        try expectEqual(.{ 3, 2, 2 }, reshaped.shape);
        try expectEqual(data[0], reshaped.scalar(.{ 0, 0, 0 }));
        try expectEqual(data[1], reshaped.scalar(.{ 0, 0, 1 }));
        try expectEqual(data[2], reshaped.scalar(.{ 0, 1, 0 }));
        try expectEqual(data[3], reshaped.scalar(.{ 0, 1, 1 }));
        try expectEqual(data[4], reshaped.scalar(.{ 1, 0, 0 }));
        try expectEqual(data[5], reshaped.scalar(.{ 1, 0, 1 }));
        try expectEqual(data[6], reshaped.scalar(.{ 1, 1, 0 }));
        try expectEqual(data[7], reshaped.scalar(.{ 1, 1, 1 }));
        try expectEqual(data[8], reshaped.scalar(.{ 2, 0, 0 }));
        try expectEqual(data[9], reshaped.scalar(.{ 2, 0, 1 }));
        try expectEqual(data[10], reshaped.scalar(.{ 2, 1, 0 }));
        try expectEqual(data[11], reshaped.scalar(.{ 2, 1, 1 }));
    }

    test "view operations - wise" {
        var data1: [8]f64 = .{ 1, 2, 3, 4, 5, 6, 7, 8 };
        var data2: [8]f64 = .{ 10, 20, 30, 40, 50, 60, 70, 80 };
        var tensor1 = Tensor(f64, .{ 2, 2, 2 }).init(data1[0..]);
        var tensor2 = Tensor(f64, .{ 2, 2, 2 }).init(data2[0..]);
        const view1 = tensor1.view(.{});
        const view2 = tensor2.view(.{});
        var result = Tensor(f64, .{ 2, 2, 2 }).init(data1[0..]);

        view1.wise(.{&view2}, (struct {
            pub fn func(a: f64, b: f64) f64 {
                return a + b;
            }
        }).func);

        try expectEqual(11, result.clone(.{ 0, 0, 0 }));
        try expectEqual(22, result.clone(.{ 0, 0, 1 }));
        try expectEqual(33, result.clone(.{ 0, 1, 0 }));
        try expectEqual(44, result.clone(.{ 0, 1, 1 }));
        try expectEqual(55, result.clone(.{ 1, 0, 0 }));
        try expectEqual(66, result.clone(.{ 1, 0, 1 }));
        try expectEqual(77, result.clone(.{ 1, 1, 0 }));
        try expectEqual(88, result.clone(.{ 1, 1, 1 }));
    }

    test "view operations - wiseNew" {
        var data1: [8]f64 = .{ 1, 2, 3, 4, 5, 6, 7, 8 };
        var data2: [8]f64 = .{ 10, 20, 30, 40, 50, 60, 70, 80 };
        var tensor1 = Tensor(f64, .{ 2, 2, 2 }).init(data1[0..]);
        var tensor2 = Tensor(f64, .{ 2, 2, 2 }).init(data2[0..]);
        const view1 = tensor1.view(.{});
        const view2 = tensor2.view(.{});

        const result = op.wise(.{ &view1, &view2 }, (struct {
            pub fn func(a: f64, b: f64) f64 {
                return a + b;
            }
        }).func);

        try expectEqual(11, result.clone(.{ 0, 0, 0 }));
        try expectEqual(22, result.clone(.{ 0, 0, 1 }));
        try expectEqual(33, result.clone(.{ 0, 1, 0 }));
        try expectEqual(44, result.clone(.{ 0, 1, 1 }));
        try expectEqual(55, result.clone(.{ 1, 0, 0 }));
        try expectEqual(66, result.clone(.{ 1, 0, 1 }));
        try expectEqual(77, result.clone(.{ 1, 1, 0 }));
        try expectEqual(88, result.clone(.{ 1, 1, 1 }));
    }

    test "view operations - slice" {
        var data: [27]f64 = createSequence(f64, 27);
        var tensor = Tensor(f64, .{ 3, 3, 3 }).init(data[0..]);
        const view = tensor.view(.{});
        const sliced = view.slice(.{
            .{ 1, 3 },
            .{ 1, 3 },
            .{ 1, 3 },
        });

        try expectEqual(.{ 2, 2, 2 }, sliced.shape);
        try expectEqual(data[13], sliced.scalar(.{ 0, 0, 0 }));
        try expectEqual(data[14], sliced.scalar(.{ 0, 0, 1 }));
        try expectEqual(data[16], sliced.scalar(.{ 0, 1, 0 }));
        try expectEqual(data[17], sliced.scalar(.{ 0, 1, 1 }));
        try expectEqual(data[22], sliced.scalar(.{ 1, 0, 0 }));
        try expectEqual(data[23], sliced.scalar(.{ 1, 0, 1 }));
        try expectEqual(data[25], sliced.scalar(.{ 1, 1, 0 }));
        try expectEqual(data[26], sliced.scalar(.{ 1, 1, 1 }));
    }

    test "view operations - transpose" {
        var data: [8]f64 = createSequence(f64, 8);
        var tensor = Tensor(f64, .{ 2, 2, 2 }).init(data[0..]);
        const view = tensor.view(.{});
        const transposed = view.transpose(.{});

        try expectEqual(.{ 2, 2, 2 }, transposed.shape);
        // Transpose swaps the last two dimensions
        try expectEqual(data[0], transposed.scalar(.{ 0, 0, 0 }));
        try expectEqual(data[2], transposed.scalar(.{ 0, 0, 1 }));
        try expectEqual(data[1], transposed.scalar(.{ 0, 1, 0 }));
        try expectEqual(data[3], transposed.scalar(.{ 0, 1, 1 }));
        try expectEqual(data[4], transposed.scalar(.{ 1, 0, 0 }));
        try expectEqual(data[6], transposed.scalar(.{ 1, 0, 1 }));
        try expectEqual(data[5], transposed.scalar(.{ 1, 1, 0 }));
        try expectEqual(data[7], transposed.scalar(.{ 1, 1, 1 }));
    }

    test "TensorView operations - all const operations" {
        var data: [8]f64 = createSequence(f64, 8);
        const view = TensorView(f64, .{ 2, 2, 2 }).init(data[0..]);

        // Test scalar access
        try expectEqual(data[0], view.scalar(.{ 0, 0, 0 }));
        try expectEqual(data[1], view.scalar(.{ 0, 0, 1 }));
        try expectEqual(data[2], view.scalar(.{ 0, 1, 0 }));
        try expectEqual(data[3], view.scalar(.{ 0, 1, 1 }));
        try expectEqual(data[4], view.scalar(.{ 1, 0, 0 }));
        try expectEqual(data[5], view.scalar(.{ 1, 0, 1 }));
        try expectEqual(data[6], view.scalar(.{ 1, 1, 0 }));
        try expectEqual(data[7], view.scalar(.{ 1, 1, 1 }));

        // Test clone
        try expectEqual(data[0], view.clone(.{ 0, 0, 0 }));
        try expectEqual(data[1], view.clone(.{ 0, 0, 1 }));
        try expectEqual(data[2], view.clone(.{ 0, 1, 0 }));
        try expectEqual(data[3], view.clone(.{ 0, 1, 1 }));
        try expectEqual(data[4], view.clone(.{ 1, 0, 0 }));
        try expectEqual(data[5], view.clone(.{ 1, 0, 1 }));
        try expectEqual(data[6], view.clone(.{ 1, 1, 0 }));
        try expectEqual(data[7], view.clone(.{ 1, 1, 1 }));

        // Test reshape
        const reshaped = view.reshape(.{8});
        try expectEqual(.{8}, reshaped.shape);
        try expectEqual(data[0], reshaped.scalar(.{0}));
        try expectEqual(data[1], reshaped.scalar(.{1}));
        try expectEqual(data[2], reshaped.scalar(.{2}));
        try expectEqual(data[3], reshaped.scalar(.{3}));
        try expectEqual(data[4], reshaped.scalar(.{4}));
        try expectEqual(data[5], reshaped.scalar(.{5}));
        try expectEqual(data[6], reshaped.scalar(.{6}));
        try expectEqual(data[7], reshaped.scalar(.{7}));
    }
};
