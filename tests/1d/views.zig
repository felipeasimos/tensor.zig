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

pub const VIEWS_1D = struct {
    test "view operations - scalar access" {
        var data: [3]f64 = createSequence(f64, 3);
        var tensor = Tensor(f64, .{3}).init(data[0..]);
        const view = tensor.view(.{});

        try expectEqual(data[0], view.scalar(.{0}));
        try expectEqual(data[1], view.scalar(.{1}));
        try expectEqual(data[2], view.scalar(.{2}));
    }

    test "view operations - clone" {
        var data: [3]f64 = createSequence(f64, 3);
        var tensor = Tensor(f64, .{3}).init(data[0..]);
        const view = tensor.view(.{});

        try expectEqual(data[0], view.clone(.{0}));
        try expectEqual(data[1], view.clone(.{1}));
        try expectEqual(data[2], view.clone(.{2}));
    }

    test "view operations - reshape" {
        var data: [6]f64 = createSequence(f64, 6);
        var tensor = Tensor(f64, .{6}).init(data[0..]);
        const view = tensor.view(.{});
        const reshaped = view.reshape(.{ 2, 3 });

        try expectEqual(.{ 2, 3 }, reshaped.shape);
        try expectEqual(data[0], reshaped.scalar(.{ 0, 0 }));
        try expectEqual(data[1], reshaped.scalar(.{ 0, 1 }));
        try expectEqual(data[2], reshaped.scalar(.{ 0, 2 }));
        try expectEqual(data[3], reshaped.scalar(.{ 1, 0 }));
        try expectEqual(data[4], reshaped.scalar(.{ 1, 1 }));
        try expectEqual(data[5], reshaped.scalar(.{ 1, 2 }));
    }

    test "view operations - wise" {
        var data1: [3]f64 = createSequence(f64, 3);
        var data2: [3]f64 = .{ 10, 20, 30 };
        var tensor1 = Tensor(f64, .{3}).init(data1[0..]);
        var tensor2 = Tensor(f64, .{3}).init(data2[0..]);
        const view1 = tensor1.view(.{});
        const view2 = tensor2.view(.{});
        var result = Tensor(f64, .{3}).init(data1[0..]);

        view1.wise(.{&view2}, (struct {
            pub fn func(a: f64, b: f64) f64 {
                return a + b;
            }
        }).func);

        try expectEqual(10, result.clone(.{0}));
        try expectEqual(21, result.clone(.{1}));
        try expectEqual(32, result.clone(.{2}));
    }

    test "view operations - wiseNew" {
        var data1: [3]f64 = createSequence(f64, 3);
        var data2: [3]f64 = .{ 10, 20, 30 };
        var tensor1 = Tensor(f64, .{3}).init(data1[0..]);
        var tensor2 = Tensor(f64, .{3}).init(data2[0..]);
        const view1 = tensor1.view(.{});
        const view2 = tensor2.view(.{});

        const result = view1.wiseNew(&view2, (struct {
            pub fn func(a: f64, b: f64) f64 {
                return a + b;
            }
        }).func);

        try expectEqual(10, result.clone(.{0}));
        try expectEqual(21, result.clone(.{1}));
        try expectEqual(32, result.clone(.{2}));
    }

    test "view operations - slice" {
        var data: [6]f64 = createSequence(f64, 6);
        var tensor = Tensor(f64, .{6}).init(data[0..]);
        const view = tensor.view(.{});
        const sliced = view.slice(.{.{ 1, 4 }});

        try expectEqual(.{3}, sliced.shape);
        try expectEqual(data[1], sliced.scalar(.{0}));
        try expectEqual(data[2], sliced.scalar(.{1}));
        try expectEqual(data[3], sliced.scalar(.{2}));
    }

    test "TensorView operations - all const operations" {
        var data: [3]f64 = createSequence(f64, 3);
        const view = TensorView(f64, .{3}).init(data[0..]);

        // Test scalar access
        try expectEqual(data[0], view.scalar(.{0}));
        try expectEqual(data[1], view.scalar(.{1}));
        try expectEqual(data[2], view.scalar(.{2}));

        // Test clone
        try expectEqual(data[0], view.clone(.{0}));
        try expectEqual(data[1], view.clone(.{1}));
        try expectEqual(data[2], view.clone(.{2}));

        // Test reshape
        const reshaped = view.reshape(.{ 1, 3 });
        try expectEqual(.{ 1, 3 }, reshaped.shape);
        try expectEqual(data[0], reshaped.scalar(.{ 0, 0 }));
        try expectEqual(data[1], reshaped.scalar(.{ 0, 1 }));
        try expectEqual(data[2], reshaped.scalar(.{ 0, 2 }));
    }
};
