const std = @import("std");
const expectEqual = std.testing.expectEqual;
const expect = std.testing.expect;
const TensorRef = @import("tensor").TensorRef;
const Tensor = @import("tensor").Tensor;
const op = @import("tensor").op;

fn createSequence(comptime dtype: type, comptime n: usize) [n]dtype {
    var seq: [n]dtype = .{1} ** n;
    inline for (0..n) |i| {
        seq[i] = i;
    }
    return seq;
}

pub const refS_1D = struct {
    test "ref operations - scalar access" {
        var data: [3]f64 = createSequence(f64, 3);
        var tensor = Tensor(f64, .{3}).init(data[0..]);
        const ref = tensor.ref(.{});

        try expectEqual(data[0], ref.scalar(.{0}));
        try expectEqual(data[1], ref.scalar(.{1}));
        try expectEqual(data[2], ref.scalar(.{2}));
    }

    test "ref operations - clone" {
        var data: [3]f64 = createSequence(f64, 3);
        var tensor = Tensor(f64, .{3}).init(data[0..]);
        const ref = tensor.ref(.{});

        try expectEqual(data[0], ref.clone(.{0}));
        try expectEqual(data[1], ref.clone(.{1}));
        try expectEqual(data[2], ref.clone(.{2}));
    }

    test "ref operations - reshape" {
        var data: [6]f64 = createSequence(f64, 6);
        var tensor = Tensor(f64, .{6}).init(data[0..]);
        const ref = tensor.ref(.{});
        const reshaped = ref.reshape(.{ 2, 3 });

        try expectEqual(.{ 2, 3 }, reshaped.shape);
        try expectEqual(data[0], reshaped.scalar(.{ 0, 0 }));
        try expectEqual(data[1], reshaped.scalar(.{ 0, 1 }));
        try expectEqual(data[2], reshaped.scalar(.{ 0, 2 }));
        try expectEqual(data[3], reshaped.scalar(.{ 1, 0 }));
        try expectEqual(data[4], reshaped.scalar(.{ 1, 1 }));
        try expectEqual(data[5], reshaped.scalar(.{ 1, 2 }));
    }

    test "ref operations - wise" {
        var data1: [3]f64 = createSequence(f64, 3);
        var data2: [3]f64 = .{ 10, 20, 30 };
        var tensor1 = Tensor(f64, .{3}).init(data1[0..]);
        var tensor2 = Tensor(f64, .{3}).init(data2[0..]);
        const ref1 = tensor1.ref(.{});
        const ref2 = tensor2.ref(.{});
        var result = Tensor(f64, .{3}).init(data1[0..]);

        result.wise(.{ &ref1, &ref2 }, (struct {
            pub fn func(args: struct { f64, f64 }) f64 {
                const a, const b = args;
                return a + b;
            }
        }).func);

        try expectEqual(10, result.clone(.{0}));
        try expectEqual(21, result.clone(.{1}));
        try expectEqual(32, result.clone(.{2}));
    }

    test "ref operations - wiseNew" {
        var data1: [3]f64 = createSequence(f64, 3);
        var data2: [3]f64 = .{ 10, 20, 30 };
        var tensor1 = Tensor(f64, .{3}).init(data1[0..]);
        var tensor2 = Tensor(f64, .{3}).init(data2[0..]);
        const ref1 = tensor1.ref(.{});
        const ref2 = tensor2.ref(.{});

        const result = op.wise(.{ &ref1, &ref2 }, (struct {
            pub fn func(args: struct { f64, f64 }) f64 {
                const a, const b = args;
                return a + b;
            }
        }).func);

        try expectEqual(10, result.clone(.{0}));
        try expectEqual(21, result.clone(.{1}));
        try expectEqual(32, result.clone(.{2}));
    }

    test "ref operations - slice" {
        var data: [6]f64 = createSequence(f64, 6);
        var tensor = Tensor(f64, .{6}).init(data[0..]);
        const ref = tensor.ref(.{});
        const sliced = ref.slice(.{.{ 1, 4 }});

        try expectEqual(.{3}, sliced.shape);
        try expectEqual(data[1], sliced.scalar(.{0}));
        try expectEqual(data[2], sliced.scalar(.{1}));
        try expectEqual(data[3], sliced.scalar(.{2}));
    }

    test "TensorRef operations - all const operations" {
        var data: [3]f64 = createSequence(f64, 3);
        const ref = TensorRef(f64, .{3}).init(data[0..]);

        // Test scalar access
        try expectEqual(data[0], ref.scalar(.{0}));
        try expectEqual(data[1], ref.scalar(.{1}));
        try expectEqual(data[2], ref.scalar(.{2}));

        // Test clone
        try expectEqual(data[0], ref.clone(.{0}));
        try expectEqual(data[1], ref.clone(.{1}));
        try expectEqual(data[2], ref.clone(.{2}));

        // Test reshape
        const reshaped = ref.reshape(.{ 1, 3 });
        try expectEqual(.{ 1, 3 }, reshaped.shape);
        try expectEqual(data[0], reshaped.scalar(.{ 0, 0 }));
        try expectEqual(data[1], reshaped.scalar(.{ 0, 1 }));
        try expectEqual(data[2], reshaped.scalar(.{ 0, 2 }));
    }
};
