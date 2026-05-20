const std = @import("std");
const expectEqual = std.testing.expectEqual;
const expect = std.testing.expect;
const Tensor = @import("tensor").Tensor;
const op = @import("tensor").op;

fn createSequence(comptime dtype: type, comptime n: usize) [n]dtype {
    var seq: [n]dtype = .{1} ** n;
    inline for (0..n) |i| {
        seq[i] = i;
    }
    return seq;
}

test "ref operations - scalar access" {
    var data: [8]f64 = createSequence(f64, 8);
    var tensor = Tensor(f64, 3).from(.rowMajor(.{ 2, 2, 2 }), data[0..]);
    const ref = tensor.ref(.{});

    try expectEqual(data[0], ref.scalar(.{ 0, 0, 0 }));
    try expectEqual(data[1], ref.scalar(.{ 0, 0, 1 }));
    try expectEqual(data[2], ref.scalar(.{ 0, 1, 0 }));
    try expectEqual(data[3], ref.scalar(.{ 0, 1, 1 }));
    try expectEqual(data[4], ref.scalar(.{ 1, 0, 0 }));
    try expectEqual(data[5], ref.scalar(.{ 1, 0, 1 }));
    try expectEqual(data[6], ref.scalar(.{ 1, 1, 0 }));
    try expectEqual(data[7], ref.scalar(.{ 1, 1, 1 }));
}

test "ref operations - reshape" {
    var data: [12]f64 = createSequence(f64, 12);
    var tensor = Tensor(f64, 3).from(.rowMajor(.{ 2, 3, 2 }), data[0..]);
    const ref = tensor.ref(.{});
    const reshaped = ref.reshape(.{ 3, 2, 2 });

    try expectEqual(.{ 3, 2, 2 }, reshaped.metadata.shape);
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

test "ref operations - wise" {
    var data1: [8]f64 = .{ 1, 2, 3, 4, 5, 6, 7, 8 };
    var data2: [8]f64 = .{ 10, 20, 30, 40, 50, 60, 70, 80 };
    var tensor1 = Tensor(f64, 3).from(.rowMajor(.{ 2, 2, 2 }), data1[0..]);
    var tensor2 = Tensor(f64, 3).from(.rowMajor(.{ 2, 2, 2 }), data2[0..]);
    const ref1 = tensor1.ref(.{});
    const ref2 = tensor2.ref(.{});
    var result = try Tensor(f64, 3).alloc(std.testing.allocator, .rowMajor(.{ 2, 2, 2 }));
    defer result.deinit(std.testing.allocator);

    _ = result.wise(.{ &ref1, &ref2 }, (struct {
        pub fn func(args: struct { f64, f64 }) f64 {
            const a, const b = args;
            return a + b;
        }
    }).func);

    try expectEqual(11, result.scalar(.{ 0, 0, 0 }));
    try expectEqual(22, result.scalar(.{ 0, 0, 1 }));
    try expectEqual(33, result.scalar(.{ 0, 1, 0 }));
    try expectEqual(44, result.scalar(.{ 0, 1, 1 }));
    try expectEqual(55, result.scalar(.{ 1, 0, 0 }));
    try expectEqual(66, result.scalar(.{ 1, 0, 1 }));
    try expectEqual(77, result.scalar(.{ 1, 1, 0 }));
    try expectEqual(88, result.scalar(.{ 1, 1, 1 }));
}

test "ref operations - wiseNew" {
    var data1: [8]f64 = .{ 1, 2, 3, 4, 5, 6, 7, 8 };
    var data2: [8]f64 = .{ 10, 20, 30, 40, 50, 60, 70, 80 };
    var tensor1 = Tensor(f64, 3).from(.rowMajor(.{ 2, 2, 2 }), data1[0..]);
    var tensor2 = Tensor(f64, 3).from(.rowMajor(.{ 2, 2, 2 }), data2[0..]);
    const ref1 = tensor1.ref(.{});
    const ref2 = tensor2.ref(.{});

    const result = try op.wise(std.testing.allocator, .{ &ref1, &ref2 }, (struct {
        pub fn func(args: struct { f64, f64 }) f64 {
            const a, const b = args;
            return a + b;
        }
    }).func);
    defer result.deinit(std.testing.allocator);

    try expectEqual(11, result.scalar(.{ 0, 0, 0 }));
    try expectEqual(22, result.scalar(.{ 0, 0, 1 }));
    try expectEqual(33, result.scalar(.{ 0, 1, 0 }));
    try expectEqual(44, result.scalar(.{ 0, 1, 1 }));
    try expectEqual(55, result.scalar(.{ 1, 0, 0 }));
    try expectEqual(66, result.scalar(.{ 1, 0, 1 }));
    try expectEqual(77, result.scalar(.{ 1, 1, 0 }));
    try expectEqual(88, result.scalar(.{ 1, 1, 1 }));
}

test "ref operations - slice" {
    var data: [27]f64 = createSequence(f64, 27);
    var tensor = Tensor(f64, 3).from(.rowMajor(.{ 3, 3, 3 }), data[0..]);
    const ref = tensor.ref(.{});
    const sliced = ref.slice(.{
        .{ 1, 3 },
        .{ 1, 3 },
        .{ 1, 3 },
    });

    try expectEqual(.{ 2, 2, 2 }, sliced.metadata.shape);
    try expectEqual(data[13], sliced.scalar(.{ 0, 0, 0 }));
    try expectEqual(data[14], sliced.scalar(.{ 0, 0, 1 }));
    try expectEqual(data[16], sliced.scalar(.{ 0, 1, 0 }));
    try expectEqual(data[17], sliced.scalar(.{ 0, 1, 1 }));
    try expectEqual(data[22], sliced.scalar(.{ 1, 0, 0 }));
    try expectEqual(data[23], sliced.scalar(.{ 1, 0, 1 }));
    try expectEqual(data[25], sliced.scalar(.{ 1, 1, 0 }));
    try expectEqual(data[26], sliced.scalar(.{ 1, 1, 1 }));
}

test "ref operations - transpose" {
    var data: [8]f64 = createSequence(f64, 8);
    var tensor = Tensor(f64, 3).from(.rowMajor(.{ 2, 2, 2 }), data[0..]);
    const ref = tensor.ref(.{});
    const transposed = ref.transpose(.{});

    try expectEqual(.{ 2, 2, 2 }, transposed.metadata.shape);
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

test "TensorRef operations - all const operations" {
    var data: [8]f64 = createSequence(f64, 8);
    const ref = Tensor(f64, 3).from(.rowMajor(.{ 2, 2, 2 }), data[0..]);

    // Test scalar access
    try expectEqual(data[0], ref.scalar(.{ 0, 0, 0 }));
    try expectEqual(data[1], ref.scalar(.{ 0, 0, 1 }));
    try expectEqual(data[2], ref.scalar(.{ 0, 1, 0 }));
    try expectEqual(data[3], ref.scalar(.{ 0, 1, 1 }));
    try expectEqual(data[4], ref.scalar(.{ 1, 0, 0 }));
    try expectEqual(data[5], ref.scalar(.{ 1, 0, 1 }));
    try expectEqual(data[6], ref.scalar(.{ 1, 1, 0 }));
    try expectEqual(data[7], ref.scalar(.{ 1, 1, 1 }));

    // Test reshape
    const reshaped = ref.reshape(.{8});
    try expectEqual(.{8}, reshaped.metadata.shape);
    try expectEqual(data[0], reshaped.scalar(.{0}));
    try expectEqual(data[1], reshaped.scalar(.{1}));
    try expectEqual(data[2], reshaped.scalar(.{2}));
    try expectEqual(data[3], reshaped.scalar(.{3}));
    try expectEqual(data[4], reshaped.scalar(.{4}));
    try expectEqual(data[5], reshaped.scalar(.{5}));
    try expectEqual(data[6], reshaped.scalar(.{6}));
    try expectEqual(data[7], reshaped.scalar(.{7}));
}
