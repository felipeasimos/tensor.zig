const std = @import("std");
const expectEqual = std.testing.expectEqual;
const expect = std.testing.expect;
const expectApproxEqAbs = std.testing.expectApproxEqAbs;
const Tensor = @import("tensor").Tensor;
const op = @import("tensor").op;

fn createSequence(comptime dtype: type, comptime n: usize) [n]dtype {
    var seq: [n]dtype = .{1} ** n;
    inline for (0..n) |i| {
        seq[i] = i;
    }
    return seq;
}

const tolerance = 10e-100;

test "ref operations - scalar access" {
    var data: [4]f64 = createSequence(f64, 4);
    var tensor = Tensor(f64, 2).from(.rowMajor(.{ 2, 2 }), data[0..]);
    const ref = tensor.ref(.{});

    try expectEqual(data[0], ref.scalar(.{ 0, 0 }));
    try expectEqual(data[1], ref.scalar(.{ 0, 1 }));
    try expectEqual(data[2], ref.scalar(.{ 1, 0 }));
    try expectEqual(data[3], ref.scalar(.{ 1, 1 }));
}

test "ref operations - reshape" {
    var data: [6]f64 = createSequence(f64, 6);
    var tensor = Tensor(f64, 2).from(.rowMajor(.{ 2, 3 }), data[0..]);
    const ref = tensor.ref(.{});
    const reshaped = ref.reshape(.{ 3, 2 });

    try expectEqual(.{ 3, 2 }, reshaped.metadata.shape);
    try expectEqual(data[0], reshaped.scalar(.{ 0, 0 }));
    try expectEqual(data[1], reshaped.scalar(.{ 0, 1 }));
    try expectEqual(data[2], reshaped.scalar(.{ 1, 0 }));
    try expectEqual(data[3], reshaped.scalar(.{ 1, 1 }));
    try expectEqual(data[4], reshaped.scalar(.{ 2, 0 }));
    try expectEqual(data[5], reshaped.scalar(.{ 2, 1 }));
}

test "ref operations - wise" {
    var data1: [4]f64 = .{ 1, 2, 3, 4 };
    var data2: [4]f64 = .{ 10, 20, 30, 40 };
    var tensor1 = Tensor(f64, 2).from(.rowMajor(.{ 2, 2 }), data1[0..]);
    var tensor2 = Tensor(f64, 2).from(.rowMajor(.{ 2, 2 }), data2[0..]);
    const ref1 = tensor1.ref(.{});
    const ref2 = tensor2.ref(.{});
    var result = try Tensor(f64, 2).alloc(std.testing.allocator, .rowMajor(.{ 2, 2 }));
    defer result.deinit(std.testing.allocator);

    result.wise(.{ &ref1, &ref2 }, (struct {
        pub fn func(args: struct { f64, f64 }) f64 {
            const a, const b = args;
            return a + b;
        }
    }).func);

    try expectEqual(11, result.scalar(.{ 0, 0 }));
    try expectEqual(22, result.scalar(.{ 0, 1 }));
    try expectEqual(33, result.scalar(.{ 1, 0 }));
    try expectEqual(44, result.scalar(.{ 1, 1 }));
}

test "ref operations - wiseNew" {
    var data1: [4]f64 = .{ 1, 2, 3, 4 };
    var data2: [4]f64 = .{ 10, 20, 30, 40 };
    var tensor1 = Tensor(f64, 2).from(.rowMajor(.{ 2, 2 }), data1[0..]);
    var tensor2 = Tensor(f64, 2).from(.rowMajor(.{ 2, 2 }), data2[0..]);
    const ref1 = tensor1.ref(.{});
    const ref2 = tensor2.ref(.{});

    const result = try op.wise(std.testing.allocator, .{ &ref1, &ref2 }, (struct {
        pub fn func(args: struct { f64, f64 }) f64 {
            const a, const b = args;
            return a + b;
        }
    }).func);
    defer result.deinit(std.testing.allocator);

    try expectEqual(11, result.scalar(.{ 0, 0 }));
    try expectEqual(22, result.scalar(.{ 0, 1 }));
    try expectEqual(33, result.scalar(.{ 1, 0 }));
    try expectEqual(44, result.scalar(.{ 1, 1 }));
}

test "ref operations - slice" {
    var data: [9]f64 = createSequence(f64, 9);
    var tensor = Tensor(f64, 2).from(.rowMajor(.{ 3, 3 }), data[0..]);
    const ref = tensor.ref(.{});
    const sliced = ref.slice(.{
        .{ 1, 3 },
        .{ 1, 3 },
    });

    try expectEqual(.{ 2, 2 }, sliced.metadata.shape);
    try expectEqual(data[4], sliced.scalar(.{ 0, 0 }));
    try expectEqual(data[5], sliced.scalar(.{ 0, 1 }));
    try expectEqual(data[7], sliced.scalar(.{ 1, 0 }));
    try expectEqual(data[8], sliced.scalar(.{ 1, 1 }));
}

test "ref operations - transpose" {
    var data: [6]f64 = createSequence(f64, 6);
    var tensor = Tensor(f64, 2).from(.rowMajor(.{ 2, 3 }), data[0..]);
    const ref = tensor.ref(.{});
    const transposed = ref.transpose(.{});

    try expectEqual(.{ 3, 2 }, transposed.metadata.shape);
    try expectEqual(data[0], transposed.scalar(.{ 0, 0 }));
    try expectEqual(data[3], transposed.scalar(.{ 0, 1 }));
    try expectEqual(data[1], transposed.scalar(.{ 1, 0 }));
    try expectEqual(data[4], transposed.scalar(.{ 1, 1 }));
    try expectEqual(data[2], transposed.scalar(.{ 2, 0 }));
    try expectEqual(data[5], transposed.scalar(.{ 2, 1 }));
}

test "ref operations - matmul" {
    var data1: [6]f64 = createSequence(f64, 6);
    var data2: [6]f64 = createSequence(f64, 6);
    var tensor1 = Tensor(f64, 2).from(.rowMajor(.{ 2, 3 }), data1[0..]);
    var tensor2 = Tensor(f64, 2).from(.rowMajor(.{ 3, 2 }), data2[0..]);
    const ref1 = tensor1.ref(.{});
    const ref2 = tensor2.ref(.{});
    var result_data: [6]f64 = .{0} ** 6;
    var result = Tensor(f64, 2).from(.rowMajor(.{ 2, 2 }), result_data[0..]);

    try result.matmul(std.testing.io, &ref1, &ref2);

    try expectEqual(10, result.scalar(.{ 0, 0 }));
    try expectEqual(13, result.scalar(.{ 0, 1 }));
    try expectEqual(28, result.scalar(.{ 1, 0 }));
    try expectEqual(40, result.scalar(.{ 1, 1 }));
}

test "ref operations - matmulNew" {
    var data1: [6]f64 = createSequence(f64, 6);
    var data2: [6]f64 = createSequence(f64, 6);
    var tensor1 = Tensor(f64, 2).from(.rowMajor(.{ 2, 3 }), data1[0..]);
    var tensor2 = Tensor(f64, 2).from(.rowMajor(.{ 3, 2 }), data2[0..]);
    const ref1 = tensor1.ref(.{});
    const ref2 = tensor2.ref(.{});

    const result = try op.matmul(std.testing.allocator, std.testing.io, &ref1, &ref2);
    defer result.deinit(std.testing.allocator);

    try expectEqual(.{ 2, 2 }, result.metadata.shape);
    try expectEqual(10, result.scalar(.{ 0, 0 }));
    try expectEqual(13, result.scalar(.{ 0, 1 }));
    try expectEqual(28, result.scalar(.{ 1, 0 }));
    try expectEqual(40, result.scalar(.{ 1, 1 }));
}

test "Tensor operations - all const operations" {
    var data: [4]f64 = createSequence(f64, 4);
    const ref = Tensor(f64, 2).from(.rowMajor(.{ 2, 2 }), data[0..]);

    // Test scalar access
    try expectEqual(data[0], ref.scalar(.{ 0, 0 }));
    try expectEqual(data[1], ref.scalar(.{ 0, 1 }));
    try expectEqual(data[2], ref.scalar(.{ 1, 0 }));
    try expectEqual(data[3], ref.scalar(.{ 1, 1 }));

    // Test reshape
    const reshaped = ref.reshape(.{4});
    try expectEqual(.{4}, reshaped.metadata.shape);
    try expectEqual(data[0], reshaped.scalar(.{0}));
    try expectEqual(data[1], reshaped.scalar(.{1}));
    try expectEqual(data[2], reshaped.scalar(.{2}));
    try expectEqual(data[3], reshaped.scalar(.{3}));
}

test "ref operations - matmulNew rectangular" {
    var data1 = [_]f64{
        1, 2,
        3, 4,
        5, 6,
    };
    var data2 = [_]f64{
        7,  8,  9,  10,
        11, 12, 13, 14,
    };

    var tensor1 = Tensor(f64, 2).from(.rowMajor(.{ 3, 2 }), data1[0..]);
    var tensor2 = Tensor(f64, 2).from(.rowMajor(.{ 2, 4 }), data2[0..]);

    const ref1 = tensor1.ref(.{});
    const ref2 = tensor2.ref(.{});

    const result = try op.matmul(std.testing.allocator, std.testing.io, &ref1, &ref2);
    defer result.deinit(std.testing.allocator);

    try expectEqual(.{ 3, 4 }, result.metadata.shape);

    try expectApproxEqAbs(29, result.scalar(.{ 0, 0 }), tolerance);
    try expectApproxEqAbs(32, result.scalar(.{ 0, 1 }), tolerance);
    try expectApproxEqAbs(35, result.scalar(.{ 0, 2 }), tolerance);
    try expectApproxEqAbs(38, result.scalar(.{ 0, 3 }), tolerance);

    try expectApproxEqAbs(65, result.scalar(.{ 1, 0 }), tolerance);
    try expectApproxEqAbs(72, result.scalar(.{ 1, 1 }), tolerance);
    try expectApproxEqAbs(79, result.scalar(.{ 1, 2 }), tolerance);
    try expectApproxEqAbs(86, result.scalar(.{ 1, 3 }), tolerance);

    try expectApproxEqAbs(101, result.scalar(.{ 2, 0 }), tolerance);
    try expectApproxEqAbs(112, result.scalar(.{ 2, 1 }), tolerance);
    try expectApproxEqAbs(123, result.scalar(.{ 2, 2 }), tolerance);
    try expectApproxEqAbs(134, result.scalar(.{ 2, 3 }), tolerance);
}

test "ref operations - matmulNew identity" {
    var data1 = [_]f64{
        1, 2, 3,
        4, 5, 6,
        7, 8, 9,
    };

    var identity = [_]f64{
        1, 0, 0,
        0, 1, 0,
        0, 0, 1,
    };

    var tensor1 = Tensor(f64, 2).from(.rowMajor(.{ 3, 3 }), data1[0..]);
    var tensor2 = Tensor(f64, 2).from(.rowMajor(.{ 3, 3 }), identity[0..]);

    const ref1 = tensor1.ref(.{});
    const ref2 = tensor2.ref(.{});

    const result = try op.matmul(std.testing.allocator, std.testing.io, &ref1, &ref2);
    defer result.deinit(std.testing.allocator);

    try expectEqual(.{ 3, 3 }, result.metadata.shape);

    inline for (0..3) |i| {
        inline for (0..3) |j| {
            try expectApproxEqAbs(
                tensor1.scalar(.{ i, j }),
                result.scalar(.{ i, j }),
                tolerance,
            );
        }
    }
}

test "ref operations - matmulNew zero matrix" {
    var data1 = [_]f64{
        1, 2, 3,
        4, 5, 6,
    };

    var zeros = [_]f64{0} ** 12;

    var tensor1 = Tensor(f64, 2).from(.rowMajor(.{ 2, 3 }), data1[0..]);
    var tensor2 = Tensor(f64, 2).from(.rowMajor(.{ 3, 4 }), zeros[0..]);

    const ref1 = tensor1.ref(.{});
    const ref2 = tensor2.ref(.{});

    const result = try op.matmul(std.testing.allocator, std.testing.io, &ref1, &ref2);
    defer result.deinit(std.testing.allocator);

    try expectEqual(.{ 2, 4 }, result.metadata.shape);

    inline for (0..2) |i| {
        inline for (0..4) |j| {
            try std.testing.expectApproxEqAbs(0, result.scalar(.{ i, j }), tolerance);
        }
    }
}

test "ref operations - matmulNew 1x1" {
    var data1 = [_]f64{6};
    var data2 = [_]f64{7};

    var tensor1 = Tensor(f64, 2).from(.rowMajor(.{ 1, 1 }), data1[0..]);
    var tensor2 = Tensor(f64, 2).from(.rowMajor(.{ 1, 1 }), data2[0..]);

    const ref1 = tensor1.ref(.{});
    const ref2 = tensor2.ref(.{});

    const result = try op.matmul(std.testing.allocator, std.testing.io, &ref1, &ref2);
    defer result.deinit(std.testing.allocator);

    try expectEqual(.{ 1, 1 }, result.metadata.shape);
    try expectApproxEqAbs(42, result.scalar(.{ 0, 0 }), tolerance);
}

test "ref operations - matmulNew non square inner dimension" {
    var data1 = [_]f64{
        1, 2, 3, 4,
    };

    var data2 = [_]f64{
        5,
        6,
        7,
        8,
    };

    var tensor1 = Tensor(f64, 2).from(.rowMajor(.{ 1, 4 }), data1[0..]);
    var tensor2 = Tensor(f64, 2).from(.rowMajor(.{ 4, 1 }), data2[0..]);

    const ref1 = tensor1.ref(.{});
    const ref2 = tensor2.ref(.{});

    const result = try op.matmul(std.testing.allocator, std.testing.io, &ref1, &ref2);
    defer result.deinit(std.testing.allocator);

    try expectEqual(.{ 1, 1 }, result.metadata.shape);
    try expectApproxEqAbs(70, result.scalar(.{ 0, 0 }), tolerance);
}

test "ref operations - matmulNew negative values" {
    var data1 = [_]f64{
        1, -2,
        3, -4,
    };

    var data2 = [_]f64{
        5,  6,
        -7, 8,
    };

    var tensor1 = Tensor(f64, 2).from(.rowMajor(.{ 2, 2 }), data1[0..]);
    var tensor2 = Tensor(f64, 2).from(.rowMajor(.{ 2, 2 }), data2[0..]);

    const ref1 = tensor1.ref(.{});
    const ref2 = tensor2.ref(.{});

    const result = try op.matmul(std.testing.allocator, std.testing.io, &ref1, &ref2);
    defer result.deinit(std.testing.allocator);

    try expectEqual(.{ 2, 2 }, result.metadata.shape);

    try expectApproxEqAbs(19, result.scalar(.{ 0, 0 }), tolerance);
    try expectApproxEqAbs(-10, result.scalar(.{ 0, 1 }), tolerance);

    try expectApproxEqAbs(43, result.scalar(.{ 1, 0 }), tolerance);
    try expectApproxEqAbs(-14, result.scalar(.{ 1, 1 }), tolerance);
}
