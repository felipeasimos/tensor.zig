const std = @import("std");
const expectEqual = std.testing.expectEqual;
const expect = std.testing.expect;
const Tensor = @import("tensor").Tensor;
const func = @import("tensor").func;
const op = @import("tensor").op;

fn createSequence(comptime dtype: type, comptime n: usize) [n]dtype {
    var seq: [n]dtype = .{1} ** n;
    inline for (0..n) |i| {
        seq[i] = i;
    }
    return seq;
}

test "check shape" {
    var data: [9]f64 = createSequence(f64, 9);
    const tensor = Tensor(f64, 2).from(.rowMajor(.{ 3, 3 }), data[0..]);
    try expectEqual(.{ 3, 3 }, tensor.metadata.shape);
}
test "check stride" {
    var data: [9]f64 = createSequence(f64, 9);
    const tensor = Tensor(f64, 2).from(.rowMajor(.{ 3, 3 }), data[0..]);
    try expectEqual(.{ 3, 1 }, tensor.metadata.strides);
}
test "indexing scalars" {
    var data: [9]f64 = createSequence(f64, 9);
    var tensor = Tensor(f64, 2).from(.rowMajor(.{ 3, 3 }), data[0..]);

    try expectEqual(data[0], tensor.scalar(.{ 0, 0 }));
    try expectEqual(data[1], tensor.scalar(.{ 0, 1 }));
    try expectEqual(data[2], tensor.scalar(.{ 0, 2 }));

    try expectEqual(data[3], tensor.scalar(.{ 1, 0 }));
    try expectEqual(data[4], tensor.scalar(.{ 1, 1 }));
    try expectEqual(data[5], tensor.scalar(.{ 1, 2 }));

    try expectEqual(data[6], tensor.scalar(.{ 2, 0 }));
    try expectEqual(data[7], tensor.scalar(.{ 2, 1 }));
    try expectEqual(data[8], tensor.scalar(.{ 2, 2 }));
}
test "ref sub tensor content (scalar)" {
    var data: [9]f64 = createSequence(f64, 9);
    var tensor = Tensor(f64, 2).from(.rowMajor(.{ 3, 3 }), data[0..]);

    try expectEqual(data[0], tensor.ref(.{ 0, 0 }).*);
    try expectEqual(data[1], tensor.ref(.{ 0, 1 }).*);
    try expectEqual(data[2], tensor.ref(.{ 0, 2 }).*);

    try expectEqual(data[3], tensor.ref(.{ 1, 0 }).*);
    try expectEqual(data[4], tensor.ref(.{ 1, 1 }).*);
    try expectEqual(data[5], tensor.ref(.{ 1, 2 }).*);

    try expectEqual(data[6], tensor.ref(.{ 2, 0 }).*);
    try expectEqual(data[7], tensor.ref(.{ 2, 1 }).*);
    try expectEqual(data[8], tensor.ref(.{ 2, 2 }).*);
}
test "ref sub tensor content (vector)" {
    var data: [9]f64 = createSequence(f64, 9);
    var tensor = Tensor(f64, 2).from(.rowMajor(.{ 3, 3 }), data[0..]);

    try expectEqual(data[0..3], tensor.ref(.{0}).data);
    try expectEqual(data[3..6], tensor.ref(.{1}).data);
    try expectEqual(data[6..9], tensor.ref(.{2}).data);
}
test "reshape 4x4 to 2x8" {
    var data: [16]f64 = createSequence(f64, 16);
    var tensor = Tensor(f64, 2).from(.rowMajor(.{ 4, 4 }), data[0..]);

    var new_tensor = tensor.reshape(.{ 2, 8 });

    try expectEqual(data[0..8], new_tensor.ref(.{0}).data);
    try expectEqual(data[8..16], new_tensor.ref(.{1}).data);
}
test "slice" {
    var data: [16]f64 = createSequence(f64, 16);
    var tensor = Tensor(f64, 2).from(.rowMajor(.{ 4, 4 }), data[0..]);
    var subtensor = tensor.slice(.{
        .{ 1, 3 },
        .{ 1, 3 },
    });
    try expectEqual(data[5], subtensor.scalar(.{ 0, 0 }));
    try expectEqual(data[6], subtensor.scalar(.{ 0, 1 }));
    try expectEqual(data[9], subtensor.scalar(.{ 1, 0 }));
    try expectEqual(data[10], subtensor.scalar(.{ 1, 1 }));
}
test "matmul 3x4 4x2" {
    var data1: [12]f64 = createSequence(f64, 12);
    const tensor1 = Tensor(f64, 2).from(.rowMajor(.{ 3, 4 }), &data1);
    var data2: [8]f64 = createSequence(f64, 8);
    const tensor2 = Tensor(f64, 2).from(.rowMajor(.{ 4, 2 }), &data2);

    var data3: [6]f64 = createSequence(f64, 6);
    var result = Tensor(f64, 2).from(.rowMajor(.{ 3, 2 }), &data3);
    try result.matmul(std.testing.allocator, std.testing.io, tensor1, tensor2);
    try expectEqual(result.metadata.shape, [_]usize{ 3, 2 });
}
test "matmulNew 3x4 4x2" {
    var data1: [12]f64 = createSequence(f64, 12);
    const tensor1 = Tensor(f64, 2).from(.rowMajor(.{ 3, 4 }), &data1);
    var data2: [8]f64 = createSequence(f64, 8);
    const tensor2 = Tensor(f64, 2).from(.rowMajor(.{ 4, 2 }), &data2);

    const result = try op.matmul(std.testing.allocator, std.testing.io, tensor1, tensor2);
    defer result.deinit(std.testing.allocator);
    try expectEqual(result.metadata.shape, [_]usize{ 3, 2 });
}
test "transpose" {
    var data: [12]f64 = createSequence(f64, 12);
    var tensor = Tensor(f64, 2).from(.rowMajor(.{ 3, 4 }), &data);
    const transpose = tensor.transpose(.{});
    try expectEqual(.{ 4, 3 }, transpose.metadata.shape);

    try expectEqual(data[0], transpose.scalar(.{ 0, 0 }));
    try expectEqual(data[4], transpose.scalar(.{ 0, 1 }));
    try expectEqual(data[8], transpose.scalar(.{ 0, 2 }));

    try expectEqual(data[1], transpose.scalar(.{ 1, 0 }));
    try expectEqual(data[5], transpose.scalar(.{ 1, 1 }));
    try expectEqual(data[9], transpose.scalar(.{ 1, 2 }));

    try expectEqual(data[2], transpose.scalar(.{ 2, 0 }));
    try expectEqual(data[6], transpose.scalar(.{ 2, 1 }));
    try expectEqual(data[10], transpose.scalar(.{ 2, 2 }));

    try expectEqual(data[3], transpose.scalar(.{ 3, 0 }));
    try expectEqual(data[7], transpose.scalar(.{ 3, 1 }));
    try expectEqual(data[11], transpose.scalar(.{ 3, 2 }));
}
test "wise element-wise addition with scalar (in place)" {
    var data: [4]f64 = .{ 1, 2, 3, 4 };
    var tensor = Tensor(f64, 2).from(.rowMajor(.{ 2, 2 }), data[0..]);
    var result = try Tensor(f64, 2).alloc(std.testing.allocator, .rowMajor(.{ 2, 2 }));
    _ = result.wise(.{ @as(f64, 10), &tensor }, (struct {
        pub fn func(args: struct { f64, f64 }) f64 {
            const a, const b = args;
            return a + b;
        }
    }).func);
    defer result.deinit(std.testing.allocator);
    try expectEqual(11, result.scalar(.{ 0, 0 }));
    try expectEqual(12, result.scalar(.{ 0, 1 }));
    try expectEqual(13, result.scalar(.{ 1, 0 }));
    try expectEqual(14, result.scalar(.{ 1, 1 }));
}
test "wise element-wise addition with tensor (in place)" {
    var data1: [4]f64 = .{ 1, 2, 3, 4 };
    var data2: [4]f64 = .{ 10, 20, 30, 40 };
    var tensor1 = Tensor(f64, 2).from(.rowMajor(.{ 2, 2 }), data1[0..]);
    var tensor2 = Tensor(f64, 2).from(.rowMajor(.{ 2, 2 }), data2[0..]);
    var result = try Tensor(f64, 2).alloc(std.testing.allocator, .rowMajor(.{ 2, 2 }));
    _ = result.wise(.{ &tensor1, &tensor2 }, (struct {
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
    // tensor2 should remain unchanged
    try expectEqual(10, tensor2.scalar(.{ 0, 0 }));
    try expectEqual(20, tensor2.scalar(.{ 0, 1 }));
    try expectEqual(30, tensor2.scalar(.{ 1, 0 }));
    try expectEqual(40, tensor2.scalar(.{ 1, 1 }));
}
test "wiseNew element-wise addition with scalar" {
    var data: [4]f64 = .{ 1, 2, 3, 4 };
    const tensor = Tensor(f64, 2).from(.rowMajor(.{ 2, 2 }), data[0..]);
    const result = try op.wise(std.testing.allocator, .{ @as(f64, 10), &tensor }, (struct {
        pub fn func(args: struct { f64, f64 }) f64 {
            const a, const b = args;
            return a + b;
        }
    }).func);
    defer result.deinit(std.testing.allocator);
    try expectEqual(11, result.scalar(.{ 0, 0 }));
    try expectEqual(12, result.scalar(.{ 0, 1 }));
    try expectEqual(13, result.scalar(.{ 1, 0 }));
    try expectEqual(14, result.scalar(.{ 1, 1 }));
    // Original tensor should remain unchanged
    try expectEqual(1, tensor.scalar(.{ 0, 0 }));
    try expectEqual(2, tensor.scalar(.{ 0, 1 }));
    try expectEqual(3, tensor.scalar(.{ 1, 0 }));
    try expectEqual(4, tensor.scalar(.{ 1, 1 }));
}
test "wiseNew element-wise addition with tensor" {
    var data1: [4]f64 = .{ 1, 2, 3, 4 };
    var data2: [4]f64 = .{ 10, 20, 30, 40 };
    const tensor1 = Tensor(f64, 2).from(.rowMajor(.{ 2, 2 }), data1[0..]);
    const tensor2 = Tensor(f64, 2).from(.rowMajor(.{ 2, 2 }), data2[0..]);
    const result = try op.wise(std.testing.allocator, .{ &tensor1, &tensor2 }, (struct {
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
    // Original tensors should remain unchanged
    try expectEqual(1, tensor1.scalar(.{ 0, 0 }));
    try expectEqual(2, tensor1.scalar(.{ 0, 1 }));
    try expectEqual(3, tensor1.scalar(.{ 1, 0 }));
    try expectEqual(4, tensor1.scalar(.{ 1, 1 }));
    try expectEqual(10, tensor2.scalar(.{ 0, 0 }));
    try expectEqual(20, tensor2.scalar(.{ 0, 1 }));
    try expectEqual(30, tensor2.scalar(.{ 1, 0 }));
    try expectEqual(40, tensor2.scalar(.{ 1, 1 }));
}
test "broadcast 2D [1,3] to [2,3]" {
    var data: [3]f64 = .{ 1, 2, 3 };
    var tensor = Tensor(f64, 2).from(.rowMajor(.{ 1, 3 }), data[0..]);
    const broadcasted = tensor.broadcast(.{ 2, 3 });

    try expectEqual(.{ 2, 3 }, broadcasted.metadata.shape);
    try expectEqual(.{ 0, 1 }, broadcasted.metadata.strides);
    try expectEqual(1, broadcasted.scalar(.{ 0, 0 }));
    try expectEqual(2, broadcasted.scalar(.{ 0, 1 }));
    try expectEqual(3, broadcasted.scalar(.{ 0, 2 }));
    try expectEqual(1, broadcasted.scalar(.{ 1, 0 }));
    try expectEqual(2, broadcasted.scalar(.{ 1, 1 }));
    try expectEqual(3, broadcasted.scalar(.{ 1, 2 }));
}
test "broadcast 2D [2,1] to [2,4]" {
    var data: [2]f64 = .{ 10, 20 };
    var tensor = Tensor(f64, 2).from(.rowMajor(.{ 2, 1 }), data[0..]);
    const broadcasted = tensor.broadcast(.{ 2, 4 });

    try expectEqual(.{ 2, 4 }, broadcasted.metadata.shape);
    try expectEqual(.{ 1, 0 }, broadcasted.metadata.strides);
    try expectEqual(10, broadcasted.scalar(.{ 0, 0 }));
    try expectEqual(10, broadcasted.scalar(.{ 0, 1 }));
    try expectEqual(10, broadcasted.scalar(.{ 0, 2 }));
    try expectEqual(10, broadcasted.scalar(.{ 0, 3 }));
    try expectEqual(20, broadcasted.scalar(.{ 1, 0 }));
    try expectEqual(20, broadcasted.scalar(.{ 1, 1 }));
    try expectEqual(20, broadcasted.scalar(.{ 1, 2 }));
    try expectEqual(20, broadcasted.scalar(.{ 1, 3 }));
}
test "broadcast 2D [1,1] to [3,4]" {
    var data: [1]f64 = .{99};
    var tensor = Tensor(f64, 2).from(.rowMajor(.{ 1, 1 }), data[0..]);
    const broadcasted = tensor.broadcast(.{ 3, 4 });

    try expectEqual(.{ 3, 4 }, broadcasted.metadata.shape);
    try expectEqual(.{ 0, 0 }, broadcasted.metadata.strides);
    try expectEqual(99, broadcasted.scalar(.{ 0, 0 }));
    try expectEqual(99, broadcasted.scalar(.{ 0, 3 }));
    try expectEqual(99, broadcasted.scalar(.{ 2, 0 }));
    try expectEqual(99, broadcasted.scalar(.{ 2, 3 }));
}
test "broadcast 2D to 3D" {
    var data: [6]f64 = .{ 1, 2, 3, 4, 5, 6 };
    var tensor = Tensor(f64, 2).from(.rowMajor(.{ 2, 3 }), &data);
    const broadcasted = tensor.broadcast(.{ 4, 2, 3 });

    try expectEqual(.{ 4, 2, 3 }, broadcasted.metadata.shape);
    try expectEqual(.{ 0, 3, 1 }, broadcasted.metadata.strides);
    try expectEqual(1, broadcasted.scalar(.{ 0, 0, 0 }));
    try expectEqual(2, broadcasted.scalar(.{ 0, 0, 1 }));
    try expectEqual(4, broadcasted.scalar(.{ 0, 1, 0 }));
    try expectEqual(1, broadcasted.scalar(.{ 3, 0, 0 }));
    try expectEqual(4, broadcasted.scalar(.{ 3, 1, 0 }));
}
test "iterator basic 2x2" {
    var data: [4]f64 = .{ 1, 2, 3, 4 };
    var tensor = Tensor(f64, 2).from(.rowMajor(.{ 2, 2 }), data[0..]);
    var iter = tensor.iter();

    var count: usize = 0;
    while (iter.next()) |item| {
        count += 1;
        switch (count) {
            1 => {
                try expectEqual(.{ 0, 0 }, item.indices);
                try expectEqual(1, item.value);
            },
            2 => {
                try expectEqual(.{ 0, 1 }, item.indices);
                try expectEqual(2, item.value);
            },
            3 => {
                try expectEqual(.{ 1, 0 }, item.indices);
                try expectEqual(3, item.value);
            },
            4 => {
                try expectEqual(.{ 1, 1 }, item.indices);
                try expectEqual(4, item.value);
            },
            else => unreachable,
        }
    }
    try expectEqual(4, count);
}
test "iterator broadcasted tensor" {
    var data: [1]f64 = .{42};
    var tensor = Tensor(f64, 2).from(.rowMajor(.{ 1, 1 }), data[0..]);
    var broadcasted = tensor.broadcast(.{ 2, 3 });
    var iter = broadcasted.iter();

    var count: usize = 0;
    while (iter.next()) |item| {
        count += 1;
        try expectEqual(42, item.value);
    }
    try expectEqual(6, count);
}

test "clone broadcast" {
    var data: [1]f64 = .{42};
    var tensor = Tensor(f64, 2).from(.rowMajor(.{ 1, 1 }), data[0..]);
    var broadcasted = tensor.broadcast(.{ 2, 3 });

    var clone = try broadcasted.clone(std.testing.allocator, .{});
    defer clone.deinit(std.testing.allocator);

    var iter = clone.iter();

    var count: usize = 0;
    while (iter.next()) |item| {
        count += 1;
        try expectEqual(42, item.value);
    }
    try expectEqual(6, count);
}

test "colum major" {
    var data = createSequence(f64, 12);
    const tensor: Tensor(f64, 2) = .from(.columnMajor(.{ 4, 3 }), data[0..]);
    try expectEqual(.{ 4, 3 }, tensor.metadata.shape);
    try expectEqual(.{ 1, 4 }, tensor.metadata.strides);

    try expectEqual(0, tensor.scalar(.{ 0, 0 }));
    try expectEqual(1, tensor.scalar(.{ 1, 0 }));
    try expectEqual(2, tensor.scalar(.{ 2, 0 }));
    try expectEqual(3, tensor.scalar(.{ 3, 0 }));

    try expectEqual(4, tensor.scalar(.{ 0, 1 }));
    try expectEqual(5, tensor.scalar(.{ 1, 1 }));
    try expectEqual(6, tensor.scalar(.{ 2, 1 }));
    try expectEqual(7, tensor.scalar(.{ 3, 1 }));

    try expectEqual(8, tensor.scalar(.{ 0, 2 }));
    try expectEqual(9, tensor.scalar(.{ 1, 2 }));
    try expectEqual(10, tensor.scalar(.{ 2, 2 }));
    try expectEqual(11, tensor.scalar(.{ 3, 2 }));
}

test "pack into different major" {
    var data = createSequence(f64, 12);
    const original: Tensor(f64, 2) = .from(.columnMajor(.{ 4, 3 }), data[0..]);
    const tensor = try original.pack(std.testing.allocator, .RowMajor);
    defer tensor.deinit(std.testing.allocator);

    try expectEqual(.{ 4, 3 }, tensor.metadata.shape);
    try expectEqual(.{ 3, 1 }, tensor.metadata.strides);
    try expectEqual(0, tensor.scalar(.{ 0, 0 }));
    try expectEqual(1, tensor.scalar(.{ 1, 0 }));
    try expectEqual(2, tensor.scalar(.{ 2, 0 }));
    try expectEqual(3, tensor.scalar(.{ 3, 0 }));

    try expectEqual(4, tensor.scalar(.{ 0, 1 }));
    try expectEqual(5, tensor.scalar(.{ 1, 1 }));
    try expectEqual(6, tensor.scalar(.{ 2, 1 }));
    try expectEqual(7, tensor.scalar(.{ 3, 1 }));

    try expectEqual(8, tensor.scalar(.{ 0, 2 }));
    try expectEqual(9, tensor.scalar(.{ 1, 2 }));
    try expectEqual(10, tensor.scalar(.{ 2, 2 }));
    try expectEqual(11, tensor.scalar(.{ 3, 2 }));
}
