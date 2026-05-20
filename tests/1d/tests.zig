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

test "check shape" {
    var data: [9]f64 = createSequence(f64, 9);
    const tensor = Tensor(f64, 1).from(.rowMajor(.{3}), data[0..]);
    try expectEqual(.{3}, tensor.metadata.shape);
}
test "check stride" {
    var data: [9]f64 = createSequence(f64, 9);
    const tensor = Tensor(f64, 1).from(.rowMajor(.{3}), data[0..]);
    try expectEqual(.{1}, tensor.metadata.strides);
}
test "indexing scalars" {
    var data: [9]f64 = createSequence(f64, 9);
    var tensor = Tensor(f64, 1).from(.rowMajor(.{3}), data[0..]);
    try expectEqual(data[0], tensor.scalar(.{0}));
    try expectEqual(data[1], tensor.scalar(.{1}));
    try expectEqual(data[2], tensor.scalar(.{2}));
}
test "indexing scalars references" {
    var data: [9]f64 = createSequence(f64, 9);
    var tensor = Tensor(f64, 1).from(.rowMajor(.{3}), data[0..]);
    try expectEqual(&data[0], tensor.scalarRef(.{0}));
    try expectEqual(&data[1], tensor.scalarRef(.{1}));
    try expectEqual(&data[2], tensor.scalarRef(.{2}));
}
test "sub tensor content" {
    var data: [9]f64 = createSequence(f64, 9);
    var tensor = Tensor(f64, 1).from(.rowMajor(.{3}), data[0..]);
    try expectEqual(data[0], tensor.clone(std.testing.allocator, .{0}));
    try expectEqual(data[1], tensor.clone(std.testing.allocator, .{1}));
    try expectEqual(data[2], tensor.clone(std.testing.allocator, .{2}));
}
test "slice" {
    var data: [9]f64 = createSequence(f64, 9);
    var tensor = Tensor(f64, 1).from(.rowMajor(.{3}), data[0..]);
    var subtensor = tensor.slice(&.{
        .{ 1, 3 },
    });
    try expectEqual(data[1], subtensor.scalar(.{0}));
    try expectEqual(data[2], subtensor.scalar(.{1}));
}
test "view tensor from view" {
    var data: [3]f64 = createSequence(f64, 3);
    var view = Tensor(f64, 1).from(.rowMajor(.{3}), data[0..]);
    var tensor = view.ref(.{});
    try expectEqual(tensor.data[0], tensor.scalar(.{0}));
    try expectEqual(tensor.data[1], tensor.scalar(.{1}));
    try expectEqual(tensor.data[2], tensor.scalar(.{2}));

    try expectEqual(tensor.data[0], tensor.scalar(.{0}));
    try expectEqual(tensor.data[1], tensor.scalar(.{1}));
    try expectEqual(tensor.data[2], tensor.scalar(.{2}));
}
test "view from view from view" {
    var data: [3]f64 = createSequence(f64, 3);
    var view_tmp = Tensor(f64, 1).from(.rowMajor(.{3}), data[0..]);
    var tensor_tmp = view_tmp.ref(.{});
    var view = tensor_tmp.ref(.{});
    try expectEqual(view.data[0], view.scalar(.{0}));
    try expectEqual(view.data[1], view.scalar(.{1}));
    try expectEqual(view.data[2], view.scalar(.{2}));
}
test "element wise operation with a scalar (wise - in place)" {
    var data: [3]f64 = createSequence(f64, 3);
    var tensor1 = try Tensor(f64, 1).dupe(std.testing.allocator, .rowMajor(.{3}), data[0..]);
    defer tensor1.deinit(std.testing.allocator);
    var result = try Tensor(f64, 1).dupe(std.testing.allocator, .rowMajor(.{3}), data[0..]);
    defer result.deinit(std.testing.allocator);
    _ = result.wise(.{ @as(u32, 2), tensor1.ref(.{}) }, (struct {
        pub fn func(args: struct { u32, f64 }) f64 {
            const a, const b = args;
            return @as(f64, @floatFromInt(a)) + b;
        }
    }).func);
    try expectEqual(2, result.scalar(.{0}));
    try expectEqual(3, result.scalar(.{1}));
    try expectEqual(4, result.scalar(.{2}));
}
test "element wise operation with a tensor (wise - in place)" {
    var data: [3]f64 = createSequence(f64, 3);
    var tensor1 = try Tensor(f64, 1).dupe(std.testing.allocator, .rowMajor(.{3}), data[0..]);
    defer tensor1.deinit(std.testing.allocator);
    var tensor2 = try Tensor(f64, 1).dupe(std.testing.allocator, .rowMajor(.{3}), data[0..]);
    defer tensor2.deinit(std.testing.allocator);
    var result = try Tensor(f64, 1).dupe(std.testing.allocator, .rowMajor(.{3}), data[0..]);
    defer result.deinit(std.testing.allocator);
    _ = result.wise(.{ tensor1.ref(.{}), tensor2.ref(.{}) }, (struct {
        pub fn func(args: struct { f64, f64 }) f64 {
            const a, const b = args;
            return a + b;
        }
    }).func);
    try expectEqual(0, result.scalar(.{0}));
    try expectEqual(2, result.scalar(.{1}));
    try expectEqual(4, result.scalar(.{2}));

    try expectEqual(0, tensor2.scalar(.{0}));
    try expectEqual(1, tensor2.scalar(.{1}));
    try expectEqual(2, tensor2.scalar(.{2}));
}
test "wiseNew element-wise addition with scalar" {
    var data: [3]f64 = createSequence(f64, 3);
    var tensor1 = try Tensor(f64, 1).dupe(std.testing.allocator, .rowMajor(.{3}), data[0..]);
    defer tensor1.deinit(std.testing.allocator);
    const result = try op.wise(std.testing.allocator, .{ @as(u32, 2), tensor1.ref(.{}) }, (struct {
        pub fn func(args: struct { u32, f64 }) f64 {
            const a, const b = args;
            return @as(f64, @floatFromInt(a)) + b;
        }
    }).func);
    defer result.deinit(std.testing.allocator);
    try expectEqual(2, result.scalar(.{0}));
    try expectEqual(3, result.scalar(.{1}));
    try expectEqual(4, result.scalar(.{2}));
    // Original tensor should remain unchanged
    try expectEqual(0, tensor1.scalar(.{0}));
    try expectEqual(1, tensor1.scalar(.{1}));
    try expectEqual(2, tensor1.scalar(.{2}));
}
test "wiseNew element-wise addition with tensor" {
    var data1: [3]f64 = createSequence(f64, 3);
    var data2: [3]f64 = .{ 10, 20, 30 };
    var tensor1 = Tensor(f64, 1).from(.rowMajor(.{3}), data1[0..]);
    var tensor2 = Tensor(f64, 1).from(.rowMajor(.{3}), data2[0..]);
    var result: Tensor(f64, 1) = try .zeroes(std.testing.allocator, .rowMajor(.{3}));
    defer result.deinit(std.testing.allocator);
    _ = result.wise(.{ tensor1.ref(.{}), tensor2.ref(.{}) }, (struct {
        pub fn func(args: struct { f64, f64 }) f64 {
            const a, const b = args;
            return a + b;
        }
    }).func);
    try expectEqual(10, result.scalar(.{0}));
    try expectEqual(21, result.scalar(.{1}));
    try expectEqual(32, result.scalar(.{2}));
    // Original tensors should remain unchanged
    try expectEqual(0, tensor1.scalar(.{0}));
    try expectEqual(1, tensor1.scalar(.{1}));
    try expectEqual(2, tensor1.scalar(.{2}));
    try expectEqual(10, tensor2.scalar(.{0}));
    try expectEqual(20, tensor2.scalar(.{1}));
    try expectEqual(30, tensor2.scalar(.{2}));
}
test "broadcast 1D size 1 to size 3" {
    var data: [1]f64 = .{42};
    var tensor = Tensor(f64, 1).from(.rowMajor(.{1}), data[0..]);
    const broadcasted = tensor.ref(.{}).broadcast(.{3});

    try expectEqual(.{3}, broadcasted.metadata.shape);
    try expectEqual(.{0}, broadcasted.metadata.strides);
    try expectEqual(42, broadcasted.scalar(.{0}));
    try expectEqual(42, broadcasted.scalar(.{1}));
    try expectEqual(42, broadcasted.scalar(.{2}));
}
test "broadcast 1D to 2D" {
    var data: [1]f64 = .{5};
    var tensor = Tensor(f64, 1).from(.rowMajor(.{1}), data[0..]);
    const broadcasted = tensor.ref(.{}).broadcast(.{ 2, 3 });

    try expectEqual(.{ 2, 3 }, broadcasted.metadata.shape);
    try expectEqual(.{ 0, 0 }, broadcasted.metadata.strides);
    try expectEqual(5, broadcasted.scalar(.{ 0, 0 }));
    try expectEqual(5, broadcasted.scalar(.{ 0, 1 }));
    try expectEqual(5, broadcasted.scalar(.{ 0, 2 }));
    try expectEqual(5, broadcasted.scalar(.{ 1, 0 }));
    try expectEqual(5, broadcasted.scalar(.{ 1, 1 }));
    try expectEqual(5, broadcasted.scalar(.{ 1, 2 }));
}
test "broadcast 1D to 3D" {
    var data: [3]f64 = .{ 1, 2, 3 };
    var tensor = Tensor(f64, 1).from(.rowMajor(.{3}), data[0..]);
    const broadcasted = tensor.broadcast(.{ 2, 4, 3 });

    try expectEqual(.{ 2, 4, 3 }, broadcasted.metadata.shape);
    try expectEqual(.{ 0, 0, 1 }, broadcasted.metadata.strides);
    try expectEqual(1, broadcasted.scalar(.{ 0, 0, 0 }));
    try expectEqual(2, broadcasted.scalar(.{ 0, 0, 1 }));
    try expectEqual(3, broadcasted.scalar(.{ 0, 0, 2 }));
    try expectEqual(1, broadcasted.scalar(.{ 1, 3, 0 }));
    try expectEqual(2, broadcasted.scalar(.{ 1, 3, 1 }));
    try expectEqual(3, broadcasted.scalar(.{ 1, 3, 2 }));
}
test "iterator basic 1D" {
    var data: [3]f64 = .{ 5, 10, 15 };
    var tensor = Tensor(f64, 1).from(.rowMajor(.{3}), data[0..]);
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

test "reduce with scalar sum - in place" {
    var data: [3]f64 = .{ 1, 2, 3 };
    var tensor = Tensor(f64, 1).from(.rowMajor(.{3}), data[0..]);
    var result = try Tensor(f64, 1).alloc(std.testing.allocator, .rowMajor(.{1}));
    defer result.deinit(std.testing.allocator);

    result.reduce(@as(f64, 0), .{tensor.ref(.{})}, (struct {
        pub fn func(args: struct { f64 }, acc: f64) f64 {
            return args[0] + acc;
        }
    }).func);

    try expectEqual(6, result.ref(.{0}).*);
}

test "reduce with scalar product - in place" {
    var data: [3]f64 = .{ 2, 3, 4 };
    const tensor = Tensor(f64, 1).from(.rowMajor(.{3}), data[0..]);
    var result = try Tensor(f64, 1).alloc(std.testing.allocator, .rowMajor(.{1}));
    defer result.deinit(std.testing.allocator);

    result.reduce(@as(f64, 1), .{tensor}, (struct {
        pub fn func(args: struct { f64 }, acc: f64) f64 {
            return args[0] * acc;
        }
    }).func);

    try expectEqual(24, result.scalar(.{0}));
}

test "reduce with scalar product - in place using address of tensor" {
    var data: [3]f64 = .{ 2, 3, 4 };
    var tensor = try Tensor(f64, 1).dupe(std.testing.allocator, .rowMajor(.{3}), data[0..]);
    defer tensor.deinit(std.testing.allocator);
    var result = try Tensor(f64, 1).alloc(std.testing.allocator, .rowMajor(.{1}));
    defer result.deinit(std.testing.allocator);

    result.reduce(@as(f64, 1), .{tensor}, (struct {
        pub fn func(args: struct { f64 }, acc: f64) f64 {
            return args[0] * acc;
        }
    }).func);

    try expectEqual(24, result.constRef(.{0}));
}
