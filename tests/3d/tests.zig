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
    var data: [27]f64 = createSequence(f64, 27);
    const tensor = Tensor(f64, 3).from(.rowMajor(.{ 3, 3, 3 }), data[0..]);
    try expectEqual(.{ 3, 3, 3 }, tensor.metadata.shape);
}
test "check stride" {
    var data: [27]f64 = createSequence(f64, 27);
    const tensor = Tensor(f64, 3).from(.rowMajor(.{ 3, 3, 3 }), data[0..]);
    try expectEqual(.{ 9, 3, 1 }, tensor.metadata.strides);
}
test "indexing scalars" {
    var data: [27]f64 = createSequence(f64, 27);
    var tensor = Tensor(f64, 3).from(.rowMajor(.{ 3, 3, 3 }), data[0..]);

    try expectEqual(data[0], tensor.scalar(.{ 0, 0, 0 }));
    try expectEqual(data[1], tensor.scalar(.{ 0, 0, 1 }));
    try expectEqual(data[2], tensor.scalar(.{ 0, 0, 2 }));
    try expectEqual(data[3], tensor.scalar(.{ 0, 1, 0 }));
    try expectEqual(data[4], tensor.scalar(.{ 0, 1, 1 }));
    try expectEqual(data[5], tensor.scalar(.{ 0, 1, 2 }));
    try expectEqual(data[6], tensor.scalar(.{ 0, 2, 0 }));
    try expectEqual(data[7], tensor.scalar(.{ 0, 2, 1 }));
    try expectEqual(data[8], tensor.scalar(.{ 0, 2, 2 }));

    try expectEqual(data[9], tensor.scalar(.{ 1, 0, 0 }));
    try expectEqual(data[10], tensor.scalar(.{ 1, 0, 1 }));
    try expectEqual(data[11], tensor.scalar(.{ 1, 0, 2 }));
    try expectEqual(data[12], tensor.scalar(.{ 1, 1, 0 }));
    try expectEqual(data[13], tensor.scalar(.{ 1, 1, 1 }));
    try expectEqual(data[14], tensor.scalar(.{ 1, 1, 2 }));
    try expectEqual(data[15], tensor.scalar(.{ 1, 2, 0 }));
    try expectEqual(data[16], tensor.scalar(.{ 1, 2, 1 }));
    try expectEqual(data[17], tensor.scalar(.{ 1, 2, 2 }));

    try expectEqual(data[18], tensor.scalar(.{ 2, 0, 0 }));
    try expectEqual(data[19], tensor.scalar(.{ 2, 0, 1 }));
    try expectEqual(data[20], tensor.scalar(.{ 2, 0, 2 }));
    try expectEqual(data[21], tensor.scalar(.{ 2, 1, 0 }));
    try expectEqual(data[22], tensor.scalar(.{ 2, 1, 1 }));
    try expectEqual(data[23], tensor.scalar(.{ 2, 1, 2 }));
    try expectEqual(data[24], tensor.scalar(.{ 2, 2, 0 }));
    try expectEqual(data[25], tensor.scalar(.{ 2, 2, 1 }));
    try expectEqual(data[26], tensor.scalar(.{ 2, 2, 2 }));
}
test "ref sub tensor content (scalar)" {
    var data: [27]f64 = createSequence(f64, 27);
    var tensor = Tensor(f64, 3).from(.rowMajor(.{ 3, 3, 3 }), data[0..]);

    try expectEqual(data[0], tensor.ref(.{ 0, 0, 0 }).*);
    try expectEqual(data[1], tensor.ref(.{ 0, 0, 1 }).*);
    try expectEqual(data[2], tensor.ref(.{ 0, 0, 2 }).*);
    try expectEqual(data[3], tensor.ref(.{ 0, 1, 0 }).*);
    try expectEqual(data[4], tensor.ref(.{ 0, 1, 1 }).*);
    try expectEqual(data[5], tensor.ref(.{ 0, 1, 2 }).*);
    try expectEqual(data[6], tensor.ref(.{ 0, 2, 0 }).*);
    try expectEqual(data[7], tensor.ref(.{ 0, 2, 1 }).*);
    try expectEqual(data[8], tensor.ref(.{ 0, 2, 2 }).*);

    try expectEqual(data[9], tensor.ref(.{ 1, 0, 0 }).*);
    try expectEqual(data[10], tensor.ref(.{ 1, 0, 1 }).*);
    try expectEqual(data[11], tensor.ref(.{ 1, 0, 2 }).*);
    try expectEqual(data[12], tensor.ref(.{ 1, 1, 0 }).*);
    try expectEqual(data[13], tensor.ref(.{ 1, 1, 1 }).*);
    try expectEqual(data[14], tensor.ref(.{ 1, 1, 2 }).*);
    try expectEqual(data[15], tensor.ref(.{ 1, 2, 0 }).*);
    try expectEqual(data[16], tensor.ref(.{ 1, 2, 1 }).*);
    try expectEqual(data[17], tensor.ref(.{ 1, 2, 2 }).*);

    try expectEqual(data[18], tensor.ref(.{ 2, 0, 0 }).*);
    try expectEqual(data[19], tensor.ref(.{ 2, 0, 1 }).*);
    try expectEqual(data[20], tensor.ref(.{ 2, 0, 2 }).*);
    try expectEqual(data[21], tensor.ref(.{ 2, 1, 0 }).*);
    try expectEqual(data[22], tensor.ref(.{ 2, 1, 1 }).*);
    try expectEqual(data[23], tensor.ref(.{ 2, 1, 2 }).*);
    try expectEqual(data[24], tensor.ref(.{ 2, 2, 0 }).*);
    try expectEqual(data[25], tensor.ref(.{ 2, 2, 1 }).*);
    try expectEqual(data[26], tensor.ref(.{ 2, 2, 2 }).*);
}
test "ref sub tensor content (vector)" {
    var data: [27]f64 = createSequence(f64, 27);
    var tensor = Tensor(f64, 3).from(.rowMajor(.{ 3, 3, 3 }), data[0..]);

    try expectEqual(data[0..3], tensor.ref(.{ 0, 0 }).data);
    try expectEqual(data[3..6], tensor.ref(.{ 0, 1 }).data);
    try expectEqual(data[6..9], tensor.ref(.{ 0, 2 }).data);

    try expectEqual(data[9..12], tensor.ref(.{ 1, 0 }).data);
    try expectEqual(data[12..15], tensor.ref(.{ 1, 1 }).data);
    try expectEqual(data[15..18], tensor.ref(.{ 1, 2 }).data);

    try expectEqual(data[18..21], tensor.ref(.{ 2, 0 }).data);
    try expectEqual(data[21..24], tensor.ref(.{ 2, 1 }).data);
    try expectEqual(data[24..27], tensor.ref(.{ 2, 2 }).data);
}
test "ref sub tensor content (matrix)" {
    var data: [27]f64 = createSequence(f64, 27);
    var tensor = Tensor(f64, 3).from(.rowMajor(.{ 3, 3, 3 }), data[0..]);

    try expectEqual(data[0..9], tensor.ref(.{0}).data);

    try expectEqual(data[9..18], tensor.ref(.{1}).data);

    try expectEqual(data[18..27], tensor.ref(.{2}).data);
}
test "reshape 2x3x4 to 4x3x2" {
    var data: [24]f64 = createSequence(f64, 24);
    var tensor = Tensor(f64, 3).from(.rowMajor(.{ 2, 3, 4 }), data[0..]);

    try expectEqual(.{ 12, 4, 1 }, tensor.metadata.strides);

    var new_tensor = tensor.reshape(.{ 4, 3, 2 });

    try expectEqual(.{ 6, 2, 1 }, new_tensor.metadata.strides);

    try expectEqual(data[0..6], new_tensor.ref(.{0}).data);
    try expectEqual(data[6..12], new_tensor.ref(.{1}).data);
    try expectEqual(data[12..18], new_tensor.ref(.{2}).data);
    try expectEqual(data[18..24], new_tensor.ref(.{3}).data);
}
test "slice" {
    var data: [24]f64 = createSequence(f64, 24);
    var tensor = Tensor(f64, 3).from(.rowMajor(.{ 3, 4, 2 }), data[0..]);
    var subtensor = tensor.slice(.{
        .{ 1, 3 },
        .{ 2, 4 },
        .{ 0, 2 },
    });
    try expectEqual(data[12], subtensor.scalar(.{ 0, 0, 0 }));
    try expectEqual(data[13], subtensor.scalar(.{ 0, 0, 1 }));

    try expectEqual(data[14], subtensor.scalar(.{ 0, 1, 0 }));
    try expectEqual(data[15], subtensor.scalar(.{ 0, 1, 1 }));

    try expectEqual(data[20], subtensor.scalar(.{ 1, 0, 0 }));
    try expectEqual(data[21], subtensor.scalar(.{ 1, 0, 1 }));

    try expectEqual(data[22], subtensor.scalar(.{ 1, 1, 0 }));
    try expectEqual(data[23], subtensor.scalar(.{ 1, 1, 1 }));
}
test "ref reference to matrix" {
    var data: [24]f64 = createSequence(f64, 24);
    var tensor = Tensor(f64, 3).from(.rowMajor(.{ 3, 4, 2 }), data[0..]);

    var subtensor1 = tensor.ref(.{0});

    try expectEqual(.{ 4, 2 }, subtensor1.metadata.shape);

    try expectEqual(data[0], subtensor1.scalar(.{ 0, 0 }));
    try expectEqual(data[1], subtensor1.scalar(.{ 0, 1 }));

    try expectEqual(data[2], subtensor1.scalar(.{ 1, 0 }));
    try expectEqual(data[3], subtensor1.scalar(.{ 1, 1 }));

    try expectEqual(data[4], subtensor1.scalar(.{ 2, 0 }));
    try expectEqual(data[5], subtensor1.scalar(.{ 2, 1 }));

    try expectEqual(data[6], subtensor1.scalar(.{ 3, 0 }));
    try expectEqual(data[7], subtensor1.scalar(.{ 3, 1 }));

    var subtensor2 = tensor.ref(.{1});

    try expectEqual(.{ 4, 2 }, subtensor2.metadata.shape);

    try expectEqual(data[8], subtensor2.scalar(.{ 0, 0 }));
    try expectEqual(data[9], subtensor2.scalar(.{ 0, 1 }));

    try expectEqual(data[10], subtensor2.scalar(.{ 1, 0 }));
    try expectEqual(data[11], subtensor2.scalar(.{ 1, 1 }));

    try expectEqual(data[12], subtensor2.scalar(.{ 2, 0 }));
    try expectEqual(data[13], subtensor2.scalar(.{ 2, 1 }));

    try expectEqual(data[14], subtensor2.scalar(.{ 3, 0 }));
    try expectEqual(data[15], subtensor2.scalar(.{ 3, 1 }));

    var subtensor3 = tensor.ref(.{2});

    try expectEqual(.{ 4, 2 }, subtensor3.metadata.shape);

    try expectEqual(data[16], subtensor3.scalar(.{ 0, 0 }));
    try expectEqual(data[17], subtensor3.scalar(.{ 0, 1 }));

    try expectEqual(data[18], subtensor3.scalar(.{ 1, 0 }));
    try expectEqual(data[19], subtensor3.scalar(.{ 1, 1 }));

    try expectEqual(data[20], subtensor3.scalar(.{ 2, 0 }));
    try expectEqual(data[21], subtensor3.scalar(.{ 2, 1 }));

    try expectEqual(data[22], subtensor3.scalar(.{ 3, 0 }));
    try expectEqual(data[23], subtensor3.scalar(.{ 3, 1 }));
}

test "matmul" {
    var data: [24]f64 = createSequence(f64, 24);
    var tensor = Tensor(f64, 3).from(.rowMajor(.{ 3, 4, 2 }), data[0..]);

    var subtensor1 = tensor.ref(.{0});
    try expectEqual(.{ 4, 2 }, subtensor1.metadata.shape);
    try expectEqual(8, subtensor1.metadata.numScalars());
    var subtensor2 = tensor.ref(.{1});
    try expectEqual(.{ 4, 2 }, subtensor2.metadata.shape);
    try expectEqual(8, subtensor2.metadata.numScalars());
    var ref_to_2 = subtensor2.ref(.{});
    try expectEqual(.{ 4, 2 }, ref_to_2.metadata.shape);
    try expectEqual(8, ref_to_2.metadata.numScalars());
    const reshaped_ref_to_2 = ref_to_2.reshape(.{ 2, 4 });
    try expectEqual(.{ 2, 4 }, reshaped_ref_to_2.metadata.shape);
    try expectEqual(8, reshaped_ref_to_2.metadata.numScalars());
    const result = try op.matmul(std.testing.allocator, std.testing.io, &subtensor1, subtensor2.ref(.{}).reshape(.{ 2, 4 }));
    defer result.deinit(std.testing.allocator);

    try expectEqual(.{ 4, 4 }, result.metadata.shape);
}

test "wise element-wise addition with scalar (in place)" {
    var data: [8]f64 = .{ 1, 2, 3, 4, 5, 6, 7, 8 };
    const tensor = Tensor(f64, 3).from(.rowMajor(.{ 2, 2, 2 }), data[0..]);
    var result = try Tensor(f64, 3).alloc(std.testing.allocator, .rowMajor(.{ 2, 2, 2 }));
    defer result.deinit(std.testing.allocator);
    result.wise(.{ @as(f64, 10), &tensor }, (struct {
        pub fn func(args: struct { f64, f64 }) f64 {
            const a, const b = args;
            return a + b;
        }
    }).func);
    try expectEqual(11, result.scalar(.{ 0, 0, 0 }));
    try expectEqual(12, result.scalar(.{ 0, 0, 1 }));
    try expectEqual(13, result.scalar(.{ 0, 1, 0 }));
    try expectEqual(14, result.scalar(.{ 0, 1, 1 }));
    try expectEqual(15, result.scalar(.{ 1, 0, 0 }));
    try expectEqual(16, result.scalar(.{ 1, 0, 1 }));
    try expectEqual(17, result.scalar(.{ 1, 1, 0 }));
    try expectEqual(18, result.scalar(.{ 1, 1, 1 }));
}
test "wise element-wise addition with tensor (in place)" {
    var data1: [8]f64 = .{ 1, 2, 3, 4, 5, 6, 7, 8 };
    var data2: [8]f64 = .{ 10, 20, 30, 40, 50, 60, 70, 80 };
    var tensor1 = Tensor(f64, 3).from(.rowMajor(.{ 2, 2, 2 }), data1[0..]);
    var tensor2 = Tensor(f64, 3).from(.rowMajor(.{ 2, 2, 2 }), data2[0..]);
    var result = try Tensor(f64, 3).alloc(std.testing.allocator, .rowMajor(.{ 2, 2, 2 }));
    defer result.deinit(std.testing.allocator);
    result.wise(.{ &tensor1, &tensor2 }, (struct {
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
    // tensor2 should remain unchanged
    try expectEqual(10, tensor2.scalar(.{ 0, 0, 0 }));
    try expectEqual(20, tensor2.scalar(.{ 0, 0, 1 }));
    try expectEqual(30, tensor2.scalar(.{ 0, 1, 0 }));
    try expectEqual(40, tensor2.scalar(.{ 0, 1, 1 }));
    try expectEqual(50, tensor2.scalar(.{ 1, 0, 0 }));
    try expectEqual(60, tensor2.scalar(.{ 1, 0, 1 }));
    try expectEqual(70, tensor2.scalar(.{ 1, 1, 0 }));
    try expectEqual(80, tensor2.scalar(.{ 1, 1, 1 }));
}
test "wiseNew element-wise addition with scalar" {
    var data: [8]f64 = .{ 1, 2, 3, 4, 5, 6, 7, 8 };
    const tensor = Tensor(f64, 3).from(.rowMajor(.{ 2, 2, 2 }), data[0..]);
    const result = try op.wise(std.testing.allocator, .{ @as(f64, 10), &tensor }, (struct {
        pub fn func(args: struct { f64, f64 }) f64 {
            const a, const b = args;
            return a + b;
        }
    }).func);
    defer result.deinit(std.testing.allocator);
    try expectEqual(11, result.scalar(.{ 0, 0, 0 }));
    try expectEqual(12, result.scalar(.{ 0, 0, 1 }));
    try expectEqual(13, result.scalar(.{ 0, 1, 0 }));
    try expectEqual(14, result.scalar(.{ 0, 1, 1 }));
    try expectEqual(15, result.scalar(.{ 1, 0, 0 }));
    try expectEqual(16, result.scalar(.{ 1, 0, 1 }));
    try expectEqual(17, result.scalar(.{ 1, 1, 0 }));
    try expectEqual(18, result.scalar(.{ 1, 1, 1 }));
    // Original tensor should remain unchanged
    try expectEqual(1, tensor.scalar(.{ 0, 0, 0 }));
    try expectEqual(2, tensor.scalar(.{ 0, 0, 1 }));
    try expectEqual(3, tensor.scalar(.{ 0, 1, 0 }));
    try expectEqual(4, tensor.scalar(.{ 0, 1, 1 }));
    try expectEqual(5, tensor.scalar(.{ 1, 0, 0 }));
    try expectEqual(6, tensor.scalar(.{ 1, 0, 1 }));
    try expectEqual(7, tensor.scalar(.{ 1, 1, 0 }));
    try expectEqual(8, tensor.scalar(.{ 1, 1, 1 }));
}
test "wiseNew element-wise addition with tensor" {
    var data1: [8]f64 = .{ 1, 2, 3, 4, 5, 6, 7, 8 };
    var data2: [8]f64 = .{ 10, 20, 30, 40, 50, 60, 70, 80 };
    const tensor1 = Tensor(f64, 3).from(.rowMajor(.{ 2, 2, 2 }), data1[0..]);
    const tensor2 = Tensor(f64, 3).from(.rowMajor(.{ 2, 2, 2 }), data2[0..]);
    const result = try op.wise(std.testing.allocator, .{ &tensor1, &tensor2 }, (struct {
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
    // Original tensors should remain unchanged
    try expectEqual(1, tensor1.scalar(.{ 0, 0, 0 }));
    try expectEqual(2, tensor1.scalar(.{ 0, 0, 1 }));
    try expectEqual(3, tensor1.scalar(.{ 0, 1, 0 }));
    try expectEqual(4, tensor1.scalar(.{ 0, 1, 1 }));
    try expectEqual(5, tensor1.scalar(.{ 1, 0, 0 }));
    try expectEqual(6, tensor1.scalar(.{ 1, 0, 1 }));
    try expectEqual(7, tensor1.scalar(.{ 1, 1, 0 }));
    try expectEqual(8, tensor1.scalar(.{ 1, 1, 1 }));
    try expectEqual(10, tensor2.scalar(.{ 0, 0, 0 }));
    try expectEqual(20, tensor2.scalar(.{ 0, 0, 1 }));
    try expectEqual(30, tensor2.scalar(.{ 0, 1, 0 }));
    try expectEqual(40, tensor2.scalar(.{ 0, 1, 1 }));
    try expectEqual(50, tensor2.scalar(.{ 1, 0, 0 }));
    try expectEqual(60, tensor2.scalar(.{ 1, 0, 1 }));
    try expectEqual(70, tensor2.scalar(.{ 1, 1, 0 }));
    try expectEqual(80, tensor2.scalar(.{ 1, 1, 1 }));
}
test "broadcast 3D [1,2,3] to [2,2,3]" {
    var data: [6]f64 = .{ 1, 2, 3, 4, 5, 6 };
    var tensor = Tensor(f64, 3).from(.rowMajor(.{ 1, 2, 3 }), data[0..]);
    const broadcasted = tensor.broadcast(.{ 2, 2, 3 });

    try expectEqual(.{ 2, 2, 3 }, broadcasted.metadata.shape);
    try expectEqual(.{ 0, 3, 1 }, broadcasted.metadata.strides);
    try expectEqual(1, broadcasted.scalar(.{ 0, 0, 0 }));
    try expectEqual(2, broadcasted.scalar(.{ 0, 0, 1 }));
    try expectEqual(3, broadcasted.scalar(.{ 0, 0, 2 }));
    try expectEqual(4, broadcasted.scalar(.{ 0, 1, 0 }));
    try expectEqual(5, broadcasted.scalar(.{ 0, 1, 1 }));
    try expectEqual(6, broadcasted.scalar(.{ 0, 1, 2 }));
    try expectEqual(1, broadcasted.scalar(.{ 1, 0, 0 }));
    try expectEqual(4, broadcasted.scalar(.{ 1, 1, 0 }));
}
test "broadcast 3D [2,1,1] to [2,3,4]" {
    var data: [2]f64 = .{ 100, 200 };
    var tensor = Tensor(f64, 3).from(.rowMajor(.{ 2, 1, 1 }), data[0..]);
    const broadcasted = tensor.broadcast(.{ 2, 3, 4 });

    try expectEqual(.{ 2, 3, 4 }, broadcasted.metadata.shape);
    try expectEqual(.{ 1, 0, 0 }, broadcasted.metadata.strides);
    try expectEqual(100, broadcasted.scalar(.{ 0, 0, 0 }));
    try expectEqual(100, broadcasted.scalar(.{ 0, 1, 1 }));
    try expectEqual(100, broadcasted.scalar(.{ 0, 2, 3 }));
    try expectEqual(200, broadcasted.scalar(.{ 1, 0, 0 }));
    try expectEqual(200, broadcasted.scalar(.{ 1, 1, 1 }));
    try expectEqual(200, broadcasted.scalar(.{ 1, 2, 3 }));
}
test "broadcast 3D [1,1,3] to [2,4,3]" {
    var data: [3]f64 = .{ 7, 8, 9 };
    var tensor = Tensor(f64, 3).from(.rowMajor(.{ 1, 1, 3 }), data[0..]);
    const broadcasted = tensor.broadcast(.{ 2, 4, 3 });

    try expectEqual(.{ 2, 4, 3 }, broadcasted.metadata.shape);
    try expectEqual(.{ 0, 0, 1 }, broadcasted.metadata.strides);
    try expectEqual(7, broadcasted.scalar(.{ 0, 0, 0 }));
    try expectEqual(8, broadcasted.scalar(.{ 0, 0, 1 }));
    try expectEqual(9, broadcasted.scalar(.{ 0, 0, 2 }));
    try expectEqual(7, broadcasted.scalar(.{ 0, 3, 0 }));
    try expectEqual(7, broadcasted.scalar(.{ 1, 0, 0 }));
    try expectEqual(9, broadcasted.scalar(.{ 1, 3, 2 }));
}
test "broadcast 3D [1,1,1] to [3,4,5]" {
    var data: [1]f64 = .{42};
    var tensor = Tensor(f64, 3).from(.rowMajor(.{ 1, 1, 1 }), data[0..]);
    const broadcasted = tensor.broadcast(.{ 3, 4, 5 });

    try expectEqual(.{ 3, 4, 5 }, broadcasted.metadata.shape);
    try expectEqual(.{ 0, 0, 0 }, broadcasted.metadata.strides);
    try expectEqual(42, broadcasted.scalar(.{ 0, 0, 0 }));
    try expectEqual(42, broadcasted.scalar(.{ 1, 2, 3 }));
    try expectEqual(42, broadcasted.scalar(.{ 2, 3, 4 }));
}
