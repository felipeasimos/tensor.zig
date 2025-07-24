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

pub const TENSOR_2D = struct {
    test "check shape" {
        var data: [9]f64 = createSequence(f64, 9);
        const tensor = TensorView(f64, .{ 3, 3 }).init(data[0..]);
        try expectEqual(.{ 3, 3 }, tensor.shape);
    }
    test "check stride" {
        var data: [9]f64 = createSequence(f64, 9);
        const tensor = TensorView(f64, .{ 3, 3 }).init(data[0..]);
        try expectEqual(.{ 3, 1 }, tensor.strides);
    }
    test "indexing scalars" {
        var data: [9]f64 = createSequence(f64, 9);
        var tensor = TensorView(f64, .{ 3, 3 }).init(data[0..]);

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
    test "mut sub tensor content (scalar)" {
        var data: [9]f64 = createSequence(f64, 9);
        var tensor = TensorView(f64, .{ 3, 3 }).init(data[0..]);

        try expectEqual(data[0], tensor.mut(.{ 0, 0 }));
        try expectEqual(data[1], tensor.mut(.{ 0, 1 }));
        try expectEqual(data[2], tensor.mut(.{ 0, 2 }));

        try expectEqual(data[3], tensor.mut(.{ 1, 0 }));
        try expectEqual(data[4], tensor.mut(.{ 1, 1 }));
        try expectEqual(data[5], tensor.mut(.{ 1, 2 }));

        try expectEqual(data[6], tensor.mut(.{ 2, 0 }));
        try expectEqual(data[7], tensor.mut(.{ 2, 1 }));
        try expectEqual(data[8], tensor.mut(.{ 2, 2 }));
    }
    test "mut sub tensor content (vector)" {
        var data: [9]f64 = createSequence(f64, 9);
        var tensor = TensorView(f64, .{ 3, 3 }).init(data[0..]);

        try expectEqual(data[0..3], tensor.mut(.{0}).data);
        try expectEqual(data[3..6], tensor.mut(.{1}).data);
        try expectEqual(data[6..9], tensor.mut(.{2}).data);
    }
    test "reshape 4x4 to 2x8" {
        var data: [16]f64 = createSequence(f64, 16);
        var tensor = TensorView(f64, .{ 4, 4 }).init(data[0..]);

        var new_tensor = tensor.reshape(.{ 2, 8 });

        try expectEqual(data[0..8], new_tensor.mut(.{0}).data);
        try expectEqual(data[8..16], new_tensor.mut(.{1}).data);
    }
    test "slice" {
        var data: [16]f64 = createSequence(f64, 16);
        var tensor = TensorView(f64, .{ 4, 4 }).init(data[0..]);
        var subtensor = tensor.slice(.{
            .{ 1, 3 },
            .{ 1, 3 },
        });
        try expectEqual(data[5], subtensor.mut(.{ 0, 0 }));
        try expectEqual(data[6], subtensor.mut(.{ 0, 1 }));
        try expectEqual(data[9], subtensor.mut(.{ 1, 0 }));
        try expectEqual(data[10], subtensor.mut(.{ 1, 1 }));
    }
    test "matmul 3x4 4x2" {
        var data1: [12]f64 = createSequence(f64, 12);
        var tensor1 = Tensor(f64, .{ 3, 4 }).init(&data1);
        var data2: [8]f64 = createSequence(f64, 8);
        var tensor2 = Tensor(f64, .{ 4, 2 }).init(&data2);

        var data3: [6]f64 = createSequence(f64, 6);
        var result = Tensor(f64, .{ 3, 2 }).init(&data3);
        op.matmul(&tensor1, &tensor2, &result);
        try expectEqual(result.shape, [_]usize{ 3, 2 });
    }
    test "matmulNew 3x4 4x2" {
        var data1: [12]f64 = createSequence(f64, 12);
        var tensor1 = Tensor(f64, .{ 3, 4 }).init(&data1);
        var data2: [8]f64 = createSequence(f64, 8);
        var tensor2 = Tensor(f64, .{ 4, 2 }).init(&data2);

        const result = op.matmulNew(&tensor1, &tensor2);
        try expectEqual(result.shape, [_]usize{ 3, 2 });
    }
    test "transpose" {
        var data: [12]f64 = createSequence(f64, 12);
        var tensor = Tensor(f64, .{ 3, 4 }).init(&data);
        const transpose = tensor.transpose(.{});
        try expectEqual(.{ 4, 3 }, transpose.shape);

        try expectEqual(data[0], transpose.clone(.{ 0, 0 }));
        try expectEqual(data[4], transpose.clone(.{ 0, 1 }));
        try expectEqual(data[8], transpose.clone(.{ 0, 2 }));

        try expectEqual(data[1], transpose.clone(.{ 1, 0 }));
        try expectEqual(data[5], transpose.clone(.{ 1, 1 }));
        try expectEqual(data[9], transpose.clone(.{ 1, 2 }));

        try expectEqual(data[2], transpose.clone(.{ 2, 0 }));
        try expectEqual(data[6], transpose.clone(.{ 2, 1 }));
        try expectEqual(data[10], transpose.clone(.{ 2, 2 }));

        try expectEqual(data[3], transpose.clone(.{ 3, 0 }));
        try expectEqual(data[7], transpose.clone(.{ 3, 1 }));
        try expectEqual(data[11], transpose.clone(.{ 3, 2 }));
    }
    test "wise element-wise addition with scalar (in place)" {
        var data: [4]f64 = .{ 1, 2, 3, 4 };
        var tensor = Tensor(f64, .{ 2, 2 }).init(data[0..]);
        var result = Tensor(f64, .{ 2, 2 }).init(data[0..]);
        tensor.wise(10, &result, (struct {
            pub fn func(a: f64, b: f64) f64 {
                return a + b;
            }
        }).func);
        try expectEqual(11, result.clone(.{ 0, 0 }));
        try expectEqual(12, result.clone(.{ 0, 1 }));
        try expectEqual(13, result.clone(.{ 1, 0 }));
        try expectEqual(14, result.clone(.{ 1, 1 }));
    }
    test "wise element-wise addition with tensor (in place)" {
        var data1: [4]f64 = .{ 1, 2, 3, 4 };
        var data2: [4]f64 = .{ 10, 20, 30, 40 };
        var tensor1 = Tensor(f64, .{ 2, 2 }).init(data1[0..]);
        var tensor2 = Tensor(f64, .{ 2, 2 }).init(data2[0..]);
        var result = Tensor(f64, .{ 2, 2 }).init(data1[0..]);
        tensor1.wise(&tensor2, &result, (struct {
            pub fn func(a: f64, b: f64) f64 {
                return a + b;
            }
        }).func);
        try expectEqual(11, result.clone(.{ 0, 0 }));
        try expectEqual(22, result.clone(.{ 0, 1 }));
        try expectEqual(33, result.clone(.{ 1, 0 }));
        try expectEqual(44, result.clone(.{ 1, 1 }));
        // tensor2 should remain unchanged
        try expectEqual(10, tensor2.clone(.{ 0, 0 }));
        try expectEqual(20, tensor2.clone(.{ 0, 1 }));
        try expectEqual(30, tensor2.clone(.{ 1, 0 }));
        try expectEqual(40, tensor2.clone(.{ 1, 1 }));
    }
    test "wiseNew element-wise addition with scalar" {
        var data: [4]f64 = .{ 1, 2, 3, 4 };
        const tensor = Tensor(f64, .{ 2, 2 }).init(data[0..]);
        const result = tensor.wiseNew(10, (struct {
            pub fn func(a: f64, b: f64) f64 {
                return a + b;
            }
        }).func);
        try expectEqual(11, result.clone(.{ 0, 0 }));
        try expectEqual(12, result.clone(.{ 0, 1 }));
        try expectEqual(13, result.clone(.{ 1, 0 }));
        try expectEqual(14, result.clone(.{ 1, 1 }));
        // Original tensor should remain unchanged
        try expectEqual(1, tensor.clone(.{ 0, 0 }));
        try expectEqual(2, tensor.clone(.{ 0, 1 }));
        try expectEqual(3, tensor.clone(.{ 1, 0 }));
        try expectEqual(4, tensor.clone(.{ 1, 1 }));
    }
    test "wiseNew element-wise addition with tensor" {
        var data1: [4]f64 = .{ 1, 2, 3, 4 };
        var data2: [4]f64 = .{ 10, 20, 30, 40 };
        const tensor1 = Tensor(f64, .{ 2, 2 }).init(data1[0..]);
        const tensor2 = Tensor(f64, .{ 2, 2 }).init(data2[0..]);
        const result = tensor1.wiseNew(&tensor2, (struct {
            pub fn func(a: f64, b: f64) f64 {
                return a + b;
            }
        }).func);
        try expectEqual(11, result.clone(.{ 0, 0 }));
        try expectEqual(22, result.clone(.{ 0, 1 }));
        try expectEqual(33, result.clone(.{ 1, 0 }));
        try expectEqual(44, result.clone(.{ 1, 1 }));
        // Original tensors should remain unchanged
        try expectEqual(1, tensor1.clone(.{ 0, 0 }));
        try expectEqual(2, tensor1.clone(.{ 0, 1 }));
        try expectEqual(3, tensor1.clone(.{ 1, 0 }));
        try expectEqual(4, tensor1.clone(.{ 1, 1 }));
        try expectEqual(10, tensor2.clone(.{ 0, 0 }));
        try expectEqual(20, tensor2.clone(.{ 0, 1 }));
        try expectEqual(30, tensor2.clone(.{ 1, 0 }));
        try expectEqual(40, tensor2.clone(.{ 1, 1 }));
    }
    test "broadcast 2D [1,3] to [2,3]" {
        var data: [3]f64 = .{ 1, 2, 3 };
        var tensor = TensorView(f64, .{ 1, 3 }).init(data[0..]);
        const broadcasted = tensor.broadcast(.{ 2, 3 });

        try expectEqual(.{ 2, 3 }, broadcasted.shape);
        try expectEqual(.{ 0, 1 }, broadcasted.strides);
        try expectEqual(1, broadcasted.scalar(.{ 0, 0 }));
        try expectEqual(2, broadcasted.scalar(.{ 0, 1 }));
        try expectEqual(3, broadcasted.scalar(.{ 0, 2 }));
        try expectEqual(1, broadcasted.scalar(.{ 1, 0 }));
        try expectEqual(2, broadcasted.scalar(.{ 1, 1 }));
        try expectEqual(3, broadcasted.scalar(.{ 1, 2 }));
    }
    test "broadcast 2D [2,1] to [2,4]" {
        var data: [2]f64 = .{ 10, 20 };
        var tensor = TensorView(f64, .{ 2, 1 }).init(data[0..]);
        const broadcasted = tensor.broadcast(.{ 2, 4 });

        try expectEqual(.{ 2, 4 }, broadcasted.shape);
        try expectEqual(.{ 1, 0 }, broadcasted.strides);
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
        var tensor = TensorView(f64, .{ 1, 1 }).init(data[0..]);
        const broadcasted = tensor.broadcast(.{ 3, 4 });

        try expectEqual(.{ 3, 4 }, broadcasted.shape);
        try expectEqual(.{ 0, 0 }, broadcasted.strides);
        try expectEqual(99, broadcasted.scalar(.{ 0, 0 }));
        try expectEqual(99, broadcasted.scalar(.{ 0, 3 }));
        try expectEqual(99, broadcasted.scalar(.{ 2, 0 }));
        try expectEqual(99, broadcasted.scalar(.{ 2, 3 }));
    }
    test "broadcast 2D to 3D" {
        var data: [6]f64 = .{ 1, 2, 3, 4, 5, 6 };
        var tensor = TensorView(f64, .{ 2, 3 }).init(data[0..]);
        const broadcasted = tensor.broadcast(.{ 4, 2, 3 });

        try expectEqual(.{ 4, 2, 3 }, broadcasted.shape);
        try expectEqual(.{ 0, 3, 1 }, broadcasted.strides);
        try expectEqual(1, broadcasted.scalar(.{ 0, 0, 0 }));
        try expectEqual(2, broadcasted.scalar(.{ 0, 0, 1 }));
        try expectEqual(4, broadcasted.scalar(.{ 0, 1, 0 }));
        try expectEqual(1, broadcasted.scalar(.{ 3, 0, 0 }));
        try expectEqual(4, broadcasted.scalar(.{ 3, 1, 0 }));
    }
};
