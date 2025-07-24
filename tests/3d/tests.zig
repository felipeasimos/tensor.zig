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

pub const TENSOR_3D = struct {
    test "check shape" {
        var data: [27]f64 = createSequence(f64, 27);
        const tensor = TensorView(f64, .{ 3, 3, 3 }).init(data[0..]);
        try expectEqual(.{ 3, 3, 3 }, tensor.shape);
    }
    test "check stride" {
        var data: [27]f64 = createSequence(f64, 27);
        const tensor = TensorView(f64, .{ 3, 3, 3 }).init(data[0..]);
        try expectEqual(.{ 9, 3, 1 }, tensor.strides);
    }
    test "indexing scalars" {
        var data: [27]f64 = createSequence(f64, 27);
        var tensor = TensorView(f64, .{ 3, 3, 3 }).init(data[0..]);

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
    test "mut sub tensor content (scalar)" {
        var data: [27]f64 = createSequence(f64, 27);
        var tensor = TensorView(f64, .{ 3, 3, 3 }).init(data[0..]);

        try expectEqual(data[0], tensor.mut(.{ 0, 0, 0 }));
        try expectEqual(data[1], tensor.mut(.{ 0, 0, 1 }));
        try expectEqual(data[2], tensor.mut(.{ 0, 0, 2 }));
        try expectEqual(data[3], tensor.mut(.{ 0, 1, 0 }));
        try expectEqual(data[4], tensor.mut(.{ 0, 1, 1 }));
        try expectEqual(data[5], tensor.mut(.{ 0, 1, 2 }));
        try expectEqual(data[6], tensor.mut(.{ 0, 2, 0 }));
        try expectEqual(data[7], tensor.mut(.{ 0, 2, 1 }));
        try expectEqual(data[8], tensor.mut(.{ 0, 2, 2 }));

        try expectEqual(data[9], tensor.mut(.{ 1, 0, 0 }));
        try expectEqual(data[10], tensor.mut(.{ 1, 0, 1 }));
        try expectEqual(data[11], tensor.mut(.{ 1, 0, 2 }));
        try expectEqual(data[12], tensor.mut(.{ 1, 1, 0 }));
        try expectEqual(data[13], tensor.mut(.{ 1, 1, 1 }));
        try expectEqual(data[14], tensor.mut(.{ 1, 1, 2 }));
        try expectEqual(data[15], tensor.mut(.{ 1, 2, 0 }));
        try expectEqual(data[16], tensor.mut(.{ 1, 2, 1 }));
        try expectEqual(data[17], tensor.mut(.{ 1, 2, 2 }));

        try expectEqual(data[18], tensor.mut(.{ 2, 0, 0 }));
        try expectEqual(data[19], tensor.mut(.{ 2, 0, 1 }));
        try expectEqual(data[20], tensor.mut(.{ 2, 0, 2 }));
        try expectEqual(data[21], tensor.mut(.{ 2, 1, 0 }));
        try expectEqual(data[22], tensor.mut(.{ 2, 1, 1 }));
        try expectEqual(data[23], tensor.mut(.{ 2, 1, 2 }));
        try expectEqual(data[24], tensor.mut(.{ 2, 2, 0 }));
        try expectEqual(data[25], tensor.mut(.{ 2, 2, 1 }));
        try expectEqual(data[26], tensor.mut(.{ 2, 2, 2 }));
    }
    test "mut sub tensor content (vector)" {
        var data: [27]f64 = createSequence(f64, 27);
        var tensor = TensorView(f64, .{ 3, 3, 3 }).init(data[0..]);

        try expectEqual(data[0..3], tensor.mut(.{ 0, 0 }).data);
        try expectEqual(data[3..6], tensor.mut(.{ 0, 1 }).data);
        try expectEqual(data[6..9], tensor.mut(.{ 0, 2 }).data);

        try expectEqual(data[9..12], tensor.mut(.{ 1, 0 }).data);
        try expectEqual(data[12..15], tensor.mut(.{ 1, 1 }).data);
        try expectEqual(data[15..18], tensor.mut(.{ 1, 2 }).data);

        try expectEqual(data[18..21], tensor.mut(.{ 2, 0 }).data);
        try expectEqual(data[21..24], tensor.mut(.{ 2, 1 }).data);
        try expectEqual(data[24..27], tensor.mut(.{ 2, 2 }).data);
    }
    test "mut sub tensor content (matrix)" {
        var data: [27]f64 = createSequence(f64, 27);
        var tensor = TensorView(f64, .{ 3, 3, 3 }).init(data[0..]);

        try expectEqual(data[0..9], tensor.mut(.{0}).data);

        try expectEqual(data[9..18], tensor.mut(.{1}).data);

        try expectEqual(data[18..27], tensor.mut(.{2}).data);
    }
    test "reshape 2x3x4 to 4x3x2" {
        var data: [24]f64 = createSequence(f64, 24);
        var tensor = TensorView(f64, .{ 2, 3, 4 }).init(data[0..]);

        try expectEqual(.{ 12, 4, 1 }, tensor.strides);

        var new_tensor = tensor.reshape(.{ 4, 3, 2 });

        try expectEqual(.{ 6, 2, 1 }, new_tensor.strides);

        try expectEqual(data[0..6], new_tensor.mut(.{0}).data);
        try expectEqual(data[6..12], new_tensor.mut(.{1}).data);
        try expectEqual(data[12..18], new_tensor.mut(.{2}).data);
        try expectEqual(data[18..24], new_tensor.mut(.{3}).data);
    }
    test "slice" {
        var data: [24]f64 = createSequence(f64, 24);
        var tensor = TensorView(f64, .{ 3, 4, 2 }).init(data[0..]);
        var subtensor = tensor.slice(.{
            .{ 1, 3 },
            .{ 2, 4 },
            .{ 0, 2 },
        });
        try expectEqual(data[12], subtensor.mut(.{ 0, 0, 0 }));
        try expectEqual(data[13], subtensor.mut(.{ 0, 0, 1 }));

        try expectEqual(data[14], subtensor.mut(.{ 0, 1, 0 }));
        try expectEqual(data[15], subtensor.mut(.{ 0, 1, 1 }));

        try expectEqual(data[20], subtensor.mut(.{ 1, 0, 0 }));
        try expectEqual(data[21], subtensor.mut(.{ 1, 0, 1 }));

        try expectEqual(data[22], subtensor.mut(.{ 1, 1, 0 }));
        try expectEqual(data[23], subtensor.mut(.{ 1, 1, 1 }));
    }
    test "mut muterence to matrix" {
        var data: [24]f64 = createSequence(f64, 24);
        var tensor = Tensor(f64, .{ 3, 4, 2 }).init(data[0..]);

        var subtensor1 = tensor.mut(.{0});

        try expectEqual(.{ 4, 2 }, subtensor1.shape);

        try expectEqual(data[0], subtensor1.scalar(.{ 0, 0 }).*);
        try expectEqual(data[1], subtensor1.scalar(.{ 0, 1 }).*);

        try expectEqual(data[2], subtensor1.scalar(.{ 1, 0 }).*);
        try expectEqual(data[3], subtensor1.scalar(.{ 1, 1 }).*);

        try expectEqual(data[4], subtensor1.scalar(.{ 2, 0 }).*);
        try expectEqual(data[5], subtensor1.scalar(.{ 2, 1 }).*);

        try expectEqual(data[6], subtensor1.scalar(.{ 3, 0 }).*);
        try expectEqual(data[7], subtensor1.scalar(.{ 3, 1 }).*);

        var subtensor2 = tensor.mut(.{1});

        try expectEqual(.{ 4, 2 }, subtensor2.shape);

        try expectEqual(data[8], subtensor2.scalar(.{ 0, 0 }).*);
        try expectEqual(data[9], subtensor2.scalar(.{ 0, 1 }).*);

        try expectEqual(data[10], subtensor2.scalar(.{ 1, 0 }).*);
        try expectEqual(data[11], subtensor2.scalar(.{ 1, 1 }).*);

        try expectEqual(data[12], subtensor2.scalar(.{ 2, 0 }).*);
        try expectEqual(data[13], subtensor2.scalar(.{ 2, 1 }).*);

        try expectEqual(data[14], subtensor2.scalar(.{ 3, 0 }).*);
        try expectEqual(data[15], subtensor2.scalar(.{ 3, 1 }).*);

        var subtensor3 = tensor.mut(.{2});

        try expectEqual(.{ 4, 2 }, subtensor3.shape);

        try expectEqual(data[16], subtensor3.scalar(.{ 0, 0 }).*);
        try expectEqual(data[17], subtensor3.scalar(.{ 0, 1 }).*);

        try expectEqual(data[18], subtensor3.scalar(.{ 1, 0 }).*);
        try expectEqual(data[19], subtensor3.scalar(.{ 1, 1 }).*);

        try expectEqual(data[20], subtensor3.scalar(.{ 2, 0 }).*);
        try expectEqual(data[21], subtensor3.scalar(.{ 2, 1 }).*);

        try expectEqual(data[22], subtensor3.scalar(.{ 3, 0 }).*);
        try expectEqual(data[23], subtensor3.scalar(.{ 3, 1 }).*);
    }

    test "matmul" {
        var data: [24]f64 = createSequence(f64, 24);
        var tensor = Tensor(f64, .{ 3, 4, 2 }).init(data[0..]);

        var subtensor1 = tensor.mut(.{0});
        try expectEqual(.{ 4, 2 }, subtensor1.shape);
        try expectEqual(8, subtensor1.num_scalars);
        var subtensor2 = tensor.mut(.{1});
        try expectEqual(.{ 4, 2 }, subtensor2.shape);
        try expectEqual(8, subtensor2.num_scalars);
        var mut_to_2 = subtensor2.mut(.{});
        try expectEqual(.{ 4, 2 }, mut_to_2.shape);
        try expectEqual(8, mut_to_2.num_scalars);
        const reshaped_mut_to_2 = mut_to_2.reshape(.{ 2, 4 });
        try expectEqual(.{ 2, 4 }, reshaped_mut_to_2.shape);
        try expectEqual(8, reshaped_mut_to_2.num_scalars);
        const result = op.matmulNew(&subtensor1, subtensor2.mut(.{}).reshape(.{ 2, 4 }));

        try expectEqual(.{ 4, 4 }, result.shape);
    }

    test "wise element-wise addition with scalar (in place)" {
        var data: [8]f64 = .{ 1, 2, 3, 4, 5, 6, 7, 8 };
        var tensor = Tensor(f64, .{ 2, 2, 2 }).init(data[0..]);
        var result = Tensor(f64, .{ 2, 2, 2 }).init(data[0..]);
        tensor.wise(10, &result, (struct {
            pub fn func(a: f64, b: f64) f64 {
                return a + b;
            }
        }).func);
        try expectEqual(11, result.clone(.{ 0, 0, 0 }));
        try expectEqual(12, result.clone(.{ 0, 0, 1 }));
        try expectEqual(13, result.clone(.{ 0, 1, 0 }));
        try expectEqual(14, result.clone(.{ 0, 1, 1 }));
        try expectEqual(15, result.clone(.{ 1, 0, 0 }));
        try expectEqual(16, result.clone(.{ 1, 0, 1 }));
        try expectEqual(17, result.clone(.{ 1, 1, 0 }));
        try expectEqual(18, result.clone(.{ 1, 1, 1 }));
    }
    test "wise element-wise addition with tensor (in place)" {
        var data1: [8]f64 = .{ 1, 2, 3, 4, 5, 6, 7, 8 };
        var data2: [8]f64 = .{ 10, 20, 30, 40, 50, 60, 70, 80 };
        var tensor1 = Tensor(f64, .{ 2, 2, 2 }).init(data1[0..]);
        var tensor2 = Tensor(f64, .{ 2, 2, 2 }).init(data2[0..]);
        var result = Tensor(f64, .{ 2, 2, 2 }).init(data1[0..]);
        tensor1.wise(&tensor2, &result, (struct {
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
        // tensor2 should remain unchanged
        try expectEqual(10, tensor2.clone(.{ 0, 0, 0 }));
        try expectEqual(20, tensor2.clone(.{ 0, 0, 1 }));
        try expectEqual(30, tensor2.clone(.{ 0, 1, 0 }));
        try expectEqual(40, tensor2.clone(.{ 0, 1, 1 }));
        try expectEqual(50, tensor2.clone(.{ 1, 0, 0 }));
        try expectEqual(60, tensor2.clone(.{ 1, 0, 1 }));
        try expectEqual(70, tensor2.clone(.{ 1, 1, 0 }));
        try expectEqual(80, tensor2.clone(.{ 1, 1, 1 }));
    }
    test "wiseNew element-wise addition with scalar" {
        var data: [8]f64 = .{ 1, 2, 3, 4, 5, 6, 7, 8 };
        const tensor = Tensor(f64, .{ 2, 2, 2 }).init(data[0..]);
        const result = tensor.wiseNew(10, (struct {
            pub fn func(a: f64, b: f64) f64 {
                return a + b;
            }
        }).func);
        try expectEqual(11, result.clone(.{ 0, 0, 0 }));
        try expectEqual(12, result.clone(.{ 0, 0, 1 }));
        try expectEqual(13, result.clone(.{ 0, 1, 0 }));
        try expectEqual(14, result.clone(.{ 0, 1, 1 }));
        try expectEqual(15, result.clone(.{ 1, 0, 0 }));
        try expectEqual(16, result.clone(.{ 1, 0, 1 }));
        try expectEqual(17, result.clone(.{ 1, 1, 0 }));
        try expectEqual(18, result.clone(.{ 1, 1, 1 }));
        // Original tensor should remain unchanged
        try expectEqual(1, tensor.clone(.{ 0, 0, 0 }));
        try expectEqual(2, tensor.clone(.{ 0, 0, 1 }));
        try expectEqual(3, tensor.clone(.{ 0, 1, 0 }));
        try expectEqual(4, tensor.clone(.{ 0, 1, 1 }));
        try expectEqual(5, tensor.clone(.{ 1, 0, 0 }));
        try expectEqual(6, tensor.clone(.{ 1, 0, 1 }));
        try expectEqual(7, tensor.clone(.{ 1, 1, 0 }));
        try expectEqual(8, tensor.clone(.{ 1, 1, 1 }));
    }
    test "wiseNew element-wise addition with tensor" {
        var data1: [8]f64 = .{ 1, 2, 3, 4, 5, 6, 7, 8 };
        var data2: [8]f64 = .{ 10, 20, 30, 40, 50, 60, 70, 80 };
        const tensor1 = Tensor(f64, .{ 2, 2, 2 }).init(data1[0..]);
        const tensor2 = Tensor(f64, .{ 2, 2, 2 }).init(data2[0..]);
        const result = tensor1.wiseNew(&tensor2, (struct {
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
        // Original tensors should remain unchanged
        try expectEqual(1, tensor1.clone(.{ 0, 0, 0 }));
        try expectEqual(2, tensor1.clone(.{ 0, 0, 1 }));
        try expectEqual(3, tensor1.clone(.{ 0, 1, 0 }));
        try expectEqual(4, tensor1.clone(.{ 0, 1, 1 }));
        try expectEqual(5, tensor1.clone(.{ 1, 0, 0 }));
        try expectEqual(6, tensor1.clone(.{ 1, 0, 1 }));
        try expectEqual(7, tensor1.clone(.{ 1, 1, 0 }));
        try expectEqual(8, tensor1.clone(.{ 1, 1, 1 }));
        try expectEqual(10, tensor2.clone(.{ 0, 0, 0 }));
        try expectEqual(20, tensor2.clone(.{ 0, 0, 1 }));
        try expectEqual(30, tensor2.clone(.{ 0, 1, 0 }));
        try expectEqual(40, tensor2.clone(.{ 0, 1, 1 }));
        try expectEqual(50, tensor2.clone(.{ 1, 0, 0 }));
        try expectEqual(60, tensor2.clone(.{ 1, 0, 1 }));
        try expectEqual(70, tensor2.clone(.{ 1, 1, 0 }));
        try expectEqual(80, tensor2.clone(.{ 1, 1, 1 }));
    }
    test "broadcast 3D [1,2,3] to [2,2,3]" {
        var data: [6]f64 = .{ 1, 2, 3, 4, 5, 6 };
        var tensor = TensorView(f64, .{ 1, 2, 3 }).init(data[0..]);
        const broadcasted = tensor.broadcast(.{ 2, 2, 3 });

        try expectEqual(.{ 2, 2, 3 }, broadcasted.shape);
        try expectEqual(.{ 0, 3, 1 }, broadcasted.strides);
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
        var tensor = TensorView(f64, .{ 2, 1, 1 }).init(data[0..]);
        const broadcasted = tensor.broadcast(.{ 2, 3, 4 });

        try expectEqual(.{ 2, 3, 4 }, broadcasted.shape);
        try expectEqual(.{ 1, 0, 0 }, broadcasted.strides);
        try expectEqual(100, broadcasted.scalar(.{ 0, 0, 0 }));
        try expectEqual(100, broadcasted.scalar(.{ 0, 1, 1 }));
        try expectEqual(100, broadcasted.scalar(.{ 0, 2, 3 }));
        try expectEqual(200, broadcasted.scalar(.{ 1, 0, 0 }));
        try expectEqual(200, broadcasted.scalar(.{ 1, 1, 1 }));
        try expectEqual(200, broadcasted.scalar(.{ 1, 2, 3 }));
    }
    test "broadcast 3D [1,1,3] to [2,4,3]" {
        var data: [3]f64 = .{ 7, 8, 9 };
        var tensor = TensorView(f64, .{ 1, 1, 3 }).init(data[0..]);
        const broadcasted = tensor.broadcast(.{ 2, 4, 3 });

        try expectEqual(.{ 2, 4, 3 }, broadcasted.shape);
        try expectEqual(.{ 0, 0, 1 }, broadcasted.strides);
        try expectEqual(7, broadcasted.scalar(.{ 0, 0, 0 }));
        try expectEqual(8, broadcasted.scalar(.{ 0, 0, 1 }));
        try expectEqual(9, broadcasted.scalar(.{ 0, 0, 2 }));
        try expectEqual(7, broadcasted.scalar(.{ 0, 3, 0 }));
        try expectEqual(7, broadcasted.scalar(.{ 1, 0, 0 }));
        try expectEqual(9, broadcasted.scalar(.{ 1, 3, 2 }));
    }
    test "broadcast 3D [1,1,1] to [3,4,5]" {
        var data: [1]f64 = .{42};
        var tensor = TensorView(f64, .{ 1, 1, 1 }).init(data[0..]);
        const broadcasted = tensor.broadcast(.{ 3, 4, 5 });

        try expectEqual(.{ 3, 4, 5 }, broadcasted.shape);
        try expectEqual(.{ 0, 0, 0 }, broadcasted.strides);
        try expectEqual(42, broadcasted.scalar(.{ 0, 0, 0 }));
        try expectEqual(42, broadcasted.scalar(.{ 1, 2, 3 }));
        try expectEqual(42, broadcasted.scalar(.{ 2, 3, 4 }));
    }
};
