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

test {
    std.testing.refAllDeclsRecursive(@This());
    _ = TENSOR_1D;
    _ = TENSOR_2D;
    _ = TENSOR_3D;
}

const TENSOR_1D = struct {
    test "check shape" {
        var data: [9]f64 = createSequence(f64, 9);
        const tensor = TensorView(f64, .{3}).init(data[0..]);
        try expectEqual(.{3}, tensor.shape);
    }
    test "check stride" {
        var data: [9]f64 = createSequence(f64, 9);
        const tensor = TensorView(f64, .{3}).init(data[0..]);
        try expectEqual(.{1}, tensor.strides);
    }
    test "indexing scalars" {
        var data: [9]f64 = createSequence(f64, 9);
        var tensor = TensorView(f64, .{3}).init(data[0..]);
        try expectEqual(data[0], tensor.scalar(.{0}));
        try expectEqual(data[1], tensor.scalar(.{1}));
        try expectEqual(data[2], tensor.scalar(.{2}));
    }
    test "get sub tensor content (scalar)" {
        var data: [9]f64 = createSequence(f64, 9);
        var tensor = TensorView(f64, .{3}).init(data[0..]);
        try expectEqual(data[0], tensor.get(.{0}));
        try expectEqual(data[1], tensor.get(.{1}));
        try expectEqual(data[2], tensor.get(.{2}));
    }
    test "slice" {
        var data: [9]f64 = createSequence(f64, 9);
        var tensor = TensorView(f64, .{3}).init(data[0..]);
        var subtensor = tensor.slice(.{.{ 1, 3 }});
        try expectEqual(data[1], subtensor.get(.{0}));
        try expectEqual(data[2], subtensor.get(.{1}));
    }
    test "get tensor from view" {
        var data: [3]f64 = createSequence(f64, 3);
        var view = TensorView(f64, .{3}).init(data[0..]);
        var tensor = view.mut();
        try expectEqual(tensor.data[0], tensor.clone(.{0}));
        try expectEqual(tensor.data[1], tensor.clone(.{1}));
        try expectEqual(tensor.data[2], tensor.clone(.{2}));

        try expectEqual(tensor.data[0], tensor.clone(.{0}));
        try expectEqual(tensor.data[1], tensor.clone(.{1}));
        try expectEqual(tensor.data[2], tensor.clone(.{2}));
    }
    test "get view from tensor from view" {
        var data: [3]f64 = createSequence(f64, 3);
        var view_tmp = TensorView(f64, .{3}).init(data[0..]);
        var tensor_tmp = view_tmp.mut();
        var view = tensor_tmp.view();
        try expectEqual(view.data[0], view.get(.{0}));
        try expectEqual(view.data[1], view.get(.{1}));
        try expectEqual(view.data[2], view.get(.{2}));
    }
    test "element wise operation with a scalar" {
        var data: [3]f64 = createSequence(f64, 3);
        var tensor1 = Tensor(f64, .{3}).init(data[0..]);
        _ = tensor1.wise(2, (struct {
            pub fn func(a: f64, b: f64) f64 {
                return a + b;
            }
        }).func);
        try expectEqual(2, tensor1.clone(.{0}));
        try expectEqual(3, tensor1.clone(.{1}));
        try expectEqual(4, tensor1.clone(.{2}));
    }
    test "element wise operation with a tensor, in_place" {
        var data: [3]f64 = createSequence(f64, 3);
        var tensor1 = Tensor(f64, .{3}).init(data[0..]);
        var tensor2 = Tensor(f64, .{3}).init(data[0..]);
        _ = tensor1.wise(&tensor2, (struct {
            pub fn func(a: f64, b: f64) f64 {
                return a + b;
            }
        }).func);
        try expectEqual(0, tensor1.clone(.{0}));
        try expectEqual(2, tensor1.clone(.{1}));
        try expectEqual(4, tensor1.clone(.{2}));

        try expectEqual(0, tensor2.clone(.{0}));
        try expectEqual(1, tensor2.clone(.{1}));
        try expectEqual(2, tensor2.clone(.{2}));
    }
};

const TENSOR_2D = struct {
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
    test "get sub tensor content (scalar)" {
        var data: [9]f64 = createSequence(f64, 9);
        var tensor = TensorView(f64, .{ 3, 3 }).init(data[0..]);

        try expectEqual(data[0], tensor.get(.{ 0, 0 }));
        try expectEqual(data[1], tensor.get(.{ 0, 1 }));
        try expectEqual(data[2], tensor.get(.{ 0, 2 }));

        try expectEqual(data[3], tensor.get(.{ 1, 0 }));
        try expectEqual(data[4], tensor.get(.{ 1, 1 }));
        try expectEqual(data[5], tensor.get(.{ 1, 2 }));

        try expectEqual(data[6], tensor.get(.{ 2, 0 }));
        try expectEqual(data[7], tensor.get(.{ 2, 1 }));
        try expectEqual(data[8], tensor.get(.{ 2, 2 }));
    }
    test "get sub tensor content (vector)" {
        var data: [9]f64 = createSequence(f64, 9);
        var tensor = TensorView(f64, .{ 3, 3 }).init(data[0..]);

        try expectEqual(data[0..3], tensor.get(.{0}).data);
        try expectEqual(data[3..6], tensor.get(.{1}).data);
        try expectEqual(data[6..9], tensor.get(.{2}).data);
    }
    test "reshape 4x4 to 2x8" {
        var data: [16]f64 = createSequence(f64, 16);
        var tensor = TensorView(f64, .{ 4, 4 }).init(data[0..]);

        var new_tensor = tensor.reshape(.{ 2, 8 });

        try expectEqual(data[0..8], new_tensor.get(.{0}).data);
        try expectEqual(data[8..16], new_tensor.get(.{1}).data);
    }
    test "slice" {
        var data: [16]f64 = createSequence(f64, 16);
        var tensor = TensorView(f64, .{ 4, 4 }).init(data[0..]);
        var subtensor = tensor.slice(.{
            .{ 1, 3 },
            .{ 1, 3 },
        });
        try expectEqual(data[5], subtensor.get(.{ 0, 0 }));
        try expectEqual(data[6], subtensor.get(.{ 0, 1 }));
        try expectEqual(data[9], subtensor.get(.{ 1, 0 }));
        try expectEqual(data[10], subtensor.get(.{ 1, 1 }));
    }
    test "matmul 3x4 4x2" {
        var data1: [12]f64 = createSequence(f64, 12);
        var tensor1 = Tensor(f64, .{ 3, 4 }).init(&data1);
        var data2: [8]f64 = createSequence(f64, 8);
        var tensor2 = Tensor(f64, .{ 4, 2 }).init(&data2);

        var data3: [6]f64 = createSequence(f64, 6);
        var result = Tensor(f64, .{ 3, 2 }).init(&data3);
        tensor1.matmul(&tensor2, &result);
        try expectEqual(result.shape, [_]usize{ 3, 2 });
    }
    test "matmulNew 3x4 4x2" {
        var data1: [12]f64 = createSequence(f64, 12);
        var tensor1 = Tensor(f64, .{ 3, 4 }).init(&data1);
        var data2: [8]f64 = createSequence(f64, 8);
        var tensor2 = Tensor(f64, .{ 4, 2 }).init(&data2);

        const result = tensor1.matmulNew(&tensor2);
        try expectEqual(result.shape, [_]usize{ 3, 2 });
    }
};

const TENSOR_3D = struct {
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
    test "get sub tensor content (scalar)" {
        var data: [27]f64 = createSequence(f64, 27);
        var tensor = TensorView(f64, .{ 3, 3, 3 }).init(data[0..]);

        try expectEqual(data[0], tensor.get(.{ 0, 0, 0 }));
        try expectEqual(data[1], tensor.get(.{ 0, 0, 1 }));
        try expectEqual(data[2], tensor.get(.{ 0, 0, 2 }));
        try expectEqual(data[3], tensor.get(.{ 0, 1, 0 }));
        try expectEqual(data[4], tensor.get(.{ 0, 1, 1 }));
        try expectEqual(data[5], tensor.get(.{ 0, 1, 2 }));
        try expectEqual(data[6], tensor.get(.{ 0, 2, 0 }));
        try expectEqual(data[7], tensor.get(.{ 0, 2, 1 }));
        try expectEqual(data[8], tensor.get(.{ 0, 2, 2 }));

        try expectEqual(data[9], tensor.get(.{ 1, 0, 0 }));
        try expectEqual(data[10], tensor.get(.{ 1, 0, 1 }));
        try expectEqual(data[11], tensor.get(.{ 1, 0, 2 }));
        try expectEqual(data[12], tensor.get(.{ 1, 1, 0 }));
        try expectEqual(data[13], tensor.get(.{ 1, 1, 1 }));
        try expectEqual(data[14], tensor.get(.{ 1, 1, 2 }));
        try expectEqual(data[15], tensor.get(.{ 1, 2, 0 }));
        try expectEqual(data[16], tensor.get(.{ 1, 2, 1 }));
        try expectEqual(data[17], tensor.get(.{ 1, 2, 2 }));

        try expectEqual(data[18], tensor.get(.{ 2, 0, 0 }));
        try expectEqual(data[19], tensor.get(.{ 2, 0, 1 }));
        try expectEqual(data[20], tensor.get(.{ 2, 0, 2 }));
        try expectEqual(data[21], tensor.get(.{ 2, 1, 0 }));
        try expectEqual(data[22], tensor.get(.{ 2, 1, 1 }));
        try expectEqual(data[23], tensor.get(.{ 2, 1, 2 }));
        try expectEqual(data[24], tensor.get(.{ 2, 2, 0 }));
        try expectEqual(data[25], tensor.get(.{ 2, 2, 1 }));
        try expectEqual(data[26], tensor.get(.{ 2, 2, 2 }));
    }
    test "get sub tensor content (vector)" {
        var data: [27]f64 = createSequence(f64, 27);
        var tensor = TensorView(f64, .{ 3, 3, 3 }).init(data[0..]);

        try expectEqual(data[0..3], tensor.get(.{ 0, 0 }).data);
        try expectEqual(data[3..6], tensor.get(.{ 0, 1 }).data);
        try expectEqual(data[6..9], tensor.get(.{ 0, 2 }).data);

        try expectEqual(data[9..12], tensor.get(.{ 1, 0 }).data);
        try expectEqual(data[12..15], tensor.get(.{ 1, 1 }).data);
        try expectEqual(data[15..18], tensor.get(.{ 1, 2 }).data);

        try expectEqual(data[18..21], tensor.get(.{ 2, 0 }).data);
        try expectEqual(data[21..24], tensor.get(.{ 2, 1 }).data);
        try expectEqual(data[24..27], tensor.get(.{ 2, 2 }).data);
    }
    test "get sub tensor content (matrix)" {
        var data: [27]f64 = createSequence(f64, 27);
        var tensor = TensorView(f64, .{ 3, 3, 3 }).init(data[0..]);

        try expectEqual(data[0..9], tensor.get(.{0}).data);

        try expectEqual(data[9..18], tensor.get(.{1}).data);

        try expectEqual(data[18..27], tensor.get(.{2}).data);
    }
    test "reshape 2x3x4 to 4x3x2" {
        var data: [24]f64 = createSequence(f64, 24);
        var tensor = TensorView(f64, .{ 2, 3, 4 }).init(data[0..]);

        var new_tensor = tensor.reshape(.{ 4, 3, 2 });

        try expectEqual(data[0..6], new_tensor.get(.{0}).data);
        try expectEqual(data[6..12], new_tensor.get(.{1}).data);
        try expectEqual(data[12..18], new_tensor.get(.{2}).data);
        try expectEqual(data[18..24], new_tensor.get(.{3}).data);
    }
    test "slice" {
        var data: [24]f64 = createSequence(f64, 24);
        var tensor = TensorView(f64, .{ 3, 4, 2 }).init(data[0..]);
        var subtensor = tensor.slice(.{
            .{ 1, 3 },
            .{ 2, 4 },
            .{ 0, 2 },
        });
        try expectEqual(data[12], subtensor.get(.{ 0, 0, 0 }));
        try expectEqual(data[13], subtensor.get(.{ 0, 0, 1 }));

        try expectEqual(data[14], subtensor.get(.{ 0, 1, 0 }));
        try expectEqual(data[15], subtensor.get(.{ 0, 1, 1 }));

        try expectEqual(data[20], subtensor.get(.{ 1, 0, 0 }));
        try expectEqual(data[21], subtensor.get(.{ 1, 0, 1 }));

        try expectEqual(data[22], subtensor.get(.{ 1, 1, 0 }));
        try expectEqual(data[23], subtensor.get(.{ 1, 1, 1 }));
    }
    test "get reference to matrix" {
        var data: [24]f64 = createSequence(f64, 24);
        var tensor = Tensor(f64, .{ 3, 4, 2 }).init(data[0..]);

        var subtensor1 = tensor.ref(.{0});

        try expectEqual(.{ 4, 2 }, subtensor1.shape);

        try expectEqual(data[0], subtensor1.scalar(.{ 0, 0 }).*);
        try expectEqual(data[1], subtensor1.scalar(.{ 0, 1 }).*);

        try expectEqual(data[2], subtensor1.scalar(.{ 1, 0 }).*);
        try expectEqual(data[3], subtensor1.scalar(.{ 1, 1 }).*);

        try expectEqual(data[4], subtensor1.scalar(.{ 2, 0 }).*);
        try expectEqual(data[5], subtensor1.scalar(.{ 2, 1 }).*);

        try expectEqual(data[6], subtensor1.scalar(.{ 3, 0 }).*);
        try expectEqual(data[7], subtensor1.scalar(.{ 3, 1 }).*);

        var subtensor2 = tensor.ref(.{1});

        try expectEqual(.{ 4, 2 }, subtensor2.shape);

        try expectEqual(data[8], subtensor2.scalar(.{ 0, 0 }).*);
        try expectEqual(data[9], subtensor2.scalar(.{ 0, 1 }).*);

        try expectEqual(data[10], subtensor2.scalar(.{ 1, 0 }).*);
        try expectEqual(data[11], subtensor2.scalar(.{ 1, 1 }).*);

        try expectEqual(data[12], subtensor2.scalar(.{ 2, 0 }).*);
        try expectEqual(data[13], subtensor2.scalar(.{ 2, 1 }).*);

        try expectEqual(data[14], subtensor2.scalar(.{ 3, 0 }).*);
        try expectEqual(data[15], subtensor2.scalar(.{ 3, 1 }).*);

        var subtensor3 = tensor.ref(.{2});

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
};
