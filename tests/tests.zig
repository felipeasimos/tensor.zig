const std = @import("std");
const expectEqual = std.testing.expectEqual;
const TensorView = @import("tensor").TensorView;
const createSequence = @import("tensor").createSequence;

test {
    std.testing.refAllDeclsRecursive(@This());
    _ = TENSOR_1D;
    _ = TENSOR_2D;
    _ = TENSOR_3D;
}

const TENSOR_1D = struct {
    test "check shape" {
        var data: [3]f64 = .{ 1, 2, 3 };
        const tensor = TensorView(f64, .{3}).init(data[0..]);
        try expectEqual(.{3}, tensor.shape);
    }
    test "check stride" {
        var data: [3]f64 = .{ 1, 2, 3 };
        const tensor = TensorView(f64, .{3}).init(data[0..]);
        try expectEqual(.{1}, tensor.strides);
    }
    test "indexing" {
        var data: [3]f64 = .{ 1, 2, 3 };
        var tensor = TensorView(f64, .{3}).init(data[0..]);
        try expectEqual(1, tensor.scalar(.{0}));
        try expectEqual(2, tensor.scalar(.{1}));
        try expectEqual(3, tensor.scalar(.{2}));
    }
    test "get sub tensor (scalar)" {
        var data: [3]f64 = .{ 1, 2, 3 };
        var tensor = TensorView(f64, .{3}).init(data[0..]);
        try expectEqual(&data[2], tensor.get(.{2}));
    }
};

const TENSOR_2D = struct {
    test "check shape" {
        var data: [9]f64 = .{ 1, 2, 3, 4, 5, 6, 7, 8, 9 };
        const tensor = TensorView(f64, .{ 3, 3 }).init(data[0..]);
        try expectEqual(.{ 3, 3 }, tensor.shape);
    }
    test "check stride" {
        var data: [9]f64 = .{ 1, 2, 3, 4, 5, 6, 7, 8, 9 };
        const tensor = TensorView(f64, .{ 3, 3 }).init(data[0..]);
        try expectEqual(.{ 3, 1 }, tensor.strides);
    }
    test "indexing" {
        var data: [9]f64 = .{ 1, 2, 3, 4, 5, 6, 7, 8, 9 };
        var tensor = TensorView(f64, .{ 3, 3 }).init(data[0..]);

        try expectEqual(1, tensor.scalar(.{ 0, 0 }));
        try expectEqual(2, tensor.scalar(.{ 0, 1 }));
        try expectEqual(3, tensor.scalar(.{ 0, 2 }));

        try expectEqual(4, tensor.scalar(.{ 1, 0 }));
        try expectEqual(5, tensor.scalar(.{ 1, 1 }));
        try expectEqual(6, tensor.scalar(.{ 1, 2 }));

        try expectEqual(7, tensor.scalar(.{ 2, 0 }));
        try expectEqual(8, tensor.scalar(.{ 2, 1 }));
        try expectEqual(9, tensor.scalar(.{ 2, 2 }));
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
    test "indexing" {
        var data: [27]f64 = createSequence(f64, 27);
        var tensor = TensorView(f64, .{ 3, 3, 3 }).init(data[0..]);

        try expectEqual(0, tensor.scalar(.{ 0, 0, 0 }));
        try expectEqual(1, tensor.scalar(.{ 0, 0, 1 }));
        try expectEqual(2, tensor.scalar(.{ 0, 0, 2 }));
        try expectEqual(3, tensor.scalar(.{ 0, 1, 0 }));
        try expectEqual(4, tensor.scalar(.{ 0, 1, 1 }));
        try expectEqual(5, tensor.scalar(.{ 0, 1, 2 }));
        try expectEqual(6, tensor.scalar(.{ 0, 2, 0 }));
        try expectEqual(7, tensor.scalar(.{ 0, 2, 1 }));
        try expectEqual(8, tensor.scalar(.{ 0, 2, 2 }));

        try expectEqual(9, tensor.scalar(.{ 1, 0, 0 }));
        try expectEqual(10, tensor.scalar(.{ 1, 0, 1 }));
        try expectEqual(11, tensor.scalar(.{ 1, 0, 2 }));
        try expectEqual(12, tensor.scalar(.{ 1, 1, 0 }));
        try expectEqual(13, tensor.scalar(.{ 1, 1, 1 }));
        try expectEqual(14, tensor.scalar(.{ 1, 1, 2 }));
        try expectEqual(15, tensor.scalar(.{ 1, 2, 0 }));
        try expectEqual(16, tensor.scalar(.{ 1, 2, 1 }));
        try expectEqual(17, tensor.scalar(.{ 1, 2, 2 }));

        try expectEqual(18, tensor.scalar(.{ 2, 0, 0 }));
        try expectEqual(19, tensor.scalar(.{ 2, 0, 1 }));
        try expectEqual(20, tensor.scalar(.{ 2, 0, 2 }));
        try expectEqual(21, tensor.scalar(.{ 2, 1, 0 }));
        try expectEqual(22, tensor.scalar(.{ 2, 1, 1 }));
        try expectEqual(23, tensor.scalar(.{ 2, 1, 2 }));
        try expectEqual(24, tensor.scalar(.{ 2, 2, 0 }));
        try expectEqual(25, tensor.scalar(.{ 2, 2, 1 }));
        try expectEqual(26, tensor.scalar(.{ 2, 2, 2 }));
    }
};
