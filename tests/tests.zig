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
        try expectEqual(&data[0], tensor.get(.{0}));
        try expectEqual(&data[1], tensor.get(.{1}));
        try expectEqual(&data[2], tensor.get(.{2}));
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

        try expectEqual(data[0], tensor.get(.{ 0, 0 }).*);
        try expectEqual(data[1], tensor.get(.{ 0, 1 }).*);
        try expectEqual(data[2], tensor.get(.{ 0, 2 }).*);

        try expectEqual(data[3], tensor.get(.{ 1, 0 }).*);
        try expectEqual(data[4], tensor.get(.{ 1, 1 }).*);
        try expectEqual(data[5], tensor.get(.{ 1, 2 }).*);

        try expectEqual(data[6], tensor.get(.{ 2, 0 }).*);
        try expectEqual(data[7], tensor.get(.{ 2, 1 }).*);
        try expectEqual(data[8], tensor.get(.{ 2, 2 }).*);
    }
    test "get sub tensor content (vector)" {
        var data: [9]f64 = .{ 1, 2, 3, 4, 5, 6, 7, 8, 9 };
        var tensor = TensorView(f64, .{ 3, 3 }).init(data[0..]);

        try expectEqual(data[0..3], tensor.get(.{0}).data);
        try expectEqual(data[3..6], tensor.get(.{1}).data);
        try expectEqual(data[6..9], tensor.get(.{2}).data);
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

        try expectEqual(data[0], tensor.get(.{ 0, 0, 0 }).*);
        try expectEqual(data[1], tensor.get(.{ 0, 0, 1 }).*);
        try expectEqual(data[2], tensor.get(.{ 0, 0, 2 }).*);
        try expectEqual(data[3], tensor.get(.{ 0, 1, 0 }).*);
        try expectEqual(data[4], tensor.get(.{ 0, 1, 1 }).*);
        try expectEqual(data[5], tensor.get(.{ 0, 1, 2 }).*);
        try expectEqual(data[6], tensor.get(.{ 0, 2, 0 }).*);
        try expectEqual(data[7], tensor.get(.{ 0, 2, 1 }).*);
        try expectEqual(data[8], tensor.get(.{ 0, 2, 2 }).*);

        try expectEqual(data[9], tensor.get(.{ 1, 0, 0 }).*);
        try expectEqual(data[10], tensor.get(.{ 1, 0, 1 }).*);
        try expectEqual(data[11], tensor.get(.{ 1, 0, 2 }).*);
        try expectEqual(data[12], tensor.get(.{ 1, 1, 0 }).*);
        try expectEqual(data[13], tensor.get(.{ 1, 1, 1 }).*);
        try expectEqual(data[14], tensor.get(.{ 1, 1, 2 }).*);
        try expectEqual(data[15], tensor.get(.{ 1, 2, 0 }).*);
        try expectEqual(data[16], tensor.get(.{ 1, 2, 1 }).*);
        try expectEqual(data[17], tensor.get(.{ 1, 2, 2 }).*);

        try expectEqual(data[18], tensor.get(.{ 2, 0, 0 }).*);
        try expectEqual(data[19], tensor.get(.{ 2, 0, 1 }).*);
        try expectEqual(data[20], tensor.get(.{ 2, 0, 2 }).*);
        try expectEqual(data[21], tensor.get(.{ 2, 1, 0 }).*);
        try expectEqual(data[22], tensor.get(.{ 2, 1, 1 }).*);
        try expectEqual(data[23], tensor.get(.{ 2, 1, 2 }).*);
        try expectEqual(data[24], tensor.get(.{ 2, 2, 0 }).*);
        try expectEqual(data[25], tensor.get(.{ 2, 2, 1 }).*);
        try expectEqual(data[26], tensor.get(.{ 2, 2, 2 }).*);
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
};
