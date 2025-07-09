const std = @import("std");
const utils = @import("./utils.zig");

pub inline fn matmul(a: anytype, b: anytype, result: anytype) void {
    if (comptime a.dtype != b.dtype) {
        @compileError("types don't match");
    }
    // (P, Q) x (Q, R) -> (P, R)
    const P = comptime a.shape[0];
    const Q = comptime a.shape[1];
    const R = comptime b.shape[1];
    if (comptime (result.shape[0] != P or result.shape[1] != R or b.shape[0] != Q)) {
        @compileError(std.fmt.comptimePrint("Number of columns don't match with number of rows: {any} x {any} -> {any}", .{ a.shape, b.shape, result.shape }));
    }
    for (0..P) |i| {
        for (0..R) |j| {
            var tmp: a.dtype = 0;
            for (0..Q) |k| {
                const index_self = utils.getIndexAt(.{ i, k }, a.strides);
                const index_other = utils.getIndexAt(.{ k, j }, b.strides);
                tmp += a.data[index_self] * b.data[index_other];
            }
            const index_result = utils.getIndexAt(.{ i, j }, result.strides);
            result.data[index_result] = tmp;
        }
    }
}

fn MatMulNewResult(other_shape: [shape_arr.len]usize) type {
    const other_length = GetTypeLength(@TypeOf(other_shape));
    if (other_length != 2 or shape_arr.len != 2) {
        @compileError("Incompatible shape with matmul");
    }

    // (P, Q1) x (Q2, R) -> (P, R)
    const P = shape_arr[0];
    const Q1 = shape_arr[1];
    const Q2 = other_shape[0];
    const R = other_shape[1];
    if (Q1 != Q2) {
        @compileError(std.fmt.comptimePrint("Number of columns don't match with number of rows: {any} x {any}", .{ shape_arr, other_shape }));
    }

    const new_shape = comptime .{ P, R };
    const new_strides = calculateStrides(new_shape);
    return InnerTensor(dtype, new_shape, new_strides, false, false);
}

pub inline fn matmulNew(self: anytype, other: anytype) MatMulNewResult(other.shape) {
    var result = MatMulNewResult(other.shape){ .data = undefined };
    self.matmul(other, &result);
    return result;
}
