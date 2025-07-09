const std = @import("std");
const utils = @import("./utils.zig");
const tensor = @import("tensor.zig");

pub inline fn matmul(a: anytype, b: anytype, result: anytype) void {
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

pub fn MatMulNewResult(dtype: type, a_shape: anytype, b_shape: anytype) type {
    const b_length = utils.GetTypeLength(@TypeOf(b_shape));
    if (b_length != 2 or a_shape.len != 2) {
        @compileError("Incompatible shape with matmul");
    }

    // (P, Q1) x (Q2, R) -> (P, R)
    const P = a_shape[0];
    const Q1 = a_shape[1];
    const Q2 = b_shape[0];
    const R = b_shape[1];
    if (Q1 != Q2) {
        @compileError(std.fmt.comptimePrint("Number of columns don't match with number of rows: {any} x {any}", .{ a_shape, b_shape }));
    }

    const new_shape = comptime .{ P, R };
    const new_strides = utils.calculateStrides(new_shape);
    return tensor.InnerTensor(dtype, new_shape, new_strides, false, false);
}

pub inline fn matmulNew(a: anytype, b: anytype) MatMulNewResult(b.shape) {
    var result = MatMulNewResult(a.dtype, a.shape, b.shape){ .data = undefined };
    matmul(a, b, &result);
    return result;
}
