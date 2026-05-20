const std = @import("std");
const utils = @import("./utils.zig");
const tensor = @import("tensor.zig");
const iterator = @import("iterator.zig");

pub inline fn isTensor(comptime T: type) bool {
    const info = @typeInfo(T);
    if (info != .@"struct" and info != .@"union") {
        return false;
    }
    if (@hasDecl(T, "Marker")) {
        return T.Marker == tensor.Tensor;
    }
    return false;
}

pub fn MatMulNewResult(dtype: type, a_shape: anytype, b_shape: anytype) type {
    const b_length = utils.getTypeLength(@TypeOf(b_shape));
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
    return tensor.InnerTensor(dtype, utils.asTuple(usize, new_shape), utils.asTuple(usize, new_strides));
}

pub inline fn matmul(allocator: std.mem.Allocator, io: std.Io, a: anytype, b: anytype) !utils.getChildType(@TypeOf(a)) {
    const A = utils.getChildType(@TypeOf(a));
    const ResultType = tensor.Tensor(A.ScalarType, A.n_dims);
    var result: ResultType = try .alloc(allocator, ResultType.Metadata.rowMajor(.{
        a.metadata.shape[0],
        b.metadata.shape[1],
    }));
    try result.matmul(allocator, io, a, b);
    return result;
}

pub inline fn wise(allocator: std.mem.Allocator, tensors: anytype, f: anytype) !tensor.Tensor(@typeInfo(@TypeOf(f)).@"fn".return_type.?, utils.getNumberOfDimensions(@TypeOf(tensors))) {
    const Dtype = comptime @typeInfo(@TypeOf(f)).@"fn".return_type.?;
    const n_dims = comptime utils.getNumberOfDimensions(@TypeOf(tensors));
    const result_shape = utils.getTensorInTupleShape(tensors);
    var result: tensor.Tensor(Dtype, n_dims) = try .alloc(allocator, .rowMajor(result_shape));
    return result.wise(tensors, f);
}
