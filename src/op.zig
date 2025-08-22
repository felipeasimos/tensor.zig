const std = @import("std");
const utils = @import("./utils.zig");
const tensor = @import("tensor.zig");
const iterator = @import("iterator.zig");

pub inline fn isTensor(comptime T: type) bool {
    const info = @typeInfo(T);
    if (info != .@"struct" and info != .@"union") {
        return false;
    }
    if (utils.getComptimeFieldValue(T, "factory_function")) |val| {
        return val == tensor.InnerTensor;
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
    return tensor.InnerTensor(dtype, utils.asTuple(usize, new_shape), utils.asTuple(usize, new_strides), false);
}

pub inline fn matmul(a: anytype, b: anytype) MatMulNewResult(a.dtype, a.shape, b.shape) {
    var result = MatMulNewResult(a.dtype, a.shape, b.shape){ .data = undefined };
    result.matmul(a, b);
    return result;
}

fn WiseResult(comptime FnType: type, comptime tensorsType: type) type {
    const Dtype = @typeInfo(FnType).@"fn".return_type.?;
    const length = utils.getTypeLength(tensorsType);
    for (0..length) |i| {
        const index_as_str = std.fmt.comptimePrint("{}", .{i});
        const T = utils.getChildType(@FieldType(tensorsType, index_as_str));
        if (isTensor(T)) {
            const shape = utils.getComptimeFieldValue(T, "shape").?;
            const strides = utils.calculateStrides(shape);
            return tensor.InnerTensor(Dtype, utils.asTuple(usize, shape), utils.asTuple(usize, strides), false);
        }
    }
    @compileError("At least one of the arguments must be a tensor");
}

pub inline fn wise(tensors: anytype, f: anytype) WiseResult(@TypeOf(f), @TypeOf(tensors)) {
    var result: WiseResult(@TypeOf(f), @TypeOf(tensors)) = undefined;
    result.wise(tensors, f);
    return result;
}

fn ReduceResult(AccumulatorType: type, FnType: type) type {
    const ReturnType = @typeInfo(FnType).@"fn".return_type.?;
    if (AccumulatorType != ReturnType) {
        @compileError("Accumulator and return type don't match");
    }

    if (isTensor(ReturnType)) {
        return ReturnType;
    }
    return ReturnType;
}

fn ReduceTensorResult(AccumulatorType: type, FnType: type) type {
    const ReturnType = @typeInfo(FnType).@"fn".return_type.?;
    if (AccumulatorType != ReturnType) {
        @compileError("Accumulator and return type don't match");
    }

    if (isTensor(ReturnType)) {
        return ReturnType;
    }
    return tensor.InnerTensor(ReturnType, .{1}, .{1}, false);
}

pub inline fn reduce(initial: anytype, tensors: anytype, f: anytype) ReduceResult(@TypeOf(initial), @TypeOf(f)) {
    var result: ReduceTensorResult(@TypeOf(initial), @TypeOf(f)) = undefined;
    result.reduce(initial, tensors, f);
    if (comptime result.shape.len == 1 and result.shape[0] == 1) {
        return result.data[0];
    }
    return result;
}
