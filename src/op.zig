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
    return tensor.InnerTensor(dtype, new_shape, new_strides, false);
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
            return tensor.InnerTensor(Dtype, shape, strides, false);
        }
    }
    @compileError("At least one of the arguments must be a tensor");
}

pub inline fn wise(tensors: anytype, f: anytype) WiseResult(@TypeOf(f), @TypeOf(tensors)) {
    var result: WiseResult(@TypeOf(f), @TypeOf(tensors)) = undefined;
    result.wise(tensors, f);
    return result;
}

fn ReduceResult(FnType: type, TensorsType: type) type {
    const ReturnType = @typeInfo(FnType).@"fn".return_type.?;
    const tuple_length = utils.getTypeLength(TensorsType);

    if (isTensor(ReturnType)) {
        const dtype = comptime utils.getComptimeFieldValue(ReturnType, "dtype").?;
        for (0..tuple_length) |i| {
            const index_as_str = std.fmt.comptimePrint("{}", .{i});
            const T = utils.getChildType(@FieldType(TensorsType, index_as_str));
            if (isTensor(T)) {
                const tensor_shape = comptime utils.getComptimeFieldValue(T, "shape").?;
                const return_shape = comptime utils.getComptimeFieldValue(ReturnType, "shape").?;
                const result_shape = comptime (.{tensor_shape[0]} ++ return_shape);
                const strides = utils.calculateStrides(result_shape);
                return tensor.InnerTensor(dtype, result_shape, strides, false);
            }
        }
    }
    for (0..tuple_length) |i| {
        const index_as_str = std.fmt.comptimePrint("{}", .{i});
        const T = utils.getChildType(@FieldType(TensorsType, index_as_str));
        if (isTensor(T)) {
            const tensor_shape = utils.getComptimeFieldValue(T, "shape").?;
            const result_shape = .{tensor_shape[0]};
            const strides = utils.calculateStrides(result_shape);
            return tensor.InnerTensor(ReturnType, result_shape, strides, false);
        }
    }
    @compileError("At least one of the arguments must be a tensor");
}

pub inline fn reduce(initial: anytype, tensors: anytype, f: anytype) ReduceResult(@TypeOf(f), @TypeOf(tensors)) {
    var result: ReduceResult(@TypeOf(f), @TypeOf(tensors)) = undefined;
    result.reduce(initial, tensors, f);
    return result;
}
