const std = @import("std");

pub fn createSequence(comptime dtype: type, comptime n: usize) [n]dtype {
    var seq: [n]dtype = .{1} ** n;
    inline for (0..n) |i| {
        seq[i] = i;
    }
    return seq;
}

pub fn getTypeLength(comptime T: type) usize {
    const type_info = @typeInfo(T);
    const type_info_data = @field(type_info, @tagName(std.meta.activeTag(type_info)));
    return if (comptime @hasField(@TypeOf(type_info_data), "len")) type_info_data.len else std.meta.fields(T).len;
}

pub fn asArray(comptime T: type, tuple: anytype) [getTypeLength(@TypeOf(tuple))]T {
    if (@typeInfo(T) == .array) return T;
    const field_count = comptime getTypeLength(@TypeOf(tuple));

    var array: [field_count]T = undefined;
    inline for (0..field_count) |i| {
        array[i] = tuple[i];
    }
    return array;
}

pub fn asSubArray(comptime T: type, arr: anytype, start_idx: usize, end_idx: usize) [end_idx - start_idx + 1]T {
    const size = end_idx - start_idx + 1;
    var result: [size]T = undefined;
    for (0..size) |i| {
        result[i] = arr[start_idx + i];
    }
    return result;
}

pub fn asSubVector(comptime T: type, arr: anytype, start_idx: usize, end_idx: usize) @Vector(end_idx - start_idx + 1, T) {
    const size = end_idx - start_idx + 1;
    const seq_vec: @Vector(size, T) = createSequence(T, size);
    const mask = seq_vec + @as(@Vector(size, T), @splat(start_idx));
    return @shuffle(
        usize,
        arr,
        undefined,
        mask,
    );
}

pub fn getIndexAt(idxs: anytype, comptime strides: anytype) usize {
    const strides_to = comptime asSubVector(usize, strides, 0, idxs.len - 1);
    const idxs_vec: @Vector(idxs.len, usize) = @bitCast(asArray(usize, idxs));
    return @reduce(.Add, strides_to * idxs_vec);
}

pub fn calculateStrides(comptime shape: anytype) @Vector(shape.len, usize) {
    var strides: [shape.len]usize = .{1} ** shape.len;
    for (1..shape.len) |i| {
        const idx = shape.len - i - 1;
        strides[idx] = shape[idx + 1] * strides[idx + 1];
    }
    return strides;
}

pub fn getComptimeFieldValue(comptime T: type, comptime field_name: []const u8) ?@FieldType(T, field_name) {
    const type_info = @typeInfo(T);
    if (@TypeOf(type_info) == void) return null;
    inline for (type_info.@"struct".fields) |field| {
        if (std.mem.eql(u8, field.name, field_name)) {
            if (field.default_value_ptr) |default_ptr| {
                return @as(*const field.type, @ptrCast(@alignCast(default_ptr))).*;
            }
        }
    }
    return null;
}

pub fn getChildType(comptime T: type) type {
    const type_info = @typeInfo(T);
    const active_tag = std.meta.activeTag(type_info);
    const info = @field(type_info, @tagName(active_tag));
    if (@TypeOf(info) == void) {
        return T;
    }
    if (@hasField(@TypeOf(info), "child")) {
        return getChildType(info.child);
    }
    return T;
}

pub fn stridesAreContiguous(comptime shape_arr: anytype, comptime strides_arr: anytype) bool {
    const contiguous_strides: [shape_arr.len]usize = calculateStrides(shape_arr);
    return std.mem.eql(usize, &strides_arr, &contiguous_strides);
}

pub fn isTuple(comptime T: type) bool {
    const type_info = @typeInfo(T);
    if (type_info != .@"struct") return false;
    const struct_info = type_info.@"struct";
    // Empty struct is considered a tuple (empty tuple)
    if (struct_info.fields.len == 0) return true;

    // Check if all field names are numbers starting from "0"
    for (struct_info.fields, 0..) |field, i| {
        const expected_name = std.fmt.comptimePrint("{}", .{i});
        if (!std.mem.eql(u8, field.name, expected_name)) {
            return false;
        }
    }

    return true;
}
