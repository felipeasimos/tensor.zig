const std = @import("std");

pub fn createSequence(comptime dtype: type, comptime n: usize) [n]dtype {
    var seq: [n]dtype = .{1} ** n;
    inline for (0..n) |i| {
        seq[i] = i;
    }
    return seq;
}

pub fn GetTypeLength(comptime T: type) usize {
    const type_info = @typeInfo(T);
    const type_info_data = @field(type_info, @tagName(std.meta.activeTag(type_info)));
    return if (comptime @hasField(@TypeOf(type_info_data), "len")) type_info_data.len else std.meta.fields(T).len;
}

pub fn asArray(comptime T: type, tuple: anytype) [GetTypeLength(@TypeOf(tuple))]T {
    if (@typeInfo(T) == .array) return T;
    const field_count = comptime GetTypeLength(@TypeOf(tuple));

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
