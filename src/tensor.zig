const std = @import("std");

pub fn Tensor(comptime dtype: type, comptime _shape: anytype) type {
    const dtype_info = @typeInfo(dtype);
    const shape_vec = asVector(dtype, _shape);
    const total_num_scalars = @reduce(.Mul, shape_vec);
    if (dtype_info != .float and dtype_info != .int) {
        @compileError("Only floats and integers are valid tensor dtypes");
    }
    const _strides = calculateStrides(dtype, _shape);
    return struct {
        comptime shape: @Vector(_shape.len, usize) = shape_vec,
        comptime strides: @Vector(_shape.len, usize) = _strides,

        data: []dtype,

        pub fn init(data: []dtype) @This() {
            std.debug.assert(total_num_scalars == data.len);
            return .{
                .data = data,
            };
        }
        pub fn randomize(self: *@This(), random: std.Random) void {
            random.bytes(std.mem.asBytes(&self.data));
        }

        pub fn scalar_mut(self: *@This(), idxs: @Vector(_shape.len, usize)) *dtype {
            const idx = @reduce(.Add, self.strides * idxs);
            return &self.data[idx];
        }

        pub fn scalar(self: *@This(), idxs: @Vector(_shape.len, usize)) dtype {
            return self.scalar_mut(idxs).*;
        }
    };
}

fn asVector(comptime dtype: type, seq: anytype) @Vector(seq.len, dtype) {
    var vec: @Vector(seq.len, dtype) = undefined;
    for (seq, 0..) |a, i| {
        vec[i] = a;
    }
    return vec;
}

fn calculateStrides(comptime dtype: type, comptime shape: anytype) @Vector(shape.len, dtype) {
    var strides: [shape.len]usize = .{1} ** shape.len;
    for (1..shape.len) |i| {
        const idx = shape.len - i - 1;
        strides[idx] = shape[idx + 1] * strides[idx + 1];
    }
    return strides;
}
