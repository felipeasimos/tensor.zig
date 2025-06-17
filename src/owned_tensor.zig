const std = @import("std");
const TensorView = @import("tensor_view.zig").TensorView;

/// OwnedTensor owns the underlying tensor data and can make changes to it
/// read-only tensor view can be accessed with the `view()` method
pub fn OwnedTensor(comptime dtype: type, comptime _shape: anytype) type {
    @setEvalBranchQuota(100000);
    return InnerTensor(dtype, _shape, .{ .cpu = .in_place });
}

pub const OpConfig = union(enum) {
    cpu: enum {
        in_place,
        write_to,
        new,
    },
    gpu: enum {
        in_place,
    },
};

fn InnerTensor(comptime dtype: type, comptime _shape: anytype, comptime _op_config: OpConfig) type {
    const dtype_info = @typeInfo(dtype);

    const shape_arr = asArray(usize, _shape);
    const shape_vec = asVector(usize, _shape);

    const strides_vec = calculateStrides(_shape);
    const strides_arr = strides_vec;

    const total_num_scalars = @reduce(.Mul, shape_vec);

    if (dtype_info != .float and dtype_info != .int) {
        @compileError("Only floats and integers are valid tensor dtypes");
    }
    return struct {
        comptime shape: @Vector(_shape.len, usize) = _shape,
        comptime strides: @Vector(_shape.len, usize) = strides_vec,
        comptime num_scalars: usize = total_num_scalars,
        comptime op_config: OpConfig = _op_config,

        data: [total_num_scalars]dtype,

        pub fn init(data: []dtype) @This() {
            var new: @This() = .{ .data = undefined };
            std.mem.copyForwards(dtype, &new.data, data);
            return new;
        }

        pub inline fn view(self: *@This()) TensorView(dtype, shape_arr) {
            return TensorView(dtype, shape_arr).init(&self.data);
        }

        pub inline fn scalar(self: *@This(), idxs: @Vector(_shape.len, usize)) *dtype {
            const idx = @reduce(.Add, self.strides * idxs);
            return &self.data[idx];
        }

        fn GetResult(comptime size: usize) type {
            if (comptime _shape.len - size == 0) {
                return *dtype;
            }
            return InnerTensor(
                dtype,
                comptime asSubArray(usize, shape_arr, size, shape_arr.len - 1),
                comptime asSubArray(usize, strides_arr, size, strides_arr.len - 1),
            );
        }

        /// get a subtensor. `idxs` needs to be an array.
        pub inline fn get(self: *@This(), idxs: anytype) GetResult(idxs.len) {
            if (comptime idxs.len == 0) {
                @compileError("index sequence must have a positive non-zero length");
            }
            if (comptime _shape.len - idxs.len == 0) {
                return self.scalar(idxs);
            }
            const strides_to_sub_tensor = comptime asSubVector(usize, self.strides, 0, idxs.len - 1);
            const start_idx = getIndexAt(idxs, self.strides);
            const final_idx = start_idx + strides_to_sub_tensor[idxs.len - 1];
            return GetResult(idxs.len).init(self.data[start_idx..final_idx]);
        }

        fn stridesAreContiguous() bool {
            const contiguous_strides: [strides_arr.len]usize = calculateStrides(shape_arr);
            return std.mem.eql(usize, &strides_arr, &contiguous_strides);
        }

        pub inline fn reshape(self: *@This(), comptime shape: anytype) OwnedTensor(dtype, shape) {
            if (comptime !stridesAreContiguous()) {
                @compileError("Can't reshape a tensor without contiguous strides");
            }
            const result = OwnedTensor(dtype, shape).init(self.data);
            if (comptime result.num_scalars != self.num_scalars) {
                @compileError("Invalid reshape size (the final number of scalars don't match the current tensor)");
            }
            return result;
        }

        fn resultTensor(self: *@This(), other: anytype) *@This() {
            return switch (comptime self.op_config) {
                .cpu => |cpu_config| switch (comptime cpu_config) {
                    .in_place => self,
                    .write_to => other,
                    .new => @This(){ .data = undefined },
                },
                .gpu => |gpu_config| switch (comptime gpu_config) {
                    .in_place => self,
                },
            };
        }

        fn otherValue(other: anytype, comptime i: usize) dtype {
            const T = @TypeOf(other);
            switch (@typeInfo(T)) {
                .comptime_int, .comptime_float, .int, .float => return other,
                .pointer => |p| if (p.child == @This()) {
                    return other.data[i];
                },
                .@"struct" => return other.data[i],
                inline else => {},
            }
            @compileError(std.fmt.comptimePrint("Invalid operand type {} for {}", .{ T, @This() }));
        }

        pub inline fn wise(self: *@This(), other: anytype, func: fn (dtype, dtype) dtype) *@This() {
            const result_tensor = self.resultTensor(other);
            inline for (0..self.data.len) |i| {
                const other_value = otherValue(other, i);
                result_tensor.data[i] = func(self.data[i], other_value);
            }
            return result_tensor;
        }

        pub inline fn opConfig(self: *@This(), new_op_config: OpConfig) InnerTensor(dtype, shape_arr, strides_arr, new_op_config) {
            return InnerTensor(dtype, shape_arr, strides_arr, new_op_config).init(self.data);
        }
    };
}

fn createSequence(comptime dtype: type, comptime n: usize) [n]dtype {
    var seq: [n]dtype = .{1} ** n;
    inline for (0..n) |i| {
        seq[i] = i;
    }
    return seq;
}

fn GetTypeLength(comptime T: type) usize {
    const type_info = @typeInfo(T);
    const type_info_data = @field(type_info, @tagName(std.meta.activeTag(type_info)));
    return if (comptime @hasField(@TypeOf(type_info_data), "len")) type_info_data.len else std.meta.fields(T).len;
}

fn GetChildType(comptime T: type) type {
    const type_info = @typeInfo(T);
    const type_info_data = @field(type_info, @tagName(std.meta.activeTag(type_info)));
    return if (comptime @hasDecl(@TypeOf(type_info_data), "child")) type_info_data.child else @FieldType(T, "0");
}

fn asArray(comptime T: type, tuple: anytype) [GetTypeLength(@TypeOf(tuple))]T {
    if (@typeInfo(T) == .array) return T;
    const field_count = comptime GetTypeLength(@TypeOf(tuple));

    var array: [field_count]T = undefined;
    inline for (0..field_count) |i| {
        array[i] = tuple[i];
    }
    return array;
}

fn asSubArray(comptime T: type, arr: anytype, start_idx: usize, end_idx: usize) [end_idx - start_idx + 1]T {
    const size = end_idx - start_idx + 1;
    var result: [size]T = undefined;
    for (0..size) |i| {
        result[i] = arr[start_idx + i];
    }
    return result;
}

fn asSubVector(comptime T: type, arr: anytype, start_idx: usize, end_idx: usize) @Vector(end_idx - start_idx + 1, T) {
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

fn getIndexAt(comptime idxs: anytype, comptime strides: anytype) usize {
    const strides_to = comptime asSubVector(usize, strides, 0, idxs.len - 1);
    return @reduce(.Add, strides_to * asVector(usize, idxs));
}

fn asVector(comptime T: type, seq: anytype) @Vector(GetTypeLength(@TypeOf(seq)), T) {
    return @bitCast(asArray(T, seq));
}

fn calculateStrides(comptime shape: anytype) @Vector(shape.len, usize) {
    var strides: [shape.len]usize = .{1} ** shape.len;
    for (1..shape.len) |i| {
        const idx = shape.len - i - 1;
        strides[idx] = shape[idx + 1] * strides[idx + 1];
    }
    return strides;
}
