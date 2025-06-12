const std = @import("std");
const Range = @import("range.zig").Range;

pub fn TensorView(comptime dtype: type, comptime _shape: anytype) type {
    return InnerTensorView(dtype, _shape, calculateStrides(_shape));
}

fn InnerTensorView(comptime dtype: type, comptime _shape: anytype, comptime _strides: anytype) type {
    const dtype_info = @typeInfo(dtype);

    const shape_arr = asArray(usize, _shape);
    const shape_vec = asVector(usize, _shape);

    const strides_arr = asArray(usize, _strides);
    // const strides_vec = asVector(dtype, _strides);

    const total_num_scalars = @reduce(.Mul, shape_vec);
    if (dtype_info != .float and dtype_info != .int) {
        @compileError("Only floats and integers are valid tensor dtypes");
    }
    return struct {
        comptime shape: @Vector(_shape.len, usize) = shape_vec,
        comptime strides: @Vector(_shape.len, usize) = _strides,
        comptime n_scalars: usize = total_num_scalars,

        data: []dtype,

        pub fn init(data: []dtype) @This() {
            return .{
                .data = data[0..total_num_scalars],
            };
        }
        pub fn randomize(self: *@This(), random: std.Random) void {
            random.bytes(std.mem.asBytes(&self.data));
        }

        pub inline fn scalar_mut(self: *@This(), idxs: @Vector(_shape.len, usize)) *dtype {
            const idx = @reduce(.Add, self.strides * idxs);
            return &self.data[idx];
        }

        pub inline fn scalar(self: *@This(), idxs: @Vector(_shape.len, usize)) dtype {
            return self.scalar_mut(idxs).*;
        }

        fn GetSubTensorViewResult(comptime size: usize) type {
            if (comptime _shape.len - size == 0) {
                return *dtype;
            }
            return InnerTensorView(
                dtype,
                comptime asSubArray(usize, shape_arr, size, shape_arr.len - 1),
                comptime asSubArray(usize, strides_arr, size, strides_arr.len - 1),
            );
        }

        /// get a subtensor. `idxs` needs to be an array.
        pub inline fn get(self: *@This(), idxs: anytype) GetSubTensorViewResult(idxs.len) {
            if (comptime idxs.len == 0) {
                @compileError("index sequence must have a positive non-zero length");
            }
            if (comptime _shape.len - idxs.len == 0) {
                return self.scalar_mut(idxs);
            }
            const to_sub_tensor_mask = comptime createSequence(usize, idxs.len);
            const strides_to_sub_tensor = comptime @shuffle(
                usize,
                self.strides,
                undefined,
                to_sub_tensor_mask,
            );
            const start_idx = @reduce(.Add, strides_to_sub_tensor * asVector(usize, idxs));
            const final_idx = start_idx + strides_to_sub_tensor[idxs.len - 1];
            return GetSubTensorViewResult(idxs.len).init(self.data[start_idx..final_idx]);
        }

        fn stridesAreContiguous() bool {
            const contiguous_strides: [strides_arr.len]usize = calculateStrides(shape_arr);
            return std.mem.eql(usize, &strides_arr, &contiguous_strides);
        }

        pub inline fn reshape(self: *@This(), comptime shape: anytype) TensorView(dtype, shape) {
            if (comptime !stridesAreContiguous()) {
                @compileError("Can't reshape a tensor without contiguous strides");
            }
            const result = TensorView(dtype, shape).init(self.data);
            if (comptime result.n_scalars != self.n_scalars) {
                @compileError("Invalid reshape size (the final number of scalars don't match the current tensor)");
            }
            return result;
        }
    };
}

pub fn createSequence(comptime dtype: type, comptime n: usize) [n]dtype {
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
    const size = comptime end_idx - start_idx + 1;
    var result: [size]T = undefined;
    for (0..size) |i| {
        result[i] = arr[start_idx + i];
    }
    return result;
}

fn asVector(comptime T: type, seq: anytype) @Vector(GetTypeLength(@TypeOf(seq)), T) {
    return asArray(T, seq);
}

fn calculateStrides(comptime shape: anytype) @Vector(shape.len, usize) {
    var strides: [shape.len]usize = .{1} ** shape.len;
    for (1..shape.len) |i| {
        const idx = shape.len - i - 1;
        strides[idx] = shape[idx + 1] * strides[idx + 1];
    }
    return strides;
}
