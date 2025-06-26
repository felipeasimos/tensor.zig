const std = @import("std");
const TensorView = @import("tensor_view.zig").TensorView;

/// OwnedTensor owns the underlying tensor data and can make changes to it
/// read-only tensor view can be accessed with the `view()` method
pub fn OwnedTensor(comptime dtype: type, comptime _shape: anytype) type {
    @setEvalBranchQuota(100000);
    return InnerTensor(dtype, _shape, calculateStrides(_shape), false);
}

fn InnerTensor(comptime dtype: type, comptime _shape: anytype, comptime _strides: anytype, comptime is_ref: bool) type {
    const dtype_info = @typeInfo(dtype);

    const shape_arr = asArray(usize, _shape);
    const shape_vec = asVector(usize, _shape);

    const strides_arr = asArray(usize, _strides);
    const strides_vec = asVector(usize, _strides);

    const total_num_scalars = @reduce(.Mul, shape_vec);
    const DataSequenceType = if (is_ref) []dtype else [total_num_scalars]dtype;

    if (dtype_info != .float and dtype_info != .int) {
        @compileError("Only floats and integers are valid tensor dtypes");
    }
    return struct {
        comptime shape: @Vector(shape_arr.len, usize) = shape_vec,
        comptime strides: @Vector(strides_arr.len, usize) = strides_vec,
        comptime num_scalars: usize = total_num_scalars,
        comptime is_reference: bool = is_ref,

        data: DataSequenceType,

        pub fn init(data: []dtype) @This() {
            if (comptime is_ref) {
                return .{ .data = data };
            }
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

        fn RefResult(comptime size: usize) type {
            if (comptime _shape.len - size == 0) {
                return *dtype;
            }
            const new_shape = comptime asSubArray(usize, shape_arr, size, shape_arr.len - 1);
            const new_strides = comptime calculateStrides(new_shape);
            return InnerTensor(
                dtype,
                new_shape,
                new_strides,
                true,
            );
        }

        /// get a mutable view
        pub inline fn ref(self: *@This(), idxs: anytype) RefResult(idxs.len) {
            if (comptime idxs.len == 0) {
                return RefResult(0).init(self.data[0..]);
            }
            if (comptime _shape.len - idxs.len == 0) {
                return self.scalar(idxs);
            }
            const strides_to_sub_tensor = comptime asSubVector(usize, self.strides, 0, idxs.len - 1);
            const start_idx = getIndexAt(idxs, self.strides);
            const final_idx = start_idx + strides_to_sub_tensor[idxs.len - 1];
            return RefResult(idxs.len).init(self.data[start_idx..final_idx]);
        }

        fn CloneResult(comptime size: usize) type {
            if (comptime shape_arr.len - size == 0) {
                return dtype;
            }
            return InnerTensor(
                dtype,
                comptime asSubArray(usize, shape_arr, size, shape_arr.len - 1),
                false,
            );
        }

        /// get a subtensor. `idxs` needs to be an array.
        pub inline fn clone(self: *const @This(), idxs: anytype) CloneResult(idxs.len) {
            if (comptime idxs.len == 0) {
                return RefResult(0).init(&self.data);
            }
            if (comptime shape_arr.len - idxs.len == 0) {
                return self.data[getIndexAt(idxs, self.strides)];
            }
            const strides_to_sub_tensor = comptime asSubVector(usize, self.strides, 0, idxs.len - 1);
            const start_idx = getIndexAt(idxs, self.strides);
            const final_idx = start_idx + strides_to_sub_tensor[idxs.len - 1];
            return RefResult(idxs.len).init(self.data[start_idx..final_idx]);
        }

        fn stridesAreContiguous() bool {
            const contiguous_strides: [strides_arr.len]usize = calculateStrides(shape_arr);
            return std.mem.eql(usize, &strides_arr, &contiguous_strides);
        }

        fn ReshapeResult(comptime shape: anytype) type {
            return InnerTensor(dtype, shape, calculateStrides(shape), is_ref);
        }

        pub inline fn reshape(self: *const @This(), comptime shape: anytype) ReshapeResult(shape) {
            if (comptime !stridesAreContiguous()) {
                @compileError("Can't reshape a tensor without contiguous strides");
            }
            const result = ReshapeResult(shape).init(self.data);
            if (comptime result.num_scalars != self.num_scalars) {
                @compileError("Invalid reshape size (the final number of scalars don't match the current tensor)");
            }
            return result;
        }

        fn otherValue(other: anytype, comptime i: usize) dtype {
            const T = @TypeOf(other);
            switch (@typeInfo(T)) {
                .comptime_int, .comptime_float, .int, .float => return other,
                .pointer, .@"struct" => return other.data[i],
                inline else => {},
            }
            @compileError(std.fmt.comptimePrint("Invalid operand type {} for {}", .{ T, @This() }));
        }

        pub inline fn wise(self: *@This(), other: anytype, func: fn (dtype, dtype) dtype) *@This() {
            inline for (0..self.data.len) |i| {
                const other_value = otherValue(other, i);
                self.data[i] = func(self.data[i], other_value);
            }
            return self;
        }

        pub inline fn matmul(self: *const @This(), other: anytype, result: anytype) void {
            // (P, Q) x (Q, R) -> (P, R)
            const P = comptime shape_arr[0];
            const Q = comptime shape_arr[1];
            const R = comptime other.shape[1];
            if (comptime (result.shape[0] != P or result.shape[1] != R or other.shape[0] != Q)) {
                @compileError("Number of columns don't match with number of rows");
            }
            for (0..P) |i| {
                for (0..R) |j| {
                    var tmp: dtype = 0;
                    for (0..Q) |k| {
                        const index_self = getIndexAt(.{ i, k }, self.strides);
                        const index_other = getIndexAt(.{ k, j }, other.strides);
                        tmp += self.data[index_self] * other.data[index_other];
                    }
                    const index_result = getIndexAt(.{ i, j }, result.strides);
                    result.data[index_result] = tmp;
                }
            }
        }

        fn MatMulNewResult(other_shape: @Vector(shape_arr.len, usize)) type {
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
                @compileError("Number of columns don't match with number of rows");
            }

            const new_shape = comptime .{ P, R };
            const new_strides = calculateStrides(new_shape);
            return InnerTensor(dtype, new_shape, new_strides, false);
        }

        pub inline fn matmulNew(self: *const @This(), other: anytype) MatMulNewResult(other.shape) {
            var result = MatMulNewResult(other.shape){ .data = undefined };
            self.matmul(other, &result);
            return result;
        }

        fn TransposeResult(comptime shuffled_axises: anytype) type {
            if (comptime shuffled_axises.len == 0) {
                var mask = createSequence(usize, strides_arr.len);
                const tmp = mask[strides_arr.len - 1];
                mask[strides_arr.len - 1] = mask[strides_arr.len - 2];
                mask[strides_arr.len - 2] = tmp;
                return TransposeResult(mask);
            }
            const new_strides = @shuffle(
                usize,
                strides_vec,
                undefined,
                shuffled_axises,
            );
            const new_shape = @shuffle(
                usize,
                shape_vec,
                undefined,
                shuffled_axises,
            );
            return InnerTensor(
                dtype,
                new_shape,
                new_strides,
                true,
            );
        }

        pub inline fn transpose(self: *@This(), comptime shuffled_axises: anytype) TransposeResult(shuffled_axises) {
            return TransposeResult(.{}).init(&self.data);
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

fn getIndexAt(idxs: anytype, comptime strides: anytype) usize {
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
