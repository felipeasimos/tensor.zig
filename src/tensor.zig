const std = @import("std");

/// OwnedTensor owns the underlying tensor data and can make changes to it
/// read-only tensor view can be accessed with the `view()` method
pub fn Tensor(comptime dtype: type, comptime _shape: anytype) type {
    @setEvalBranchQuota(10000);
    return InnerTensor(dtype, _shape, calculateStrides(_shape), false, false);
}

/// TensorView is a compile-time type that represents a view into a tensor.
/// Never writes to the underlying data
pub fn TensorView(comptime dtype: type, comptime _shape: anytype) type {
    @setEvalBranchQuota(10000);
    return InnerTensor(dtype, _shape, calculateStrides(_shape), true, true);
}

fn InnerTensor(comptime dtype: type, comptime _shape: anytype, comptime _strides: anytype, comptime is_ref: bool, comptime readonly: bool) type {
    const dtype_info = @typeInfo(dtype);

    if (dtype_info != .float and dtype_info != .int) {
        @compileError("Only floats and integers are valid tensor dtypes");
    }

    const shape_arr = asArray(usize, _shape);
    const shape_vec: @Vector(shape_arr.len, usize) = shape_arr;

    const strides_arr = asArray(usize, _strides);
    const strides_vec: @Vector(strides_arr.len, usize) = strides_arr;

    const _is_view = (is_ref and readonly);

    const total_num_scalars = @reduce(.Mul, shape_vec);
    const highest_idx = @reduce(
        .Add,
        (shape_vec - @as(@Vector(shape_arr.len, usize), @splat(1))) * strides_vec,
    );
    const DataSequenceType = if (is_ref) []dtype else [total_num_scalars]dtype;

    const ScalarResult = if (readonly) dtype else *dtype;

    return struct {
        comptime shape: @Vector(shape_arr.len, usize) = shape_vec,
        comptime strides: @Vector(strides_arr.len, usize) = strides_vec,
        comptime num_scalars: usize = total_num_scalars,
        comptime is_reference: bool = is_ref,
        comptime is_view: bool = _is_view,

        data: DataSequenceType,

        pub fn init(data: []dtype) @This() {
            if (comptime is_ref) {
                return .{ .data = data[0 .. highest_idx + 1] };
            }
            var new: @This() = .{ .data = undefined };
            std.mem.copyForwards(dtype, &new.data, data[0..]);
            return new;
        }

        pub inline fn scalar(self: *const @This(), idxs: @Vector(_shape.len, usize)) ScalarResult {
            const idx = @reduce(.Add, self.strides * idxs);
            if (comptime readonly) {
                return self.data[idx];
            }
            return &self.data[idx];
        }

        fn ViewResult(comptime size: usize) type {
            if (comptime _shape.len - size == 0) {
                return ScalarResult;
            }
            const new_shape = comptime asSubArray(usize, shape_arr, size, shape_arr.len - 1);
            const new_strides = comptime calculateStrides(new_shape);
            return InnerTensor(
                dtype,
                new_shape,
                new_strides,
                true,
                true,
            );
        }

        pub inline fn view(self: *const @This(), idxs: anytype) ViewResult(idxs.len) {
            if (comptime is_ref) {
                return ViewResult(idxs.len).init(self.data);
            }
            return ViewResult(idxs.len).init(&self.data);
        }

        fn MutResult(comptime size: usize) type {
            if (comptime _shape.len - size == 0) {
                return ScalarResult;
            }
            const new_shape = comptime asSubArray(usize, shape_arr, size, shape_arr.len - 1);
            const new_strides = comptime calculateStrides(new_shape);
            return InnerTensor(
                dtype,
                new_shape,
                new_strides,
                true,
                false,
            );
        }

        /// get a mutable view
        pub inline fn mut(self: *@This(), idxs: anytype) MutResult(idxs.len) {
            if (comptime idxs.len == 0) {
                return MutResult(0).init(self.data[0..]);
            }
            if (comptime _shape.len - idxs.len == 0) {
                return self.scalar(idxs);
            }
            const strides_to_sub_tensor = comptime asSubVector(usize, self.strides, 0, idxs.len - 1);
            const start_idx = getIndexAt(idxs, self.strides);
            const final_idx = start_idx + strides_to_sub_tensor[idxs.len - 1];
            return MutResult(idxs.len).init(self.data[start_idx..final_idx]);
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
                return CloneResult(0).init(&self.data);
            }
            if (comptime shape_arr.len - idxs.len == 0) {
                return self.data[getIndexAt(idxs, self.strides)];
            }
            const strides_to_sub_tensor = comptime asSubVector(usize, self.strides, 0, idxs.len - 1);
            const start_idx = getIndexAt(idxs, self.strides);
            const final_idx = start_idx + strides_to_sub_tensor[idxs.len - 1];
            return CloneResult(idxs.len).init(self.data[start_idx..final_idx]);
        }

        fn stridesAreContiguous() bool {
            const contiguous_strides: [strides_arr.len]usize = calculateStrides(shape_arr);
            return std.mem.eql(usize, &strides_arr, &contiguous_strides);
        }

        fn ReshapeResult(comptime shape: anytype) type {
            return InnerTensor(dtype, shape, calculateStrides(shape), is_ref, readonly);
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

        pub inline fn wise(self: *const @This(), other: anytype, result: anytype, func: fn (dtype, dtype) dtype) void {
            inline for (0..result.data.len) |i| {
                const other_value = otherValue(other, i);
                result.data[i] = func(self.data[i], other_value);
            }
        }

        fn WiseNewResult() type {
            return InnerTensor(dtype, shape_arr, strides_arr, false, false);
        }

        pub inline fn wiseNew(self: *const @This(), other: anytype, func: fn (dtype, dtype) dtype) WiseNewResult() {
            var result = InnerTensor(dtype, shape_arr, strides_arr, is_ref, readonly){ .data = undefined };
            _ = self.wise(other, &result, func);
            return result;
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
            return InnerTensor(dtype, new_shape, new_strides, false, false);
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
                readonly,
            );
        }

        pub inline fn transpose(self: *const @This(), comptime shuffled_axises: anytype) TransposeResult(shuffled_axises) {
            if (comptime is_ref) {
                return TransposeResult(.{}).init(self.data);
            }
            return TransposeResult(.{}).init(@constCast(self.data[0..]));
        }

        fn SliceResult(comptime ranges: anytype) type {
            var new_shape: [ranges.len]usize = undefined;
            for (0..ranges.len) |i| {
                new_shape[i] = ranges[i][1] - ranges[i][0];
            }
            var new_strides: [ranges.len]usize = undefined;
            for (0..ranges.len) |i| {
                new_strides[i] = strides_arr[i];
            }
            return InnerTensor(dtype, new_shape, new_strides, is_ref, readonly);
        }

        fn validateRanges(comptime ranges: anytype) bool {
            for (ranges, 0..) |range, i| {
                if (range[1] <= range[0] or range[1] > shape_arr[i]) {
                    return false;
                }
            }
            return true;
        }

        pub inline fn slice(self: *const @This(), comptime ranges: anytype) SliceResult(ranges) {
            if (comptime !validateRanges(ranges)) {
                @compileError("Invalid slicing ranges");
            }
            if (comptime !stridesAreContiguous()) {
                @compileError("Can't slice a tensor without contiguous strides");
            }
            const start_idx, const final_idx = comptime idxs: {
                var low_ranges_arr: [ranges.len]usize = undefined;
                var high_ranges_arr: [ranges.len]usize = undefined;
                for (0..ranges.len) |i| {
                    low_ranges_arr[i] = ranges[i][0];
                    high_ranges_arr[i] = ranges[i][1] - 1;
                }
                break :idxs .{
                    getIndexAt(low_ranges_arr, strides_arr),
                    getIndexAt(high_ranges_arr, strides_arr),
                };
            };
            return SliceResult(ranges).init(self.data[start_idx .. final_idx + 1]);
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
    const idxs_vec: @Vector(idxs.len, usize) = @bitCast(asArray(usize, idxs));
    return @reduce(.Add, strides_to * idxs_vec);
}

fn calculateStrides(comptime shape: anytype) @Vector(shape.len, usize) {
    var strides: [shape.len]usize = .{1} ** shape.len;
    for (1..shape.len) |i| {
        const idx = shape.len - i - 1;
        strides[idx] = shape[idx + 1] * strides[idx + 1];
    }
    return strides;
}
