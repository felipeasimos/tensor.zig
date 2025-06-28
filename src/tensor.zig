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
        comptime shape: @TypeOf(shape_arr) = shape_arr,
        comptime strides: @TypeOf(strides_arr) = strides_arr,
        comptime num_scalars: usize = total_num_scalars,
        comptime is_reference: bool = is_ref,
        comptime is_view: bool = _is_view,

        data: DataSequenceType,

        pub fn init(data: []dtype) @This() {
            if (comptime is_ref) {
                return .{ .data = data[0 .. highest_idx + 1] };
            }
            var new: @This() = .{ .data = undefined };
            std.mem.copyForwards(dtype, &new.data, data[0 .. highest_idx + 1]);
            return new;
        }

        fn Self() type {
            if (comptime readonly) {
                return *const @This();
            }
            return *@This();
        }

        pub inline fn scalar(self: Self(), idxs: @Vector(shape_arr.len, usize)) ScalarResult {
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

        pub inline fn view(self: Self(), idxs: anytype) ViewResult(idxs.len) {
            if (comptime is_ref) {
                return ViewResult(idxs.len).init(self.data);
            }
            return ViewResult(idxs.len).init(@as([]dtype, self.data[0..]));
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
            inline for (0..self.num_scalars) |i| {
                const other_value = otherValue(other, i);
                result.data[i] = func(self.data[i], other_value);
            }
        }

        fn WiseNewResult() type {
            return InnerTensor(dtype, shape_arr, strides_arr, false, false);
        }

        pub inline fn wiseNew(self: *const @This(), other: anytype, func: fn (dtype, dtype) dtype) WiseNewResult() {
            var result = WiseNewResult(){ .data = undefined };
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

        pub inline fn convolution(self: *const @This(), kernel: anytype, result: anytype) void {
            // Perform convolution
            const kernel_strides = comptime asArray(usize, kernel.strides);
            const result_strides = comptime asArray(usize, result.strides);
            // Iterate over all output positions
            var output_pos: [shape_arr.len]usize = .{0} ** shape_arr.len;
            var pos: usize = 0;
            while (pos < result.num_scalars) {
                // Calculate output position from linear index
                var temp_pos = pos;
                for (0..shape_arr.len) |dim| {
                    const stride = result_strides[dim];
                    output_pos[dim] = temp_pos / stride;
                    temp_pos = temp_pos % stride;
                }
                // Perform convolution at this position
                var conv_sum: dtype = 0;
                // Iterate over kernel positions
                var kernel_pos: [kernel.shape.len]usize = .{0} ** kernel.shape.len;
                var kernel_idx: usize = 0;
                while (kernel_idx < kernel.num_scalars) {
                    // Calculate kernel position from linear index
                    var temp_kernel_idx = kernel_idx;
                    for (0..kernel.shape.len) |dim| {
                        const kernel_stride = kernel_strides[dim];
                        kernel_pos[dim] = temp_kernel_idx / kernel_stride;
                        temp_kernel_idx = temp_kernel_idx % kernel_stride;
                    }
                    // Calculate input position
                    var input_pos: [shape_arr.len]usize = undefined;
                    for (0..shape_arr.len) |dim| {
                        if (kernel.shape[dim] == 1) {
                            input_pos[dim] = output_pos[dim];
                        } else {
                            input_pos[dim] = output_pos[dim] + kernel_pos[dim];
                        }
                    }
                    // Get input and kernel values
                    const input_val = self.clone(input_pos);
                    const kernel_val = kernel.clone(kernel_pos);
                    // Accumulate convolution sum
                    conv_sum += input_val * kernel_val;
                    kernel_idx += 1;
                }
                // Store result
                const result_idx = getIndexAt(output_pos, result.strides);
                result.data[result_idx] = conv_sum;
                pos += 1;
            }
        }

        fn ConvolutionNewResult(kernel_shape: @Vector(shape_arr.len, usize)) type {
            const kernel_shape_arr = comptime asArray(usize, kernel_shape);

            // Calculate output shape for valid convolution
            var output_shape: [shape_arr.len]usize = undefined;
            for (0..shape_arr.len) |i| {
                if (shape_arr[i] < kernel_shape_arr[i]) {
                    @compileError("Input tensor dimension must be >= kernel dimension");
                }
                output_shape[i] = shape_arr[i] - kernel_shape_arr[i] + 1;
            }

            const output_strides = calculateStrides(output_shape);
            return InnerTensor(dtype, output_shape, output_strides, false, false);
        }

        /// Perform N-D convolution and return a new tensor with the result.
        pub inline fn convolutionNew(self: *const @This(), kernel: anytype) ConvolutionNewResult(kernel.shape) {
            var result = ConvolutionNewResult(kernel.shape){ .data = undefined };
            self.convolution(kernel, &result);
            return result;
        }

        /// Perform N-D pooling with a custom aggregation function.
        /// The kernel shape defines the pooling window size.
        /// The aggregation function takes: (accumulator, kernel_index, current_value) -> new_accumulator
        pub inline fn pooling(self: *const @This(), comptime kernel_shape: anytype, result: anytype, func: fn (dtype, usize, dtype) dtype) void {
            const kernel_shape_arr = comptime asArray(usize, kernel_shape);

            // Perform pooling
            const result_strides = comptime asArray(usize, result.strides);

            // Calculate total kernel size
            const total_kernel_size = comptime @reduce(.Mul, @as(@Vector(kernel_shape.len, usize), kernel_shape));
            // Iterate over all output positions
            var output_pos: [shape_arr.len]usize = .{0} ** shape_arr.len;
            var pos: usize = 0;

            while (pos < result.num_scalars) {
                // Calculate output position from linear index
                var temp_pos = pos;
                for (0..shape_arr.len) |dim| {
                    const stride = result_strides[dim];
                    output_pos[dim] = temp_pos / stride;
                    temp_pos = temp_pos % stride;
                }

                // Perform pooling at this position
                var accumulator: dtype = 0;
                var kernel_idx: usize = 0;

                // Iterate over kernel positions
                var kernel_pos: [kernel_shape_arr.len]usize = .{0} ** kernel_shape_arr.len;
                var kernel_linear_idx: usize = 0;

                while (kernel_linear_idx < total_kernel_size) {
                    // Calculate kernel position from linear index
                    var temp_kernel_idx = kernel_linear_idx;
                    for (0..kernel_shape_arr.len) |dim| {
                        const kernel_stride = kernel_stride: {
                            if (dim == kernel_shape_arr.len - 1) {
                                break :kernel_stride 1;
                            } else {
                                var stride: usize = 1;
                                for (dim + 1..kernel_shape_arr.len) |i| {
                                    stride *= kernel_shape_arr[i];
                                }
                                break :kernel_stride stride;
                            }
                        };
                        kernel_pos[dim] = temp_kernel_idx / kernel_stride;
                        temp_kernel_idx = temp_kernel_idx % kernel_stride;
                    }

                    // Calculate input position
                    var input_pos: [shape_arr.len]usize = undefined;
                    for (0..shape_arr.len) |dim| {
                        input_pos[dim] = output_pos[dim] + kernel_pos[dim];
                    }

                    // Get input value
                    const input_val = self.clone(input_pos);

                    // Apply aggregation function
                    accumulator = @call(.always_inline, func, .{ accumulator, kernel_idx, input_val });

                    kernel_linear_idx += 1;
                    kernel_idx += 1;
                }

                // Store result
                const result_idx = getIndexAt(output_pos, result.strides);
                result.data[result_idx] = accumulator;

                pos += 1;
            }
        }

        fn PoolingNewResult(comptime kernel_shape: anytype) type {
            const kernel_shape_arr = comptime asArray(usize, kernel_shape);

            // Calculate output shape for valid pooling
            var output_shape: [shape_arr.len]usize = undefined;
            inline for (0..shape_arr.len) |i| {
                if (shape_arr[i] < kernel_shape_arr[i]) {
                    @compileError("Input tensor dimension must be >= kernel dimension");
                }
                output_shape[i] = shape_arr[i] - kernel_shape_arr[i] + 1;
            }

            const output_strides = calculateStrides(output_shape);
            return InnerTensor(dtype, output_shape, output_strides, false, false);
        }

        /// Perform N-D pooling and return a new tensor with the result.
        pub inline fn poolingNew(self: *const @This(), comptime kernel_shape: anytype, func: fn (dtype, usize, dtype) dtype) PoolingNewResult(kernel_shape) {
            var result = PoolingNewResult(kernel_shape){ .data = undefined };
            self.pooling(kernel_shape, &result, func);
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

        pub inline fn transpose(self: Self(), comptime shuffled_axises: anytype) TransposeResult(shuffled_axises) {
            if (comptime is_ref) {
                return TransposeResult(.{}).init(self.data);
            }
            return TransposeResult(.{}).init(&self.data);
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
