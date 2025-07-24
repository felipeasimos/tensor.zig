const std = @import("std");
const utils = @import("utils.zig");
pub const op = @import("op.zig");
pub const func = @import("func.zig");
pub const iterator = @import("iterator.zig");

/// OwnedTensor owns the underlying tensor data and can make changes to it
/// read-only tensor view can be accessed with the `view()` method
pub fn Tensor(comptime dtype: type, comptime _shape: anytype) type {
    @setEvalBranchQuota(10000);
    return InnerTensor(dtype, _shape, utils.calculateStrides(_shape), false, false);
}

/// TensorView is a compile-time type that represents a view into a tensor.
/// Never writes to the underlying data
pub fn TensorView(comptime dtype: type, comptime _shape: anytype) type {
    @setEvalBranchQuota(10000);
    return InnerTensor(dtype, _shape, utils.calculateStrides(_shape), true, true);
}

pub fn InnerTensor(comptime dtype: type, comptime _shape: anytype, comptime _strides: anytype, comptime is_ref: bool, comptime readonly: bool) type {
    const dtype_info = @typeInfo(dtype);

    if (dtype_info != .float and dtype_info != .int) {
        @compileError("Only floats and integers are valid tensor dtypes");
    }

    const shape_arr = utils.asArray(usize, _shape);
    const strides_arr = utils.asArray(usize, _strides);

    const _is_view = (is_ref and readonly);

    const total_num_scalars = @reduce(.Mul, @as(@Vector(shape_arr.len, usize), shape_arr));
    const highest_idx = @reduce(
        .Add,
        (shape_arr - @as(@Vector(shape_arr.len, usize), @splat(1))) * strides_arr,
    );
    const DataSequenceType = if (is_ref)
        []dtype
    else
        [total_num_scalars]dtype;

    const ScalarResult = if (readonly) dtype else *dtype;

    return struct {
        comptime shape: @TypeOf(shape_arr) = shape_arr,
        comptime strides: @TypeOf(strides_arr) = strides_arr,
        comptime dtype: type = dtype,
        comptime num_scalars: usize = total_num_scalars,
        comptime is_reference: bool = is_ref,
        comptime is_view: bool = _is_view,
        comptime is_readonly: bool = readonly,
        comptime ScalarType: type = ScalarResult,

        data: DataSequenceType,

        pub fn random(rand: std.Random) @This() {
            var new = @This(){ .data = undefined };
            new.randomize(rand);
            return new;
        }

        pub fn randomize(self: anytype, rand: std.Random) void {
            switch (@typeInfo(dtype)) {
                .comptime_int, .int => {
                    for (0..self.data.len) |i| {
                        self.data[i] = rand.int(dtype);
                    }
                },
                .comptime_float, .float => {
                    for (0..self.data.len) |i| {
                        self.data[i] = rand.floatNorm(dtype);
                    }
                },
                else => @compileError("invalid dtype"),
            }
            rand.bytes(std.mem.asBytes(self.data[0..]));
        }

        pub fn init(data: []dtype) @This() {
            if (comptime is_ref) {
                return .{ .data = data[0 .. highest_idx + 1] };
            }
            var new: @This() = .{ .data = undefined };
            std.mem.copyForwards(dtype, &new.data, data[0 .. highest_idx + 1]);
            return new;
        }

        pub fn zeroes() @This() {
            var new: @This() = undefined;
            @memset(&new.data, 0);
            return new;
        }

        pub inline fn scalar(self: anytype, idxs: @TypeOf(shape_arr)) ScalarResult {
            const idx = utils.getIndexAt(idxs, self.strides);
            if (comptime readonly) {
                return self.data[idx];
            }
            return &self.data[idx];
        }

        fn ViewResult(comptime size: usize) type {
            if (comptime _shape.len - size == 0) {
                return ScalarResult;
            }
            const new_shape = comptime utils.asSubArray(usize, shape_arr, size, shape_arr.len - 1);
            const new_strides = comptime utils.calculateStrides(new_shape);
            return InnerTensor(
                dtype,
                new_shape,
                new_strides,
                true,
                true,
            );
        }

        pub inline fn view(self: anytype, idxs: anytype) ViewResult(idxs.len) {
            if (comptime is_ref) {
                return ViewResult(idxs.len).init(self.data);
            }
            return ViewResult(idxs.len).init(@as([]dtype, self.data[0..]));
        }

        fn MutResult(comptime size: usize) type {
            if (comptime _shape.len - size == 0) {
                return ScalarResult;
            }
            const new_shape = comptime utils.asSubArray(usize, shape_arr, size, shape_arr.len - 1);
            const new_strides = comptime utils.calculateStrides(new_shape);
            return InnerTensor(
                dtype,
                new_shape,
                new_strides,
                true,
                false,
            );
        }

        /// get a mutable view
        pub inline fn mut(self: anytype, idxs: anytype) MutResult(idxs.len) {
            if (comptime idxs.len == 0) {
                return MutResult(0).init(self.data[0..]);
            }
            if (comptime _shape.len - idxs.len == 0) {
                return self.scalar(idxs);
            }
            const strides_to_sub_tensor = comptime utils.asSubVector(usize, self.strides, 0, idxs.len - 1);
            const start_idx = utils.getIndexAt(idxs, self.strides);
            const final_idx = start_idx + strides_to_sub_tensor[idxs.len - 1];
            return MutResult(idxs.len).init(self.data[start_idx..final_idx]);
        }

        fn CloneResult(comptime size: usize) type {
            if (comptime shape_arr.len - size == 0) {
                return dtype;
            }
            const new_shape = comptime utils.asSubArray(usize, shape_arr, size, shape_arr.len - 1);
            const new_strides = comptime utils.calculateStrides(new_shape);
            return InnerTensor(
                dtype,
                new_shape,
                new_strides,
                false,
                readonly,
            );
        }

        /// get a subtensor. `idxs` needs to be an array.
        pub inline fn clone(self: anytype, idxs: anytype) CloneResult(idxs.len) {
            if (comptime idxs.len == 0) {
                return CloneResult(0).init(&self.data);
            }
            if (comptime shape_arr.len - idxs.len == 0) {
                return self.data[utils.getIndexAt(idxs, self.strides)];
            }
            const strides_to_sub_tensor = comptime utils.asSubVector(usize, self.strides, 0, idxs.len - 1);
            const start_idx = utils.getIndexAt(idxs, self.strides);
            const final_idx = start_idx + strides_to_sub_tensor[idxs.len - 1];
            return CloneResult(idxs.len).init(self.data[start_idx..final_idx]);
        }

        fn stridesAreContiguous() bool {
            const contiguous_strides: [strides_arr.len]usize = utils.calculateStrides(shape_arr);
            return std.mem.eql(usize, &strides_arr, &contiguous_strides);
        }

        fn ReshapeResult(comptime shape: anytype) type {
            return InnerTensor(dtype, shape, utils.calculateStrides(shape), is_ref, readonly);
        }

        pub inline fn reshape(self: anytype, comptime shape: anytype) ReshapeResult(shape) {
            if (comptime !stridesAreContiguous()) {
                @compileError("Can't reshape a tensor without contiguous strides");
            }
            const result = ReshapeResult(shape).init(self.data[0..]);
            if (comptime result.num_scalars != self.num_scalars) {
                @compileError("Invalid reshape size (the final number of scalars don't match the current tensor)");
            }
            return result;
        }

        fn otherValue(other: anytype, i: usize) dtype {
            const T = @TypeOf(other);
            switch (@typeInfo(T)) {
                .comptime_int, .comptime_float, .int, .float => return other,
                .pointer, .@"struct" => return other.data[i],
                inline else => other,
            }
            @compileError(std.fmt.comptimePrint("Invalid operand type {} for {}", .{ T, @This() }));
        }

        pub inline fn copy(self: anytype, from: anytype) void {
            for (0..self.num_scalars) |i| {
                self.data[i] = from.data[i];
            }
        }

        pub inline fn apply(self: anytype, f: fn (dtype) dtype) void {
            if (comptime readonly) {
                @compileError("Cannot apply function to readonly tensor");
            }
            var it = self.iter();
            while (it.next()) |item| {
                const data_idx = utils.getIndexAt(item.indices, strides_arr);
                self.data[data_idx] = f(self.data[data_idx]);
            }
        }

        pub inline fn wise(self: anytype, other: anytype, result: anytype, f: fn (dtype, dtype) dtype) void {
            for (0..self.num_scalars) |i| {
                const other_value = otherValue(other, i);
                result.data[i] = f(self.data[i], other_value);
            }
        }

        fn WiseNewResult() type {
            return InnerTensor(dtype, shape_arr, strides_arr, false, false);
        }

        pub inline fn wiseNew(self: anytype, other: anytype, f: fn (dtype, dtype) dtype) WiseNewResult() {
            var result = WiseNewResult(){ .data = undefined };
            _ = self.wise(other, &result, f);
            return result;
        }

        fn TransposeResult(comptime shuffled_axises: anytype) type {
            if (comptime shuffled_axises.len == 0) {
                var mask = utils.createSequence(usize, strides_arr.len);
                const tmp = mask[strides_arr.len - 1];
                mask[strides_arr.len - 1] = mask[strides_arr.len - 2];
                mask[strides_arr.len - 2] = tmp;
                return TransposeResult(mask);
            }
            const new_strides = @shuffle(
                usize,
                strides_arr,
                undefined,
                shuffled_axises,
            );
            const new_shape = @shuffle(
                usize,
                shape_arr,
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

        pub inline fn transpose(self: anytype, comptime shuffled_axises: anytype) TransposeResult(shuffled_axises) {
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

        pub inline fn slice(self: anytype, comptime ranges: anytype) SliceResult(ranges) {
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
                    utils.getIndexAt(low_ranges_arr, strides_arr),
                    utils.getIndexAt(high_ranges_arr, strides_arr),
                };
            };
            return SliceResult(ranges).init(self.data[start_idx .. final_idx + 1]);
        }

        fn BroadcastResult(comptime target_shape: anytype) type {
            const target_arr = utils.asArray(usize, target_shape);
            const new_strides = calculateBroadcastStrides(target_shape);
            return InnerTensor(dtype, target_arr, new_strides, true, readonly);
        }

        fn calculateBroadcastStrides(comptime target_shape: anytype) @Vector(target_shape.len, usize) {
            const target_arr = utils.asArray(usize, target_shape);
            var new_strides: [target_arr.len]usize = undefined;

            if (target_arr.len < shape_arr.len) {
                @compileError("Target shape must have at least as many dimensions as source shape");
            }

            const shape_offset = target_arr.len - shape_arr.len;

            for (0..target_arr.len) |i| {
                if (i < shape_offset) {
                    new_strides[i] = 0;
                } else {
                    const orig_dim = i - shape_offset;
                    if (shape_arr[orig_dim] == 1 and target_arr[i] > 1) {
                        new_strides[i] = 0;
                    } else if (shape_arr[orig_dim] == target_arr[i]) {
                        new_strides[i] = strides_arr[orig_dim];
                    } else {
                        @compileError("Invalid broadcast: incompatible dimensions");
                    }
                }
            }

            return new_strides;
        }

        pub inline fn broadcast(self: anytype, comptime target_shape: anytype) BroadcastResult(target_shape) {
            return BroadcastResult(target_shape).init(self.data);
        }

        pub fn indicesIter() iterator.IndicesIterator(@This()) {
            return iterator.IndicesIterator(@This()).init();
        }

        pub fn dataIter(self: anytype) iterator.DataIterator(@This()) {
            return iterator.DataIterator(@This()).init(self);
        }

        pub fn iter(self: anytype) iterator.Iterator(@This()) {
            return iterator.Iterator(@This()).init(self);
        }
    };
}
