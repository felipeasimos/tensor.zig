const std = @import("std");
const utils = @import("utils.zig");
pub const op = @import("op.zig");
pub const func = @import("func.zig");
pub const iterator = @import("iterator.zig");

/// OwnedTensor owns the underlying tensor data and can make changes to it
/// read-only tensor view can be accessed with the `view()` method
pub fn Tensor(comptime dtype: type, comptime _shape: anytype) type {
    @setEvalBranchQuota(10000);
    return InnerTensor(dtype, _shape, utils.calculateStrides(_shape), false);
}

/// TensorRef is a type that doesn't store the data, just point to it.
pub fn TensorRef(comptime dtype: type, comptime _shape: anytype) type {
    @setEvalBranchQuota(10000);
    return InnerTensor(dtype, _shape, utils.calculateStrides(_shape), true);
}

const type_factory_marker: u8 = undefined;

pub fn InnerTensor(comptime dtype: type, comptime _shape: anytype, comptime _strides: anytype, comptime is_ref: bool) type {
    const dtype_info = @typeInfo(dtype);
    if (dtype_info != .float and dtype_info != .int) {
        @compileError("Only floats and integers are valid tensor dtypes");
    }

    const shape_arr = utils.asArray(usize, _shape);
    const strides_arr = utils.asArray(usize, _strides);

    const total_num_scalars = @reduce(.Mul, @as(@Vector(shape_arr.len, usize), shape_arr));
    const highest_idx = @reduce(
        .Add,
        (shape_arr - @as(@Vector(shape_arr.len, usize), @splat(1))) * strides_arr,
    );
    const DataSequenceType = if (is_ref)
        []dtype
    else
        [total_num_scalars]dtype;

    return struct {
        comptime shape: @TypeOf(shape_arr) = shape_arr,
        comptime strides: @TypeOf(strides_arr) = strides_arr,
        comptime dtype: type = dtype,
        comptime num_scalars: usize = total_num_scalars,
        comptime is_reference: bool = is_ref,
        comptime strides_are_contiguous: bool = utils.stridesAreContiguous(shape_arr, strides_arr),
        comptime factory_function: @TypeOf(InnerTensor) = InnerTensor,

        data: DataSequenceType,

        pub fn random(rand: std.Random) @This() {
            var new = @This(){ .data = undefined };
            new.randomize(rand);
            return new;
        }

        pub fn randomize(self: anytype, rand: std.Random) void {
            const sampler = comptime switch (@typeInfo(dtype)) {
                .comptime_int, .int => struct {
                    inline fn sample(r: std.Random) dtype {
                        return r.int(dtype);
                    }
                }.sample,
                .comptime_float, .float => struct {
                    inline fn sample(r: std.Random) dtype {
                        // For uniform:
                        return r.float(dtype);
                        // Or, for normal:
                        // var n = std.rand.Normal(dtype).init(r);
                        // return n.sample();
                    }
                }.sample,
                else => @compileError("invalid dtype"),
            };

            if (comptime self.strides_are_contiguous) {
                for (0..self.num_scalars) |i| {
                    self.data[i] = sampler(rand);
                }
            } else {
                var it = self.dataRefIter();
                while (it.next()) |data_ptr| {
                    data_ptr.* = sampler(rand);
                }
            }
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

        pub inline fn scalar(self: anytype, idxs: @TypeOf(shape_arr)) dtype {
            const idx = utils.getIndexAt(idxs, self.strides);
            return self.data[idx];
        }

        pub inline fn scalarRef(self: anytype, idxs: @TypeOf(shape_arr)) *dtype {
            const idx = utils.getIndexAt(idxs, self.strides);
            return &self.data[idx];
        }

        fn RefResult(comptime size: usize) type {
            if (comptime _shape.len - size == 0) {
                return *dtype;
            }
            const new_shape = comptime utils.asSubArray(usize, shape_arr, size, shape_arr.len - 1);
            const new_strides = comptime utils.calculateStrides(new_shape);
            return InnerTensor(
                dtype,
                new_shape,
                new_strides,
                true,
            );
        }

        pub inline fn ref(self: *@This(), idxs: anytype) RefResult(idxs.len) {
            if (comptime idxs.len == 0) {
                return RefResult(0).init(self.data[0..]);
            }
            if (comptime _shape.len - idxs.len == 0) {
                return self.scalarRef(idxs);
            }
            const strides_to_sub_tensor = comptime utils.asSubVector(usize, self.strides, 0, idxs.len - 1);
            const start_idx = utils.getIndexAt(idxs, self.strides);
            const final_idx = start_idx + strides_to_sub_tensor[idxs.len - 1];
            return RefResult(idxs.len).init(self.data[start_idx..final_idx]);
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

        fn ReshapeResult(comptime shape: anytype) type {
            return InnerTensor(dtype, shape, utils.calculateStrides(shape), is_ref);
        }

        pub inline fn reshape(self: anytype, comptime shape: anytype) ReshapeResult(shape) {
            if (comptime !self.strides_are_contiguous) {
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
            if (comptime self.strides_are_contiguous) {
                for (0..self.num_scalars) |i| {
                    self.data[i] = from.data[i];
                }
            } else {
                var it = self.indicesIter();
                while (it.next()) |indices| {
                    const data_idx = utils.getIndexAt(indices, strides_arr);
                    self.data[data_idx] = from.data[data_idx];
                }
            }
        }

        pub inline fn matmul(self: anytype, a: anytype, b: anytype) void {
            // (P, Q) x (Q, R) -> (P, R)
            const P = comptime a.shape[0];
            const Q = comptime a.shape[1];
            const R = comptime b.shape[1];
            if (comptime (self.shape[0] != P or self.shape[1] != R or b.shape[0] != Q)) {
                @compileError(std.fmt.comptimePrint("Number of columns don't match with number of rows: {any} x {any} -> {any}", .{ a.shape, b.shape, self.shape }));
            }
            for (0..P) |i| {
                for (0..R) |j| {
                    var tmp: a.dtype = 0;
                    for (0..Q) |k| {
                        const index_self = utils.getIndexAt(.{ i, k }, a.strides);
                        const index_other = utils.getIndexAt(.{ k, j }, b.strides);
                        tmp += a.data[index_self] * b.data[index_other];
                    }
                    const index_result = utils.getIndexAt(.{ i, j }, self.strides);
                    self.data[index_result] = tmp;
                }
            }
        }

        fn TupleOfIteratorsType(comptime tensorsType: type) type {
            const length = utils.getTypeLength(tensorsType);
            var types: [length]type = undefined;
            inline for (0..length) |i| {
                const index_as_str = std.fmt.comptimePrint("{}", .{i});
                const T = @FieldType(tensorsType, index_as_str);
                if (comptime op.isTensor(utils.getChildType(T))) {
                    types[i] = iterator.IndicesIterator(T);
                } else {
                    types[i] = T;
                }
            }
            return std.meta.Tuple(&types);
        }

        fn TupleOfDtypes(comptime tensorsType: type) type {
            const length = utils.getTypeLength(tensorsType);
            var types: [length]type = undefined;
            for (0..length) |i| {
                const index_as_str = std.fmt.comptimePrint("{}", .{i});
                const T = @FieldType(tensorsType, index_as_str);
                if (comptime op.isTensor(utils.getChildType(T))) {
                    const TensorType = utils.getChildType(T);
                    const current_dtype = utils.getComptimeFieldValue(TensorType, "dtype").?;
                    types[i] = current_dtype;
                } else {
                    types[i] = T;
                }
            }
            return std.meta.Tuple(&types);
        }

        pub inline fn wise(self: *@This(), tuple: anytype, f: anytype) void {
            if (comptime !utils.isTuple(@TypeOf(tuple))) {
                @compileError("argument should be a tuple");
            }
            const length = comptime utils.getTypeLength(@TypeOf(tuple));
            comptime var iters: TupleOfIteratorsType(@TypeOf(tuple)) = undefined;
            // initialize iterators
            comptime {
                for (0..length) |i| {
                    const index_as_str = std.fmt.comptimePrint("{}", .{i});
                    const T = @FieldType(@TypeOf(tuple), index_as_str);
                    if (op.isTensor(utils.getChildType(T))) {
                        const TensorType = utils.getChildType(T);
                        iters[i] = TensorType.indicesIter();
                    } else {
                        iters[i] = tuple[i];
                    }
                }
            }
            // change every element
            var dtypes: TupleOfDtypes(@TypeOf(tuple)) = undefined;
            comptime var result_iter = utils.getChildType(@TypeOf(self)).indicesIter();

            inline while (comptime result_iter.next()) |result_idxs| {
                inline for (0..length) |i| {
                    const index_as_str = comptime std.fmt.comptimePrint("{}", .{i});
                    const T = comptime @FieldType(@TypeOf(tuple), index_as_str);
                    if (comptime op.isTensor(utils.getChildType(T))) {
                        const TensorType = comptime utils.getChildType(T);
                        const strides = comptime utils.getComptimeFieldValue(TensorType, "strides").?;
                        const idxs = (comptime iters[i].next()).?;
                        const data_idx = comptime utils.getIndexAt(idxs, strides);
                        dtypes[i] = tuple[i].data[data_idx];
                    } else {
                        dtypes[i] = tuple[i];
                    }
                }
                self.scalarRef(result_idxs).* = f(dtypes);
            }
        }

        // pub inline fn reduce(self: *@This(), initial: dtype, tuple: anytype, f: anytype) void {
        //     if (comptime !utils.isTuple(@TypeOf(tuple))) {
        //         @compileError("argument should be a tuple");
        //     }
        //
        //     const subtensor_size = comptime (self.num_scalars / self.shape[0]);
        //
        //     const length = comptime utils.getTypeLength(@TypeOf(tuple));
        //     comptime var iters: TupleOfIteratorsType(@TypeOf(tuple)) = undefined;
        //     // initialize iterators
        //     comptime {
        //         for (0..length) |i| {
        //             const index_as_str = std.fmt.comptimePrint("{}", .{i});
        //             const T = @FieldType(@TypeOf(tuple), index_as_str);
        //             if (op.isTensor(utils.getChildType(T))) {
        //                 const TensorType = utils.getChildType(T);
        //                 iters[i] = TensorType.indicesIter();
        //             } else {
        //                 iters[i] = tuple[i];
        //             }
        //         }
        //     }
        //     // change every element
        //     var dtypes: TupleOfDtypes(@TypeOf(tuple)) = undefined;
        //     comptime var result_iter = utils.getChildType(@TypeOf(self)).indicesIter();
        //
        //     var accumulator = initial;
        //     inline while (comptime result_iter.next(), 0..subtensor_size) |result_idxs| {
        //         inline for (0..length) |i| {
        //             const index_as_str = comptime std.fmt.comptimePrint("{}", .{i});
        //             const T = comptime @FieldType(@TypeOf(tuple), index_as_str);
        //             if (comptime op.isTensor(utils.getChildType(T))) {
        //                 const TensorType = comptime utils.getChildType(T);
        //                 const strides = comptime utils.getComptimeFieldValue(TensorType, "strides").?;
        //                 const idxs = (comptime iters[i].next()).?;
        //                 const data_idx = comptime utils.getIndexAt(idxs, strides);
        //                 dtypes[i] = tuple[i].data[data_idx];
        //             } else {
        //                 dtypes[i] = tuple[i];
        //             }
        //         }
        //         accumulator = f(dtypes, accumulator);
        //     }
        //     self.scalarRef(result_idxs).*
        // }

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
            return InnerTensor(dtype, new_shape, new_strides, is_ref);
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
            if (comptime !self.strides_are_contiguous) {
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
            return InnerTensor(dtype, target_arr, new_strides, true);
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
            return BroadcastResult(target_shape).init(self.data[0..]);
        }

        pub inline fn indicesIter() iterator.IndicesIterator(@This()) {
            return iterator.IndicesIterator(@This()).init();
        }

        pub inline fn dataIter(self: *const @This()) iterator.DataIterator(@This()) {
            return iterator.DataIterator(@This()).init(self);
        }

        pub inline fn dataRefIter(self: *@This()) iterator.DataIterator(@This()) {
            return iterator.DataIterator(@This()).init(self);
        }

        pub inline fn iter(self: *@This()) iterator.Iterator(@This()) {
            return iterator.Iterator(@This()).init(self);
        }
    };
}
