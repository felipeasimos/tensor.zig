const std = @import("std");
const utils = @import("utils.zig");
const CPUGEMM = @import("matmul/cpu.zig").matmul;
pub const op = @import("op.zig");
pub const func = @import("func.zig");
pub const iterator = @import("iterator.zig");

pub fn Tensor(comptime ElementType: type, comptime NDims: usize) type {
    return struct {
        /// used to tell if this type is a tensor
        pub const Marker: @TypeOf(Tensor) = Tensor;
        pub const ScalarType: type = ElementType;
        pub const n_dims: usize = NDims;
        pub const ElemHasFormat: bool = has_format: {
            break :has_format switch (@typeInfo(ElementType)) {
                .@"struct", .@"enum", .@"union", .@"opaque" => @hasDecl(ElementType, "format"),
                else => false,
            };
        };

        /// runtime metadata values
        pub const Metadata = struct {
            strides: [n_dims]usize,
            shape: [n_dims]usize,
            pub fn eqlShape(self: @This(), other: @This()) bool {
                for (self.shape, other.shape) |s, o| {
                    if (s != o) return false;
                }
                return true;
            }
            pub fn eqlStrides(self: @This(), other: @This()) bool {
                for (self.strides, other.strides) |s, o| {
                    if (s != o) return false;
                }
                return true;
            }
            pub fn eql(self: @This(), other: @This()) bool {
                return self.eqlShape(other) and self.eqlStrides(other);
            }
            pub fn major(layout: utils.MemoryLayout, shape: [n_dims]usize) @This() {
                return switch (layout) {
                    .ColumnMajor => columnMajor(shape),
                    .RowMajor => rowMajor(shape),
                };
            }
            pub fn rowMajor(shape: [n_dims]usize) @This() {
                return .{
                    .shape = shape,
                    .strides = utils.calculateStridesRowMajor(shape),
                };
            }
            pub fn columnMajor(shape: [n_dims]usize) @This() {
                return .{
                    .shape = shape,
                    .strides = utils.calculateStridesColumnMajor(shape),
                };
            }
            pub fn isContinuous(self: *const @This()) bool {
                return utils.stridesAreContiguous(self.shape, self.strides);
            }
            pub fn highestIndex(self: *const @This()) usize {
                if (comptime n_dims == 0) {
                    return 0;
                }
                return @reduce(
                    .Add,
                    (self.shape - @as(@Vector(self.shape.len, usize), @splat(1))) * self.strides,
                );
            }
            pub fn numScalars(self: *const @This()) usize {
                return @reduce(.Mul, @as(@Vector(self.shape.len, usize), self.shape));
            }
        };
        metadata: Metadata,
        data: []ScalarType,

        pub fn randomize(self: *const @This(), rand: std.Random) @This() {
            const sampler = comptime switch (@typeInfo(ScalarType)) {
                .comptime_int, .int => struct {
                    inline fn sample(r: std.Random) ScalarType {
                        return r.int(ScalarType);
                    }
                }.sample,
                .comptime_float, .float => struct {
                    inline fn sample(r: std.Random) ScalarType {
                        if (comptime ScalarType == f16) {
                            var f: ScalarType = undefined;
                            r.bytes(std.mem.asBytes(&f));
                            return f;
                        } else {
                            return r.float(ScalarType);
                        }
                        // For uniform:
                        // Or, for normal:
                        // var n = std.rand.Normal(dtype).init(r);
                        // return n.sample();
                    }
                }.sample,
                .bool => struct {
                    inline fn sample(r: std.Random) ScalarType {
                        return r.boolean();
                    }
                }.sample,
                else => @compileError("invalid dtype"),
            };
            var it = self.dataRefIter();
            while (it.next()) |data_ptr| {
                data_ptr.* = sampler(rand);
            }
            return self.*;
        }

        pub fn from(metadata: Metadata, data: []ScalarType) @This() {
            const highest_idx = metadata.highestIndex();
            return .{
                .data = data[0 .. highest_idx + 1],
                .metadata = metadata,
            };
        }

        pub fn dupe(allocator: std.mem.Allocator, metadata: Metadata, data: []ScalarType) !@This() {
            const highest_idx = metadata.highestIndex();
            const new_data = try allocator.alloc(ScalarType, highest_idx + 1);
            @memcpy(new_data, data);
            return .{
                .data = new_data,
                .metadata = metadata,
            };
        }

        pub fn zeroes(allocator: std.mem.Allocator, metadata: Metadata) !@This() {
            const highest_idx = metadata.highestIndex();
            const data = try allocator.alloc(ScalarType, highest_idx + 1);
            @memset(data, 0);
            return .{
                .data = data,
                .metadata = metadata,
            };
        }

        pub fn ones(allocator: std.mem.Allocator, metadata: Metadata) !@This() {
            const highest_idx = metadata.highestIndex();
            const data = try allocator.alloc(ScalarType, highest_idx + 1);
            @memset(data, 1);
            return .{
                .data = data,
                .metadata = metadata,
            };
        }

        pub fn alloc(allocator: std.mem.Allocator, metadata: Metadata) !@This() {
            const highest_idx = metadata.highestIndex();
            return .{
                .data = try allocator.alloc(ScalarType, highest_idx + 1),
                .metadata = metadata,
            };
        }

        pub fn deinit(self: *const @This(), allocator: std.mem.Allocator) void {
            allocator.free(self.data);
        }

        pub fn memset(self: *const @This(), value: ScalarType) @This() {
            var self_it = utils.getChildType(@TypeOf(self)).indicesIter();
            while (self_it.next()) |self_indices| {
                const self_idx = utils.getIndexAt(self_indices, self.strides);
                self.data[self_idx] = value;
            }
            return self;
        }

        fn checkIndexOutOfBounds(self: *const @This(), idxs: anytype) bool {
            if (idxs.len > n_dims) return false;
            for (idxs, 0..) |d, i| {
                if (d >= self.metadata.shape[i]) {
                    return false;
                }
            }
            return true;
        }

        pub inline fn scalar(self: *const @This(), idxs: [n_dims]usize) ScalarType {
            std.debug.assert(self.checkIndexOutOfBounds(idxs));
            const idx = utils.getIndexAt(idxs, self.metadata.strides);
            return self.data[idx];
        }

        pub inline fn scalarRef(self: *const @This(), idxs: [n_dims]usize) *ScalarType {
            const idx = utils.getIndexAt(idxs, self.metadata.strides);
            return &self.data[idx];
        }

        pub fn AsVectorResult() type {
            return;
        }

        pub inline fn asVector(self: anytype, comptime N: usize) @Vector(N, ScalarType) {
            std.debug.assert(self.metadata.isContinuous());
            return self.data[0..N].*;
        }

        pub fn SubTensorRef(comptime size: usize) type {
            if (size > n_dims) {
                @compileError("Can't resolve subtensor: too much dimensional subtraction");
            }
            if (size == n_dims) {
                return *ElementType;
            }
            return Tensor(ElementType, n_dims - size);
        }

        pub fn SubTensor(comptime size: usize) type {
            if (size > n_dims) {
                @compileError("Can't resolve subtensor: too much dimensional subtraction");
            }
            if (size == n_dims) {
                return ElementType;
            }
            return Tensor(ElementType, n_dims - size);
        }

        pub inline fn constRef(self: *const @This(), idxs: anytype) SubTensor(idxs.len) {
            if (comptime idxs.len == 0) {
                return .from(self.metadata, self.data);
            }
            if (comptime n_dims == idxs.len) {
                return self.scalar(idxs);
            }
            const strides_to_sub_tensor: [idxs.len]usize = self.metadata.strides[0..idxs.len].*;
            const start_idx = utils.getIndexAt(idxs, self.metadata.strides);
            const final_idx = start_idx + strides_to_sub_tensor[idxs.len - 1];

            const NewTensorType = SubTensor(idxs.len);
            const new_shape = self.metadata.shape[idxs.len..n_dims].*;
            const new_metadata = switch (utils.MemoryLayout.detectLayout(self.metadata.strides).?) {
                .RowMajor => NewTensorType.Metadata.rowMajor(new_shape),
                .ColumnMajor => NewTensorType.Metadata.columnMajor(new_shape),
            };

            return NewTensorType.from(new_metadata, self.data[start_idx..final_idx]);
        }

        pub inline fn ref(self: *const @This(), idxs: anytype) SubTensorRef(idxs.len) {
            if (comptime idxs.len == 0) {
                return .from(self.metadata, self.data);
            }
            if (comptime n_dims == idxs.len) {
                return self.scalarRef(idxs);
            }
            const strides_to_sub_tensor: [idxs.len]usize = self.metadata.strides[0..idxs.len].*;
            const start_idx = utils.getIndexAt(idxs, self.metadata.strides);
            const final_idx = start_idx + strides_to_sub_tensor[idxs.len - 1];

            const NewTensorType = SubTensor(idxs.len);
            const new_shape = self.metadata.shape[idxs.len..n_dims].*;
            const new_metadata = switch (utils.MemoryLayout.detectLayout(self.metadata.strides).?) {
                .RowMajor => NewTensorType.Metadata.rowMajor(new_shape),
                .ColumnMajor => NewTensorType.Metadata.columnMajor(new_shape),
            };

            return NewTensorType.from(new_metadata, self.data[start_idx..final_idx]);
        }

        /// get a subtensor. `idxs` needs to be an array.
        pub inline fn clone(self: *const @This(), allocator: std.mem.Allocator, idxs: anytype) !SubTensor(idxs.len) {
            if (comptime n_dims == idxs.len) {
                return self.scalar(idxs);
            }
            var clone_ptr = try alloc(allocator, self.metadata);
            _ = clone_ptr.copy(self.*);
            return clone_ptr;
        }

        pub inline fn reshape(self: *const @This(), new_shape: anytype) Tensor(ScalarType, new_shape.len) {
            std.debug.assert(self.metadata.isContinuous());
            const NewTensorType = Tensor(ScalarType, new_shape.len);
            const new_metadata = switch (utils.MemoryLayout.detectLayout(self.metadata.strides).?) {
                .RowMajor => NewTensorType.Metadata.rowMajor(new_shape),
                .ColumnMajor => NewTensorType.Metadata.columnMajor(new_shape),
            };
            const result = NewTensorType.from(new_metadata, self.data[0..]);
            return result;
        }

        /// like 'copy', but also copy the metadata
        pub inline fn deepCopy(self: *const @This(), other: @This()) void {
            self.metadata = other.metadata;
            self.copy(other);
        }

        /// doesn't have assert check, which allows it to be use by things like 'pack'
        pub inline fn copy(self: *const @This(), other: @This()) void {
            var self_it = other.indicesIter();
            var from_it = other.indicesIter();

            while (self_it.next()) |self_indices| {
                const from_indices = from_it.next().?;
                // will work for things like 'pack', since we abstract over majors!
                self.scalarRef(self_indices).* = other.scalar(from_indices);
            }
        }

        /// copy contents into a tensor with the desired major
        pub inline fn pack(self: *const @This(), allocator: std.mem.Allocator, major: utils.MemoryLayout) !@This() {
            const current_layout = utils.MemoryLayout.detectLayout(self.metadata.strides);
            if (current_layout == major) return self.clone(allocator, .{});
            const new_metadata = Metadata.major(major, self.metadata.shape);
            var clone_ptr = try alloc(allocator, new_metadata);
            clone_ptr.copy(self.*);
            return clone_ptr;
        }

        /// high performance multithreaded CPU GEMM
        /// allocator is needed to pack major friendly matrices inside
        pub inline fn matmul(self: *const @This(), allocator: std.mem.Allocator, io: std.Io, a: Tensor(ScalarType, 2), b: Tensor(ScalarType, 2)) !void {
            return CPUGEMM(allocator, io, self, a, b);
        }

        fn TupleOfIteratorsAndResults(tupleType: type, iteratorType: iterator.IteratorType) struct { type, type } {
            const length = utils.getTypeLength(tupleType);
            var iterTypes: []const type = &.{};
            var dtypes: []const type = &.{};
            for (0..length) |i| {
                const index_as_str = std.fmt.comptimePrint("{}", .{i});
                const T = utils.getChildType(@FieldType(tupleType, index_as_str));
                if (op.isTensor(T)) {
                    iterTypes = iterTypes ++ .{iteratorType.GetIteratorType(T)};
                    dtypes = dtypes ++ .{iteratorType.GetIteratorResultType(T)};
                } else {
                    iterTypes = iterTypes ++ .{T};
                    dtypes = dtypes ++ .{T};
                }
            }
            return .{ @Tuple(iterTypes), @Tuple(dtypes) };
        }

        pub inline fn wise(self: *const @This(), tuple: anytype, f: anytype) @This() {
            if (comptime !utils.isTuple(@TypeOf(tuple))) {
                @compileError("argument should be a tuple");
            }

            _, const Dtypes = TupleOfIteratorsAndResults(@TypeOf(tuple), .data);
            var dtypes: Dtypes = undefined;
            // change every element
            var result_iter = self.indicesIter();
            const length = comptime utils.getTypeLength(@TypeOf(tuple));

            while (result_iter.next()) |result_idxs| {
                inline for (0..length) |i| {
                    const T = utils.getChildType(@TypeOf(tuple[i]));
                    if (comptime op.isTensor(T)) {
                        dtypes[i] = tuple[i].scalar(result_idxs);
                    } else {
                        dtypes[i] = tuple[i];
                    }
                }
                self.scalarRef(result_idxs).* = f(dtypes);
            }
            return self.*;
        }

        fn initTupleIterators(tuple: anytype, iters: anytype) void {
            const length = comptime utils.getTypeLength(utils.getChildType(@TypeOf(iters)));
            inline for (0..length) |i| {
                const T = utils.getChildType(@TypeOf(tuple[i]));
                if (comptime op.isTensor(T)) {
                    iters[i] = utils.getChildType(@TypeOf(iters[i])).Type.initIterator(tuple[i]);
                } else {
                    iters[i] = tuple[i];
                }
            }
        }

        fn getTupleIteratorsResults(tuple: anytype, iters: anytype, dtypes: anytype) void {
            const length = comptime utils.getTypeLength(utils.getChildType(@TypeOf(iters)));
            inline for (0..length) |i| {
                const T = utils.getChildType(@TypeOf(tuple[i]));
                if (comptime op.isTensor(T)) {
                    dtypes[i] = iters[i].next().?;
                } else {
                    dtypes[i] = iters[i];
                }
            }
        }

        pub inline fn reduce(self: *const @This(), initial: anytype, tuple: anytype, f: anytype) void {
            const AccumulatorType = @TypeOf(initial);
            if (comptime !utils.isTuple(@TypeOf(tuple))) {
                @compileError("argument should be a tuple");
            }
            const num_iterations = utils.getTensorInTupleShape(tuple)[0];

            const IterTypes, const Dtypes = TupleOfIteratorsAndResults(@TypeOf(tuple), .subtensor);
            var dtypes: Dtypes = undefined;
            var iters: IterTypes = undefined;
            initTupleIterators(tuple, &iters);

            var accumulator = initial;
            for (0..num_iterations) |_| {
                getTupleIteratorsResults(tuple, &iters, &dtypes);
                accumulator = f(dtypes, accumulator);
            }
            if (comptime op.isTensor(AccumulatorType)) {
                _ = self.copy(accumulator);
            } else {
                self.scalarRef(.{0}).* = accumulator;
            }
        }

        pub inline fn transpose(self: *const @This(), shuffled_axises: anytype) @This() {
            if (comptime shuffled_axises.len == 0) {
                var shuffle_mask: [n_dims]usize = utils.createSequence(usize, n_dims);
                shuffle_mask[n_dims - 1] = n_dims - 2;
                shuffle_mask[n_dims - 2] = n_dims - 1;
                return self.transpose(shuffle_mask);
            }
            var new_strides: [n_dims]usize = undefined;
            var new_shape: [n_dims]usize = undefined;
            inline for (shuffled_axises, 0..) |axis_idxs, i| {
                new_strides[i] = self.metadata.strides[axis_idxs];
                new_shape[i] = self.metadata.shape[axis_idxs];
            }
            return .{
                .data = self.data,
                .metadata = .{
                    .shape = new_shape,
                    .strides = new_strides,
                },
            };
        }

        fn checkSliceRanges(self: *const @This(), ranges: anytype) bool {
            inline for (ranges, 0..) |range, i| {
                if (range[1] <= range[0] or range[1] > self.metadata.shape[i]) {
                    return false;
                }
            }
            return true;
        }

        pub inline fn slice(self: *const @This(), ranges: anytype) @This() {
            std.debug.assert(self.checkSliceRanges(ranges));

            var offset: usize = 0;

            inline for (0..ranges.len) |i| {
                offset += ranges[i][0] * self.metadata.strides[i];
            }

            var new_shape: [n_dims]usize = self.metadata.shape;
            inline for (0..ranges.len) |i| {
                const start = ranges[i][0];
                const end = ranges[i][1];
                new_shape[i] = end - start;
            }
            var new_strides: [n_dims]usize = self.metadata.strides;
            for (0..ranges.len) |i| {
                new_strides[i] = self.metadata.strides[i];
            }

            return .from(.{
                .strides = new_strides,
                .shape = new_shape,
            }, self.data[offset..]);
        }

        fn calculateBroadcastStrides(self: *const @This(), target_shape: anytype) @Vector(target_shape.len, usize) {
            const target_arr = utils.asArray(usize, target_shape);
            var new_strides: [target_arr.len]usize = undefined;

            if (target_arr.len < n_dims) {
                @compileError("Target shape must have at least as many dimensions as source shape");
            }

            const shape_offset = target_arr.len - n_dims;

            for (0..target_arr.len) |i| {
                if (i < shape_offset) {
                    new_strides[i] = 0;
                } else {
                    const orig_dim = i - shape_offset;
                    if (self.metadata.shape[orig_dim] == 1 and target_arr[i] > 1) {
                        new_strides[i] = 0;
                    } else if (self.metadata.shape[orig_dim] == target_arr[i]) {
                        new_strides[i] = self.metadata.strides[orig_dim];
                    } else {
                        std.debug.assert(false);
                    }
                }
            }

            return new_strides;
        }

        pub inline fn broadcast(self: anytype, target_shape: anytype) Tensor(ScalarType, target_shape.len) {
            return .from(.{
                .shape = target_shape,
                .strides = self.calculateBroadcastStrides(target_shape),
            }, self.data);
        }

        pub inline fn indicesIter(self: *const @This()) iterator.IndicesIterator(@This()) {
            return .init(self);
        }

        pub inline fn dataIter(self: *const @This()) iterator.DataIterator(@This()) {
            return .init(self);
        }

        pub inline fn dataRefIter(self: *const @This()) iterator.DataRefIterator(@This()) {
            return .init(self);
        }

        pub inline fn iter(self: *const @This()) iterator.Iterator(@This()) {
            return .init(self);
        }

        pub inline fn subTensorIter(self: *const @This()) iterator.SubTensorIterator(@This()) {
            return .init(self);
        }

        pub fn format(self: *const @This(), writer: anytype) !void {
            var indices: [n_dims]usize = [_]usize{0} ** n_dims;

            try self.formatRecursive(writer, 0, &indices);
            try writer.writeByte('\n');
        }

        fn formatRecursive(
            self: *const @This(),
            writer: anytype,
            dim: usize,
            indices: *[n_dims]usize,
        ) !void {
            try writer.writeByte('{');

            // Last dimension: print scalars
            if (dim == n_dims - 1) {
                for (0..self.metadata.shape[dim]) |i| {
                    if (i != 0)
                        try writer.writeAll(", ");

                    indices[dim] = i;

                    if (comptime ElemHasFormat) {
                        try writer.print("{f}", .{
                            self.scalar(indices.*),
                        });
                    } else {
                        try writer.print("{any}", .{
                            self.scalar(indices.*),
                        });
                    }
                }

                try writer.writeByte('}');
                return;
            }

            // Higher dimensions
            for (0..self.metadata.shape[dim]) |i| {
                if (i != 0) {
                    try writer.writeAll(",\n");

                    // indentation
                    for (0..dim + 1) |_| {
                        try writer.writeByte('\t');
                    }
                }

                indices[dim] = i;

                try self.formatRecursive(
                    writer,
                    dim + 1,
                    indices,
                );
            }

            try writer.writeByte('}');
        }
    };
}
