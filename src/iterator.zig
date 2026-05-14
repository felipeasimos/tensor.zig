const utils = @import("utils.zig");

pub const IteratorType = enum {
    data,
    dataRef,
    indices,
    both,
    subtensor,

    pub fn GetIteratorType(self: @This(), TensorType: type) type {
        return switch (self) {
            .both => Iterator(TensorType),
            .dataRef => DataRefIterator(TensorType),
            .data => DataIterator(TensorType),
            .indices => IndicesIterator(TensorType),
            .subtensor => SubTensorIterator(TensorType),
        };
    }
    pub fn GetIteratorResultType(self: @This(), TensorType: type) type {
        return switch (self) {
            .both => struct {
                indices: [TensorType.n_dims]usize,
                value: TensorType.ScalarType,
            },
            .dataRef => *TensorType.ScalarType,
            .data => TensorType.ScalarType,
            .indices => [TensorType.n_dims]usize,
            .subtensor => TensorType.SubTensor(1),
        };
    }
    pub inline fn initIterator(comptime self: @This(), tensor: anytype) GetIteratorType(self, @TypeOf(tensor)) {
        return switch (comptime self) {
            .both => tensor.iter(),
            .dataRef => tensor.dataRefIter(),
            .data => tensor.dataIter(),
            .indices => tensor.indicesIter(),
            .subtensor => tensor.subTensorIter(),
        };
    }
};

inline fn incrementIndices(iter: anytype, shape_arr: anytype) void {
    inline for (0..shape_arr.len) |rev_i| {
        const dim = shape_arr.len - 1 - rev_i;
        iter.current_indices[dim] += 1;
        if (iter.current_indices[dim] < shape_arr[dim]) {
            return;
        }
        iter.current_indices[dim] = 0;
    }
    iter.finished = true;
}

pub fn Iterator(comptime TensorType: type) type {
    const dtype = TensorType.ScalarType;
    return struct {
        pub const Type: IteratorType = .both;
        tensor: TensorType,
        current_indices: [TensorType.n_dims]usize,
        finished: bool,

        pub fn init(tensor: *const TensorType) @This() {
            return .{
                .tensor = tensor.*,
                .current_indices = @as(@Vector(TensorType.n_dims, usize), @splat(0)),
                .finished = false,
            };
        }

        pub fn next(self: *@This()) ?struct {
            indices: [TensorType.n_dims]usize,
            value: dtype,
        } {
            if (self.finished) return null;

            const current_idx = self.current_indices;
            incrementIndices(self, self.tensor.metadata.shape);
            return .{
                .indices = current_idx,
                .value = self.tensor.scalar(current_idx),
            };
        }
    };
}

pub fn IndicesIterator(comptime T: type) type {
    const TensorType = utils.getChildType(T);
    return struct {
        pub const Type: IteratorType = .indices;
        const n_dims = TensorType.n_dims;
        shape: [n_dims]usize,
        current_indices: [n_dims]usize,
        finished: bool,

        pub fn init(tensor: *const TensorType) @This() {
            return .{
                .shape = tensor.metadata.shape,
                .current_indices = .{0} ** n_dims,
                .finished = false,
            };
        }

        pub fn next(self: *@This()) ?[n_dims]usize {
            if (self.finished) return null;

            const current_idx = self.current_indices;
            incrementIndices(self, self.shape);
            return current_idx;
        }
    };
}

pub fn DataIterator(comptime T: type) type {
    const TensorType = utils.getChildType(T);
    const n_dims = TensorType.n_dims;
    return struct {
        pub const Type: IteratorType = .data;
        const Self = @This();

        tensor: TensorType,
        current_indices: [n_dims]usize,
        finished: bool,

        pub fn init(tensor: *const TensorType) Self {
            return Self{
                .tensor = tensor.*,
                .current_indices = @as(@Vector(n_dims, usize), @splat(0)),
                .finished = false,
            };
        }

        pub fn next(self: *Self) ?TensorType.ScalarType {
            if (self.finished) return null;

            const current_idx = self.current_indices;
            incrementIndices(self, self.tensor.metadata.shape);
            return self.tensor.scalar(current_idx);
        }
    };
}

pub fn DataRefIterator(comptime TensorType: type) type {
    const T = utils.getChildType(TensorType);
    const shape_arr = T.Shape;
    const strides_arr = T.Strides;
    const dtype = T.Dtype;

    return struct {
        pub const Type: IteratorType = .dataRef;
        const Self = @This();

        tensor: *T,
        current_indices: @TypeOf(shape_arr),
        finished: bool,

        pub fn init(tensor: *T) Self {
            return Self{
                .tensor = tensor,
                .current_indices = @as(@Vector(shape_arr.len, usize), @splat(0)),
                .finished = false,
            };
        }

        pub fn next(self: *Self) ?*dtype {
            if (self.finished) return null;

            const current_idx = self.current_indices;
            incrementIndices(self, shape_arr);
            const data_idx = utils.getIndexAt(current_idx, strides_arr);
            return &self.tensor.data[data_idx];
        }
    };
}

pub fn SubTensorIterator(comptime TensorType: type) type {
    return struct {
        pub const Type: IteratorType = .subtensor;
        const Self = @This();

        tensor: TensorType,
        current_index: usize = 0,
        finished: bool,

        pub fn init(tensor: *const TensorType) Self {
            return Self{
                .tensor = tensor.*,
                .finished = tensor.metadata.shape.len == 0,
            };
        }

        pub fn next(self: *Self) ?TensorType.SubTensor(1) {
            if (self.finished) return null;
            if (self.current_index >= self.tensor.metadata.shape[0]) {
                self.finished = true;
                return null;
            }
            const current_idx = self.current_index;
            self.current_index += 1;
            return self.tensor.constRef(.{current_idx});
        }
    };
}
