const utils = @import("utils.zig");

inline fn incrementIndices(iter: anytype, comptime shape_arr: anytype) void {
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
    const T = utils.getChildType(TensorType);
    const shape_arr = utils.getComptimeFieldValue(T, "shape").?;
    const strides_arr = utils.getComptimeFieldValue(T, "strides").?;
    const dtype = utils.getComptimeFieldValue(T, "dtype").?;

    return struct {
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

        pub inline fn next(self: *Self) ?struct { indices: @TypeOf(shape_arr), value: dtype } {
            if (self.finished) return null;

            const current_idx = self.current_indices;
            incrementIndices(self, shape_arr);
            const data_idx = utils.getIndexAt(current_idx, strides_arr);
            const value = self.tensor.data[data_idx];
            return .{ .indices = current_idx, .value = value };
        }
    };
}

pub fn IndicesIterator(comptime TensorType: type) type {
    const T = utils.getChildType(TensorType);
    const shape_arr = utils.getComptimeFieldValue(T, "shape").?;
    return struct {
        const Self = @This();

        current_indices: @TypeOf(shape_arr),
        finished: bool,

        pub fn init() Self {
            return Self{
                .current_indices = @as(@Vector(shape_arr.len, usize), @splat(0)),
                .finished = false,
            };
        }

        pub fn next(self: *Self) ?@TypeOf(shape_arr) {
            if (self.finished) return null;

            const current_idx = self.current_indices;
            incrementIndices(self, shape_arr);
            return current_idx;
        }
    };
}

pub fn DataIterator(comptime TensorType: type) type {
    const T = utils.getChildType(TensorType);
    const shape_arr = utils.getComptimeFieldValue(T, "shape").?;
    const strides_arr = utils.getComptimeFieldValue(T, "strides").?;
    const dtype = utils.getComptimeFieldValue(T, "dtype").?;

    return struct {
        const Self = @This();

        tensor: *const T,
        current_indices: @TypeOf(shape_arr),
        finished: bool,

        pub fn init(tensor: *const T) Self {
            return Self{
                .tensor = tensor,
                .current_indices = @as(@Vector(shape_arr.len, usize), @splat(0)),
                .finished = false,
            };
        }

        pub fn next(self: *Self) ?dtype {
            if (self.finished) return null;

            const current_idx = self.current_indices;
            self.incrementIndices(shape_arr);
            const data_idx = utils.getIndexAt(current_idx, strides_arr);
            return self.tensor.data[data_idx];
        }
    };
}

pub fn DataRefIterator(comptime TensorType: type) type {
    const T = utils.getChildType(TensorType);
    const shape_arr = utils.getComptimeFieldValue(T, "shape").?;
    const strides_arr = utils.getComptimeFieldValue(T, "strides").?;
    const dtype = utils.getComptimeFieldValue(T, "dtype").?;

    return struct {
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
    const T = utils.getChildType(TensorType);
    const shape_arr = utils.getComptimeFieldValue(T, "shape").?;

    return struct {
        const Self = @This();

        tensor: *T,
        current_index: usize = 0,

        pub fn init(tensor: *T) Self {
            return Self{
                .tensor = tensor,
            };
        }

        pub fn next(self: *Self) ?@TypeOf(self.tensor.RefResult(1)) {
            if (self.current_index >= shape_arr[0]) return null;

            const current_idx = self.current_index;
            self.current_index += 1;
            return self.tensor.ref(.{current_idx});
        }
    };
}
