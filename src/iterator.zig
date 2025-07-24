const utils = @import("utils.zig");

pub fn Iterator(comptime TensorType: type) type {
    const shape_arr = utils.getComptimeFieldValue(TensorType, "shape").?;
    const strides_arr = utils.getComptimeFieldValue(TensorType, "strides").?;
    const ScalarType = utils.getComptimeFieldValue(TensorType, "ScalarType").?;
    const readonly = utils.getComptimeFieldValue(TensorType, "is_readonly").?;

    const TensorPtrType = if (comptime readonly)
        *const TensorType
    else
        *TensorType;
    return struct {
        const Self = @This();

        tensor: TensorPtrType,
        current_indices: @TypeOf(shape_arr),
        finished: bool,

        pub fn init(tensor: TensorPtrType) Self {
            return Self{
                .tensor = tensor,
                .current_indices = @as(@Vector(shape_arr.len, usize), @splat(0)),
                .finished = false,
            };
        }

        pub inline fn next(self: *Self) ?struct { indices: @TypeOf(shape_arr), value: ScalarType } {
            if (self.finished) return null;

            const current_idx = self.current_indices;
            self.incrementIndices();
            const data_idx = utils.getIndexAt(current_idx, strides_arr);
            const value = if (comptime readonly)
                self.tensor.data[data_idx]
            else
                &self.tensor.data[data_idx];
            return .{ .indices = current_idx, .value = value };
        }

        fn incrementIndices(self: *Self) void {
            var dim: usize = shape_arr.len - 1;
            while (true) {
                self.current_indices[dim] += 1;
                if (self.current_indices[dim] < shape_arr[dim]) {
                    return;
                }

                self.current_indices[dim] = 0;
                if (dim == 0) {
                    self.finished = true;
                    return;
                }
                dim -= 1;
            }
        }
    };
}

pub fn IndicesIterator(comptime TensorType: type) type {
    const shape_arr = utils.getComptimeFieldValue(TensorType, "shape").?;
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
            self.incrementIndices();
            return current_idx;
        }

        fn incrementIndices(self: *Self) void {
            var dim: usize = shape_arr.len - 1;
            while (true) {
                self.current_indices[dim] += 1;
                if (self.current_indices[dim] < shape_arr[dim]) {
                    return;
                }

                self.current_indices[dim] = 0;
                if (dim == 0) {
                    self.finished = true;
                    return;
                }
                dim -= 1;
            }
        }
    };
}

pub fn DataIterator(comptime TensorType: type) type {
    const shape_arr = utils.getComptimeFieldValue(TensorType, "shape").?;
    const strides_arr = utils.getComptimeFieldValue(TensorType, "strides").?;
    const ScalarType = utils.getComptimeFieldValue(TensorType, "ScalarType").?;
    const readonly = utils.getComptimeFieldValue(TensorType, "is_readonly").?;

    const TensorPtrType = if (comptime readonly)
        *const TensorType
    else
        *TensorType;
    return struct {
        const Self = @This();

        tensor: TensorPtrType,
        current_indices: @TypeOf(shape_arr),
        finished: bool,

        pub fn init(tensor: TensorPtrType) Self {
            return Self{
                .tensor = tensor,
                .current_indices = @as(@Vector(shape_arr.len, usize), @splat(0)),
                .finished = false,
            };
        }

        pub fn next(self: *Self) ?ScalarType {
            if (self.finished) return null;

            const current_idx = self.current_indices;
            self.incrementIndices();
            const data_idx = utils.getIndexAt(current_idx, strides_arr);
            return if (comptime readonly)
                self.tensor.data[data_idx]
            else
                &self.tensor.data[data_idx];
        }

        fn incrementIndices(self: *Self) void {
            var dim: usize = shape_arr.len - 1;
            while (true) {
                self.current_indices[dim] += 1;
                if (self.current_indices[dim] < shape_arr[dim]) {
                    return;
                }

                self.current_indices[dim] = 0;
                if (dim == 0) {
                    self.finished = true;
                    return;
                }
                dim -= 1;
            }
        }
    };
}
