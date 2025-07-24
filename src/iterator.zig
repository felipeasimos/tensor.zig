const utils = @import("utils.zig");

pub fn Iterator(comptime TensorType: type) type {
    const shape_arr = utils.getComptimeFieldValue(TensorType, "shape").?;
    const strides_arr = utils.getComptimeFieldValue(TensorType, "strides").?;
    const scalar_type = utils.getComptimeFieldValue(TensorType, "scalar_type").?;
    const readonly = utils.getComptimeFieldValue(TensorType, "is_readonly").?;
    return struct {
        const Self = @This();

        tensor: *const TensorType,
        current_indices: @TypeOf(shape_arr),
        finished: bool,

        pub fn init(tensor: *const TensorType) Self {
            return Self{
                .tensor = tensor,
                .current_indices = @as(@Vector(shape_arr.len, usize), @splat(0)),
                .finished = false,
            };
        }

        pub fn next(self: *Self) ?struct { indices: @TypeOf(shape_arr), value: scalar_type } {
            if (self.finished) return null;

            const current_idx = self.current_indices;
            const data_idx = utils.getIndexAt(current_idx, strides_arr);
            const value = if (comptime readonly)
                self.tensor.data[data_idx]
            else
                &self.tensor.data[data_idx];

            self.incrementIndices();

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
