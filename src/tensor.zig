const std = @import("std");

pub fn TensorView(comptime dtype: type, comptime _shape: anytype) type {
    const dtype_info = @typeInfo(dtype);
    const shape_vec = asVector(dtype, _shape);
    const total_num_scalars = @reduce(.Mul, shape_vec);
    if (dtype_info != .float and dtype_info != .int) {
        @compileError("Only floats and integers are valid tensor dtypes");
    }
    const _strides = calculateStrides(dtype, _shape);
    return struct {
        comptime shape: @Vector(_shape.len, usize) = shape_vec,
        comptime strides: @Vector(_shape.len, usize) = _strides,

        data: []dtype,

        pub fn init(data: []dtype) @This() {
            return .{
                .data = data[0..total_num_scalars],
            };
        }
        pub fn randomize(self: *@This(), random: std.Random) void {
            random.bytes(std.mem.asBytes(&self.data));
        }

        pub fn scalar_mut(self: *@This(), idxs: @Vector(_shape.len, usize)) *dtype {
            const idx = @reduce(.Add, self.strides * idxs);
            return &self.data[idx];
        }

        pub fn scalar(self: *@This(), idxs: @Vector(_shape.len, usize)) dtype {
            return self.scalar_mut(idxs).*;
        }

        fn SubTensorView(comptime size: usize) type {
            return TensorView(dtype, toArray(_shape)[size..]);
        }

        fn GetSubTensorViewResult(comptime size: usize) type {
            if (_shape.len - size == 0) {
                return *dtype;
            }
            return SubTensorView(size);
        }

        /// get a subtensor. `idxs` needs to be an array.
        pub fn get(self: *@This(), idxs: anytype) GetSubTensorViewResult(idxs.len) {
            if (comptime _shape.len - idxs.len == 0) {
                return self.scalar_mut(idxs);
            }
            const to_sub_tensor_mask = comptime createSequence(usize, idxs.len);
            const strides_to_sub_tensor = comptime @shuffle(
                usize,
                self.strides,
                undefined,
                to_sub_tensor_mask,
            );
            const start_idx = @reduce(.Add, strides_to_sub_tensor * idxs);
            const final_idx = start_idx + strides_to_sub_tensor[strides_to_sub_tensor.len - 1];
            return SubTensorView(idxs.len).init(self.data[start_idx..final_idx]);
        }
    };
}

pub fn createSequence(comptime dtype: type, comptime n: usize) [n]dtype {
    var seq: [n]dtype = .{1} ** n;
    inline for (0..n) |i| {
        seq[i] = i;
    }
    return seq;
}

fn toArray(tuple: anytype) [tuple.len]@FieldType(@TypeOf(tuple), "0") {
    const TupleType = @TypeOf(tuple);
    const field_count = if (comptime @hasDecl(TupleType, "len")) tuple.len else std.meta.fields(TupleType).len;
    const T = comptime @FieldType(@TypeOf(tuple), "0");

    var array: [field_count]T = undefined;
    for (0..field_count) |i| {
        @compileLog(@TypeOf(tuple[i]));
        array[i] = tuple[i];
    }
    return array;
}

fn asVector(comptime dtype: type, seq: anytype) @Vector(seq.len, dtype) {
    var vec: @Vector(seq.len, dtype) = undefined;
    for (seq, 0..) |a, i| {
        vec[i] = a;
    }
    return vec;
}

fn calculateStrides(comptime dtype: type, comptime shape: anytype) @Vector(shape.len, dtype) {
    var strides: [shape.len]usize = .{1} ** shape.len;
    for (1..shape.len) |i| {
        const idx = shape.len - i - 1;
        strides[idx] = shape[idx + 1] * strides[idx + 1];
    }
    return strides;
}
