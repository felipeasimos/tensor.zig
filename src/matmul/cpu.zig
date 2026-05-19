//! Macrokernel:
//! - distributed among threads
//! - own one C tile
//! - receives references to strides of A and B associated with C tile
//! - packs A and B strides in proper majors for microkernel
//! Microkernel:
//! - partially compute C tile through FMA
//! - receives packed A and B in column and row major respectively
const std = @import("std");
const utils = @import("../utils.zig");
const tensor = @import("../tensor.zig");

fn calculateBlockside(comptime T: type) usize {
    const target_cache_size_in_bytes = 128 * 1024;
    const side_of_block_in_bytes = std.math.sqrt(target_cache_size_in_bytes);
    return side_of_block_in_bytes / @sizeOf(T);
}

fn WorkerRowMajor(comptime T: type, comptime n_accumulators: usize) type {
    return struct {
        pub const VecLen = std.simd.suggestVectorLength(T).?;
        pub const blockside = calculateBlockside(T);

        /// operate on a single C tile, using its associated A and B strides
        fn macrokernelFull(c: *tensor.Tensor(T, 2), a: tensor.Tensor(T, 2), b: tensor.Tensor(T, 2)) void {
            var i: usize = 0;
            while (i < a.metadata.shape[0]) : (i += n_accumulators) {
                const a_microstride = a.slice(.{
                    .{ i, i + n_accumulators },
                    .{ 0, a.metadata.shape[1] },
                });
                var j: usize = 0;
                while (j < b.metadata.shape[1]) : (j += VecLen) {
                    var c_microtile = c.slice(.{
                        .{ i, i + n_accumulators },
                        .{ j, j + VecLen },
                    });
                    const b_microstride = b.slice(.{
                        .{ 0, b.metadata.shape[0] },
                        .{ j, j + VecLen },
                    });
                    microkernelFull(
                        &c_microtile,
                        a_microstride,
                        b_microstride,
                    );
                }
            }
        }

        /// operate on a single C tile, using its associated A and B strides
        fn macrokernelPartial(c: *tensor.Tensor(T, 2), a: tensor.Tensor(T, 2), b: tensor.Tensor(T, 2)) void {
            var i: usize = 0;
            while (i < a.metadata.shape[0]) : (i += n_accumulators) {
                const I = @min(i + n_accumulators, a.metadata.shape[0]);
                const a_microstride = a.slice(.{
                    .{ i, I },
                    .{ 0, a.metadata.shape[1] },
                });
                var j: usize = 0;
                while (j < b.metadata.shape[1]) : (j += VecLen) {
                    const J = @min(j + VecLen, b.metadata.shape[1]);
                    var c_microtile = c.slice(.{
                        .{ i, I },
                        .{ j, J },
                    });
                    const b_microstride = b.slice(.{
                        .{ 0, b.metadata.shape[0] },
                        .{ j, J },
                    });
                    microkernelPartial(
                        &c_microtile,
                        a_microstride,
                        b_microstride,
                    );
                }
            }
        }

        pub fn macrokernel(allocator: std.mem.Allocator, c: tensor.Tensor(T, 2), unpacked_a: tensor.Tensor(T, 2), unpacked_b: tensor.Tensor(T, 2)) void {
            const row_remainder = c.metadata.shape[0] % n_accumulators;
            const column_remainder = c.metadata.shape[1] % VecLen;

            var a = unpacked_a.pack(allocator, .ColumnMajor) catch @panic("Couldn't allocate memory for packed A");
            defer a.deinit(allocator);
            var b = unpacked_b.pack(allocator, .RowMajor) catch @panic("Couldn't allocate memory for packed B");
            defer b.deinit(allocator);

            // full part
            if (c.metadata.shape[0] > n_accumulators and c.metadata.shape[1] > VecLen) {
                var c_slice = c.slice(.{
                    .{ 0, c.metadata.shape[0] - row_remainder },
                    .{ 0, c.metadata.shape[1] - column_remainder },
                });
                const a_slice = a.slice(.{
                    .{ 0, c.metadata.shape[0] - row_remainder },
                    .{ 0, a.metadata.shape[1] },
                });
                const b_slice = b.slice(.{
                    .{ 0, b.metadata.shape[0] },
                    .{ 0, b.metadata.shape[1] - column_remainder },
                });
                macrokernelFull(&c_slice, a_slice, b_slice);
            }
            if (row_remainder > 0 or column_remainder > 0) {
                const first_row_index = if (c.metadata.shape[0] <= n_accumulators)
                    0
                else
                    c.metadata.shape[0] - row_remainder;
                const first_column_index = if (c.metadata.shape[1] <= VecLen)
                    0
                else
                    c.metadata.shape[1] - column_remainder;
                var c_slice = c.slice(.{
                    .{ first_row_index, c.metadata.shape[0] },
                    .{ first_column_index, c.metadata.shape[1] },
                });
                const a_slice = a.slice(.{
                    .{ first_row_index, c.metadata.shape[0] },
                    .{ 0, a.metadata.shape[1] },
                });
                const b_slice = b.slice(.{
                    .{ 0, b.metadata.shape[0] },
                    .{ first_column_index, b.metadata.shape[1] },
                });
                macrokernelPartial(
                    &c_slice,
                    a_slice,
                    b_slice,
                );
            }
        }
        /// no edge cases, full SIMD
        fn microkernelFull(c: *tensor.Tensor(T, 2), a: tensor.Tensor(T, 2), b: tensor.Tensor(T, 2)) void {
            var accs: [n_accumulators]@Vector(VecLen, T) = .{.{0} ** VecLen} ** n_accumulators;

            var k: usize = 0;
            while (k < a.metadata.shape[1]) : (k += 1) {
                inline for (0..n_accumulators) |acc_idx| {
                    const a_splat: @Vector(VecLen, T) = @splat(a.scalar(.{ acc_idx, k }));
                    const b_row: @Vector(VecLen, T) = b.constRef(.{k}).asVector(VecLen);

                    accs[acc_idx] = switch (@typeInfo(T)) {
                        .float => @mulAdd(@Vector(VecLen, T), a_splat, b_row, accs[acc_idx]),
                        .int => a_splat *% b_row +% accs[acc_idx],
                        inline else => @compileError("matmul not available"),
                    };
                }
            }
            inline for (0..n_accumulators) |acc_idx| {
                var c_ref = c.ref(.{acc_idx});
                var vec: @Vector(VecLen, T) = c_ref.data[0..VecLen].*;
                switch (@typeInfo(T)) {
                    .float => vec += accs[acc_idx],
                    .int => vec +%= accs[acc_idx],
                    inline else => @compileError("matmul not available"),
                }
                c_ref.data[0..VecLen].* = vec;
            }
        }
        /// no edge cases, full SIMD
        fn microkernelPartial(c: *tensor.Tensor(T, 2), a: tensor.Tensor(T, 2), b: tensor.Tensor(T, 2)) void {
            switch (b.metadata.shape[1]) {
                inline 0...VecLen => |N| {
                    var accs: [n_accumulators]@Vector(N, T) = .{.{0} ** N} ** n_accumulators;
                    var k: usize = 0;
                    while (k < a.metadata.shape[1]) : (k += 1) {
                        for (0..a.metadata.shape[0]) |acc_idx| {
                            const a_splat: @Vector(N, T) = @splat(a.scalar(.{ acc_idx, k }));
                            const b_row: @Vector(N, T) = b.constRef(.{k}).asVector(N);

                            accs[acc_idx] = switch (@typeInfo(T)) {
                                .float => @mulAdd(@Vector(N, T), a_splat, b_row, accs[acc_idx]),
                                .int => a_splat *% b_row +% accs[acc_idx],
                                inline else => @compileError("matmul not available"),
                            };
                        }
                    }
                    for (0..a.metadata.shape[0]) |acc_idx| {
                        var c_ref = c.ref(.{acc_idx});
                        var vec: @Vector(N, T) = c_ref.data[0..N].*;
                        switch (@typeInfo(T)) {
                            .float => vec += accs[acc_idx],
                            .int => vec +%= accs[acc_idx],
                            inline else => @compileError("matmul not available"),
                        }
                        c_ref.data[0..N].* = vec;
                    }
                },
                else => @panic("microkernelPartial for row major shouldn't be called when number of columns in B is greater than VecLen"),
            }
        }
        pub inline fn naive(c: *tensor.Tensor(T, 2), a: tensor.Tensor(T, 2), b: tensor.Tensor(T, 2)) void {
            // return CPUGEMM(io, self, a, b);
            // blocked gemm
            // (P, Q) x (Q, R) -> (P, R)
            const P = a.metadata.shape[0];
            const Q = a.metadata.shape[1];
            const R = b.metadata.shape[1];
            std.debug.assert(!(c.metadata.shape[0] != P or c.metadata.shape[1] != R or b.metadata.shape[0] != Q));
            for (0..P) |i| {
                for (0..R) |j| {
                    var tmp: @TypeOf(a.data[0]) = 0;
                    for (0..Q) |k| {
                        const index_self = utils.getIndexAt(.{ i, k }, a.metadata.strides);
                        const index_other = utils.getIndexAt(.{ k, j }, b.metadata.strides);
                        tmp += a.data[index_self] * b.data[index_other];
                    }
                    const index_result = utils.getIndexAt(.{ i, j }, c.metadata.strides);
                    c.data[index_result] = tmp;
                }
            }
        }
    };
}

fn checkDimensions(c: anytype, a: anytype, b: anytype) bool {
    if (c.metadata.shape.len != 2 or a.metadata.shape.len != 2 or b.metadata.shape.len != 2) {
        return false;
    }
    if (c.metadata.shape[0] != a.metadata.shape[0] or
        c.metadata.shape[1] != b.metadata.shape[1] or
        a.metadata.shape[1] != b.metadata.shape[0])
    {
        return false;
    }
    return true;
}

/// TODO: actually pack the results
/// TODO: make it work for column major (WorkerColumnMajor is WIP)
/// C is (i, j) (row-major)
/// A is (i, k) (row-major)
/// B is (k, j) (column-major)
pub fn matmul(allocator: std.mem.Allocator, io: std.Io, c: anytype, a: anytype, b: anytype) !void {
    std.debug.assert(checkDimensions(c, a, b));
    var group = std.Io.Group.init;

    const ni = a.metadata.shape[0];
    const nj = b.metadata.shape[1];

    const major: utils.MemoryLayout = major: {
        const opt = utils.MemoryLayout.detectLayout(c.metadata.strides);
        if (opt) |major| {
            break :major major;
        }
        break :major .RowMajor;
    };
    switch (major) {
        inline else => |M| {
            const Worker = if (M == .RowMajor) WorkerRowMajor else WorkerRowMajor;
            const W = Worker(utils.getChildType(@TypeOf(c)).ScalarType, 4);

            // full macrokernels only
            var bi: usize = 0;
            while (bi < ni) : (bi += W.blockside) {
                const I = @min(W.blockside, a.metadata.shape[0] - bi);
                const a_stride = a.slice(.{
                    .{ bi, bi + I },
                    .{ 0, a.metadata.shape[1] },
                });
                if (I == 0) continue;
                var bj: usize = 0;
                while (bj < nj) : (bj += W.blockside) {
                    const J = @min(W.blockside, b.metadata.shape[1] - bj);
                    if (J == 0) continue;
                    const b_stride = b.slice(.{
                        .{ 0, b.metadata.shape[0] },
                        .{ bj, bj + J },
                    });
                    const c_tile = c.slice(.{
                        .{ bi, bi + I },
                        .{ bj, bj + J },
                    });
                    try group.concurrent(io, W.macrokernel, .{
                        allocator,
                        c_tile,
                        a_stride,
                        b_stride,
                    });
                }
            }
        },
    }

    try group.await(io);
}
