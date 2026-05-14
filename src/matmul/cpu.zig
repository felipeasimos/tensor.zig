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

fn calculateBlockside(comptime T: usize) usize {
    const target_cache_size_in_bytes = 128 * 1024;
    const side_of_block_in_bytes = std.math.sqrt(target_cache_size_in_bytes);
    return side_of_block_in_bytes / @sizeOf(T);
}

fn Worker(comptime CType: type, comptime AType: type, comptime BType: type, comptime n_accumulators: usize) type {
    const T = switch (@typeInfo(CType)) {
        .pointer => |p| p.child.Dtype,
        else => CType.Dtype,
    };
    return struct {
        pub const VecLen = std.simd.suggestVectorLength(T);
        pub const blockside = calculateBlockside(T);

        /// operate on a single C tile, using its associated A and B strides
        fn macrokernelFull(c: anytype, a: anytype, b: anytype) void {
            var k: usize = 0;
            inline while (k < a.shape[1]) : (k += blockside) {
                var i: usize = 0;
                inline while (i < a.shape[0]) : (i += n_accumulators) {
                    var j: usize = 0;
                    inline while (j < b.shape[1]) : (j += VecLen) {
                        microkernelFull(
                            c.slice(.{
                                .{ i, i + n_accumulators },
                                .{ j, j + VecLen },
                            }),
                            a.slice(.{
                                .{ i, i + n_accumulators },
                                .{ 0, a.shape[1] },
                            }),
                            b.slice(.{
                                .{ 0, b.shape[0] },
                                .{ j, j + VecLen },
                            }),
                        );
                    }
                }
            }
        }
        pub fn macrokernel(c: CType, a: AType, b: BType) void {
            if (c.shape[0] > n_accumulators and c.shape[1] > VecLen) {
                const row_remainder = c.shape[0] % n_accumulators;
                const column_remainder = c.shape[1] % VecLen;
                macrokernelFull(
                    c.slice(.{
                        .{ 0, c.shape[0] - row_remainder },
                        .{ 0, c.shape[1] - column_remainder },
                    }),
                    a.slice(.{
                        .{ 0, c.shape[0] - row_remainder },
                        .{ 0, a.shape[1] },
                    }),
                    b.slice(.{
                        .{ 0, b.shape[0] },
                        .{ 0, b.shape[1] - column_remainder },
                    }),
                );
            }
        }
        /// no edge cases, full SIMD
        fn microkernelFull(c: anytype, a: anytype, b: anytype) void {
            var accs: [n_accumulators]@Vector(VecLen, T) = .{.{0} ** VecLen} ** n_accumulators;

            var k: usize = 0;
            inline while (k < a.shape[1]) : (k += 1) {
                inline for (0..n_accumulators) |acc_idx| {
                    const a_splat: @Vector(VecLen, T) = @splat(a.scalar(.{ acc_idx, k }));
                    const b_row: @Vector(VecLen, T) = b.ref(.{k}).asVector();

                    accs[acc_idx] = switch (@typeInfo(T)) {
                        .float => @mulAdd(@Vector(VecLen, T), a_splat, b_row, accs[acc_idx]),
                        .int => a_splat *% b_row +% accs[acc_idx],
                        inline else => @compileError("matmul not available"),
                    };
                }
            }
            inline for (0..n_accumulators) |acc_idx| {
                switch (@typeInfo(T)) {
                    .float => c.ref(.{acc_idx}).data[0..VecLen].* += accs[acc_idx],
                    .int => c.ref(.{acc_idx}).data[0..VecLen].* +%= accs[acc_idx],
                    inline else => @compileError("matmul not available"),
                }
            }
        }
    };
}

fn checkDimensions(c: anytype, a: anytype, b: anytype) void {
    if (c.shape.len != 2 or a.shape.len != 2 or b.shape.len != 2) {
        @compileError("matmul only available for 2D tensors");
    }
    if (c.shape[0] != a.shape[0] or
        c.shape[1] != b.shape[1] or
        a.shape[1] != b.shape[0])
    {
        @compileError("Dimensions don't match for matmul operation");
    }
}
/// TODO: un-unroll loops
/// TODO: column major friendly
/// C is (i, j) (row-major)
/// A is (i, k) (row-major)
/// B is (k, j) (column-major)
pub fn matmul(io: std.Io, c: anytype, a: anytype, b: anytype) !void {
    checkDimensions(c, a, b);
    var group = std.Io.Group.init;

    const ni = comptime a.shape[0];
    const nj = comptime b.shape[1];

    const blockside = calculateBlockside(c.dtype);
    const W = Worker(@TypeOf(c), @TypeOf(a), @TypeOf(b), 4);

    // full macrokernels only
    var bi: usize = 0;
    while (bi < ni) : (bi += blockside) {
        const I = @min(blockside, a.shape[0] - bi);
        const a_stride = a.slice(.{
            .{ bi, bi + I },
            .{ 0, a.shape[1] },
        });
        var bj: usize = 0;
        while (bj < nj) : (bj += blockside) {
            const J = @min(blockside, b.shape[1] - bj);
            const b_stride = b.slice(.{
                .{ 0, b.shape[0] },
                .{ bj, bj + J },
            });
            const c_tile = c.slice(.{
                .{ bi, bi + I },
                .{ bj, bj + J },
            });
            try group.concurrent(io, W.macrokernel, .{
                c_tile,
                a_stride,
                b_stride,
            });
        }
    }

    try group.await(io);
}
