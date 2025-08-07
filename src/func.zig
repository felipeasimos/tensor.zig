const std = @import("std");
const utils = @import("utils.zig");

fn ArgsTuple(comptime T: type, comptime n_args: usize) type {
    const types: [n_args]type = .{T} ** n_args;
    return std.meta.Tuple(&types);
}

pub fn addFactory(comptime T: type, comptime n_args: usize) fn (ArgsTuple(T, n_args)) T {
    return (struct {
        pub fn func(args: ArgsTuple(T, n_args)) T {
            var total: T = 0;
            inline for (0..n_args) |i| {
                const index_as_str = std.fmt.comptimePrint("{}", .{i});
                total += @field(args, index_as_str);
            }
            return total;
        }
    }).func;
}

pub fn subFactory(comptime T: type, comptime n_args: usize) fn (ArgsTuple(T, n_args)) T {
    return (struct {
        pub fn func(args: ArgsTuple(T, n_args)) T {
            var total: T = args[0];
            inline for (1..n_args) |i| {
                const index_as_str = std.fmt.comptimePrint("{}", .{i});
                total -= @field(args, index_as_str);
            }
            return total;
        }
    }).func;
}

pub fn mulFactory(comptime T: type, comptime n_args: usize) fn (ArgsTuple(T, n_args)) T {
    return (struct {
        pub fn func(args: ArgsTuple(T, n_args)) T {
            var total: T = 1;
            inline for (0..n_args) |i| {
                const index_as_str = std.fmt.comptimePrint("{}", .{i});
                total *= @field(args, index_as_str);
            }
            return total;
        }
    }).func;
}

pub fn divFactory(comptime T: type, comptime n_args: usize) fn (ArgsTuple(T, n_args)) T {
    return (struct {
        pub fn func(args: ArgsTuple(T, n_args)) T {
            var total: T = args[0];
            inline for (1..n_args) |i| {
                const index_as_str = std.fmt.comptimePrint("{}", .{i});
                total /= @field(args, index_as_str);
            }
            return total;
        }
    }).func;
}
