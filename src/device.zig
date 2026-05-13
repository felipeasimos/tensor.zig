const std = @import("std");
const wgpu = @import("wgpu.zig");

pub const DeviceType = enum {
    GPU,
    CPU,
};

pub fn Device(comptime device_type: DeviceType) type {
    return switch (device_type) {
        .CPU => std.Io,
        .GPU => wgpu.WGPU,
    };
}
