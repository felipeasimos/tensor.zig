test "all" {
    _ = @import("std");
    _ = @import("1d/tests.zig");
    _ = @import("2d/tests.zig");
    _ = @import("3d/tests.zig");
    _ = @import("1d/views.zig");
    _ = @import("2d/views.zig");
    _ = @import("3d/views.zig");
    _ = @import("1d/refs.zig");
    _ = @import("2d/refs.zig");
    _ = @import("3d/refs.zig");
}
