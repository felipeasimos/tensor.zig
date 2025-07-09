const std = @import("std");
const tensor_1d = @import("1d/tests.zig");
const tensor_2d = @import("2d/tests.zig");
const tensor_3d = @import("3d/tests.zig");
const views_1d = @import("1d/views.zig");
const views_2d = @import("2d/views.zig");
const views_3d = @import("3d/views.zig");
const refs_1d = @import("1d/refs.zig");
const refs_2d = @import("2d/refs.zig");
const refs_3d = @import("3d/refs.zig");

test {
    std.testing.refAllDeclsRecursive(@This());
    std.testing.refAllDeclsRecursive(tensor_1d);
    std.testing.refAllDeclsRecursive(tensor_2d);
    std.testing.refAllDeclsRecursive(tensor_3d);
    std.testing.refAllDeclsRecursive(views_1d);
    std.testing.refAllDeclsRecursive(views_2d);
    std.testing.refAllDeclsRecursive(views_3d);
    std.testing.refAllDeclsRecursive(refs_1d);
    std.testing.refAllDeclsRecursive(refs_2d);
    std.testing.refAllDeclsRecursive(refs_3d);
}
