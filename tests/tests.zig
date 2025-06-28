const std = @import("std");
const tensor_1d = @import("1d/tests.zig");
const tensor_2d = @import("2d/tests.zig");
const tensor_3d = @import("3d/tests.zig");
const views_and_refs_1d = @import("1d/views_and_refs.zig");
const views_and_refs_2d = @import("2d/views_and_refs.zig");
const views_and_refs_3d = @import("3d/views_and_refs.zig");

test {
    std.testing.refAllDeclsRecursive(@This());
    std.testing.refAllDeclsRecursive(tensor_1d);
    std.testing.refAllDeclsRecursive(tensor_2d);
    std.testing.refAllDeclsRecursive(tensor_3d);
    std.testing.refAllDeclsRecursive(views_and_refs_1d);
    std.testing.refAllDeclsRecursive(views_and_refs_2d);
    std.testing.refAllDeclsRecursive(views_and_refs_3d);
}
