const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const lib = b.addModule("tensor", .{
        .root_source_file = b.path("src/tensor.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{},
    });

    const test_mod = b.createModule(.{
        .root_source_file = b.path("tests/tests.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{
            .{ .name = "tensor", .module = lib },
        },
    });
    // Creates a step for unit testing. This only builds the test executable
    // but does not run it.
    const lib_unit_tests = b.addTest(.{
        .root_module = test_mod,
    });

    const run_lib_unit_tests = b.addRunArtifact(lib_unit_tests);

    const test_step = b.step("test", "Run unit tests");
    test_step.dependOn(&run_lib_unit_tests.step);

    const check_lib_tests = b.addTest(.{
        .root_module = test_mod,
    });
    const check = b.step("check", "Check if the library compiles");
    check.dependOn(&check_lib_tests.step);
}
