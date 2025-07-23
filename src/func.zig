const std = @import("std");

pub fn addFactory(comptime T: type) fn (T, T) T {
    return (struct {
        pub fn func(a: T, b: T) T {
            return a + b;
        }
    }).func;
}

pub fn subFactory(comptime T: type) fn (T, T) T {
    return (struct {
        pub fn func(a: T, b: T) T {
            return a - b;
        }
    }).func;
}

pub fn mulFactory(comptime T: type) fn (T, T) T {
    return (struct {
        pub fn func(a: T, b: T) T {
            return a * b;
        }
    }).func;
}

pub fn divFactory(comptime T: type) fn (T, T) T {
    return (struct {
        pub fn func(a: T, b: T) T {
            return a / b;
        }
    }).func;
}
