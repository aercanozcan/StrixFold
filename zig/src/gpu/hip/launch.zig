//! Kernel launches: argument values packed where hipModuleLaunchKernel reads them (the CUDA runtime's Args and Config,
//! so the family's launch code is unchanged). CUDA's programmatic dependent launch, clusters and cooperative launches
//! have no counterpart here and are refused, never dropped: plain stream order is the stronger guarantee, and a family
//! asks for them only when `caps.pdl` / `caps.clusters` say they exist.

const std = @import("std");
const abi = @import("abi.zig");
const Function = @import("module.zig").Function;
const Stream = @import("stream.zig").Stream;
const Error = @import("driver.zig").Error;

pub const Dim3 = abi.Dim3;

/// Argument values in kernel order; each must have exactly the C type of its kernel parameter.
pub const Args = struct {
    pub const max_args = 48;
    pub const max_bytes = 2048;

    storage: [max_bytes]u8 align(16) = undefined,
    offsets: [max_args]u16 = undefined,
    sizes: [max_args]u16 = undefined,
    ptrs: [max_args]?*anyopaque = undefined,
    used: usize = 0,
    count: usize = 0,

    /// Copies `value` (a pointer-sized address, scalar, or extern struct passed by value) at its C alignment.
    pub fn add(self: *Args, value: anytype) void {
        const T = @TypeOf(value);
        comptime std.debug.assert(@sizeOf(T) > 0 and @alignOf(T) <= 16);
        const at = std.mem.alignForward(usize, self.used, @alignOf(T));
        std.debug.assert(self.count < max_args and at + @sizeOf(T) <= max_bytes);
        @memcpy(self.storage[at..][0..@sizeOf(T)], std.mem.asBytes(&value));
        self.offsets[self.count] = @intCast(at);
        self.sizes[self.count] = @sizeOf(T);
        self.count += 1;
        self.used = at + @sizeOf(T);
    }

    /// Argument `i` read back as an unsigned integer (4- and 8-byte values only).
    pub fn integer(self: *const Args, i: usize) ?u64 {
        if (i >= self.count) return null;
        const at = self.storage[self.offsets[i]..];
        return switch (self.sizes[i]) {
            4 => std.mem.readInt(u32, at[0..4], .little),
            8 => std.mem.readInt(u64, at[0..8], .little),
            else => null,
        };
    }

    /// The address array, rebuilt here so a moved Args never hands out stale addresses.
    pub fn pointers(self: *Args) ?[*]?*anyopaque {
        if (self.count == 0) return null;
        for (self.offsets[0..self.count], 0..) |off, i| self.ptrs[i] = &self.storage[off];
        return &self.ptrs;
    }
};

pub const Config = struct {
    grid: Dim3,
    block: Dim3,
    shared: u32 = 0,
    cluster: ?Dim3 = null,
    cluster_spread: bool = false,
    pdl: bool = false,
    cooperative: bool = false,

    pub fn validate(self: Config) Error!void {
        const g = self.grid;
        const b = self.block;
        if (g.x == 0 or g.y == 0 or g.z == 0 or b.x == 0 or b.y == 0 or b.z == 0) return error.Invalid;
        if (@as(u64, b.x) * b.y * b.z > 1024) return error.Invalid;
        // HIP's total-threads limit per dimension (grid * block fits in 32 bits)
        if (@as(u64, g.x) * b.x > std.math.maxInt(u32) or @as(u64, g.y) * b.y > std.math.maxInt(u32) or @as(u64, g.z) * b.z > std.math.maxInt(u32)) return error.Invalid;
        if (self.shared > 65536) return error.Invalid;
    }

    /// The CUDA-only launch attributes: refused on HIP rather than dropped.
    pub fn plain(self: Config) bool {
        return self.cluster == null and !self.pdl and !self.cooperative;
    }
};

/// Kernel launches so far on this thread (a profile's count by phase; the vision encoder's thread keeps its own).
pub threadlocal var count: u64 = 0;

/// A check run before every launch when set (the kit's strict mode: arguments against the kernel's metadata).
pub var before: ?*const fn (f: Function, cfg: Config, args: *const Args) Error!void = null;

pub fn launch(f: Function, cfg: Config, stream: Stream, args: *Args) Error!void {
    try cfg.validate();
    if (!cfg.plain()) return error.Unsupported;
    if (before) |hook| try hook(f, cfg, args);
    count += 1;
    const g = cfg.grid;
    const b = cfg.block;
    try f.d.check(f.d.api.hipModuleLaunchKernel(f.handle, g.x, g.y, g.z, b.x, b.y, b.z, cfg.shared, stream.handle, args.pointers(), null), "hipModuleLaunchKernel");
}

test "args keep C alignment and sizes" {
    var a: Args = .{};
    a.add(@as(u64, 0x1122334455667788));
    a.add(@as(i32, -3));
    a.add(@as(bool, true));
    a.add(@as(f64, 2.5));
    const S = extern struct { p: u64, n: c_int, s: f32 };
    a.add(S{ .p = 7, .n = 8, .s = 9 });
    try std.testing.expectEqualSlices(u16, &.{ 0, 8, 12, 16, 24 }, a.offsets[0..a.count]);
    try std.testing.expectEqualSlices(u16, &.{ 8, 4, 1, 8, 16 }, a.sizes[0..a.count]);
    const ptrs = a.pointers().?;
    try std.testing.expectEqual(@as(i32, -3), @as(*align(1) const i32, @ptrCast(ptrs[1].?)).*);
    try std.testing.expectEqual(@as(f64, 2.5), @as(*align(1) const f64, @ptrCast(ptrs[3].?)).*);
    var moved = a;
    const p2 = moved.pointers().?;
    try std.testing.expectEqual(@intFromPtr(&moved.storage[8]), @intFromPtr(p2[1].?));
}

test "CUDA-only launch attributes and bad geometry are refused before HIP is called" {
    var d: @import("driver.zig").Driver = undefined;
    const f: Function = .{ .d = &d, .handle = @ptrFromInt(64) };
    const s: Stream = .{ .d = &d, .handle = @ptrFromInt(32) };
    var args: Args = .{};
    try std.testing.expectError(error.Unsupported, launch(f, .{ .grid = .{ .x = 1 }, .block = .{ .x = 32 }, .pdl = true }, s, &args));
    try std.testing.expectError(error.Unsupported, launch(f, .{ .grid = .{ .x = 1 }, .block = .{ .x = 32 }, .cluster = .{ .x = 2 } }, s, &args));
    try std.testing.expectError(error.Unsupported, launch(f, .{ .grid = .{ .x = 1 }, .block = .{ .x = 32 }, .cooperative = true }, s, &args));
    try std.testing.expectError(error.Invalid, launch(f, .{ .grid = .{ .x = 0 }, .block = .{ .x = 32 } }, s, &args));
    try std.testing.expectError(error.Invalid, launch(f, .{ .grid = .{ .x = 1 }, .block = .{ .x = 1024, .y = 2 } }, s, &args));
    try std.testing.expectError(error.Invalid, launch(f, .{ .grid = .{ .x = 1 }, .block = .{ .x = 32 }, .shared = 65537 }, s, &args));
}
