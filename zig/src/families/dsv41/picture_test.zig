//! A picture too thin to keep one row or column once ImageOps.pad fits it is refused, as PIL's resize refuses it.
const std = @import("std");
const picture = @import("picture.zig");

fn prepare(w: usize, h: usize) !picture.Picture {
    const a = std.testing.allocator;
    const px = try a.alloc(u8, w * h * 3);
    defer a.free(px);
    @memset(px, 200);
    return picture.fromRgb(a, .{ .w = w, .h = h, .px = px }, .{});
}

test "the thinnest pictures whose fitted side rounds to 0 are refused, and the thinnest kept ones fill a span" {
    for ([_][2]usize{ .{ 85764, 1 }, .{ 100000, 1 }, .{ 1, 42924 }, .{ 1, 50000 } }) |s| {
        try std.testing.expectError(error.EmptySide, prepare(s[0], s[1]));
    }
    // (w, h) -> best_w, best_h, n_llm_h, n_llm_w
    for ([_][6]usize{ .{ 85763, 1, 42882, 42, 1, 1021 }, .{ 1, 42923, 42, 21462, 511, 1 } }) |c| {
        var p = try prepare(c[0], c[1]);
        defer p.deinit(std.testing.allocator);
        try std.testing.expectEqual(c[2], p.grid.best_w);
        try std.testing.expectEqual(c[3], p.grid.best_h);
        try std.testing.expectEqual(c[4], p.grid.n_llm_h);
        try std.testing.expectEqual(c[5], p.grid.n_llm_w);
        try std.testing.expectEqual(@as(usize, 1024), p.grid.spanTokens());
        try std.testing.expectEqual(p.grid.patches() * 3 * 14 * 14, p.patches.len);
    }
}
