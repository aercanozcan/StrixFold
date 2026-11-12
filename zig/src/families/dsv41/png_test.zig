//! A PNG's inflated data is bounded by the rows its header declares: longer valid data is not inflated past them.
const std = @import("std");
const picture = @import("picture.zig");

/// Adam7's passes as (x0, y0, dx, dy).
const adam7: [7][4]usize = .{ .{ 0, 0, 8, 8 }, .{ 4, 0, 8, 8 }, .{ 0, 4, 4, 8 }, .{ 2, 0, 4, 4 }, .{ 0, 2, 2, 4 }, .{ 1, 0, 2, 2 }, .{ 0, 1, 1, 2 } };

fn chunk(a: std.mem.Allocator, out: *std.ArrayList(u8), kind: *const [4]u8, body: []const u8) !void {
    var len: [4]u8 = undefined;
    std.mem.writeInt(u32, &len, @intCast(body.len), .big);
    try out.appendSlice(a, &len);
    try out.appendSlice(a, kind);
    try out.appendSlice(a, body);
    var hc = std.hash.Crc32.init();
    hc.update(kind);
    hc.update(body);
    std.mem.writeInt(u32, &len, hc.final(), .big);
    try out.appendSlice(a, &len);
}

/// A PNG of w x h at this depth and colour type whose one IDAT is `zlib` (a PLTE first when `plte` has entries).
fn png(a: std.mem.Allocator, w: u32, h: u32, depth: u8, ctype: u8, interlace: u8, plte: []const u8, zlib: []const u8) ![]u8 {
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(a);
    try out.appendSlice(a, "\x89PNG\r\n\x1a\n");
    var ihdr: [13]u8 = .{ 0, 0, 0, 0, 0, 0, 0, 0, depth, ctype, 0, 0, interlace };
    std.mem.writeInt(u32, ihdr[0..4], w, .big);
    std.mem.writeInt(u32, ihdr[4..8], h, .big);
    try chunk(a, &out, "IHDR", &ihdr);
    if (plte.len > 0) try chunk(a, &out, "PLTE", plte);
    try chunk(a, &out, "IDAT", zlib);
    try chunk(a, &out, "IEND", "");
    return out.toOwnedSlice(a);
}

/// The rows (filter 0) of samples [h][w][channels] at `depth` bits, pass by pass when interlaced.
fn rows(a: std.mem.Allocator, w: usize, h: usize, interlace: u8, depth: u5, channels: usize, s: []const u16) ![]u8 {
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(a);
    const one: [1][4]usize = .{.{ 0, 0, 1, 1 }};
    for (if (interlace == 1) adam7[0..] else one[0..]) |ps| {
        var y = ps[1];
        while (y < h and ps[0] < w) : (y += ps[3]) {
            try out.append(a, 0);
            var acc: u32 = 0;
            var n: u5 = 0;
            var x = ps[0];
            while (x < w) : (x += ps[2]) {
                for (s[(y * w + x) * channels ..][0..channels]) |v| {
                    if (depth == 16) {
                        try out.appendSlice(a, &[2]u8{ @intCast(v >> 8), @truncate(v) });
                        continue;
                    }
                    acc = acc << depth | v;
                    n += depth;
                    if (n == 8) {
                        try out.append(a, @intCast(acc));
                        acc = 0;
                        n = 0;
                    }
                }
            }
            if (n > 0) try out.append(a, @intCast(acc << @intCast(8 - n)));
        }
    }
    return out.toOwnedSlice(a);
}

/// `raw` as a zlib stream of one stored block.
fn stored(a: std.mem.Allocator, raw: []const u8) ![]u8 {
    const z = try a.alloc(u8, raw.len + 11);
    @memcpy(z[0..3], &[_]u8{ 0x78, 0x01, 0x01 });
    std.mem.writeInt(u16, z[3..5], @intCast(raw.len), .little);
    std.mem.writeInt(u16, z[5..7], ~@as(u16, @intCast(raw.len)), .little);
    @memcpy(z[7..][0..raw.len], raw);
    std.mem.writeInt(u32, z[z.len - 4 ..][0..4], std.hash.Adler32.hash(raw), .big);
    return z;
}

/// Bits in deflate's order: fields least significant bit first, Huffman codes most significant bit first.
const Bits = struct {
    out: std.ArrayList(u8) = .empty,
    acc: u64 = 0,
    n: u6 = 0,

    fn put(b: *Bits, a: std.mem.Allocator, v: u64, n: u6) !void {
        b.acc |= v << b.n;
        b.n += n;
        while (b.n >= 8) : (b.n -= 8) {
            try b.out.append(a, @truncate(b.acc));
            b.acc >>= 8;
        }
    }

    fn code(b: *Bits, a: std.mem.Allocator, c: u16, n: u4) !void {
        try b.put(a, @bitReverse(c) >> @intCast(16 - @as(u5, n)), n);
    }

    /// A zlib header and a final fixed-Huffman block's header.
    fn start(a: std.mem.Allocator) !Bits {
        var b: Bits = .{};
        try b.out.appendSlice(a, &[_]u8{ 0x78, 0x01 });
        try b.put(a, 0b011, 3);
        return b;
    }

    /// The last byte, then `sum` as zlib's Adler-32 unless null.
    fn finish(b: *Bits, a: std.mem.Allocator, sum: ?u32) ![]u8 {
        if (b.n > 0) try b.out.append(a, @truncate(b.acc));
        var be: [4]u8 = undefined;
        std.mem.writeInt(u32, &be, sum orelse 0, .big);
        if (sum != null) try b.out.appendSlice(a, &be);
        return b.out.toOwnedSlice(a);
    }
};

/// A zlib stream of 1 + 258 * matches zero bytes: a fixed-Huffman literal 0, then matches of 258 at distance 1.
fn zeros(a: std.mem.Allocator, matches: usize) ![]u8 {
    var b = try Bits.start(a);
    errdefer b.out.deinit(a);
    try b.code(a, 0x30, 8);
    for (0..matches) |_| {
        try b.code(a, 0xc5, 8);
        try b.put(a, 0, 5);
    }
    try b.code(a, 0, 7);
    var adler: std.hash.Adler32 = .{};
    const block: [258]u8 = @splat(0);
    adler.update(block[0..1]);
    for (0..matches) |_| adler.update(&block);
    return b.finish(a, adler.adler);
}

/// `raw` as fixed-Huffman literals, then the block's end, or length code 286 (invalid in deflate) when broken.
fn literals(a: std.mem.Allocator, raw: []const u8, broken: bool) ![]u8 {
    var b = try Bits.start(a);
    errdefer b.out.deinit(a);
    for (raw) |v| try if (v < 144) b.code(a, 0x30 + @as(u16, v), 8) else b.code(a, 0x190 + @as(u16, v - 144), 9);
    if (broken) {
        try b.code(a, 0xc6, 8);
        return b.finish(a, null);
    }
    try b.code(a, 0, 7);
    return b.finish(a, std.hash.Adler32.hash(raw));
}

/// The rows of a black 1x1 RGB PNG (a filter byte, then 3 zeros) as a literal 0 and a match of 3 at distance code `d`.
fn match(a: std.mem.Allocator, d: u16) ![]u8 {
    var b = try Bits.start(a);
    errdefer b.out.deinit(a);
    try b.code(a, 0x30, 8);
    try b.code(a, 1, 7);
    try b.code(a, d, 5);
    try b.code(a, 0, 7);
    return b.finish(a, std.hash.Adler32.hash(&.{ 0, 0, 0, 0 }));
}

/// Every byte `z` inflates to, or its error.
fn inflated(a: std.mem.Allocator, z: []const u8) !usize {
    const window = try a.alloc(u8, std.compress.flate.max_window_len);
    defer a.free(window);
    var in = std.Io.Reader.fixed(z);
    var d: std.compress.flate.Decompress = .init(&in, .zlib, window);
    return d.reader.discardRemaining();
}

test "a 1x1 PNG whose IDAT inflates to 16 MiB decodes from its first 4 bytes without holding the rest" {
    const a = std.testing.allocator;
    const z = try zeros(a, 65025);
    defer a.free(z);
    try std.testing.expectEqual(@as(usize, 1 + 258 * 65025), try inflated(a, z));
    const data = try png(a, 1, 1, 8, 2, 0, "", z);
    defer a.free(data);
    var counted = std.testing.FailingAllocator.init(a, .{});
    var rgb = try picture.decodePng(counted.allocator(), data);
    defer rgb.deinit(counted.allocator());
    try std.testing.expectEqualSlices(u8, &.{ 0, 0, 0 }, rgb.px);
    try std.testing.expect(counted.allocated_bytes < 1 << 20);
}

test "a PNG decodes from its rows and valid data past them; a stream short, or broken just after the rows, is refused" {
    const a = std.testing.allocator;
    const w = 5;
    const h = 3;
    var s: [w * h * 3]u16 = undefined;
    var want: [w * h * 3]u8 = undefined;
    for (&s, &want, 0..) |*v, *b, i| {
        b.* = @intCast((i * 37 + 11) % 256);
        v.* = b.*;
    }
    for ([_]u8{ 0, 1 }) |interlace| {
        const exact = try rows(a, w, h, interlace, 8, 3, &s);
        defer a.free(exact);
        const more = try std.mem.concat(a, u8, &.{ exact, "\x07" });
        defer a.free(more);
        // decoded: the rows, a byte more stored, a byte more as literals; refused: a bad code after the rows, one short
        const streams = [_][]u8{
            try stored(a, exact),
            try stored(a, more),
            try literals(a, more, false),
            try literals(a, exact, true),
            try stored(a, exact[0 .. exact.len - 1]),
        };
        defer for (streams) |z| a.free(z);
        try std.testing.expectEqual(more.len, try inflated(a, streams[2]));
        try std.testing.expectError(error.ReadFailed, inflated(a, streams[3]));
        for (streams, 0..) |z, i| {
            const data = try png(a, w, h, 8, 2, interlace, "", z);
            defer a.free(data);
            if (i >= 3) {
                try std.testing.expectError(error.BadPng, picture.decodePng(a, data));
                continue;
            }
            var rgb = try picture.decodePng(a, data);
            defer rgb.deinit(a);
            try std.testing.expectEqualSlices(u8, &want, rgb.px);
        }
    }
}

test "a back-reference past the bytes written is refused; the same stream at distance 1 decodes" {
    const a = std.testing.allocator;
    for ([_]u16{ 0, 1 }) |d| {
        const z = try match(a, d);
        defer a.free(z);
        const data = try png(a, 1, 1, 8, 2, 0, "", z);
        defer a.free(data);
        if (d == 1) {
            try std.testing.expectError(error.ReadFailed, inflated(a, z));
            try std.testing.expectError(error.BadPng, picture.decodePng(a, data));
            continue;
        }
        try std.testing.expectEqual(@as(usize, 4), try inflated(a, z));
        var rgb = try picture.decodePng(a, data);
        defer rgb.deinit(a);
        try std.testing.expectEqualSlices(u8, &.{ 0, 0, 0 }, rgb.px);
    }
}

test "1-bit grey, 4-bit palette and 16-bit RGB PNGs decode from exactly their rows, plain and interlaced" {
    const a = std.testing.allocator;
    const w = 13;
    const h = 5;
    var plte: [48]u8 = undefined;
    for (&plte, 0..) |*b, i| b.* = @intCast((i * 53 + 7) % 256);
    // depth, colour type, channels
    for ([_][3]u8{ .{ 1, 0, 1 }, .{ 4, 3, 1 }, .{ 16, 2, 3 } }) |c| {
        const ch = c[2];
        var s: [w * h * 3]u16 = undefined;
        for (s[0 .. w * h * ch], 0..) |*v, i| v.* = @intCast((i * 7919 + 13) % (@as(usize, 1) << @intCast(c[0])));
        var want: [w * h * 3]u8 = undefined;
        for (0..w * h) |i| want[i * 3 ..][0..3].* = switch (c[1]) {
            0 => if (s[i] != 0) .{ 255, 255, 255 } else .{ 0, 0, 0 },
            3 => plte[s[i] * 3 ..][0..3].*,
            else => .{ @intCast(s[i * 3] >> 8), @intCast(s[i * 3 + 1] >> 8), @intCast(s[i * 3 + 2] >> 8) },
        };
        for ([_]u8{ 0, 1 }) |interlace| {
            const r = try rows(a, w, h, interlace, @intCast(c[0]), ch, s[0 .. w * h * ch]);
            defer a.free(r);
            for ([_][]const u8{ r, r[0 .. r.len - 1] }, 0..) |raw, short| {
                const z = try stored(a, raw);
                defer a.free(z);
                const data = try png(a, w, h, c[0], c[1], interlace, if (c[1] == 3) &plte else "", z);
                defer a.free(data);
                if (short == 1) {
                    try std.testing.expectError(error.BadPng, picture.decodePng(a, data));
                    continue;
                }
                var rgb = try picture.decodePng(a, data);
                defer rgb.deinit(a);
                try std.testing.expectEqualSlices(u8, &want, rgb.px);
            }
        }
    }
}
