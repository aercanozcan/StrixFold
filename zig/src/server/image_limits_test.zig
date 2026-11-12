//! A request's image parts against --vision-max-images (messages.Images.max): counted before any part's base64 is read.
const std = @import("std");
const json = @import("json.zig");
const messages = @import("messages.zig");
const Cx = @import("errors.zig").Cx;

const placeholder = "<\u{ff5c}deepseek_image\u{ff5c}>";

/// One user message: a text part, then `n` image parts whose data URLs carry `data`.
fn request(a: std.mem.Allocator, n: usize, data: []const u8) !json.Value {
    var text: std.ArrayList(u8) = .empty;
    try text.appendSlice(a, "[{\"role\": \"user\", \"content\": [{\"type\": \"text\", \"text\": \"compare\"}");
    for (0..n) |_| try text.print(a, ", {{\"type\": \"image_url\", \"image_url\": {{\"url\": \"data:image/png;base64,{s}\"}}}}", .{data});
    try text.appendSlice(a, "]}]");
    return (try json.parseText(a, text.items)).ok;
}

fn normalize(cx: *Cx, list: json.Value, max: u32) !usize {
    var images: messages.Images = .{ .placeholder = placeholder, .max = max };
    _ = try messages.normalizeImages(cx, list, "system", false, .generic, &images);
    return images.list.items.len;
}

test "more image parts than --vision-max-images are refused before any is decoded; up to it they pass" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var cx: Cx = .{ .a = a };
    // three parts whose base64 is cut short: past a limit of 2 the count is refused, not the base64
    try std.testing.expectError(error.Refused, normalize(&cx, try request(a, 3, "aGVsbG8hZ"), 2));
    try std.testing.expect(cx.kind == .request);
    try std.testing.expect(std.mem.indexOf(u8, cx.message, "3 images") != null);
    try std.testing.expect(std.mem.indexOf(u8, cx.message, "--vision-max-images") != null);
    // at a limit of 3 the same parts are read, and the first one's base64 is refused
    try std.testing.expectError(error.Refused, normalize(&cx, try request(a, 3, "aGVsbG8hZ"), 3));
    try std.testing.expect(std.mem.indexOf(u8, cx.message, "base64") != null);
    // two good parts at a limit of 2
    try std.testing.expectEqual(@as(usize, 2), try normalize(&cx, try request(a, 2, "aGVsbG8="), 2));
}

test "the count takes every message's image parts, earlier turns and tool results too" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var cx: Cx = .{ .a = a };
    const part = "{\"type\": \"image_url\", \"image_url\": {\"url\": \"data:image/png;base64,aGVsbG8=\"}}";
    const list = (try json.parseText(a, "[{\"role\": \"user\", \"content\": [" ++ part ++ "]}, {\"role\": \"assistant\", \"content\": \"seen\"}, " ++
        "{\"role\": \"user\", \"content\": [" ++ part ++ "]}, {\"role\": \"tool\", \"tool_call_id\": \"c1\", \"content\": [" ++ part ++ "]}]")).ok;
    try std.testing.expectEqual(@as(usize, 3), try normalize(&cx, list, 3));
    try std.testing.expectError(error.Refused, normalize(&cx, list, 2));
    try std.testing.expect(std.mem.indexOf(u8, cx.message, "3 images") != null);
}
