#!/usr/bin/env python3
"""Generate simple PNG icons for SeeChords extension."""
import struct, zlib, math, os

def create_png(size, filename):
    pixels = []
    cx, cy = size // 2, size // 2
    r = size // 2 - 2
    for y in range(size):
        row = []
        for x in range(size):
            dx, dy = x - cx, y - cy
            dist = (dx * dx + dy * dy) ** 0.5
            if dist <= r:
                ratio = dist / r
                R = int(108 - 40 * ratio)
                G = int(71 - 30 * ratio)
                B = int(255 - 60 * ratio)
                A = 255
                angle = math.atan2(dy, dx)
                inner_r = r * 0.35
                outer_r = r * 0.65
                if inner_r < dist < outer_r and (angle < -0.5 or angle > 0.5):
                    R, G, B = 255, 215, 0
                row.append(bytes([R, G, B, A]))
            else:
                row.append(bytes([0, 0, 0, 0]))
        pixels.append(b'\x00' + b''.join(row))

    raw_data = b''.join(pixels)

    def chunk(ctype, data):
        c = ctype + data
        return struct.pack('>I', len(data)) + c + struct.pack('>I', zlib.crc32(c) & 0xffffffff)

    sig = b'\x89PNG\r\n\x1a\n'
    ihdr = struct.pack('>IIBBBBB', size, size, 8, 6, 0, 0, 0)
    idat = zlib.compress(raw_data)

    with open(filename, 'wb') as f:
        f.write(sig)
        f.write(chunk(b'IHDR', ihdr))
        f.write(chunk(b'IDAT', idat))
        f.write(chunk(b'IEND', b''))

out_dir = os.path.dirname(os.path.abspath(__file__))
icons_dir = os.path.join(out_dir, 'extension', 'icons')
os.makedirs(icons_dir, exist_ok=True)

create_png(48, os.path.join(icons_dir, 'icon48.png'))
create_png(128, os.path.join(icons_dir, 'icon128.png'))
print('Icons created successfully')
