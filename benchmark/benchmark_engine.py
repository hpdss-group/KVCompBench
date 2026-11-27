"""PNG renderer and compatibility entrypoint for the generic benchmark."""
import struct,zlib

BACKGROUND=(247,249,252,255)
INK=(22,38,56,255)
MUTED=(90,110,130,255)

def _cell_color(scale):
    # light teal ramp: pale mint -> deep teal, readable on a light page
    return (int(230-214*scale), int(244-122*scale), int(240-134*scale), 255)

def _readable(color):
    r,g,b=color[0],color[1],color[2]
    return INK if (0.299*r+0.587*g+0.114*b)>150 else (255,255,255,255)

def write_png(path, rows, codec_names):
    # Dependency-free PNG heatmap: rows=datasets, columns=codecs, teal=larger ratio.
    margin, cell_w, cell_h = 220, 210, 95
    datasets = sorted({row["dataset"] for row in rows})
    width, height = max(980, margin + len(codec_names) * cell_w + 30), max(340, 95 + len(datasets) * cell_h + 55)
    pixels = bytearray(BACKGROUND) * (width * height)
    font = {
        'C':('01111','10000','10000','10000','10000','10000','01111'),
        'U':('10001','10001','10001','10001','10001','10001','01110'),
        'O':('01110','10001','10001','10001','10001','10001','01110'),
        'B':('11110','10001','10001','11110','10001','10001','11110'),
        'M':('10001','11011','10101','10101','10001','10001','10001'),
        'V':('10001','10001','10001','10001','10001','01010','00100'),
        'H':('10001','10001','10001','11111','10001','10001','10001'),
        '-':('00000','00000','00000','11111','00000','00000','00000'),
        ':':('00000','00100','00100','00000','00100','00100','00000'),
        "A":("01110","10001","10001","11111","10001","10001","10001"), "D":("11110","10001","10001","10001","10001","10001","11110"),
        "E":("11111","10000","10000","11110","10000","10000","11111"), "F":("11111","10000","10000","11110","10000","10000","10000"),
        "G":("01111","10000","10000","10111","10001","10001","01110"), "I":("11111","00100","00100","00100","00100","00100","11111"),
        "K":("10001","10010","10100","11000","10100","10010","10001"), "L":("10000","10000","10000","10000","10000","10000","11111"),
        "N":("10001","11001","10101","10011","10001","10001","10001"), "P":("11110","10001","10001","11110","10000","10000","10000"),
        "Q":("01110","10001","10001","10001","10101","10010","01101"), "R":("11110","10001","10001","11110","10100","10010","10001"),
        "S":("01111","10000","10000","01110","00001","00001","11110"), "T":("11111","00100","00100","00100","00100","00100","00100"),
        "W":("10001","10001","10001","10101","10101","10101","01010"), "Y":("10001","10001","01010","00100","00100","00100","00100"),
        "Z":("11111","00001","00010","00100","01000","10000","11111"), "0":("01110","10001","10011","10101","11001","10001","01110"),
        "1":("00100","01100","00100","00100","00100","00100","01110"), "2":("01110","10001","00001","00010","00100","01000","11111"),
        "3":("11110","00001","00001","01110","00001","00001","11110"), "4":("00010","00110","01010","10010","11111","00010","00010"),
        "5":("11111","10000","10000","11110","00001","00001","11110"), "6":("01110","10000","10000","11110","10001","10001","01110"),
        "7":("11111","00001","00010","00100","01000","01000","01000"), "8":("01110","10001","10001","01110","10001","10001","01110"),
        "9":("01110","10001","10001","01111","00001","00001","01110"), ".":("00000","00000","00000","00000","00000","00110","00110"), " ":("00000",)*7,
    }
    def pixel(x, y, color):
        if 0 <= x < width and 0 <= y < height:
            pos = 4 * (y * width + x); pixels[pos:pos + 4] = bytes(color)
    def text(x, y, value, color=INK, scale=2):
        for char in value.upper():
            for ry, row in enumerate(font.get(char, font[" "])):
                for rx, bit in enumerate(row):
                    if bit == "1":
                        for dy in range(scale):
                            for dx in range(scale): pixel(x + rx * scale + dx, y + ry * scale + dy, color)
            x += 6 * scale
    text(margin, 12, 'CODEC PATH', scale=2)
    maximum = max(1.0, *(row["compression_ratio"] for row in rows))
    for y, dataset in enumerate(datasets):
        for x, codec in enumerate(codec_names):
            row = next((r for r in rows if r["dataset"] == dataset and r["codec"] == codec), None)
            if not row:
                continue
            scale = min(1.0, row["compression_ratio"] / maximum)
            color = _cell_color(scale)
            x0, y0 = margin + x * cell_w, 95 + y * cell_h
            for py in range(y0, y0 + cell_h - 8):
                for px in range(x0, x0 + cell_w - 8):
                    offset = 4 * (py * width + px)
                    pixels[offset:offset + 4] = bytes(color)
            text(x0 + 12, y0 + 20, f"{row['compression_ratio']:.2f}", color=_readable(color), scale=3)
    headers = {"zstd": "ZSTD", "snappy": "SNAPPY", "gzip": "GZIP", "lz4_block": "LZ4", "raw_deflate": "RAW DEFLATE"}
    for index, codec in enumerate(codec_names):
        text(margin + index * cell_w, 47, headers.get(codec, codec.upper()), color=INK, scale=1)
        mode = next(r['execution_path'] for r in rows if r['codec'] == codec)
        text(margin + index * cell_w, 67, mode, color=MUTED, scale=2)
    for index, dataset in enumerate(datasets):
        label = dataset.replace("_bf16", "")
        label_scale = 1 if len(label) * 12 > margin - 44 else 2
        text(22, 125 + index * cell_h, label, color=INK, scale=label_scale)
    raw = b"".join(b"\0" + pixels[y * width * 4:(y + 1) * width * 4] for y in range(height))
    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xffffffff)
    png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0)) + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b"")
    path.write_bytes(png)


def main():
    from run_full_cpu_encode_gpu_decode import main as run
    run()

if __name__ == "__main__":main()
