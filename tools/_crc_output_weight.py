import json, struct, sys, zlib

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")

manifest_path = r"E:\models\GPT2\sdds\model_dds_manifest.json"
sdds_dir      = r"E:\models\GPT2\sdds"
CHUNK_HDR     = 80
STREAM_SZ     = 64 * 1024 * 1024

with open(manifest_path, encoding="utf-8") as f:
    manifest = json.load(f)

output_slices = []
for shard in manifest:
    for sl in shard["tensors"]:
        if sl["name"] == "output.weight":
            output_slices.append((shard["dds_file"], sl))

output_slices.sort(key=lambda x: x[1]["offset_in_tensor"])
print(f"output.weight: {len(output_slices)} slice(s)")

all_ok = True
for dds_file, sl in output_slices:
    path      = sdds_dir + "\\" + dds_file
    off       = sl["offset_in_file"]
    sz        = sl["size"]
    off_t     = sl["offset_in_tensor"]

    with open(path, "rb") as fh:
        fh.seek(off - CHUNK_HDR)
        hdr = fh.read(CHUNK_HDR)
        stored_crc, = struct.unpack_from("<I", hdr, 76)
        fh.seek(off)
        crc = 0
        remaining = sz
        while remaining > 0:
            chunk = fh.read(min(remaining, STREAM_SZ))
            if not chunk:
                break
            crc = zlib.crc32(chunk, crc)
            remaining -= len(chunk)
        crc &= 0xFFFFFFFF

    ok = crc == stored_crc
    if not ok:
        all_ok = False
    tag = "PASS" if ok else "FAIL"
    print(f"  [{tag}]  {dds_file}  off={off}  sz={sz:,}  stored=0x{stored_crc:08x}  computed=0x{crc:08x}  off_tensor={off_t}")

print()
print("output.weight CRC:", "PASS" if all_ok else "FAIL")
