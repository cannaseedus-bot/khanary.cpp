#include "sha256.h"
#include <cstring>
#include <sstream>
#include <iomanip>

namespace khanary::util {

namespace {

constexpr uint32_t K[64] = {
    0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
    0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
    0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
    0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
    0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
    0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
    0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
    0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2
};

inline uint32_t rotr(uint32_t x, uint32_t n) { return (x >> n) | (x << (32 - n)); }
inline uint32_t ch(uint32_t x,uint32_t y,uint32_t z){return (x&y)^(~x&z);}
inline uint32_t maj(uint32_t x,uint32_t y,uint32_t z){return (x&y)^(x&z)^(y&z);}
inline uint32_t ep0(uint32_t x){return rotr(x,2)^rotr(x,13)^rotr(x,22);}
inline uint32_t ep1(uint32_t x){return rotr(x,6)^rotr(x,11)^rotr(x,25);}
inline uint32_t sig0(uint32_t x){return rotr(x,7)^rotr(x,18)^(x>>3);}
inline uint32_t sig1(uint32_t x){return rotr(x,17)^rotr(x,19)^(x>>10);}

} // anon

std::array<uint8_t, 32> sha256_bytes(const void* data, size_t len) {
    uint32_t h[8] = {
        0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,
        0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19
    };

    auto* ptr = static_cast<const uint8_t*>(data);
    uint64_t bit_len = (uint64_t)len * 8;

    auto process_block = [&](const uint8_t* blk) {
        uint32_t w[64];
        for (int i = 0; i < 16; ++i)
            w[i] = ((uint32_t)blk[i*4]<<24)|((uint32_t)blk[i*4+1]<<16)|
                   ((uint32_t)blk[i*4+2]<<8)|(uint32_t)blk[i*4+3];
        for (int i = 16; i < 64; ++i)
            w[i] = sig1(w[i-2])+w[i-7]+sig0(w[i-15])+w[i-16];

        uint32_t a=h[0],b=h[1],c=h[2],d=h[3],e=h[4],f=h[5],g=h[6],hh=h[7];
        for (int i = 0; i < 64; ++i) {
            uint32_t t1 = hh+ep1(e)+ch(e,f,g)+K[i]+w[i];
            uint32_t t2 = ep0(a)+maj(a,b,c);
            hh=g; g=f; f=e; e=d+t1; d=c; c=b; b=a; a=t1+t2;
        }
        h[0]+=a; h[1]+=b; h[2]+=c; h[3]+=d;
        h[4]+=e; h[5]+=f; h[6]+=g; h[7]+=hh;
    };

    size_t i = 0;
    for (; i + 64 <= len; i += 64)
        process_block(ptr + i);

    uint8_t tail[128] = {};
    size_t rem = len - i;
    memcpy(tail, ptr + i, rem);
    tail[rem] = 0x80;

    size_t pad_len = (rem < 56) ? 64 : 128;
    for (int k = 7; k >= 0; --k) {
        tail[pad_len - 1 - k] = (uint8_t)(bit_len >> (k * 8));
    }
    process_block(tail);
    if (pad_len == 128)
        process_block(tail + 64);

    std::array<uint8_t, 32> out;
    for (int j = 0; j < 8; ++j) {
        out[j*4+0] = (h[j]>>24)&0xFF;
        out[j*4+1] = (h[j]>>16)&0xFF;
        out[j*4+2] = (h[j]>>8)&0xFF;
        out[j*4+3] = h[j]&0xFF;
    }
    return out;
}

std::string sha256_hex(const void* data, size_t len) {
    auto bytes = sha256_bytes(data, len);
    std::ostringstream ss;
    ss << std::hex << std::setfill('0');
    for (auto b : bytes) ss << std::setw(2) << (int)b;
    return ss.str();
}

std::string sha256_hex(const std::string& s) {
    return sha256_hex(s.data(), s.size());
}

} // namespace khanary::util
