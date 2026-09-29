#pragma once
#include <cstdint>

namespace khanary::klsl {

// Low-level KLSL bytecode opcodes, mirrored from E:\models\.ASX.cpp\klsl\klsl_opcodes.h
// Each instruction is 4 bytes: [op:1][a:1][b:1][c:1]
// Operand fields a/b/c are register indices into a 256-register file.
enum KLSLOp : uint8_t {
    OP_NOP          = 0x00,

    // ---- Register / Immediate ----
    OP_LOAD_IMMU    = 0x01, // a = imm_u32 (b:c as 16-bit imm, zero-extended)
    OP_LOAD_CONST   = 0x02, // a = const_table[b]
    OP_MOV          = 0x03, // a = b
    OP_ZERO         = 0x04, // a = 0

    // ---- Integer Arithmetic ----
    OP_IADD         = 0x10, // a = b + c
    OP_ISUB         = 0x11, // a = b - c
    OP_IMUL         = 0x12, // a = b * c
    OP_IDIV         = 0x13, // a = b / c
    OP_IMOD         = 0x14, // a = b % c
    OP_INEG         = 0x15, // a = -b
    OP_IABS         = 0x16, // a = abs(b)
    OP_ILT          = 0x17, // a = (b < c) ? 1 : 0
    OP_ILE          = 0x18, // a = (b <= c) ? 1 : 0
    OP_IGT          = 0x19, // a = (b > c) ? 1 : 0
    OP_IEQ          = 0x1A, // a = (b == c) ? 1 : 0

    // ---- Float Arithmetic ----
    OP_FADD         = 0x20, // a = b + c
    OP_FSUB         = 0x21, // a = b - c
    OP_FMUL         = 0x22, // a = b * c
    OP_FDIV         = 0x23, // a = b / c
    OP_FMAD         = 0x24, // a = b * c + a  (fused multiply-add)
    OP_FNEG         = 0x25, // a = -b
    OP_FABS         = 0x26, // a = abs(b)
    OP_FSQRT        = 0x27, // a = sqrt(b)
    OP_FMAX         = 0x28, // a = max(b, c)
    OP_FMIN         = 0x29, // a = min(b, c)
    OP_RELU         = 0x2A, // a = max(b, 0.f)
    OP_FLT          = 0x2B, // a = (float)(b < c)
    OP_FEQ          = 0x2C, // a = (float)(b == c)

    // ---- Transcendentals ----
    OP_FSIN         = 0x30, // a = sin(b)
    OP_FCOS         = 0x31, // a = cos(b)
    OP_FTAN         = 0x32, // a = tan(b)
    OP_FEXP         = 0x33, // a = exp(b)
    OP_FLOG         = 0x34, // a = log(b)
    OP_FLOG2        = 0x35, // a = log2(b)
    OP_FPOW         = 0x36, // a = pow(b, c)

    // ---- Memory ----
    OP_LOAD_SHARED  = 0x40, // a = shared[b]
    OP_STORE_SHARED = 0x41, // shared[a] = b
    OP_LOAD_IN0     = 0x42, // a = in0[b]
    OP_LOAD_IN1     = 0x43, // a = in1[b]
    OP_LOAD_IN2     = 0x44, // a = in2[b]
    OP_STORE_OUT0   = 0x45, // out0[a] = b
    OP_STORE_OUT1   = 0x46, // out1[a] = b

    // ---- Atomics ----
    OP_ATOMIC_ADD   = 0x50, // InterlockedAdd(shared[a], b, c)
    OP_ATOMIC_MAX   = 0x51, // InterlockedMax(shared[a], b, c)
    OP_ATOMIC_MIN   = 0x52, // InterlockedMin(shared[a], b, c)
    OP_ATOMIC_CAS   = 0x53, // InterlockedCompareExchange(shared[a], b, c, a)

    // ---- Thread / Dispatch ----
    OP_THREAD_ID    = 0x60, // a = SV_DispatchThreadID.x
    OP_THREAD_CNT   = 0x61, // a = total thread count (M*N dispatch)
    OP_GROUP_ID     = 0x62, // a = SV_GroupID.x
    OP_LOCAL_ID     = 0x63, // a = SV_GroupThreadID.x
    OP_BARRIER      = 0x64, // GroupMemoryBarrierWithGroupSync()
    OP_SPAWN        = 0x65, // dispatch sub-kernel (bridge call)

    // ---- Control Flow ----
    OP_JMP          = 0xD0, // unconditional jump to label a
    OP_JZ           = 0xD1, // jump to a if b == 0
    OP_JNZ          = 0xD2, // jump to a if b != 0
    OP_CALL         = 0xD3, // call function a
    OP_RET_V        = 0xD4, // return value in a

    OP_RET          = 0xFF, // return void
};

// 4-byte KLSL instruction word
struct KLSLInstr {
    KLSLOp  op;
    uint8_t a; // destination / address register
    uint8_t b; // first source register
    uint8_t c; // second source register / immediate low byte
};

} // namespace khanary::klsl
