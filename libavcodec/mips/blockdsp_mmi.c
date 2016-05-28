/*
 * Loongson SIMD optimized blockdsp
 *
 * Copyright (c) 2015 Loongson Technology Corporation Limited
 * Copyright (c) 2015 Zhou Xiaoyong <zhouxiaoyong@loongson.cn>
 *
 * This file is part of FFmpeg.
 *
 * FFmpeg is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * FFmpeg is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with FFmpeg; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA
 */

#include "blockdsp_mips.h"
#include "libavutil/mips/asmdefs.h"

void ff_fill_block16_mmi(uint8_t *block, uint8_t value, int line_size, int h)
{
    double ftmp[1];
    uint64_t all64;

    __asm__ volatile (
        "mtc1       %[value],   %[ftmp0]                                \n\t"
        "punpcklbh  %[ftmp0],   %[ftmp0],       %[ftmp0]                \n\t"
        "punpcklbh  %[ftmp0],   %[ftmp0],       %[ftmp0]                \n\t"
        "punpcklbh  %[ftmp0],   %[ftmp0],       %[ftmp0]                \n\t"
        "1:                                                             \n\t"
#if HAVE_LOONGSON3
        "gssdlc1    %[ftmp0],   0x07(%[block])                          \n\t"
        "gssdrc1    %[ftmp0],   0x00(%[block])                          \n\t"
        PTR_ADDI    "%[h],      %[h],           -0x01                   \n\t"
        "gssdlc1    %[ftmp0],   0x0f(%[block])                          \n\t"
        "gssdrc1    %[ftmp0],   0x08(%[block])                          \n\t"
#elif HAVE_LOONGSON2
        "dmfc1      %[all64],   %[ftmp0]                                \n\t"
        "usd        %[all64],   0x00(%[block])                          \n\t"
        PTR_ADDI   "%[h],       %[h],           -0x01                   \n\t"
        "usd        %[all64],   0x08(%[block])                          \n\t"
#endif
        PTR_ADDU   "%[block],   %[block],       %[line_size]            \n\t"
        "bnez       %[h],       1b                                      \n\t"
        : [block]"+&r"(block),              [h]"+&r"(h),
          [ftmp0]"=&f"(ftmp[0]),
          [all64]"=&r"(all64)
        : [value]"r"(value),                [line_size]"r"((mips_reg)line_size)
        : "memory"
    );
}

void ff_fill_block8_mmi(uint8_t *block, uint8_t value, int line_size, int h)
{
    double ftmp0;
    uint64_t all64;

    __asm__ volatile (
        "mtc1       %[value],   %[ftmp0]                                \n\t"
        "punpcklbh  %[ftmp0],   %[ftmp0],       %[ftmp0]                \n\t"
        "punpcklbh  %[ftmp0],   %[ftmp0],       %[ftmp0]                \n\t"
        "punpcklbh  %[ftmp0],   %[ftmp0],       %[ftmp0]                \n\t"
        "1:                                                             \n\t"
#if HAVE_LOONGSON3
        "gssdlc1    %[ftmp0],   0x07(%[block])                          \n\t"
        "gssdrc1    %[ftmp0],   0x00(%[block])                          \n\t"
#elif HAVE_LOONGSON2
        "dmfc1      %[all64],   %[ftmp0]                                \n\t"
        "usd        %[all64],   0x00(%[block])                          \n\t"
#endif
        PTR_ADDI   "%[h],       %[h],           -0x01                   \n\t"
        PTR_ADDU   "%[block],   %[block],       %[line_size]            \n\t"
        "bnez       %[h],       1b                                      \n\t"
        : [block]"+&r"(block),              [h]"+&r"(h),
          [ftmp0]"=&f"(ftmp0),
          [all64]"=&r"(all64)
        : [value]"r"(value),                [line_size]"r"((mips_reg)line_size)
        : "memory"
    );
}

void ff_clear_block_mmi(int16_t *block)
{
    double ftmp[2];

    __asm__ volatile (
        "xor        %[ftmp0],   %[ftmp0],       %[ftmp0]                \n\t"
        "xor        %[ftmp1],   %[ftmp1],       %[ftmp1]                \n\t"
#if HAVE_LOONGSON3
        "gssqc1     %[ftmp0],   %[ftmp1],       0x00(%[block])          \n\t"
        "gssqc1     %[ftmp0],   %[ftmp1],       0x10(%[block])          \n\t"
        "gssqc1     %[ftmp0],   %[ftmp1],       0x20(%[block])          \n\t"
        "gssqc1     %[ftmp0],   %[ftmp1],       0x30(%[block])          \n\t"
        "gssqc1     %[ftmp0],   %[ftmp1],       0x40(%[block])          \n\t"
        "gssqc1     %[ftmp0],   %[ftmp1],       0x50(%[block])          \n\t"
        "gssqc1     %[ftmp0],   %[ftmp1],       0x60(%[block])          \n\t"
        "gssqc1     %[ftmp0],   %[ftmp1],       0x70(%[block])          \n\t"
#elif HAVE_LOONGSON2
        "sdc1       %[ftmp0],   0x00(%[block])                          \n\t"
        "sdc1       %[ftmp1],   0x08(%[block])                          \n\t"
        "sdc1       %[ftmp0],   0x10(%[block])                          \n\t"
        "sdc1       %[ftmp1],   0x18(%[block])                          \n\t"
        "sdc1       %[ftmp0],   0x20(%[block])                          \n\t"
        "sdc1       %[ftmp1],   0x28(%[block])                          \n\t"
        "sdc1       %[ftmp0],   0x30(%[block])                          \n\t"
        "sdc1       %[ftmp1],   0x38(%[block])                          \n\t"
        "sdc1       %[ftmp0],   0x40(%[block])                          \n\t"
        "sdc1       %[ftmp1],   0x48(%[block])                          \n\t"
        "sdc1       %[ftmp0],   0x50(%[block])                          \n\t"
        "sdc1       %[ftmp1],   0x58(%[block])                          \n\t"
        "sdc1       %[ftmp0],   0x60(%[block])                          \n\t"
        "sdc1       %[ftmp1],   0x68(%[block])                          \n\t"
        "sdc1       %[ftmp0],   0x70(%[block])                          \n\t"
        "sdc1       %[ftmp1],   0x78(%[block])                          \n\t"
#endif
        : [ftmp0]"=&f"(ftmp[0]),            [ftmp1]"=&f"(ftmp[1])
        : [block]"r"(block)
        : "memory"
    );
}

void ff_clear_blocks_mmi(int16_t *block)
{
    double ftmp[2];

    __asm__ volatile (
        "xor        %[ftmp0],   %[ftmp0],       %[ftmp0]                \n\t"
        "xor        %[ftmp1],   %[ftmp1],       %[ftmp1]                \n\t"
#if HAVE_LOONGSON3
        "gssqc1     %[ftmp0],   %[ftmp1],       0x00(%[block])          \n\t"
        "gssqc1     %[ftmp0],   %[ftmp1],       0x10(%[block])          \n\t"
        "gssqc1     %[ftmp0],   %[ftmp1],       0x20(%[block])          \n\t"
        "gssqc1     %[ftmp0],   %[ftmp1],       0x30(%[block])          \n\t"
        "gssqc1     %[ftmp0],   %[ftmp1],       0x40(%[block])          \n\t"
        "gssqc1     %[ftmp0],   %[ftmp1],       0x50(%[block])          \n\t"
        "gssqc1     %[ftmp0],   %[ftmp1],       0x60(%[block])          \n\t"
        "gssqc1     %[ftmp0],   %[ftmp1],       0x70(%[block])          \n\t"

        "gssqc1     %[ftmp0],   %[ftmp1],       0x80(%[block])          \n\t"
        "gssqc1     %[ftmp0],   %[ftmp1],       0x90(%[block])          \n\t"
        "gssqc1     %[ftmp0],   %[ftmp1],       0xa0(%[block])          \n\t"
        "gssqc1     %[ftmp0],   %[ftmp1],       0xb0(%[block])          \n\t"
        "gssqc1     %[ftmp0],   %[ftmp1],       0xc0(%[block])          \n\t"
        "gssqc1     %[ftmp0],   %[ftmp1],       0xd0(%[block])          \n\t"
        "gssqc1     %[ftmp0],   %[ftmp1],       0xe0(%[block])          \n\t"
        "gssqc1     %[ftmp0],   %[ftmp1],       0xf0(%[block])          \n\t"

        "gssqc1     %[ftmp0],   %[ftmp1],       0x100(%[block])         \n\t"
        "gssqc1     %[ftmp0],   %[ftmp1],       0x110(%[block])         \n\t"
        "gssqc1     %[ftmp0],   %[ftmp1],       0x120(%[block])         \n\t"
        "gssqc1     %[ftmp0],   %[ftmp1],       0x130(%[block])         \n\t"
        "gssqc1     %[ftmp0],   %[ftmp1],       0x140(%[block])         \n\t"
        "gssqc1     %[ftmp0],   %[ftmp1],       0x150(%[block])         \n\t"
        "gssqc1     %[ftmp0],   %[ftmp1],       0x160(%[block])         \n\t"
        "gssqc1     %[ftmp0],   %[ftmp1],       0x170(%[block])         \n\t"

        "gssqc1     %[ftmp0],   %[ftmp1],       0x180(%[block])         \n\t"
        "gssqc1     %[ftmp0],   %[ftmp1],       0x190(%[block])         \n\t"
        "gssqc1     %[ftmp0],   %[ftmp1],       0x1a0(%[block])         \n\t"
        "gssqc1     %[ftmp0],   %[ftmp1],       0x1b0(%[block])         \n\t"
        "gssqc1     %[ftmp0],   %[ftmp1],       0x1c0(%[block])         \n\t"
        "gssqc1     %[ftmp0],   %[ftmp1],       0x1d0(%[block])         \n\t"
        "gssqc1     %[ftmp0],   %[ftmp1],       0x1e0(%[block])         \n\t"
        "gssqc1     %[ftmp0],   %[ftmp1],       0x1f0(%[block])         \n\t"

        "gssqc1     %[ftmp0],   %[ftmp1],       0x200(%[block])         \n\t"
        "gssqc1     %[ftmp0],   %[ftmp1],       0x210(%[block])         \n\t"
        "gssqc1     %[ftmp0],   %[ftmp1],       0x220(%[block])         \n\t"
        "gssqc1     %[ftmp0],   %[ftmp1],       0x230(%[block])         \n\t"
        "gssqc1     %[ftmp0],   %[ftmp1],       0x240(%[block])         \n\t"
        "gssqc1     %[ftmp0],   %[ftmp1],       0x250(%[block])         \n\t"
        "gssqc1     %[ftmp0],   %[ftmp1],       0x260(%[block])         \n\t"
        "gssqc1     %[ftmp0],   %[ftmp1],       0x270(%[block])         \n\t"

        "gssqc1     %[ftmp0],   %[ftmp1],       0x280(%[block])         \n\t"
        "gssqc1     %[ftmp0],   %[ftmp1],       0x290(%[block])         \n\t"
        "gssqc1     %[ftmp0],   %[ftmp1],       0x2a0(%[block])         \n\t"
        "gssqc1     %[ftmp0],   %[ftmp1],       0x2b0(%[block])         \n\t"
        "gssqc1     %[ftmp0],   %[ftmp1],       0x2c0(%[block])         \n\t"
        "gssqc1     %[ftmp0],   %[ftmp1],       0x2d0(%[block])         \n\t"
        "gssqc1     %[ftmp0],   %[ftmp1],       0x2e0(%[block])         \n\t"
        "gssqc1     %[ftmp0],   %[ftmp1],       0x2f0(%[block])         \n\t"
#elif HAVE_LOONGSON2
        "sdc1       %[ftmp0],   0x00(%[block])                          \n\t"
        "sdc1       %[ftmp1],   0x08(%[block])                          \n\t"
        "sdc1       %[ftmp0],   0x10(%[block])                          \n\t"
        "sdc1       %[ftmp1],   0x18(%[block])                          \n\t"
        "sdc1       %[ftmp0],   0x20(%[block])                          \n\t"
        "sdc1       %[ftmp1],   0x28(%[block])                          \n\t"
        "sdc1       %[ftmp0],   0x30(%[block])                          \n\t"
        "sdc1       %[ftmp1],   0x38(%[block])                          \n\t"
        "sdc1       %[ftmp0],   0x40(%[block])                          \n\t"
        "sdc1       %[ftmp1],   0x48(%[block])                          \n\t"
        "sdc1       %[ftmp0],   0x50(%[block])                          \n\t"
        "sdc1       %[ftmp1],   0x58(%[block])                          \n\t"
        "sdc1       %[ftmp0],   0x60(%[block])                          \n\t"
        "sdc1       %[ftmp1],   0x68(%[block])                          \n\t"
        "sdc1       %[ftmp0],   0x70(%[block])                          \n\t"
        "sdc1       %[ftmp1],   0x78(%[block])                          \n\t"

        "sdc1       %[ftmp0],   0x80(%[block])                          \n\t"
        "sdc1       %[ftmp1],   0x88(%[block])                          \n\t"
        "sdc1       %[ftmp0],   0x90(%[block])                          \n\t"
        "sdc1       %[ftmp1],   0x98(%[block])                          \n\t"
        "sdc1       %[ftmp0],   0xa0(%[block])                          \n\t"
        "sdc1       %[ftmp1],   0xa8(%[block])                          \n\t"
        "sdc1       %[ftmp0],   0xb0(%[block])                          \n\t"
        "sdc1       %[ftmp1],   0xb8(%[block])                          \n\t"
        "sdc1       %[ftmp0],   0xc0(%[block])                          \n\t"
        "sdc1       %[ftmp1],   0xc8(%[block])                          \n\t"
        "sdc1       %[ftmp0],   0xd0(%[block])                          \n\t"
        "sdc1       %[ftmp1],   0xd8(%[block])                          \n\t"
        "sdc1       %[ftmp0],   0xe0(%[block])                          \n\t"
        "sdc1       %[ftmp1],   0xe8(%[block])                          \n\t"
        "sdc1       %[ftmp0],   0xf0(%[block])                          \n\t"
        "sdc1       %[ftmp1],   0xf8(%[block])                          \n\t"

        "sdc1       %[ftmp0],   0x100(%[block])                         \n\t"
        "sdc1       %[ftmp1],   0x108(%[block])                         \n\t"
        "sdc1       %[ftmp0],   0x110(%[block])                         \n\t"
        "sdc1       %[ftmp1],   0x118(%[block])                         \n\t"
        "sdc1       %[ftmp0],   0x120(%[block])                         \n\t"
        "sdc1       %[ftmp1],   0x128(%[block])                         \n\t"
        "sdc1       %[ftmp0],   0x130(%[block])                         \n\t"
        "sdc1       %[ftmp1],   0x138(%[block])                         \n\t"
        "sdc1       %[ftmp0],   0x140(%[block])                         \n\t"
        "sdc1       %[ftmp1],   0x148(%[block])                         \n\t"
        "sdc1       %[ftmp0],   0x150(%[block])                         \n\t"
        "sdc1       %[ftmp1],   0x158(%[block])                         \n\t"
        "sdc1       %[ftmp0],   0x160(%[block])                         \n\t"
        "sdc1       %[ftmp1],   0x168(%[block])                         \n\t"
        "sdc1       %[ftmp0],   0x170(%[block])                         \n\t"
        "sdc1       %[ftmp1],   0x178(%[block])                         \n\t"

        "sdc1       %[ftmp0],   0x180(%[block])                         \n\t"
        "sdc1       %[ftmp1],   0x188(%[block])                         \n\t"
        "sdc1       %[ftmp0],   0x190(%[block])                         \n\t"
        "sdc1       %[ftmp1],   0x198(%[block])                         \n\t"
        "sdc1       %[ftmp0],   0x1a0(%[block])                         \n\t"
        "sdc1       %[ftmp1],   0x1a8(%[block])                         \n\t"
        "sdc1       %[ftmp0],   0x1b0(%[block])                         \n\t"
        "sdc1       %[ftmp1],   0x1b8(%[block])                         \n\t"
        "sdc1       %[ftmp0],   0x1c0(%[block])                         \n\t"
        "sdc1       %[ftmp1],   0x1c8(%[block])                         \n\t"
        "sdc1       %[ftmp0],   0x1d0(%[block])                         \n\t"
        "sdc1       %[ftmp1],   0x1d8(%[block])                         \n\t"
        "sdc1       %[ftmp0],   0x1e0(%[block])                         \n\t"
        "sdc1       %[ftmp1],   0x1e8(%[block])                         \n\t"
        "sdc1       %[ftmp0],   0x1f0(%[block])                         \n\t"
        "sdc1       %[ftmp1],   0x1f8(%[block])                         \n\t"

        "sdc1       %[ftmp0],   0x200(%[block])                         \n\t"
        "sdc1       %[ftmp1],   0x208(%[block])                         \n\t"
        "sdc1       %[ftmp0],   0x210(%[block])                         \n\t"
        "sdc1       %[ftmp1],   0x218(%[block])                         \n\t"
        "sdc1       %[ftmp0],   0x220(%[block])                         \n\t"
        "sdc1       %[ftmp1],   0x228(%[block])                         \n\t"
        "sdc1       %[ftmp0],   0x230(%[block])                         \n\t"
        "sdc1       %[ftmp1],   0x238(%[block])                         \n\t"
        "sdc1       %[ftmp0],   0x240(%[block])                         \n\t"
        "sdc1       %[ftmp1],   0x248(%[block])                         \n\t"
        "sdc1       %[ftmp0],   0x250(%[block])                         \n\t"
        "sdc1       %[ftmp1],   0x258(%[block])                         \n\t"
        "sdc1       %[ftmp0],   0x260(%[block])                         \n\t"
        "sdc1       %[ftmp1],   0x268(%[block])                         \n\t"
        "sdc1       %[ftmp0],   0x270(%[block])                         \n\t"
        "sdc1       %[ftmp1],   0x278(%[block])                         \n\t"

        "sdc1       %[ftmp0],   0x280(%[block])                         \n\t"
        "sdc1       %[ftmp1],   0x288(%[block])                         \n\t"
        "sdc1       %[ftmp0],   0x290(%[block])                         \n\t"
        "sdc1       %[ftmp1],   0x298(%[block])                         \n\t"
        "sdc1       %[ftmp0],   0x2a0(%[block])                         \n\t"
        "sdc1       %[ftmp1],   0x2a8(%[block])                         \n\t"
        "sdc1       %[ftmp0],   0x2b0(%[block])                         \n\t"
        "sdc1       %[ftmp1],   0x2b8(%[block])                         \n\t"
        "sdc1       %[ftmp0],   0x2c0(%[block])                         \n\t"
        "sdc1       %[ftmp1],   0x2c8(%[block])                         \n\t"
        "sdc1       %[ftmp0],   0x2d0(%[block])                         \n\t"
        "sdc1       %[ftmp1],   0x2d8(%[block])                         \n\t"
        "sdc1       %[ftmp0],   0x2e0(%[block])                         \n\t"
        "sdc1       %[ftmp1],   0x2e8(%[block])                         \n\t"
        "sdc1       %[ftmp0],   0x2f0(%[block])                         \n\t"
        "sdc1       %[ftmp1],   0x2f8(%[block])                         \n\t"
#endif
        : [ftmp0]"=&f"(ftmp[0]),            [ftmp1]"=&f"(ftmp[1])
        : [block]"r"((uint64_t *)block)
        : "memory"
    );
}
