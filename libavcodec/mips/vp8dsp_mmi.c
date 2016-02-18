/*
 * Loongson SIMD optimized vp8dsp
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

#include "vp8dsp_mips.h"
#include "constants.h"
#include "libavutil/mips/asmdefs.h"

#define OK      1
#define NOTOK   1

#define TRANSPOSE_4H(f2, f4, f6, f8, f10, f12, f14, f16, f18, r8, f0, f30) \
        "li "#r8", 0x93                          \n\t" \
        "xor "#f0", "#f0","#f0"                  \n\t" \
        "mtc1 "#r8", "#f30"                      \n\t" \
        "punpcklhw "#f10", "#f2", "#f0"          \n\t" \
        "punpcklhw "#f18", "#f4", "#f0"          \n\t" \
        "pshufh "#f18", "#f18", "#f30"           \n\t" \
        "or "#f10", "#f10", "#f18"               \n\t" \
        "punpckhhw "#f12", "#f2", "#f0"          \n\t" \
        "punpckhhw "#f18", "#f4", "#f0"          \n\t" \
        "pshufh "#f18", "#f18", "#f30"           \n\t" \
        "or "#f12", "#f12", "#f18"               \n\t" \
        "punpcklhw "#f14", "#f6", "#f0"          \n\t" \
        "punpcklhw "#f18", "#f8", "#f0"          \n\t" \
        "pshufh "#f18", "#f18", "#f30"           \n\t" \
        "or "#f14", "#f14", "#f18"               \n\t" \
        "punpckhhw "#f16", "#f6", "#f0"          \n\t" \
        "punpckhhw "#f18", "#f8", "#f0"          \n\t" \
        "pshufh "#f18", "#f18", "#f30"           \n\t" \
        "or "#f16", "#f16", "#f18"               \n\t" \
        "punpcklwd "#f2", "#f10", "#f14"         \n\t" \
        "punpckhwd "#f4", "#f10", "#f14"         \n\t" \
        "punpcklwd "#f6", "#f12", "#f16"         \n\t" \
        "punpckhwd "#f8", "#f12", "#f16"         \n\t" \

#define clip_int8(n) (cm[(n) + 0x80] - 0x80)
static av_always_inline void vp8_filter_common_is4tap(uint8_t *p,
        ptrdiff_t stride)
{
    int av_unused p1 = p[-2 * stride];
    int av_unused p0 = p[-1 * stride];
    int av_unused q0 = p[ 0 * stride];
    int av_unused q1 = p[ 1 * stride];
    int a, f1, f2;
    const uint8_t *cm = ff_crop_tab + MAX_NEG_CROP;

    a = 3 * (q0 - p0);
    a += clip_int8(p1 - q1);
    a = clip_int8(a);

    // We deviate from the spec here with c(a+3) >> 3
    // since that's what libvpx does.
    f1 = FFMIN(a + 4, 127) >> 3;
    f2 = FFMIN(a + 3, 127) >> 3;

    // Despite what the spec says, we do need to clamp here to
    // be bitexact with libvpx.
    p[-1 * stride] = cm[p0 + f2];
    p[ 0 * stride] = cm[q0 - f1];
}

static av_always_inline void vp8_filter_common_isnot4tap(uint8_t *p,
        ptrdiff_t stride)
{
    int av_unused p1 = p[-2 * stride];
    int av_unused p0 = p[-1 * stride];
    int av_unused q0 = p[ 0 * stride];
    int av_unused q1 = p[ 1 * stride];
    int a, f1, f2;
    const uint8_t *cm = ff_crop_tab + MAX_NEG_CROP;

    a = 3 * (q0 - p0);
    a = clip_int8(a);

    // We deviate from the spec here with c(a+3) >> 3
    // since that's what libvpx does.
    f1 = FFMIN(a + 4, 127) >> 3;
    f2 = FFMIN(a + 3, 127) >> 3;

    // Despite what the spec says, we do need to clamp here to
    // be bitexact with libvpx.
    p[-1 * stride] = cm[p0 + f2];
    p[ 0 * stride] = cm[q0 - f1];
    a              = (f1 + 1) >> 1;
    p[-2 * stride] = cm[p1 + a];
    p[ 1 * stride] = cm[q1 - a];
}

static av_always_inline int vp8_simple_limit(uint8_t *p, ptrdiff_t stride,
        int flim)
{
    int av_unused p1 = p[-2 * stride];
    int av_unused p0 = p[-1 * stride];
    int av_unused q0 = p[ 0 * stride];
    int av_unused q1 = p[ 1 * stride];

    return 2 * FFABS(p0 - q0) + (FFABS(p1 - q1) >> 1) <= flim;
}

static av_always_inline int hev(uint8_t *p, ptrdiff_t stride, int thresh)
{
    int av_unused p1 = p[-2 * stride];
    int av_unused p0 = p[-1 * stride];
    int av_unused q0 = p[ 0 * stride];
    int av_unused q1 = p[ 1 * stride];

    return FFABS(p1 - p0) > thresh || FFABS(q1 - q0) > thresh;
}

static av_always_inline void filter_mbedge(uint8_t *p, ptrdiff_t stride)
{
    int a0, a1, a2, w;
    const uint8_t *cm = ff_crop_tab + MAX_NEG_CROP;

    int av_unused p2 = p[-3 * stride];
    int av_unused p1 = p[-2 * stride];
    int av_unused p0 = p[-1 * stride];
    int av_unused q0 = p[ 0 * stride];
    int av_unused q1 = p[ 1 * stride];
    int av_unused q2 = p[ 2 * stride];

    w = clip_int8(p1 - q1);
    w = clip_int8(w + 3 * (q0 - p0));

    a0 = (27 * w + 63) >> 7;
    a1 = (18 * w + 63) >> 7;
    a2 =  (9 * w + 63) >> 7;

    p[-3 * stride] = cm[p2 + a2];
    p[-2 * stride] = cm[p1 + a1];
    p[-1 * stride] = cm[p0 + a0];
    p[ 0 * stride] = cm[q0 - a0];
    p[ 1 * stride] = cm[q1 - a1];
    p[ 2 * stride] = cm[q2 - a2];
}

static av_always_inline int vp8_normal_limit(uint8_t *p, ptrdiff_t stride,
        int E, int I)
{
    int av_unused p3 = p[-4 * stride];
    int av_unused p2 = p[-3 * stride];
    int av_unused p1 = p[-2 * stride];
    int av_unused p0 = p[-1 * stride];
    int av_unused q0 = p[ 0 * stride];
    int av_unused q1 = p[ 1 * stride];
    int av_unused q2 = p[ 2 * stride];
    int av_unused q3 = p[ 3 * stride];

    return vp8_simple_limit(p, stride, E) &&
           FFABS(p3 - p2) <= I && FFABS(p2 - p1) <= I &&
           FFABS(p1 - p0) <= I && FFABS(q3 - q2) <= I &&
           FFABS(q2 - q1) <= I && FFABS(q1 - q0) <= I;
}

static av_always_inline void vp8_v_loop_filter8_mmi(uint8_t *dst,
        ptrdiff_t stride, int flim_E, int flim_I, int hev_thresh)
{
    int i;

    for (i = 0; i < 8; i++)
        if (vp8_normal_limit(dst + i * 1, stride, flim_E, flim_I)) {
            if (hev(dst + i * 1, stride, hev_thresh))
                vp8_filter_common_is4tap(dst + i * 1, stride);
            else
                filter_mbedge(dst + i * 1, stride);
        }
}

static av_always_inline void vp8_v_loop_filter8_inner_mmi(uint8_t *dst,
        ptrdiff_t stride, int flim_E, int flim_I, int hev_thresh)
{
    int i;

    for (i = 0; i < 8; i++)
        if (vp8_normal_limit(dst + i * 1, stride, flim_E, flim_I)) {
            int hv = hev(dst + i * 1, stride, hev_thresh);
            if (hv)
                vp8_filter_common_is4tap(dst + i * 1, stride);
            else
                vp8_filter_common_isnot4tap(dst + i * 1, stride);
        }
}

static av_always_inline void vp8_h_loop_filter8_mmi(uint8_t *dst,
        ptrdiff_t stride, int flim_E, int flim_I, int hev_thresh)
{
    int i;

    for (i = 0; i < 8; i++)
        if (vp8_normal_limit(dst + i * stride, 1, flim_E, flim_I)) {
            if (hev(dst + i * stride, 1, hev_thresh))
                vp8_filter_common_is4tap(dst + i * stride, 1);
            else
                filter_mbedge(dst + i * stride, 1);
        }
}

static av_always_inline void vp8_h_loop_filter8_inner_mmi(uint8_t *dst,
        ptrdiff_t stride, int flim_E, int flim_I, int hev_thresh)
{
    int i;

    for (i = 0; i < 8; i++)
        if (vp8_normal_limit(dst + i * stride, 1, flim_E, flim_I)) {
            int hv = hev(dst + i * stride, 1, hev_thresh);
            if (hv)
                vp8_filter_common_is4tap(dst + i * stride, 1);
            else
                vp8_filter_common_isnot4tap(dst + i * stride, 1);
        }
}

void ff_vp8_luma_dc_wht_mmi(int16_t block[4][4][16], int16_t dc[16])
{
#if OK
    double ftmp[8];

    __asm__ volatile (
        "gsldlc1    %[ftmp0],   0x07(%[dc])                         \n\t"
        "gsldrc1    %[ftmp0],   0x00(%[dc])                         \n\t"
        "gsldlc1    %[ftmp1],   0x0f(%[dc])                         \n\t"
        "gsldrc1    %[ftmp1],   0x08(%[dc])                         \n\t"
        "gsldlc1    %[ftmp2],   0x17(%[dc])                         \n\t"
        "gsldrc1    %[ftmp2],   0x10(%[dc])                         \n\t"
        "gsldlc1    %[ftmp3],   0x1f(%[dc])                         \n\t"
        "gsldrc1    %[ftmp3],   0x18(%[dc])                         \n\t"
        "paddsh     %[ftmp4],   %[ftmp0],       %[ftmp3]            \n\t"
        "psubsh     %[ftmp5],   %[ftmp0],       %[ftmp3]            \n\t"
        "paddsh     %[ftmp6],   %[ftmp1],       %[ftmp2]            \n\t"
        "psubsh     %[ftmp7],   %[ftmp1],       %[ftmp2]            \n\t"
        "paddsh     %[ftmp0],   %[ftmp4],       %[ftmp6]            \n\t"
        "paddsh     %[ftmp1],   %[ftmp5],       %[ftmp7]            \n\t"
        "psubsh     %[ftmp2],   %[ftmp4],       %[ftmp6]            \n\t"
        "psubsh     %[ftmp3],   %[ftmp5],       %[ftmp7]            \n\t"
        "gssdlc1    %[ftmp0],   0x07(%[dc])                         \n\t"
        "gssdrc1    %[ftmp0],   0x00(%[dc])                         \n\t"
        "gssdlc1    %[ftmp1],   0x0f(%[dc])                         \n\t"
        "gssdrc1    %[ftmp1],   0x08(%[dc])                         \n\t"
        "gssdlc1    %[ftmp2],   0x17(%[dc])                         \n\t"
        "gssdrc1    %[ftmp2],   0x10(%[dc])                         \n\t"
        "gssdlc1    %[ftmp3],   0x1f(%[dc])                         \n\t"
        "gssdrc1    %[ftmp3],   0x18(%[dc])                         \n\t"
        : [ftmp0]"=&f"(ftmp[0]),            [ftmp1]"=&f"(ftmp[1]),
          [ftmp2]"=&f"(ftmp[2]),            [ftmp3]"=&f"(ftmp[3]),
          [ftmp4]"=&f"(ftmp[4]),            [ftmp5]"=&f"(ftmp[5]),
          [ftmp6]"=&f"(ftmp[6]),            [ftmp7]"=&f"(ftmp[7])
        : [dc]"r"((uint8_t*)dc)
        : "memory"
    );

    block[0][0][0] = (dc[0] + dc[3] + 3 + dc[1] + dc[2]) >> 3;
    block[0][1][0] = (dc[0] - dc[3] + 3 + dc[1] - dc[2]) >> 3;
    block[0][2][0] = (dc[0] + dc[3] + 3 - dc[1] - dc[2]) >> 3;
    block[0][3][0] = (dc[0] - dc[3] + 3 - dc[1] + dc[2]) >> 3;

    block[1][0][0] = (dc[4] + dc[7] + 3 + dc[5] + dc[6]) >> 3;
    block[1][1][0] = (dc[4] - dc[7] + 3 + dc[5] - dc[6]) >> 3;
    block[1][2][0] = (dc[4] + dc[7] + 3 - dc[5] - dc[6]) >> 3;
    block[1][3][0] = (dc[4] - dc[7] + 3 - dc[5] + dc[6]) >> 3;

    block[2][0][0] = (dc[8] + dc[11] + 3 + dc[9] + dc[10]) >> 3;
    block[2][1][0] = (dc[8] - dc[11] + 3 + dc[9] - dc[10]) >> 3;
    block[2][2][0] = (dc[8] + dc[11] + 3 - dc[9] - dc[10]) >> 3;
    block[2][3][0] = (dc[8] - dc[11] + 3 - dc[9] + dc[10]) >> 3;

    block[3][0][0] = (dc[12] + dc[15] + 3 + dc[13] + dc[14]) >> 3;
    block[3][1][0] = (dc[12] - dc[15] + 3 + dc[13] - dc[14]) >> 3;
    block[3][2][0] = (dc[12] + dc[15] + 3 - dc[13] - dc[14]) >> 3;
    block[3][3][0] = (dc[12] - dc[15] + 3 - dc[13] + dc[14]) >> 3;

    __asm__ volatile (
        "xor        %[ftmp0],   %[ftmp0],       %[ftmp0]            \n\t"
        "gssdlc1    %[ftmp0],   0x07(%[dc])                         \n\t"
        "gssdrc1    %[ftmp0],   0x00(%[dc])                         \n\t"
        "gssdlc1    %[ftmp0],   0x0f(%[dc])                         \n\t"
        "gssdrc1    %[ftmp0],   0x08(%[dc])                         \n\t"
        "gssdlc1    %[ftmp0],   0x17(%[dc])                         \n\t"
        "gssdrc1    %[ftmp0],   0x10(%[dc])                         \n\t"
        "gssdlc1    %[ftmp0],   0x1f(%[dc])                         \n\t"
        "gssdrc1    %[ftmp0],   0x18(%[dc])                         \n\t"
        : [ftmp0]"=&f"(ftmp[0])
        : [dc]"r"((uint8_t *)dc)
        : "memory"
    );
#else
    int t00, t01, t02, t03, t10, t11, t12, t13, t20, t21, t22, t23, t30, t31, t32, t33;

    t00 = dc[0] + dc[12];
    t10 = dc[1] + dc[13];
    t20 = dc[2] + dc[14];
    t30 = dc[3] + dc[15];

    t03 = dc[0] - dc[12];
    t13 = dc[1] - dc[13];
    t23 = dc[2] - dc[14];
    t33 = dc[3] - dc[15];

    t01 = dc[4] + dc[ 8];
    t11 = dc[5] + dc[ 9];
    t21 = dc[6] + dc[10];
    t31 = dc[7] + dc[11];

    t02 = dc[4] - dc[ 8];
    t12 = dc[5] - dc[ 9];
    t22 = dc[6] - dc[10];
    t32 = dc[7] - dc[11];

    dc[ 0] = t00 + t01;
    dc[ 1] = t10 + t11;
    dc[ 2] = t20 + t21;
    dc[ 3] = t30 + t31;

    dc[ 4] = t03 + t02;
    dc[ 5] = t13 + t12;
    dc[ 6] = t23 + t22;
    dc[ 7] = t33 + t32;

    dc[ 8] = t00 - t01;
    dc[ 9] = t10 - t11;
    dc[10] = t20 - t21;
    dc[11] = t30 - t31;

    dc[12] = t03 - t02;
    dc[13] = t13 - t12;
    dc[14] = t23 - t22;
    dc[15] = t33 - t32;

    block[0][0][0] = (dc[0] + dc[3] + 3 + dc[1] + dc[2]) >> 3;
    block[0][1][0] = (dc[0] - dc[3] + 3 + dc[1] - dc[2]) >> 3;
    block[0][2][0] = (dc[0] + dc[3] + 3 - dc[1] - dc[2]) >> 3;
    block[0][3][0] = (dc[0] - dc[3] + 3 - dc[1] + dc[2]) >> 3;

    block[1][0][0] = (dc[4] + dc[7] + 3 + dc[5] + dc[6]) >> 3;
    block[1][1][0] = (dc[4] - dc[7] + 3 + dc[5] - dc[6]) >> 3;
    block[1][2][0] = (dc[4] + dc[7] + 3 - dc[5] - dc[6]) >> 3;
    block[1][3][0] = (dc[4] - dc[7] + 3 - dc[5] + dc[6]) >> 3;

    block[2][0][0] = (dc[8] + dc[11] + 3 + dc[9] + dc[10]) >> 3;
    block[2][1][0] = (dc[8] - dc[11] + 3 + dc[9] - dc[10]) >> 3;
    block[2][2][0] = (dc[8] + dc[11] + 3 - dc[9] - dc[10]) >> 3;
    block[2][3][0] = (dc[8] - dc[11] + 3 - dc[9] + dc[10]) >> 3;

    block[3][0][0] = (dc[12] + dc[15] + 3 + dc[13] + dc[14]) >> 3;
    block[3][1][0] = (dc[12] - dc[15] + 3 + dc[13] - dc[14]) >> 3;
    block[3][2][0] = (dc[12] + dc[15] + 3 - dc[13] - dc[14]) >> 3;
    block[3][3][0] = (dc[12] - dc[15] + 3 - dc[13] + dc[14]) >> 3;

    AV_ZERO64(dc + 0);
    AV_ZERO64(dc + 4);
    AV_ZERO64(dc + 8);
    AV_ZERO64(dc + 12);
#endif
}

void ff_vp8_luma_dc_wht_dc_mmi(int16_t block[4][4][16], int16_t dc[16])
{
    int val = (dc[0] + 3) >> 3;

    dc[0] = 0;

    block[0][0][0] = val;
    block[0][1][0] = val;
    block[0][2][0] = val;
    block[0][3][0] = val;
    block[1][0][0] = val;
    block[1][1][0] = val;
    block[1][2][0] = val;
    block[1][3][0] = val;
    block[2][0][0] = val;
    block[2][1][0] = val;
    block[2][2][0] = val;
    block[2][3][0] = val;
    block[3][0][0] = val;
    block[3][1][0] = val;
    block[3][2][0] = val;
    block[3][3][0] = val;
}

#define MUL_20091(a) ((((a) * 20091) >> 16) + (a))
#define MUL_35468(a)  (((a) * 35468) >> 16)
void ff_vp8_idct_add_mmi(uint8_t *dst, int16_t block[16], ptrdiff_t stride)
{
#if NOTOK //FIXME fate-vp8-test-vector-xxx
    DECLARE_ALIGNED(8, const uint64_t, ff_ph_20091) = {0x4e7b4e7b4e7b4e7bULL};
    DECLARE_ALIGNED(8, const uint64_t, ff_ph_35468) = {0x8a8c8a8c8a8c8a8cULL};
    double ftmp[15];
    uint64_t tmp[1];

    __asm__ volatile (
        /*
        t0  = block[0] + block[ 8];
        t4  = block[1] + block[ 9];
        t8  = block[2] + block[10];
        t12 = block[3] + block[11];

        t1  = block[0] - block[ 8];
        t5  = block[1] - block[ 9];
        t9  = block[2] - block[10];
        t13 = block[3] - block[11];

        t2  = MUL_35468(block[4]) - MUL_20091(block[12]);
        t6  = MUL_35468(block[5]) - MUL_20091(block[13]);
        t10 = MUL_35468(block[6]) - MUL_20091(block[14]);
        t14 = MUL_35468(block[7]) - MUL_20091(block[15]);

        t3  = MUL_20091(block[4]) + MUL_35468(block[12]);
        t7  = MUL_20091(block[5]) + MUL_35468(block[13]);
        t11 = MUL_20091(block[6]) + MUL_35468(block[14]);
        t15 = MUL_20091(block[7]) + MUL_35468(block[15]);

        block[ 0] = 0;
        block[ 4] = 0;
        block[ 8] = 0;
        block[12] = 0;
        block[ 1] = 0;
        block[ 5] = 0;
        block[ 9] = 0;
        block[13] = 0;
        block[ 2] = 0;
        block[ 6] = 0;
        block[10] = 0;
        block[14] = 0;
        block[ 3] = 0;
        block[ 7] = 0;
        block[11] = 0;
        block[15] = 0;

        tmp[ 0] = t0  + t3;
        tmp[ 4] = t4  + t7;
        tmp[ 8] = t8  + t11;
        tmp[12] = t12 + t15;

        tmp[ 1] = t1  + t2;
        tmp[ 5] = t5  + t6;
        tmp[ 9] = t9  + t10;
        tmp[13] = t13 + t14;

        tmp[ 2] = t1  - t2;
        tmp[ 6] = t5  - t6;
        tmp[10] = t9  - t10;
        tmp[14] = t13 - t14;

        tmp[ 3] = t0  - t3;
        tmp[ 7] = t4  - t7;
        tmp[11] = t8  - t11;
        tmp[15] = t12 - t15;
        */

        "xor        %[ftmp0],   %[ftmp0],       %[ftmp0]                \n\t"
        "dmtc1      %[ff_ph_35468],             %[ftmp13]               \n\t"
        "dmtc1      %[ff_ph_20091],             %[ftmp14]               \n\t"
        "gsldlc1    %[ftmp1],   0x07(%[block])                          \n\t"
        "gsldrc1    %[ftmp1],   0x00(%[block])                          \n\t"
        "gsldlc1    %[ftmp2],   0x0f(%[block])                          \n\t"
        "gsldrc1    %[ftmp2],   0x08(%[block])                          \n\t"
        "gsldlc1    %[ftmp3],   0x17(%[block])                          \n\t"
        "gsldrc1    %[ftmp3],   0x10(%[block])                          \n\t"
        "gsldlc1    %[ftmp4],   0x1f(%[block])                          \n\t"
        "gsldrc1    %[ftmp4],   0x18(%[block])                          \n\t"

        "pmulhh     %[ftmp9],   %[ftmp2],       %[ftmp13]               \n\t"
        "pmulhh     %[ftmp10],  %[ftmp4],       %[ftmp14]               \n\t"
        "paddh      %[ftmp10],  %[ftmp10],      %[ftmp4]                \n\t"
        "pmulhh     %[ftmp11],  %[ftmp2],       %[ftmp14]               \n\t"
        "paddh      %[ftmp11],  %[ftmp11],      %[ftmp2]                \n\t"
        "pmulhh     %[ftmp12],  %[ftmp4],       %[ftmp13]               \n\t"

        "paddh      %[ftmp5],   %[ftmp1],       %[ftmp3]                \n\t"
        "psubh      %[ftmp6],   %[ftmp1],       %[ftmp3]                \n\t"
        "psubh      %[ftmp7],   %[ftmp9],       %[ftmp10]               \n\t"
        "paddh      %[ftmp8],   %[ftmp11],      %[ftmp12]               \n\t"

        "gssdlc1    %[ftmp0],   0x07(%[block])                          \n\t"
        "gssdrc1    %[ftmp0],   0x00(%[block])                          \n\t"
        "gssdlc1    %[ftmp0],   0x0f(%[block])                          \n\t"
        "gssdrc1    %[ftmp0],   0x08(%[block])                          \n\t"
        "gssdlc1    %[ftmp0],   0x17(%[block])                          \n\t"
        "gssdrc1    %[ftmp0],   0x10(%[block])                          \n\t"
        "gssdlc1    %[ftmp0],   0x1f(%[block])                          \n\t"
        "gssdrc1    %[ftmp0],   0x18(%[block])                          \n\t"

        "paddh     %[ftmp1],   %[ftmp5],       %[ftmp8]                 \n\t"
        "paddh     %[ftmp2],   %[ftmp6],       %[ftmp7]                 \n\t"
        "psubh     %[ftmp3],   %[ftmp6],       %[ftmp7]                 \n\t"
        "psubh     %[ftmp4],   %[ftmp5],       %[ftmp8]                 \n\t"

        TRANSPOSE_4H(%[ftmp1], %[ftmp2], %[ftmp3], %[ftmp4],
                     %[ftmp5], %[ftmp6], %[ftmp7], %[ftmp8],
                     %[ftmp9], %[tmp0],  %[ftmp0], %[ftmp14])
        /*
        t0  = tmp[0] + tmp[ 8];
        t4  = tmp[1] + tmp[ 9];
        t8  = tmp[2] + tmp[10];
        t12 = tmp[3] + tmp[11];

        t1  = tmp[0] - tmp[ 8];
        t5  = tmp[1] - tmp[ 9];
        t9  = tmp[2] - tmp[10];
        t13 = tmp[3] - tmp[11];

        t2  = MUL_35468(tmp[4]) - MUL_20091(tmp[12]);
        t6  = MUL_35468(tmp[5]) - MUL_20091(tmp[13]);
        t10 = MUL_35468(tmp[6]) - MUL_20091(tmp[14]);
        t14 = MUL_35468(tmp[7]) - MUL_20091(tmp[15]);

        t3  = MUL_20091(tmp[4]) + MUL_35468(tmp[12]);
        t7  = MUL_20091(tmp[5]) + MUL_35468(tmp[13]);
        t11 = MUL_20091(tmp[6]) + MUL_35468(tmp[14]);
        t15 = MUL_20091(tmp[7]) + MUL_35468(tmp[15]);

        dst[0]          = av_clip_uint8(dst[0] +          ((t0  + t3  + 4) >> 3));
        dst[0+1*stride] = av_clip_uint8(dst[0+1*stride] + ((t4  + t7  + 4) >> 3));
        dst[0+2*stride] = av_clip_uint8(dst[0+2*stride] + ((t8  + t11 + 4) >> 3));
        dst[0+3*stride] = av_clip_uint8(dst[0+3*stride] + ((t12 + t15 + 4) >> 3));

        dst[1]          = av_clip_uint8(dst[1] +          ((t1  + t2  + 4) >> 3));
        dst[1+1*stride] = av_clip_uint8(dst[1+1*stride] + ((t5  + t6  + 4) >> 3));
        dst[1+2*stride] = av_clip_uint8(dst[1+2*stride] + ((t9  + t10 + 4) >> 3));
        dst[1+3*stride] = av_clip_uint8(dst[1+3*stride] + ((t13 + t14 + 4) >> 3));

        dst[2]          = av_clip_uint8(dst[2] +          ((t1  - t2  + 4) >> 3));
        dst[2+1*stride] = av_clip_uint8(dst[2+1*stride] + ((t5  - t6  + 4) >> 3));
        dst[2+2*stride] = av_clip_uint8(dst[2+2*stride] + ((t9  - t10 + 4) >> 3));
        dst[2+3*stride] = av_clip_uint8(dst[2+3*stride] + ((t13 - t14 + 4) >> 3));

        dst[3]          = av_clip_uint8(dst[3] +          ((t0  - t3  + 4) >> 3));
        dst[3+1*stride] = av_clip_uint8(dst[3+1*stride] + ((t4  - t7  + 4) >> 3));
        dst[3+2*stride] = av_clip_uint8(dst[3+2*stride] + ((t8  - t11 + 4) >> 3));
        dst[3+3*stride] = av_clip_uint8(dst[3+3*stride] + ((t12 - t15 + 4) >> 3));
        */

        "li         %[tmp0],    0x03                                    \n\t"
        "dmtc1      %[ff_ph_35468],             %[ftmp0]                \n\t"
        "dmtc1      %[ff_ph_20091],             %[ftmp14]               \n\t"
        "paddh      %[ftmp5],   %[ftmp1],       %[ftmp3]                \n\t"
        "psubh      %[ftmp6],   %[ftmp1],       %[ftmp3]                \n\t"
        "pmulhh     %[ftmp7],   %[ftmp2],       %[ftmp0]                \n\t"
        "pmulhh     %[ftmp8],   %[ftmp4],       %[ftmp14]               \n\t"
        "paddh      %[ftmp8],   %[ftmp8],       %[ftmp4]                \n\t"
        "psubh      %[ftmp7],   %[ftmp7],       %[ftmp8]                \n\t"
        "pmulhh     %[ftmp8],   %[ftmp2],       %[ftmp14]               \n\t"
        "paddh      %[ftmp8],   %[ftmp8],       %[ftmp2]                \n\t"
        "pmulhh     %[ftmp9],   %[ftmp4],       %[ftmp0]                \n\t"
        "paddh      %[ftmp8],   %[ftmp8],       %[ftmp9]                \n\t"

        "mtc1       %[tmp0],    %[ftmp0]                                \n\t"
        "paddh      %[ftmp10],  %[ftmp5],       %[ftmp8]                \n\t"
        "paddh      %[ftmp10],  %[ftmp10],      %[ff_pw_4]              \n\t"
        "psrah      %[ftmp10],  %[ftmp10],      %[ftmp0]                \n\t"

        "paddh      %[ftmp11],  %[ftmp6],       %[ftmp7]                \n\t"
        "paddh      %[ftmp11],  %[ftmp11],      %[ff_pw_4]              \n\t"
        "psrah      %[ftmp11],  %[ftmp11],      %[ftmp0]                \n\t"

        "psubh      %[ftmp12],  %[ftmp6],       %[ftmp7]                \n\t"
        "paddh      %[ftmp12],  %[ftmp12],      %[ff_pw_4]              \n\t"
        "psrah      %[ftmp12],  %[ftmp12],      %[ftmp0]                \n\t"

        "psubh      %[ftmp13],  %[ftmp5],       %[ftmp8]                \n\t"
        "paddh      %[ftmp13],  %[ftmp13],      %[ff_pw_4]              \n\t"
        "psrah      %[ftmp13],  %[ftmp13],      %[ftmp0]                \n\t"

        "xor        %[ftmp0],   %[ftmp0],       %[ftmp0]                \n\t"
        "gslwlc1    %[ftmp1],   0x03(%[dst0])                           \n\t"
        "gslwrc1    %[ftmp1],   0x00(%[dst0])                           \n\t"
        "gslwlc1    %[ftmp2],   0x03(%[dst1])                           \n\t"
        "gslwrc1    %[ftmp2],   0x00(%[dst1])                           \n\t"
        "gslwlc1    %[ftmp3],   0x03(%[dst2])                           \n\t"
        "gslwrc1    %[ftmp3],   0x00(%[dst2])                           \n\t"
        "gslwlc1    %[ftmp4],   0x03(%[dst3])                           \n\t"
        "gslwrc1    %[ftmp4],   0x00(%[dst3])                           \n\t"
        "punpcklbh  %[ftmp1],   %[ftmp1],       %[ftmp0]                \n\t"
        "punpcklbh  %[ftmp2],   %[ftmp2],       %[ftmp0]                \n\t"
        "punpcklbh  %[ftmp3],   %[ftmp3],       %[ftmp0]                \n\t"
        "punpcklbh  %[ftmp4],   %[ftmp4],       %[ftmp0]                \n\t"

        TRANSPOSE_4H(%[ftmp1], %[ftmp2], %[ftmp3], %[ftmp4],
                     %[ftmp5], %[ftmp6], %[ftmp7], %[ftmp8],
                     %[ftmp9], %[tmp0],  %[ftmp0], %[ftmp14])

        "paddh      %[ftmp1],   %[ftmp1],       %[ftmp10]               \n\t"
        "paddh      %[ftmp2],   %[ftmp2],       %[ftmp11]               \n\t"
        "paddh      %[ftmp3],   %[ftmp3],       %[ftmp12]               \n\t"
        "paddh      %[ftmp4],   %[ftmp4],       %[ftmp13]               \n\t"

        TRANSPOSE_4H(%[ftmp1], %[ftmp2], %[ftmp3], %[ftmp4],
                     %[ftmp5], %[ftmp6], %[ftmp7], %[ftmp8],
                     %[ftmp9], %[tmp0],  %[ftmp0], %[ftmp14])

        "packushb   %[ftmp1],   %[ftmp1],       %[ftmp0]                \n\t"
        "packushb   %[ftmp2],   %[ftmp2],       %[ftmp0]                \n\t"
        "packushb   %[ftmp3],   %[ftmp3],       %[ftmp0]                \n\t"
        "packushb   %[ftmp4],   %[ftmp4],       %[ftmp0]                \n\t"
        "gsswlc1    %[ftmp1],   0x03(%[dst0])                           \n\t"
        "gsswrc1    %[ftmp1],   0x00(%[dst0])                           \n\t"
        "gsswlc1    %[ftmp2],   0x03(%[dst1])                           \n\t"
        "gsswrc1    %[ftmp2],   0x00(%[dst1])                           \n\t"
        "gsswlc1    %[ftmp3],   0x03(%[dst2])                           \n\t"
        "gsswrc1    %[ftmp3],   0x00(%[dst2])                           \n\t"
        "gsswlc1    %[ftmp4],   0x03(%[dst3])                           \n\t"
        "gsswrc1    %[ftmp4],   0x00(%[dst3])                           \n\t"
        : [ftmp0]"=&f"(ftmp[0]),            [ftmp1]"=&f"(ftmp[1]),
          [ftmp2]"=&f"(ftmp[2]),            [ftmp3]"=&f"(ftmp[3]),
          [ftmp4]"=&f"(ftmp[4]),            [ftmp5]"=&f"(ftmp[5]),
          [ftmp6]"=&f"(ftmp[6]),            [ftmp7]"=&f"(ftmp[7]),
          [ftmp8]"=&f"(ftmp[8]),            [ftmp9]"=&f"(ftmp[9]),
          [ftmp10]"=&f"(ftmp[10]),          [ftmp11]"=&f"(ftmp[11]),
          [ftmp12]"=&f"(ftmp[12]),          [ftmp13]"=&f"(ftmp[13]),
          [ftmp14]"=&f"(ftmp[14]),
          [tmp0]"=&r"(tmp[0])
        : [dst0]"r"(dst),                   [dst1]"r"(dst+stride),
          [dst2]"r"(dst+2*stride),          [dst3]"r"(dst+3*stride),
          [block]"r"(block),                [ff_pw_4]"f"(ff_pw_4),
          [ff_ph_20091]"r"(ff_ph_20091),    [ff_ph_35468]"r"(ff_ph_35468)
        : "memory"
    );
#else
    int i, t0, t1, t2, t3;
    int16_t tmp[16];

    for (i = 0; i < 4; i++) {
        t0 = block[0 + i] + block[8 + i];
        t1 = block[0 + i] - block[8 + i];
        t2 = MUL_35468(block[4 + i]) - MUL_20091(block[12 + i]);
        t3 = MUL_20091(block[4 + i]) + MUL_35468(block[12 + i]);
        block[ 0 + i] = 0;
        block[ 4 + i] = 0;
        block[ 8 + i] = 0;
        block[12 + i] = 0;

        tmp[i * 4 + 0] = t0 + t3;
        tmp[i * 4 + 1] = t1 + t2;
        tmp[i * 4 + 2] = t1 - t2;
        tmp[i * 4 + 3] = t0 - t3;
    }

    for (i = 0; i < 4; i++) {
        t0 = tmp[0 + i] + tmp[8 + i];
        t1 = tmp[0 + i] - tmp[8 + i];
        t2 = MUL_35468(tmp[4 + i]) - MUL_20091(tmp[12 + i]);
        t3 = MUL_20091(tmp[4 + i]) + MUL_35468(tmp[12 + i]);

        dst[0] = av_clip_uint8(dst[0] + ((t0 + t3 + 4) >> 3));
        dst[1] = av_clip_uint8(dst[1] + ((t1 + t2 + 4) >> 3));
        dst[2] = av_clip_uint8(dst[2] + ((t1 - t2 + 4) >> 3));
        dst[3] = av_clip_uint8(dst[3] + ((t0 - t3 + 4) >> 3));
        dst   += stride;
    }
#endif
}

void ff_vp8_idct_dc_add_mmi(uint8_t *dst, int16_t block[16], ptrdiff_t stride)
{
#if OK
    int dc = (block[0] + 4) >> 3;
    double ftmp[6];

    block[0] = 0;

    __asm__ volatile (
        "xor        %[ftmp0],   %[ftmp0],       %[ftmp0]             \n\t"
        "dmtc1      %[dc],      %[ftmp5]                             \n\t"
        "gslwlc1    %[ftmp1],   0x03(%[dst0])                        \n\t"
        "gslwrc1    %[ftmp1],   0x00(%[dst0])                        \n\t"
        "gslwlc1    %[ftmp2],   0x03(%[dst1])                        \n\t"
        "gslwrc1    %[ftmp2],   0x00(%[dst1])                        \n\t"
        "gslwlc1    %[ftmp3],   0x03(%[dst2])                        \n\t"
        "gslwrc1    %[ftmp3],   0x00(%[dst2])                        \n\t"
        "gslwlc1    %[ftmp4],   0x03(%[dst3])                        \n\t"
        "gslwrc1    %[ftmp4],   0x00(%[dst3])                        \n\t"
        "pshufh     %[ftmp5],   %[ftmp5],       %[ftmp0]             \n\t"
        "punpcklbh  %[ftmp1],   %[ftmp1],       %[ftmp0]             \n\t"
        "punpcklbh  %[ftmp2],   %[ftmp2],       %[ftmp0]             \n\t"
        "punpcklbh  %[ftmp3],   %[ftmp3],       %[ftmp0]             \n\t"
        "punpcklbh  %[ftmp4],   %[ftmp4],       %[ftmp0]             \n\t"
        "paddsh     %[ftmp1],   %[ftmp1],       %[ftmp5]             \n\t"
        "paddsh     %[ftmp2],   %[ftmp2],       %[ftmp5]             \n\t"
        "paddsh     %[ftmp3],   %[ftmp3],       %[ftmp5]             \n\t"
        "paddsh     %[ftmp4],   %[ftmp4],       %[ftmp5]             \n\t"
        "packushb   %[ftmp1],   %[ftmp1],       %[ftmp0]             \n\t"
        "packushb   %[ftmp2],   %[ftmp2],       %[ftmp0]             \n\t"
        "packushb   %[ftmp3],   %[ftmp3],       %[ftmp0]             \n\t"
        "packushb   %[ftmp4],   %[ftmp4],       %[ftmp0]             \n\t"
        "gsswlc1    %[ftmp1],   0x03(%[dst0])                        \n\t"
        "gsswrc1    %[ftmp1],   0x00(%[dst0])                        \n\t"
        "gsswlc1    %[ftmp2],   0x03(%[dst1])                        \n\t"
        "gsswrc1    %[ftmp2],   0x00(%[dst1])                        \n\t"
        "gsswlc1    %[ftmp3],   0x03(%[dst2])                        \n\t"
        "gsswrc1    %[ftmp3],   0x00(%[dst2])                        \n\t"
        "gsswlc1    %[ftmp4],   0x03(%[dst3])                        \n\t"
        "gsswrc1    %[ftmp4],   0x00(%[dst3])                        \n\t"
        : [ftmp0]"=&f"(ftmp[0]),            [ftmp1]"=&f"(ftmp[1]),
          [ftmp2]"=&f"(ftmp[2]),            [ftmp3]"=&f"(ftmp[3]),
          [ftmp4]"=&f"(ftmp[4]),            [ftmp5]"=&f"(ftmp[5])
        : [dst0]"r"(dst),                   [dst1]"r"(dst+stride),
          [dst2]"r"(dst+2*stride),          [dst3]"r"(dst+3*stride),
          [dc]"r"(dc)
        : "memory"
    );
#else
    int i, dc = (block[0] + 4) >> 3;

    block[0] = 0;

    for (i = 0; i < 4; i++) {
        dst[0] = av_clip_uint8(dst[0] + dc);
        dst[1] = av_clip_uint8(dst[1] + dc);
        dst[2] = av_clip_uint8(dst[2] + dc);
        dst[3] = av_clip_uint8(dst[3] + dc);
        dst   += stride;
    }
#endif
}

void ff_vp8_idct_dc_add4y_mmi(uint8_t *dst, int16_t block[4][16],
        ptrdiff_t stride)
{
    ff_vp8_idct_dc_add_mmi(dst +  0, block[0], stride);
    ff_vp8_idct_dc_add_mmi(dst +  4, block[1], stride);
    ff_vp8_idct_dc_add_mmi(dst +  8, block[2], stride);
    ff_vp8_idct_dc_add_mmi(dst + 12, block[3], stride);
}

void ff_vp8_idct_dc_add4uv_mmi(uint8_t *dst, int16_t block[4][16],
        ptrdiff_t stride)
{
    ff_vp8_idct_dc_add_mmi(dst + stride * 0 + 0, block[0], stride);
    ff_vp8_idct_dc_add_mmi(dst + stride * 0 + 4, block[1], stride);
    ff_vp8_idct_dc_add_mmi(dst + stride * 4 + 0, block[2], stride);
    ff_vp8_idct_dc_add_mmi(dst + stride * 4 + 4, block[3], stride);
}

// loop filter applied to edges between macroblocks
void ff_vp8_v_loop_filter16_mmi(uint8_t *dst, ptrdiff_t stride, int flim_E,
        int flim_I, int hev_thresh)
{
    int i;

    for (i = 0; i < 16; i++)
        if (vp8_normal_limit(dst + i * 1, stride, flim_E, flim_I)) {
            if (hev(dst + i * 1, stride, hev_thresh))
                vp8_filter_common_is4tap(dst + i * 1, stride);
            else
                filter_mbedge(dst + i * 1, stride);
        }
}

void ff_vp8_h_loop_filter16_mmi(uint8_t *dst, ptrdiff_t stride, int flim_E,
        int flim_I, int hev_thresh)
{
    int i;

    for (i = 0; i < 16; i++)
        if (vp8_normal_limit(dst + i * stride, 1, flim_E, flim_I)) {
            if (hev(dst + i * stride, 1, hev_thresh))
                vp8_filter_common_is4tap(dst + i * stride, 1);
            else
                filter_mbedge(dst + i * stride, 1);
        }
}

void ff_vp8_v_loop_filter8uv_mmi(uint8_t *dstU, uint8_t *dstV, ptrdiff_t stride,
        int flim_E, int flim_I, int hev_thresh)
{
    vp8_v_loop_filter8_mmi(dstU, stride, flim_E, flim_I, hev_thresh);
    vp8_v_loop_filter8_mmi(dstV, stride, flim_E, flim_I, hev_thresh);
}

void ff_vp8_h_loop_filter8uv_mmi(uint8_t *dstU, uint8_t *dstV, ptrdiff_t stride,
        int flim_E, int flim_I, int hev_thresh)
{
    vp8_h_loop_filter8_mmi(dstU, stride, flim_E, flim_I, hev_thresh);
    vp8_h_loop_filter8_mmi(dstV, stride, flim_E, flim_I, hev_thresh);
}

// loop filter applied to inner macroblock edges
void ff_vp8_v_loop_filter16_inner_mmi(uint8_t *dst, ptrdiff_t stride,
        int flim_E, int flim_I, int hev_thresh)
{
    int i;

    for (i = 0; i < 16; i++)
        if (vp8_normal_limit(dst + i * 1, stride, flim_E, flim_I)) {
            int hv = hev(dst + i * 1, stride, hev_thresh);
            if (hv)
                vp8_filter_common_is4tap(dst + i * 1, stride);
            else
                vp8_filter_common_isnot4tap(dst + i * 1, stride);
        }
}

void ff_vp8_h_loop_filter16_inner_mmi(uint8_t *dst, ptrdiff_t stride,
        int flim_E, int flim_I, int hev_thresh)
{
    int i;

    for (i = 0; i < 16; i++)
        if (vp8_normal_limit(dst + i * stride, 1, flim_E, flim_I)) {
            int hv = hev(dst + i * stride, 1, hev_thresh);
            if (hv)
                vp8_filter_common_is4tap(dst + i * stride, 1);
            else
                vp8_filter_common_isnot4tap(dst + i * stride, 1);
        }
}

void ff_vp8_v_loop_filter8uv_inner_mmi(uint8_t *dstU, uint8_t *dstV,
        ptrdiff_t stride, int flim_E, int flim_I, int hev_thresh)
{
    vp8_v_loop_filter8_inner_mmi(dstU, stride, flim_E, flim_I, hev_thresh);
    vp8_v_loop_filter8_inner_mmi(dstV, stride, flim_E, flim_I, hev_thresh);
}

void ff_vp8_h_loop_filter8uv_inner_mmi(uint8_t *dstU, uint8_t *dstV,
        ptrdiff_t stride, int flim_E, int flim_I, int hev_thresh)
{
    vp8_h_loop_filter8_inner_mmi(dstU, stride, flim_E, flim_I, hev_thresh);
    vp8_h_loop_filter8_inner_mmi(dstV, stride, flim_E, flim_I, hev_thresh);
}

void ff_vp8_v_loop_filter_simple_mmi(uint8_t *dst, ptrdiff_t stride, int flim)
{
    int i;

    for (i = 0; i < 16; i++)
        if (vp8_simple_limit(dst + i, stride, flim))
            vp8_filter_common_is4tap(dst + i, stride);
}

void ff_vp8_h_loop_filter_simple_mmi(uint8_t *dst, ptrdiff_t stride, int flim)
{
    int i;

    for (i = 0; i < 16; i++)
        if (vp8_simple_limit(dst + i * stride, 1, flim))
            vp8_filter_common_is4tap(dst + i * stride, 1);
}

void ff_put_vp8_pixels16_mmi(uint8_t *dst, ptrdiff_t dststride, uint8_t *src,
        ptrdiff_t srcstride, int h, int x, int y)
{
#if OK
    double ftmp[2];
    uint64_t tmp[2];
    mips_reg addr[2];

    __asm__ volatile (
        "1:                                                         \n\t"
        PTR_ADDU   "%[addr0],   %[src],         %[srcstride]        \n\t"
        "gsldlc1    %[ftmp0],   0x07(%[src])                        \n\t"
        "gsldrc1    %[ftmp0],   0x00(%[src])                        \n\t"
        "ldl        %[tmp0],    0x0f(%[src])                        \n\t"
        "ldr        %[tmp0],    0x08(%[src])                        \n\t"
        "gsldlc1    %[ftmp1],   0x07(%[addr0])                      \n\t"
        "gsldrc1    %[ftmp1],   0x00(%[addr0])                      \n\t"
        "ldl        %[tmp1],    0x0f(%[addr0])                      \n\t"
        "ldr        %[tmp1],    0x08(%[addr0])                      \n\t"
        PTR_ADDU   "%[addr1],   %[dst],         %[dststride]        \n\t"
        "gssdlc1    %[ftmp0],   0x07(%[dst])                        \n\t"
        "gssdrc1    %[ftmp0],   0x00(%[dst])                        \n\t"
        "sdl        %[tmp0],    0x0f(%[dst])                        \n\t"
        "sdr        %[tmp0],    0x08(%[dst])                        \n\t"
        "daddi      %[h],       %[h],           -0x02               \n\t"
        "gssdlc1    %[ftmp1],   0x07(%[addr1])                      \n\t"
        "gssdrc1    %[ftmp1],   0x00(%[addr1])                      \n\t"
        PTR_ADDU   "%[src],     %[addr0],       %[srcstride]        \n\t"
        "sdl        %[tmp1],    0x0f(%[addr1])                      \n\t"
        "sdr        %[tmp1],    0x08(%[addr1])                      \n\t"
        PTR_ADDU   "%[dst],     %[addr1],       %[dststride]        \n\t"
        "bnez       %[h],       1b                                  \n\t"
        : [ftmp0]"=&f"(ftmp[0]),            [ftmp1]"=&f"(ftmp[1]),
          [tmp0]"=&r"(tmp[0]),              [tmp1]"=&r"(tmp[1]),
          [addr0]"=&r"(addr[0]),            [addr1]"=&r"(addr[1]),
          [dst]"+&r"(dst),                  [src]"+&r"(src),
          [h]"+&r"(h)
        : [dststride]"r"((mips_reg)dststride),
          [srcstride]"r"((mips_reg)srcstride)
        : "memory"
    );
#else
    int i;

    for (i = 0; i < h; i++, dst += dststride, src += srcstride)
        memcpy(dst, src, 16);
#endif
}

void ff_put_vp8_pixels8_mmi(uint8_t *dst, ptrdiff_t dststride, uint8_t *src,
        ptrdiff_t srcstride, int h, int x, int y)
{
#if OK
    double ftmp[1];
    uint64_t tmp[1];
    mips_reg addr[2];

    __asm__ volatile (
        "1:                                                         \n\t"
        PTR_ADDU   "%[addr0],   %[src],         %[srcstride]        \n\t"
        "gsldlc1    %[ftmp0],   0x07(%[src])                        \n\t"
        "gsldrc1    %[ftmp0],   0x00(%[src])                        \n\t"
        "ldl        %[tmp0],    0x07(%[addr0])                      \n\t"
        "ldr        %[tmp0],    0x00(%[addr0])                      \n\t"
        PTR_ADDU   "%[addr1],   %[dst],         %[dststride]        \n\t"
        "gssdlc1    %[ftmp0],   0x07(%[dst])                        \n\t"
        "gssdrc1    %[ftmp0],   0x00(%[dst])                        \n\t"
        "daddi      %[h],       %[h],           -0x02               \n\t"
        "sdl        %[tmp0],    0x07(%[addr1])                      \n\t"
        "sdr        %[tmp0],    0x00(%[addr1])                      \n\t"
        PTR_ADDU   "%[src],     %[addr0],       %[srcstride]        \n\t"
        PTR_ADDU   "%[dst],     %[addr1],       %[dststride]        \n\t"
        "bnez       %[h],       1b                                  \n\t"
        : [ftmp0]"=&f"(ftmp[0]),            [tmp0]"=&r"(tmp[0]),
          [addr0]"=&r"(addr[0]),            [addr1]"=&r"(addr[1]),
          [dst]"+&r"(dst),                  [src]"+&r"(src),
          [h]"+&r"(h)
        : [dststride]"r"((mips_reg)dststride),
          [srcstride]"r"((mips_reg)srcstride)
        : "memory"
    );
#else
    int i;

    for (i = 0; i < h; i++, dst += dststride, src += srcstride)
        memcpy(dst, src, 8);
#endif
}

void ff_put_vp8_pixels4_mmi(uint8_t *dst, ptrdiff_t dststride, uint8_t *src,
        ptrdiff_t srcstride, int h, int x, int y)
{
#if OK
    double ftmp[1];
    uint64_t tmp[1];
    mips_reg addr[2];

    __asm__ volatile (
        "1:                                                         \n\t"
        PTR_ADDU   "%[addr0],   %[src],         %[srcstride]        \n\t"
        "gslwlc1    %[ftmp0],   0x03(%[src])                        \n\t"
        "gslwrc1    %[ftmp0],   0x00(%[src])                        \n\t"
        "lwl        %[tmp0],    0x03(%[addr0])                      \n\t"
        "lwr        %[tmp0],    0x00(%[addr0])                      \n\t"
        PTR_ADDU   "%[addr1],   %[dst],         %[dststride]        \n\t"
        "gsswlc1    %[ftmp0],   0x03(%[dst])                        \n\t"
        "gsswrc1    %[ftmp0],   0x00(%[dst])                        \n\t"
        "daddi      %[h],       %[h],           -0x02               \n\t"
        "swl        %[tmp0],    0x03(%[addr1])                      \n\t"
        "swr        %[tmp0],    0x00(%[addr1])                      \n\t"
        PTR_ADDU   "%[src],     %[addr0],       %[srcstride]        \n\t"
        PTR_ADDU   "%[dst],     %[addr1],       %[dststride]        \n\t"
        "bnez       %[h],       1b                                  \n\t"
        : [ftmp0]"=&f"(ftmp[0]),            [tmp0]"=&r"(tmp[0]),
          [addr0]"=&r"(addr[0]),            [addr1]"=&r"(addr[1]),
          [dst]"+&r"(dst),                  [src]"+&r"(src),
          [h]"+&r"(h)
        : [dststride]"r"((mips_reg)dststride),
          [srcstride]"r"((mips_reg)srcstride)
        : "memory"
    );
#else
    int i;

    for (i = 0; i < h; i++, dst += dststride, src += srcstride)
        memcpy(dst, src, 4);
#endif
}

static const uint8_t subpel_filters[7][6] = {
    { 0,  6, 123,  12,  1, 0 },
    { 2, 11, 108,  36,  8, 1 },
    { 0,  9,  93,  50,  6, 0 },
    { 3, 16,  77,  77, 16, 3 },
    { 0,  6,  50,  93,  9, 0 },
    { 1,  8,  36, 108, 11, 2 },
    { 0,  1,  12, 123,  6, 0 },
};

#define FILTER_6TAP(src, F, stride)                                           \
    cm[(F[2] * src[x + 0 * stride] - F[1] * src[x - 1 * stride] +             \
        F[0] * src[x - 2 * stride] + F[3] * src[x + 1 * stride] -             \
        F[4] * src[x + 2 * stride] + F[5] * src[x + 3 * stride] + 64) >> 7]

#define FILTER_4TAP(src, F, stride)                                           \
    cm[(F[2] * src[x + 0 * stride] - F[1] * src[x - 1 * stride] +             \
        F[3] * src[x + 1 * stride] - F[4] * src[x + 2 * stride] + 64) >> 7]

void ff_put_vp8_epel16_h4_mmi(uint8_t *dst, ptrdiff_t dststride, uint8_t *src,
        ptrdiff_t srcstride, int h, int mx, int my)
{
#if OK
    const uint8_t *filter = subpel_filters[mx - 1];
    int y;
    double ftmp[10];
    uint64_t tmp[1];

    for (y = 0; y < h; y++) {
        /*
        dst[0] = cm[(filter[2] * src[0] - filter[1] * src[-1] + filter[3] * src[1] - filter[4] * src[2] + 64) >> 7];
        dst[1] = cm[(filter[2] * src[1] - filter[1] * src[ 0] + filter[3] * src[2] - filter[4] * src[3] + 64) >> 7];
        dst[2] = cm[(filter[2] * src[2] - filter[1] * src[ 1] + filter[3] * src[3] - filter[4] * src[4] + 64) >> 7];
        dst[3] = cm[(filter[2] * src[3] - filter[1] * src[ 2] + filter[3] * src[4] - filter[4] * src[5] + 64) >> 7];
        dst[4] = cm[(filter[2] * src[4] - filter[1] * src[ 3] + filter[3] * src[5] - filter[4] * src[6] + 64) >> 7];
        dst[5] = cm[(filter[2] * src[5] - filter[1] * src[ 4] + filter[3] * src[6] - filter[4] * src[7] + 64) >> 7];
        dst[6] = cm[(filter[2] * src[6] - filter[1] * src[ 5] + filter[3] * src[7] - filter[4] * src[8] + 64) >> 7];
        dst[7] = cm[(filter[2] * src[7] - filter[1] * src[ 6] + filter[3] * src[8] - filter[4] * src[9] + 64) >> 7];

        dst[ 8] = cm[(filter[2] * src[ 8] - filter[1] * src[ 7] + filter[3] * src[ 9] - filter[4] * src[10] + 64) >> 7];
        dst[ 9] = cm[(filter[2] * src[ 9] - filter[1] * src[ 8] + filter[3] * src[10] - filter[4] * src[11] + 64) >> 7];
        dst[10] = cm[(filter[2] * src[10] - filter[1] * src[ 9] + filter[3] * src[11] - filter[4] * src[12] + 64) >> 7];
        dst[11] = cm[(filter[2] * src[11] - filter[1] * src[10] + filter[3] * src[12] - filter[4] * src[13] + 64) >> 7];
        dst[12] = cm[(filter[2] * src[12] - filter[1] * src[11] + filter[3] * src[13] - filter[4] * src[14] + 64) >> 7];
        dst[13] = cm[(filter[2] * src[13] - filter[1] * src[12] + filter[3] * src[14] - filter[4] * src[15] + 64) >> 7];
        dst[14] = cm[(filter[2] * src[14] - filter[1] * src[13] + filter[3] * src[15] - filter[4] * src[16] + 64) >> 7];
        dst[15] = cm[(filter[2] * src[15] - filter[1] * src[14] + filter[3] * src[16] - filter[4] * src[17] + 64) >> 7];
        */
        __asm__ volatile (
            "xor        %[ftmp0],   %[ftmp0],       %[ftmp0]            \n\t"
            "gsldlc1    %[ftmp1],   0x07(%[src])                        \n\t"
            "gsldrc1    %[ftmp1],   0x00(%[src])                        \n\t"
            "gsldlc1    %[ftmp7],   0x0f(%[src])                        \n\t"
            "mtc1       %[filter2], %[ftmp4]                            \n\t"
            "gsldrc1    %[ftmp7],   0x08(%[src])                        \n\t"
            "pshufh     %[ftmp4],   %[ftmp4],       %[ftmp0]            \n\t"
            "punpcklbh  %[ftmp2],   %[ftmp1],       %[ftmp0]            \n\t"
            "punpckhbh  %[ftmp3],   %[ftmp1],       %[ftmp0]            \n\t"
            "pmullh     %[ftmp5],   %[ftmp2],       %[ftmp4]            \n\t"
            "pmullh     %[ftmp6],   %[ftmp3],       %[ftmp4]            \n\t"
            "punpcklbh  %[ftmp2],   %[ftmp7],       %[ftmp0]            \n\t"
            "punpckhbh  %[ftmp3],   %[ftmp7],       %[ftmp0]            \n\t"
            "pmullh     %[ftmp8],   %[ftmp2],       %[ftmp4]            \n\t"
            "pmullh     %[ftmp9],   %[ftmp3],       %[ftmp4]            \n\t"

            "gsldlc1    %[ftmp1],   0x06(%[src])                        \n\t"
            "gsldrc1    %[ftmp1],  -0x01(%[src])                        \n\t"
            "gsldlc1    %[ftmp7],   0x0e(%[src])                        \n\t"
            "mtc1       %[filter1], %[ftmp4]                            \n\t"
            "gsldrc1    %[ftmp7],   0x07(%[src])                        \n\t"
            "pshufh     %[ftmp4],   %[ftmp4],       %[ftmp0]            \n\t"
            "punpcklbh  %[ftmp2],   %[ftmp1],       %[ftmp0]            \n\t"
            "punpckhbh  %[ftmp3],   %[ftmp1],       %[ftmp0]            \n\t"
            "pmullh     %[ftmp2],   %[ftmp2],       %[ftmp4]            \n\t"
            "pmullh     %[ftmp3],   %[ftmp3],       %[ftmp4]            \n\t"
            "psubush    %[ftmp5],   %[ftmp5],       %[ftmp2]            \n\t"
            "psubush    %[ftmp6],   %[ftmp6],       %[ftmp3]            \n\t"
            "punpcklbh  %[ftmp2],   %[ftmp7],       %[ftmp0]            \n\t"
            "punpckhbh  %[ftmp3],   %[ftmp7],       %[ftmp0]            \n\t"
            "pmullh     %[ftmp2],   %[ftmp2],       %[ftmp4]            \n\t"
            "pmullh     %[ftmp3],   %[ftmp3],       %[ftmp4]            \n\t"
            "psubush    %[ftmp8],   %[ftmp8],       %[ftmp2]            \n\t"
            "psubush    %[ftmp9],   %[ftmp9],       %[ftmp3]            \n\t"

            "gsldlc1    %[ftmp1],   0x08(%[src])                        \n\t"
            "gsldrc1    %[ftmp1],   0x01(%[src])                        \n\t"
            "gsldlc1    %[ftmp7],   0x0c(%[src])                        \n\t"
            "mtc1       %[filter3], %[ftmp4]                            \n\t"
            "gsldrc1    %[ftmp7],   0x06(%[src])                        \n\t"
            "pshufh     %[ftmp4],   %[ftmp4],       %[ftmp0]            \n\t"
            "punpcklbh  %[ftmp2],   %[ftmp1],       %[ftmp0]            \n\t"
            "punpckhbh  %[ftmp3],   %[ftmp1],       %[ftmp0]            \n\t"
            "pmullh     %[ftmp2],   %[ftmp2],       %[ftmp4]            \n\t"
            "pmullh     %[ftmp3],   %[ftmp3],       %[ftmp4]            \n\t"
            "paddush    %[ftmp5],   %[ftmp5],       %[ftmp2]            \n\t"
            "paddush    %[ftmp6],   %[ftmp6],       %[ftmp3]            \n\t"
            "punpcklbh  %[ftmp2],   %[ftmp7],       %[ftmp0]            \n\t"
            "punpckhbh  %[ftmp3],   %[ftmp7],       %[ftmp0]            \n\t"
            "pmullh     %[ftmp2],   %[ftmp2],       %[ftmp4]            \n\t"
            "pmullh     %[ftmp3],   %[ftmp3],       %[ftmp4]            \n\t"
            "paddush    %[ftmp8],   %[ftmp8],       %[ftmp2]            \n\t"
            "paddush    %[ftmp9],   %[ftmp9],       %[ftmp3]            \n\t"

            "gsldlc1    %[ftmp1],   0x09(%[src])                        \n\t"
            "gsldrc1    %[ftmp1],   0x02(%[src])                        \n\t"
            "gsldlc1    %[ftmp7],   0x11(%[src])                        \n\t"
            "mtc1       %[filter4], %[ftmp4]                            \n\t"
            "gsldrc1    %[ftmp7],   0x0a(%[src])                        \n\t"
            "pshufh     %[ftmp4],   %[ftmp4],       %[ftmp0]            \n\t"
            "punpcklbh  %[ftmp2],   %[ftmp1],       %[ftmp0]            \n\t"
            "punpckhbh  %[ftmp3],   %[ftmp1],       %[ftmp0]            \n\t"
            "pmullh     %[ftmp2],   %[ftmp2],       %[ftmp4]            \n\t"
            "pmullh     %[ftmp3],   %[ftmp3],       %[ftmp4]            \n\t"
            "psubush    %[ftmp5],   %[ftmp5],       %[ftmp2]            \n\t"
            "psubush    %[ftmp6],   %[ftmp6],       %[ftmp3]            \n\t"
            "punpcklbh  %[ftmp2],   %[ftmp7],       %[ftmp0]            \n\t"
            "punpckhbh  %[ftmp3],   %[ftmp7],       %[ftmp0]            \n\t"
            "pmullh     %[ftmp2],   %[ftmp2],       %[ftmp4]            \n\t"
            "pmullh     %[ftmp3],   %[ftmp3],       %[ftmp4]            \n\t"
            "psubush    %[ftmp8],   %[ftmp8],       %[ftmp2]            \n\t"
            "psubush    %[ftmp9],   %[ftmp9],       %[ftmp3]            \n\t"

            "li         %[tmp0],    0x07                                \n\t"
            "paddush    %[ftmp5],   %[ftmp5],       %[ff_pw_64]         \n\t"
            "paddush    %[ftmp6],   %[ftmp6],       %[ff_pw_64]         \n\t"
            "mtc1       %[tmp0],    %[ftmp4]                            \n\t"
            "psrlh      %[ftmp5],   %[ftmp5],       %[ftmp4]            \n\t"
            "psrlh      %[ftmp6],   %[ftmp6],       %[ftmp4]            \n\t"

            "packushb   %[ftmp1],   %[ftmp5],       %[ftmp6]            \n\t"
            "gssdlc1    %[ftmp1],   0x07(%[dst])                        \n\t"
            "gssdrc1    %[ftmp1],   0x00(%[dst])                        \n\t"
            "gssdlc1    %[ftmp1],   0x0f(%[dst])                        \n\t"
            "gssdrc1    %[ftmp1],   0x08(%[dst])                        \n\t"
            : [ftmp0]"=&f"(ftmp[0]),            [ftmp1]"=&f"(ftmp[1]),
              [ftmp2]"=&f"(ftmp[2]),            [ftmp3]"=&f"(ftmp[3]),
              [ftmp4]"=&f"(ftmp[4]),            [ftmp5]"=&f"(ftmp[5]),
              [ftmp6]"=&f"(ftmp[6]),            [ftmp7]"=&f"(ftmp[7]),
              [ftmp8]"=&f"(ftmp[8]),            [ftmp9]"=&f"(ftmp[9]),
              [tmp0]"=&r"(tmp[0])
            : [dst]"r"(dst),                    [src]"r"(src),
              [ff_pw_64]"f"(ff_pw_64),
              [filter1]"r"(filter[1]),          [filter2]"r"(filter[2]),
              [filter3]"r"(filter[3]),          [filter4]"r"(filter[4])
            : "memory"
        );
        dst += dststride;
        src += srcstride;
    }
#else
    const uint8_t *filter = subpel_filters[mx - 1];
    const uint8_t *cm     = ff_crop_tab + MAX_NEG_CROP;
    int x, y;

    for (y = 0; y < h; y++) {
        for (x = 0; x < 16; x++)
            dst[x] = FILTER_4TAP(src, filter, 1);
        dst += dststride;
        src += srcstride;
    }
#endif
}

void ff_put_vp8_epel8_h4_mmi(uint8_t *dst, ptrdiff_t dststride, uint8_t *src,
        ptrdiff_t srcstride, int h, int mx, int my)
{
#if NOTOK //FIXME fate-vp8-test-vector-006
    const uint8_t *filter = subpel_filters[mx - 1];
    int y;
    double ftmp[7];
    uint64_t tmp[1];

    for (y = 0; y < h; y++) {
        /*
        dst[0] = cm[(filter[2] * src[0] - filter[1] * src[-1] + filter[3] * src[1] - filter[4] * src[2] + 64) >> 7];
        dst[1] = cm[(filter[2] * src[1] - filter[1] * src[ 0] + filter[3] * src[2] - filter[4] * src[3] + 64) >> 7];
        dst[2] = cm[(filter[2] * src[2] - filter[1] * src[ 1] + filter[3] * src[3] - filter[4] * src[4] + 64) >> 7];
        dst[3] = cm[(filter[2] * src[3] - filter[1] * src[ 2] + filter[3] * src[4] - filter[4] * src[5] + 64) >> 7];
        dst[4] = cm[(filter[2] * src[4] - filter[1] * src[ 3] + filter[3] * src[5] - filter[4] * src[6] + 64) >> 7];
        dst[5] = cm[(filter[2] * src[5] - filter[1] * src[ 4] + filter[3] * src[6] - filter[4] * src[7] + 64) >> 7];
        dst[6] = cm[(filter[2] * src[6] - filter[1] * src[ 5] + filter[3] * src[7] - filter[4] * src[8] + 64) >> 7];
        dst[7] = cm[(filter[2] * src[7] - filter[1] * src[ 6] + filter[3] * src[8] - filter[4] * src[9] + 64) >> 7];
        */
        __asm__ volatile (
            "xor        %[ftmp0],   %[ftmp0],       %[ftmp0]            \n\t"
            "gsldlc1    %[ftmp1],   0x07(%[src])                        \n\t"
            "mtc1       %[filter2], %[ftmp4]                            \n\t"
            "gsldrc1    %[ftmp1],   0x00(%[src])                        \n\t"
            "pshufh     %[ftmp4],   %[ftmp4],       %[ftmp0]            \n\t"
            "punpcklbh  %[ftmp2],   %[ftmp1],       %[ftmp0]            \n\t"
            "punpckhbh  %[ftmp3],   %[ftmp1],       %[ftmp0]            \n\t"
            "pmullh     %[ftmp5],   %[ftmp2],       %[ftmp4]            \n\t"
            "pmullh     %[ftmp6],   %[ftmp3],       %[ftmp4]            \n\t"

            "gsldlc1    %[ftmp1],   0x06(%[src])                        \n\t"
            "mtc1       %[filter1], %[ftmp4]                            \n\t"
            "gsldrc1    %[ftmp1],  -0x01(%[src])                        \n\t"
            "pshufh     %[ftmp4],   %[ftmp4],       %[ftmp0]            \n\t"
            "punpcklbh  %[ftmp2],   %[ftmp1],       %[ftmp0]            \n\t"
            "punpckhbh  %[ftmp3],   %[ftmp1],       %[ftmp0]            \n\t"
            "pmullh     %[ftmp2],   %[ftmp2],       %[ftmp4]            \n\t"
            "pmullh     %[ftmp3],   %[ftmp3],       %[ftmp4]            \n\t"
            "psubush    %[ftmp5],   %[ftmp5],       %[ftmp2]            \n\t"
            "psubush    %[ftmp6],   %[ftmp6],       %[ftmp3]            \n\t"

            "gsldlc1    %[ftmp1],   0x08(%[src])                        \n\t"
            "mtc1       %[filter3], %[ftmp4]                            \n\t"
            "gsldrc1    %[ftmp1],   0x01(%[src])                        \n\t"
            "pshufh     %[ftmp4],   %[ftmp4],       %[ftmp0]            \n\t"
            "punpcklbh  %[ftmp2],   %[ftmp1],       %[ftmp0]            \n\t"
            "punpckhbh  %[ftmp3],   %[ftmp1],       %[ftmp0]            \n\t"
            "pmullh     %[ftmp2],   %[ftmp2],       %[ftmp4]            \n\t"
            "pmullh     %[ftmp3],   %[ftmp3],       %[ftmp4]            \n\t"
            "paddush    %[ftmp5],   %[ftmp5],       %[ftmp2]            \n\t"
            "paddush    %[ftmp6],   %[ftmp6],       %[ftmp3]            \n\t"

            "gsldlc1    %[ftmp1],   0x09(%[src])                        \n\t"
            "mtc1       %[filter4], %[ftmp4]                            \n\t"
            "gsldrc1    %[ftmp1],   0x02(%[src])                        \n\t"
            "pshufh     %[ftmp4],   %[ftmp4],       %[ftmp0]            \n\t"
            "punpcklbh  %[ftmp2],   %[ftmp1],       %[ftmp0]            \n\t"
            "punpckhbh  %[ftmp3],   %[ftmp1],       %[ftmp0]            \n\t"
            "pmullh     %[ftmp2],   %[ftmp2],       %[ftmp4]            \n\t"
            "pmullh     %[ftmp3],   %[ftmp3],       %[ftmp4]            \n\t"
            "psubush    %[ftmp5],   %[ftmp5],       %[ftmp2]            \n\t"
            "psubush    %[ftmp6],   %[ftmp6],       %[ftmp3]            \n\t"

            "li         %[tmp0],    0x07                                \n\t"
            "paddush    %[ftmp5],   %[ftmp5],       %[ff_pw_64]         \n\t"
            "mtc1       %[tmp0],    %[ftmp4]                            \n\t"
            "paddush    %[ftmp6],   %[ftmp6],       %[ff_pw_64]         \n\t"
            "psrlh      %[ftmp5],   %[ftmp5],       %[ftmp4]            \n\t"
            "psrlh      %[ftmp6],   %[ftmp6],       %[ftmp4]            \n\t"

            "packushb   %[ftmp1],   %[ftmp5],       %[ftmp6]            \n\t"
            "gssdlc1    %[ftmp1],   0x07(%[dst])                        \n\t"
            "gssdrc1    %[ftmp1],   0x00(%[dst])                        \n\t"
            : [ftmp0]"=&f"(ftmp[0]),            [ftmp1]"=&f"(ftmp[1]),
              [ftmp2]"=&f"(ftmp[2]),            [ftmp3]"=&f"(ftmp[3]),
              [ftmp4]"=&f"(ftmp[4]),            [ftmp5]"=&f"(ftmp[5]),
              [ftmp6]"=&f"(ftmp[6]),
              [tmp0]"=&r"(tmp[0])
            : [dst]"r"(dst),                    [src]"r"(src),
              [ff_pw_64]"f"(ff_pw_64),
              [filter1]"r"(filter[1]),          [filter2]"r"(filter[2]),
              [filter3]"r"(filter[3]),          [filter4]"r"(filter[4])
            : "memory"
        );
        dst += dststride;
        src += srcstride;
    }
#else
    const uint8_t *filter = subpel_filters[mx - 1];
    const uint8_t *cm     = ff_crop_tab + MAX_NEG_CROP;
    int x, y;

    for (y = 0; y < h; y++) {
        for (x = 0; x < 8; x++)
            dst[x] = FILTER_4TAP(src, filter, 1);
        dst += dststride;
        src += srcstride;
    }
#endif
}

void ff_put_vp8_epel4_h4_mmi(uint8_t *dst, ptrdiff_t dststride, uint8_t *src,
        ptrdiff_t srcstride, int h, int mx, int my)
{
#if NOTOK //FIXME fate-vp8-test-vector-006
    const uint8_t *filter = subpel_filters[mx - 1];
    int y;
    double ftmp[5];
    uint64_t tmp[1];

    for (y = 0; y < h; y++) {
        /*
        dst[0] = cm[(filter[2] * src[0] - filter[1] * src[-1] + filter[3] * src[1] - filter[4] * src[2] + 64) >> 7];
        dst[1] = cm[(filter[2] * src[1] - filter[1] * src[ 0] + filter[3] * src[2] - filter[4] * src[3] + 64) >> 7];
        dst[2] = cm[(filter[2] * src[2] - filter[1] * src[ 1] + filter[3] * src[3] - filter[4] * src[4] + 64) >> 7];
        dst[3] = cm[(filter[2] * src[3] - filter[1] * src[ 2] + filter[3] * src[4] - filter[4] * src[5] + 64) >> 7];
        */
        __asm__ volatile (
            "xor        %[ftmp0],   %[ftmp0],       %[ftmp0]            \n\t"
            "gslwlc1    %[ftmp1],   0x03(%[src])                        \n\t"
            "mtc1       %[filter2], %[ftmp3]                            \n\t"
            "gslwrc1    %[ftmp1],   0x00(%[src])                        \n\t"
            "pshufh     %[ftmp3],   %[ftmp3],       %[ftmp0]            \n\t"
            "punpcklbh  %[ftmp2],   %[ftmp1],       %[ftmp0]            \n\t"
            "pmullh     %[ftmp4],   %[ftmp2],       %[ftmp3]            \n\t"

            "gslwlc1    %[ftmp1],   0x02(%[src])                        \n\t"
            "mtc1       %[filter1], %[ftmp3]                            \n\t"
            "gslwrc1    %[ftmp1],  -0x01(%[src])                        \n\t"
            "pshufh     %[ftmp3],   %[ftmp3],       %[ftmp0]            \n\t"
            "punpcklbh  %[ftmp2],   %[ftmp1],       %[ftmp0]            \n\t"
            "pmullh     %[ftmp2],   %[ftmp2],       %[ftmp3]            \n\t"
            "psubush    %[ftmp4],   %[ftmp4],       %[ftmp2]            \n\t"

            "gslwlc1    %[ftmp1],   0x04(%[src])                        \n\t"
            "mtc1       %[filter3], %[ftmp3]                            \n\t"
            "gslwrc1    %[ftmp1],   0x01(%[src])                        \n\t"
            "pshufh     %[ftmp3],   %[ftmp3],       %[ftmp0]            \n\t"
            "punpcklbh  %[ftmp2],   %[ftmp1],       %[ftmp0]            \n\t"
            "pmullh     %[ftmp2],   %[ftmp2],       %[ftmp3]            \n\t"
            "paddush    %[ftmp4],   %[ftmp4],       %[ftmp2]            \n\t"

            "gslwlc1    %[ftmp1],   0x05(%[src])                        \n\t"
            "mtc1       %[filter4], %[ftmp3]                            \n\t"
            "gslwrc1    %[ftmp1],   0x02(%[src])                        \n\t"
            "pshufh     %[ftmp3],   %[ftmp3],       %[ftmp0]            \n\t"
            "punpcklbh  %[ftmp2],   %[ftmp1],       %[ftmp0]            \n\t"
            "pmullh     %[ftmp2],   %[ftmp2],       %[ftmp3]            \n\t"
            "psubush    %[ftmp4],   %[ftmp4],       %[ftmp2]            \n\t"

            "li         %[tmp0],    0x07                                \n\t"
            "paddush    %[ftmp4],   %[ftmp4],       %[ff_pw_64]         \n\t"
            "mtc1       %[tmp0],    %[ftmp3]                            \n\t"
            "psrlh      %[ftmp4],   %[ftmp4],       %[ftmp3]            \n\t"

            "packushb   %[ftmp1],   %[ftmp4],       %[ftmp0]            \n\t"
            "gsswlc1    %[ftmp1],   0x03(%[dst])                        \n\t"
            "gsswrc1    %[ftmp1],   0x00(%[dst])                        \n\t"
            : [ftmp0]"=&f"(ftmp[0]),            [ftmp1]"=&f"(ftmp[1]),
              [ftmp2]"=&f"(ftmp[2]),            [ftmp3]"=&f"(ftmp[3]),
              [ftmp4]"=&f"(ftmp[4]),
              [tmp0]"=&r"(tmp[0])
            : [dst]"r"(dst),                    [src]"r"(src),
              [ff_pw_64]"f"(ff_pw_64),
              [filter1]"r"(filter[1]),          [filter2]"r"(filter[2]),
              [filter3]"r"(filter[3]),          [filter4]"r"(filter[4])
            : "memory"
        );
        dst += dststride;
        src += srcstride;
    }
#else
    const uint8_t *filter = subpel_filters[mx - 1];
    const uint8_t *cm     = ff_crop_tab + MAX_NEG_CROP;
    int x, y;

    for (y = 0; y < h; y++) {
        for (x = 0; x < 4; x++)
            dst[x] = FILTER_4TAP(src, filter, 1);
        dst += dststride;
        src += srcstride;
    }
#endif
}

void ff_put_vp8_epel16_h6_mmi(uint8_t *dst, ptrdiff_t dststride, uint8_t *src,
        ptrdiff_t srcstride, int h, int mx, int my)
{
#if NOTOK //FIXME fate-vp8-test-vector-002 006 009
    const uint8_t *filter = subpel_filters[mx - 1];
    int y;
    double ftmp[10];
    uint64_t tmp[1];

    for (y = 0; y < h; y++) {
        /*
        dst[0] = cm[(filter[2]*src[0] - filter[1]*src[-1] + filter[0]*src[-2] + filter[3]*src[1] - filter[4]*src[2] + filter[5]*src[ 3] + 64) >> 7];
        dst[1] = cm[(filter[2]*src[1] - filter[1]*src[ 0] + filter[0]*src[-1] + filter[3]*src[2] - filter[4]*src[3] + filter[5]*src[ 4] + 64) >> 7];
        dst[2] = cm[(filter[2]*src[2] - filter[1]*src[ 1] + filter[0]*src[ 0] + filter[3]*src[3] - filter[4]*src[4] + filter[5]*src[ 5] + 64) >> 7];
        dst[3] = cm[(filter[2]*src[3] - filter[1]*src[ 2] + filter[0]*src[ 1] + filter[3]*src[4] - filter[4]*src[5] + filter[5]*src[ 6] + 64) >> 7];
        dst[4] = cm[(filter[2]*src[4] - filter[1]*src[ 3] + filter[0]*src[ 2] + filter[3]*src[5] - filter[4]*src[6] + filter[5]*src[ 7] + 64) >> 7];
        dst[5] = cm[(filter[2]*src[5] - filter[1]*src[ 4] + filter[0]*src[ 3] + filter[3]*src[6] - filter[4]*src[7] + filter[5]*src[ 8] + 64) >> 7];
        dst[6] = cm[(filter[2]*src[6] - filter[1]*src[ 5] + filter[0]*src[ 4] + filter[3]*src[7] - filter[4]*src[8] + filter[5]*src[ 9] + 64) >> 7];
        dst[7] = cm[(filter[2]*src[7] - filter[1]*src[ 6] + filter[0]*src[ 5] + filter[3]*src[8] - filter[4]*src[9] + filter[5]*src[10] + 64) >> 7];

        dst[ 8] = cm[(filter[2]*src[ 8] - filter[1]*src[ 7] + filter[0]*src[ 6] + filter[3]*src[ 9] - filter[4]*src[10] + filter[5]*src[11] + 64) >> 7];
        dst[ 9] = cm[(filter[2]*src[ 9] - filter[1]*src[ 8] + filter[0]*src[ 7] + filter[3]*src[10] - filter[4]*src[11] + filter[5]*src[12] + 64) >> 7];
        dst[10] = cm[(filter[2]*src[10] - filter[1]*src[ 9] + filter[0]*src[ 8] + filter[3]*src[11] - filter[4]*src[12] + filter[5]*src[13] + 64) >> 7];
        dst[11] = cm[(filter[2]*src[11] - filter[1]*src[10] + filter[0]*src[ 9] + filter[3]*src[12] - filter[4]*src[13] + filter[5]*src[14] + 64) >> 7];
        dst[12] = cm[(filter[2]*src[12] - filter[1]*src[11] + filter[0]*src[10] + filter[3]*src[13] - filter[4]*src[14] + filter[5]*src[15] + 64) >> 7];
        dst[13] = cm[(filter[2]*src[13] - filter[1]*src[12] + filter[0]*src[11] + filter[3]*src[14] - filter[4]*src[15] + filter[5]*src[16] + 64) >> 7];
        dst[14] = cm[(filter[2]*src[14] - filter[1]*src[13] + filter[0]*src[12] + filter[3]*src[15] - filter[4]*src[16] + filter[5]*src[17] + 64) >> 7];
        dst[15] = cm[(filter[2]*src[15] - filter[1]*src[14] + filter[0]*src[13] + filter[3]*src[16] - filter[4]*src[17] + filter[5]*src[18] + 64) >> 7];
        */
        __asm__ volatile (
            "xor        %[ftmp0],   %[ftmp0],       %[ftmp0]            \n\t"
            "gsldlc1    %[ftmp1],   0x07(%[src])                        \n\t"
            "gsldrc1    %[ftmp1],   0x00(%[src])                        \n\t"
            "gsldlc1    %[ftmp7],   0x0f(%[src])                        \n\t"
            "mtc1       %[filter2], %[ftmp4]                            \n\t"
            "gsldrc1    %[ftmp7],   0x08(%[src])                        \n\t"
            "pshufh     %[ftmp4],   %[ftmp4],       %[ftmp0]            \n\t"
            "punpcklbh  %[ftmp2],   %[ftmp1],       %[ftmp0]            \n\t"
            "punpckhbh  %[ftmp3],   %[ftmp1],       %[ftmp0]            \n\t"
            "pmullh     %[ftmp5],   %[ftmp2],       %[ftmp4]            \n\t"
            "pmullh     %[ftmp6],   %[ftmp3],       %[ftmp4]            \n\t"
            "punpcklbh  %[ftmp2],   %[ftmp7],       %[ftmp0]            \n\t"
            "punpckhbh  %[ftmp3],   %[ftmp7],       %[ftmp0]            \n\t"
            "pmullh     %[ftmp8],   %[ftmp2],       %[ftmp4]            \n\t"
            "pmullh     %[ftmp9],   %[ftmp3],       %[ftmp4]            \n\t"

            "gsldlc1    %[ftmp1],   0x06(%[src])                        \n\t"
            "gsldrc1    %[ftmp1],  -0x01(%[src])                        \n\t"
            "gsldlc1    %[ftmp7],   0x0e(%[src])                        \n\t"
            "mtc1       %[filter1], %[ftmp4]                            \n\t"
            "gsldrc1    %[ftmp7],   0x07(%[src])                        \n\t"
            "pshufh     %[ftmp4],   %[ftmp4],       %[ftmp0]            \n\t"
            "punpcklbh  %[ftmp2],   %[ftmp1],       %[ftmp0]            \n\t"
            "punpckhbh  %[ftmp3],   %[ftmp1],       %[ftmp0]            \n\t"
            "pmullh     %[ftmp2],   %[ftmp2],       %[ftmp4]            \n\t"
            "pmullh     %[ftmp3],   %[ftmp3],       %[ftmp4]            \n\t"
            "psubush    %[ftmp5],   %[ftmp5],       %[ftmp2]            \n\t"
            "psubush    %[ftmp6],   %[ftmp6],       %[ftmp3]            \n\t"
            "punpcklbh  %[ftmp2],   %[ftmp7],       %[ftmp0]            \n\t"
            "punpckhbh  %[ftmp3],   %[ftmp7],       %[ftmp0]            \n\t"
            "pmullh     %[ftmp2],   %[ftmp2],       %[ftmp4]            \n\t"
            "pmullh     %[ftmp3],   %[ftmp3],       %[ftmp4]            \n\t"
            "psubush    %[ftmp8],   %[ftmp8],       %[ftmp2]            \n\t"
            "psubush    %[ftmp9],   %[ftmp9],       %[ftmp3]            \n\t"

            "gsldlc1    %[ftmp1],   0x05(%[src])                        \n\t"
            "gsldrc1    %[ftmp1],  -0x02(%[src])                        \n\t"
            "gsldlc1    %[ftmp7],   0x0d(%[src])                        \n\t"
            "mtc1       %[filter0], %[ftmp4]                            \n\t"
            "gsldrc1    %[ftmp7],   0x06(%[src])                        \n\t"
            "pshufh     %[ftmp4],   %[ftmp4],       %[ftmp0]            \n\t"
            "punpcklbh  %[ftmp2],   %[ftmp1],       %[ftmp0]            \n\t"
            "punpckhbh  %[ftmp3],   %[ftmp1],       %[ftmp0]            \n\t"
            "pmullh     %[ftmp2],   %[ftmp2],       %[ftmp4]            \n\t"
            "pmullh     %[ftmp3],   %[ftmp3],       %[ftmp4]            \n\t"
            "paddush    %[ftmp5],   %[ftmp5],       %[ftmp2]            \n\t"
            "paddush    %[ftmp6],   %[ftmp6],       %[ftmp3]            \n\t"
            "punpcklbh  %[ftmp2],   %[ftmp7],       %[ftmp0]            \n\t"
            "punpckhbh  %[ftmp3],   %[ftmp7],       %[ftmp0]            \n\t"
            "pmullh     %[ftmp2],   %[ftmp2],       %[ftmp4]            \n\t"
            "pmullh     %[ftmp3],   %[ftmp3],       %[ftmp4]            \n\t"
            "paddush    %[ftmp8],   %[ftmp8],       %[ftmp2]            \n\t"
            "paddush    %[ftmp9],   %[ftmp9],       %[ftmp3]            \n\t"

            "gsldlc1    %[ftmp1],   0x08(%[src])                        \n\t"
            "gsldrc1    %[ftmp1],   0x01(%[src])                        \n\t"
            "gsldlc1    %[ftmp7],   0x10(%[src])                        \n\t"
            "mtc1       %[filter3], %[ftmp4]                            \n\t"
            "gsldrc1    %[ftmp7],   0x09(%[src])                        \n\t"
            "pshufh     %[ftmp4],   %[ftmp4],       %[ftmp0]            \n\t"
            "punpcklbh  %[ftmp2],   %[ftmp1],       %[ftmp0]            \n\t"
            "punpckhbh  %[ftmp3],   %[ftmp1],       %[ftmp0]            \n\t"
            "pmullh     %[ftmp2],   %[ftmp2],       %[ftmp4]            \n\t"
            "pmullh     %[ftmp3],   %[ftmp3],       %[ftmp4]            \n\t"
            "paddush    %[ftmp5],   %[ftmp5],       %[ftmp2]            \n\t"
            "paddush    %[ftmp6],   %[ftmp6],       %[ftmp3]            \n\t"
            "punpcklbh  %[ftmp2],   %[ftmp7],       %[ftmp0]            \n\t"
            "punpckhbh  %[ftmp3],   %[ftmp7],       %[ftmp0]            \n\t"
            "pmullh     %[ftmp2],   %[ftmp2],       %[ftmp4]            \n\t"
            "pmullh     %[ftmp3],   %[ftmp3],       %[ftmp4]            \n\t"
            "paddush    %[ftmp8],   %[ftmp8],       %[ftmp2]            \n\t"
            "paddush    %[ftmp9],   %[ftmp9],       %[ftmp3]            \n\t"

            "gsldlc1    %[ftmp1],   0x09(%[src])                        \n\t"
            "gsldrc1    %[ftmp1],   0x02(%[src])                        \n\t"
            "gsldlc1    %[ftmp7],   0x11(%[src])                        \n\t"
            "mtc1       %[filter4], %[ftmp4]                            \n\t"
            "gsldrc1    %[ftmp7],   0x0a(%[src])                        \n\t"
            "pshufh     %[ftmp4],   %[ftmp4],       %[ftmp0]            \n\t"
            "punpcklbh  %[ftmp2],   %[ftmp1],       %[ftmp0]            \n\t"
            "punpckhbh  %[ftmp3],   %[ftmp1],       %[ftmp0]            \n\t"
            "pmullh     %[ftmp2],   %[ftmp2],       %[ftmp4]            \n\t"
            "pmullh     %[ftmp3],   %[ftmp3],       %[ftmp4]            \n\t"
            "psubush    %[ftmp5],   %[ftmp5],       %[ftmp2]            \n\t"
            "psubush    %[ftmp6],   %[ftmp6],       %[ftmp3]            \n\t"
            "punpcklbh  %[ftmp2],   %[ftmp7],       %[ftmp0]            \n\t"
            "punpckhbh  %[ftmp3],   %[ftmp7],       %[ftmp0]            \n\t"
            "pmullh     %[ftmp2],   %[ftmp2],       %[ftmp4]            \n\t"
            "pmullh     %[ftmp3],   %[ftmp3],       %[ftmp4]            \n\t"
            "psubush    %[ftmp8],   %[ftmp8],       %[ftmp2]            \n\t"
            "psubush    %[ftmp9],   %[ftmp9],       %[ftmp3]            \n\t"

            "gsldlc1    %[ftmp1],   0x0a(%[src])                        \n\t"
            "gsldrc1    %[ftmp1],   0x03(%[src])                        \n\t"
            "gsldlc1    %[ftmp7],   0x12(%[src])                        \n\t"
            "mtc1       %[filter5], %[ftmp4]                            \n\t"
            "gsldrc1    %[ftmp7],   0x0b(%[src])                        \n\t"
            "pshufh     %[ftmp4],   %[ftmp4],       %[ftmp0]            \n\t"
            "punpcklbh  %[ftmp2],   %[ftmp1],       %[ftmp0]            \n\t"
            "punpckhbh  %[ftmp3],   %[ftmp1],       %[ftmp0]            \n\t"
            "pmullh     %[ftmp2],   %[ftmp2],       %[ftmp4]            \n\t"
            "pmullh     %[ftmp3],   %[ftmp3],       %[ftmp4]            \n\t"
            "paddush    %[ftmp5],   %[ftmp5],       %[ftmp2]            \n\t"
            "paddush    %[ftmp6],   %[ftmp6],       %[ftmp3]            \n\t"
            "punpcklbh  %[ftmp2],   %[ftmp7],       %[ftmp0]            \n\t"
            "punpckhbh  %[ftmp3],   %[ftmp7],       %[ftmp0]            \n\t"
            "pmullh     %[ftmp2],   %[ftmp2],       %[ftmp4]            \n\t"
            "pmullh     %[ftmp3],   %[ftmp3],       %[ftmp4]            \n\t"
            "paddush    %[ftmp8],   %[ftmp8],       %[ftmp2]            \n\t"
            "paddush    %[ftmp9],   %[ftmp9],       %[ftmp3]            \n\t"

            "li         %[tmp0],    0x07                                \n\t"
            "paddush    %[ftmp5],   %[ftmp5],       %[ff_pw_64]         \n\t"
            "paddush    %[ftmp6],   %[ftmp6],       %[ff_pw_64]         \n\t"
            "paddush    %[ftmp8],   %[ftmp8],       %[ff_pw_64]         \n\t"
            "paddush    %[ftmp9],   %[ftmp9],       %[ff_pw_64]         \n\t"
            "mtc1       %[tmp0],    %[ftmp4]                            \n\t"
            "psrlh      %[ftmp5],   %[ftmp5],       %[ftmp4]            \n\t"
            "psrlh      %[ftmp6],   %[ftmp6],       %[ftmp4]            \n\t"
            "psrlh      %[ftmp8],   %[ftmp8],       %[ftmp4]            \n\t"
            "psrlh      %[ftmp9],   %[ftmp9],       %[ftmp4]            \n\t"

            "packushb   %[ftmp1],   %[ftmp5],       %[ftmp6]            \n\t"
            "packushb   %[ftmp2],   %[ftmp8],       %[ftmp9]            \n\t"
            "gssdlc1    %[ftmp1],   0x07(%[dst])                        \n\t"
            "gssdrc1    %[ftmp1],   0x00(%[dst])                        \n\t"
            "gssdlc1    %[ftmp2],   0x0f(%[dst])                        \n\t"
            "gssdrc1    %[ftmp2],   0x08(%[dst])                        \n\t"
            : [ftmp0]"=&f"(ftmp[0]),            [ftmp1]"=&f"(ftmp[1]),
              [ftmp2]"=&f"(ftmp[2]),            [ftmp3]"=&f"(ftmp[3]),
              [ftmp4]"=&f"(ftmp[4]),            [ftmp5]"=&f"(ftmp[5]),
              [ftmp6]"=&f"(ftmp[6]),            [ftmp7]"=&f"(ftmp[7]),
              [ftmp8]"=&f"(ftmp[8]),            [ftmp9]"=&f"(ftmp[9]),
              [tmp0]"=&r"(tmp[0])
            : [dst]"r"(dst),                    [src]"r"(src),
              [ff_pw_64]"f"(ff_pw_64),
              [filter0]"r"(filter[0]),          [filter1]"r"(filter[1]),
              [filter2]"r"(filter[2]),          [filter3]"r"(filter[3]),
              [filter4]"r"(filter[4]),          [filter5]"r"(filter[5])
            : "memory"
        );

        dst += dststride;
        src += srcstride;
    }
#else
    const uint8_t *filter = subpel_filters[mx - 1];
    const uint8_t *cm     = ff_crop_tab + MAX_NEG_CROP;
    int x, y;

    for (y = 0; y < h; y++) {
        for (x = 0; x < 16; x++)
            dst[x] = FILTER_6TAP(src, filter, 1);
        dst += dststride;
        src += srcstride;
    }
#endif
}

void ff_put_vp8_epel8_h6_mmi(uint8_t *dst, ptrdiff_t dststride, uint8_t *src,
        ptrdiff_t srcstride, int h, int mx, int my)
{
#if NOTOK //FIXME fate-vp8-test-vector-006
    const uint8_t *filter = subpel_filters[mx - 1];
    int y;
    double ftmp[7];
    uint64_t tmp[1];

    for (y = 0; y < h; y++) {
        /*
        dst[0] = cm[(filter[2]*src[0] - filter[1]*src[-1] + filter[0]*src[-2] + filter[3]*src[1] - filter[4]*src[2] + filter[5]*src[ 3] + 64) >> 7];
        dst[1] = cm[(filter[2]*src[1] - filter[1]*src[ 0] + filter[0]*src[-1] + filter[3]*src[2] - filter[4]*src[3] + filter[5]*src[ 4] + 64) >> 7];
        dst[2] = cm[(filter[2]*src[2] - filter[1]*src[ 1] + filter[0]*src[ 0] + filter[3]*src[3] - filter[4]*src[4] + filter[5]*src[ 5] + 64) >> 7];
        dst[3] = cm[(filter[2]*src[3] - filter[1]*src[ 2] + filter[0]*src[ 1] + filter[3]*src[4] - filter[4]*src[5] + filter[5]*src[ 6] + 64) >> 7];
        dst[4] = cm[(filter[2]*src[4] - filter[1]*src[ 3] + filter[0]*src[ 2] + filter[3]*src[5] - filter[4]*src[6] + filter[5]*src[ 7] + 64) >> 7];
        dst[5] = cm[(filter[2]*src[5] - filter[1]*src[ 4] + filter[0]*src[ 3] + filter[3]*src[6] - filter[4]*src[7] + filter[5]*src[ 8] + 64) >> 7];
        dst[6] = cm[(filter[2]*src[6] - filter[1]*src[ 5] + filter[0]*src[ 4] + filter[3]*src[7] - filter[4]*src[8] + filter[5]*src[ 9] + 64) >> 7];
        dst[7] = cm[(filter[2]*src[7] - filter[1]*src[ 6] + filter[0]*src[ 5] + filter[3]*src[8] - filter[4]*src[9] + filter[5]*src[10] + 64) >> 7];
        */
        __asm__ volatile (
            "xor        %[ftmp0],   %[ftmp0],       %[ftmp0]            \n\t"
            "gsldlc1    %[ftmp1],   0x07(%[src])                        \n\t"
            "mtc1       %[filter2], %[ftmp4]                            \n\t"
            "gsldrc1    %[ftmp1],   0x00(%[src])                        \n\t"
            "pshufh     %[ftmp4],   %[ftmp4],       %[ftmp0]            \n\t"
            "punpcklbh  %[ftmp2],   %[ftmp1],       %[ftmp0]            \n\t"
            "punpckhbh  %[ftmp3],   %[ftmp1],       %[ftmp0]            \n\t"
            "pmullh     %[ftmp5],   %[ftmp2],       %[ftmp4]            \n\t"
            "pmullh     %[ftmp6],   %[ftmp3],       %[ftmp4]            \n\t"

            "gsldlc1    %[ftmp1],   0x06(%[src])                        \n\t"
            "mtc1       %[filter1], %[ftmp4]                            \n\t"
            "gsldrc1    %[ftmp1],  -0x01(%[src])                        \n\t"
            "pshufh     %[ftmp4],   %[ftmp4],       %[ftmp0]            \n\t"
            "punpcklbh  %[ftmp2],   %[ftmp1],       %[ftmp0]            \n\t"
            "punpckhbh  %[ftmp3],   %[ftmp1],       %[ftmp0]            \n\t"
            "pmullh     %[ftmp2],   %[ftmp2],       %[ftmp4]            \n\t"
            "pmullh     %[ftmp3],   %[ftmp3],       %[ftmp4]            \n\t"
            "psubush    %[ftmp5],   %[ftmp5],       %[ftmp2]            \n\t"
            "psubush    %[ftmp6],   %[ftmp6],       %[ftmp3]            \n\t"

            "gsldlc1    %[ftmp1],   0x05(%[src])                        \n\t"
            "mtc1       %[filter0], %[ftmp4]                            \n\t"
            "gsldrc1    %[ftmp1],  -0x02(%[src])                        \n\t"
            "pshufh     %[ftmp4],   %[ftmp4],       %[ftmp0]            \n\t"
            "punpcklbh  %[ftmp2],   %[ftmp1],       %[ftmp0]            \n\t"
            "punpckhbh  %[ftmp3],   %[ftmp1],       %[ftmp0]            \n\t"
            "pmullh     %[ftmp2],   %[ftmp2],       %[ftmp4]            \n\t"
            "pmullh     %[ftmp3],   %[ftmp3],       %[ftmp4]            \n\t"
            "paddush    %[ftmp5],   %[ftmp5],       %[ftmp2]            \n\t"
            "paddush    %[ftmp6],   %[ftmp6],       %[ftmp3]            \n\t"

            "gsldlc1    %[ftmp1],   0x08(%[src])                        \n\t"
            "mtc1       %[filter3], %[ftmp4]                            \n\t"
            "gsldrc1    %[ftmp1],   0x01(%[src])                        \n\t"
            "pshufh     %[ftmp4],   %[ftmp4],       %[ftmp0]            \n\t"
            "punpcklbh  %[ftmp2],   %[ftmp1],       %[ftmp0]            \n\t"
            "punpckhbh  %[ftmp3],   %[ftmp1],       %[ftmp0]            \n\t"
            "pmullh     %[ftmp2],   %[ftmp2],       %[ftmp4]            \n\t"
            "pmullh     %[ftmp3],   %[ftmp3],       %[ftmp4]            \n\t"
            "paddush    %[ftmp5],   %[ftmp5],       %[ftmp2]            \n\t"
            "paddush    %[ftmp6],   %[ftmp6],       %[ftmp3]            \n\t"

            "gsldlc1    %[ftmp1],   0x09(%[src])                        \n\t"
            "mtc1       %[filter4], %[ftmp4]                            \n\t"
            "gsldrc1    %[ftmp1],   0x02(%[src])                        \n\t"
            "pshufh     %[ftmp4],   %[ftmp4],       %[ftmp0]            \n\t"
            "punpcklbh  %[ftmp2],   %[ftmp1],       %[ftmp0]            \n\t"
            "punpckhbh  %[ftmp3],   %[ftmp1],       %[ftmp0]            \n\t"
            "pmullh     %[ftmp2],   %[ftmp2],       %[ftmp4]            \n\t"
            "pmullh     %[ftmp3],   %[ftmp3],       %[ftmp4]            \n\t"
            "psubush    %[ftmp5],   %[ftmp5],       %[ftmp2]            \n\t"
            "psubush    %[ftmp6],   %[ftmp6],       %[ftmp3]            \n\t"

            "gsldlc1    %[ftmp1],   0x0a(%[src])                        \n\t"
            "mtc1       %[filter5], %[ftmp4]                            \n\t"
            "gsldrc1    %[ftmp1],   0x03(%[src])                        \n\t"
            "pshufh     %[ftmp4],   %[ftmp4],       %[ftmp0]            \n\t"
            "punpcklbh  %[ftmp2],   %[ftmp1],       %[ftmp0]            \n\t"
            "punpckhbh  %[ftmp3],   %[ftmp1],       %[ftmp0]            \n\t"
            "pmullh     %[ftmp2],   %[ftmp2],       %[ftmp4]            \n\t"
            "pmullh     %[ftmp3],   %[ftmp3],       %[ftmp4]            \n\t"
            "paddush    %[ftmp5],   %[ftmp5],       %[ftmp2]            \n\t"
            "paddush    %[ftmp6],   %[ftmp6],       %[ftmp3]            \n\t"

            "li         %[tmp0],    0x07                                \n\t"
            "paddush    %[ftmp5],   %[ftmp5],       %[ff_pw_64]         \n\t"
            "paddush    %[ftmp6],   %[ftmp6],       %[ff_pw_64]         \n\t"
            "mtc1       %[tmp0],    %[ftmp4]                            \n\t"
            "psrlh      %[ftmp5],   %[ftmp5],       %[ftmp4]            \n\t"
            "psrlh      %[ftmp6],   %[ftmp6],       %[ftmp4]            \n\t"

            "packushb   %[ftmp1],   %[ftmp5],       %[ftmp6]            \n\t"
            "gssdlc1    %[ftmp1],   0x07(%[dst])                        \n\t"
            "gssdrc1    %[ftmp1],   0x00(%[dst])                        \n\t"
            : [ftmp0]"=&f"(ftmp[0]),            [ftmp1]"=&f"(ftmp[1]),
              [ftmp2]"=&f"(ftmp[2]),            [ftmp3]"=&f"(ftmp[3]),
              [ftmp4]"=&f"(ftmp[4]),            [ftmp5]"=&f"(ftmp[5]),
              [ftmp6]"=&f"(ftmp[6]),
              [tmp0]"=&r"(tmp[0])
            : [dst]"r"(dst),                    [src]"r"(src),
              [ff_pw_64]"f"(ff_pw_64),
              [filter0]"r"(filter[0]),          [filter1]"r"(filter[1]),
              [filter2]"r"(filter[2]),          [filter3]"r"(filter[3]),
              [filter4]"r"(filter[4]),          [filter5]"r"(filter[5])
            : "memory"
        );

        dst += dststride;
        src += srcstride;
    }
#else
    const uint8_t *filter = subpel_filters[mx - 1];
    const uint8_t *cm     = ff_crop_tab + MAX_NEG_CROP;
    int x, y;

    for (y = 0; y < h; y++) {
        for (x = 0; x < 8; x++)
            dst[x] = FILTER_6TAP(src, filter, 1);
        dst += dststride;
        src += srcstride;
    }
#endif
}

void ff_put_vp8_epel4_h6_mmi(uint8_t *dst, ptrdiff_t dststride, uint8_t *src,
        ptrdiff_t srcstride, int h, int mx, int my)
{
#if NOTOK //FIXME fate-vp8-test-vector-002 006 009
    const uint8_t *filter = subpel_filters[mx - 1];
    int y;
    double ftmp[6];
    uint64_t tmp[1];

    for (y = 0; y < h; y++) {
        /*
        dst[0] = cm[(filter[2]*src[0] - filter[1]*src[-1] + filter[0]*src[-2] + filter[3]*src[1] - filter[4]*src[2] + filter[5]*src[ 3] + 64) >> 7];
        dst[1] = cm[(filter[2]*src[1] - filter[1]*src[ 0] + filter[0]*src[-1] + filter[3]*src[2] - filter[4]*src[3] + filter[5]*src[ 4] + 64) >> 7];
        dst[2] = cm[(filter[2]*src[2] - filter[1]*src[ 1] + filter[0]*src[ 0] + filter[3]*src[3] - filter[4]*src[4] + filter[5]*src[ 5] + 64) >> 7];
        dst[3] = cm[(filter[2]*src[3] - filter[1]*src[ 2] + filter[0]*src[ 1] + filter[3]*src[4] - filter[4]*src[5] + filter[5]*src[ 6] + 64) >> 7];
        */
        __asm__ volatile (
            "xor        %[ftmp0],   %[ftmp0],       %[ftmp0]            \n\t"
            "gslwlc1    %[ftmp1],   0x03(%[src])                        \n\t"
            "mtc1       %[filter2], %[ftmp4]                            \n\t"
            "gslwrc1    %[ftmp1],   0x00(%[src])                        \n\t"
            "pshufh     %[ftmp4],   %[ftmp4],       %[ftmp0]            \n\t"
            "punpcklbh  %[ftmp2],   %[ftmp1],       %[ftmp0]            \n\t"
            "pmullh     %[ftmp5],   %[ftmp2],       %[ftmp4]            \n\t"

            "gslwlc1    %[ftmp1],   0x02(%[src])                        \n\t"
            "mtc1       %[filter1], %[ftmp4]                            \n\t"
            "gslwrc1    %[ftmp1],  -0x01(%[src])                        \n\t"
            "pshufh     %[ftmp4],   %[ftmp4],       %[ftmp0]            \n\t"
            "punpcklbh  %[ftmp2],   %[ftmp1],       %[ftmp0]            \n\t"
            "pmullh     %[ftmp2],   %[ftmp2],       %[ftmp4]            \n\t"
            "psubush    %[ftmp5],   %[ftmp5],       %[ftmp2]            \n\t"

            "gslwlc1    %[ftmp1],   0x01(%[src])                        \n\t"
            "mtc1       %[filter0], %[ftmp4]                            \n\t"
            "gslwrc1    %[ftmp1],  -0x02(%[src])                        \n\t"
            "pshufh     %[ftmp4],   %[ftmp4],       %[ftmp0]            \n\t"
            "punpcklbh  %[ftmp2],   %[ftmp1],       %[ftmp0]            \n\t"
            "pmullh     %[ftmp2],   %[ftmp2],       %[ftmp4]            \n\t"
            "paddush    %[ftmp5],   %[ftmp5],       %[ftmp2]            \n\t"

            "gslwlc1    %[ftmp1],   0x04(%[src])                        \n\t"
            "mtc1       %[filter3], %[ftmp4]                            \n\t"
            "gslwrc1    %[ftmp1],   0x01(%[src])                        \n\t"
            "pshufh     %[ftmp4],   %[ftmp4],       %[ftmp0]            \n\t"
            "punpcklbh  %[ftmp2],   %[ftmp1],       %[ftmp0]            \n\t"
            "pmullh     %[ftmp2],   %[ftmp2],       %[ftmp4]            \n\t"
            "paddush    %[ftmp5],   %[ftmp5],       %[ftmp2]            \n\t"

            "gslwlc1    %[ftmp1],   0x05(%[src])                        \n\t"
            "mtc1       %[filter4], %[ftmp4]                            \n\t"
            "gslwrc1    %[ftmp1],   0x02(%[src])                        \n\t"
            "pshufh     %[ftmp4],   %[ftmp4],       %[ftmp0]            \n\t"
            "punpcklbh  %[ftmp2],   %[ftmp1],       %[ftmp0]            \n\t"
            "pmullh     %[ftmp2],   %[ftmp2],       %[ftmp4]            \n\t"
            "psubush    %[ftmp5],   %[ftmp5],       %[ftmp2]            \n\t"

            "gslwlc1    %[ftmp1],   0x06(%[src])                        \n\t"
            "mtc1       %[filter5], %[ftmp4]                            \n\t"
            "gslwrc1    %[ftmp1],   0x03(%[src])                        \n\t"
            "pshufh     %[ftmp4],   %[ftmp4],       %[ftmp0]            \n\t"
            "punpcklbh  %[ftmp2],   %[ftmp1],       %[ftmp0]            \n\t"
            "pmullh     %[ftmp2],   %[ftmp2],       %[ftmp4]            \n\t"
            "paddush    %[ftmp5],   %[ftmp5],       %[ftmp2]            \n\t"

            "li         %[tmp0],    0x07                                \n\t"
            "paddush    %[ftmp5],   %[ftmp5],       %[ff_pw_64]         \n\t"
            "mtc1       %[tmp0],    %[ftmp4]                            \n\t"
            "psrlh      %[ftmp5],   %[ftmp5],       %[ftmp4]            \n\t"

            "packushb   %[ftmp1],   %[ftmp5],       %[ftmp0]            \n\t"
            "gsswlc1    %[ftmp1],   0x03(%[dst])                        \n\t"
            "gsswrc1    %[ftmp1],   0x00(%[dst])                        \n\t"
            : [ftmp0]"=&f"(ftmp[0]),            [ftmp1]"=&f"(ftmp[1]),
              [ftmp2]"=&f"(ftmp[2]),            [ftmp3]"=&f"(ftmp[3]),
              [ftmp4]"=&f"(ftmp[4]),            [ftmp5]"=&f"(ftmp[5]),
              [tmp0]"=&r"(tmp[0])
            : [dst]"r"(dst),                    [src]"r"(src),
              [ff_pw_64]"f"(ff_pw_64),
              [filter0]"r"(filter[0]),          [filter1]"r"(filter[1]),
              [filter2]"r"(filter[2]),          [filter3]"r"(filter[3]),
              [filter4]"r"(filter[4]),          [filter5]"r"(filter[5])
            : "memory"
        );

        dst += dststride;
        src += srcstride;
    }
#else
    const uint8_t *filter = subpel_filters[mx - 1];
    const uint8_t *cm     = ff_crop_tab + MAX_NEG_CROP;
    int x, y;

    for (y = 0; y < h; y++) {
        for (x = 0; x < 4; x++)
            dst[x] = FILTER_6TAP(src, filter, 1);
        dst += dststride;
        src += srcstride;
    }
#endif
}

void ff_put_vp8_epel16_v4_mmi(uint8_t *dst, ptrdiff_t dststride, uint8_t *src,
        ptrdiff_t srcstride, int h, int mx, int my)
{
#if NOTOK //FIXME fate-vp8-test-vector-002 006 009
    const uint8_t *filter = subpel_filters[my - 1];
    int y;
    double ftmp[10];
    uint64_t tmp[1];

    for (y = 0; y < h; y++) {
        /*
        dst[0] = cm[(filter[2] * src[0] - filter[1] * src[ -srcstride] + filter[3] * src[  srcstride] - filter[4] * src[  2*srcstride] + 64) >> 7];
        dst[1] = cm[(filter[2] * src[1] - filter[1] * src[1-srcstride] + filter[3] * src[1+srcstride] - filter[4] * src[1+2*srcstride] + 64) >> 7];
        dst[2] = cm[(filter[2] * src[2] - filter[1] * src[2-srcstride] + filter[3] * src[2+srcstride] - filter[4] * src[2+2*srcstride] + 64) >> 7];
        dst[3] = cm[(filter[2] * src[3] - filter[1] * src[3-srcstride] + filter[3] * src[3+srcstride] - filter[4] * src[3+2*srcstride] + 64) >> 7];
        dst[4] = cm[(filter[2] * src[4] - filter[1] * src[4-srcstride] + filter[3] * src[4+srcstride] - filter[4] * src[4+2*srcstride] + 64) >> 7];
        dst[5] = cm[(filter[2] * src[5] - filter[1] * src[5-srcstride] + filter[3] * src[5+srcstride] - filter[4] * src[5+2*srcstride] + 64) >> 7];
        dst[6] = cm[(filter[2] * src[6] - filter[1] * src[6-srcstride] + filter[3] * src[6+srcstride] - filter[4] * src[6+2*srcstride] + 64) >> 7];
        dst[7] = cm[(filter[2] * src[7] - filter[1] * src[7-srcstride] + filter[3] * src[7+srcstride] - filter[4] * src[7+2*srcstride] + 64) >> 7];

        dst[ 8] = cm[(filter[2] * src[ 8] - filter[1] * src[ 8-srcstride] + filter[3] * src[ 8+srcstride] - filter[4] * src[ 8+2*srcstride] + 64) >> 7];
        dst[ 9] = cm[(filter[2] * src[ 9] - filter[1] * src[ 9-srcstride] + filter[3] * src[ 9+srcstride] - filter[4] * src[ 9+2*srcstride] + 64) >> 7];
        dst[10] = cm[(filter[2] * src[10] - filter[1] * src[10-srcstride] + filter[3] * src[10+srcstride] - filter[4] * src[10+2*srcstride] + 64) >> 7];
        dst[11] = cm[(filter[2] * src[11] - filter[1] * src[11-srcstride] + filter[3] * src[11+srcstride] - filter[4] * src[11+2*srcstride] + 64) >> 7];
        dst[12] = cm[(filter[2] * src[12] - filter[1] * src[12-srcstride] + filter[3] * src[12+srcstride] - filter[4] * src[12+2*srcstride] + 64) >> 7];
        dst[13] = cm[(filter[2] * src[13] - filter[1] * src[13-srcstride] + filter[3] * src[13+srcstride] - filter[4] * src[13+2*srcstride] + 64) >> 7];
        dst[14] = cm[(filter[2] * src[14] - filter[1] * src[14-srcstride] + filter[3] * src[14+srcstride] - filter[4] * src[14+2*srcstride] + 64) >> 7];
        dst[15] = cm[(filter[2] * src[15] - filter[1] * src[15-srcstride] + filter[3] * src[15+srcstride] - filter[4] * src[15+2*srcstride] + 64) >> 7];
        */
        __asm__ volatile (
            "xor        %[ftmp0],   %[ftmp0],       %[ftmp0]            \n\t"
            "gsldlc1    %[ftmp1],   0x07(%[src0])                       \n\t"
            "gsldrc1    %[ftmp1],   0x00(%[src0])                       \n\t"
            "gsldlc1    %[ftmp7],   0x0f(%[src0])                       \n\t"
            "mtc1       %[filter2], %[ftmp4]                            \n\t"
            "gsldrc1    %[ftmp7],   0x08(%[src0])                       \n\t"
            "pshufh     %[ftmp4],   %[ftmp4],       %[ftmp0]            \n\t"
            "punpcklbh  %[ftmp2],   %[ftmp1],       %[ftmp0]            \n\t"
            "punpckhbh  %[ftmp3],   %[ftmp1],       %[ftmp0]            \n\t"
            "pmullh     %[ftmp5],   %[ftmp2],       %[ftmp4]            \n\t"
            "pmullh     %[ftmp6],   %[ftmp3],       %[ftmp4]            \n\t"
            "punpcklbh  %[ftmp2],   %[ftmp7],       %[ftmp0]            \n\t"
            "punpckhbh  %[ftmp3],   %[ftmp7],       %[ftmp0]            \n\t"
            "pmullh     %[ftmp8],   %[ftmp2],       %[ftmp4]            \n\t"
            "pmullh     %[ftmp9],   %[ftmp3],       %[ftmp4]            \n\t"

            "gsldlc1    %[ftmp1],   0x07(%[src1])                       \n\t"
            "gsldrc1    %[ftmp1],   0x00(%[src1])                       \n\t"
            "gsldlc1    %[ftmp7],   0x0f(%[src1])                       \n\t"
            "mtc1       %[filter1], %[ftmp4]                            \n\t"
            "gsldrc1    %[ftmp7],   0x08(%[src1])                       \n\t"
            "pshufh     %[ftmp4],   %[ftmp4],       %[ftmp0]            \n\t"
            "punpcklbh  %[ftmp2],   %[ftmp1],       %[ftmp0]            \n\t"
            "punpckhbh  %[ftmp3],   %[ftmp1],       %[ftmp0]            \n\t"
            "pmullh     %[ftmp2],   %[ftmp2],       %[ftmp4]            \n\t"
            "pmullh     %[ftmp3],   %[ftmp3],       %[ftmp4]            \n\t"
            "psubush    %[ftmp5],   %[ftmp5],       %[ftmp2]            \n\t"
            "psubush    %[ftmp6],   %[ftmp6],       %[ftmp3]            \n\t"
            "punpcklbh  %[ftmp2],   %[ftmp7],       %[ftmp0]            \n\t"
            "punpckhbh  %[ftmp3],   %[ftmp7],       %[ftmp0]            \n\t"
            "pmullh     %[ftmp2],   %[ftmp2],       %[ftmp4]            \n\t"
            "pmullh     %[ftmp3],   %[ftmp3],       %[ftmp4]            \n\t"
            "psubush    %[ftmp8],   %[ftmp8],       %[ftmp2]            \n\t"
            "psubush    %[ftmp9],   %[ftmp9],       %[ftmp3]            \n\t"

            "gsldlc1    %[ftmp1],   0x07(%[src2])                       \n\t"
            "gsldrc1    %[ftmp1],   0x00(%[src2])                       \n\t"
            "gsldlc1    %[ftmp7],   0x0f(%[src2])                       \n\t"
            "mtc1       %[filter3], %[ftmp4]                            \n\t"
            "gsldrc1    %[ftmp7],   0x08(%[src2])                       \n\t"
            "pshufh     %[ftmp4],   %[ftmp4],       %[ftmp0]            \n\t"
            "punpcklbh  %[ftmp2],   %[ftmp1],       %[ftmp0]            \n\t"
            "punpckhbh  %[ftmp3],   %[ftmp1],       %[ftmp0]            \n\t"
            "pmullh     %[ftmp2],   %[ftmp2],       %[ftmp4]            \n\t"
            "pmullh     %[ftmp3],   %[ftmp3],       %[ftmp4]            \n\t"
            "paddush    %[ftmp5],   %[ftmp5],       %[ftmp2]            \n\t"
            "paddush    %[ftmp6],   %[ftmp6],       %[ftmp3]            \n\t"
            "punpcklbh  %[ftmp2],   %[ftmp7],       %[ftmp0]            \n\t"
            "punpckhbh  %[ftmp3],   %[ftmp7],       %[ftmp0]            \n\t"
            "pmullh     %[ftmp2],   %[ftmp2],       %[ftmp4]            \n\t"
            "pmullh     %[ftmp3],   %[ftmp3],       %[ftmp4]            \n\t"
            "paddush    %[ftmp8],   %[ftmp8],       %[ftmp2]            \n\t"
            "paddush    %[ftmp9],   %[ftmp9],       %[ftmp3]            \n\t"

            "gsldlc1    %[ftmp1],   0x07(%[src3])                       \n\t"
            "gsldrc1    %[ftmp1],   0x00(%[src3])                       \n\t"
            "gsldlc1    %[ftmp7],   0x0f(%[src3])                       \n\t"
            "mtc1       %[filter4], %[ftmp4]                            \n\t"
            "gsldrc1    %[ftmp7],   0x08(%[src3])                       \n\t"
            "pshufh     %[ftmp4],   %[ftmp4],       %[ftmp0]            \n\t"
            "punpcklbh  %[ftmp2],   %[ftmp1],       %[ftmp0]            \n\t"
            "punpckhbh  %[ftmp3],   %[ftmp1],       %[ftmp0]            \n\t"
            "pmullh     %[ftmp2],   %[ftmp2],       %[ftmp4]            \n\t"
            "pmullh     %[ftmp3],   %[ftmp3],       %[ftmp4]            \n\t"
            "psubush    %[ftmp5],   %[ftmp5],       %[ftmp2]            \n\t"
            "psubush    %[ftmp6],   %[ftmp6],       %[ftmp3]            \n\t"
            "punpcklbh  %[ftmp2],   %[ftmp7],       %[ftmp0]            \n\t"
            "punpckhbh  %[ftmp3],   %[ftmp7],       %[ftmp0]            \n\t"
            "pmullh     %[ftmp2],   %[ftmp2],       %[ftmp4]            \n\t"
            "pmullh     %[ftmp3],   %[ftmp3],       %[ftmp4]            \n\t"
            "psubush    %[ftmp8],   %[ftmp8],       %[ftmp2]            \n\t"
            "psubush    %[ftmp9],   %[ftmp9],       %[ftmp3]            \n\t"

            "li         %[tmp0],    0x07                                \n\t"
            "paddush    %[ftmp5],   %[ftmp5],       %[ff_pw_64]         \n\t"
            "paddush    %[ftmp6],   %[ftmp6],       %[ff_pw_64]         \n\t"
            "mtc1       %[tmp0],    %[ftmp4]                            \n\t"
            "psrlh      %[ftmp5],   %[ftmp5],       %[ftmp4]            \n\t"
            "psrlh      %[ftmp6],   %[ftmp6],       %[ftmp4]            \n\t"

            "packushb   %[ftmp1],   %[ftmp5],       %[ftmp6]            \n\t"
            "gssdlc1    %[ftmp1],   0x07(%[dst])                        \n\t"
            "gssdrc1    %[ftmp1],   0x00(%[dst])                        \n\t"
            "gssdlc1    %[ftmp1],   0x0f(%[dst])                        \n\t"
            "gssdrc1    %[ftmp1],   0x08(%[dst])                        \n\t"
            : [ftmp0]"=&f"(ftmp[0]),            [ftmp1]"=&f"(ftmp[1]),
              [ftmp2]"=&f"(ftmp[2]),            [ftmp3]"=&f"(ftmp[3]),
              [ftmp4]"=&f"(ftmp[4]),            [ftmp5]"=&f"(ftmp[5]),
              [ftmp6]"=&f"(ftmp[6]),            [ftmp7]"=&f"(ftmp[7]),
              [ftmp8]"=&f"(ftmp[8]),            [ftmp9]"=&f"(ftmp[9]),
              [tmp0]"=&r"(tmp[0])
            : [dst]"r"(dst),                    [src0]"r"(src),
              [src1]"r"(src-srcstride),         [src2]"r"(src+srcstride),
              [src3]"r"(src+2*srcstride),       [ff_pw_64]"f"(ff_pw_64),
              [filter1]"r"(filter[1]),          [filter2]"r"(filter[2]),
              [filter3]"r"(filter[3]),          [filter4]"r"(filter[4])
            : "memory"
        );

        dst += dststride;
        src += srcstride;
    }
#else
    const uint8_t *filter = subpel_filters[my - 1];
    const uint8_t *cm     = ff_crop_tab + MAX_NEG_CROP;
    int x, y;

    for (y = 0; y < h; y++) {
        for (x = 0; x < 16; x++)
            dst[x] = FILTER_4TAP(src, filter, srcstride);
        dst += dststride;
        src += srcstride;
    }
#endif
}

void ff_put_vp8_epel8_v4_mmi(uint8_t *dst, ptrdiff_t dststride, uint8_t *src,
        ptrdiff_t srcstride, int h, int mx, int my)
{
#if NOTOK //FIXME fate-vp8-test-vector-002 006 009
    const uint8_t *filter = subpel_filters[my - 1];
    int y;
    double ftmp[7];
    uint64_t tmp[1];

    for (y = 0; y < h; y++) {
        /*
        dst[0] = cm[(filter[2] * src[0] - filter[1] * src[ -srcstride] + filter[3] * src[  srcstride] - filter[4] * src[  2*srcstride] + 64) >> 7];
        dst[1] = cm[(filter[2] * src[1] - filter[1] * src[1-srcstride] + filter[3] * src[1+srcstride] - filter[4] * src[1+2*srcstride] + 64) >> 7];
        dst[2] = cm[(filter[2] * src[2] - filter[1] * src[2-srcstride] + filter[3] * src[2+srcstride] - filter[4] * src[2+2*srcstride] + 64) >> 7];
        dst[3] = cm[(filter[2] * src[3] - filter[1] * src[3-srcstride] + filter[3] * src[3+srcstride] - filter[4] * src[3+2*srcstride] + 64) >> 7];
        dst[4] = cm[(filter[2] * src[4] - filter[1] * src[4-srcstride] + filter[3] * src[4+srcstride] - filter[4] * src[4+2*srcstride] + 64) >> 7];
        dst[5] = cm[(filter[2] * src[5] - filter[1] * src[5-srcstride] + filter[3] * src[5+srcstride] - filter[4] * src[5+2*srcstride] + 64) >> 7];
        dst[6] = cm[(filter[2] * src[6] - filter[1] * src[6-srcstride] + filter[3] * src[6+srcstride] - filter[4] * src[6+2*srcstride] + 64) >> 7];
        dst[7] = cm[(filter[2] * src[7] - filter[1] * src[7-srcstride] + filter[3] * src[7+srcstride] - filter[4] * src[7+2*srcstride] + 64) >> 7];
        */
        __asm__ volatile (
            "xor        %[ftmp0],   %[ftmp0],       %[ftmp0]            \n\t"
            "gsldlc1    %[ftmp1],   0x07(%[src0])                       \n\t"
            "mtc1       %[filter2], %[ftmp4]                            \n\t"
            "gsldrc1    %[ftmp1],   0x00(%[src0])                       \n\t"
            "pshufh     %[ftmp4],   %[ftmp4],       %[ftmp0]            \n\t"
            "punpcklbh  %[ftmp2],   %[ftmp1],       %[ftmp0]            \n\t"
            "punpckhbh  %[ftmp3],   %[ftmp1],       %[ftmp0]            \n\t"
            "pmullh     %[ftmp5],   %[ftmp2],       %[ftmp4]            \n\t"
            "pmullh     %[ftmp6],   %[ftmp3],       %[ftmp4]            \n\t"

            "gsldlc1    %[ftmp1],   0x07(%[src1])                       \n\t"
            "mtc1       %[filter1], %[ftmp4]                            \n\t"
            "gsldrc1    %[ftmp1],   0x00(%[src1])                       \n\t"
            "pshufh     %[ftmp4],   %[ftmp4],       %[ftmp0]            \n\t"
            "punpcklbh  %[ftmp2],   %[ftmp1],       %[ftmp0]            \n\t"
            "punpckhbh  %[ftmp3],   %[ftmp1],       %[ftmp0]            \n\t"
            "pmullh     %[ftmp2],   %[ftmp2],       %[ftmp4]            \n\t"
            "pmullh     %[ftmp3],   %[ftmp3],       %[ftmp4]            \n\t"
            "psubush    %[ftmp5],   %[ftmp5],       %[ftmp2]            \n\t"
            "psubush    %[ftmp6],   %[ftmp6],       %[ftmp3]            \n\t"

            "gsldlc1    %[ftmp1],   0x07(%[src2])                       \n\t"
            "mtc1       %[filter3], %[ftmp4]                            \n\t"
            "gsldrc1    %[ftmp1],   0x00(%[src2])                       \n\t"
            "pshufh     %[ftmp4],   %[ftmp4],       %[ftmp0]            \n\t"
            "punpcklbh  %[ftmp2],   %[ftmp1],       %[ftmp0]            \n\t"
            "punpckhbh  %[ftmp3],   %[ftmp1],       %[ftmp0]            \n\t"
            "pmullh     %[ftmp2],   %[ftmp2],       %[ftmp4]            \n\t"
            "pmullh     %[ftmp3],   %[ftmp3],       %[ftmp4]            \n\t"
            "paddush    %[ftmp5],   %[ftmp5],       %[ftmp2]            \n\t"
            "paddush    %[ftmp6],   %[ftmp6],       %[ftmp3]            \n\t"

            "gsldlc1    %[ftmp1],   0x07(%[src3])                       \n\t"
            "mtc1       %[filter4], %[ftmp4]                            \n\t"
            "gsldrc1    %[ftmp1],   0x00(%[src3])                       \n\t"
            "pshufh     %[ftmp4],   %[ftmp4],       %[ftmp0]            \n\t"
            "punpcklbh  %[ftmp2],   %[ftmp1],       %[ftmp0]            \n\t"
            "punpckhbh  %[ftmp3],   %[ftmp1],       %[ftmp0]            \n\t"
            "pmullh     %[ftmp2],   %[ftmp2],       %[ftmp4]            \n\t"
            "pmullh     %[ftmp3],   %[ftmp3],       %[ftmp4]            \n\t"
            "psubush    %[ftmp5],   %[ftmp5],       %[ftmp2]            \n\t"
            "psubush    %[ftmp6],   %[ftmp6],       %[ftmp3]            \n\t"

            "li         %[tmp0],    0x07                                \n\t"
            "paddush    %[ftmp5],   %[ftmp5],       %[ff_pw_64]         \n\t"
            "paddush    %[ftmp6],   %[ftmp6],       %[ff_pw_64]         \n\t"
            "mtc1       %[tmp0],    %[ftmp4]                            \n\t"
            "psrlh      %[ftmp5],   %[ftmp5],       %[ftmp4]            \n\t"
            "psrlh      %[ftmp6],   %[ftmp6],       %[ftmp4]            \n\t"

            "packushb   %[ftmp1],   %[ftmp5],       %[ftmp6]            \n\t"
            "gssdlc1    %[ftmp1],   0x07(%[dst])                        \n\t"
            "gssdrc1    %[ftmp1],   0x00(%[dst])                        \n\t"
            : [ftmp0]"=&f"(ftmp[0]),            [ftmp1]"=&f"(ftmp[1]),
              [ftmp2]"=&f"(ftmp[2]),            [ftmp3]"=&f"(ftmp[3]),
              [ftmp4]"=&f"(ftmp[4]),            [ftmp5]"=&f"(ftmp[5]),
              [ftmp6]"=&f"(ftmp[6]),
              [tmp0]"=&r"(tmp[0])
            : [dst]"r"(dst),                    [src0]"r"(src),
              [src1]"r"(src-srcstride),         [src2]"r"(src+srcstride),
              [src3]"r"(src+2*srcstride),       [ff_pw_64]"f"(ff_pw_64),
              [filter1]"r"(filter[1]),          [filter2]"r"(filter[2]),
              [filter3]"r"(filter[3]),          [filter4]"r"(filter[4])
            : "memory"
        );

        dst += dststride;
        src += srcstride;
    }
#else
    const uint8_t *filter = subpel_filters[my - 1];
    const uint8_t *cm     = ff_crop_tab + MAX_NEG_CROP;
    int x, y;

    for (y = 0; y < h; y++) {
        for (x = 0; x < 8; x++)
            dst[x] = FILTER_4TAP(src, filter, srcstride);
        dst += dststride;
        src += srcstride;
    }
#endif
}

void ff_put_vp8_epel4_v4_mmi(uint8_t *dst, ptrdiff_t dststride, uint8_t *src,
        ptrdiff_t srcstride, int h, int mx, int my)
{
#if NOTOK //FIXME fate-vp8-test-vector-002 006 009
    const uint8_t *filter = subpel_filters[my - 1];
    int y;
    double ftmp[6];
    uint64_t tmp[1];

    for (y = 0; y < h; y++) {
        /*
        dst[0] = cm[(filter[2] * src[0] - filter[1] * src[ -srcstride] + filter[3] * src[  srcstride] - filter[4] * src[  2*srcstride] + 64) >> 7];
        dst[1] = cm[(filter[2] * src[1] - filter[1] * src[1-srcstride] + filter[3] * src[1+srcstride] - filter[4] * src[1+2*srcstride] + 64) >> 7];
        dst[2] = cm[(filter[2] * src[2] - filter[1] * src[2-srcstride] + filter[3] * src[2+srcstride] - filter[4] * src[2+2*srcstride] + 64) >> 7];
        dst[3] = cm[(filter[2] * src[3] - filter[1] * src[3-srcstride] + filter[3] * src[3+srcstride] - filter[4] * src[3+2*srcstride] + 64) >> 7];
        */
        __asm__ volatile (
            "xor        %[ftmp0],   %[ftmp0],       %[ftmp0]            \n\t"
            "gslwlc1    %[ftmp1],   0x03(%[src0])                       \n\t"
            "mtc1       %[filter2], %[ftmp3]                            \n\t"
            "gslwrc1    %[ftmp1],   0x00(%[src0])                       \n\t"
            "pshufh     %[ftmp3],   %[ftmp3],       %[ftmp0]            \n\t"
            "punpcklbh  %[ftmp2],   %[ftmp1],       %[ftmp0]            \n\t"
            "pmullh     %[ftmp4],   %[ftmp2],       %[ftmp3]            \n\t"

            "gslwlc1    %[ftmp1],   0x03(%[src1])                       \n\t"
            "mtc1       %[filter1], %[ftmp3]                            \n\t"
            "gslwrc1    %[ftmp1],   0x00(%[src1])                       \n\t"
            "pshufh     %[ftmp3],   %[ftmp3],       %[ftmp0]            \n\t"
            "punpcklbh  %[ftmp2],   %[ftmp1],       %[ftmp0]            \n\t"
            "pmullh     %[ftmp2],   %[ftmp2],       %[ftmp3]            \n\t"
            "psubush    %[ftmp4],   %[ftmp4],       %[ftmp2]            \n\t"

            "gslwlc1    %[ftmp1],   0x03(%[src2])                       \n\t"
            "mtc1       %[filter3], %[ftmp3]                            \n\t"
            "gslwrc1    %[ftmp1],   0x00(%[src2])                       \n\t"
            "pshufh     %[ftmp3],   %[ftmp3],       %[ftmp0]            \n\t"
            "punpcklbh  %[ftmp2],   %[ftmp1],       %[ftmp0]            \n\t"
            "pmullh     %[ftmp2],   %[ftmp2],       %[ftmp3]            \n\t"
            "paddush    %[ftmp4],   %[ftmp4],       %[ftmp2]            \n\t"

            "gslwlc1    %[ftmp1],   0x03(%[src3])                       \n\t"
            "mtc1       %[filter4], %[ftmp3]                            \n\t"
            "gslwrc1    %[ftmp1],   0x00(%[src3])                       \n\t"
            "pshufh     %[ftmp3],   %[ftmp3],       %[ftmp0]            \n\t"
            "punpcklbh  %[ftmp2],   %[ftmp1],       %[ftmp0]            \n\t"
            "pmullh     %[ftmp2],   %[ftmp2],       %[ftmp3]            \n\t"
            "psubush    %[ftmp4],   %[ftmp4],       %[ftmp2]            \n\t"

            "li         %[tmp0],    0x07                                \n\t"
            "paddush    %[ftmp4],   %[ftmp4],       %[ff_pw_64]         \n\t"
            "mtc1       %[tmp0],    %[ftmp3]                            \n\t"
            "psrlh      %[ftmp4],   %[ftmp4],       %[ftmp3]            \n\t"

            "packushb   %[ftmp1],   %[ftmp4],       %[ftmp0]            \n\t"
            "gsswlc1    %[ftmp1],   0x03(%[dst])                        \n\t"
            "gsswrc1    %[ftmp1],   0x00(%[dst])                        \n\t"
            : [ftmp0]"=&f"(ftmp[0]),            [ftmp1]"=&f"(ftmp[1]),
              [ftmp2]"=&f"(ftmp[2]),            [ftmp3]"=&f"(ftmp[3]),
              [ftmp4]"=&f"(ftmp[4]),            [ftmp5]"=&f"(ftmp[5]),
              [tmp0]"=&r"(tmp[0])
            : [dst]"r"(dst),                    [src0]"r"(src),
              [src1]"r"(src-srcstride),         [src2]"r"(src+srcstride),
              [src3]"r"(src+2*srcstride),       [ff_pw_64]"f"(ff_pw_64),
              [filter1]"r"(filter[1]),          [filter2]"r"(filter[2]),
              [filter3]"r"(filter[3]),          [filter4]"r"(filter[4])
            : "memory"
        );

        dst += dststride;
        src += srcstride;
    }
#else
    const uint8_t *filter = subpel_filters[my - 1];
    const uint8_t *cm     = ff_crop_tab + MAX_NEG_CROP;
    int x, y;

    for (y = 0; y < h; y++) {
        for (x = 0; x < 4; x++)
            dst[x] = FILTER_4TAP(src, filter, srcstride);
        dst += dststride;
        src += srcstride;
    }
#endif
}

void ff_put_vp8_epel16_v6_mmi(uint8_t *dst, ptrdiff_t dststride, uint8_t *src,
        ptrdiff_t srcstride, int h, int mx, int my)
{
#if NOTOK //FIXME fate-vp8-test-vector-002 006 009
    const uint8_t *filter = subpel_filters[my - 1];
    int y;
    double ftmp[10];
    uint64_t tmp[1];

    for (y = 0; y < h; y++) {
        /*
        dst[0] = cm[(filter[2]*src[0] - filter[1]*src[0-srcstride] + filter[0]*src[0-2*srcstride] + filter[3]*src[0+srcstride] - filter[4]*src[0+2*srcstride] + filter[5]*src[0+3*srcstride] + 64) >> 7];
        dst[1] = cm[(filter[2]*src[1] - filter[1]*src[1-srcstride] + filter[0]*src[1-2*srcstride] + filter[3]*src[1+srcstride] - filter[4]*src[1+2*srcstride] + filter[5]*src[1+3*srcstride] + 64) >> 7];
        dst[2] = cm[(filter[2]*src[2] - filter[1]*src[2-srcstride] + filter[0]*src[2-2*srcstride] + filter[3]*src[2+srcstride] - filter[4]*src[2+2*srcstride] + filter[5]*src[2+3*srcstride] + 64) >> 7];
        dst[3] = cm[(filter[2]*src[3] - filter[1]*src[3-srcstride] + filter[0]*src[3-2*srcstride] + filter[3]*src[3+srcstride] - filter[4]*src[3+2*srcstride] + filter[5]*src[3+3*srcstride] + 64) >> 7];
        dst[4] = cm[(filter[2]*src[4] - filter[1]*src[4-srcstride] + filter[0]*src[4-2*srcstride] + filter[3]*src[4+srcstride] - filter[4]*src[4+2*srcstride] + filter[5]*src[4+3*srcstride] + 64) >> 7];
        dst[5] = cm[(filter[2]*src[5] - filter[1]*src[5-srcstride] + filter[0]*src[5-2*srcstride] + filter[3]*src[5+srcstride] - filter[4]*src[5+2*srcstride] + filter[5]*src[5+3*srcstride] + 64) >> 7];
        dst[6] = cm[(filter[2]*src[6] - filter[1]*src[6-srcstride] + filter[0]*src[6-2*srcstride] + filter[3]*src[6+srcstride] - filter[4]*src[6+2*srcstride] + filter[5]*src[6+3*srcstride] + 64) >> 7];
        dst[7] = cm[(filter[2]*src[7] - filter[1]*src[7-srcstride] + filter[0]*src[7-2*srcstride] + filter[3]*src[7+srcstride] - filter[4]*src[7+2*srcstride] + filter[5]*src[7+3*srcstride] + 64) >> 7];

        dst[ 8] = cm[(filter[2]*src[ 8] - filter[1]*src[ 8-srcstride] + filter[0]*src[ 8-2*srcstride] + filter[3]*src[ 8+srcstride] - filter[4]*src[ 8+2*srcstride] + filter[5]*src[ 8+3*srcstride] + 64) >> 7];
        dst[ 9] = cm[(filter[2]*src[ 9] - filter[1]*src[ 9-srcstride] + filter[0]*src[ 9-2*srcstride] + filter[3]*src[ 9+srcstride] - filter[4]*src[ 9+2*srcstride] + filter[5]*src[ 9+3*srcstride] + 64) >> 7];
        dst[10] = cm[(filter[2]*src[10] - filter[1]*src[10-srcstride] + filter[0]*src[10-2*srcstride] + filter[3]*src[10+srcstride] - filter[4]*src[10+2*srcstride] + filter[5]*src[10+3*srcstride] + 64) >> 7];
        dst[11] = cm[(filter[2]*src[11] - filter[1]*src[11-srcstride] + filter[0]*src[11-2*srcstride] + filter[3]*src[11+srcstride] - filter[4]*src[11+2*srcstride] + filter[5]*src[11+3*srcstride] + 64) >> 7];
        dst[12] = cm[(filter[2]*src[12] - filter[1]*src[12-srcstride] + filter[0]*src[12-2*srcstride] + filter[3]*src[12+srcstride] - filter[4]*src[12+2*srcstride] + filter[5]*src[12+3*srcstride] + 64) >> 7];
        dst[13] = cm[(filter[2]*src[13] - filter[1]*src[13-srcstride] + filter[0]*src[13-2*srcstride] + filter[3]*src[13+srcstride] - filter[4]*src[13+2*srcstride] + filter[5]*src[13+3*srcstride] + 64) >> 7];
        dst[14] = cm[(filter[2]*src[14] - filter[1]*src[14-srcstride] + filter[0]*src[14-2*srcstride] + filter[3]*src[14+srcstride] - filter[4]*src[14+2*srcstride] + filter[5]*src[14+3*srcstride] + 64) >> 7];
        dst[15] = cm[(filter[2]*src[15] - filter[1]*src[15-srcstride] + filter[0]*src[15-2*srcstride] + filter[3]*src[15+srcstride] - filter[4]*src[15+2*srcstride] + filter[5]*src[15+3*srcstride] + 64) >> 7];
        */
        __asm__ volatile (
            "xor        %[ftmp0],   %[ftmp0],       %[ftmp0]            \n\t"
            "gsldlc1    %[ftmp1],   0x07(%[src0])                       \n\t"
            "gsldrc1    %[ftmp1],   0x00(%[src0])                       \n\t"
            "gsldlc1    %[ftmp7],   0x0f(%[src0])                       \n\t"
            "mtc1       %[filter2], %[ftmp4]                            \n\t"
            "gsldrc1    %[ftmp7],   0x08(%[src0])                       \n\t"
            "pshufh     %[ftmp4],   %[ftmp4],       %[ftmp0]            \n\t"
            "punpcklbh  %[ftmp2],   %[ftmp1],       %[ftmp0]            \n\t"
            "punpckhbh  %[ftmp3],   %[ftmp1],       %[ftmp0]            \n\t"
            "pmullh     %[ftmp5],   %[ftmp2],       %[ftmp4]            \n\t"
            "pmullh     %[ftmp6],   %[ftmp3],       %[ftmp4]            \n\t"
            "punpcklbh  %[ftmp2],   %[ftmp7],       %[ftmp0]            \n\t"
            "punpckhbh  %[ftmp3],   %[ftmp7],       %[ftmp0]            \n\t"
            "pmullh     %[ftmp8],   %[ftmp2],       %[ftmp4]            \n\t"
            "pmullh     %[ftmp9],   %[ftmp3],       %[ftmp4]            \n\t"

            "gsldlc1    %[ftmp1],   0x07(%[src1])                       \n\t"
            "gsldrc1    %[ftmp1],   0x00(%[src1])                       \n\t"
            "gsldlc1    %[ftmp7],   0x0f(%[src1])                       \n\t"
            "mtc1       %[filter1], %[ftmp4]                            \n\t"
            "gsldrc1    %[ftmp7],   0x08(%[src1])                       \n\t"
            "pshufh     %[ftmp4],   %[ftmp4],       %[ftmp0]            \n\t"
            "punpcklbh  %[ftmp2],   %[ftmp1],       %[ftmp0]            \n\t"
            "punpckhbh  %[ftmp3],   %[ftmp1],       %[ftmp0]            \n\t"
            "pmullh     %[ftmp2],   %[ftmp2],       %[ftmp4]            \n\t"
            "pmullh     %[ftmp3],   %[ftmp3],       %[ftmp4]            \n\t"
            "psubush    %[ftmp5],   %[ftmp5],       %[ftmp2]            \n\t"
            "psubush    %[ftmp6],   %[ftmp6],       %[ftmp3]            \n\t"
            "punpcklbh  %[ftmp2],   %[ftmp7],       %[ftmp0]            \n\t"
            "punpckhbh  %[ftmp3],   %[ftmp7],       %[ftmp0]            \n\t"
            "pmullh     %[ftmp2],   %[ftmp2],       %[ftmp4]            \n\t"
            "pmullh     %[ftmp3],   %[ftmp3],       %[ftmp4]            \n\t"
            "psubush    %[ftmp8],   %[ftmp8],       %[ftmp2]            \n\t"
            "psubush    %[ftmp9],   %[ftmp9],       %[ftmp3]            \n\t"

            "gsldlc1    %[ftmp1],   0x07(%[src2])                       \n\t"
            "gsldrc1    %[ftmp1],   0x00(%[src2])                       \n\t"
            "gsldlc1    %[ftmp7],   0x0f(%[src2])                       \n\t"
            "mtc1       %[filter0], %[ftmp4]                            \n\t"
            "gsldrc1    %[ftmp7],   0x08(%[src2])                       \n\t"
            "pshufh     %[ftmp4],   %[ftmp4],       %[ftmp0]            \n\t"
            "punpcklbh  %[ftmp2],   %[ftmp1],       %[ftmp0]            \n\t"
            "punpckhbh  %[ftmp3],   %[ftmp1],       %[ftmp0]            \n\t"
            "pmullh     %[ftmp2],   %[ftmp2],       %[ftmp4]            \n\t"
            "pmullh     %[ftmp3],   %[ftmp3],       %[ftmp4]            \n\t"
            "paddush    %[ftmp5],   %[ftmp5],       %[ftmp2]            \n\t"
            "paddush    %[ftmp6],   %[ftmp6],       %[ftmp3]            \n\t"
            "punpcklbh  %[ftmp2],   %[ftmp7],       %[ftmp0]            \n\t"
            "punpckhbh  %[ftmp3],   %[ftmp7],       %[ftmp0]            \n\t"
            "pmullh     %[ftmp2],   %[ftmp2],       %[ftmp4]            \n\t"
            "pmullh     %[ftmp3],   %[ftmp3],       %[ftmp4]            \n\t"
            "paddush    %[ftmp8],   %[ftmp8],       %[ftmp2]            \n\t"
            "paddush    %[ftmp9],   %[ftmp9],       %[ftmp3]            \n\t"

            "gsldlc1    %[ftmp1],   0x07(%[src3])                       \n\t"
            "gsldrc1    %[ftmp1],   0x00(%[src3])                       \n\t"
            "gsldlc1    %[ftmp7],   0x0f(%[src3])                       \n\t"
            "mtc1       %[filter3], %[ftmp4]                            \n\t"
            "gsldrc1    %[ftmp7],   0x08(%[src3])                       \n\t"
            "pshufh     %[ftmp4],   %[ftmp4],       %[ftmp0]            \n\t"
            "punpcklbh  %[ftmp2],   %[ftmp1],       %[ftmp0]            \n\t"
            "punpckhbh  %[ftmp3],   %[ftmp1],       %[ftmp0]            \n\t"
            "pmullh     %[ftmp2],   %[ftmp2],       %[ftmp4]            \n\t"
            "pmullh     %[ftmp3],   %[ftmp3],       %[ftmp4]            \n\t"
            "paddush    %[ftmp5],   %[ftmp5],       %[ftmp2]            \n\t"
            "paddush    %[ftmp6],   %[ftmp6],       %[ftmp3]            \n\t"
            "punpcklbh  %[ftmp2],   %[ftmp7],       %[ftmp0]            \n\t"
            "punpckhbh  %[ftmp3],   %[ftmp7],       %[ftmp0]            \n\t"
            "pmullh     %[ftmp2],   %[ftmp2],       %[ftmp4]            \n\t"
            "pmullh     %[ftmp3],   %[ftmp3],       %[ftmp4]            \n\t"
            "paddush    %[ftmp8],   %[ftmp8],       %[ftmp2]            \n\t"
            "paddush    %[ftmp9],   %[ftmp9],       %[ftmp3]            \n\t"

            "gsldlc1    %[ftmp1],   0x07(%[src4])                       \n\t"
            "gsldrc1    %[ftmp1],   0x00(%[src4])                       \n\t"
            "gsldlc1    %[ftmp7],   0x0f(%[src4])                       \n\t"
            "mtc1       %[filter4], %[ftmp4]                            \n\t"
            "gsldrc1    %[ftmp7],   0x08(%[src4])                       \n\t"
            "pshufh     %[ftmp4],   %[ftmp4],       %[ftmp0]            \n\t"
            "punpcklbh  %[ftmp2],   %[ftmp1],       %[ftmp0]            \n\t"
            "punpckhbh  %[ftmp3],   %[ftmp1],       %[ftmp0]            \n\t"
            "pmullh     %[ftmp2],   %[ftmp2],       %[ftmp4]            \n\t"
            "pmullh     %[ftmp3],   %[ftmp3],       %[ftmp4]            \n\t"
            "psubush    %[ftmp5],   %[ftmp5],       %[ftmp2]            \n\t"
            "psubush    %[ftmp6],   %[ftmp6],       %[ftmp3]            \n\t"
            "punpcklbh  %[ftmp2],   %[ftmp7],       %[ftmp0]            \n\t"
            "punpckhbh  %[ftmp3],   %[ftmp7],       %[ftmp0]            \n\t"
            "pmullh     %[ftmp2],   %[ftmp2],       %[ftmp4]            \n\t"
            "pmullh     %[ftmp3],   %[ftmp3],       %[ftmp4]            \n\t"
            "psubush    %[ftmp8],   %[ftmp8],       %[ftmp2]            \n\t"
            "psubush    %[ftmp9],   %[ftmp9],       %[ftmp3]            \n\t"

            "gsldlc1    %[ftmp1],   0x07(%[src5])                       \n\t"
            "gsldrc1    %[ftmp1],   0x00(%[src5])                       \n\t"
            "gsldlc1    %[ftmp7],   0x0f(%[src5])                       \n\t"
            "mtc1       %[filter5], %[ftmp4]                            \n\t"
            "gsldrc1    %[ftmp7],   0x08(%[src5])                       \n\t"
            "pshufh     %[ftmp4],   %[ftmp4],       %[ftmp0]            \n\t"
            "punpcklbh  %[ftmp2],   %[ftmp1],       %[ftmp0]            \n\t"
            "punpckhbh  %[ftmp3],   %[ftmp1],       %[ftmp0]            \n\t"
            "pmullh     %[ftmp2],   %[ftmp2],       %[ftmp4]            \n\t"
            "pmullh     %[ftmp3],   %[ftmp3],       %[ftmp4]            \n\t"
            "paddush    %[ftmp5],   %[ftmp5],       %[ftmp2]            \n\t"
            "paddush    %[ftmp6],   %[ftmp6],       %[ftmp3]            \n\t"
            "punpcklbh  %[ftmp2],   %[ftmp7],       %[ftmp0]            \n\t"
            "punpckhbh  %[ftmp3],   %[ftmp7],       %[ftmp0]            \n\t"
            "pmullh     %[ftmp2],   %[ftmp2],       %[ftmp4]            \n\t"
            "pmullh     %[ftmp3],   %[ftmp3],       %[ftmp4]            \n\t"
            "paddush    %[ftmp8],   %[ftmp8],       %[ftmp2]            \n\t"
            "paddush    %[ftmp9],   %[ftmp9],       %[ftmp3]            \n\t"

            "li         %[tmp0],    0x07                                \n\t"
            "paddush    %[ftmp5],   %[ftmp5],       %[ff_pw_64]         \n\t"
            "paddush    %[ftmp6],   %[ftmp6],       %[ff_pw_64]         \n\t"
            "paddush    %[ftmp8],   %[ftmp8],       %[ff_pw_64]         \n\t"
            "paddush    %[ftmp9],   %[ftmp9],       %[ff_pw_64]         \n\t"
            "mtc1       %[tmp0],    %[ftmp4]                            \n\t"
            "psrlh      %[ftmp5],   %[ftmp5],       %[ftmp4]            \n\t"
            "psrlh      %[ftmp6],   %[ftmp6],       %[ftmp4]            \n\t"
            "psrlh      %[ftmp8],   %[ftmp8],       %[ftmp4]            \n\t"
            "psrlh      %[ftmp9],   %[ftmp9],       %[ftmp4]            \n\t"

            "packushb   %[ftmp1],   %[ftmp5],       %[ftmp6]            \n\t"
            "packushb   %[ftmp2],   %[ftmp8],       %[ftmp9]            \n\t"
            "gssdlc1    %[ftmp1],   0x07(%[dst])                        \n\t"
            "gssdrc1    %[ftmp1],   0x00(%[dst])                        \n\t"
            "gssdlc1    %[ftmp2],   0x0f(%[dst])                        \n\t"
            "gssdrc1    %[ftmp2],   0x08(%[dst])                        \n\t"
            : [ftmp0]"=&f"(ftmp[0]),            [ftmp1]"=&f"(ftmp[1]),
              [ftmp2]"=&f"(ftmp[2]),            [ftmp3]"=&f"(ftmp[3]),
              [ftmp4]"=&f"(ftmp[4]),            [ftmp5]"=&f"(ftmp[5]),
              [ftmp6]"=&f"(ftmp[6]),            [ftmp7]"=&f"(ftmp[7]),
              [ftmp8]"=&f"(ftmp[8]),            [ftmp9]"=&f"(ftmp[9]),
              [tmp0]"=&r"(tmp[0])
            : [dst]"r"(dst),                    [src0]"r"(src),
              [src1]"r"(src-srcstride),         [src2]"r"(src-2*srcstride),
              [src3]"r"(src+srcstride),         [src4]"r"(src+2*srcstride),
              [src5]"r"(src+3*srcstride),       [ff_pw_64]"f"(ff_pw_64),
              [filter0]"r"(filter[0]),          [filter1]"r"(filter[1]),
              [filter2]"r"(filter[2]),          [filter3]"r"(filter[3]),
              [filter4]"r"(filter[4]),          [filter5]"r"(filter[5])
            : "memory"
        );

        dst += dststride;
        src += srcstride;
    }
#else
    const uint8_t *filter = subpel_filters[my - 1];
    const uint8_t *cm     = ff_crop_tab + MAX_NEG_CROP;
    int x, y;

    for (y = 0; y < h; y++) {
        for (x = 0; x < 16; x++)
            dst[x] = FILTER_6TAP(src, filter, srcstride);
        dst += dststride;
        src += srcstride;
    }
#endif
}

void ff_put_vp8_epel8_v6_mmi(uint8_t *dst, ptrdiff_t dststride, uint8_t *src,
        ptrdiff_t srcstride, int h, int mx, int my)
{
#if NOTOK //FIXME fate-vp8-test-vector-002 006 009
    const uint8_t *filter = subpel_filters[my - 1];
    int y;
    double ftmp[7];
    uint64_t tmp[1];

    for (y = 0; y < h; y++) {
        /*
        dst[0] = cm[(filter[2]*src[0] - filter[1]*src[0-srcstride] + filter[0]*src[0-2*srcstride] + filter[3]*src[0+srcstride] - filter[4]*src[0+2*srcstride] + filter[5]*src[0+3*srcstride] + 64) >> 7];
        dst[1] = cm[(filter[2]*src[1] - filter[1]*src[1-srcstride] + filter[0]*src[1-2*srcstride] + filter[3]*src[1+srcstride] - filter[4]*src[1+2*srcstride] + filter[5]*src[1+3*srcstride] + 64) >> 7];
        dst[2] = cm[(filter[2]*src[2] - filter[1]*src[2-srcstride] + filter[0]*src[2-2*srcstride] + filter[3]*src[2+srcstride] - filter[4]*src[2+2*srcstride] + filter[5]*src[2+3*srcstride] + 64) >> 7];
        dst[3] = cm[(filter[2]*src[3] - filter[1]*src[3-srcstride] + filter[0]*src[3-2*srcstride] + filter[3]*src[3+srcstride] - filter[4]*src[3+2*srcstride] + filter[5]*src[3+3*srcstride] + 64) >> 7];
        dst[4] = cm[(filter[2]*src[4] - filter[1]*src[4-srcstride] + filter[0]*src[4-2*srcstride] + filter[3]*src[4+srcstride] - filter[4]*src[4+2*srcstride] + filter[5]*src[4+3*srcstride] + 64) >> 7];
        dst[5] = cm[(filter[2]*src[5] - filter[1]*src[5-srcstride] + filter[0]*src[5-2*srcstride] + filter[3]*src[5+srcstride] - filter[4]*src[5+2*srcstride] + filter[5]*src[5+3*srcstride] + 64) >> 7];
        dst[6] = cm[(filter[2]*src[6] - filter[1]*src[6-srcstride] + filter[0]*src[6-2*srcstride] + filter[3]*src[6+srcstride] - filter[4]*src[6+2*srcstride] + filter[5]*src[6+3*srcstride] + 64) >> 7];
        dst[7] = cm[(filter[2]*src[7] - filter[1]*src[7-srcstride] + filter[0]*src[7-2*srcstride] + filter[3]*src[7+srcstride] - filter[4]*src[7+2*srcstride] + filter[5]*src[7+3*srcstride] + 64) >> 7];
        */
        __asm__ volatile (
            "xor        %[ftmp0],   %[ftmp0],       %[ftmp0]            \n\t"
            "gsldlc1    %[ftmp1],   0x07(%[src0])                       \n\t"
            "mtc1       %[filter2], %[ftmp4]                            \n\t"
            "gsldrc1    %[ftmp1],   0x00(%[src0])                       \n\t"
            "pshufh     %[ftmp4],   %[ftmp4],       %[ftmp0]            \n\t"
            "punpcklbh  %[ftmp2],   %[ftmp1],       %[ftmp0]            \n\t"
            "punpckhbh  %[ftmp3],   %[ftmp1],       %[ftmp0]            \n\t"
            "pmullh     %[ftmp5],   %[ftmp2],       %[ftmp4]            \n\t"
            "pmullh     %[ftmp6],   %[ftmp3],       %[ftmp4]            \n\t"

            "gsldlc1    %[ftmp1],   0x07(%[src1])                       \n\t"
            "mtc1       %[filter1], %[ftmp4]                            \n\t"
            "gsldrc1    %[ftmp1],   0x00(%[src1])                       \n\t"
            "pshufh     %[ftmp4],   %[ftmp4],       %[ftmp0]            \n\t"
            "punpcklbh  %[ftmp2],   %[ftmp1],       %[ftmp0]            \n\t"
            "punpckhbh  %[ftmp3],   %[ftmp1],       %[ftmp0]            \n\t"
            "pmullh     %[ftmp2],   %[ftmp2],       %[ftmp4]            \n\t"
            "pmullh     %[ftmp3],   %[ftmp3],       %[ftmp4]            \n\t"
            "psubush    %[ftmp5],   %[ftmp5],       %[ftmp2]            \n\t"
            "psubush    %[ftmp6],   %[ftmp6],       %[ftmp3]            \n\t"

            "gsldlc1    %[ftmp1],   0x07(%[src2])                       \n\t"
            "mtc1       %[filter0], %[ftmp4]                            \n\t"
            "gsldrc1    %[ftmp1],   0x00(%[src2])                       \n\t"
            "pshufh     %[ftmp4],   %[ftmp4],       %[ftmp0]            \n\t"
            "punpcklbh  %[ftmp2],   %[ftmp1],       %[ftmp0]            \n\t"
            "punpckhbh  %[ftmp3],   %[ftmp1],       %[ftmp0]            \n\t"
            "pmullh     %[ftmp2],   %[ftmp2],       %[ftmp4]            \n\t"
            "pmullh     %[ftmp3],   %[ftmp3],       %[ftmp4]            \n\t"
            "paddush    %[ftmp5],   %[ftmp5],       %[ftmp2]            \n\t"
            "paddush    %[ftmp6],   %[ftmp6],       %[ftmp3]            \n\t"

            "gsldlc1    %[ftmp1],   0x07(%[src3])                       \n\t"
            "mtc1       %[filter3], %[ftmp4]                            \n\t"
            "gsldrc1    %[ftmp1],   0x00(%[src3])                       \n\t"
            "pshufh     %[ftmp4],   %[ftmp4],       %[ftmp0]            \n\t"
            "punpcklbh  %[ftmp2],   %[ftmp1],       %[ftmp0]            \n\t"
            "punpckhbh  %[ftmp3],   %[ftmp1],       %[ftmp0]            \n\t"
            "pmullh     %[ftmp2],   %[ftmp2],       %[ftmp4]            \n\t"
            "pmullh     %[ftmp3],   %[ftmp3],       %[ftmp4]            \n\t"
            "paddush    %[ftmp5],   %[ftmp5],       %[ftmp2]            \n\t"
            "paddush    %[ftmp6],   %[ftmp6],       %[ftmp3]            \n\t"

            "gsldlc1    %[ftmp1],   0x07(%[src4])                       \n\t"
            "mtc1       %[filter4], %[ftmp4]                            \n\t"
            "gsldrc1    %[ftmp1],   0x00(%[src4])                       \n\t"
            "pshufh     %[ftmp4],   %[ftmp4],       %[ftmp0]            \n\t"
            "punpcklbh  %[ftmp2],   %[ftmp1],       %[ftmp0]            \n\t"
            "punpckhbh  %[ftmp3],   %[ftmp1],       %[ftmp0]            \n\t"
            "pmullh     %[ftmp2],   %[ftmp2],       %[ftmp4]            \n\t"
            "pmullh     %[ftmp3],   %[ftmp3],       %[ftmp4]            \n\t"
            "psubush    %[ftmp5],   %[ftmp5],       %[ftmp2]            \n\t"
            "psubush    %[ftmp6],   %[ftmp6],       %[ftmp3]            \n\t"

            "gsldlc1    %[ftmp1],   0x07(%[src5])                       \n\t"
            "mtc1       %[filter5], %[ftmp4]                            \n\t"
            "gsldrc1    %[ftmp1],   0x00(%[src5])                       \n\t"
            "pshufh     %[ftmp4],   %[ftmp4],       %[ftmp0]            \n\t"
            "punpcklbh  %[ftmp2],   %[ftmp1],       %[ftmp0]            \n\t"
            "punpckhbh  %[ftmp3],   %[ftmp1],       %[ftmp0]            \n\t"
            "pmullh     %[ftmp2],   %[ftmp2],       %[ftmp4]            \n\t"
            "pmullh     %[ftmp3],   %[ftmp3],       %[ftmp4]            \n\t"
            "paddush    %[ftmp5],   %[ftmp5],       %[ftmp2]            \n\t"
            "paddush    %[ftmp6],   %[ftmp6],       %[ftmp3]            \n\t"

            "li         %[tmp0],    0x07                                \n\t"
            "paddush    %[ftmp5],   %[ftmp5],       %[ff_pw_64]         \n\t"
            "paddush    %[ftmp6],   %[ftmp6],       %[ff_pw_64]         \n\t"
            "mtc1       %[tmp0],    %[ftmp4]                            \n\t"
            "psrlh      %[ftmp5],   %[ftmp5],       %[ftmp4]            \n\t"
            "psrlh      %[ftmp6],   %[ftmp6],       %[ftmp4]            \n\t"

            "packushb   %[ftmp1],   %[ftmp5],       %[ftmp6]            \n\t"
            "gssdlc1    %[ftmp1],   0x07(%[dst])                        \n\t"
            "gssdrc1    %[ftmp1],   0x00(%[dst])                        \n\t"
            : [ftmp0]"=&f"(ftmp[0]),            [ftmp1]"=&f"(ftmp[1]),
              [ftmp2]"=&f"(ftmp[2]),            [ftmp3]"=&f"(ftmp[3]),
              [ftmp4]"=&f"(ftmp[4]),            [ftmp5]"=&f"(ftmp[5]),
              [ftmp6]"=&f"(ftmp[6]),
              [tmp0]"=&r"(tmp[0])
            : [dst]"r"(dst),                    [src0]"r"(src),
              [src1]"r"(src-srcstride),         [src2]"r"(src-2*srcstride),
              [src3]"r"(src+srcstride),         [src4]"r"(src+2*srcstride),
              [src5]"r"(src+3*srcstride),       [ff_pw_64]"f"(ff_pw_64),
              [filter0]"r"(filter[0]),          [filter1]"r"(filter[1]),
              [filter2]"r"(filter[2]),          [filter3]"r"(filter[3]),
              [filter4]"r"(filter[4]),          [filter5]"r"(filter[5])
            : "memory"
        );

        dst += dststride;
        src += srcstride;
    }
#else
    const uint8_t *filter = subpel_filters[my - 1];
    const uint8_t *cm     = ff_crop_tab + MAX_NEG_CROP;
    int x, y;

    for (y = 0; y < h; y++) {
        for (x = 0; x < 8; x++)
            dst[x] = FILTER_6TAP(src, filter, srcstride);
        dst += dststride;
        src += srcstride;
    }
#endif
}

void ff_put_vp8_epel4_v6_mmi(uint8_t *dst, ptrdiff_t dststride, uint8_t *src,
        ptrdiff_t srcstride, int h, int mx, int my)
{
#if NOTOK //FIXME fate-vp8-test-vector-002 006 009
    const uint8_t *filter = subpel_filters[my - 1];
    int y;
    double ftmp[5];
    uint64_t tmp[1];

    for (y = 0; y < h; y++) {
        /*
        dst[0] = cm[(filter[2]*src[0] - filter[1]*src[0-srcstride] + filter[0]*src[0-2*srcstride] + filter[3]*src[0+srcstride] - filter[4]*src[0+2*srcstride] + filter[5]*src[0+3*srcstride] + 64) >> 7];
        dst[1] = cm[(filter[2]*src[1] - filter[1]*src[1-srcstride] + filter[0]*src[1-2*srcstride] + filter[3]*src[1+srcstride] - filter[4]*src[1+2*srcstride] + filter[5]*src[1+3*srcstride] + 64) >> 7];
        dst[2] = cm[(filter[2]*src[2] - filter[1]*src[2-srcstride] + filter[0]*src[2-2*srcstride] + filter[3]*src[2+srcstride] - filter[4]*src[2+2*srcstride] + filter[5]*src[2+3*srcstride] + 64) >> 7];
        dst[3] = cm[(filter[2]*src[3] - filter[1]*src[3-srcstride] + filter[0]*src[3-2*srcstride] + filter[3]*src[3+srcstride] - filter[4]*src[3+2*srcstride] + filter[5]*src[3+3*srcstride] + 64) >> 7];
        */
        __asm__ volatile (
            "xor        %[ftmp0],   %[ftmp0],       %[ftmp0]            \n\t"
            "gslwlc1    %[ftmp1],   0x03(%[src0])                       \n\t"
            "mtc1       %[filter2], %[ftmp3]                            \n\t"
            "gslwrc1    %[ftmp1],   0x00(%[src0])                       \n\t"
            "pshufh     %[ftmp3],   %[ftmp3],       %[ftmp0]            \n\t"
            "punpcklbh  %[ftmp2],   %[ftmp1],       %[ftmp0]            \n\t"
            "pmullh     %[ftmp4],   %[ftmp2],       %[ftmp3]            \n\t"

            "gslwlc1    %[ftmp1],   0x03(%[src1])                       \n\t"
            "mtc1       %[filter1], %[ftmp3]                            \n\t"
            "gslwrc1    %[ftmp1],   0x00(%[src1])                       \n\t"
            "pshufh     %[ftmp3],   %[ftmp3],       %[ftmp0]            \n\t"
            "punpcklbh  %[ftmp2],   %[ftmp1],       %[ftmp0]            \n\t"
            "pmullh     %[ftmp2],   %[ftmp2],       %[ftmp3]            \n\t"
            "psubush    %[ftmp4],   %[ftmp4],       %[ftmp2]            \n\t"

            "gslwlc1    %[ftmp1],   0x03(%[src2])                       \n\t"
            "mtc1       %[filter0], %[ftmp3]                            \n\t"
            "gslwrc1    %[ftmp1],   0x00(%[src2])                       \n\t"
            "pshufh     %[ftmp3],   %[ftmp3],       %[ftmp0]            \n\t"
            "punpcklbh  %[ftmp2],   %[ftmp1],       %[ftmp0]            \n\t"
            "pmullh     %[ftmp2],   %[ftmp2],       %[ftmp3]            \n\t"
            "paddush    %[ftmp4],   %[ftmp4],       %[ftmp2]            \n\t"

            "gslwlc1    %[ftmp1],   0x03(%[src3])                       \n\t"
            "mtc1       %[filter3], %[ftmp3]                            \n\t"
            "gslwrc1    %[ftmp1],   0x00(%[src3])                       \n\t"
            "pshufh     %[ftmp3],   %[ftmp3],       %[ftmp0]            \n\t"
            "punpcklbh  %[ftmp2],   %[ftmp1],       %[ftmp0]            \n\t"
            "pmullh     %[ftmp2],   %[ftmp2],       %[ftmp3]            \n\t"
            "paddush    %[ftmp4],   %[ftmp4],       %[ftmp2]            \n\t"

            "gslwlc1    %[ftmp1],   0x03(%[src4])                       \n\t"
            "mtc1       %[filter4], %[ftmp3]                            \n\t"
            "gslwrc1    %[ftmp1],   0x00(%[src4])                       \n\t"
            "pshufh     %[ftmp3],   %[ftmp3],       %[ftmp0]            \n\t"
            "punpcklbh  %[ftmp2],   %[ftmp1],       %[ftmp0]            \n\t"
            "pmullh     %[ftmp2],   %[ftmp2],       %[ftmp3]            \n\t"
            "psubush    %[ftmp4],   %[ftmp4],       %[ftmp2]            \n\t"

            "gslwlc1    %[ftmp1],   0x03(%[src5])                       \n\t"
            "mtc1       %[filter5], %[ftmp3]                            \n\t"
            "gslwrc1    %[ftmp1],   0x00(%[src5])                       \n\t"
            "pshufh     %[ftmp3],   %[ftmp3],       %[ftmp0]            \n\t"
            "punpcklbh  %[ftmp2],   %[ftmp1],       %[ftmp0]            \n\t"
            "pmullh     %[ftmp2],   %[ftmp2],       %[ftmp3]            \n\t"
            "paddush    %[ftmp4],   %[ftmp4],       %[ftmp2]            \n\t"

            "li         %[tmp0],    0x07                                \n\t"
            "paddush    %[ftmp4],   %[ftmp4],       %[ff_pw_64]         \n\t"
            "mtc1       %[tmp0],    %[ftmp3]                            \n\t"
            "psrlh      %[ftmp4],   %[ftmp4],       %[ftmp3]            \n\t"

            "packushb   %[ftmp1],   %[ftmp4],       %[ftmp0]            \n\t"
            "gsswlc1    %[ftmp1],   0x03(%[dst])                        \n\t"
            "gsswrc1    %[ftmp1],   0x00(%[dst])                        \n\t"
            : [ftmp0]"=&f"(ftmp[0]),            [ftmp1]"=&f"(ftmp[1]),
              [ftmp2]"=&f"(ftmp[2]),            [ftmp3]"=&f"(ftmp[3]),
              [ftmp4]"=&f"(ftmp[4]),
              [tmp0]"=&r"(tmp[0])
            : [dst]"r"(dst),                    [src0]"r"(src),
              [src1]"r"(src-srcstride),         [src2]"r"(src-2*srcstride),
              [src3]"r"(src+srcstride),         [src4]"r"(src+2*srcstride),
              [src5]"r"(src+3*srcstride),       [ff_pw_64]"f"(ff_pw_64),
              [filter0]"r"(filter[0]),          [filter1]"r"(filter[1]),
              [filter2]"r"(filter[2]),          [filter3]"r"(filter[3]),
              [filter4]"r"(filter[4]),          [filter5]"r"(filter[5])
            : "memory"
        );

        dst += dststride;
        src += srcstride;
    }
#else
    const uint8_t *filter = subpel_filters[my - 1];
    const uint8_t *cm     = ff_crop_tab + MAX_NEG_CROP;
    int x, y;

    for (y = 0; y < h; y++) {
        for (x = 0; x < 4; x++)
            dst[x] = FILTER_6TAP(src, filter, srcstride);
        dst += dststride;
        src += srcstride;
    }
#endif
}

void ff_put_vp8_epel16_h4v4_mmi(uint8_t *dst, ptrdiff_t dststride, uint8_t *src,
        ptrdiff_t srcstride, int h, int mx, int my)
{
#if OK
    const uint8_t *filter = subpel_filters[mx - 1];
    int y;
    uint8_t tmp_array[560];
    uint8_t *tmp = tmp_array;
    double ftmp[10];
    uint64_t tmp0;

    src -= srcstride;

    for (y = 0; y < h + 3; y++) {
        /*
        tmp[0] = cm[(filter[2] * src[0] - filter[1] * src[-1] + filter[3] * src[1] - filter[4] * src[2] + 64) >> 7];
        tmp[1] = cm[(filter[2] * src[1] - filter[1] * src[ 0] + filter[3] * src[2] - filter[4] * src[3] + 64) >> 7];
        tmp[2] = cm[(filter[2] * src[2] - filter[1] * src[ 1] + filter[3] * src[3] - filter[4] * src[4] + 64) >> 7];
        tmp[3] = cm[(filter[2] * src[3] - filter[1] * src[ 2] + filter[3] * src[4] - filter[4] * src[5] + 64) >> 7];
        tmp[4] = cm[(filter[2] * src[4] - filter[1] * src[ 3] + filter[3] * src[5] - filter[4] * src[6] + 64) >> 7];
        tmp[5] = cm[(filter[2] * src[5] - filter[1] * src[ 4] + filter[3] * src[6] - filter[4] * src[7] + 64) >> 7];
        tmp[6] = cm[(filter[2] * src[6] - filter[1] * src[ 5] + filter[3] * src[7] - filter[4] * src[8] + 64) >> 7];
        tmp[7] = cm[(filter[2] * src[7] - filter[1] * src[ 6] + filter[3] * src[8] - filter[4] * src[9] + 64) >> 7];

        tmp[ 8] = cm[(filter[2] * src[ 8] - filter[1] * src[ 7] + filter[3] * src[ 9] - filter[4] * src[10] + 64) >> 7];
        tmp[ 9] = cm[(filter[2] * src[ 9] - filter[1] * src[ 8] + filter[3] * src[10] - filter[4] * src[11] + 64) >> 7];
        tmp[10] = cm[(filter[2] * src[10] - filter[1] * src[ 9] + filter[3] * src[11] - filter[4] * src[12] + 64) >> 7];
        tmp[11] = cm[(filter[2] * src[11] - filter[1] * src[10] + filter[3] * src[12] - filter[4] * src[13] + 64) >> 7];
        tmp[12] = cm[(filter[2] * src[12] - filter[1] * src[11] + filter[3] * src[13] - filter[4] * src[14] + 64) >> 7];
        tmp[13] = cm[(filter[2] * src[13] - filter[1] * src[12] + filter[3] * src[14] - filter[4] * src[15] + 64) >> 7];
        tmp[14] = cm[(filter[2] * src[14] - filter[1] * src[13] + filter[3] * src[15] - filter[4] * src[16] + 64) >> 7];
        tmp[15] = cm[(filter[2] * src[15] - filter[1] * src[14] + filter[3] * src[16] - filter[4] * src[17] + 64) >> 7];
        */
        __asm__ volatile (
            "xor        %[ftmp0],   %[ftmp0], %[ftmp0]                  \n\t"
            "gsldlc1    %[ftmp1],   0x07(%[src])                        \n\t"
            "gsldrc1    %[ftmp1],   0x00(%[src])                        \n\t"
            "gsldlc1    %[ftmp7],   0x0f(%[src])                        \n\t"
            "mtc1       %[filter2], %[ftmp4]                            \n\t"
            "gsldrc1    %[ftmp7],   0x08(%[src])                        \n\t"
            "pshufh     %[ftmp4],   %[ftmp4], %[ftmp0]                  \n\t"
            "punpcklbh  %[ftmp2],   %[ftmp1], %[ftmp0]                  \n\t"
            "punpckhbh  %[ftmp3],   %[ftmp1], %[ftmp0]                  \n\t"
            "pmullh     %[ftmp5],   %[ftmp2], %[ftmp4]                  \n\t"
            "pmullh     %[ftmp6],   %[ftmp3], %[ftmp4]                  \n\t"
            "punpcklbh  %[ftmp2],   %[ftmp7], %[ftmp0]                  \n\t"
            "punpckhbh  %[ftmp3],   %[ftmp7], %[ftmp0]                  \n\t"
            "pmullh     %[ftmp8],   %[ftmp2], %[ftmp4]                  \n\t"
            "pmullh     %[ftmp9],   %[ftmp3], %[ftmp4]                  \n\t"

            "gsldlc1    %[ftmp1],   0x06(%[src])                        \n\t"
            "gsldrc1    %[ftmp1],  -0x01(%[src])                        \n\t"
            "gsldlc1    %[ftmp7],   0x0e(%[src])                        \n\t"
            "mtc1       %[filter1], %[ftmp4]                            \n\t"
            "gsldrc1    %[ftmp7],   0x07(%[src])                        \n\t"
            "pshufh     %[ftmp4],   %[ftmp4], %[ftmp0]                  \n\t"
            "punpcklbh  %[ftmp2],   %[ftmp1], %[ftmp0]                  \n\t"
            "punpckhbh  %[ftmp3],   %[ftmp1], %[ftmp0]                  \n\t"
            "pmullh     %[ftmp2],   %[ftmp2], %[ftmp4]                  \n\t"
            "pmullh     %[ftmp3],   %[ftmp3], %[ftmp4]                  \n\t"
            "psubush    %[ftmp5],   %[ftmp5], %[ftmp2]                  \n\t"
            "psubush    %[ftmp6],   %[ftmp6], %[ftmp3]                  \n\t"
            "punpcklbh  %[ftmp2],   %[ftmp7], %[ftmp0]                  \n\t"
            "punpckhbh  %[ftmp3],   %[ftmp7], %[ftmp0]                  \n\t"
            "pmullh     %[ftmp2],   %[ftmp2], %[ftmp4]                  \n\t"
            "pmullh     %[ftmp3],   %[ftmp3], %[ftmp4]                  \n\t"
            "psubush    %[ftmp8],   %[ftmp8], %[ftmp2]                  \n\t"
            "psubush    %[ftmp9],   %[ftmp9], %[ftmp3]                  \n\t"

            "gsldlc1    %[ftmp1],   0x08(%[src])                        \n\t"
            "gsldrc1    %[ftmp1],   0x01(%[src])                        \n\t"
            "gsldlc1    %[ftmp7],   0x0d(%[src])                        \n\t"
            "mtc1       %[filter3], %[ftmp4]                            \n\t"
            "gsldrc1    %[ftmp7],   0x06(%[src])                        \n\t"
            "pshufh     %[ftmp4],   %[ftmp4], %[ftmp0]                  \n\t"
            "punpcklbh  %[ftmp2],   %[ftmp1], %[ftmp0]                  \n\t"
            "punpckhbh  %[ftmp3],   %[ftmp1], %[ftmp0]                  \n\t"
            "pmullh     %[ftmp2],   %[ftmp2], %[ftmp4]                  \n\t"
            "pmullh     %[ftmp3],   %[ftmp3], %[ftmp4]                  \n\t"
            "paddush    %[ftmp5],   %[ftmp5], %[ftmp2]                  \n\t"
            "paddush    %[ftmp6],   %[ftmp6], %[ftmp3]                  \n\t"
            "punpcklbh  %[ftmp2],   %[ftmp7], %[ftmp0]                  \n\t"
            "punpckhbh  %[ftmp3],   %[ftmp7], %[ftmp0]                  \n\t"
            "pmullh     %[ftmp2],   %[ftmp2], %[ftmp4]                  \n\t"
            "pmullh     %[ftmp3],   %[ftmp3], %[ftmp4]                  \n\t"
            "paddush    %[ftmp8],   %[ftmp8], %[ftmp2]                  \n\t"
            "paddush    %[ftmp9],   %[ftmp9], %[ftmp3]                  \n\t"

            "gsldlc1    %[ftmp1],   0x09(%[src])                        \n\t"
            "gsldrc1    %[ftmp1],   0x02(%[src])                        \n\t"
            "gsldlc1    %[ftmp7],   0x11(%[src])                        \n\t"
            "mtc1       %[filter4], %[ftmp4]                            \n\t"
            "gsldrc1    %[ftmp7],   0x0a(%[src])                        \n\t"
            "pshufh     %[ftmp4],   %[ftmp4], %[ftmp0]                  \n\t"
            "punpcklbh  %[ftmp2],   %[ftmp1], %[ftmp0]                  \n\t"
            "punpckhbh  %[ftmp3],   %[ftmp1], %[ftmp0]                  \n\t"
            "pmullh     %[ftmp2],   %[ftmp2], %[ftmp4]                  \n\t"
            "pmullh     %[ftmp3],   %[ftmp3], %[ftmp4]                  \n\t"
            "psubush    %[ftmp5],   %[ftmp5], %[ftmp2]                  \n\t"
            "psubush    %[ftmp6],   %[ftmp6], %[ftmp3]                  \n\t"
            "punpcklbh  %[ftmp2],   %[ftmp7], %[ftmp0]                  \n\t"
            "punpckhbh  %[ftmp3],   %[ftmp7], %[ftmp0]                  \n\t"
            "pmullh     %[ftmp2],   %[ftmp2], %[ftmp4]                  \n\t"
            "pmullh     %[ftmp3],   %[ftmp3], %[ftmp4]                  \n\t"
            "psubush    %[ftmp8],   %[ftmp8], %[ftmp2]                  \n\t"
            "psubush    %[ftmp9],   %[ftmp9], %[ftmp3]                  \n\t"

            "li         %[tmp0],    0x07                                \n\t"
            "paddush    %[ftmp5],   %[ftmp5], %[ff_pw_64]               \n\t"
            "paddush    %[ftmp6],   %[ftmp6], %[ff_pw_64]               \n\t"
            "mtc1       %[tmp0],    %[ftmp4]                            \n\t"
            "psrlh      %[ftmp5],   %[ftmp5], %[ftmp4]                  \n\t"
            "psrlh      %[ftmp6],   %[ftmp6], %[ftmp4]                  \n\t"

            "packushb   %[ftmp1],   %[ftmp5], %[ftmp6]                  \n\t"
            "gssdlc1    %[ftmp1],   0x07(%[tmp])                        \n\t"
            "gssdrc1    %[ftmp1],   0x00(%[tmp])                        \n\t"
            "gssdlc1    %[ftmp1],   0x0f(%[tmp])                        \n\t"
            "gssdrc1    %[ftmp1],   0x08(%[tmp])                        \n\t"
            : [ftmp0]"=&f"(ftmp[0]),            [ftmp1]"=&f"(ftmp[1]),
              [ftmp2]"=&f"(ftmp[2]),            [ftmp3]"=&f"(ftmp[3]),
              [ftmp4]"=&f"(ftmp[4]),            [ftmp5]"=&f"(ftmp[5]),
              [ftmp6]"=&f"(ftmp[6]),            [ftmp7]"=&f"(ftmp[7]),
              [ftmp8]"=&f"(ftmp[8]),            [ftmp9]"=&f"(ftmp[9]),
              [tmp0]"=&r"(tmp0)
            : [tmp]"r"(tmp),                    [src]"r"(src),
              [ff_pw_64]"f"(ff_pw_64),
              [filter1]"r"(filter[1]),          [filter2]"r"(filter[2]),
              [filter3]"r"(filter[3]),          [filter4]"r"(filter[4])
            : "memory"
        );

        tmp += 16;
        src += srcstride;
    }
#else
    const uint8_t *filter = subpel_filters[mx - 1];
    const uint8_t *cm     = ff_crop_tab + MAX_NEG_CROP;
    int x, y;
    uint8_t tmp_array[560];
    uint8_t *tmp = tmp_array;

    src -= srcstride;

    for (y = 0; y < h + 3; y++) {
        for (x = 0; x < 16; x++)
            tmp[x] = FILTER_4TAP(src, filter, 1);
        tmp += 16;
        src += srcstride;
    }
#endif

    tmp    = tmp_array + 16;
    filter = subpel_filters[my - 1];

#if OK
    for (y = 0; y < h; y++) {
        /*
        dst[0] = cm[(filter[2] * tmp[0] - filter[1] * tmp[-16] + filter[3] * tmp[16] - filter[4] * tmp[32] + 64) >> 7];
        dst[1] = cm[(filter[2] * tmp[1] - filter[1] * tmp[-15] + filter[3] * tmp[17] - filter[4] * tmp[33] + 64) >> 7];
        dst[2] = cm[(filter[2] * tmp[2] - filter[1] * tmp[-14] + filter[3] * tmp[18] - filter[4] * tmp[34] + 64) >> 7];
        dst[3] = cm[(filter[2] * tmp[3] - filter[1] * tmp[-13] + filter[3] * tmp[19] - filter[4] * tmp[35] + 64) >> 7];
        dst[4] = cm[(filter[2] * tmp[4] - filter[1] * tmp[-12] + filter[3] * tmp[20] - filter[4] * tmp[36] + 64) >> 7];
        dst[5] = cm[(filter[2] * tmp[5] - filter[1] * tmp[-11] + filter[3] * tmp[21] - filter[4] * tmp[37] + 64) >> 7];
        dst[6] = cm[(filter[2] * tmp[6] - filter[1] * tmp[-10] + filter[3] * tmp[22] - filter[4] * tmp[38] + 64) >> 7];
        dst[7] = cm[(filter[2] * tmp[7] - filter[1] * tmp[ -9] + filter[3] * tmp[23] - filter[4] * tmp[39] + 64) >> 7];

        dst[ 8] = cm[(filter[2] * tmp[ 8] - filter[1] * tmp[-8] + filter[3] * tmp[24] - filter[4] * tmp[40] + 64) >> 7];
        dst[ 9] = cm[(filter[2] * tmp[ 9] - filter[1] * tmp[-7] + filter[3] * tmp[25] - filter[4] * tmp[41] + 64) >> 7];
        dst[10] = cm[(filter[2] * tmp[10] - filter[1] * tmp[-6] + filter[3] * tmp[26] - filter[4] * tmp[42] + 64) >> 7];
        dst[11] = cm[(filter[2] * tmp[11] - filter[1] * tmp[-5] + filter[3] * tmp[27] - filter[4] * tmp[43] + 64) >> 7];
        dst[12] = cm[(filter[2] * tmp[12] - filter[1] * tmp[-4] + filter[3] * tmp[28] - filter[4] * tmp[44] + 64) >> 7];
        dst[13] = cm[(filter[2] * tmp[13] - filter[1] * tmp[-3] + filter[3] * tmp[29] - filter[4] * tmp[45] + 64) >> 7];
        dst[14] = cm[(filter[2] * tmp[14] - filter[1] * tmp[-2] + filter[3] * tmp[30] - filter[4] * tmp[46] + 64) >> 7];
        dst[15] = cm[(filter[2] * tmp[15] - filter[1] * tmp[-1] + filter[3] * tmp[31] - filter[4] * tmp[47] + 64) >> 7];
        */
        __asm__ volatile (
            "xor        %[ftmp0],   %[ftmp0],       %[ftmp0]            \n\t"
            "gsldlc1    %[ftmp1],   0x07(%[tmp])                        \n\t"
            "gsldrc1    %[ftmp1],   0x00(%[tmp])                        \n\t"
            "gsldlc1    %[ftmp7],   0x0f(%[tmp])                        \n\t"
            "mtc1       %[filter2], %[ftmp4]                            \n\t"
            "gsldrc1    %[ftmp7],   0x08(%[tmp])                        \n\t"
            "pshufh     %[ftmp4],   %[ftmp4],       %[ftmp0]            \n\t"
            "punpcklbh  %[ftmp2],   %[ftmp1],       %[ftmp0]            \n\t"
            "punpckhbh  %[ftmp3],   %[ftmp1],       %[ftmp0]            \n\t"
            "pmullh     %[ftmp5],   %[ftmp2],       %[ftmp4]            \n\t"
            "pmullh     %[ftmp6],   %[ftmp3],       %[ftmp4]            \n\t"
            "punpcklbh  %[ftmp2],   %[ftmp7],       %[ftmp0]            \n\t"
            "punpckhbh  %[ftmp3],   %[ftmp7],       %[ftmp0]            \n\t"
            "pmullh     %[ftmp8],   %[ftmp2],       %[ftmp4]            \n\t"
            "pmullh     %[ftmp9],   %[ftmp3],       %[ftmp4]            \n\t"

            "gsldlc1    %[ftmp1],  -0x09(%[tmp])                        \n\t"
            "gsldrc1    %[ftmp1],  -0x0f(%[tmp])                        \n\t"
            "gsldlc1    %[ftmp7],  -0x01(%[tmp])                        \n\t"
            "mtc1       %[filter1], %[ftmp4]                            \n\t"
            "gsldrc1    %[ftmp7],  -0x08(%[tmp])                        \n\t"
            "pshufh     %[ftmp4],   %[ftmp4],       %[ftmp0]            \n\t"
            "punpcklbh  %[ftmp2],   %[ftmp1],       %[ftmp0]            \n\t"
            "punpckhbh  %[ftmp3],   %[ftmp1],       %[ftmp0]            \n\t"
            "pmullh     %[ftmp2],   %[ftmp2],       %[ftmp4]            \n\t"
            "pmullh     %[ftmp3],   %[ftmp3],       %[ftmp4]            \n\t"
            "psubush    %[ftmp5],   %[ftmp5],       %[ftmp2]            \n\t"
            "psubush    %[ftmp6],   %[ftmp6],       %[ftmp3]            \n\t"
            "punpcklbh  %[ftmp2],   %[ftmp7],       %[ftmp0]            \n\t"
            "punpckhbh  %[ftmp3],   %[ftmp7],       %[ftmp0]            \n\t"
            "pmullh     %[ftmp2],   %[ftmp2],       %[ftmp4]            \n\t"
            "pmullh     %[ftmp3],   %[ftmp3],       %[ftmp4]            \n\t"
            "psubush    %[ftmp8],   %[ftmp8],       %[ftmp2]            \n\t"
            "psubush    %[ftmp9],   %[ftmp9],       %[ftmp3]            \n\t"

            "gsldlc1    %[ftmp1],   0x17(%[tmp])                        \n\t"
            "gsldrc1    %[ftmp1],   0x10(%[tmp])                        \n\t"
            "gsldlc1    %[ftmp7],   0x1f(%[tmp])                        \n\t"
            "mtc1       %[filter3], %[ftmp4]                            \n\t"
            "gsldrc1    %[ftmp7],   0x18(%[tmp])                        \n\t"
            "pshufh     %[ftmp4],   %[ftmp4],       %[ftmp0]            \n\t"
            "punpcklbh  %[ftmp2],   %[ftmp1],       %[ftmp0]            \n\t"
            "punpckhbh  %[ftmp3],   %[ftmp1],       %[ftmp0]            \n\t"
            "pmullh     %[ftmp2],   %[ftmp2],       %[ftmp4]            \n\t"
            "pmullh     %[ftmp3],   %[ftmp3],       %[ftmp4]            \n\t"
            "paddush    %[ftmp5],   %[ftmp5],       %[ftmp2]            \n\t"
            "paddush    %[ftmp6],   %[ftmp6],       %[ftmp3]            \n\t"
            "punpcklbh  %[ftmp2],   %[ftmp7],       %[ftmp0]            \n\t"
            "punpckhbh  %[ftmp3],   %[ftmp7],       %[ftmp0]            \n\t"
            "pmullh     %[ftmp2],   %[ftmp2],       %[ftmp4]            \n\t"
            "pmullh     %[ftmp3],   %[ftmp3],       %[ftmp4]            \n\t"
            "paddush    %[ftmp8],   %[ftmp8],       %[ftmp2]            \n\t"
            "paddush    %[ftmp9],   %[ftmp9],       %[ftmp3]            \n\t"

            "gsldlc1    %[ftmp1],   0x27(%[tmp])                        \n\t"
            "gsldrc1    %[ftmp1],   0x20(%[tmp])                        \n\t"
            "gsldlc1    %[ftmp7],   0x2f(%[tmp])                        \n\t"
            "mtc1       %[filter4], %[ftmp4]                            \n\t"
            "gsldrc1    %[ftmp7],   0x28(%[tmp])                        \n\t"
            "pshufh     %[ftmp4],   %[ftmp4],       %[ftmp0]            \n\t"
            "punpcklbh  %[ftmp2],   %[ftmp1],       %[ftmp0]            \n\t"
            "punpckhbh  %[ftmp3],   %[ftmp1],       %[ftmp0]            \n\t"
            "pmullh     %[ftmp2],   %[ftmp2],       %[ftmp4]            \n\t"
            "pmullh     %[ftmp3],   %[ftmp3],       %[ftmp4]            \n\t"
            "psubush    %[ftmp5],   %[ftmp5],       %[ftmp2]            \n\t"
            "psubush    %[ftmp6],   %[ftmp6],       %[ftmp3]            \n\t"
            "punpcklbh  %[ftmp2],   %[ftmp7],       %[ftmp0]            \n\t"
            "punpckhbh  %[ftmp3],   %[ftmp7],       %[ftmp0]            \n\t"
            "pmullh     %[ftmp2],   %[ftmp2],       %[ftmp4]            \n\t"
            "pmullh     %[ftmp3],   %[ftmp3],       %[ftmp4]            \n\t"
            "psubush    %[ftmp8],   %[ftmp8],       %[ftmp2]            \n\t"
            "psubush    %[ftmp9],   %[ftmp9],       %[ftmp3]            \n\t"

            "li         %[tmp0],    0x07                                \n\t"
            "paddush    %[ftmp5],   %[ftmp5],       %[ff_pw_64]         \n\t"
            "paddush    %[ftmp6],   %[ftmp6],       %[ff_pw_64]         \n\t"
            "mtc1       %[tmp0],    %[ftmp4]                            \n\t"
            "psrlh      %[ftmp5],   %[ftmp5],       %[ftmp4]            \n\t"
            "psrlh      %[ftmp6],   %[ftmp6],       %[ftmp4]            \n\t"

            "packushb   %[ftmp1],   %[ftmp5],       %[ftmp6]            \n\t"
            "gssdlc1    %[ftmp1],   0x07(%[dst])                        \n\t"
            "gssdrc1    %[ftmp1],   0x00(%[dst])                        \n\t"
            "gssdlc1    %[ftmp1],   0x0f(%[dst])                        \n\t"
            "gssdrc1    %[ftmp1],   0x08(%[dst])                        \n\t"
            : [ftmp0]"=&f"(ftmp[0]),            [ftmp1]"=&f"(ftmp[1]),
              [ftmp2]"=&f"(ftmp[2]),            [ftmp3]"=&f"(ftmp[3]),
              [ftmp4]"=&f"(ftmp[4]),            [ftmp5]"=&f"(ftmp[5]),
              [ftmp6]"=&f"(ftmp[6]),            [ftmp7]"=&f"(ftmp[7]),
              [ftmp8]"=&f"(ftmp[8]),            [ftmp9]"=&f"(ftmp[9]),
              [tmp0]"=&r"(tmp0)
            : [dst]"r"(dst),                    [tmp]"r"(tmp),
              [ff_pw_64]"f"(ff_pw_64),
              [filter1]"r"(filter[1]),          [filter2]"r"(filter[2]),
              [filter3]"r"(filter[3]),          [filter4]"r"(filter[4])
            : "memory"
        );

        dst += dststride;
        tmp += 16;
    }
#else
    for (y = 0; y < h; y++) {
        for (x = 0; x < 16; x++)
            dst[x] = FILTER_4TAP(tmp, filter, 16);
        dst += dststride;
        tmp += 16;
    }
#endif
}

void ff_put_vp8_epel8_h4v4_mmi(uint8_t *dst, ptrdiff_t dststride, uint8_t *src,
        ptrdiff_t srcstride, int h, int mx, int my)
{
#if NOTOK //FIXME fate-vp8-test-vector-006
    const uint8_t *filter = subpel_filters[mx - 1];
    int y;
    uint8_t tmp_array[152];
    uint8_t *tmp = tmp_array;
    double ftmp[7];
    uint64_t tmp0;

    src -= srcstride;

    for (y = 0; y < h + 3; y++) {
        /*
        tmp[0] = cm[(filter[2] * src[0] - filter[1] * src[-1] + filter[3] * src[1] - filter[4] * src[2] + 64) >> 7];
        tmp[1] = cm[(filter[2] * src[1] - filter[1] * src[ 0] + filter[3] * src[2] - filter[4] * src[3] + 64) >> 7];
        tmp[2] = cm[(filter[2] * src[2] - filter[1] * src[ 1] + filter[3] * src[3] - filter[4] * src[4] + 64) >> 7];
        tmp[3] = cm[(filter[2] * src[3] - filter[1] * src[ 2] + filter[3] * src[4] - filter[4] * src[5] + 64) >> 7];
        tmp[4] = cm[(filter[2] * src[4] - filter[1] * src[ 3] + filter[3] * src[5] - filter[4] * src[6] + 64) >> 7];
        tmp[5] = cm[(filter[2] * src[5] - filter[1] * src[ 4] + filter[3] * src[6] - filter[4] * src[7] + 64) >> 7];
        tmp[6] = cm[(filter[2] * src[6] - filter[1] * src[ 5] + filter[3] * src[7] - filter[4] * src[8] + 64) >> 7];
        tmp[7] = cm[(filter[2] * src[7] - filter[1] * src[ 6] + filter[3] * src[8] - filter[4] * src[9] + 64) >> 7];
        */
        __asm__ volatile (
            "xor        %[ftmp0],   %[ftmp0],       %[ftmp0]            \n\t"
            "gsldlc1    %[ftmp1],   0x07(%[src])                        \n\t"
            "mtc1       %[filter2], %[ftmp4]                            \n\t"
            "gsldrc1    %[ftmp1],   0x00(%[src])                        \n\t"
            "pshufh     %[ftmp4],   %[ftmp4],       %[ftmp0]            \n\t"
            "punpcklbh  %[ftmp2],   %[ftmp1],       %[ftmp0]            \n\t"
            "punpckhbh  %[ftmp3],   %[ftmp1],       %[ftmp0]            \n\t"
            "pmullh     %[ftmp5],   %[ftmp2],       %[ftmp4]            \n\t"
            "pmullh     %[ftmp6],   %[ftmp3],       %[ftmp4]            \n\t"

            "gsldlc1    %[ftmp1],   0x06(%[src])                        \n\t"
            "mtc1       %[filter1], %[ftmp4]                            \n\t"
            "gsldrc1    %[ftmp1],  -0x01(%[src])                        \n\t"
            "pshufh     %[ftmp4],   %[ftmp4],       %[ftmp0]            \n\t"
            "punpcklbh  %[ftmp2],   %[ftmp1],       %[ftmp0]            \n\t"
            "punpckhbh  %[ftmp3],   %[ftmp1],       %[ftmp0]            \n\t"
            "pmullh     %[ftmp2],   %[ftmp2],       %[ftmp4]            \n\t"
            "pmullh     %[ftmp3],   %[ftmp3],       %[ftmp4]            \n\t"
            "psubush    %[ftmp5],   %[ftmp5],       %[ftmp2]            \n\t"
            "psubush    %[ftmp6],   %[ftmp6],       %[ftmp3]            \n\t"

            "gsldlc1    %[ftmp1],   0x08(%[src])                        \n\t"
            "mtc1       %[filter3], %[ftmp4]                            \n\t"
            "gsldrc1    %[ftmp1],   0x01(%[src])                        \n\t"
            "pshufh     %[ftmp4],   %[ftmp4],       %[ftmp0]            \n\t"
            "punpcklbh  %[ftmp2],   %[ftmp1],       %[ftmp0]            \n\t"
            "punpckhbh  %[ftmp3],   %[ftmp1],       %[ftmp0]            \n\t"
            "pmullh     %[ftmp2],   %[ftmp2],       %[ftmp4]            \n\t"
            "pmullh     %[ftmp3],   %[ftmp3],       %[ftmp4]            \n\t"
            "paddush    %[ftmp5],   %[ftmp5],       %[ftmp2]            \n\t"
            "paddush    %[ftmp6],   %[ftmp6],       %[ftmp3]            \n\t"

            "gsldlc1    %[ftmp1],   0x09(%[src])                        \n\t"
            "mtc1       %[filter4], %[ftmp4]                            \n\t"
            "gsldrc1    %[ftmp1],   0x02(%[src])                        \n\t"
            "pshufh     %[ftmp4],   %[ftmp4],       %[ftmp0]            \n\t"
            "punpcklbh  %[ftmp2],   %[ftmp1],       %[ftmp0]            \n\t"
            "punpckhbh  %[ftmp3],   %[ftmp1],       %[ftmp0]            \n\t"
            "pmullh     %[ftmp2],   %[ftmp2],       %[ftmp4]            \n\t"
            "pmullh     %[ftmp3],   %[ftmp3],       %[ftmp4]            \n\t"
            "psubush    %[ftmp5],   %[ftmp5],       %[ftmp2]            \n\t"
            "psubush    %[ftmp6],   %[ftmp6],       %[ftmp3]            \n\t"

            "li         %[tmp0],    0x07                                \n\t"
            "paddush    %[ftmp5],   %[ftmp5],       %[ff_pw_64]         \n\t"
            "paddush    %[ftmp6],   %[ftmp6],       %[ff_pw_64]         \n\t"
            "mtc1       %[tmp0],    %[ftmp4]                            \n\t"
            "psrlh      %[ftmp5],   %[ftmp5],       %[ftmp4]            \n\t"
            "psrlh      %[ftmp6],   %[ftmp6],       %[ftmp4]            \n\t"

            "packushb   %[ftmp1],   %[ftmp5],       %[ftmp6]            \n\t"
            "gssdlc1    %[ftmp1],   0x07(%[tmp])                        \n\t"
            "gssdrc1    %[ftmp1],   0x00(%[tmp])                        \n\t"
            : [ftmp0]"=&f"(ftmp[0]),            [ftmp1]"=&f"(ftmp[1]),
              [ftmp2]"=&f"(ftmp[2]),            [ftmp3]"=&f"(ftmp[3]),
              [ftmp4]"=&f"(ftmp[4]),            [ftmp5]"=&f"(ftmp[5]),
              [ftmp6]"=&f"(ftmp[6]),
              [tmp0]"=&r"(tmp0)
            : [tmp]"r"(tmp),                    [src]"r"(src),
              [ff_pw_64]"f"(ff_pw_64),
              [filter1]"r"(filter[1]),          [filter2]"r"(filter[2]),
              [filter3]"r"(filter[3]),          [filter4]"r"(filter[4])
            : "memory"
        );

        tmp += 8;
        src += srcstride;
    }
#else
    const uint8_t *filter = subpel_filters[mx - 1];
    const uint8_t *cm     = ff_crop_tab + MAX_NEG_CROP;
    int x, y;
    uint8_t tmp_array[152];
    uint8_t *tmp = tmp_array;

    src -= srcstride;

    for (y = 0; y < h + 3; y++) {
        for (x = 0; x < 8; x++)
            tmp[x] = FILTER_4TAP(src, filter, 1);
        tmp += 8;
        src += srcstride;
    }
#endif

    tmp    = tmp_array + 8;
    filter = subpel_filters[my - 1];

#if NOTOK //FIXME fate-vp8-test-vector-006
    for (y = 0; y < h; y++) {
        /*
        dst[0] = cm[(filter[2] * tmp[0] - filter[1] * tmp[-8] + filter[3] * tmp[ 8] - filter[4] * tmp[16] + 64) >> 7];
        dst[1] = cm[(filter[2] * tmp[1] - filter[1] * tmp[-7] + filter[3] * tmp[ 9] - filter[4] * tmp[17] + 64) >> 7];
        dst[2] = cm[(filter[2] * tmp[2] - filter[1] * tmp[-6] + filter[3] * tmp[10] - filter[4] * tmp[18] + 64) >> 7];
        dst[3] = cm[(filter[2] * tmp[3] - filter[1] * tmp[-5] + filter[3] * tmp[11] - filter[4] * tmp[19] + 64) >> 7];
        dst[4] = cm[(filter[2] * tmp[4] - filter[1] * tmp[-4] + filter[3] * tmp[12] - filter[4] * tmp[20] + 64) >> 7];
        dst[5] = cm[(filter[2] * tmp[5] - filter[1] * tmp[-3] + filter[3] * tmp[13] - filter[4] * tmp[21] + 64) >> 7];
        dst[6] = cm[(filter[2] * tmp[6] - filter[1] * tmp[-2] + filter[3] * tmp[14] - filter[4] * tmp[22] + 64) >> 7];
        dst[7] = cm[(filter[2] * tmp[7] - filter[1] * tmp[-1] + filter[3] * tmp[15] - filter[4] * tmp[23] + 64) >> 7];
        */
        __asm__ volatile (
            "xor        %[ftmp0],   %[ftmp0],       %[ftmp0]            \n\t"
            "gsldlc1    %[ftmp1],   0x07(%[tmp])                        \n\t"
            "mtc1       %[filter2], %[ftmp4]                            \n\t"
            "gsldrc1    %[ftmp1],   0x00(%[tmp])                        \n\t"
            "pshufh     %[ftmp4],   %[ftmp4],       %[ftmp0]            \n\t"
            "punpcklbh  %[ftmp2],   %[ftmp1],       %[ftmp0]            \n\t"
            "punpckhbh  %[ftmp3],   %[ftmp1],       %[ftmp0]            \n\t"
            "pmullh     %[ftmp5],   %[ftmp2],       %[ftmp4]            \n\t"
            "pmullh     %[ftmp6],   %[ftmp3],       %[ftmp4]            \n\t"

            "gsldlc1    %[ftmp1],  -0x01(%[tmp])                        \n\t"
            "mtc1       %[filter1], %[ftmp4]                            \n\t"
            "gsldrc1    %[ftmp1],  -0x08(%[tmp])                        \n\t"
            "pshufh     %[ftmp4],   %[ftmp4],       %[ftmp0]            \n\t"
            "punpcklbh  %[ftmp2],   %[ftmp1],       %[ftmp0]            \n\t"
            "punpckhbh  %[ftmp3],   %[ftmp1],       %[ftmp0]            \n\t"
            "pmullh     %[ftmp2],   %[ftmp2],       %[ftmp4]            \n\t"
            "pmullh     %[ftmp3],   %[ftmp3],       %[ftmp4]            \n\t"
            "psubush    %[ftmp5],   %[ftmp5],       %[ftmp2]            \n\t"
            "psubush    %[ftmp6],   %[ftmp6],       %[ftmp3]            \n\t"

            "gsldlc1    %[ftmp1],   0x0f(%[tmp])                        \n\t"
            "mtc1       %[filter3], %[ftmp4]                            \n\t"
            "gsldrc1    %[ftmp1],   0x08(%[tmp])                        \n\t"
            "pshufh     %[ftmp4],   %[ftmp4],       %[ftmp0]            \n\t"
            "punpcklbh  %[ftmp2],   %[ftmp1],       %[ftmp0]            \n\t"
            "punpckhbh  %[ftmp3],   %[ftmp1],       %[ftmp0]            \n\t"
            "pmullh     %[ftmp2],   %[ftmp2],       %[ftmp4]            \n\t"
            "pmullh     %[ftmp3],   %[ftmp3],       %[ftmp4]            \n\t"
            "paddush    %[ftmp5],   %[ftmp5],       %[ftmp2]            \n\t"
            "paddush    %[ftmp6],   %[ftmp6],       %[ftmp3]            \n\t"

            "gsldlc1    %[ftmp1],   0x17(%[tmp])                        \n\t"
            "mtc1       %[filter4], %[ftmp4]                            \n\t"
            "gsldrc1    %[ftmp1],   0x10(%[tmp])                        \n\t"
            "pshufh     %[ftmp4],   %[ftmp4],       %[ftmp0]            \n\t"
            "punpcklbh  %[ftmp2],   %[ftmp1],       %[ftmp0]            \n\t"
            "punpckhbh  %[ftmp3],   %[ftmp1],       %[ftmp0]            \n\t"
            "pmullh     %[ftmp2],   %[ftmp2],       %[ftmp4]            \n\t"
            "pmullh     %[ftmp3],   %[ftmp3],       %[ftmp4]            \n\t"
            "psubush    %[ftmp5],   %[ftmp5],       %[ftmp2]            \n\t"
            "psubush    %[ftmp6],   %[ftmp6],       %[ftmp3]            \n\t"

            "li         %[tmp0],    0x07                                \n\t"
            "paddush    %[ftmp5],   %[ftmp5],       %[ff_pw_64]         \n\t"
            "paddush    %[ftmp6],   %[ftmp6],       %[ff_pw_64]         \n\t"
            "mtc1       %[tmp0],    %[ftmp4]                            \n\t"
            "psrlh      %[ftmp5],   %[ftmp5],       %[ftmp4]            \n\t"
            "psrlh      %[ftmp6],   %[ftmp6],       %[ftmp4]            \n\t"

            "packushb   %[ftmp1],   %[ftmp5],       %[ftmp6]            \n\t"
            "gssdlc1    %[ftmp1],   0x07(%[dst])                        \n\t"
            "gssdrc1    %[ftmp1],   0x00(%[dst])                        \n\t"
            : [ftmp0]"=&f"(ftmp[0]),            [ftmp1]"=&f"(ftmp[1]),
              [ftmp2]"=&f"(ftmp[2]),            [ftmp3]"=&f"(ftmp[3]),
              [ftmp4]"=&f"(ftmp[4]),            [ftmp5]"=&f"(ftmp[5]),
              [ftmp6]"=&f"(ftmp[6]),
              [tmp0]"=&r"(tmp0)
            : [dst]"r"(dst),                    [tmp]"r"(tmp),
              [ff_pw_64]"f"(ff_pw_64),
              [filter1]"r"(filter[1]),          [filter2]"r"(filter[2]),
              [filter3]"r"(filter[3]),          [filter4]"r"(filter[4])
            : "memory"
        );

        dst += dststride;
        tmp += 8;
    }
#else
    for (y = 0; y < h; y++) {
        for (x = 0; x < 8; x++)
            dst[x] = FILTER_4TAP(tmp, filter, 8);
        dst += dststride;
        tmp += 8;
    }
#endif
}

void ff_put_vp8_epel4_h4v4_mmi(uint8_t *dst, ptrdiff_t dststride, uint8_t *src,
        ptrdiff_t srcstride, int h, int mx, int my)
{
#if NOTOK //FIXME fate-vp8-test-vector-006
    const uint8_t *filter = subpel_filters[mx - 1];
    int y;
    uint8_t tmp_array[44];
    uint8_t *tmp = tmp_array;
    double ftmp[5];
    uint64_t tmp0;

    src -= srcstride;

    for (y = 0; y < h + 3; y++) {
        /*
        tmp[0] = cm[(filter[2] * src[0] - filter[1] * src[-1] + filter[3] * src[1] - filter[4] * src[2] + 64) >> 7];
        tmp[1] = cm[(filter[2] * src[1] - filter[1] * src[ 0] + filter[3] * src[2] - filter[4] * src[3] + 64) >> 7];
        tmp[2] = cm[(filter[2] * src[2] - filter[1] * src[ 1] + filter[3] * src[3] - filter[4] * src[4] + 64) >> 7];
        tmp[3] = cm[(filter[2] * src[3] - filter[1] * src[ 2] + filter[3] * src[4] - filter[4] * src[5] + 64) >> 7];
        */
        __asm__ volatile (
            "xor        %[ftmp0],   %[ftmp0],       %[ftmp0]            \n\t"
            "gslwlc1    %[ftmp1],   0x03(%[src])                        \n\t"
            "mtc1       %[filter2], %[ftmp3]                            \n\t"
            "gslwrc1    %[ftmp1],   0x00(%[src])                        \n\t"
            "pshufh     %[ftmp3],   %[ftmp3],       %[ftmp0]            \n\t"
            "punpcklbh  %[ftmp2],   %[ftmp1],       %[ftmp0]            \n\t"
            "pmullh     %[ftmp4],   %[ftmp2],       %[ftmp3]            \n\t"

            "gslwlc1    %[ftmp1],   0x02(%[src])                        \n\t"
            "mtc1       %[filter1], %[ftmp3]                            \n\t"
            "gslwrc1    %[ftmp1],  -0x01(%[src])                        \n\t"
            "pshufh     %[ftmp3],   %[ftmp3],       %[ftmp0]            \n\t"
            "punpcklbh  %[ftmp2],   %[ftmp1],       %[ftmp0]            \n\t"
            "pmullh     %[ftmp2],   %[ftmp2],       %[ftmp3]            \n\t"
            "psubush    %[ftmp4],   %[ftmp4],       %[ftmp2]            \n\t"

            "gslwlc1    %[ftmp1],   0x04(%[src])                        \n\t"
            "mtc1       %[filter3], %[ftmp3]                            \n\t"
            "gslwrc1    %[ftmp1],   0x01(%[src])                        \n\t"
            "pshufh     %[ftmp3],   %[ftmp3],       %[ftmp0]            \n\t"
            "punpcklbh  %[ftmp2],   %[ftmp1],       %[ftmp0]            \n\t"
            "pmullh     %[ftmp2],   %[ftmp2],       %[ftmp3]            \n\t"
            "paddush    %[ftmp4],   %[ftmp4],       %[ftmp2]            \n\t"

            "gslwlc1    %[ftmp1],   0x05(%[src])                        \n\t"
            "mtc1       %[filter4], %[ftmp3]                            \n\t"
            "gslwrc1    %[ftmp1],   0x02(%[src])                        \n\t"
            "pshufh     %[ftmp3],   %[ftmp3],       %[ftmp0]            \n\t"
            "punpcklbh  %[ftmp2],   %[ftmp1],       %[ftmp0]            \n\t"
            "pmullh     %[ftmp2],   %[ftmp2],       %[ftmp3]            \n\t"
            "psubush    %[ftmp4],   %[ftmp4],       %[ftmp2]            \n\t"

            "li         %[tmp0],    0x07                                \n\t"
            "paddush    %[ftmp4],   %[ftmp4],       %[ff_pw_64]         \n\t"
            "mtc1       %[tmp0],    %[ftmp3]                            \n\t"
            "psrlh      %[ftmp4],   %[ftmp4],       %[ftmp3]            \n\t"

            "packushb   %[ftmp1],   %[ftmp4],       %[ftmp0]            \n\t"
            "gsswlc1    %[ftmp1],   0x03(%[tmp])                        \n\t"
            "gsswrc1    %[ftmp1],   0x00(%[tmp])                        \n\t"
            : [ftmp0]"=&f"(ftmp[0]),            [ftmp1]"=&f"(ftmp[1]),
              [ftmp2]"=&f"(ftmp[2]),            [ftmp3]"=&f"(ftmp[3]),
              [ftmp4]"=&f"(ftmp[4]),
              [tmp0]"=&r"(tmp0)
            : [tmp]"r"(tmp),                    [src]"r"(src),
              [ff_pw_64]"f"(ff_pw_64),
              [filter1]"r"(filter[1]),          [filter2]"r"(filter[2]),
              [filter3]"r"(filter[3]),          [filter4]"r"(filter[4])
            : "memory"
        );

        tmp += 4;
        src += srcstride;
    }
#else
    const uint8_t *filter = subpel_filters[mx - 1];
    const uint8_t *cm     = ff_crop_tab + MAX_NEG_CROP;
    int x, y;
    uint8_t tmp_array[44];
    uint8_t *tmp = tmp_array;

    src -= srcstride;

    for (y = 0; y < h + 3; y++) {
        for (x = 0; x < 4; x++)
            tmp[x] = FILTER_4TAP(src, filter, 1);
        tmp += 4;
        src += srcstride;
    }
#endif

    tmp    = tmp_array + 4;
    filter = subpel_filters[my - 1];

#if NOTOK //FIXME fate-vp8-test-vector-006
    for (y = 0; y < h; y++) {
        /*
        dst[0] = cm[(filter[2] * tmp[0] - filter[1] * tmp[-4] + filter[3] * tmp[4] - filter[4] * tmp[ 8] + 64) >> 7];
        dst[1] = cm[(filter[2] * tmp[1] - filter[1] * tmp[-3] + filter[3] * tmp[5] - filter[4] * tmp[ 9] + 64) >> 7];
        dst[2] = cm[(filter[2] * tmp[2] - filter[1] * tmp[-2] + filter[3] * tmp[6] - filter[4] * tmp[10] + 64) >> 7];
        dst[3] = cm[(filter[2] * tmp[3] - filter[1] * tmp[-1] + filter[3] * tmp[7] - filter[4] * tmp[11] + 64) >> 7];
        */
        __asm__ volatile (
            "xor        %[ftmp0],   %[ftmp0],       %[ftmp0]            \n\t"
            "gslwlc1    %[ftmp1],   0x03(%[tmp])                        \n\t"
            "mtc1       %[filter2], %[ftmp3]                            \n\t"
            "gslwrc1    %[ftmp1],   0x00(%[tmp])                        \n\t"
            "pshufh     %[ftmp3],   %[ftmp3],       %[ftmp0]            \n\t"
            "punpcklbh  %[ftmp2],   %[ftmp1],       %[ftmp0]            \n\t"
            "pmullh     %[ftmp4],   %[ftmp2],       %[ftmp3]            \n\t"

            "gslwlc1    %[ftmp1],  -0x01(%[tmp])                        \n\t"
            "mtc1       %[filter1], %[ftmp3]                            \n\t"
            "gslwrc1    %[ftmp1],  -0x04(%[tmp])                        \n\t"
            "pshufh     %[ftmp3],   %[ftmp3],       %[ftmp0]            \n\t"
            "punpcklbh  %[ftmp2],   %[ftmp1],       %[ftmp0]            \n\t"
            "pmullh     %[ftmp2],   %[ftmp2],       %[ftmp3]            \n\t"
            "psubush    %[ftmp4],   %[ftmp4],       %[ftmp2]            \n\t"

            "gslwlc1    %[ftmp1],   0x07(%[tmp])                        \n\t"
            "mtc1       %[filter3], %[ftmp3]                            \n\t"
            "gslwrc1    %[ftmp1],   0x04(%[tmp])                        \n\t"
            "pshufh     %[ftmp3],   %[ftmp3],       %[ftmp0]            \n\t"
            "punpcklbh  %[ftmp2],   %[ftmp1],       %[ftmp0]            \n\t"
            "pmullh     %[ftmp2],   %[ftmp2],       %[ftmp3]            \n\t"
            "paddush    %[ftmp4],   %[ftmp4],       %[ftmp2]            \n\t"

            "gslwlc1    %[ftmp1],   0x0b(%[tmp])                        \n\t"
            "mtc1       %[filter4], %[ftmp3]                            \n\t"
            "gslwrc1    %[ftmp1],   0x08(%[tmp])                        \n\t"
            "pshufh     %[ftmp3],   %[ftmp3],       %[ftmp0]            \n\t"
            "punpcklbh  %[ftmp2],   %[ftmp1],       %[ftmp0]            \n\t"
            "pmullh     %[ftmp2],   %[ftmp2],       %[ftmp3]            \n\t"
            "psubush    %[ftmp4],   %[ftmp4],       %[ftmp2]            \n\t"

            "li         %[tmp0],    0x07                                \n\t"
            "paddush    %[ftmp4],   %[ftmp4],       %[ff_pw_64]         \n\t"
            "mtc1       %[tmp0],    %[ftmp3]                            \n\t"
            "psrlh      %[ftmp4],   %[ftmp4],       %[ftmp3]            \n\t"

            "packushb   %[ftmp1],   %[ftmp4],       %[ftmp0]            \n\t"
            "gsswlc1    %[ftmp1],   0x03(%[dst])                        \n\t"
            "gsswrc1    %[ftmp1],   0x00(%[dst])                        \n\t"
            : [ftmp0]"=&f"(ftmp[0]),            [ftmp1]"=&f"(ftmp[1]),
              [ftmp2]"=&f"(ftmp[2]),            [ftmp3]"=&f"(ftmp[3]),
              [ftmp4]"=&f"(ftmp[4]),
              [tmp0]"=&r"(tmp0)
            : [dst]"r"(dst),                    [tmp]"r"(tmp),
              [ff_pw_64]"f"(ff_pw_64),
              [filter1]"r"(filter[1]),          [filter2]"r"(filter[2]),
              [filter3]"r"(filter[3]),          [filter4]"r"(filter[4])
            : "memory"
        );

        dst += dststride;
        tmp += 4;
    }
#else
    for (y = 0; y < h; y++) {
        for (x = 0; x < 4; x++)
            dst[x] = FILTER_4TAP(tmp, filter, 4);
        dst += dststride;
        tmp += 4;
    }
#endif
}

void ff_put_vp8_epel16_h4v6_mmi(uint8_t *dst, ptrdiff_t dststride, uint8_t *src,
        ptrdiff_t srcstride, int h, int mx, int my)
{
#if NOTOK //FIXME fate-vp8-test-vector-002 006 009
    const uint8_t *filter = subpel_filters[mx - 1];
    int y;
    uint8_t tmp_array[592];
    uint8_t *tmp = tmp_array;
    double ftmp[10];
    uint64_t tmp0;

    src -= 2 * srcstride;

    for (y = 0; y < h + 5; y++) {
        /*
        tmp[0] = cm[(filter[2] * src[0] - filter[1] * src[-1] + filter[3] * src[1] - filter[4] * src[2] + 64) >> 7];
        tmp[1] = cm[(filter[2] * src[1] - filter[1] * src[ 0] + filter[3] * src[2] - filter[4] * src[3] + 64) >> 7];
        tmp[2] = cm[(filter[2] * src[2] - filter[1] * src[ 1] + filter[3] * src[3] - filter[4] * src[4] + 64) >> 7];
        tmp[3] = cm[(filter[2] * src[3] - filter[1] * src[ 2] + filter[3] * src[4] - filter[4] * src[5] + 64) >> 7];
        tmp[4] = cm[(filter[2] * src[4] - filter[1] * src[ 3] + filter[3] * src[5] - filter[4] * src[6] + 64) >> 7];
        tmp[5] = cm[(filter[2] * src[5] - filter[1] * src[ 4] + filter[3] * src[6] - filter[4] * src[7] + 64) >> 7];
        tmp[6] = cm[(filter[2] * src[6] - filter[1] * src[ 5] + filter[3] * src[7] - filter[4] * src[8] + 64) >> 7];
        tmp[7] = cm[(filter[2] * src[7] - filter[1] * src[ 6] + filter[3] * src[8] - filter[4] * src[9] + 64) >> 7];

        tmp[ 8] = cm[(filter[2] * src[ 8] - filter[1] * src[ 7] + filter[3] * src[ 9] - filter[4] * src[10] + 64) >> 7];
        tmp[ 9] = cm[(filter[2] * src[ 9] - filter[1] * src[ 8] + filter[3] * src[10] - filter[4] * src[11] + 64) >> 7];
        tmp[10] = cm[(filter[2] * src[10] - filter[1] * src[ 9] + filter[3] * src[11] - filter[4] * src[12] + 64) >> 7];
        tmp[11] = cm[(filter[2] * src[11] - filter[1] * src[10] + filter[3] * src[12] - filter[4] * src[13] + 64) >> 7];
        tmp[12] = cm[(filter[2] * src[12] - filter[1] * src[11] + filter[3] * src[13] - filter[4] * src[14] + 64) >> 7];
        tmp[13] = cm[(filter[2] * src[13] - filter[1] * src[12] + filter[3] * src[14] - filter[4] * src[15] + 64) >> 7];
        tmp[14] = cm[(filter[2] * src[14] - filter[1] * src[13] + filter[3] * src[15] - filter[4] * src[16] + 64) >> 7];
        tmp[15] = cm[(filter[2] * src[15] - filter[1] * src[14] + filter[3] * src[16] - filter[4] * src[17] + 64) >> 7];
        */
        __asm__ volatile (
            "xor        %[ftmp0],   %[ftmp0],       %[ftmp0]            \n\t"
            "gsldlc1    %[ftmp1],   0x07(%[src])                        \n\t"
            "gsldrc1    %[ftmp1],   0x00(%[src])                        \n\t"
            "gsldlc1    %[ftmp7],   0x0f(%[src])                        \n\t"
            "mtc1       %[filter2], %[ftmp4]                            \n\t"
            "gsldrc1    %[ftmp7],   0x08(%[src])                        \n\t"
            "pshufh     %[ftmp4],   %[ftmp4],       %[ftmp0]            \n\t"
            "punpcklbh  %[ftmp2],   %[ftmp1],       %[ftmp0]            \n\t"
            "punpckhbh  %[ftmp3],   %[ftmp1],       %[ftmp0]            \n\t"
            "pmullh     %[ftmp5],   %[ftmp2],       %[ftmp4]            \n\t"
            "pmullh     %[ftmp6],   %[ftmp3],       %[ftmp4]            \n\t"
            "punpcklbh  %[ftmp2],   %[ftmp7],       %[ftmp0]            \n\t"
            "punpckhbh  %[ftmp3],   %[ftmp7],       %[ftmp0]            \n\t"
            "pmullh     %[ftmp8],   %[ftmp2],       %[ftmp4]            \n\t"
            "pmullh     %[ftmp9],   %[ftmp3],       %[ftmp4]            \n\t"

            "gsldlc1    %[ftmp1],   0x06(%[src])                        \n\t"
            "gsldrc1    %[ftmp1],  -0x01(%[src])                        \n\t"
            "gsldlc1    %[ftmp7],   0x0e(%[src])                        \n\t"
            "mtc1       %[filter1], %[ftmp4]                            \n\t"
            "gsldrc1    %[ftmp7],   0x07(%[src])                        \n\t"
            "pshufh     %[ftmp4],   %[ftmp4],       %[ftmp0]            \n\t"
            "punpcklbh  %[ftmp2],   %[ftmp1],       %[ftmp0]            \n\t"
            "punpckhbh  %[ftmp3],   %[ftmp1],       %[ftmp0]            \n\t"
            "pmullh     %[ftmp2],   %[ftmp2],       %[ftmp4]            \n\t"
            "pmullh     %[ftmp3],   %[ftmp3],       %[ftmp4]            \n\t"
            "psubush    %[ftmp5],   %[ftmp5],       %[ftmp2]            \n\t"
            "psubush    %[ftmp6],   %[ftmp6],       %[ftmp3]            \n\t"
            "punpcklbh  %[ftmp2],   %[ftmp7],       %[ftmp0]            \n\t"
            "punpckhbh  %[ftmp3],   %[ftmp7],       %[ftmp0]            \n\t"
            "pmullh     %[ftmp2],   %[ftmp2],       %[ftmp4]            \n\t"
            "pmullh     %[ftmp3],   %[ftmp3],       %[ftmp4]            \n\t"
            "psubush    %[ftmp8],   %[ftmp8],       %[ftmp2]            \n\t"
            "psubush    %[ftmp9],   %[ftmp9],       %[ftmp3]            \n\t"

            "gsldlc1    %[ftmp1],   0x08(%[src])                        \n\t"
            "gsldrc1    %[ftmp1],   0x01(%[src])                        \n\t"
            "gsldlc1    %[ftmp7],   0x0d(%[src])                        \n\t"
            "mtc1       %[filter3], %[ftmp4]                            \n\t"
            "gsldrc1    %[ftmp7],   0x06(%[src])                        \n\t"
            "pshufh     %[ftmp4],   %[ftmp4],       %[ftmp0]            \n\t"
            "punpcklbh  %[ftmp2],   %[ftmp1],       %[ftmp0]            \n\t"
            "punpckhbh  %[ftmp3],   %[ftmp1],       %[ftmp0]            \n\t"
            "pmullh     %[ftmp2],   %[ftmp2],       %[ftmp4]            \n\t"
            "pmullh     %[ftmp3],   %[ftmp3],       %[ftmp4]            \n\t"
            "paddush    %[ftmp5],   %[ftmp5],       %[ftmp2]            \n\t"
            "paddush    %[ftmp6],   %[ftmp6],       %[ftmp3]            \n\t"
            "punpcklbh  %[ftmp2],   %[ftmp7],       %[ftmp0]            \n\t"
            "punpckhbh  %[ftmp3],   %[ftmp7],       %[ftmp0]            \n\t"
            "pmullh     %[ftmp2],   %[ftmp2],       %[ftmp4]            \n\t"
            "pmullh     %[ftmp3],   %[ftmp3],       %[ftmp4]            \n\t"
            "paddush    %[ftmp8],   %[ftmp8],       %[ftmp2]            \n\t"
            "paddush    %[ftmp9],   %[ftmp9],       %[ftmp3]            \n\t"

            "gsldlc1    %[ftmp1],   0x09(%[src])                        \n\t"
            "gsldrc1    %[ftmp1],   0x02(%[src])                        \n\t"
            "gsldlc1    %[ftmp7],   0x11(%[src])                        \n\t"
            "mtc1       %[filter4], %[ftmp4]                            \n\t"
            "gsldrc1    %[ftmp7],   0x0a(%[src])                        \n\t"
            "pshufh     %[ftmp4],   %[ftmp4],       %[ftmp0]            \n\t"
            "punpcklbh  %[ftmp2],   %[ftmp1],       %[ftmp0]            \n\t"
            "punpckhbh  %[ftmp3],   %[ftmp1],       %[ftmp0]            \n\t"
            "pmullh     %[ftmp2],   %[ftmp2],       %[ftmp4]            \n\t"
            "pmullh     %[ftmp3],   %[ftmp3],       %[ftmp4]            \n\t"
            "psubush    %[ftmp5],   %[ftmp5],       %[ftmp2]            \n\t"
            "psubush    %[ftmp6],   %[ftmp6],       %[ftmp3]            \n\t"
            "punpcklbh  %[ftmp2],   %[ftmp7],       %[ftmp0]            \n\t"
            "punpckhbh  %[ftmp3],   %[ftmp7],       %[ftmp0]            \n\t"
            "pmullh     %[ftmp2],   %[ftmp2],       %[ftmp4]            \n\t"
            "pmullh     %[ftmp3],   %[ftmp3],       %[ftmp4]            \n\t"
            "psubush    %[ftmp8],   %[ftmp8],       %[ftmp2]            \n\t"
            "psubush    %[ftmp9],   %[ftmp9],       %[ftmp3]            \n\t"

            "li         %[tmp0],    0x07                                \n\t"
            "paddush    %[ftmp5],   %[ftmp5],       %[ff_pw_64]         \n\t"
            "paddush    %[ftmp6],   %[ftmp6],       %[ff_pw_64]         \n\t"
            "mtc1       %[tmp0],    %[ftmp4]                            \n\t"
            "psrlh      %[ftmp5],   %[ftmp5],       %[ftmp4]            \n\t"
            "psrlh      %[ftmp6],   %[ftmp6],       %[ftmp4]            \n\t"

            "packushb   %[ftmp1],   %[ftmp5],       %[ftmp6]            \n\t"
            "gssdlc1    %[ftmp1],   0x07(%[tmp])                        \n\t"
            "gssdrc1    %[ftmp1],   0x00(%[tmp])                        \n\t"
            "gssdlc1    %[ftmp1],   0x0f(%[tmp])                        \n\t"
            "gssdrc1    %[ftmp1],   0x08(%[tmp])                        \n\t"
            : [ftmp0]"=&f"(ftmp[0]),            [ftmp1]"=&f"(ftmp[1]),
              [ftmp2]"=&f"(ftmp[2]),            [ftmp3]"=&f"(ftmp[3]),
              [ftmp4]"=&f"(ftmp[4]),            [ftmp5]"=&f"(ftmp[5]),
              [ftmp6]"=&f"(ftmp[6]),            [ftmp7]"=&f"(ftmp[7]),
              [ftmp8]"=&f"(ftmp[8]),            [ftmp9]"=&f"(ftmp[9]),
              [tmp0]"=&r"(tmp0)
            : [tmp]"r"(tmp),                    [src]"r"(src),
              [ff_pw_64]"f"(ff_pw_64),
              [filter1]"r"(filter[1]),          [filter2]"r"(filter[2]),
              [filter3]"r"(filter[3]),          [filter4]"r"(filter[4])
            : "memory"
        );

        tmp += 16;
        src += srcstride;
    }
#else
    const uint8_t *filter = subpel_filters[mx - 1];
    const uint8_t *cm     = ff_crop_tab + MAX_NEG_CROP;
    int x, y;
    uint8_t tmp_array[592];
    uint8_t *tmp = tmp_array;

    src -= 2 * srcstride;

    for (y = 0; y < h + 5; y++) {
        for (x = 0; x < 16; x++)
            tmp[x] = FILTER_4TAP(src, filter, 1);
        tmp += 16;
        src += srcstride;
    }
#endif

    tmp    = tmp_array + 32;
    filter = subpel_filters[my - 1];

#if NOTOK //FIXME fate-vp8-test-vector-002 006 008
    for (y = 0; y < h; y++) {
        /*
        dst[0] = cm[(filter[2]*tmp[0] - filter[1]*tmp[-16] + filter[0]*tmp[-32] + filter[3]*tmp[17] - filter[4]*tmp[32] + filter[5]*tmp[48] + 64) >> 7];
        dst[1] = cm[(filter[2]*tmp[1] - filter[1]*tmp[-15] + filter[0]*tmp[-31] + filter[3]*tmp[18] - filter[4]*tmp[33] + filter[5]*tmp[49] + 64) >> 7];
        dst[2] = cm[(filter[2]*tmp[2] - filter[1]*tmp[-14] + filter[0]*tmp[-30] + filter[3]*tmp[19] - filter[4]*tmp[34] + filter[5]*tmp[50] + 64) >> 7];
        dst[3] = cm[(filter[2]*tmp[3] - filter[1]*tmp[-13] + filter[0]*tmp[-29] + filter[3]*tmp[20] - filter[4]*tmp[35] + filter[5]*tmp[51] + 64) >> 7];
        dst[4] = cm[(filter[2]*tmp[4] - filter[1]*tmp[-12] + filter[0]*tmp[-28] + filter[3]*tmp[21] - filter[4]*tmp[36] + filter[5]*tmp[52] + 64) >> 7];
        dst[5] = cm[(filter[2]*tmp[5] - filter[1]*tmp[-11] + filter[0]*tmp[-27] + filter[3]*tmp[22] - filter[4]*tmp[37] + filter[5]*tmp[53] + 64) >> 7];
        dst[6] = cm[(filter[2]*tmp[6] - filter[1]*tmp[-10] + filter[0]*tmp[-26] + filter[3]*tmp[23] - filter[4]*tmp[38] + filter[5]*tmp[54] + 64) >> 7];
        dst[7] = cm[(filter[2]*tmp[7] - filter[1]*tmp[ -9] + filter[0]*tmp[-25] + filter[3]*tmp[24] - filter[4]*tmp[39] + filter[5]*tmp[55] + 64) >> 7];

        dst[ 8] = cm[(filter[2]*tmp[ 8] - filter[1]*tmp[-8] + filter[0]*tmp[-24] + filter[3]*tmp[25] - filter[4]*tmp[40] + filter[5]*tmp[56] + 64) >> 7];
        dst[ 9] = cm[(filter[2]*tmp[ 9] - filter[1]*tmp[-7] + filter[0]*tmp[-23] + filter[3]*tmp[26] - filter[4]*tmp[41] + filter[5]*tmp[57] + 64) >> 7];
        dst[10] = cm[(filter[2]*tmp[10] - filter[1]*tmp[-6] + filter[0]*tmp[-22] + filter[3]*tmp[27] - filter[4]*tmp[42] + filter[5]*tmp[58] + 64) >> 7];
        dst[11] = cm[(filter[2]*tmp[11] - filter[1]*tmp[-5] + filter[0]*tmp[-21] + filter[3]*tmp[28] - filter[4]*tmp[43] + filter[5]*tmp[59] + 64) >> 7];
        dst[12] = cm[(filter[2]*tmp[12] - filter[1]*tmp[-4] + filter[0]*tmp[-20] + filter[3]*tmp[29] - filter[4]*tmp[44] + filter[5]*tmp[60] + 64) >> 7];
        dst[13] = cm[(filter[2]*tmp[13] - filter[1]*tmp[-3] + filter[0]*tmp[-19] + filter[3]*tmp[30] - filter[4]*tmp[45] + filter[5]*tmp[61] + 64) >> 7];
        dst[14] = cm[(filter[2]*tmp[14] - filter[1]*tmp[-2] + filter[0]*tmp[-18] + filter[3]*tmp[31] - filter[4]*tmp[46] + filter[5]*tmp[62] + 64) >> 7];
        dst[15] = cm[(filter[2]*tmp[15] - filter[1]*tmp[-1] + filter[0]*tmp[-17] + filter[3]*tmp[32] - filter[4]*tmp[47] + filter[5]*tmp[63] + 64) >> 7];
        */
        __asm__ volatile (
            "xor        %[ftmp0],   %[ftmp0],       %[ftmp0]            \n\t"
            "gsldlc1    %[ftmp1],   0x07(%[tmp])                        \n\t"
            "gsldrc1    %[ftmp1],   0x00(%[tmp])                        \n\t"
            "gsldlc1    %[ftmp7],   0x0f(%[tmp])                        \n\t"
            "mtc1       %[filter2], %[ftmp4]                            \n\t"
            "gsldrc1    %[ftmp7],   0x08(%[tmp])                        \n\t"
            "pshufh     %[ftmp4],   %[ftmp4],       %[ftmp0]            \n\t"
            "punpcklbh  %[ftmp2],   %[ftmp1],       %[ftmp0]            \n\t"
            "punpckhbh  %[ftmp3],   %[ftmp1],       %[ftmp0]            \n\t"
            "pmullh     %[ftmp5],   %[ftmp2],       %[ftmp4]            \n\t"
            "pmullh     %[ftmp6],   %[ftmp3],       %[ftmp4]            \n\t"
            "punpcklbh  %[ftmp2],   %[ftmp7],       %[ftmp0]            \n\t"
            "punpckhbh  %[ftmp3],   %[ftmp7],       %[ftmp0]            \n\t"
            "pmullh     %[ftmp8],   %[ftmp2],       %[ftmp4]            \n\t"
            "pmullh     %[ftmp9],   %[ftmp3],       %[ftmp4]            \n\t"

            "gsldlc1    %[ftmp1],  -0x09(%[tmp])                        \n\t"
            "gsldrc1    %[ftmp1],  -0x10(%[tmp])                        \n\t"
            "gsldlc1    %[ftmp7],  -0x08(%[tmp])                        \n\t"
            "mtc1       %[filter1], %[ftmp4]                            \n\t"
            "gsldrc1    %[ftmp7],  -0x01(%[tmp])                        \n\t"
            "pshufh     %[ftmp4],   %[ftmp4],       %[ftmp0]            \n\t"
            "punpcklbh  %[ftmp2],   %[ftmp1],       %[ftmp0]            \n\t"
            "punpckhbh  %[ftmp3],   %[ftmp1],       %[ftmp0]            \n\t"
            "pmullh     %[ftmp2],   %[ftmp2],       %[ftmp4]            \n\t"
            "pmullh     %[ftmp3],   %[ftmp3],       %[ftmp4]            \n\t"
            "psubush    %[ftmp5],   %[ftmp5],       %[ftmp2]            \n\t"
            "psubush    %[ftmp6],   %[ftmp6],       %[ftmp3]            \n\t"
            "punpcklbh  %[ftmp2],   %[ftmp7],       %[ftmp0]            \n\t"
            "punpckhbh  %[ftmp3],   %[ftmp7],       %[ftmp0]            \n\t"
            "pmullh     %[ftmp2],   %[ftmp2],       %[ftmp4]            \n\t"
            "pmullh     %[ftmp3],   %[ftmp3],       %[ftmp4]            \n\t"
            "psubush    %[ftmp8],   %[ftmp8],       %[ftmp2]            \n\t"
            "psubush    %[ftmp9],   %[ftmp9],       %[ftmp3]            \n\t"

            "gsldlc1    %[ftmp1],  -0x19(%[tmp])                        \n\t"
            "gsldrc1    %[ftmp1],  -0x20(%[tmp])                        \n\t"
            "gsldlc1    %[ftmp7],  -0x11(%[tmp])                        \n\t"
            "mtc1       %[filter0], %[ftmp4]                            \n\t"
            "gsldrc1    %[ftmp7],  -0x18(%[tmp])                        \n\t"
            "pshufh     %[ftmp4],   %[ftmp4],       %[ftmp0]            \n\t"
            "punpcklbh  %[ftmp2],   %[ftmp1],       %[ftmp0]            \n\t"
            "punpckhbh  %[ftmp3],   %[ftmp1],       %[ftmp0]            \n\t"
            "pmullh     %[ftmp2],   %[ftmp2],       %[ftmp4]            \n\t"
            "pmullh     %[ftmp3],   %[ftmp3],       %[ftmp4]            \n\t"
            "paddush    %[ftmp5],   %[ftmp5],       %[ftmp2]            \n\t"
            "paddush    %[ftmp6],   %[ftmp6],       %[ftmp3]            \n\t"
            "punpcklbh  %[ftmp2],   %[ftmp7],       %[ftmp0]            \n\t"
            "punpckhbh  %[ftmp3],   %[ftmp7],       %[ftmp0]            \n\t"
            "pmullh     %[ftmp2],   %[ftmp2],       %[ftmp4]            \n\t"
            "pmullh     %[ftmp3],   %[ftmp3],       %[ftmp4]            \n\t"
            "paddush    %[ftmp8],   %[ftmp8],       %[ftmp2]            \n\t"
            "paddush    %[ftmp9],   %[ftmp9],       %[ftmp3]            \n\t"

            "gsldlc1    %[ftmp1],   0x18(%[tmp])                        \n\t"
            "gsldrc1    %[ftmp1],   0x11(%[tmp])                        \n\t"
            "gsldlc1    %[ftmp7],   0x20(%[tmp])                        \n\t"
            "mtc1       %[filter3], %[ftmp4]                            \n\t"
            "gsldrc1    %[ftmp7],   0x19(%[tmp])                        \n\t"
            "pshufh     %[ftmp4],   %[ftmp4],       %[ftmp0]            \n\t"
            "punpcklbh  %[ftmp2],   %[ftmp1],       %[ftmp0]            \n\t"
            "punpckhbh  %[ftmp3],   %[ftmp1],       %[ftmp0]            \n\t"
            "pmullh     %[ftmp2],   %[ftmp2],       %[ftmp4]            \n\t"
            "pmullh     %[ftmp3],   %[ftmp3],       %[ftmp4]            \n\t"
            "paddush    %[ftmp5],   %[ftmp5],       %[ftmp2]            \n\t"
            "paddush    %[ftmp6],   %[ftmp6],       %[ftmp3]            \n\t"
            "punpcklbh  %[ftmp2],   %[ftmp7],       %[ftmp0]            \n\t"
            "punpckhbh  %[ftmp3],   %[ftmp7],       %[ftmp0]            \n\t"
            "pmullh     %[ftmp2],   %[ftmp2],       %[ftmp4]            \n\t"
            "pmullh     %[ftmp3],   %[ftmp3],       %[ftmp4]            \n\t"
            "paddush    %[ftmp8],   %[ftmp8],       %[ftmp2]            \n\t"
            "paddush    %[ftmp9],   %[ftmp9],       %[ftmp3]            \n\t"

            "gsldlc1    %[ftmp1],   0x27(%[tmp])                        \n\t"
            "gsldrc1    %[ftmp1],   0x20(%[tmp])                        \n\t"
            "gsldlc1    %[ftmp7],   0x2f(%[tmp])                        \n\t"
            "mtc1       %[filter4], %[ftmp4]                            \n\t"
            "gsldrc1    %[ftmp7],   0x28(%[tmp])                        \n\t"
            "pshufh     %[ftmp4],   %[ftmp4],       %[ftmp0]            \n\t"
            "punpcklbh  %[ftmp2],   %[ftmp1],       %[ftmp0]            \n\t"
            "punpckhbh  %[ftmp3],   %[ftmp1],       %[ftmp0]            \n\t"
            "pmullh     %[ftmp2],   %[ftmp2],       %[ftmp4]            \n\t"
            "pmullh     %[ftmp3],   %[ftmp3],       %[ftmp4]            \n\t"
            "psubush    %[ftmp5],   %[ftmp5],       %[ftmp2]            \n\t"
            "psubush    %[ftmp6],   %[ftmp6],       %[ftmp3]            \n\t"
            "punpcklbh  %[ftmp2],   %[ftmp7],       %[ftmp0]            \n\t"
            "punpckhbh  %[ftmp3],   %[ftmp7],       %[ftmp0]            \n\t"
            "pmullh     %[ftmp2],   %[ftmp2],       %[ftmp4]            \n\t"
            "pmullh     %[ftmp3],   %[ftmp3],       %[ftmp4]            \n\t"
            "psubush    %[ftmp8],   %[ftmp8],       %[ftmp2]            \n\t"
            "psubush    %[ftmp9],   %[ftmp9],       %[ftmp3]            \n\t"

            "gsldlc1    %[ftmp1],   0x37(%[tmp])                        \n\t"
            "gsldrc1    %[ftmp1],   0x30(%[tmp])                        \n\t"
            "gsldlc1    %[ftmp7],   0x3f(%[tmp])                        \n\t"
            "mtc1       %[filter5], %[ftmp4]                            \n\t"
            "gsldrc1    %[ftmp7],   0x38(%[tmp])                        \n\t"
            "pshufh     %[ftmp4],   %[ftmp4],       %[ftmp0]            \n\t"
            "punpcklbh  %[ftmp2],   %[ftmp1],       %[ftmp0]            \n\t"
            "punpckhbh  %[ftmp3],   %[ftmp1],       %[ftmp0]            \n\t"
            "pmullh     %[ftmp2],   %[ftmp2],       %[ftmp4]            \n\t"
            "pmullh     %[ftmp3],   %[ftmp3],       %[ftmp4]            \n\t"
            "paddush    %[ftmp5],   %[ftmp5],       %[ftmp2]            \n\t"
            "paddush    %[ftmp6],   %[ftmp6],       %[ftmp3]            \n\t"
            "punpcklbh  %[ftmp2],   %[ftmp7],       %[ftmp0]            \n\t"
            "punpckhbh  %[ftmp3],   %[ftmp7],       %[ftmp0]            \n\t"
            "pmullh     %[ftmp2],   %[ftmp2],       %[ftmp4]            \n\t"
            "pmullh     %[ftmp3],   %[ftmp3],       %[ftmp4]            \n\t"
            "paddush    %[ftmp8],   %[ftmp8],       %[ftmp2]            \n\t"
            "paddush    %[ftmp9],   %[ftmp9],       %[ftmp3]            \n\t"

            "li         %[tmp0],    0x07                                \n\t"
            "paddush    %[ftmp5],   %[ftmp5],       %[ff_pw_64]         \n\t"
            "paddush    %[ftmp6],   %[ftmp6],       %[ff_pw_64]         \n\t"
            "paddush    %[ftmp8],   %[ftmp8],       %[ff_pw_64]         \n\t"
            "paddush    %[ftmp9],   %[ftmp9],       %[ff_pw_64]         \n\t"
            "mtc1       %[tmp0],    %[ftmp4]                            \n\t"
            "psrlh      %[ftmp5],   %[ftmp5],       %[ftmp4]            \n\t"
            "psrlh      %[ftmp6],   %[ftmp6],       %[ftmp4]            \n\t"
            "psrlh      %[ftmp8],   %[ftmp8],       %[ftmp4]            \n\t"
            "psrlh      %[ftmp9],   %[ftmp9],       %[ftmp4]            \n\t"

            "packushb   %[ftmp1],   %[ftmp5],       %[ftmp6]            \n\t"
            "packushb   %[ftmp2],   %[ftmp8],       %[ftmp9]            \n\t"
            "gssdlc1    %[ftmp1],   0x07(%[dst])                        \n\t"
            "gssdrc1    %[ftmp1],   0x00(%[dst])                        \n\t"
            "gssdlc1    %[ftmp2],   0x0f(%[dst])                        \n\t"
            "gssdrc1    %[ftmp2],   0x08(%[dst])                        \n\t"
            : [ftmp0]"=&f"(ftmp[0]),        [ftmp1]"=&f"(ftmp[1]),
              [ftmp2]"=&f"(ftmp[2]),        [ftmp3]"=&f"(ftmp[3]),
              [ftmp4]"=&f"(ftmp[4]),        [ftmp5]"=&f"(ftmp[5]),
              [ftmp6]"=&f"(ftmp[6]),        [ftmp7]"=&f"(ftmp[7]),
              [ftmp8]"=&f"(ftmp[8]),        [ftmp9]"=&f"(ftmp[9]),
              [tmp0]"=&r"(tmp0)
            : [dst]"r"(dst),                [tmp]"r"(tmp),
              [ff_pw_64]"f"(ff_pw_64),
              [filter0]"r"(filter[0]),      [filter1]"r"(filter[1]),
              [filter2]"r"(filter[2]),      [filter3]"r"(filter[3]),
              [filter4]"r"(filter[4]),      [filter5]"r"(filter[5])
            : "memory"
        );

        dst += dststride;
        tmp += 16;
    }
#else
    for (y = 0; y < h; y++) {
        for (x = 0; x < 16; x++)
            dst[x] = FILTER_6TAP(tmp, filter, 16);
        dst += dststride;
        tmp += 16;
    }
#endif
}

void ff_put_vp8_epel8_h4v6_mmi(uint8_t *dst, ptrdiff_t dststride, uint8_t *src,
        ptrdiff_t srcstride, int h, int mx, int my)
{
#if NOTOK //FIXME fate-vp8-test-vector-006
    const uint8_t *filter = subpel_filters[mx - 1];
    int y;
    uint8_t tmp_array[168];
    uint8_t *tmp = tmp_array;
    double ftmp[7];
    uint64_t tmp0;

    src -= 2 * srcstride;

    for (y = 0; y < h + 6 - 1; y++) {
        /*
        tmp[0] = cm[(filter[2] * src[0] - filter[1] * src[-1] + filter[3] * src[1] - filter[4] * src[2] + 64) >> 7];
        tmp[1] = cm[(filter[2] * src[1] - filter[1] * src[ 0] + filter[3] * src[2] - filter[4] * src[3] + 64) >> 7];
        tmp[2] = cm[(filter[2] * src[2] - filter[1] * src[ 1] + filter[3] * src[3] - filter[4] * src[4] + 64) >> 7];
        tmp[3] = cm[(filter[2] * src[3] - filter[1] * src[ 2] + filter[3] * src[4] - filter[4] * src[5] + 64) >> 7];
        tmp[4] = cm[(filter[2] * src[4] - filter[1] * src[ 3] + filter[3] * src[5] - filter[4] * src[6] + 64) >> 7];
        tmp[5] = cm[(filter[2] * src[5] - filter[1] * src[ 4] + filter[3] * src[6] - filter[4] * src[7] + 64) >> 7];
        tmp[6] = cm[(filter[2] * src[6] - filter[1] * src[ 5] + filter[3] * src[7] - filter[4] * src[8] + 64) >> 7];
        tmp[7] = cm[(filter[2] * src[7] - filter[1] * src[ 6] + filter[3] * src[8] - filter[4] * src[9] + 64) >> 7];
        */
        __asm__ volatile (
            "xor        %[ftmp0],   %[ftmp0],       %[ftmp0]            \n\t"
            "gsldlc1    %[ftmp1],   0x07(%[src])                        \n\t"
            "mtc1       %[filter2], %[ftmp4]                            \n\t"
            "gsldrc1    %[ftmp1],   0x00(%[src])                        \n\t"
            "pshufh     %[ftmp4],   %[ftmp4],       %[ftmp0]            \n\t"
            "punpcklbh  %[ftmp2],   %[ftmp1],       %[ftmp0]            \n\t"
            "punpckhbh  %[ftmp3],   %[ftmp1],       %[ftmp0]            \n\t"
            "pmullh     %[ftmp5],   %[ftmp2],       %[ftmp4]            \n\t"
            "pmullh     %[ftmp6],   %[ftmp3],       %[ftmp4]            \n\t"

            "gsldlc1    %[ftmp1],   0x06(%[src])                        \n\t"
            "mtc1       %[filter1], %[ftmp4]                            \n\t"
            "gsldrc1    %[ftmp1],  -0x01(%[src])                        \n\t"
            "pshufh     %[ftmp4],   %[ftmp4],       %[ftmp0]            \n\t"
            "punpcklbh  %[ftmp2],   %[ftmp1],       %[ftmp0]            \n\t"
            "punpckhbh  %[ftmp3],   %[ftmp1],       %[ftmp0]            \n\t"
            "pmullh     %[ftmp2],   %[ftmp2],       %[ftmp4]            \n\t"
            "pmullh     %[ftmp3],   %[ftmp3],       %[ftmp4]            \n\t"
            "psubush    %[ftmp5],   %[ftmp5],       %[ftmp2]            \n\t"
            "psubush    %[ftmp6],   %[ftmp6],       %[ftmp3]            \n\t"

            "gsldlc1    %[ftmp1],   0x08(%[src])                        \n\t"
            "mtc1       %[filter3], %[ftmp4]                            \n\t"
            "gsldrc1    %[ftmp1],   0x01(%[src])                        \n\t"
            "pshufh     %[ftmp4],   %[ftmp4],       %[ftmp0]            \n\t"
            "punpcklbh  %[ftmp2],   %[ftmp1],       %[ftmp0]            \n\t"
            "punpckhbh  %[ftmp3],   %[ftmp1],       %[ftmp0]            \n\t"
            "pmullh     %[ftmp2],   %[ftmp2],       %[ftmp4]            \n\t"
            "pmullh     %[ftmp3],   %[ftmp3],       %[ftmp4]            \n\t"
            "paddush    %[ftmp5],   %[ftmp5],       %[ftmp2]            \n\t"
            "paddush    %[ftmp6],   %[ftmp6],       %[ftmp3]            \n\t"

            "gsldlc1    %[ftmp1],   0x09(%[src])                        \n\t"
            "mtc1       %[filter4], %[ftmp4]                            \n\t"
            "gsldrc1    %[ftmp1],   0x02(%[src])                        \n\t"
            "pshufh     %[ftmp4],   %[ftmp4],       %[ftmp0]            \n\t"
            "punpcklbh  %[ftmp2],   %[ftmp1],       %[ftmp0]            \n\t"
            "punpckhbh  %[ftmp3],   %[ftmp1],       %[ftmp0]            \n\t"
            "pmullh     %[ftmp2],   %[ftmp2],       %[ftmp4]            \n\t"
            "pmullh     %[ftmp3],   %[ftmp3],       %[ftmp4]            \n\t"
            "psubush    %[ftmp5],   %[ftmp5],       %[ftmp2]            \n\t"
            "psubush    %[ftmp6],   %[ftmp6],       %[ftmp3]            \n\t"

            "li         %[tmp0],    0x07                                \n\t"
            "paddush    %[ftmp5],   %[ftmp5],       %[ff_pw_64]         \n\t"
            "paddush    %[ftmp6],   %[ftmp6],       %[ff_pw_64]         \n\t"
            "mtc1       %[tmp0],    %[ftmp4]                            \n\t"
            "psrlh      %[ftmp5],   %[ftmp5],       %[ftmp4]            \n\t"
            "psrlh      %[ftmp6],   %[ftmp6],       %[ftmp4]            \n\t"

            "packushb   %[ftmp1],   %[ftmp5],       %[ftmp6]            \n\t"
            "gssdlc1    %[ftmp1],   0x07(%[tmp])                        \n\t"
            "gssdrc1    %[ftmp1],   0x00(%[tmp])                        \n\t"
            : [ftmp0]"=&f"(ftmp[0]),        [ftmp1]"=&f"(ftmp[1]),
              [ftmp2]"=&f"(ftmp[2]),        [ftmp3]"=&f"(ftmp[3]),
              [ftmp4]"=&f"(ftmp[4]),        [ftmp5]"=&f"(ftmp[5]),
              [ftmp6]"=&f"(ftmp[6]),
              [tmp0]"=&r"(tmp0)
            : [tmp]"r"(tmp),                [src]"r"(src),
              [ff_pw_64]"f"(ff_pw_64),
              [filter1]"r"(filter[1]),      [filter2]"r"(filter[2]),
              [filter3]"r"(filter[3]),      [filter4]"r"(filter[4])
            : "memory"
        );

        tmp += 8;
        src += srcstride;
    }
#else
    const uint8_t *filter = subpel_filters[mx - 1];
    const uint8_t *cm     = ff_crop_tab + MAX_NEG_CROP;
    int x, y;
    uint8_t tmp_array[168];
    uint8_t *tmp = tmp_array;

    src -= 2 * srcstride;

    for (y = 0; y < h + 6 - 1; y++) {
        for (x = 0; x < 8; x++)
            tmp[x] = FILTER_4TAP(src, filter, 1);
        tmp += 8;
        src += srcstride;
    }
#endif

    tmp    = tmp_array + 16;
    filter = subpel_filters[my - 1];

#if NOTOK //FIXME fate-vp8-test-vector-006
    for (y = 0; y < h; y++) {
        /*
        dst[0] = cm[(filter[2]*tmp[0] - filter[1]*tmp[-8] + filter[0]*tmp[-16] + filter[3]*tmp[ 8] - filter[4]*tmp[16] + filter[5]*tmp[24] + 64) >> 7];
        dst[1] = cm[(filter[2]*tmp[1] - filter[1]*tmp[-7] + filter[0]*tmp[-15] + filter[3]*tmp[ 9] - filter[4]*tmp[17] + filter[5]*tmp[25] + 64) >> 7];
        dst[2] = cm[(filter[2]*tmp[2] - filter[1]*tmp[-6] + filter[0]*tmp[-14] + filter[3]*tmp[10] - filter[4]*tmp[18] + filter[5]*tmp[26] + 64) >> 7];
        dst[3] = cm[(filter[2]*tmp[3] - filter[1]*tmp[-5] + filter[0]*tmp[-13] + filter[3]*tmp[11] - filter[4]*tmp[19] + filter[5]*tmp[27] + 64) >> 7];
        dst[4] = cm[(filter[2]*tmp[4] - filter[1]*tmp[-4] + filter[0]*tmp[-12] + filter[3]*tmp[12] - filter[4]*tmp[20] + filter[5]*tmp[28] + 64) >> 7];
        dst[5] = cm[(filter[2]*tmp[5] - filter[1]*tmp[-3] + filter[0]*tmp[-11] + filter[3]*tmp[13] - filter[4]*tmp[21] + filter[5]*tmp[29] + 64) >> 7];
        dst[6] = cm[(filter[2]*tmp[6] - filter[1]*tmp[-2] + filter[0]*tmp[-10] + filter[3]*tmp[14] - filter[4]*tmp[22] + filter[5]*tmp[30] + 64) >> 7];
        dst[7] = cm[(filter[2]*tmp[7] - filter[1]*tmp[-1] + filter[0]*tmp[ -9] + filter[3]*tmp[15] - filter[4]*tmp[23] + filter[5]*tmp[31] + 64) >> 7];
        */
        __asm__ volatile (
            "xor        %[ftmp0],   %[ftmp0],       %[ftmp0]            \n\t"
            "gsldlc1    %[ftmp1],   0x07(%[tmp])                        \n\t"
            "mtc1       %[filter2], %[ftmp4]                            \n\t"
            "gsldrc1    %[ftmp1],   0x00(%[tmp])                        \n\t"
            "pshufh     %[ftmp4],   %[ftmp4],       %[ftmp0]            \n\t"
            "punpcklbh  %[ftmp2],   %[ftmp1],       %[ftmp0]            \n\t"
            "punpckhbh  %[ftmp3],   %[ftmp1],       %[ftmp0]            \n\t"
            "pmullh     %[ftmp5],   %[ftmp2],       %[ftmp4]            \n\t"
            "pmullh     %[ftmp6],   %[ftmp3],       %[ftmp4]            \n\t"

            "gsldlc1    %[ftmp1], -0x01(%[tmp])                         \n\t"
            "mtc1       %[filter1], %[ftmp4]                            \n\t"
            "gsldrc1    %[ftmp1], -0x08(%[tmp])                         \n\t"
            "pshufh     %[ftmp4],   %[ftmp4],       %[ftmp0]            \n\t"
            "punpcklbh  %[ftmp2],   %[ftmp1],       %[ftmp0]            \n\t"
            "punpckhbh  %[ftmp3],   %[ftmp1],       %[ftmp0]            \n\t"
            "pmullh     %[ftmp2],   %[ftmp2],       %[ftmp4]            \n\t"
            "pmullh     %[ftmp3],   %[ftmp3],       %[ftmp4]            \n\t"
            "psubush    %[ftmp5],   %[ftmp5],       %[ftmp2]            \n\t"
            "psubush    %[ftmp6],   %[ftmp6],       %[ftmp3]            \n\t"

            "gsldlc1    %[ftmp1],  -0x09(%[tmp])                        \n\t"
            "mtc1       %[filter0], %[ftmp4]                            \n\t"
            "gsldrc1    %[ftmp1],  -0x10(%[tmp])                        \n\t"
            "pshufh     %[ftmp4],   %[ftmp4],       %[ftmp0]            \n\t"
            "punpcklbh  %[ftmp2],   %[ftmp1],       %[ftmp0]            \n\t"
            "punpckhbh  %[ftmp3],   %[ftmp1],       %[ftmp0]            \n\t"
            "pmullh     %[ftmp2],   %[ftmp2],       %[ftmp4]            \n\t"
            "pmullh     %[ftmp3],   %[ftmp3],       %[ftmp4]            \n\t"
            "paddush    %[ftmp5],   %[ftmp5],       %[ftmp2]            \n\t"
            "paddush    %[ftmp6],   %[ftmp6],       %[ftmp3]            \n\t"

            "gsldlc1    %[ftmp1],   0x0f(%[tmp])                        \n\t"
            "mtc1       %[filter3], %[ftmp4]                            \n\t"
            "gsldrc1    %[ftmp1],   0x08(%[tmp])                        \n\t"
            "pshufh     %[ftmp4],   %[ftmp4],       %[ftmp0]            \n\t"
            "punpcklbh  %[ftmp2],   %[ftmp1],       %[ftmp0]            \n\t"
            "punpckhbh  %[ftmp3],   %[ftmp1],       %[ftmp0]            \n\t"
            "pmullh     %[ftmp2],   %[ftmp2],       %[ftmp4]            \n\t"
            "pmullh     %[ftmp3],   %[ftmp3],       %[ftmp4]            \n\t"
            "paddush    %[ftmp5],   %[ftmp5],       %[ftmp2]            \n\t"
            "paddush    %[ftmp6],   %[ftmp6],       %[ftmp3]            \n\t"

            "gsldlc1    %[ftmp1],   0x17(%[tmp])                        \n\t"
            "mtc1       %[filter4], %[ftmp4]                            \n\t"
            "gsldrc1    %[ftmp1],   0x10(%[tmp])                        \n\t"
            "pshufh     %[ftmp4],   %[ftmp4],       %[ftmp0]            \n\t"
            "punpcklbh  %[ftmp2],   %[ftmp1],       %[ftmp0]            \n\t"
            "punpckhbh  %[ftmp3],   %[ftmp1],       %[ftmp0]            \n\t"
            "pmullh     %[ftmp2],   %[ftmp2],       %[ftmp4]            \n\t"
            "pmullh     %[ftmp3],   %[ftmp3],       %[ftmp4]            \n\t"
            "psubush    %[ftmp5],   %[ftmp5],       %[ftmp2]            \n\t"
            "psubush    %[ftmp6],   %[ftmp6],       %[ftmp3]            \n\t"

            "gsldlc1    %[ftmp1],   0x1f(%[tmp])                        \n\t"
            "mtc1       %[filter5], %[ftmp4]                            \n\t"
            "gsldrc1    %[ftmp1],   0x18(%[tmp])                        \n\t"
            "pshufh     %[ftmp4],   %[ftmp4],       %[ftmp0]            \n\t"
            "punpcklbh  %[ftmp2],   %[ftmp1],       %[ftmp0]            \n\t"
            "punpckhbh  %[ftmp3],   %[ftmp1],       %[ftmp0]            \n\t"
            "pmullh     %[ftmp2],   %[ftmp2],       %[ftmp4]            \n\t"
            "pmullh     %[ftmp3],   %[ftmp3],       %[ftmp4]            \n\t"
            "paddush    %[ftmp5],   %[ftmp5],       %[ftmp2]            \n\t"
            "paddush    %[ftmp6],   %[ftmp6],       %[ftmp3]            \n\t"

            "li         %[tmp0],    0x07                                \n\t"
            "paddush    %[ftmp5],   %[ftmp5],       %[ff_pw_64]         \n\t"
            "paddush    %[ftmp6],   %[ftmp6],       %[ff_pw_64]         \n\t"
            "mtc1       %[tmp0],    %[ftmp4]                            \n\t"
            "psrlh      %[ftmp5],   %[ftmp5],       %[ftmp4]            \n\t"
            "psrlh      %[ftmp6],   %[ftmp6],       %[ftmp4]            \n\t"

            "packushb   %[ftmp1],   %[ftmp5],       %[ftmp6]            \n\t"
            "gssdlc1    %[ftmp1],   0x07(%[dst])                        \n\t"
            "gssdrc1    %[ftmp1],   0x00(%[dst])                        \n\t"
            : [ftmp0]"=&f"(ftmp[0]),        [ftmp1]"=&f"(ftmp[1]),
              [ftmp2]"=&f"(ftmp[2]),        [ftmp3]"=&f"(ftmp[3]),
              [ftmp4]"=&f"(ftmp[4]),        [ftmp5]"=&f"(ftmp[5]),
              [ftmp6]"=&f"(ftmp[6]),
              [tmp0]"=&r"(tmp0)
            : [dst]"r"(dst),                [tmp]"r"(tmp),
              [ff_pw_64]"f"(ff_pw_64),
              [filter0]"r"(filter[0]),      [filter1]"r"(filter[1]),
              [filter2]"r"(filter[2]),      [filter3]"r"(filter[3]),
              [filter4]"r"(filter[4]),      [filter5]"r"(filter[5])
            : "memory"
        );

        dst += dststride;
        tmp += 8;
    }
#else
    for (y = 0; y < h; y++) {
        for (x = 0; x < 8; x++)
            dst[x] = FILTER_6TAP(tmp, filter, 8);
        dst += dststride;
        tmp += 8;
    }
#endif
}

void ff_put_vp8_epel4_h4v6_mmi(uint8_t *dst, ptrdiff_t dststride, uint8_t *src,
        ptrdiff_t srcstride, int h, int mx, int my)
{
#if NOTOK //FIXME fate-vp8-test-vector-006
    const uint8_t *filter = subpel_filters[mx - 1];
    int y;
    uint8_t tmp_array[52];
    uint8_t *tmp = tmp_array;
    double ftmp[5];
    uint64_t tmp0;

    src -= 2 * srcstride;

    for (y = 0; y < h + 5; y++) {
        /*
        tmp[0] = cm[(filter[2] * src[0] - filter[1] * src[-1] + filter[3] * src[1] - filter[4] * src[2] + 64) >> 7];
        tmp[1] = cm[(filter[2] * src[1] - filter[1] * src[ 0] + filter[3] * src[2] - filter[4] * src[3] + 64) >> 7];
        tmp[2] = cm[(filter[2] * src[2] - filter[1] * src[ 1] + filter[3] * src[3] - filter[4] * src[4] + 64) >> 7];
        tmp[3] = cm[(filter[2] * src[3] - filter[1] * src[ 2] + filter[3] * src[4] - filter[4] * src[5] + 64) >> 7];
        */
        __asm__ volatile (
            "xor        %[ftmp0],   %[ftmp0],       %[ftmp0]            \n\t"
            "gslwlc1    %[ftmp1],   0x03(%[src])                        \n\t"
            "mtc1       %[filter2], %[ftmp3]                            \n\t"
            "gslwrc1    %[ftmp1],   0x00(%[src])                        \n\t"
            "pshufh     %[ftmp3],   %[ftmp3],       %[ftmp0]            \n\t"
            "punpcklbh  %[ftmp2],   %[ftmp1],       %[ftmp0]            \n\t"
            "pmullh     %[ftmp4],   %[ftmp2],       %[ftmp3]            \n\t"

            "gslwlc1    %[ftmp1],   0x02(%[src])                        \n\t"
            "mtc1       %[filter1], %[ftmp3]                            \n\t"
            "gslwrc1    %[ftmp1],  -0x01(%[src])                        \n\t"
            "pshufh     %[ftmp3],   %[ftmp3],       %[ftmp0]            \n\t"
            "punpcklbh  %[ftmp2],   %[ftmp1],       %[ftmp0]            \n\t"
            "pmullh     %[ftmp2],   %[ftmp2],       %[ftmp3]            \n\t"
            "psubush    %[ftmp4],   %[ftmp4],       %[ftmp2]            \n\t"

            "gslwlc1    %[ftmp1],   0x04(%[src])                        \n\t"
            "mtc1       %[filter3], %[ftmp3]                            \n\t"
            "gslwrc1    %[ftmp1],   0x01(%[src])                        \n\t"
            "pshufh     %[ftmp3],   %[ftmp3],       %[ftmp0]            \n\t"
            "punpcklbh  %[ftmp2],   %[ftmp1],       %[ftmp0]            \n\t"
            "pmullh     %[ftmp2],   %[ftmp2],       %[ftmp3]            \n\t"
            "paddush    %[ftmp4],   %[ftmp4],       %[ftmp2]            \n\t"

            "gslwlc1    %[ftmp1],   0x05(%[src])                        \n\t"
            "mtc1       %[filter4], %[ftmp3]                            \n\t"
            "gslwrc1    %[ftmp1],   0x02(%[src])                        \n\t"
            "pshufh     %[ftmp3],   %[ftmp3],       %[ftmp0]            \n\t"
            "punpcklbh  %[ftmp2],   %[ftmp1],       %[ftmp0]            \n\t"
            "pmullh     %[ftmp2],   %[ftmp2],       %[ftmp3]            \n\t"
            "psubush    %[ftmp4],   %[ftmp4],       %[ftmp2]            \n\t"

            "li         %[tmp0],    0x07                                \n\t"
            "paddush    %[ftmp4],   %[ftmp4],       %[ff_pw_64]         \n\t"
            "mtc1       %[tmp0],    %[ftmp3]                            \n\t"
            "psrlh      %[ftmp4],   %[ftmp4],       %[ftmp3]            \n\t"

            "packushb   %[ftmp1],   %[ftmp4],       %[ftmp0]            \n\t"
            "gsswlc1    %[ftmp1],   0x03(%[tmp])                        \n\t"
            "gsswrc1    %[ftmp1],   0x00(%[tmp])                        \n\t"
            : [ftmp0]"=&f"(ftmp[0]),        [ftmp1]"=&f"(ftmp[1]),
              [ftmp2]"=&f"(ftmp[2]),        [ftmp3]"=&f"(ftmp[3]),
              [ftmp4]"=&f"(ftmp[4]),
              [tmp0]"=&r"(tmp0)
            : [tmp]"r"(tmp),                [src]"r"(src),
              [ff_pw_64]"f"(ff_pw_64),
              [filter1]"r"(filter[1]),      [filter2]"r"(filter[2]),
              [filter3]"r"(filter[3]),      [filter4]"r"(filter[4])
            : "memory"
        );

        tmp += 4;
        src += srcstride;
    }
#else
    const uint8_t *filter = subpel_filters[mx - 1];
    const uint8_t *cm     = ff_crop_tab + MAX_NEG_CROP;
    int x, y;
    uint8_t tmp_array[52];
    uint8_t *tmp = tmp_array;

    src -= 2 * srcstride;

    for (y = 0; y < h + 5; y++) {
        for (x = 0; x < 4; x++)
            tmp[x] = FILTER_4TAP(src, filter, 1);
        tmp += 4;
        src += srcstride;
    }
#endif

    tmp    = tmp_array + 8;
    filter = subpel_filters[my - 1];

#if NOTOK //FIXME fate-vp8-test-vector-006
    for (y = 0; y < h; y++) {
        /*
        dst[0] = cm[(filter[2]*tmp[0] - filter[1]*tmp[-4] + filter[0]*tmp[-8] + filter[3]*tmp[4] - filter[4]*tmp[ 8] + filter[5]*tmp[12] + 64) >> 7];
        dst[1] = cm[(filter[2]*tmp[1] - filter[1]*tmp[-3] + filter[0]*tmp[-7] + filter[3]*tmp[5] - filter[4]*tmp[ 9] + filter[5]*tmp[13] + 64) >> 7];
        dst[2] = cm[(filter[2]*tmp[2] - filter[1]*tmp[-2] + filter[0]*tmp[-6] + filter[3]*tmp[6] - filter[4]*tmp[10] + filter[5]*tmp[14] + 64) >> 7];
        dst[3] = cm[(filter[2]*tmp[3] - filter[1]*tmp[-1] + filter[0]*tmp[-5] + filter[3]*tmp[7] - filter[4]*tmp[11] + filter[5]*tmp[15] + 64) >> 7];
        */
        __asm__ volatile (
            "xor        %[ftmp0],   %[ftmp0],       %[ftmp0]            \n\t"
            "gslwlc1    %[ftmp1],   0x03(%[tmp])                        \n\t"
            "mtc1       %[filter2], %[ftmp3]                            \n\t"
            "gslwrc1    %[ftmp1],   0x00(%[tmp])                        \n\t"
            "pshufh     %[ftmp3],   %[ftmp3],       %[ftmp0]            \n\t"
            "punpcklbh  %[ftmp2],   %[ftmp1],       %[ftmp0]            \n\t"
            "pmullh     %[ftmp4],   %[ftmp2],       %[ftmp3]            \n\t"

            "gslwlc1    %[ftmp1],  -0x01(%[tmp])                        \n\t"
            "mtc1       %[filter1], %[ftmp3]                            \n\t"
            "gslwrc1    %[ftmp1],  -0x04(%[tmp])                        \n\t"
            "pshufh     %[ftmp3],   %[ftmp3],       %[ftmp0]            \n\t"
            "punpcklbh  %[ftmp2],   %[ftmp1],       %[ftmp0]            \n\t"
            "pmullh     %[ftmp2],   %[ftmp2],       %[ftmp3]            \n\t"
            "psubush    %[ftmp4],   %[ftmp4],       %[ftmp2]            \n\t"

            "gslwlc1    %[ftmp1],  -0x05(%[tmp])                        \n\t"
            "mtc1       %[filter0], %[ftmp3]                            \n\t"
            "gslwrc1    %[ftmp1],  -0x08(%[tmp])                        \n\t"
            "pshufh     %[ftmp3],   %[ftmp3],       %[ftmp0]            \n\t"
            "punpcklbh  %[ftmp2],   %[ftmp1],       %[ftmp0]            \n\t"
            "pmullh     %[ftmp2],   %[ftmp2],       %[ftmp3]            \n\t"
            "paddush    %[ftmp4],   %[ftmp4],       %[ftmp2]            \n\t"

            "gslwlc1    %[ftmp1],   0x07(%[tmp])                        \n\t"
            "mtc1       %[filter3], %[ftmp3]                            \n\t"
            "gslwrc1    %[ftmp1],   0x04(%[tmp])                        \n\t"
            "pshufh     %[ftmp3],   %[ftmp3],       %[ftmp0]            \n\t"
            "punpcklbh  %[ftmp2],   %[ftmp1],       %[ftmp0]            \n\t"
            "pmullh     %[ftmp2],   %[ftmp2],       %[ftmp3]            \n\t"
            "paddush    %[ftmp4],   %[ftmp4],       %[ftmp2]            \n\t"

            "gslwlc1    %[ftmp1],   0x0b(%[tmp])                        \n\t"
            "mtc1       %[filter4], %[ftmp3]                            \n\t"
            "gslwrc1    %[ftmp1],   0x08(%[tmp])                        \n\t"
            "pshufh     %[ftmp3],   %[ftmp3],       %[ftmp0]            \n\t"
            "punpcklbh  %[ftmp2],   %[ftmp1],       %[ftmp0]            \n\t"
            "pmullh     %[ftmp2],   %[ftmp2],       %[ftmp3]            \n\t"
            "psubush    %[ftmp4],   %[ftmp4],       %[ftmp2]            \n\t"

            "gslwlc1    %[ftmp1],   0x0f(%[tmp])                        \n\t"
            "mtc1       %[filter5], %[ftmp3]                            \n\t"
            "gslwrc1    %[ftmp1],   0x0c(%[tmp])                        \n\t"
            "pshufh     %[ftmp3],   %[ftmp3],       %[ftmp0]            \n\t"
            "punpcklbh  %[ftmp2],   %[ftmp1],       %[ftmp0]            \n\t"
            "pmullh     %[ftmp2],   %[ftmp2],       %[ftmp3]            \n\t"
            "paddush    %[ftmp4],   %[ftmp4],       %[ftmp2]            \n\t"

            "li         %[tmp0],    0x07                                \n\t"
            "paddush    %[ftmp4],   %[ftmp4],       %[ff_pw_64]         \n\t"
            "mtc1       %[tmp0],    %[ftmp3]                            \n\t"
            "psrlh      %[ftmp4],   %[ftmp4],       %[ftmp3]            \n\t"

            "packushb   %[ftmp1],   %[ftmp4],       %[ftmp0]            \n\t"
            "gsswlc1    %[ftmp1],   0x03(%[dst])                        \n\t"
            "gsswrc1    %[ftmp1],   0x00(%[dst])                        \n\t"
            : [ftmp0]"=&f"(ftmp[0]),        [ftmp1]"=&f"(ftmp[1]),
              [ftmp2]"=&f"(ftmp[2]),        [ftmp3]"=&f"(ftmp[3]),
              [ftmp4]"=&f"(ftmp[4]),
              [tmp0]"=&r"(tmp0)
            : [dst]"r"(dst),                [tmp]"r"(tmp),
              [ff_pw_64]"f"(ff_pw_64),
              [filter0]"r"(filter[0]),      [filter1]"r"(filter[1]),
              [filter2]"r"(filter[2]),      [filter3]"r"(filter[3]),
              [filter4]"r"(filter[4]),      [filter5]"r"(filter[5])
            : "memory"
        );

        dst += dststride;
        tmp += 4;
    }
#else
    for (y = 0; y < h; y++) {
        for (x = 0; x < 4; x++)
            dst[x] = FILTER_6TAP(tmp, filter, 4);
        dst += dststride;
        tmp += 4;
    }
#endif
}

void ff_put_vp8_epel16_h6v4_mmi(uint8_t *dst, ptrdiff_t dststride, uint8_t *src,
        ptrdiff_t srcstride, int h, int mx, int my)
{
#if NOTOK //FIXME fate-vp8-test-vector-002 006 009
    const uint8_t *filter = subpel_filters[mx - 1];
    int y;
    uint8_t tmp_array[560];
    uint8_t *tmp = tmp_array;
    double ftmp[10];
    uint64_t tmp0;

    src -= srcstride;

    for (y = 0; y < h + 3; y++) {
        /*
        dst[0] = cm[(filter[2]*src[0] - filter[1]*src[-1] + filter[0]*src[-2] + filter[3]*src[1] - filter[4]*src[2] + filter[5]*src[ 3] + 64) >> 7];
        dst[1] = cm[(filter[2]*src[1] - filter[1]*src[ 0] + filter[0]*src[-1] + filter[3]*src[2] - filter[4]*src[3] + filter[5]*src[ 4] + 64) >> 7];
        dst[2] = cm[(filter[2]*src[2] - filter[1]*src[ 1] + filter[0]*src[ 0] + filter[3]*src[3] - filter[4]*src[4] + filter[5]*src[ 5] + 64) >> 7];
        dst[3] = cm[(filter[2]*src[3] - filter[1]*src[ 2] + filter[0]*src[ 1] + filter[3]*src[4] - filter[4]*src[5] + filter[5]*src[ 6] + 64) >> 7];
        dst[4] = cm[(filter[2]*src[4] - filter[1]*src[ 3] + filter[0]*src[ 2] + filter[3]*src[5] - filter[4]*src[6] + filter[5]*src[ 7] + 64) >> 7];
        dst[5] = cm[(filter[2]*src[5] - filter[1]*src[ 4] + filter[0]*src[ 3] + filter[3]*src[6] - filter[4]*src[7] + filter[5]*src[ 8] + 64) >> 7];
        dst[6] = cm[(filter[2]*src[6] - filter[1]*src[ 5] + filter[0]*src[ 4] + filter[3]*src[7] - filter[4]*src[8] + filter[5]*src[ 9] + 64) >> 7];
        dst[7] = cm[(filter[2]*src[7] - filter[1]*src[ 6] + filter[0]*src[ 5] + filter[3]*src[8] - filter[4]*src[9] + filter[5]*src[10] + 64) >> 7];

        dst[ 8] = cm[(filter[2]*src[ 8] - filter[1]*src[ 7] + filter[0]*src[ 6] + filter[3]*src[ 9] - filter[4]*src[10] + filter[5]*src[11] + 64) >> 7];
        dst[ 9] = cm[(filter[2]*src[ 9] - filter[1]*src[ 8] + filter[0]*src[ 7] + filter[3]*src[10] - filter[4]*src[11] + filter[5]*src[12] + 64) >> 7];
        dst[10] = cm[(filter[2]*src[10] - filter[1]*src[ 9] + filter[0]*src[ 8] + filter[3]*src[11] - filter[4]*src[12] + filter[5]*src[13] + 64) >> 7];
        dst[11] = cm[(filter[2]*src[11] - filter[1]*src[10] + filter[0]*src[ 9] + filter[3]*src[12] - filter[4]*src[13] + filter[5]*src[14] + 64) >> 7];
        dst[12] = cm[(filter[2]*src[12] - filter[1]*src[11] + filter[0]*src[10] + filter[3]*src[13] - filter[4]*src[14] + filter[5]*src[15] + 64) >> 7];
        dst[13] = cm[(filter[2]*src[13] - filter[1]*src[12] + filter[0]*src[11] + filter[3]*src[14] - filter[4]*src[15] + filter[5]*src[16] + 64) >> 7];
        dst[14] = cm[(filter[2]*src[14] - filter[1]*src[13] + filter[0]*src[12] + filter[3]*src[15] - filter[4]*src[16] + filter[5]*src[17] + 64) >> 7];
        dst[15] = cm[(filter[2]*src[15] - filter[1]*src[14] + filter[0]*src[13] + filter[3]*src[16] - filter[4]*src[17] + filter[5]*src[18] + 64) >> 7];
        */
        __asm__ volatile (
            "xor        %[ftmp0],   %[ftmp0],       %[ftmp0]            \n\t"
            "gsldlc1    %[ftmp1],   0x07(%[src])                        \n\t"
            "gsldrc1    %[ftmp1],   0x00(%[src])                        \n\t"
            "gsldlc1    %[ftmp7],   0x0f(%[src])                        \n\t"
            "mtc1       %[filter2], %[ftmp4]                            \n\t"
            "gsldrc1    %[ftmp7],   0x08(%[src])                        \n\t"
            "pshufh     %[ftmp4],   %[ftmp4],       %[ftmp0]            \n\t"
            "punpcklbh  %[ftmp2],   %[ftmp1],       %[ftmp0]            \n\t"
            "punpckhbh  %[ftmp3],   %[ftmp1],       %[ftmp0]            \n\t"
            "pmullh     %[ftmp5],   %[ftmp2],       %[ftmp4]            \n\t"
            "pmullh     %[ftmp6],   %[ftmp3],       %[ftmp4]            \n\t"
            "punpcklbh  %[ftmp2],   %[ftmp7],       %[ftmp0]            \n\t"
            "punpckhbh  %[ftmp3],   %[ftmp7],       %[ftmp0]            \n\t"
            "pmullh     %[ftmp8],   %[ftmp2],       %[ftmp4]            \n\t"
            "pmullh     %[ftmp9],   %[ftmp3],       %[ftmp4]            \n\t"

            "gsldlc1    %[ftmp1],   0x06(%[src])                        \n\t"
            "gsldrc1    %[ftmp1],  -0x01(%[src])                        \n\t"
            "gsldlc1    %[ftmp7],   0x0e(%[src])                        \n\t"
            "mtc1       %[filter1], %[ftmp4]                            \n\t"
            "gsldrc1    %[ftmp7],   0x07(%[src])                        \n\t"
            "pshufh     %[ftmp4],   %[ftmp4],       %[ftmp0]            \n\t"
            "punpcklbh  %[ftmp2],   %[ftmp1],       %[ftmp0]            \n\t"
            "punpckhbh  %[ftmp3],   %[ftmp1],       %[ftmp0]            \n\t"
            "pmullh     %[ftmp2],   %[ftmp2],       %[ftmp4]            \n\t"
            "pmullh     %[ftmp3],   %[ftmp3],       %[ftmp4]            \n\t"
            "psubush    %[ftmp5],   %[ftmp5],       %[ftmp2]            \n\t"
            "psubush    %[ftmp6],   %[ftmp6],       %[ftmp3]            \n\t"
            "punpcklbh  %[ftmp2],   %[ftmp7],       %[ftmp0]            \n\t"
            "punpckhbh  %[ftmp3],   %[ftmp7],       %[ftmp0]            \n\t"
            "pmullh     %[ftmp2],   %[ftmp2],       %[ftmp4]            \n\t"
            "pmullh     %[ftmp3],   %[ftmp3],       %[ftmp4]            \n\t"
            "psubush    %[ftmp8],   %[ftmp8],       %[ftmp2]            \n\t"
            "psubush    %[ftmp9],   %[ftmp9],       %[ftmp3]            \n\t"

            "gsldlc1    %[ftmp1],   0x05(%[src])                        \n\t"
            "gsldrc1    %[ftmp1],  -0x02(%[src])                        \n\t"
            "gsldlc1    %[ftmp7],   0x0d(%[src])                        \n\t"
            "mtc1       %[filter0], %[ftmp4]                            \n\t"
            "gsldrc1    %[ftmp7],   0x06(%[src])                        \n\t"
            "pshufh     %[ftmp4],   %[ftmp4],       %[ftmp0]            \n\t"
            "punpcklbh  %[ftmp2],   %[ftmp1],       %[ftmp0]            \n\t"
            "punpckhbh  %[ftmp3],   %[ftmp1],       %[ftmp0]            \n\t"
            "pmullh     %[ftmp2],   %[ftmp2],       %[ftmp4]            \n\t"
            "pmullh     %[ftmp3],   %[ftmp3],       %[ftmp4]            \n\t"
            "paddush    %[ftmp5],   %[ftmp5],       %[ftmp2]            \n\t"
            "paddush    %[ftmp6],   %[ftmp6],       %[ftmp3]            \n\t"
            "punpcklbh  %[ftmp2],   %[ftmp7],       %[ftmp0]            \n\t"
            "punpckhbh  %[ftmp3],   %[ftmp7],       %[ftmp0]            \n\t"
            "pmullh     %[ftmp2],   %[ftmp2],       %[ftmp4]            \n\t"
            "pmullh     %[ftmp3],   %[ftmp3],       %[ftmp4]            \n\t"
            "paddush    %[ftmp8],   %[ftmp8],       %[ftmp2]            \n\t"
            "paddush    %[ftmp9],   %[ftmp9],       %[ftmp3]            \n\t"

            "gsldlc1    %[ftmp1],   0x08(%[src])                        \n\t"
            "gsldrc1    %[ftmp1],   0x01(%[src])                        \n\t"
            "gsldlc1    %[ftmp7],   0x10(%[src])                        \n\t"
            "mtc1       %[filter3], %[ftmp4]                            \n\t"
            "gsldrc1    %[ftmp7],   0x09(%[src])                        \n\t"
            "pshufh     %[ftmp4],   %[ftmp4],       %[ftmp0]            \n\t"
            "punpcklbh  %[ftmp2],   %[ftmp1],       %[ftmp0]            \n\t"
            "punpckhbh  %[ftmp3],   %[ftmp1],       %[ftmp0]            \n\t"
            "pmullh     %[ftmp2],   %[ftmp2],       %[ftmp4]            \n\t"
            "pmullh     %[ftmp3],   %[ftmp3],       %[ftmp4]            \n\t"
            "paddush    %[ftmp5],   %[ftmp5],       %[ftmp2]            \n\t"
            "paddush    %[ftmp6],   %[ftmp6],       %[ftmp3]            \n\t"
            "punpcklbh  %[ftmp2],   %[ftmp7],       %[ftmp0]            \n\t"
            "punpckhbh  %[ftmp3],   %[ftmp7],       %[ftmp0]            \n\t"
            "pmullh     %[ftmp2],   %[ftmp2],       %[ftmp4]            \n\t"
            "pmullh     %[ftmp3],   %[ftmp3],       %[ftmp4]            \n\t"
            "paddush    %[ftmp8],   %[ftmp8],       %[ftmp2]            \n\t"
            "paddush    %[ftmp9],   %[ftmp9],       %[ftmp3]            \n\t"

            "gsldlc1    %[ftmp1],   0x09(%[src])                        \n\t"
            "gsldrc1    %[ftmp1],   0x02(%[src])                        \n\t"
            "gsldlc1    %[ftmp7],   0x11(%[src])                        \n\t"
            "mtc1       %[filter4], %[ftmp4]                            \n\t"
            "gsldrc1    %[ftmp7],   0x0a(%[src])                        \n\t"
            "pshufh     %[ftmp4],   %[ftmp4],       %[ftmp0]            \n\t"
            "punpcklbh  %[ftmp2],   %[ftmp1],       %[ftmp0]            \n\t"
            "punpckhbh  %[ftmp3],   %[ftmp1],       %[ftmp0]            \n\t"
            "pmullh     %[ftmp2],   %[ftmp2],       %[ftmp4]            \n\t"
            "pmullh     %[ftmp3],   %[ftmp3],       %[ftmp4]            \n\t"
            "psubush    %[ftmp5],   %[ftmp5],       %[ftmp2]            \n\t"
            "psubush    %[ftmp6],   %[ftmp6],       %[ftmp3]            \n\t"
            "punpcklbh  %[ftmp2],   %[ftmp7],       %[ftmp0]            \n\t"
            "punpckhbh  %[ftmp3],   %[ftmp7],       %[ftmp0]            \n\t"
            "pmullh     %[ftmp2],   %[ftmp2],       %[ftmp4]            \n\t"
            "pmullh     %[ftmp3],   %[ftmp3],       %[ftmp4]            \n\t"
            "psubush    %[ftmp8],   %[ftmp8],       %[ftmp2]            \n\t"
            "psubush    %[ftmp9],   %[ftmp9],       %[ftmp3]            \n\t"

            "gsldlc1    %[ftmp1],   0x0a(%[src])                        \n\t"
            "gsldrc1    %[ftmp1],   0x03(%[src])                        \n\t"
            "gsldlc1    %[ftmp7],   0x12(%[src])                        \n\t"
            "mtc1       %[filter5], %[ftmp4]                            \n\t"
            "gsldrc1    %[ftmp7],   0x0b(%[src])                        \n\t"
            "pshufh     %[ftmp4],   %[ftmp4],       %[ftmp0]            \n\t"
            "punpcklbh  %[ftmp2],   %[ftmp1],       %[ftmp0]            \n\t"
            "punpckhbh  %[ftmp3],   %[ftmp1],       %[ftmp0]            \n\t"
            "pmullh     %[ftmp2],   %[ftmp2],       %[ftmp4]            \n\t"
            "pmullh     %[ftmp3],   %[ftmp3],       %[ftmp4]            \n\t"
            "paddush    %[ftmp5],   %[ftmp5],       %[ftmp2]            \n\t"
            "paddush    %[ftmp6],   %[ftmp6],       %[ftmp3]            \n\t"
            "punpcklbh  %[ftmp2],   %[ftmp7],       %[ftmp0]            \n\t"
            "punpckhbh  %[ftmp3],   %[ftmp7],       %[ftmp0]            \n\t"
            "pmullh     %[ftmp2],   %[ftmp2],       %[ftmp4]            \n\t"
            "pmullh     %[ftmp3],   %[ftmp3],       %[ftmp4]            \n\t"
            "paddush    %[ftmp8],   %[ftmp8],       %[ftmp2]            \n\t"
            "paddush    %[ftmp9],   %[ftmp9],       %[ftmp3]            \n\t"

            "li         %[tmp0],    0x07                                \n\t"
            "paddush    %[ftmp5],   %[ftmp5],       %[ff_pw_64]         \n\t"
            "paddush    %[ftmp6],   %[ftmp6],       %[ff_pw_64]         \n\t"
            "paddush    %[ftmp8],   %[ftmp8],       %[ff_pw_64]         \n\t"
            "paddush    %[ftmp9],   %[ftmp9],       %[ff_pw_64]         \n\t"
            "mtc1       %[tmp0],    %[ftmp4]                            \n\t"
            "psrlh      %[ftmp5],   %[ftmp5],       %[ftmp4]            \n\t"
            "psrlh      %[ftmp6],   %[ftmp6],       %[ftmp4]            \n\t"
            "psrlh      %[ftmp8],   %[ftmp8],       %[ftmp4]            \n\t"
            "psrlh      %[ftmp9],   %[ftmp9],       %[ftmp4]            \n\t"

            "packushb   %[ftmp1],   %[ftmp5],       %[ftmp6]            \n\t"
            "packushb   %[ftmp2],   %[ftmp8],       %[ftmp9]            \n\t"
            "gssdlc1    %[ftmp1],   0x07(%[tmp])                        \n\t"
            "gssdrc1    %[ftmp1],   0x00(%[tmp])                        \n\t"
            "gssdlc1    %[ftmp2],   0x0f(%[tmp])                        \n\t"
            "gssdrc1    %[ftmp2],   0x08(%[tmp])                        \n\t"
            : [ftmp0]"=&f"(ftmp[0]),        [ftmp1]"=&f"(ftmp[1]),
              [ftmp2]"=&f"(ftmp[2]),        [ftmp3]"=&f"(ftmp[3]),
              [ftmp4]"=&f"(ftmp[4]),        [ftmp5]"=&f"(ftmp[5]),
              [ftmp6]"=&f"(ftmp[6]),        [ftmp7]"=&f"(ftmp[7]),
              [ftmp8]"=&f"(ftmp[8]),        [ftmp9]"=&f"(ftmp[9]),
              [tmp0]"=&r"(tmp0)
            : [tmp]"r"(tmp),                [src]"r"(src),
              [ff_pw_64]"f"(ff_pw_64),
              [filter0]"r"(filter[0]),      [filter1]"r"(filter[1]),
              [filter2]"r"(filter[2]),      [filter3]"r"(filter[3]),
              [filter4]"r"(filter[4]),      [filter5]"r"(filter[5])
            : "memory"
        );

        tmp += 16;
        src += srcstride;
    }
#else
    const uint8_t *filter = subpel_filters[mx - 1];
    const uint8_t *cm     = ff_crop_tab + MAX_NEG_CROP;
    int x, y;
    uint8_t tmp_array[560];
    uint8_t *tmp = tmp_array;

    src -= srcstride;

    for (y = 0; y < h + 3; y++) {
        for (x = 0; x < 16; x++)
            tmp[x] = FILTER_6TAP(src, filter, 1);
        tmp += 16;
        src += srcstride;
    }
#endif

    tmp    = tmp_array + 16;
    filter = subpel_filters[my - 1];

#if NOTOK // this is ok
    for (y = 0; y < h; y++) {
        /*
        dst[0] = cm[(filter[2] * tmp[0] - filter[1] * tmp[-16] + filter[3] * tmp[16] - filter[4] * tmp[32] + 64) >> 7];
        dst[1] = cm[(filter[2] * tmp[1] - filter[1] * tmp[-15] + filter[3] * tmp[17] - filter[4] * tmp[33] + 64) >> 7];
        dst[2] = cm[(filter[2] * tmp[2] - filter[1] * tmp[-14] + filter[3] * tmp[18] - filter[4] * tmp[34] + 64) >> 7];
        dst[3] = cm[(filter[2] * tmp[3] - filter[1] * tmp[-13] + filter[3] * tmp[19] - filter[4] * tmp[35] + 64) >> 7];
        dst[4] = cm[(filter[2] * tmp[4] - filter[1] * tmp[-12] + filter[3] * tmp[20] - filter[4] * tmp[36] + 64) >> 7];
        dst[5] = cm[(filter[2] * tmp[5] - filter[1] * tmp[-11] + filter[3] * tmp[21] - filter[4] * tmp[37] + 64) >> 7];
        dst[6] = cm[(filter[2] * tmp[6] - filter[1] * tmp[-10] + filter[3] * tmp[22] - filter[4] * tmp[38] + 64) >> 7];
        dst[7] = cm[(filter[2] * tmp[7] - filter[1] * tmp[ -9] + filter[3] * tmp[23] - filter[4] * tmp[39] + 64) >> 7];

        dst[ 8] = cm[(filter[2] * tmp[ 8] - filter[1] * tmp[-8] + filter[3] * tmp[24] - filter[4] * tmp[40] + 64) >> 7];
        dst[ 9] = cm[(filter[2] * tmp[ 9] - filter[1] * tmp[-7] + filter[3] * tmp[25] - filter[4] * tmp[41] + 64) >> 7];
        dst[10] = cm[(filter[2] * tmp[10] - filter[1] * tmp[-6] + filter[3] * tmp[26] - filter[4] * tmp[42] + 64) >> 7];
        dst[11] = cm[(filter[2] * tmp[11] - filter[1] * tmp[-5] + filter[3] * tmp[27] - filter[4] * tmp[43] + 64) >> 7];
        dst[12] = cm[(filter[2] * tmp[12] - filter[1] * tmp[-4] + filter[3] * tmp[28] - filter[4] * tmp[44] + 64) >> 7];
        dst[13] = cm[(filter[2] * tmp[13] - filter[1] * tmp[-3] + filter[3] * tmp[29] - filter[4] * tmp[45] + 64) >> 7];
        dst[14] = cm[(filter[2] * tmp[14] - filter[1] * tmp[-2] + filter[3] * tmp[30] - filter[4] * tmp[46] + 64) >> 7];
        dst[15] = cm[(filter[2] * tmp[15] - filter[1] * tmp[-1] + filter[3] * tmp[31] - filter[4] * tmp[47] + 64) >> 7];
        */
        __asm__ volatile (
            "xor        %[ftmp0],   %[ftmp0],       %[ftmp0]            \n\t"
            "gsldlc1    %[ftmp1],   0x07(%[src])                        \n\t"
            "gsldrc1    %[ftmp1],   0x00(%[src])                        \n\t"
            "gsldlc1    %[ftmp7],   0x0f(%[src])                        \n\t"
            "mtc1       %[filter2], %[ftmp4]                            \n\t"
            "gsldrc1    %[ftmp7],   0x08(%[src])                        \n\t"
            "pshufh     %[ftmp4],   %[ftmp4],       %[ftmp0]            \n\t"
            "punpcklbh  %[ftmp2],   %[ftmp1],       %[ftmp0]            \n\t"
            "punpckhbh  %[ftmp3],   %[ftmp1],       %[ftmp0]            \n\t"
            "pmullh     %[ftmp5],   %[ftmp2],       %[ftmp4]            \n\t"
            "pmullh     %[ftmp6],   %[ftmp3],       %[ftmp4]            \n\t"
            "punpcklbh  %[ftmp2],   %[ftmp7],       %[ftmp0]            \n\t"
            "punpckhbh  %[ftmp3],   %[ftmp7],       %[ftmp0]            \n\t"
            "pmullh     %[ftmp8],   %[ftmp2],       %[ftmp4]            \n\t"
            "pmullh     %[ftmp9],   %[ftmp3],       %[ftmp4]            \n\t"

            "gsldlc1    %[ftmp1],  -0x09(%[src])                        \n\t"
            "gsldrc1    %[ftmp1],  -0x10(%[src])                        \n\t"
            "gsldlc1    %[ftmp7],  -0x01(%[src])                        \n\t"
            "mtc1       %[filter1], %[ftmp4]                            \n\t"
            "gsldrc1    %[ftmp7],  -0x08(%[src])                        \n\t"
            "pshufh     %[ftmp4],   %[ftmp4],       %[ftmp0]            \n\t"
            "punpcklbh  %[ftmp2],   %[ftmp1],       %[ftmp0]            \n\t"
            "punpckhbh  %[ftmp3],   %[ftmp1],       %[ftmp0]            \n\t"
            "pmullh     %[ftmp2],   %[ftmp2],       %[ftmp4]            \n\t"
            "pmullh     %[ftmp3],   %[ftmp3],       %[ftmp4]            \n\t"
            "psubush    %[ftmp5],   %[ftmp5],       %[ftmp2]            \n\t"
            "psubush    %[ftmp6],   %[ftmp6],       %[ftmp3]            \n\t"
            "punpcklbh  %[ftmp2],   %[ftmp7],       %[ftmp0]            \n\t"
            "punpckhbh  %[ftmp3],   %[ftmp7],       %[ftmp0]            \n\t"
            "pmullh     %[ftmp2],   %[ftmp2],       %[ftmp4]            \n\t"
            "pmullh     %[ftmp3],   %[ftmp3],       %[ftmp4]            \n\t"
            "psubush    %[ftmp8],   %[ftmp8],       %[ftmp2]            \n\t"
            "psubush    %[ftmp9],   %[ftmp9],       %[ftmp3]            \n\t"

            "gsldlc1    %[ftmp1],   0x08(%[src])                        \n\t"
            "gsldrc1    %[ftmp1],   0x01(%[src])                        \n\t"
            "gsldlc1    %[ftmp7],   0x0d(%[src])                        \n\t"
            "mtc1       %[filter3], %[ftmp4]                            \n\t"
            "gsldrc1    %[ftmp7],   0x06(%[src])                        \n\t"
            "pshufh     %[ftmp4],   %[ftmp4],       %[ftmp0]            \n\t"
            "punpcklbh  %[ftmp2],   %[ftmp1],       %[ftmp0]            \n\t"
            "punpckhbh  %[ftmp3],   %[ftmp1],       %[ftmp0]            \n\t"
            "pmullh     %[ftmp2],   %[ftmp2],       %[ftmp4]            \n\t"
            "pmullh     %[ftmp3],   %[ftmp3],       %[ftmp4]            \n\t"
            "paddush    %[ftmp5],   %[ftmp5],       %[ftmp2]            \n\t"
            "paddush    %[ftmp6],   %[ftmp6],       %[ftmp3]            \n\t"
            "punpcklbh  %[ftmp2],   %[ftmp7],       %[ftmp0]            \n\t"
            "punpckhbh  %[ftmp3],   %[ftmp7],       %[ftmp0]            \n\t"
            "pmullh     %[ftmp2],   %[ftmp2],       %[ftmp4]            \n\t"
            "pmullh     %[ftmp3],   %[ftmp3],       %[ftmp4]            \n\t"
            "paddush    %[ftmp8],   %[ftmp8],       %[ftmp2]            \n\t"
            "paddush    %[ftmp9],   %[ftmp9],       %[ftmp3]            \n\t"

            "gsldlc1    %[ftmp1],   0x09(%[src])                        \n\t"
            "gsldrc1    %[ftmp1],   0x02(%[src])                        \n\t"
            "gsldlc1    %[ftmp7],   0x11(%[src])                        \n\t"
            "mtc1       %[filter4], %[ftmp4]                            \n\t"
            "gsldrc1    %[ftmp7],   0x0a(%[src])                        \n\t"
            "pshufh     %[ftmp4],   %[ftmp4],       %[ftmp0]            \n\t"
            "punpcklbh  %[ftmp2],   %[ftmp1],       %[ftmp0]            \n\t"
            "punpckhbh  %[ftmp3],   %[ftmp1],       %[ftmp0]            \n\t"
            "pmullh     %[ftmp2],   %[ftmp2],       %[ftmp4]            \n\t"
            "pmullh     %[ftmp3],   %[ftmp3],       %[ftmp4]            \n\t"
            "psubush    %[ftmp5],   %[ftmp5],       %[ftmp2]            \n\t"
            "psubush    %[ftmp6],   %[ftmp6],       %[ftmp3]            \n\t"
            "punpcklbh  %[ftmp2],   %[ftmp7],       %[ftmp0]            \n\t"
            "punpckhbh  %[ftmp3],   %[ftmp7],       %[ftmp0]            \n\t"
            "pmullh     %[ftmp2],   %[ftmp2],       %[ftmp4]            \n\t"
            "pmullh     %[ftmp3],   %[ftmp3],       %[ftmp4]            \n\t"
            "psubush    %[ftmp8],   %[ftmp8],       %[ftmp2]            \n\t"
            "psubush    %[ftmp9],   %[ftmp9],       %[ftmp3]            \n\t"

            "li         %[tmp0],    0x07                                \n\t"
            "paddush    %[ftmp5],   %[ftmp5],       %[ff_pw_64]         \n\t"
            "paddush    %[ftmp6],   %[ftmp6],       %[ff_pw_64]         \n\t"
            "mtc1       %[tmp0],    %[ftmp4]                            \n\t"
            "psrlh      %[ftmp5],   %[ftmp5],       %[ftmp4]            \n\t"
            "psrlh      %[ftmp6],   %[ftmp6],       %[ftmp4]            \n\t"

            "packushb   %[ftmp1],   %[ftmp5],       %[ftmp6]            \n\t"
            "gssdlc1    %[ftmp1],   0x07(%[dst])                        \n\t"
            "gssdrc1    %[ftmp1],   0x00(%[dst])                        \n\t"
            "gssdlc1    %[ftmp1],   0x0f(%[dst])                        \n\t"
            "gssdrc1    %[ftmp1],   0x08(%[dst])                        \n\t"
            : [ftmp0]"=&f"(ftmp[0]),        [ftmp1]"=&f"(ftmp[1]),
              [ftmp2]"=&f"(ftmp[2]),        [ftmp3]"=&f"(ftmp[3]),
              [ftmp4]"=&f"(ftmp[4]),        [ftmp5]"=&f"(ftmp[5]),
              [ftmp6]"=&f"(ftmp[6]),        [ftmp7]"=&f"(ftmp[7]),
              [ftmp8]"=&f"(ftmp[8]),        [ftmp9]"=&f"(ftmp[9]),
              [tmp0]"=&r"(tmp0)
            : [dst]"r"(dst),                [src]"r"(src),
              [ff_pw_64]"f"(ff_pw_64),
              [filter1]"r"(filter[1]),      [filter2]"r"(filter[2]),
              [filter3]"r"(filter[3]),      [filter4]"r"(filter[4])
            : "memory"
        );

        dst += dststride;
        tmp += 16;
    }
#else
    for (y = 0; y < h; y++) {
        for (x = 0; x < 16; x++)
            dst[x] = FILTER_4TAP(tmp, filter, 16);
        dst += dststride;
        tmp += 16;
    }
#endif
}

void ff_put_vp8_epel8_h6v4_mmi(uint8_t *dst, ptrdiff_t dststride, uint8_t *src,
        ptrdiff_t srcstride, int h, int mx, int my)
{
#if NOTOK //FIXME fate-vp8-test-vector-006
    const uint8_t *filter = subpel_filters[mx - 1];
    int y;
    uint8_t tmp_array[152];
    uint8_t *tmp = tmp_array;
    double ftmp[7];
    uint64_t tmp0;

    src -= srcstride;

    for (y = 0; y < h + 3; y++) {
        /*
        tmp[0] = cm[(filter[2]*src[0] - filter[1]*src[-1] + filter[0]*src[-2] + filter[3]*src[1] - filter[4]*src[2] + filter[5]*src[ 3] + 64) >> 7];
        tmp[1] = cm[(filter[2]*src[1] - filter[1]*src[ 0] + filter[0]*src[-1] + filter[3]*src[2] - filter[4]*src[3] + filter[5]*src[ 4] + 64) >> 7];
        tmp[2] = cm[(filter[2]*src[2] - filter[1]*src[ 1] + filter[0]*src[ 0] + filter[3]*src[3] - filter[4]*src[4] + filter[5]*src[ 5] + 64) >> 7];
        tmp[3] = cm[(filter[2]*src[3] - filter[1]*src[ 2] + filter[0]*src[ 1] + filter[3]*src[4] - filter[4]*src[5] + filter[5]*src[ 6] + 64) >> 7];
        tmp[4] = cm[(filter[2]*src[4] - filter[1]*src[ 3] + filter[0]*src[ 2] + filter[3]*src[5] - filter[4]*src[6] + filter[5]*src[ 7] + 64) >> 7];
        tmp[5] = cm[(filter[2]*src[5] - filter[1]*src[ 4] + filter[0]*src[ 3] + filter[3]*src[6] - filter[4]*src[7] + filter[5]*src[ 8] + 64) >> 7];
        tmp[6] = cm[(filter[2]*src[6] - filter[1]*src[ 5] + filter[0]*src[ 4] + filter[3]*src[7] - filter[4]*src[8] + filter[5]*src[ 9] + 64) >> 7];
        tmp[7] = cm[(filter[2]*src[7] - filter[1]*src[ 6] + filter[0]*src[ 5] + filter[3]*src[8] - filter[4]*src[9] + filter[5]*src[10] + 64) >> 7];
        */
        __asm__ volatile (
            "xor        %[ftmp0],   %[ftmp0],       %[ftmp0]            \n\t"
            "gsldlc1    %[ftmp1],   0x07(%[src])                        \n\t"
            "mtc1       %[filter2], %[ftmp4]                            \n\t"
            "gsldrc1    %[ftmp1],   0x00(%[src])                        \n\t"
            "pshufh     %[ftmp4],   %[ftmp4],       %[ftmp0]            \n\t"
            "punpcklbh  %[ftmp2],   %[ftmp1],       %[ftmp0]            \n\t"
            "punpckhbh  %[ftmp3],   %[ftmp1],       %[ftmp0]            \n\t"
            "pmullh     %[ftmp5],   %[ftmp2],       %[ftmp4]            \n\t"
            "pmullh     %[ftmp6],   %[ftmp3],       %[ftmp4]            \n\t"

            "gsldlc1    %[ftmp1],   0x06(%[src])                        \n\t"
            "mtc1       %[filter1], %[ftmp4]                            \n\t"
            "gsldrc1    %[ftmp1],  -0x01(%[src])                        \n\t"
            "pshufh     %[ftmp4],   %[ftmp4],       %[ftmp0]            \n\t"
            "punpcklbh  %[ftmp2],   %[ftmp1],       %[ftmp0]            \n\t"
            "punpckhbh  %[ftmp3],   %[ftmp1],       %[ftmp0]            \n\t"
            "pmullh     %[ftmp2],   %[ftmp2],       %[ftmp4]            \n\t"
            "pmullh     %[ftmp3],   %[ftmp3],       %[ftmp4]            \n\t"
            "psubush    %[ftmp5],   %[ftmp5],       %[ftmp2]            \n\t"
            "psubush    %[ftmp6],   %[ftmp6],       %[ftmp3]            \n\t"

            "gsldlc1    %[ftmp1],   0x05(%[src])                        \n\t"
            "mtc1       %[filter0], %[ftmp4]                            \n\t"
            "gsldrc1    %[ftmp1],  -0x02(%[src])                        \n\t"
            "pshufh     %[ftmp4],   %[ftmp4],       %[ftmp0]            \n\t"
            "punpcklbh  %[ftmp2],   %[ftmp1],       %[ftmp0]            \n\t"
            "punpckhbh  %[ftmp3],   %[ftmp1],       %[ftmp0]            \n\t"
            "pmullh     %[ftmp2],   %[ftmp2],       %[ftmp4]            \n\t"
            "pmullh     %[ftmp3],   %[ftmp3],       %[ftmp4]            \n\t"
            "paddush    %[ftmp5],   %[ftmp5],       %[ftmp2]            \n\t"
            "paddush    %[ftmp6],   %[ftmp6],       %[ftmp3]            \n\t"

            "gsldlc1    %[ftmp1],   0x08(%[src])                        \n\t"
            "mtc1       %[filter3], %[ftmp4]                            \n\t"
            "gsldrc1    %[ftmp1],   0x01(%[src])                        \n\t"
            "pshufh     %[ftmp4],   %[ftmp4],       %[ftmp0]            \n\t"
            "punpcklbh  %[ftmp2],   %[ftmp1],       %[ftmp0]            \n\t"
            "punpckhbh  %[ftmp3],   %[ftmp1],       %[ftmp0]            \n\t"
            "pmullh     %[ftmp2],   %[ftmp2],       %[ftmp4]            \n\t"
            "pmullh     %[ftmp3],   %[ftmp3],       %[ftmp4]            \n\t"
            "paddush    %[ftmp5],   %[ftmp5],       %[ftmp2]            \n\t"
            "paddush    %[ftmp6],   %[ftmp6],       %[ftmp3]            \n\t"

            "gsldlc1    %[ftmp1],   0x09(%[src])                        \n\t"
            "mtc1       %[filter4], %[ftmp4]                            \n\t"
            "gsldrc1    %[ftmp1],   0x02(%[src])                        \n\t"
            "pshufh     %[ftmp4],   %[ftmp4],       %[ftmp0]            \n\t"
            "punpcklbh  %[ftmp2],   %[ftmp1],       %[ftmp0]            \n\t"
            "punpckhbh  %[ftmp3],   %[ftmp1],       %[ftmp0]            \n\t"
            "pmullh     %[ftmp2],   %[ftmp2],       %[ftmp4]            \n\t"
            "pmullh     %[ftmp3],   %[ftmp3],       %[ftmp4]            \n\t"
            "psubush    %[ftmp5],   %[ftmp5],       %[ftmp2]            \n\t"
            "psubush    %[ftmp6],   %[ftmp6],       %[ftmp3]            \n\t"

            "gsldlc1    %[ftmp1],   0x0a(%[src])                        \n\t"
            "mtc1       %[filter5], %[ftmp4]                            \n\t"
            "gsldrc1    %[ftmp1],   0x03(%[src])                        \n\t"
            "pshufh     %[ftmp4],   %[ftmp4],       %[ftmp0]            \n\t"
            "punpcklbh  %[ftmp2],   %[ftmp1],       %[ftmp0]            \n\t"
            "punpckhbh  %[ftmp3],   %[ftmp1],       %[ftmp0]            \n\t"
            "pmullh     %[ftmp2],   %[ftmp2],       %[ftmp4]            \n\t"
            "pmullh     %[ftmp3],   %[ftmp3],       %[ftmp4]            \n\t"
            "paddush    %[ftmp5],   %[ftmp5],       %[ftmp2]            \n\t"
            "paddush    %[ftmp6],   %[ftmp6],       %[ftmp3]            \n\t"

            "li         %[tmp0],    0x07                                \n\t"
            "paddush    %[ftmp5],   %[ftmp5],       %[ff_pw_64]         \n\t"
            "paddush    %[ftmp6],   %[ftmp6],       %[ff_pw_64]         \n\t"
            "mtc1       %[tmp0],    %[ftmp4]                            \n\t"
            "psrlh      %[ftmp5],   %[ftmp5],       %[ftmp4]            \n\t"
            "psrlh      %[ftmp6],   %[ftmp6],       %[ftmp4]            \n\t"

            "packushb   %[ftmp1],   %[ftmp5],       %[ftmp6]            \n\t"
            "gssdlc1    %[ftmp1],   0x07(%[tmp])                        \n\t"
            "gssdrc1    %[ftmp1],   0x00(%[tmp])                        \n\t"
            : [ftmp0]"=&f"(ftmp[0]),        [ftmp1]"=&f"(ftmp[1]),
              [ftmp2]"=&f"(ftmp[2]),        [ftmp3]"=&f"(ftmp[3]),
              [ftmp4]"=&f"(ftmp[4]),        [ftmp5]"=&f"(ftmp[5]),
              [ftmp6]"=&f"(ftmp[6]),
              [tmp0]"=&r"(tmp0)
            : [tmp]"r"(tmp),                [src]"r"(src),
              [ff_pw_64]"f"(ff_pw_64),
              [filter0]"r"(filter[0]),      [filter1]"r"(filter[1]),
              [filter2]"r"(filter[2]),      [filter3]"r"(filter[3]),
              [filter4]"r"(filter[4]),      [filter5]"r"(filter[5])
            : "memory"
        );

        tmp += 8;
        src += srcstride;
    }
#else
    const uint8_t *filter = subpel_filters[mx - 1];
    const uint8_t *cm     = ff_crop_tab + MAX_NEG_CROP;
    int x, y;
    uint8_t tmp_array[152];
    uint8_t *tmp = tmp_array;

    src -= srcstride;

    for (y = 0; y < h + 3; y++) {
        for (x = 0; x < 8; x++)
            tmp[x] = FILTER_6TAP(src, filter, 1);
        tmp += 8;
        src += srcstride;
    }
#endif

    tmp    = tmp_array + 8;
    filter = subpel_filters[my - 1];

#if NOTOK //FIXME fate-vp8-test-vector-006
    for (y = 0; y < h; y++) {
        /*
        dst[0] = cm[(filter[2] * tmp[0] - filter[1] * tmp[-8] + filter[3] * tmp[ 8] - filter[4] * tmp[16] + 64) >> 7];
        dst[1] = cm[(filter[2] * tmp[1] - filter[1] * tmp[-7] + filter[3] * tmp[ 9] - filter[4] * tmp[17] + 64) >> 7];
        dst[2] = cm[(filter[2] * tmp[2] - filter[1] * tmp[-6] + filter[3] * tmp[10] - filter[4] * tmp[18] + 64) >> 7];
        dst[3] = cm[(filter[2] * tmp[3] - filter[1] * tmp[-5] + filter[3] * tmp[11] - filter[4] * tmp[19] + 64) >> 7];
        dst[4] = cm[(filter[2] * tmp[4] - filter[1] * tmp[-4] + filter[3] * tmp[12] - filter[4] * tmp[20] + 64) >> 7];
        dst[5] = cm[(filter[2] * tmp[5] - filter[1] * tmp[-3] + filter[3] * tmp[13] - filter[4] * tmp[21] + 64) >> 7];
        dst[6] = cm[(filter[2] * tmp[6] - filter[1] * tmp[-2] + filter[3] * tmp[14] - filter[4] * tmp[22] + 64) >> 7];
        dst[7] = cm[(filter[2] * tmp[7] - filter[1] * tmp[-1] + filter[3] * tmp[15] - filter[4] * tmp[23] + 64) >> 7];
        */
        __asm__ volatile (
            "xor        %[ftmp0],   %[ftmp0],       %[ftmp0]            \n\t"
            "gsldlc1    %[ftmp1],   0x07(%[tmp])                        \n\t"
            "mtc1       %[filter2], %[ftmp4]                            \n\t"
            "gsldrc1    %[ftmp1],   0x00(%[tmp])                        \n\t"
            "pshufh     %[ftmp4],   %[ftmp4],       %[ftmp0]            \n\t"
            "punpcklbh  %[ftmp2],   %[ftmp1],       %[ftmp0]            \n\t"
            "punpckhbh  %[ftmp3],   %[ftmp1],       %[ftmp0]            \n\t"
            "pmullh     %[ftmp5],   %[ftmp2],       %[ftmp4]            \n\t"
            "pmullh     %[ftmp6],   %[ftmp3],       %[ftmp4]            \n\t"

            "gsldlc1    %[ftmp1],  -0x01(%[tmp])                        \n\t"
            "mtc1       %[filter1], %[ftmp4]                            \n\t"
            "gsldrc1    %[ftmp1],  -0x08(%[tmp])                        \n\t"
            "pshufh     %[ftmp4],   %[ftmp4],       %[ftmp0]            \n\t"
            "punpcklbh  %[ftmp2],   %[ftmp1],       %[ftmp0]            \n\t"
            "punpckhbh  %[ftmp3],   %[ftmp1],       %[ftmp0]            \n\t"
            "pmullh     %[ftmp2],   %[ftmp2],       %[ftmp4]            \n\t"
            "pmullh     %[ftmp3],   %[ftmp3],       %[ftmp4]            \n\t"
            "psubush    %[ftmp5],   %[ftmp5],       %[ftmp2]            \n\t"
            "psubush    %[ftmp6],   %[ftmp6],       %[ftmp3]            \n\t"

            "gsldlc1    %[ftmp1],   0x0f(%[tmp])                        \n\t"
            "mtc1       %[filter3], %[ftmp4]                            \n\t"
            "gsldrc1    %[ftmp1],   0x08(%[tmp])                        \n\t"
            "pshufh     %[ftmp4],   %[ftmp4],       %[ftmp0]            \n\t"
            "punpcklbh  %[ftmp2],   %[ftmp1],       %[ftmp0]            \n\t"
            "punpckhbh  %[ftmp3],   %[ftmp1],       %[ftmp0]            \n\t"
            "pmullh     %[ftmp2],   %[ftmp2],       %[ftmp4]            \n\t"
            "pmullh     %[ftmp3],   %[ftmp3],       %[ftmp4]            \n\t"
            "paddush    %[ftmp5],   %[ftmp5],       %[ftmp2]            \n\t"
            "paddush    %[ftmp6],   %[ftmp6],       %[ftmp3]            \n\t"

            "gsldlc1    %[ftmp1],   0x17(%[tmp])                        \n\t"
            "mtc1       %[filter4], %[ftmp4]                            \n\t"
            "gsldrc1    %[ftmp1],   0x10(%[tmp])                        \n\t"
            "pshufh     %[ftmp4],   %[ftmp4],       %[ftmp0]            \n\t"
            "punpcklbh  %[ftmp2],   %[ftmp1],       %[ftmp0]            \n\t"
            "punpckhbh  %[ftmp3],   %[ftmp1],       %[ftmp0]            \n\t"
            "pmullh     %[ftmp2],   %[ftmp2],       %[ftmp4]            \n\t"
            "pmullh     %[ftmp3],   %[ftmp3],       %[ftmp4]            \n\t"
            "psubush    %[ftmp5],   %[ftmp5],       %[ftmp2]            \n\t"
            "psubush    %[ftmp6],   %[ftmp6],       %[ftmp3]            \n\t"

            "li         %[tmp0],    0x07                                \n\t"
            "paddush    %[ftmp5],   %[ftmp5],       %[ff_pw_64]         \n\t"
            "paddush    %[ftmp6],   %[ftmp6],       %[ff_pw_64]         \n\t"
            "mtc1       %[tmp0],    %[ftmp4]                            \n\t"
            "psrlh      %[ftmp5],   %[ftmp5],       %[ftmp4]            \n\t"
            "psrlh      %[ftmp6],   %[ftmp6],       %[ftmp4]            \n\t"

            "packushb   %[ftmp1],   %[ftmp5],       %[ftmp6]            \n\t"
            "gssdlc1    %[ftmp1],   0x07(%[dst])                        \n\t"
            "gssdrc1    %[ftmp1],   0x00(%[dst])                        \n\t"
            : [ftmp0]"=&f"(ftmp[0]),        [ftmp1]"=&f"(ftmp[1]),
              [ftmp2]"=&f"(ftmp[2]),        [ftmp3]"=&f"(ftmp[3]),
              [ftmp4]"=&f"(ftmp[4]),        [ftmp5]"=&f"(ftmp[5]),
              [ftmp6]"=&f"(ftmp[6]),
              [tmp0]"=&r"(tmp0)
            : [dst]"r"(dst),                [tmp]"r"(tmp),
              [ff_pw_64]"f"(ff_pw_64),
              [filter1]"r"(filter[1]),      [filter2]"r"(filter[2]),
              [filter3]"r"(filter[3]),      [filter4]"r"(filter[4])
            : "memory"
        );

        dst += dststride;
        tmp += 8;
    }
#else
    for (y = 0; y < h; y++) {
        for (x = 0; x < 8; x++)
            dst[x] = FILTER_4TAP(tmp, filter, 8);
        dst += dststride;
        tmp += 8;
    }
#endif
}

void ff_put_vp8_epel4_h6v4_mmi(uint8_t *dst, ptrdiff_t dststride, uint8_t *src,
        ptrdiff_t srcstride, int h, int mx, int my)
{
#if NOTOK //FIXME fate-vp8-test-vector-006
    const uint8_t *filter = subpel_filters[mx - 1];
    int y;
    uint8_t tmp_array[44];
    uint8_t *tmp = tmp_array;
    double ftmp[7];
    uint64_t tmp0;

    src -= srcstride;

    for (y = 0; y < h + 3; y++) {
        /*
        tmp[0] = cm[(filter[2]*src[0] - filter[1]*src[-1] + filter[0]*src[-2] + filter[3]*src[1] - filter[4]*src[2] + filter[5]*src[3] + 64) >> 7];
        tmp[1] = cm[(filter[2]*src[1] - filter[1]*src[ 0] + filter[0]*src[-1] + filter[3]*src[2] - filter[4]*src[3] + filter[5]*src[4] + 64) >> 7];
        tmp[2] = cm[(filter[2]*src[2] - filter[1]*src[ 1] + filter[0]*src[ 0] + filter[3]*src[3] - filter[4]*src[4] + filter[5]*src[5] + 64) >> 7];
        tmp[3] = cm[(filter[2]*src[3] - filter[1]*src[ 2] + filter[0]*src[ 1] + filter[3]*src[4] - filter[4]*src[5] + filter[5]*src[6] + 64) >> 7];
        */
        __asm__ volatile (
            "xor        %[ftmp0],   %[ftmp0],       %[ftmp0]            \n\t"
            "gslwlc1    %[ftmp1],   0x03(%[src])                        \n\t"
            "mtc1       %[filter2], %[ftmp3]                            \n\t"
            "gslwrc1    %[ftmp1],   0x00(%[src])                        \n\t"
            "pshufh     %[ftmp3],   %[ftmp3],       %[ftmp0]            \n\t"
            "punpcklbh  %[ftmp2],   %[ftmp1],       %[ftmp0]            \n\t"
            "pmullh     %[ftmp4],   %[ftmp2],       %[ftmp3]            \n\t"

            "gslwlc1    %[ftmp1],   0x02(%[src])                        \n\t"
            "mtc1       %[filter1], %[ftmp3]                            \n\t"
            "gslwrc1    %[ftmp1],  -0x01(%[src])                        \n\t"
            "pshufh     %[ftmp3],   %[ftmp3],       %[ftmp0]            \n\t"
            "punpcklbh  %[ftmp2],   %[ftmp1],       %[ftmp0]            \n\t"
            "pmullh     %[ftmp2],   %[ftmp2],       %[ftmp3]            \n\t"
            "psubush    %[ftmp4],   %[ftmp4],       %[ftmp2]            \n\t"

            "gslwlc1    %[ftmp1],   0x01(%[src])                        \n\t"
            "mtc1       %[filter0], %[ftmp3]                            \n\t"
            "gslwrc1    %[ftmp1],  -0x02(%[src])                        \n\t"
            "pshufh     %[ftmp3],   %[ftmp3],       %[ftmp0]            \n\t"
            "punpcklbh  %[ftmp2],   %[ftmp1],       %[ftmp0]            \n\t"
            "pmullh     %[ftmp2],   %[ftmp2],       %[ftmp3]            \n\t"
            "paddush    %[ftmp4],   %[ftmp4],       %[ftmp2]            \n\t"

            "gslwlc1    %[ftmp1],   0x04(%[src])                        \n\t"
            "mtc1       %[filter3], %[ftmp3]                            \n\t"
            "gslwrc1    %[ftmp1],   0x01(%[src])                        \n\t"
            "pshufh     %[ftmp3],   %[ftmp3],       %[ftmp0]            \n\t"
            "punpcklbh  %[ftmp2],   %[ftmp1],       %[ftmp0]            \n\t"
            "pmullh     %[ftmp2],   %[ftmp2],       %[ftmp3]            \n\t"
            "paddush    %[ftmp4],   %[ftmp4],       %[ftmp2]            \n\t"

            "gslwlc1    %[ftmp1],   0x05(%[src])                        \n\t"
            "mtc1       %[filter4], %[ftmp3]                            \n\t"
            "gslwrc1    %[ftmp1],   0x02(%[src])                        \n\t"
            "pshufh     %[ftmp3],   %[ftmp3],       %[ftmp0]            \n\t"
            "punpcklbh  %[ftmp2],   %[ftmp1],       %[ftmp0]            \n\t"
            "pmullh     %[ftmp2],   %[ftmp2],       %[ftmp3]            \n\t"
            "psubush    %[ftmp4],   %[ftmp4],       %[ftmp2]            \n\t"

            "gslwlc1    %[ftmp1],   0x06(%[src])                        \n\t"
            "mtc1       %[filter5], %[ftmp3]                            \n\t"
            "gslwrc1    %[ftmp1],   0x03(%[src])                        \n\t"
            "pshufh     %[ftmp3],   %[ftmp3],       %[ftmp0]            \n\t"
            "punpcklbh  %[ftmp2],   %[ftmp1],       %[ftmp0]            \n\t"
            "pmullh     %[ftmp2],   %[ftmp2],       %[ftmp3]            \n\t"
            "paddush    %[ftmp4],   %[ftmp4],       %[ftmp2]            \n\t"

            "li         %[tmp0],    0x07                                \n\t"
            "paddush    %[ftmp4],   %[ftmp4],       %[ff_pw_64]         \n\t"
            "mtc1       %[tmp0],    %[ftmp3]                            \n\t"
            "psrlh      %[ftmp4],   %[ftmp4],       %[ftmp3]            \n\t"

            "packushb   %[ftmp1],   %[ftmp4],       %[ftmp0]            \n\t"
            "gsswlc1    %[ftmp1],   0x03(%[tmp])                        \n\t"
            "gsswrc1    %[ftmp1],   0x00(%[tmp])                        \n\t"
            : [ftmp0]"=&f"(ftmp[0]),        [ftmp1]"=&f"(ftmp[1]),
              [ftmp2]"=&f"(ftmp[2]),        [ftmp3]"=&f"(ftmp[3]),
              [ftmp4]"=&f"(ftmp[4]),
              [tmp0]"=&r"(tmp0)
            : [tmp]"r"(tmp),                [src]"r"(src),
              [ff_pw_64]"f"(ff_pw_64),
              [filter0]"r"(filter[0]),      [filter1]"r"(filter[1]),
              [filter2]"r"(filter[2]),      [filter3]"r"(filter[3]),
              [filter4]"r"(filter[4]),      [filter5]"r"(filter[5])
            : "memory"
        );

        tmp += 4;
        src += srcstride;
    }
#else
    const uint8_t *filter = subpel_filters[mx - 1];
    const uint8_t *cm     = ff_crop_tab + MAX_NEG_CROP;
    int x, y;
    uint8_t tmp_array[44];
    uint8_t *tmp = tmp_array;

    src -= srcstride;

    for (y = 0; y < h + 3; y++) {
        for (x = 0; x < 4; x++)
            tmp[x] = FILTER_6TAP(src, filter, 1);
        tmp += 4;
        src += srcstride;
    }
#endif

    tmp    = tmp_array + 4;
    filter = subpel_filters[my - 1];

#if NOTOK //FIXME fate-vp8-test-vector-006
    for (y = 0; y < h; y++) {
        /*
        dst[0] = cm[(filter[2] * tmp[0] - filter[1] * tmp[-4] + filter[3] * tmp[4] - filter[4] * tmp[ 8] + 64) >> 7];
        dst[1] = cm[(filter[2] * tmp[1] - filter[1] * tmp[-3] + filter[3] * tmp[5] - filter[4] * tmp[ 9] + 64) >> 7];
        dst[2] = cm[(filter[2] * tmp[2] - filter[1] * tmp[-2] + filter[3] * tmp[6] - filter[4] * tmp[10] + 64) >> 7];
        dst[3] = cm[(filter[2] * tmp[3] - filter[1] * tmp[-1] + filter[3] * tmp[7] - filter[4] * tmp[11] + 64) >> 7];
        */
        __asm__ volatile (
            "xor        %[ftmp0],   %[ftmp0],       %[ftmp0]            \n\t"
            "gslwlc1    %[ftmp1],   0x03(%[tmp])                        \n\t"
            "mtc1       %[filter2], %[ftmp3]                            \n\t"
            "gslwrc1    %[ftmp1],   0x00(%[tmp])                        \n\t"
            "pshufh     %[ftmp3],   %[ftmp3],       %[ftmp0]            \n\t"
            "punpcklbh  %[ftmp2],   %[ftmp1],       %[ftmp0]            \n\t"
            "pmullh     %[ftmp4],   %[ftmp2],       %[ftmp3]            \n\t"

            "gslwlc1    %[ftmp1],  -0x01(%[tmp])                        \n\t"
            "mtc1       %[filter1], %[ftmp3]                            \n\t"
            "gslwrc1    %[ftmp1],  -0x04(%[tmp])                        \n\t"
            "pshufh     %[ftmp3],   %[ftmp3],       %[ftmp0]            \n\t"
            "punpcklbh  %[ftmp2],   %[ftmp1],       %[ftmp0]            \n\t"
            "pmullh     %[ftmp2],   %[ftmp2],       %[ftmp3]            \n\t"
            "psubush    %[ftmp4],   %[ftmp4],       %[ftmp2]            \n\t"

            "gslwlc1    %[ftmp1],   0x07(%[tmp])                        \n\t"
            "mtc1       %[filter3], %[ftmp3]                            \n\t"
            "gslwrc1    %[ftmp1],   0x04(%[tmp])                        \n\t"
            "pshufh     %[ftmp3],   %[ftmp3],       %[ftmp0]            \n\t"
            "punpcklbh  %[ftmp2],   %[ftmp1],       %[ftmp0]            \n\t"
            "pmullh     %[ftmp2],   %[ftmp2],       %[ftmp3]            \n\t"
            "paddush    %[ftmp4],   %[ftmp4],       %[ftmp2]            \n\t"

            "gslwlc1    %[ftmp1],   0x0b(%[tmp])                        \n\t"
            "mtc1       %[filter4], %[ftmp3]                            \n\t"
            "gslwrc1    %[ftmp1],   0x08(%[tmp])                        \n\t"
            "pshufh     %[ftmp3],   %[ftmp3],       %[ftmp0]            \n\t"
            "punpcklbh  %[ftmp2],   %[ftmp1],       %[ftmp0]            \n\t"
            "pmullh     %[ftmp2],   %[ftmp2],       %[ftmp3]            \n\t"
            "psubush    %[ftmp4],   %[ftmp4],       %[ftmp2]            \n\t"

            "li         %[tmp0],    0x07                                \n\t"
            "paddush    %[ftmp4],   %[ftmp4],       %[ff_pw_64]         \n\t"
            "mtc1       %[tmp0],    %[ftmp3]                            \n\t"
            "psrlh      %[ftmp4],   %[ftmp4],       %[ftmp3]            \n\t"

            "packushb   %[ftmp1],   %[ftmp4],       %[ftmp0]            \n\t"
            "gsswlc1    %[ftmp1],   0x03(%[dst])                        \n\t"
            "gsswrc1    %[ftmp1],   0x00(%[dst])                        \n\t"
            : [ftmp0]"=&f"(ftmp[0]),        [ftmp1]"=&f"(ftmp[1]),
              [ftmp2]"=&f"(ftmp[2]),        [ftmp3]"=&f"(ftmp[3]),
              [ftmp4]"=&f"(ftmp[4]),        [ftmp5]"=&f"(ftmp[5]),
              [ftmp6]"=&f"(ftmp[6]),
              [tmp0]"=&r"(tmp0)
            : [dst]"r"(dst),                [tmp]"r"(tmp),
              [ff_pw_64]"f"(ff_pw_64),
              [filter1]"r"(filter[1]),      [filter2]"r"(filter[2]),
              [filter3]"r"(filter[3]),      [filter4]"r"(filter[4])
            : "memory"
        );

        dst += dststride;
        tmp += 4;
    }
#else
    for (y = 0; y < h; y++) {
        for (x = 0; x < 4; x++)
            dst[x] = FILTER_4TAP(tmp, filter, 4);
        dst += dststride;
        tmp += 4;
    }
#endif
}

void ff_put_vp8_epel16_h6v6_mmi(uint8_t *dst, ptrdiff_t dststride, uint8_t *src,
        ptrdiff_t srcstride, int h, int mx, int my)
{
#if NOTOK //FIXME fate-vp8-test-vector-006
    const uint8_t *filter = subpel_filters[mx - 1];
    int y;
    uint8_t tmp_array[592];
    uint8_t *tmp = tmp_array;
    double ftmp[10];
    uint64_t tmp0;

    src -= 2 * srcstride;

    for (y = 0; y < h + 5; y++) {
        /*
        tmp[0] = cm[(filter[2]*src[0] - filter[1]*src[-1] + filter[0]*src[-2] + filter[3]*src[1] - filter[4]*src[2] + filter[5]*src[ 3] + 64) >> 7];
        tmp[1] = cm[(filter[2]*src[1] - filter[1]*src[ 0] + filter[0]*src[-1] + filter[3]*src[2] - filter[4]*src[3] + filter[5]*src[ 4] + 64) >> 7];
        tmp[2] = cm[(filter[2]*src[2] - filter[1]*src[ 1] + filter[0]*src[ 0] + filter[3]*src[3] - filter[4]*src[4] + filter[5]*src[ 5] + 64) >> 7];
        tmp[3] = cm[(filter[2]*src[3] - filter[1]*src[ 2] + filter[0]*src[ 1] + filter[3]*src[4] - filter[4]*src[5] + filter[5]*src[ 6] + 64) >> 7];
        tmp[4] = cm[(filter[2]*src[4] - filter[1]*src[ 3] + filter[0]*src[ 2] + filter[3]*src[5] - filter[4]*src[6] + filter[5]*src[ 7] + 64) >> 7];
        tmp[5] = cm[(filter[2]*src[5] - filter[1]*src[ 4] + filter[0]*src[ 3] + filter[3]*src[6] - filter[4]*src[7] + filter[5]*src[ 8] + 64) >> 7];
        tmp[6] = cm[(filter[2]*src[6] - filter[1]*src[ 5] + filter[0]*src[ 4] + filter[3]*src[7] - filter[4]*src[8] + filter[5]*src[ 9] + 64) >> 7];
        tmp[7] = cm[(filter[2]*src[7] - filter[1]*src[ 6] + filter[0]*src[ 5] + filter[3]*src[8] - filter[4]*src[9] + filter[5]*src[10] + 64) >> 7];

        tmp[ 8] = cm[(filter[2]*src[ 8] - filter[1]*src[ 7] + filter[0]*src[ 6] + filter[3]*src[ 9] - filter[4]*src[10] + filter[5]*src[11] + 64) >> 7];
        tmp[ 9] = cm[(filter[2]*src[ 9] - filter[1]*src[ 8] + filter[0]*src[ 7] + filter[3]*src[10] - filter[4]*src[11] + filter[5]*src[12] + 64) >> 7];
        tmp[10] = cm[(filter[2]*src[10] - filter[1]*src[ 9] + filter[0]*src[ 8] + filter[3]*src[11] - filter[4]*src[12] + filter[5]*src[13] + 64) >> 7];
        tmp[11] = cm[(filter[2]*src[11] - filter[1]*src[10] + filter[0]*src[ 9] + filter[3]*src[12] - filter[4]*src[13] + filter[5]*src[14] + 64) >> 7];
        tmp[12] = cm[(filter[2]*src[12] - filter[1]*src[11] + filter[0]*src[10] + filter[3]*src[13] - filter[4]*src[14] + filter[5]*src[15] + 64) >> 7];
        tmp[13] = cm[(filter[2]*src[13] - filter[1]*src[12] + filter[0]*src[11] + filter[3]*src[14] - filter[4]*src[15] + filter[5]*src[16] + 64) >> 7];
        tmp[14] = cm[(filter[2]*src[14] - filter[1]*src[13] + filter[0]*src[12] + filter[3]*src[15] - filter[4]*src[16] + filter[5]*src[17] + 64) >> 7];
        tmp[15] = cm[(filter[2]*src[15] - filter[1]*src[14] + filter[0]*src[13] + filter[3]*src[16] - filter[4]*src[17] + filter[5]*src[18] + 64) >> 7];
        */
        __asm__ volatile (
            "xor        %[ftmp0],   %[ftmp0],       %[ftmp0]            \n\t"
            "gsldlc1    %[ftmp1],   0x07(%[src])                        \n\t"
            "gsldrc1    %[ftmp1],   0x00(%[src])                        \n\t"
            "gsldlc1    %[ftmp7],   0x0f(%[src])                        \n\t"
            "mtc1       %[filter2], %[ftmp4]                            \n\t"
            "gsldrc1    %[ftmp7],   0x08(%[src])                        \n\t"
            "pshufh     %[ftmp4],   %[ftmp4],       %[ftmp0]            \n\t"
            "punpcklbh  %[ftmp2],   %[ftmp1],       %[ftmp0]            \n\t"
            "punpckhbh  %[ftmp3],   %[ftmp1],       %[ftmp0]            \n\t"
            "pmullh     %[ftmp5],   %[ftmp2],       %[ftmp4]            \n\t"
            "pmullh     %[ftmp6],   %[ftmp3],       %[ftmp4]            \n\t"
            "punpcklbh  %[ftmp2],   %[ftmp7],       %[ftmp0]            \n\t"
            "punpckhbh  %[ftmp3],   %[ftmp7],       %[ftmp0]            \n\t"
            "pmullh     %[ftmp8],   %[ftmp2],       %[ftmp4]            \n\t"
            "pmullh     %[ftmp9],   %[ftmp3],       %[ftmp4]            \n\t"

            "gsldlc1    %[ftmp1],   0x06(%[src])                        \n\t"
            "gsldrc1    %[ftmp1],  -0x01(%[src])                        \n\t"
            "gsldlc1    %[ftmp7],   0x0e(%[src])                        \n\t"
            "mtc1       %[filter1], %[ftmp4]                            \n\t"
            "gsldrc1    %[ftmp7],   0x07(%[src])                        \n\t"
            "pshufh     %[ftmp4],   %[ftmp4],       %[ftmp0]            \n\t"
            "punpcklbh  %[ftmp2],   %[ftmp1],       %[ftmp0]            \n\t"
            "punpckhbh  %[ftmp3],   %[ftmp1],       %[ftmp0]            \n\t"
            "pmullh     %[ftmp2],   %[ftmp2],       %[ftmp4]            \n\t"
            "pmullh     %[ftmp3],   %[ftmp3],       %[ftmp4]            \n\t"
            "psubush    %[ftmp5],   %[ftmp5],       %[ftmp2]            \n\t"
            "psubush    %[ftmp6],   %[ftmp6],       %[ftmp3]            \n\t"
            "punpcklbh  %[ftmp2],   %[ftmp7],       %[ftmp0]            \n\t"
            "punpckhbh  %[ftmp3],   %[ftmp7],       %[ftmp0]            \n\t"
            "pmullh     %[ftmp2],   %[ftmp2],       %[ftmp4]            \n\t"
            "pmullh     %[ftmp3],   %[ftmp3],       %[ftmp4]            \n\t"
            "psubush    %[ftmp8],   %[ftmp8],       %[ftmp2]            \n\t"
            "psubush    %[ftmp9],   %[ftmp9],       %[ftmp3]            \n\t"

            "gsldlc1    %[ftmp1],   0x05(%[src])                        \n\t"
            "gsldrc1    %[ftmp1],  -0x02(%[src])                        \n\t"
            "gsldlc1    %[ftmp7],   0x0d(%[src])                        \n\t"
            "mtc1       %[filter0], %[ftmp4]                            \n\t"
            "gsldrc1    %[ftmp7],   0x06(%[src])                        \n\t"
            "pshufh     %[ftmp4],   %[ftmp4],       %[ftmp0]            \n\t"
            "punpcklbh  %[ftmp2],   %[ftmp1],       %[ftmp0]            \n\t"
            "punpckhbh  %[ftmp3],   %[ftmp1],       %[ftmp0]            \n\t"
            "pmullh     %[ftmp2],   %[ftmp2],       %[ftmp4]            \n\t"
            "pmullh     %[ftmp3],   %[ftmp3],       %[ftmp4]            \n\t"
            "paddush    %[ftmp5],   %[ftmp5],       %[ftmp2]            \n\t"
            "paddush    %[ftmp6],   %[ftmp6],       %[ftmp3]            \n\t"
            "punpcklbh  %[ftmp2],   %[ftmp7],       %[ftmp0]            \n\t"
            "punpckhbh  %[ftmp3],   %[ftmp7],       %[ftmp0]            \n\t"
            "pmullh     %[ftmp2],   %[ftmp2],       %[ftmp4]            \n\t"
            "pmullh     %[ftmp3],   %[ftmp3],       %[ftmp4]            \n\t"
            "paddush    %[ftmp8],   %[ftmp8],       %[ftmp2]            \n\t"
            "paddush    %[ftmp9],   %[ftmp9],       %[ftmp3]            \n\t"

            "gsldlc1    %[ftmp1],   0x08(%[src])                        \n\t"
            "gsldrc1    %[ftmp1],   0x01(%[src])                        \n\t"
            "gsldlc1    %[ftmp7],   0x10(%[src])                        \n\t"
            "mtc1       %[filter3], %[ftmp4]                            \n\t"
            "gsldrc1    %[ftmp7],   0x09(%[src])                        \n\t"
            "pshufh     %[ftmp4],   %[ftmp4],       %[ftmp0]            \n\t"
            "punpcklbh  %[ftmp2],   %[ftmp1],       %[ftmp0]            \n\t"
            "punpckhbh  %[ftmp3],   %[ftmp1],       %[ftmp0]            \n\t"
            "pmullh     %[ftmp2],   %[ftmp2],       %[ftmp4]            \n\t"
            "pmullh     %[ftmp3],   %[ftmp3],       %[ftmp4]            \n\t"
            "paddush    %[ftmp5],   %[ftmp5],       %[ftmp2]            \n\t"
            "paddush    %[ftmp6],   %[ftmp6],       %[ftmp3]            \n\t"
            "punpcklbh  %[ftmp2],   %[ftmp7],       %[ftmp0]            \n\t"
            "punpckhbh  %[ftmp3],   %[ftmp7],       %[ftmp0]            \n\t"
            "pmullh     %[ftmp2],   %[ftmp2],       %[ftmp4]            \n\t"
            "pmullh     %[ftmp3],   %[ftmp3],       %[ftmp4]            \n\t"
            "paddush    %[ftmp8],   %[ftmp8],       %[ftmp2]            \n\t"
            "paddush    %[ftmp9],   %[ftmp9],       %[ftmp3]            \n\t"

            "gsldlc1    %[ftmp1],   0x09(%[src])                        \n\t"
            "gsldrc1    %[ftmp1],   0x02(%[src])                        \n\t"
            "gsldlc1    %[ftmp7],   0x11(%[src])                        \n\t"
            "mtc1       %[filter4], %[ftmp4]                            \n\t"
            "gsldrc1    %[ftmp7],   0x0a(%[src])                        \n\t"
            "pshufh     %[ftmp4],   %[ftmp4],       %[ftmp0]            \n\t"
            "punpcklbh  %[ftmp2],   %[ftmp1],       %[ftmp0]            \n\t"
            "punpckhbh  %[ftmp3],   %[ftmp1],       %[ftmp0]            \n\t"
            "pmullh     %[ftmp2],   %[ftmp2],       %[ftmp4]            \n\t"
            "pmullh     %[ftmp3],   %[ftmp3],       %[ftmp4]            \n\t"
            "psubush    %[ftmp5],   %[ftmp5],       %[ftmp2]            \n\t"
            "psubush    %[ftmp6],   %[ftmp6],       %[ftmp3]            \n\t"
            "punpcklbh  %[ftmp2],   %[ftmp7],       %[ftmp0]            \n\t"
            "punpckhbh  %[ftmp3],   %[ftmp7],       %[ftmp0]            \n\t"
            "pmullh     %[ftmp2],   %[ftmp2],       %[ftmp4]            \n\t"
            "pmullh     %[ftmp3],   %[ftmp3],       %[ftmp4]            \n\t"
            "psubush    %[ftmp8],   %[ftmp8],       %[ftmp2]            \n\t"
            "psubush    %[ftmp9],   %[ftmp9],       %[ftmp3]            \n\t"

            "gsldlc1    %[ftmp1],   0x0a(%[src])                        \n\t"
            "gsldrc1    %[ftmp1],   0x03(%[src])                        \n\t"
            "gsldlc1    %[ftmp7],   0x12(%[src])                        \n\t"
            "mtc1       %[filter5], %[ftmp4]                            \n\t"
            "gsldrc1    %[ftmp7],   0x0b(%[src])                        \n\t"
            "pshufh     %[ftmp4],   %[ftmp4],       %[ftmp0]            \n\t"
            "punpcklbh  %[ftmp2],   %[ftmp1],       %[ftmp0]            \n\t"
            "punpckhbh  %[ftmp3],   %[ftmp1],       %[ftmp0]            \n\t"
            "pmullh     %[ftmp2],   %[ftmp2],       %[ftmp4]            \n\t"
            "pmullh     %[ftmp3],   %[ftmp3],       %[ftmp4]            \n\t"
            "paddush    %[ftmp5],   %[ftmp5],       %[ftmp2]            \n\t"
            "paddush    %[ftmp6],   %[ftmp6],       %[ftmp3]            \n\t"
            "punpcklbh  %[ftmp2],   %[ftmp7],       %[ftmp0]            \n\t"
            "punpckhbh  %[ftmp3],   %[ftmp7],       %[ftmp0]            \n\t"
            "pmullh     %[ftmp2],   %[ftmp2],       %[ftmp4]            \n\t"
            "pmullh     %[ftmp3],   %[ftmp3],       %[ftmp4]            \n\t"
            "paddush    %[ftmp8],   %[ftmp8],       %[ftmp2]            \n\t"
            "paddush    %[ftmp9],   %[ftmp9],       %[ftmp3]            \n\t"

            "li         %[tmp0],    0x07                                \n\t"
            "paddush    %[ftmp5],   %[ftmp5],       %[ff_pw_64]         \n\t"
            "paddush    %[ftmp6],   %[ftmp6],       %[ff_pw_64]         \n\t"
            "paddush    %[ftmp8],   %[ftmp8],       %[ff_pw_64]         \n\t"
            "paddush    %[ftmp9],   %[ftmp9],       %[ff_pw_64]         \n\t"
            "mtc1       %[tmp0],    %[ftmp4]                            \n\t"
            "psrlh      %[ftmp5],   %[ftmp5],       %[ftmp4]            \n\t"
            "psrlh      %[ftmp6],   %[ftmp6],       %[ftmp4]            \n\t"
            "psrlh      %[ftmp8],   %[ftmp8],       %[ftmp4]            \n\t"
            "psrlh      %[ftmp9],   %[ftmp9],       %[ftmp4]            \n\t"

            "packushb   %[ftmp1],   %[ftmp5],       %[ftmp6]            \n\t"
            "packushb   %[ftmp2],   %[ftmp8],       %[ftmp9]            \n\t"
            "gssdlc1    %[ftmp1],   0x07(%[tmp])                        \n\t"
            "gssdrc1    %[ftmp1],   0x00(%[tmp])                        \n\t"
            "gssdlc1    %[ftmp2],   0x0f(%[tmp])                        \n\t"
            "gssdrc1    %[ftmp2],   0x08(%[tmp])                        \n\t"
            : [ftmp0]"=&f"(ftmp[0]),        [ftmp1]"=&f"(ftmp[1]),
              [ftmp2]"=&f"(ftmp[2]),        [ftmp3]"=&f"(ftmp[3]),
              [ftmp4]"=&f"(ftmp[4]),        [ftmp5]"=&f"(ftmp[5]),
              [ftmp6]"=&f"(ftmp[6]),        [ftmp7]"=&f"(ftmp[7]),
              [ftmp8]"=&f"(ftmp[8]),        [ftmp9]"=&f"(ftmp[9]),
              [tmp0]"=&r"(tmp0)
            : [tmp]"r"(tmp),                [src]"r"(src),
              [ff_pw_64]"f"(ff_pw_64),
              [filter0]"r"(filter[0]),      [filter1]"r"(filter[1]),
              [filter2]"r"(filter[2]),      [filter3]"r"(filter[3]),
              [filter4]"r"(filter[4]),      [filter5]"r"(filter[5])
            : "memory"
        );

        tmp += 16;
        src += srcstride;
    }
#else
    const uint8_t *filter = subpel_filters[mx - 1];
    const uint8_t *cm     = ff_crop_tab + MAX_NEG_CROP;
    int x, y;
    uint8_t tmp_array[592];
    uint8_t *tmp = tmp_array;

    src -= 2 * srcstride;

    for (y = 0; y < h + 5; y++) {
        for (x = 0; x < 16; x++)
            tmp[x] = FILTER_6TAP(src, filter, 1);
        tmp += 16;
        src += srcstride;
    }
#endif

    tmp    = tmp_array + 32;
    filter = subpel_filters[my - 1];

#if NOTOK //FIXME fate-vp8-test-vector-006
    for (y = 0; y < h; y++) {
        /*
        dst[0] = cm[(filter[2]*tmp[0] - filter[1]*tmp[-16] + filter[0]*tmp[-32] + filter[3]*tmp[16] - filter[4]*tmp[32] + filter[5]*tmp[48] + 64) >> 7];
        dst[1] = cm[(filter[2]*tmp[1] - filter[1]*tmp[-15] + filter[0]*tmp[-31] + filter[3]*tmp[17] - filter[4]*tmp[33] + filter[5]*tmp[49] + 64) >> 7];
        dst[2] = cm[(filter[2]*tmp[2] - filter[1]*tmp[-14] + filter[0]*tmp[-30] + filter[3]*tmp[18] - filter[4]*tmp[34] + filter[5]*tmp[50] + 64) >> 7];
        dst[3] = cm[(filter[2]*tmp[3] - filter[1]*tmp[-13] + filter[0]*tmp[-29] + filter[3]*tmp[19] - filter[4]*tmp[35] + filter[5]*tmp[51] + 64) >> 7];
        dst[4] = cm[(filter[2]*tmp[4] - filter[1]*tmp[-12] + filter[0]*tmp[-28] + filter[3]*tmp[20] - filter[4]*tmp[36] + filter[5]*tmp[52] + 64) >> 7];
        dst[5] = cm[(filter[2]*tmp[5] - filter[1]*tmp[-11] + filter[0]*tmp[-27] + filter[3]*tmp[21] - filter[4]*tmp[37] + filter[5]*tmp[53] + 64) >> 7];
        dst[6] = cm[(filter[2]*tmp[6] - filter[1]*tmp[-10] + filter[0]*tmp[-26] + filter[3]*tmp[22] - filter[4]*tmp[38] + filter[5]*tmp[54] + 64) >> 7];
        dst[7] = cm[(filter[2]*tmp[7] - filter[1]*tmp[ -9] + filter[0]*tmp[-25] + filter[3]*tmp[23] - filter[4]*tmp[39] + filter[5]*tmp[55] + 64) >> 7];

        dst[ 8] = cm[(filter[2]*tmp[ 8] - filter[1]*tmp[-8] + filter[0]*tmp[-24] + filter[3]*tmp[24] - filter[4]*tmp[40] + filter[5]*tmp[56] + 64) >> 7];
        dst[ 9] = cm[(filter[2]*tmp[ 9] - filter[1]*tmp[-7] + filter[0]*tmp[-23] + filter[3]*tmp[25] - filter[4]*tmp[41] + filter[5]*tmp[57] + 64) >> 7];
        dst[10] = cm[(filter[2]*tmp[10] - filter[1]*tmp[-6] + filter[0]*tmp[-22] + filter[3]*tmp[26] - filter[4]*tmp[42] + filter[5]*tmp[58] + 64) >> 7];
        dst[11] = cm[(filter[2]*tmp[11] - filter[1]*tmp[-5] + filter[0]*tmp[-21] + filter[3]*tmp[27] - filter[4]*tmp[43] + filter[5]*tmp[59] + 64) >> 7];
        dst[12] = cm[(filter[2]*tmp[12] - filter[1]*tmp[-4] + filter[0]*tmp[-20] + filter[3]*tmp[28] - filter[4]*tmp[44] + filter[5]*tmp[60] + 64) >> 7];
        dst[13] = cm[(filter[2]*tmp[13] - filter[1]*tmp[-3] + filter[0]*tmp[-19] + filter[3]*tmp[29] - filter[4]*tmp[45] + filter[5]*tmp[61] + 64) >> 7];
        dst[14] = cm[(filter[2]*tmp[14] - filter[1]*tmp[-2] + filter[0]*tmp[-18] + filter[3]*tmp[30] - filter[4]*tmp[46] + filter[5]*tmp[62] + 64) >> 7];
        dst[15] = cm[(filter[2]*tmp[15] - filter[1]*tmp[-1] + filter[0]*tmp[-17] + filter[3]*tmp[31] - filter[4]*tmp[47] + filter[5]*tmp[63] + 64) >> 7];
        */
        __asm__ volatile (
            "xor        %[ftmp0],   %[ftmp0],       %[ftmp0]            \n\t"
            "gsldlc1    %[ftmp1],   0x07(%[tmp])                        \n\t"
            "gsldrc1    %[ftmp1],   0x00(%[tmp])                        \n\t"
            "gsldlc1    %[ftmp7],   0x0f(%[tmp])                        \n\t"
            "mtc1       %[filter2], %[ftmp4]                            \n\t"
            "gsldrc1    %[ftmp7],   0x08(%[tmp])                        \n\t"
            "pshufh     %[ftmp4],   %[ftmp4],       %[ftmp0]            \n\t"
            "punpcklbh  %[ftmp2],   %[ftmp1],       %[ftmp0]            \n\t"
            "punpckhbh  %[ftmp3],   %[ftmp1],       %[ftmp0]            \n\t"
            "pmullh     %[ftmp5],   %[ftmp2],       %[ftmp4]            \n\t"
            "pmullh     %[ftmp6],   %[ftmp3],       %[ftmp4]            \n\t"
            "punpcklbh  %[ftmp2],   %[ftmp7],       %[ftmp0]            \n\t"
            "punpckhbh  %[ftmp3],   %[ftmp7],       %[ftmp0]            \n\t"
            "pmullh     %[ftmp8],   %[ftmp2],       %[ftmp4]            \n\t"
            "pmullh     %[ftmp9],   %[ftmp3],       %[ftmp4]            \n\t"

            "gsldlc1    %[ftmp1],  -0x09(%[tmp])                        \n\t"
            "gsldrc1    %[ftmp1],  -0x10(%[tmp])                        \n\t"
            "gsldlc1    %[ftmp7],  -0x01(%[tmp])                        \n\t"
            "mtc1       %[filter1], %[ftmp4]                            \n\t"
            "gsldrc1    %[ftmp7],  -0x08(%[tmp])                        \n\t"
            "pshufh     %[ftmp4],   %[ftmp4],       %[ftmp0]            \n\t"
            "punpcklbh  %[ftmp2],   %[ftmp1],       %[ftmp0]            \n\t"
            "punpckhbh  %[ftmp3],   %[ftmp1],       %[ftmp0]            \n\t"
            "pmullh     %[ftmp2],   %[ftmp2],       %[ftmp4]            \n\t"
            "pmullh     %[ftmp3],   %[ftmp3],       %[ftmp4]            \n\t"
            "psubush    %[ftmp5],   %[ftmp5],       %[ftmp2]            \n\t"
            "psubush    %[ftmp6],   %[ftmp6],       %[ftmp3]            \n\t"
            "punpcklbh  %[ftmp2],   %[ftmp7],       %[ftmp0]            \n\t"
            "punpckhbh  %[ftmp3],   %[ftmp7],       %[ftmp0]            \n\t"
            "pmullh     %[ftmp2],   %[ftmp2],       %[ftmp4]            \n\t"
            "pmullh     %[ftmp3],   %[ftmp3],       %[ftmp4]            \n\t"
            "psubush    %[ftmp8],   %[ftmp8],       %[ftmp2]            \n\t"
            "psubush    %[ftmp9],   %[ftmp9],       %[ftmp3]            \n\t"

            "gsldlc1    %[ftmp1],  -0x19(%[tmp])                        \n\t"
            "gsldrc1    %[ftmp1],  -0x20(%[tmp])                        \n\t"
            "gsldlc1    %[ftmp7],  -0x11(%[tmp])                        \n\t"
            "mtc1       %[filter0], %[ftmp4]                            \n\t"
            "gsldrc1    %[ftmp7],  -0x18(%[tmp])                        \n\t"
            "pshufh     %[ftmp4],   %[ftmp4],       %[ftmp0]            \n\t"
            "punpcklbh  %[ftmp2],   %[ftmp1],       %[ftmp0]            \n\t"
            "punpckhbh  %[ftmp3],   %[ftmp1],       %[ftmp0]            \n\t"
            "pmullh     %[ftmp2],   %[ftmp2],       %[ftmp4]            \n\t"
            "pmullh     %[ftmp3],   %[ftmp3],       %[ftmp4]            \n\t"
            "paddush    %[ftmp5],   %[ftmp5],       %[ftmp2]            \n\t"
            "paddush    %[ftmp6],   %[ftmp6],       %[ftmp3]            \n\t"
            "punpcklbh  %[ftmp2],   %[ftmp7],       %[ftmp0]            \n\t"
            "punpckhbh  %[ftmp3],   %[ftmp7],       %[ftmp0]            \n\t"
            "pmullh     %[ftmp2],   %[ftmp2],       %[ftmp4]            \n\t"
            "pmullh     %[ftmp3],   %[ftmp3],       %[ftmp4]            \n\t"
            "paddush    %[ftmp8],   %[ftmp8],       %[ftmp2]            \n\t"
            "paddush    %[ftmp9],   %[ftmp9],       %[ftmp3]            \n\t"

            "gsldlc1    %[ftmp1],   0x17(%[tmp])                        \n\t"
            "gsldrc1    %[ftmp1],   0x10(%[tmp])                        \n\t"
            "gsldlc1    %[ftmp7],   0x1f(%[tmp])                        \n\t"
            "mtc1       %[filter3], %[ftmp4]                            \n\t"
            "gsldrc1    %[ftmp7],   0x18(%[tmp])                        \n\t"
            "pshufh     %[ftmp4],   %[ftmp4],       %[ftmp0]            \n\t"
            "punpcklbh  %[ftmp2],   %[ftmp1],       %[ftmp0]            \n\t"
            "punpckhbh  %[ftmp3],   %[ftmp1],       %[ftmp0]            \n\t"
            "pmullh     %[ftmp2],   %[ftmp2],       %[ftmp4]            \n\t"
            "pmullh     %[ftmp3],   %[ftmp3],       %[ftmp4]            \n\t"
            "paddush    %[ftmp5],   %[ftmp5],       %[ftmp2]            \n\t"
            "paddush    %[ftmp6],   %[ftmp6],       %[ftmp3]            \n\t"
            "punpcklbh  %[ftmp2],   %[ftmp7],       %[ftmp0]            \n\t"
            "punpckhbh  %[ftmp3],   %[ftmp7],       %[ftmp0]            \n\t"
            "pmullh     %[ftmp2],   %[ftmp2],       %[ftmp4]            \n\t"
            "pmullh     %[ftmp3],   %[ftmp3],       %[ftmp4]            \n\t"
            "paddush    %[ftmp8],   %[ftmp8],       %[ftmp2]            \n\t"
            "paddush    %[ftmp9],   %[ftmp9],       %[ftmp3]            \n\t"

            "gsldlc1    %[ftmp1],   0x27(%[tmp])                        \n\t"
            "gsldrc1    %[ftmp1],   0x20(%[tmp])                        \n\t"
            "gsldlc1    %[ftmp7],   0x2f(%[tmp])                        \n\t"
            "mtc1       %[filter4], %[ftmp4]                            \n\t"
            "gsldrc1    %[ftmp7],   0x28(%[tmp])                        \n\t"
            "pshufh     %[ftmp4],   %[ftmp4],       %[ftmp0]            \n\t"
            "punpcklbh  %[ftmp2],   %[ftmp1],       %[ftmp0]            \n\t"
            "punpckhbh  %[ftmp3],   %[ftmp1],       %[ftmp0]            \n\t"
            "pmullh     %[ftmp2],   %[ftmp2],       %[ftmp4]            \n\t"
            "pmullh     %[ftmp3],   %[ftmp3],       %[ftmp4]            \n\t"
            "psubush    %[ftmp5],   %[ftmp5],       %[ftmp2]            \n\t"
            "psubush    %[ftmp6],   %[ftmp6],       %[ftmp3]            \n\t"
            "punpcklbh  %[ftmp2],   %[ftmp7],       %[ftmp0]            \n\t"
            "punpckhbh  %[ftmp3],   %[ftmp7],       %[ftmp0]            \n\t"
            "pmullh     %[ftmp2],   %[ftmp2],       %[ftmp4]            \n\t"
            "pmullh     %[ftmp3],   %[ftmp3],       %[ftmp4]            \n\t"
            "psubush    %[ftmp8],   %[ftmp8],       %[ftmp2]            \n\t"
            "psubush    %[ftmp9],   %[ftmp9],       %[ftmp3]            \n\t"

            "gsldlc1    %[ftmp1],   0x37(%[tmp])                        \n\t"
            "gsldrc1    %[ftmp1],   0x30(%[tmp])                        \n\t"
            "gsldlc1    %[ftmp7],   0x3f(%[tmp])                        \n\t"
            "mtc1       %[filter5], %[ftmp4]                            \n\t"
            "gsldrc1    %[ftmp7],   0x38(%[tmp])                        \n\t"
            "pshufh     %[ftmp4],   %[ftmp4],       %[ftmp0]            \n\t"
            "punpcklbh  %[ftmp2],   %[ftmp1],       %[ftmp0]            \n\t"
            "punpckhbh  %[ftmp3],   %[ftmp1],       %[ftmp0]            \n\t"
            "pmullh     %[ftmp2],   %[ftmp2],       %[ftmp4]            \n\t"
            "pmullh     %[ftmp3],   %[ftmp3],       %[ftmp4]            \n\t"
            "paddush    %[ftmp5],   %[ftmp5],       %[ftmp2]            \n\t"
            "paddush    %[ftmp6],   %[ftmp6],       %[ftmp3]            \n\t"
            "punpcklbh  %[ftmp2],   %[ftmp7],       %[ftmp0]            \n\t"
            "punpckhbh  %[ftmp3],   %[ftmp7],       %[ftmp0]            \n\t"
            "pmullh     %[ftmp2],   %[ftmp2],       %[ftmp4]            \n\t"
            "pmullh     %[ftmp3],   %[ftmp3],       %[ftmp4]            \n\t"
            "paddush    %[ftmp8],   %[ftmp8],       %[ftmp2]            \n\t"
            "paddush    %[ftmp9],   %[ftmp9],       %[ftmp3]            \n\t"

            "li         %[tmp0],    0x07                                \n\t"
            "paddush    %[ftmp5],   %[ftmp5],       %[ff_pw_64]         \n\t"
            "paddush    %[ftmp6],   %[ftmp6],       %[ff_pw_64]         \n\t"
            "paddush    %[ftmp8],   %[ftmp8],       %[ff_pw_64]         \n\t"
            "paddush    %[ftmp9],   %[ftmp9],       %[ff_pw_64]         \n\t"
            "mtc1       %[tmp0],    %[ftmp4]                            \n\t"
            "psrlh      %[ftmp5],   %[ftmp5],       %[ftmp4]            \n\t"
            "psrlh      %[ftmp6],   %[ftmp6],       %[ftmp4]            \n\t"
            "psrlh      %[ftmp8],   %[ftmp8],       %[ftmp4]            \n\t"
            "psrlh      %[ftmp9],   %[ftmp9],       %[ftmp4]            \n\t"

            "packushb   %[ftmp1],   %[ftmp5],       %[ftmp6]            \n\t"
            "packushb   %[ftmp2],   %[ftmp8],       %[ftmp9]            \n\t"
            "gssdlc1    %[ftmp1],   0x07(%[dst])                        \n\t"
            "gssdrc1    %[ftmp1],   0x00(%[dst])                        \n\t"
            "gssdlc1    %[ftmp2],   0x0f(%[dst])                        \n\t"
            "gssdrc1    %[ftmp2],   0x08(%[dst])                        \n\t"
            : [ftmp0]"=&f"(ftmp[0]),        [ftmp1]"=&f"(ftmp[1]),
              [ftmp2]"=&f"(ftmp[2]),        [ftmp3]"=&f"(ftmp[3]),
              [ftmp4]"=&f"(ftmp[4]),        [ftmp5]"=&f"(ftmp[5]),
              [ftmp6]"=&f"(ftmp[6]),        [ftmp7]"=&f"(ftmp[7]),
              [ftmp8]"=&f"(ftmp[8]),        [ftmp9]"=&f"(ftmp[9]),
              [tmp0]"=&r"(tmp0)
            : [dst]"r"(dst),                [tmp]"r"(tmp),
              [ff_pw_64]"f"(ff_pw_64),
              [filter0]"r"(filter[0]),      [filter1]"r"(filter[1]),
              [filter2]"r"(filter[2]),      [filter3]"r"(filter[3]),
              [filter4]"r"(filter[4]),      [filter5]"r"(filter[5])
            : "memory"
        );

        dst += dststride;
        tmp += 16;
    }
#else
    for (y = 0; y < h; y++) {
        for (x = 0; x < 16; x++)
            dst[x] = FILTER_6TAP(tmp, filter, 16);
        dst += dststride;
        tmp += 16;
    }
#endif
}

void ff_put_vp8_epel8_h6v6_mmi(uint8_t *dst, ptrdiff_t dststride, uint8_t *src,
        ptrdiff_t srcstride, int h, int mx, int my)
{
#if NOTOK //FIXME fate-vp8-test-vector-002 006 009
    const uint8_t *filter = subpel_filters[mx - 1];
    int y;
    uint8_t tmp_array[168];
    uint8_t *tmp = tmp_array;
    double ftmp[7];
    uint64_t tmp0;

    src -= 2 * srcstride;

    for (y = 0; y < h + 5; y++) {
        /*
        tmp[0] = cm[(filter[2]*src[0] - filter[1]*src[-1] + filter[0]*src[-2] + filter[3]*src[1] - filter[4]*src[2] + filter[5]*src[ 3] + 64) >> 7];
        tmp[1] = cm[(filter[2]*src[1] - filter[1]*src[ 0] + filter[0]*src[-1] + filter[3]*src[2] - filter[4]*src[3] + filter[5]*src[ 4] + 64) >> 7];
        tmp[2] = cm[(filter[2]*src[2] - filter[1]*src[ 1] + filter[0]*src[ 0] + filter[3]*src[3] - filter[4]*src[4] + filter[5]*src[ 5] + 64) >> 7];
        tmp[3] = cm[(filter[2]*src[3] - filter[1]*src[ 2] + filter[0]*src[ 1] + filter[3]*src[4] - filter[4]*src[5] + filter[5]*src[ 6] + 64) >> 7];
        tmp[4] = cm[(filter[2]*src[4] - filter[1]*src[ 3] + filter[0]*src[ 2] + filter[3]*src[5] - filter[4]*src[6] + filter[5]*src[ 7] + 64) >> 7];
        tmp[5] = cm[(filter[2]*src[5] - filter[1]*src[ 4] + filter[0]*src[ 3] + filter[3]*src[6] - filter[4]*src[7] + filter[5]*src[ 8] + 64) >> 7];
        tmp[6] = cm[(filter[2]*src[6] - filter[1]*src[ 5] + filter[0]*src[ 4] + filter[3]*src[7] - filter[4]*src[8] + filter[5]*src[ 9] + 64) >> 7];
        tmp[7] = cm[(filter[2]*src[7] - filter[1]*src[ 6] + filter[0]*src[ 5] + filter[3]*src[8] - filter[4]*src[9] + filter[5]*src[10] + 64) >> 7];
        */
        __asm__ volatile (
            "xor        %[ftmp0],   %[ftmp0],       %[ftmp0]            \n\t"
            "gsldlc1    %[ftmp1],   0x07(%[src])                        \n\t"
            "mtc1       %[filter2], %[ftmp4]                            \n\t"
            "gsldrc1    %[ftmp1],   0x00(%[src])                        \n\t"
            "pshufh     %[ftmp4],   %[ftmp4],       %[ftmp0]            \n\t"
            "punpcklbh  %[ftmp2],   %[ftmp1],       %[ftmp0]            \n\t"
            "punpckhbh  %[ftmp3],   %[ftmp1],       %[ftmp0]            \n\t"
            "pmullh     %[ftmp5],   %[ftmp2],       %[ftmp4]            \n\t"
            "pmullh     %[ftmp6],   %[ftmp3],       %[ftmp4]            \n\t"

            "gsldlc1    %[ftmp1],   0x06(%[src])                        \n\t"
            "mtc1       %[filter1], %[ftmp4]                            \n\t"
            "gsldrc1    %[ftmp1],  -0x01(%[src])                        \n\t"
            "pshufh     %[ftmp4],   %[ftmp4],       %[ftmp0]            \n\t"
            "punpcklbh  %[ftmp2],   %[ftmp1],       %[ftmp0]            \n\t"
            "punpckhbh  %[ftmp3],   %[ftmp1],       %[ftmp0]            \n\t"
            "pmullh     %[ftmp2],   %[ftmp2],       %[ftmp4]            \n\t"
            "pmullh     %[ftmp3],   %[ftmp3],       %[ftmp4]            \n\t"
            "psubush    %[ftmp5],   %[ftmp5],       %[ftmp2]            \n\t"
            "psubush    %[ftmp6],   %[ftmp6],       %[ftmp3]            \n\t"

            "gsldlc1    %[ftmp1],   0x05(%[src])                        \n\t"
            "mtc1       %[filter0], %[ftmp4]                            \n\t"
            "gsldrc1    %[ftmp1],  -0x02(%[src])                        \n\t"
            "pshufh     %[ftmp4],   %[ftmp4],       %[ftmp0]            \n\t"
            "punpcklbh  %[ftmp2],   %[ftmp1],       %[ftmp0]            \n\t"
            "punpckhbh  %[ftmp3],   %[ftmp1],       %[ftmp0]            \n\t"
            "pmullh     %[ftmp2],   %[ftmp2],       %[ftmp4]            \n\t"
            "pmullh     %[ftmp3],   %[ftmp3],       %[ftmp4]            \n\t"
            "paddush    %[ftmp5],   %[ftmp5],       %[ftmp2]            \n\t"
            "paddush    %[ftmp6],   %[ftmp6],       %[ftmp3]            \n\t"

            "gsldlc1    %[ftmp1],   0x08(%[src])                        \n\t"
            "mtc1       %[filter3], %[ftmp4]                            \n\t"
            "gsldrc1    %[ftmp1],   0x01(%[src])                        \n\t"
            "pshufh     %[ftmp4],   %[ftmp4],       %[ftmp0]            \n\t"
            "punpcklbh  %[ftmp2],   %[ftmp1],       %[ftmp0]            \n\t"
            "punpckhbh  %[ftmp3],   %[ftmp1],       %[ftmp0]            \n\t"
            "pmullh     %[ftmp2],   %[ftmp2],       %[ftmp4]            \n\t"
            "pmullh     %[ftmp3],   %[ftmp3],       %[ftmp4]            \n\t"
            "paddush    %[ftmp5],   %[ftmp5],       %[ftmp2]            \n\t"
            "paddush    %[ftmp6],   %[ftmp6],       %[ftmp3]            \n\t"

            "gsldlc1    %[ftmp1],   0x09(%[src])                        \n\t"
            "mtc1       %[filter4], %[ftmp4]                            \n\t"
            "gsldrc1    %[ftmp1],   0x02(%[src])                        \n\t"
            "pshufh     %[ftmp4],   %[ftmp4],       %[ftmp0]            \n\t"
            "punpcklbh  %[ftmp2],   %[ftmp1],       %[ftmp0]            \n\t"
            "punpckhbh  %[ftmp3],   %[ftmp1],       %[ftmp0]            \n\t"
            "pmullh     %[ftmp2],   %[ftmp2],       %[ftmp4]            \n\t"
            "pmullh     %[ftmp3],   %[ftmp3],       %[ftmp4]            \n\t"
            "psubush    %[ftmp5],   %[ftmp5],       %[ftmp2]            \n\t"
            "psubush    %[ftmp6],   %[ftmp6],       %[ftmp3]            \n\t"

            "gsldlc1    %[ftmp1],   0x0a(%[src])                        \n\t"
            "mtc1       %[filter5], %[ftmp4]                            \n\t"
            "gsldrc1    %[ftmp1],   0x03(%[src])                        \n\t"
            "pshufh     %[ftmp4],   %[ftmp4],       %[ftmp0]            \n\t"
            "punpcklbh  %[ftmp2],   %[ftmp1],       %[ftmp0]            \n\t"
            "punpckhbh  %[ftmp3],   %[ftmp1],       %[ftmp0]            \n\t"
            "pmullh     %[ftmp2],   %[ftmp2],       %[ftmp4]            \n\t"
            "pmullh     %[ftmp3],   %[ftmp3],       %[ftmp4]            \n\t"
            "paddush    %[ftmp5],   %[ftmp5],       %[ftmp2]            \n\t"
            "paddush    %[ftmp6],   %[ftmp6],       %[ftmp3]            \n\t"

            "li         %[tmp0],    0x07                                \n\t"
            "paddush    %[ftmp5],   %[ftmp5],       %[ff_pw_64]         \n\t"
            "paddush    %[ftmp6],   %[ftmp6],       %[ff_pw_64]         \n\t"
            "mtc1       %[tmp0],    %[ftmp4]                            \n\t"
            "psrlh      %[ftmp5],   %[ftmp5],       %[ftmp4]            \n\t"
            "psrlh      %[ftmp6],   %[ftmp6],       %[ftmp4]            \n\t"

            "packushb   %[ftmp1],   %[ftmp5],       %[ftmp6]            \n\t"
            "gssdlc1    %[ftmp1],   0x07(%[tmp])                        \n\t"
            "gssdrc1    %[ftmp1],   0x00(%[tmp])                        \n\t"
            : [ftmp0]"=&f"(ftmp[0]),        [ftmp1]"=&f"(ftmp[1]),
              [ftmp2]"=&f"(ftmp[2]),        [ftmp3]"=&f"(ftmp[3]),
              [ftmp4]"=&f"(ftmp[4]),        [ftmp5]"=&f"(ftmp[5]),
              [ftmp6]"=&f"(ftmp[6]),
              [tmp0]"=&r"(tmp0)
            : [tmp]"r"(tmp),                [src]"r"(src),
              [ff_pw_64]"f"(ff_pw_64),
              [filter0]"r"(filter[0]),      [filter1]"r"(filter[1]),
              [filter2]"r"(filter[2]),      [filter3]"r"(filter[3]),
              [filter4]"r"(filter[4]),      [filter5]"r"(filter[5])
            : "memory"
        );

        tmp += 8;
        src += srcstride;
    }
#else
    const uint8_t *filter = subpel_filters[mx - 1];
    const uint8_t *cm     = ff_crop_tab + MAX_NEG_CROP;
    int x, y;
    uint8_t tmp_array[168];
    uint8_t *tmp = tmp_array;

    src -= 2 * srcstride;

    for (y = 0; y < h + 5; y++) {
        for (x = 0; x < 8; x++)
            tmp[x] = FILTER_6TAP(src, filter, 1);
        tmp += 8;
        src += srcstride;
    }
#endif

    tmp    = tmp_array + 16;
    filter = subpel_filters[my - 1];

#if NOTOK //FIXME fate-vp8-test-vector-006
    for (y = 0; y < h; y++) {
        /*
        dst[0] = cm[(filter[2]*tmp[0] - filter[1]*tmp[-8] + filter[0]*tmp[-16] + filter[3]*tmp[ 8] - filter[4]*tmp[16] + filter[5]*tmp[24] + 64) >> 7];
        dst[1] = cm[(filter[2]*tmp[1] - filter[1]*tmp[-7] + filter[0]*tmp[-15] + filter[3]*tmp[ 9] - filter[4]*tmp[17] + filter[5]*tmp[25] + 64) >> 7];
        dst[2] = cm[(filter[2]*tmp[2] - filter[1]*tmp[-6] + filter[0]*tmp[-14] + filter[3]*tmp[10] - filter[4]*tmp[18] + filter[5]*tmp[26] + 64) >> 7];
        dst[3] = cm[(filter[2]*tmp[3] - filter[1]*tmp[-5] + filter[0]*tmp[-13] + filter[3]*tmp[11] - filter[4]*tmp[19] + filter[5]*tmp[27] + 64) >> 7];
        dst[4] = cm[(filter[2]*tmp[4] - filter[1]*tmp[-4] + filter[0]*tmp[-12] + filter[3]*tmp[12] - filter[4]*tmp[20] + filter[5]*tmp[28] + 64) >> 7];
        dst[5] = cm[(filter[2]*tmp[5] - filter[1]*tmp[-3] + filter[0]*tmp[-11] + filter[3]*tmp[13] - filter[4]*tmp[21] + filter[5]*tmp[29] + 64) >> 7];
        dst[6] = cm[(filter[2]*tmp[6] - filter[1]*tmp[-2] + filter[0]*tmp[-10] + filter[3]*tmp[14] - filter[4]*tmp[22] + filter[5]*tmp[30] + 64) >> 7];
        dst[7] = cm[(filter[2]*tmp[7] - filter[1]*tmp[-1] + filter[0]*tmp[ -9] + filter[3]*tmp[15] - filter[4]*tmp[23] + filter[5]*tmp[31] + 64) >> 7];
        */
        __asm__ volatile (
            "xor        %[ftmp0],   %[ftmp0],       %[ftmp0]            \n\t"
            "gsldlc1    %[ftmp1],   0x07(%[tmp])                        \n\t"
            "mtc1       %[filter2], %[ftmp4]                            \n\t"
            "gsldrc1    %[ftmp1],   0x00(%[tmp])                        \n\t"
            "pshufh     %[ftmp4],   %[ftmp4],       %[ftmp0]            \n\t"
            "punpcklbh  %[ftmp2],   %[ftmp1],       %[ftmp0]            \n\t"
            "punpckhbh  %[ftmp3],   %[ftmp1],       %[ftmp0]            \n\t"
            "pmullh     %[ftmp5],   %[ftmp2],       %[ftmp4]            \n\t"
            "pmullh     %[ftmp6],   %[ftmp3],       %[ftmp4]            \n\t"

            "gsldlc1    %[ftmp1],  -0x01(%[tmp])                        \n\t"
            "mtc1       %[filter1], %[ftmp4]                            \n\t"
            "gsldrc1    %[ftmp1],  -0x08(%[tmp])                        \n\t"
            "pshufh     %[ftmp4],   %[ftmp4],       %[ftmp0]            \n\t"
            "punpcklbh  %[ftmp2],   %[ftmp1],       %[ftmp0]            \n\t"
            "punpckhbh  %[ftmp3],   %[ftmp1],       %[ftmp0]            \n\t"
            "pmullh     %[ftmp2],   %[ftmp2],       %[ftmp4]            \n\t"
            "pmullh     %[ftmp3],   %[ftmp3],       %[ftmp4]            \n\t"
            "psubush    %[ftmp5],   %[ftmp5],       %[ftmp2]            \n\t"
            "psubush    %[ftmp6],   %[ftmp6],       %[ftmp3]            \n\t"

            "gsldlc1    %[ftmp1],  -0x09(%[tmp])                        \n\t"
            "mtc1       %[filter0], %[ftmp4]                            \n\t"
            "gsldrc1    %[ftmp1],  -0x10(%[tmp])                        \n\t"
            "pshufh     %[ftmp4],   %[ftmp4],       %[ftmp0]            \n\t"
            "punpcklbh  %[ftmp2],   %[ftmp1],       %[ftmp0]            \n\t"
            "punpckhbh  %[ftmp3],   %[ftmp1],       %[ftmp0]            \n\t"
            "pmullh     %[ftmp2],   %[ftmp2],       %[ftmp4]            \n\t"
            "pmullh     %[ftmp3],   %[ftmp3],       %[ftmp4]            \n\t"
            "paddush    %[ftmp5],   %[ftmp5],       %[ftmp2]            \n\t"
            "paddush    %[ftmp6],   %[ftmp6],       %[ftmp3]            \n\t"

            "gsldlc1    %[ftmp1],   0x0f(%[tmp])                        \n\t"
            "mtc1       %[filter3], %[ftmp4]                            \n\t"
            "gsldrc1    %[ftmp1],   0x08(%[tmp])                        \n\t"
            "pshufh     %[ftmp4],   %[ftmp4],       %[ftmp0]            \n\t"
            "punpcklbh  %[ftmp2],   %[ftmp1],       %[ftmp0]            \n\t"
            "punpckhbh  %[ftmp3],   %[ftmp1],       %[ftmp0]            \n\t"
            "pmullh     %[ftmp2],   %[ftmp2],       %[ftmp4]            \n\t"
            "pmullh     %[ftmp3],   %[ftmp3],       %[ftmp4]            \n\t"
            "paddush    %[ftmp5],   %[ftmp5],       %[ftmp2]            \n\t"
            "paddush    %[ftmp6],   %[ftmp6],       %[ftmp3]            \n\t"

            "gsldlc1    %[ftmp1],   0x17(%[tmp])                        \n\t"
            "mtc1       %[filter4], %[ftmp4]                            \n\t"
            "gsldrc1    %[ftmp1],   0x10(%[tmp])                        \n\t"
            "pshufh     %[ftmp4],   %[ftmp4],       %[ftmp0]            \n\t"
            "punpcklbh  %[ftmp2],   %[ftmp1],       %[ftmp0]            \n\t"
            "punpckhbh  %[ftmp3],   %[ftmp1],       %[ftmp0]            \n\t"
            "pmullh     %[ftmp2],   %[ftmp2],       %[ftmp4]            \n\t"
            "pmullh     %[ftmp3],   %[ftmp3],       %[ftmp4]            \n\t"
            "psubush    %[ftmp5],   %[ftmp5],       %[ftmp2]            \n\t"
            "psubush    %[ftmp6],   %[ftmp6],       %[ftmp3]            \n\t"

            "gsldlc1    %[ftmp1],   0x1f(%[tmp])                        \n\t"
            "mtc1       %[filter5], %[ftmp4]                            \n\t"
            "gsldrc1    %[ftmp1],   0x18(%[tmp])                        \n\t"
            "pshufh     %[ftmp4],   %[ftmp4],       %[ftmp0]            \n\t"
            "punpcklbh  %[ftmp2],   %[ftmp1],       %[ftmp0]            \n\t"
            "punpckhbh  %[ftmp3],   %[ftmp1],       %[ftmp0]            \n\t"
            "pmullh     %[ftmp2],   %[ftmp2],       %[ftmp4]            \n\t"
            "pmullh     %[ftmp3],   %[ftmp3],       %[ftmp4]            \n\t"
            "paddush    %[ftmp5],   %[ftmp5],       %[ftmp2]            \n\t"
            "paddush    %[ftmp6],   %[ftmp6],       %[ftmp3]            \n\t"

            "li         %[tmp0],    0x07                                \n\t"
            "paddush    %[ftmp5],   %[ftmp5],       %[ff_pw_64]         \n\t"
            "paddush    %[ftmp6],   %[ftmp6],       %[ff_pw_64]         \n\t"
            "mtc1       %[tmp0],    %[ftmp4]                            \n\t"
            "psrlh      %[ftmp5],   %[ftmp5],       %[ftmp4]            \n\t"
            "psrlh      %[ftmp6],   %[ftmp6],       %[ftmp4]            \n\t"

            "packushb   %[ftmp1],   %[ftmp5],       %[ftmp6]            \n\t"
            "gssdlc1    %[ftmp1],   0x07(%[dst])                        \n\t"
            "gssdrc1    %[ftmp1],   0x00(%[dst])                        \n\t"
            : [ftmp0]"=&f"(ftmp[0]),        [ftmp1]"=&f"(ftmp[1]),
              [ftmp2]"=&f"(ftmp[2]),        [ftmp3]"=&f"(ftmp[3]),
              [ftmp4]"=&f"(ftmp[4]),        [ftmp5]"=&f"(ftmp[5]),
              [ftmp6]"=&f"(ftmp[6]),
              [tmp0]"=&r"(tmp0)
            : [dst]"r"(dst),                [tmp]"r"(tmp),
              [ff_pw_64]"f"(ff_pw_64),
              [filter0]"r"(filter[0]),      [filter1]"r"(filter[1]),
              [filter2]"r"(filter[2]),      [filter3]"r"(filter[3]),
              [filter4]"r"(filter[4]),      [filter5]"r"(filter[5])
            : "memory"
        );
        dst += dststride;
        tmp += 8;
    }
#else
    for (y = 0; y < h; y++) {
        for (x = 0; x < 8; x++)
            dst[x] = FILTER_6TAP(tmp, filter, 8);
        dst += dststride;
        tmp += 8;
    }
#endif
}

void ff_put_vp8_epel4_h6v6_mmi(uint8_t *dst, ptrdiff_t dststride, uint8_t *src,
        ptrdiff_t srcstride, int h, int mx, int my)
{
#if NOTOK //FIXME fate-vp8-test-vector-002 006 009
    const uint8_t *filter = subpel_filters[mx - 1];
    int y;
    uint8_t tmp_array[52];
    uint8_t *tmp = tmp_array;
    double ftmp[5];
    uint64_t tmp0;

    src -= 2 * srcstride;

    for (y = 0; y < h + 5; y++) {
        /*
        tmp[0] = cm[(filter[2]*src[0] - filter[1]*src[-1] + filter[0]*src[-2] + filter[3]*src[1] - filter[4]*src[2] + filter[5]*src[ 3] + 64) >> 7];
        tmp[1] = cm[(filter[2]*src[1] - filter[1]*src[ 0] + filter[0]*src[-1] + filter[3]*src[2] - filter[4]*src[3] + filter[5]*src[ 4] + 64) >> 7];
        tmp[2] = cm[(filter[2]*src[2] - filter[1]*src[ 1] + filter[0]*src[ 0] + filter[3]*src[3] - filter[4]*src[4] + filter[5]*src[ 5] + 64) >> 7];
        tmp[3] = cm[(filter[2]*src[3] - filter[1]*src[ 2] + filter[0]*src[ 1] + filter[3]*src[4] - filter[4]*src[5] + filter[5]*src[ 6] + 64) >> 7];
        */
        __asm__ volatile (
            "xor        %[ftmp0],   %[ftmp0],       %[ftmp0]            \n\t"
            "gslwlc1    %[ftmp1],   0x03(%[src])                        \n\t"
            "mtc1       %[filter2], %[ftmp3]                            \n\t"
            "gslwrc1    %[ftmp1],   0x00(%[src])                        \n\t"
            "pshufh     %[ftmp3],   %[ftmp3],       %[ftmp0]            \n\t"
            "punpcklbh  %[ftmp2],   %[ftmp1],       %[ftmp0]            \n\t"
            "pmullh     %[ftmp4],   %[ftmp2],       %[ftmp3]            \n\t"

            "gslwlc1    %[ftmp1],   0x02(%[src])                        \n\t"
            "mtc1       %[filter1], %[ftmp3]                            \n\t"
            "gslwrc1    %[ftmp1],  -0x01(%[src])                        \n\t"
            "pshufh     %[ftmp3],   %[ftmp3],       %[ftmp0]            \n\t"
            "punpcklbh  %[ftmp2],   %[ftmp1],       %[ftmp0]            \n\t"
            "pmullh     %[ftmp2],   %[ftmp2],       %[ftmp3]            \n\t"
            "psubush    %[ftmp4],   %[ftmp4],       %[ftmp2]            \n\t"

            "gslwlc1    %[ftmp1],   0x01(%[src])                        \n\t"
            "mtc1       %[filter0], %[ftmp3]                            \n\t"
            "gslwrc1    %[ftmp1],  -0x02(%[src])                        \n\t"
            "pshufh     %[ftmp3],   %[ftmp3],       %[ftmp0]            \n\t"
            "punpcklbh  %[ftmp2],   %[ftmp1],       %[ftmp0]            \n\t"
            "pmullh     %[ftmp2],   %[ftmp2],       %[ftmp3]            \n\t"
            "paddush    %[ftmp4],   %[ftmp4],       %[ftmp2]            \n\t"

            "gslwlc1    %[ftmp1],   0x04(%[src])                        \n\t"
            "mtc1       %[filter3], %[ftmp3]                            \n\t"
            "gslwrc1    %[ftmp1],   0x01(%[src])                        \n\t"
            "pshufh     %[ftmp3],   %[ftmp3],       %[ftmp0]            \n\t"
            "punpcklbh  %[ftmp2],   %[ftmp1],       %[ftmp0]            \n\t"
            "pmullh     %[ftmp2],   %[ftmp2],       %[ftmp3]            \n\t"
            "paddush    %[ftmp4],   %[ftmp4],       %[ftmp2]            \n\t"

            "gslwlc1    %[ftmp1],   0x05(%[src])                        \n\t"
            "mtc1       %[filter4], %[ftmp3]                            \n\t"
            "gslwrc1    %[ftmp1],   0x02(%[src])                        \n\t"
            "pshufh     %[ftmp3],   %[ftmp3],       %[ftmp0]            \n\t"
            "punpcklbh  %[ftmp2],   %[ftmp1],       %[ftmp0]            \n\t"
            "pmullh     %[ftmp2],   %[ftmp2],       %[ftmp3]            \n\t"
            "psubush    %[ftmp4],   %[ftmp4],       %[ftmp2]            \n\t"

            "gslwlc1    %[ftmp1],   0x06(%[src])                        \n\t"
            "mtc1       %[filter5], %[ftmp3]                            \n\t"
            "gslwrc1    %[ftmp1],   0x03(%[src])                        \n\t"
            "pshufh     %[ftmp3],   %[ftmp3],       %[ftmp0]            \n\t"
            "punpcklbh  %[ftmp2],   %[ftmp1],       %[ftmp0]            \n\t"
            "pmullh     %[ftmp2],   %[ftmp2],       %[ftmp3]            \n\t"
            "paddush    %[ftmp4],   %[ftmp4],       %[ftmp2]            \n\t"

            "li         %[tmp0],    0x07                                \n\t"
            "paddush    %[ftmp4],   %[ftmp4],       %[ff_pw_64]         \n\t"
            "mtc1       %[tmp0],    %[ftmp3]                            \n\t"
            "psrlh      %[ftmp4],   %[ftmp4],       %[ftmp3]            \n\t"

            "packushb   %[ftmp1],   %[ftmp4],       %[ftmp0]            \n\t"
            "gsswlc1    %[ftmp1],   0x03(%[tmp])                        \n\t"
            "gsswrc1    %[ftmp1],   0x00(%[tmp])                        \n\t"
            : [ftmp0]"=&f"(ftmp[0]),        [ftmp1]"=&f"(ftmp[1]),
              [ftmp2]"=&f"(ftmp[2]),        [ftmp3]"=&f"(ftmp[3]),
              [ftmp4]"=&f"(ftmp[4]),
              [tmp0]"=&r"(tmp0)
            : [tmp]"r"(tmp),                [src]"r"(src),
              [ff_pw_64]"f"(ff_pw_64),
              [filter0]"r"(filter[0]),      [filter1]"r"(filter[1]),
              [filter2]"r"(filter[2]),      [filter3]"r"(filter[3]),
              [filter4]"r"(filter[4]),      [filter5]"r"(filter[5])
            : "memory"
        );

        tmp += 4;
        src += srcstride;
    }
#else
    const uint8_t *filter = subpel_filters[mx - 1];
    const uint8_t *cm     = ff_crop_tab + MAX_NEG_CROP;
    int x, y;
    uint8_t tmp_array[52];
    uint8_t *tmp = tmp_array;

    src -= 2 * srcstride;

    for (y = 0; y < h + 5; y++) {
        for (x = 0; x < 4; x++)
            tmp[x] = FILTER_6TAP(src, filter, 1);
        tmp += 4;
        src += srcstride;
    }
#endif

    tmp    = tmp_array + 8;
    filter = subpel_filters[my - 1];

#if NOTOK //FIXME fate-vp8-test-vector-006
    for (y = 0; y < h; y++) {
        /*
        dst[0] = cm[(filter[2]*tmp[0] - filter[1]*tmp[-4] + filter[0]*tmp[-8] + filter[3]*tmp[4] - filter[4]*tmp[ 8] + filter[5]*tmp[12] + 64) >> 7];
        dst[1] = cm[(filter[2]*tmp[1] - filter[1]*tmp[-3] + filter[0]*tmp[-7] + filter[3]*tmp[5] - filter[4]*tmp[ 9] + filter[5]*tmp[13] + 64) >> 7];
        dst[2] = cm[(filter[2]*tmp[2] - filter[1]*tmp[-2] + filter[0]*tmp[-6] + filter[3]*tmp[6] - filter[4]*tmp[10] + filter[5]*tmp[14] + 64) >> 7];
        dst[3] = cm[(filter[2]*tmp[3] - filter[1]*tmp[-1] + filter[0]*tmp[-5] + filter[3]*tmp[7] - filter[4]*tmp[11] + filter[5]*tmp[15] + 64) >> 7];
        */
        __asm__ volatile (
            "xor        %[ftmp0],   %[ftmp0],       %[ftmp0]            \n\t"
            "gslwlc1    %[ftmp1],   0x03(%[tmp])                        \n\t"
            "mtc1       %[filter2], %[ftmp3]                            \n\t"
            "gslwrc1    %[ftmp1],   0x00(%[tmp])                        \n\t"
            "pshufh     %[ftmp3],   %[ftmp3],       %[ftmp0]            \n\t"
            "punpcklbh  %[ftmp2],   %[ftmp1],       %[ftmp0]            \n\t"
            "pmullh     %[ftmp4],   %[ftmp2],       %[ftmp3]            \n\t"

            "gslwlc1    %[ftmp1],  -0x01(%[tmp])                        \n\t"
            "mtc1       %[filter1], %[ftmp3]                            \n\t"
            "gslwrc1    %[ftmp1],  -0x04(%[tmp])                        \n\t"
            "pshufh     %[ftmp3],   %[ftmp3],       %[ftmp0]            \n\t"
            "punpcklbh  %[ftmp2],   %[ftmp1],       %[ftmp0]            \n\t"
            "pmullh     %[ftmp2],   %[ftmp2],       %[ftmp3]            \n\t"
            "psubush    %[ftmp4],   %[ftmp4],       %[ftmp2]            \n\t"

            "gslwlc1    %[ftmp1],  -0x05(%[tmp])                        \n\t"
            "mtc1       %[filter0], %[ftmp3]                            \n\t"
            "gslwrc1    %[ftmp1],  -0x08(%[tmp])                        \n\t"
            "pshufh     %[ftmp3],   %[ftmp3],       %[ftmp0]            \n\t"
            "punpcklbh  %[ftmp2],   %[ftmp1],       %[ftmp0]            \n\t"
            "pmullh     %[ftmp2],   %[ftmp2],       %[ftmp3]            \n\t"
            "paddush    %[ftmp4],   %[ftmp4],       %[ftmp2]            \n\t"

            "gslwlc1    %[ftmp1],   0x07(%[tmp])                        \n\t"
            "mtc1       %[filter3], %[ftmp3]                            \n\t"
            "gslwrc1    %[ftmp1],   0x04(%[tmp])                        \n\t"
            "pshufh     %[ftmp3],   %[ftmp3],       %[ftmp0]            \n\t"
            "punpcklbh  %[ftmp2],   %[ftmp1],       %[ftmp0]            \n\t"
            "pmullh     %[ftmp2],   %[ftmp2],       %[ftmp3]            \n\t"
            "paddush    %[ftmp4],   %[ftmp4],       %[ftmp2]            \n\t"

            "gslwlc1    %[ftmp1],   0x0b(%[tmp])                        \n\t"
            "mtc1       %[filter4], %[ftmp3]                            \n\t"
            "gslwrc1    %[ftmp1],   0x08(%[tmp])                        \n\t"
            "pshufh     %[ftmp3],   %[ftmp3],       %[ftmp0]            \n\t"
            "punpcklbh  %[ftmp2],   %[ftmp1],       %[ftmp0]            \n\t"
            "pmullh     %[ftmp2],   %[ftmp2],       %[ftmp3]            \n\t"
            "psubush    %[ftmp4],   %[ftmp4],       %[ftmp2]            \n\t"

            "gslwlc1    %[ftmp1],   0x0f(%[tmp])                        \n\t"
            "mtc1       %[filter5], %[ftmp3]                            \n\t"
            "gslwrc1    %[ftmp1],   0x0c(%[tmp])                        \n\t"
            "pshufh     %[ftmp3],   %[ftmp3],       %[ftmp0]            \n\t"
            "punpcklbh  %[ftmp2],   %[ftmp1],       %[ftmp0]            \n\t"
            "pmullh     %[ftmp2],   %[ftmp2],       %[ftmp3]            \n\t"
            "paddush    %[ftmp4],   %[ftmp4],       %[ftmp2]            \n\t"

            "li         %[tmp0],    0x07                                \n\t"
            "paddush    %[ftmp4],   %[ftmp4],       %[ff_pw_64]         \n\t"
            "mtc1       %[tmp0],    %[ftmp3]                            \n\t"
            "psrlh      %[ftmp4],   %[ftmp4],       %[ftmp3]            \n\t"

            "packushb   %[ftmp1],   %[ftmp4],       %[ftmp0]            \n\t"
            "gsswlc1    %[ftmp1],   0x03(%[dst])                        \n\t"
            "gsswrc1    %[ftmp1],   0x00(%[dst])                        \n\t"
            : [ftmp0]"=&f"(ftmp[0]),        [ftmp1]"=&f"(ftmp[1]),
              [ftmp2]"=&f"(ftmp[2]),        [ftmp3]"=&f"(ftmp[3]),
              [ftmp4]"=&f"(ftmp[4]),
              [tmp0]"=&r"(tmp0)
            : [dst]"r"(dst),                [tmp]"r"(tmp),
              [ff_pw_64]"f"(ff_pw_64),
              [filter0]"r"(filter[0]),      [filter1]"r"(filter[1]),
              [filter2]"r"(filter[2]),      [filter3]"r"(filter[3]),
              [filter4]"r"(filter[4]),      [filter5]"r"(filter[5])
            : "memory"
        );

        dst += dststride;
        tmp += 4;
    }
#else
    for (y = 0; y < h; y++) {
        for (x = 0; x < 4; x++)
            dst[x] = FILTER_6TAP(tmp, filter, 4);
        dst += dststride;
        tmp += 4;
    }
#endif
}

void ff_put_vp8_bilinear16_h_mmi(uint8_t *dst, ptrdiff_t dstride, uint8_t *src,
        ptrdiff_t sstride, int h, int mx, int my)
{
#if OK
    int a = 8 - mx, b = mx;
    int y;
    double ftmp[15];

    for (y = 0; y < h; y++) {
        /*
        dst[0] = (a * src[0] + b * src[1] + 4) >> 3;
        dst[1] = (a * src[1] + b * src[2] + 4) >> 3;
        dst[2] = (a * src[2] + b * src[3] + 4) >> 3;
        dst[3] = (a * src[3] + b * src[4] + 4) >> 3;
        dst[4] = (a * src[4] + b * src[5] + 4) >> 3;
        dst[5] = (a * src[5] + b * src[6] + 4) >> 3;
        dst[6] = (a * src[6] + b * src[7] + 4) >> 3;
        dst[7] = (a * src[7] + b * src[8] + 4) >> 3;

        dst[ 8] = (a * src[ 8] + b * src[ 9] + 4) >> 3;
        dst[ 9] = (a * src[ 9] + b * src[10] + 4) >> 3;
        dst[10] = (a * src[10] + b * src[11] + 4) >> 3;
        dst[11] = (a * src[11] + b * src[12] + 4) >> 3;
        dst[12] = (a * src[12] + b * src[13] + 4) >> 3;
        dst[13] = (a * src[13] + b * src[14] + 4) >> 3;
        dst[14] = (a * src[14] + b * src[15] + 4) >> 3;
        dst[15] = (a * src[15] + b * src[16] + 4) >> 3;
        */
        __asm__ volatile (
            "xor        %[ftmp0],   %[ftmp0],       %[ftmp0]            \n\t"
            "gsldlc1    %[ftmp1],   0x07(%[src])                        \n\t"
            "gsldrc1    %[ftmp1],   0x00(%[src])                        \n\t"
            "gsldlc1    %[ftmp2],   0x0f(%[src])                        \n\t"
            "gsldrc1    %[ftmp2],   0x08(%[src])                        \n\t"
            "gsldlc1    %[ftmp3],   0x08(%[src])                        \n\t"
            "mtc1       %[a],       %[ftmp13]                           \n\t"
            "gsldrc1    %[ftmp3],   0x01(%[src])                        \n\t"
            "gsldlc1    %[ftmp4],   0x10(%[src])                        \n\t"
            "mtc1       %[b],       %[ftmp14]                           \n\t"
            "gsldrc1    %[ftmp4],   0x09(%[src])                        \n\t"
            "pshufh     %[ftmp13],  %[ftmp13],      %[ftmp0]            \n\t"
            "pshufh     %[ftmp14],  %[ftmp14],      %[ftmp0]            \n\t"
            "punpcklbh  %[ftmp5],   %[ftmp1],       %[ftmp0]            \n\t"
            "punpckhbh  %[ftmp6],   %[ftmp1],       %[ftmp0]            \n\t"
            "punpcklbh  %[ftmp7],   %[ftmp2],       %[ftmp0]            \n\t"
            "punpckhbh  %[ftmp8],   %[ftmp2],       %[ftmp0]            \n\t"
            "punpcklbh  %[ftmp9],   %[ftmp3],       %[ftmp0]            \n\t"
            "punpckhbh  %[ftmp10],  %[ftmp3],       %[ftmp0]            \n\t"
            "punpcklbh  %[ftmp11],  %[ftmp4],       %[ftmp0]            \n\t"
            "punpckhbh  %[ftmp12],  %[ftmp4],       %[ftmp0]            \n\t"
            "pmullh     %[ftmp5],   %[ftmp5],       %[ftmp13]           \n\t"
            "pmullh     %[ftmp6],   %[ftmp6],       %[ftmp13]           \n\t"
            "pmullh     %[ftmp7],   %[ftmp7],       %[ftmp13]           \n\t"
            "pmullh     %[ftmp8],   %[ftmp8],       %[ftmp13]           \n\t"
            "pmullh     %[ftmp9],   %[ftmp9],       %[ftmp14]           \n\t"
            "pmullh     %[ftmp10],  %[ftmp10],      %[ftmp14]           \n\t"
            "pmullh     %[ftmp11],  %[ftmp11],      %[ftmp14]           \n\t"
            "pmullh     %[ftmp12],  %[ftmp12],      %[ftmp14]           \n\t"
            "paddsh     %[ftmp5],   %[ftmp5],       %[ftmp9]            \n\t"
            "paddsh     %[ftmp6],   %[ftmp6],       %[ftmp10]           \n\t"
            "paddsh     %[ftmp7],   %[ftmp7],       %[ftmp11]           \n\t"
            "paddsh     %[ftmp8],   %[ftmp8],       %[ftmp12]           \n\t"
            "dmtc1      %[ff_pw_4], %[ftmp14]                           \n\t"
            "dmtc1      %[ff_pw_3], %[ftmp13]                           \n\t"
            "paddsh     %[ftmp5],   %[ftmp5],       %[ftmp14]           \n\t"
            "paddsh     %[ftmp6],   %[ftmp6],       %[ftmp14]           \n\t"
            "paddsh     %[ftmp7],   %[ftmp7],       %[ftmp14]           \n\t"
            "paddsh     %[ftmp8],   %[ftmp8],       %[ftmp14]           \n\t"
            "psrlh      %[ftmp5],   %[ftmp5],       %[ftmp13]           \n\t"
            "psrlh      %[ftmp6],   %[ftmp6],       %[ftmp13]           \n\t"
            "psrlh      %[ftmp7],   %[ftmp7],       %[ftmp13]           \n\t"
            "psrlh      %[ftmp8],   %[ftmp8],       %[ftmp13]           \n\t"
            "packushb   %[ftmp5],   %[ftmp5],       %[ftmp6]            \n\t"
            "packushb   %[ftmp7],   %[ftmp7],       %[ftmp8]            \n\t"
            "gssdlc1    %[ftmp5],   0x07(%[dst])                        \n\t"
            "gssdrc1    %[ftmp5],   0x00(%[dst])                        \n\t"
            "gssdlc1    %[ftmp7],   0x0f(%[dst])                        \n\t"
            "gssdrc1    %[ftmp7],   0x08(%[dst])                        \n\t"
            : [ftmp0]"=&f"(ftmp[0]),        [ftmp1]"=&f"(ftmp[1]),
              [ftmp2]"=&f"(ftmp[2]),        [ftmp3]"=&f"(ftmp[3]),
              [ftmp4]"=&f"(ftmp[4]),        [ftmp5]"=&f"(ftmp[5]),
              [ftmp6]"=&f"(ftmp[6]),        [ftmp7]"=&f"(ftmp[7]),
              [ftmp8]"=&f"(ftmp[8]),        [ftmp9]"=&f"(ftmp[9]),
              [ftmp10]"=&f"(ftmp[10]),      [ftmp11]"=&f"(ftmp[11]),
              [ftmp12]"=&f"(ftmp[12]),      [ftmp13]"=&f"(ftmp[13]),
              [ftmp14]"=&f"(ftmp[14]),
              [dst]"+&r"(dst)
            : [src]"r"(src),                [a]"r"(a),
              [b]"r"(b),
              [ff_pw_4]"r"(ff_pw_4),        [ff_pw_3]"r"(ff_pw_3)
            : "memory"
        );

        dst += dstride;
        src += sstride;
    }
#else
    int a = 8 - mx, b = mx;
    int x, y;

    for (y = 0; y < h; y++) {
        for (x = 0; x < 16; x++)
            dst[x] = (a * src[x] + b * src[x + 1] + 4) >> 3;
        dst += dstride;
        src += sstride;
    }
#endif
}

void ff_put_vp8_bilinear16_v_mmi(uint8_t *dst, ptrdiff_t dstride, uint8_t *src,
        ptrdiff_t sstride, int h, int mx, int my)
{
#if OK
    int c = 8 - my, d = my;
    int y;
    double ftmp[15];

    for (y = 0; y < h; y++) {
        /*
        dst[0] = (c * src[0] + d * src[    sstride] + 4) >> 3;
        dst[1] = (c * src[1] + d * src[1 + sstride] + 4) >> 3;
        dst[2] = (c * src[2] + d * src[2 + sstride] + 4) >> 3;
        dst[3] = (c * src[3] + d * src[3 + sstride] + 4) >> 3;
        dst[4] = (c * src[4] + d * src[4 + sstride] + 4) >> 3;
        dst[5] = (c * src[5] + d * src[5 + sstride] + 4) >> 3;
        dst[6] = (c * src[6] + d * src[6 + sstride] + 4) >> 3;
        dst[7] = (c * src[7] + d * src[7 + sstride] + 4) >> 3;

        dst[ 8] = (c * src[ 8] + d * src[ 8 + sstride] + 4) >> 3;
        dst[ 9] = (c * src[ 9] + d * src[ 9 + sstride] + 4) >> 3;
        dst[10] = (c * src[10] + d * src[10 + sstride] + 4) >> 3;
        dst[11] = (c * src[11] + d * src[11 + sstride] + 4) >> 3;
        dst[12] = (c * src[12] + d * src[12 + sstride] + 4) >> 3;
        dst[13] = (c * src[13] + d * src[13 + sstride] + 4) >> 3;
        dst[14] = (c * src[14] + d * src[14 + sstride] + 4) >> 3;
        dst[15] = (c * src[15] + d * src[15 + sstride] + 4) >> 3;
        */
        __asm__ volatile (
            "xor        %[ftmp0],   %[ftmp0],       %[ftmp0]            \n\t"
            "gsldlc1    %[ftmp1],   0x07(%[src1])                       \n\t"
            "gsldrc1    %[ftmp1],   0x00(%[src1])                       \n\t"
            "gsldlc1    %[ftmp2],   0x0f(%[src1])                       \n\t"
            "gsldrc1    %[ftmp2],   0x08(%[src1])                       \n\t"
            "gsldlc1    %[ftmp3],   0x07(%[src2])                       \n\t"
            "mtc1       %[c],       %[ftmp13]                           \n\t"
            "gsldrc1    %[ftmp3],   0x00(%[src2])                       \n\t"
            "gsldlc1    %[ftmp4],   0x0f(%[src2])                       \n\t"
            "mtc1       %[d],       %[ftmp14]                           \n\t"
            "gsldrc1    %[ftmp4],   0x08(%[src2])                       \n\t"
            "pshufh     %[ftmp13],  %[ftmp13],      %[ftmp0]            \n\t"
            "pshufh     %[ftmp14],  %[ftmp14],      %[ftmp0]            \n\t"
            "punpcklbh  %[ftmp5],   %[ftmp1],       %[ftmp0]            \n\t"
            "punpckhbh  %[ftmp6],   %[ftmp1],       %[ftmp0]            \n\t"
            "punpcklbh  %[ftmp7],   %[ftmp2],       %[ftmp0]            \n\t"
            "punpckhbh  %[ftmp8],   %[ftmp2],       %[ftmp0]            \n\t"
            "punpcklbh  %[ftmp9],   %[ftmp3],       %[ftmp0]            \n\t"
            "punpckhbh  %[ftmp10],  %[ftmp3],       %[ftmp0]            \n\t"
            "punpcklbh  %[ftmp11],  %[ftmp4],       %[ftmp0]            \n\t"
            "punpckhbh  %[ftmp12],  %[ftmp4],       %[ftmp0]            \n\t"
            "pmullh     %[ftmp5],   %[ftmp5],       %[ftmp13]           \n\t"
            "pmullh     %[ftmp6],   %[ftmp6],       %[ftmp13]           \n\t"
            "pmullh     %[ftmp7],   %[ftmp7],       %[ftmp13]           \n\t"
            "pmullh     %[ftmp8],   %[ftmp8],       %[ftmp13]           \n\t"
            "pmullh     %[ftmp9],   %[ftmp9],       %[ftmp14]           \n\t"
            "pmullh     %[ftmp10],  %[ftmp10],      %[ftmp14]           \n\t"
            "pmullh     %[ftmp11],  %[ftmp11],      %[ftmp14]           \n\t"
            "pmullh     %[ftmp12],  %[ftmp12],      %[ftmp14]           \n\t"
            "paddsh     %[ftmp5],   %[ftmp5],       %[ftmp9]            \n\t"
            "paddsh     %[ftmp6],   %[ftmp6],       %[ftmp10]           \n\t"
            "paddsh     %[ftmp7],   %[ftmp7],       %[ftmp11]           \n\t"
            "paddsh     %[ftmp8],   %[ftmp8],       %[ftmp12]           \n\t"
            "dmtc1      %[ff_pw_4], %[ftmp14]                           \n\t"
            "dmtc1      %[ff_pw_3], %[ftmp13]                           \n\t"
            "paddsh     %[ftmp5],   %[ftmp5],       %[ftmp14]          \n\t"
            "paddsh     %[ftmp6],   %[ftmp6],       %[ftmp14]          \n\t"
            "paddsh     %[ftmp7],   %[ftmp7],       %[ftmp14]          \n\t"
            "paddsh     %[ftmp8],   %[ftmp8],       %[ftmp14]          \n\t"
            "psrlh      %[ftmp5],   %[ftmp5],       %[ftmp13]          \n\t"
            "psrlh      %[ftmp6],   %[ftmp6],       %[ftmp13]          \n\t"
            "psrlh      %[ftmp7],   %[ftmp7],       %[ftmp13]          \n\t"
            "psrlh      %[ftmp8],   %[ftmp8],       %[ftmp13]          \n\t"
            "packushb   %[ftmp5],   %[ftmp5],       %[ftmp6]            \n\t"
            "packushb   %[ftmp7],   %[ftmp7],       %[ftmp8]            \n\t"
            "gssdlc1    %[ftmp5],   0x07(%[dst])                        \n\t"
            "gssdrc1    %[ftmp5],   0x00(%[dst])                        \n\t"
            "gssdlc1    %[ftmp7],   0x0f(%[dst])                        \n\t"
            "gssdrc1    %[ftmp7],   0x08(%[dst])                        \n\t"
            : [ftmp0]"=&f"(ftmp[0]),        [ftmp1]"=&f"(ftmp[1]),
              [ftmp2]"=&f"(ftmp[2]),        [ftmp3]"=&f"(ftmp[3]),
              [ftmp4]"=&f"(ftmp[4]),        [ftmp5]"=&f"(ftmp[5]),
              [ftmp6]"=&f"(ftmp[6]),        [ftmp7]"=&f"(ftmp[7]),
              [ftmp8]"=&f"(ftmp[8]),        [ftmp9]"=&f"(ftmp[9]),
              [ftmp10]"=&f"(ftmp[10]),      [ftmp11]"=&f"(ftmp[11]),
              [ftmp12]"=&f"(ftmp[12]),      [ftmp13]"=&f"(ftmp[13]),
              [ftmp14]"=&f"(ftmp[14]),
              [dst]"+&r"(dst)
            : [src1]"r"(src),               [src2]"r"(src+sstride),
              [c]"r"(c),                    [d]"r"(d),
              [ff_pw_4]"r"(ff_pw_4),        [ff_pw_3]"r"(ff_pw_3)
            : "memory"
        );

        dst += dstride;
        src += sstride;
    }
#else
    int c = 8 - my, d = my;
    int x, y;

    for (y = 0; y < h; y++) {
        for (x = 0; x < 16; x++)
            dst[x] = (c * src[x] + d * src[x + sstride] + 4) >> 3;
        dst += dstride;
        src += sstride;
    }
#endif
}

void ff_put_vp8_bilinear16_hv_mmi(uint8_t *dst, ptrdiff_t dstride, uint8_t *src,
        ptrdiff_t sstride, int h, int mx, int my)
{
#if OK
    int a = 8 - mx, b = mx;
    int c = 8 - my, d = my;
    int y;
    uint8_t tmp_array[528];
    uint8_t *tmp = tmp_array;
    double ftmp[15];

    for (y = 0; y < h + 1; y++) {
        /*
        tmp[0] = (a * src[0] + b * src[1] + 4) >> 3;
        tmp[1] = (a * src[1] + b * src[2] + 4) >> 3;
        tmp[2] = (a * src[2] + b * src[3] + 4) >> 3;
        tmp[3] = (a * src[3] + b * src[4] + 4) >> 3;
        tmp[4] = (a * src[4] + b * src[5] + 4) >> 3;
        tmp[5] = (a * src[5] + b * src[6] + 4) >> 3;
        tmp[6] = (a * src[6] + b * src[7] + 4) >> 3;
        tmp[7] = (a * src[7] + b * src[8] + 4) >> 3;

        tmp[ 8] = (a * src[ 8] + b * src[ 9] + 4) >> 3;
        tmp[ 9] = (a * src[ 9] + b * src[10] + 4) >> 3;
        tmp[10] = (a * src[10] + b * src[11] + 4) >> 3;
        tmp[11] = (a * src[11] + b * src[12] + 4) >> 3;
        tmp[12] = (a * src[12] + b * src[13] + 4) >> 3;
        tmp[13] = (a * src[13] + b * src[14] + 4) >> 3;
        tmp[14] = (a * src[14] + b * src[15] + 4) >> 3;
        tmp[15] = (a * src[15] + b * src[16] + 4) >> 3;
        */
        __asm__ volatile (
            "xor        %[ftmp0],   %[ftmp0],       %[ftmp0]            \n\t"
            "gsldlc1    %[ftmp1],   0x07(%[src])                        \n\t"
            "gsldrc1    %[ftmp1],   0x00(%[src])                        \n\t"
            "gsldlc1    %[ftmp2],   0x0f(%[src])                        \n\t"
            "gsldrc1    %[ftmp2],   0x08(%[src])                        \n\t"
            "gsldlc1    %[ftmp3],   0x08(%[src])                        \n\t"
            "mtc1       %[a],       %[ftmp13]                           \n\t"
            "gsldrc1    %[ftmp3],   0x01(%[src])                        \n\t"
            "gsldlc1    %[ftmp4],   0x10(%[src])                        \n\t"
            "mtc1       %[b],       %[ftmp14]                           \n\t"
            "gsldrc1    %[ftmp4],   0x09(%[src])                        \n\t"
            "pshufh     %[ftmp13],  %[ftmp13],      %[ftmp0]            \n\t"
            "pshufh     %[ftmp14],  %[ftmp14],      %[ftmp0]            \n\t"
            "punpcklbh  %[ftmp5],   %[ftmp1],       %[ftmp0]            \n\t"
            "punpckhbh  %[ftmp6],   %[ftmp1],       %[ftmp0]            \n\t"
            "punpcklbh  %[ftmp7],   %[ftmp2],       %[ftmp0]            \n\t"
            "punpckhbh  %[ftmp8],   %[ftmp2],       %[ftmp0]            \n\t"
            "punpcklbh  %[ftmp9],   %[ftmp3],       %[ftmp0]            \n\t"
            "punpckhbh  %[ftmp10],  %[ftmp3],       %[ftmp0]            \n\t"
            "punpcklbh  %[ftmp11],  %[ftmp4],       %[ftmp0]            \n\t"
            "punpckhbh  %[ftmp12],  %[ftmp4],       %[ftmp0]            \n\t"
            "pmullh     %[ftmp5],   %[ftmp5],       %[ftmp13]           \n\t"
            "pmullh     %[ftmp6],   %[ftmp6],       %[ftmp13]           \n\t"
            "pmullh     %[ftmp7],   %[ftmp7],       %[ftmp13]           \n\t"
            "pmullh     %[ftmp8],   %[ftmp8],       %[ftmp13]           \n\t"
            "pmullh     %[ftmp9],   %[ftmp9],       %[ftmp14]           \n\t"
            "pmullh     %[ftmp10],  %[ftmp10],      %[ftmp14]           \n\t"
            "pmullh     %[ftmp11],  %[ftmp11],      %[ftmp14]           \n\t"
            "pmullh     %[ftmp12],  %[ftmp12],      %[ftmp14]           \n\t"
            "paddsh     %[ftmp5],   %[ftmp5],       %[ftmp9]            \n\t"
            "paddsh     %[ftmp6],   %[ftmp6],       %[ftmp10]           \n\t"
            "paddsh     %[ftmp7],   %[ftmp7],       %[ftmp11]           \n\t"
            "paddsh     %[ftmp8],   %[ftmp8],       %[ftmp12]           \n\t"
            "dmtc1      %[ff_pw_4], %[ftmp14]                           \n\t"
            "dmtc1      %[ff_pw_3], %[ftmp13]                           \n\t"
            "paddsh     %[ftmp5],   %[ftmp5],       %[ftmp14]           \n\t"
            "paddsh     %[ftmp6],   %[ftmp6],       %[ftmp14]           \n\t"
            "paddsh     %[ftmp7],   %[ftmp7],       %[ftmp14]           \n\t"
            "paddsh     %[ftmp8],   %[ftmp8],       %[ftmp14]           \n\t"
            "psrlh      %[ftmp5],   %[ftmp5],       %[ftmp13]           \n\t"
            "psrlh      %[ftmp6],   %[ftmp6],       %[ftmp13]           \n\t"
            "psrlh      %[ftmp7],   %[ftmp7],       %[ftmp13]           \n\t"
            "psrlh      %[ftmp8],   %[ftmp8],       %[ftmp13]           \n\t"
            "packushb   %[ftmp5],   %[ftmp5],       %[ftmp6]            \n\t"
            "packushb   %[ftmp7],   %[ftmp7],       %[ftmp8]            \n\t"
            "gssdlc1    %[ftmp5],   0x07(%[tmp])                        \n\t"
            "gssdrc1    %[ftmp5],   0x00(%[tmp])                        \n\t"
            "gssdlc1    %[ftmp7],   0x0f(%[tmp])                        \n\t"
            "gssdrc1    %[ftmp7],   0x08(%[tmp])                        \n\t"
            : [ftmp0]"=&f"(ftmp[0]),        [ftmp1]"=&f"(ftmp[1]),
              [ftmp2]"=&f"(ftmp[2]),        [ftmp3]"=&f"(ftmp[3]),
              [ftmp4]"=&f"(ftmp[4]),        [ftmp5]"=&f"(ftmp[5]),
              [ftmp6]"=&f"(ftmp[6]),        [ftmp7]"=&f"(ftmp[7]),
              [ftmp8]"=&f"(ftmp[8]),        [ftmp9]"=&f"(ftmp[9]),
              [ftmp10]"=&f"(ftmp[10]),      [ftmp11]"=&f"(ftmp[11]),
              [ftmp12]"=&f"(ftmp[12]),      [ftmp13]"=&f"(ftmp[13]),
              [ftmp14]"=&f"(ftmp[14]),
              [tmp]"+&r"(tmp)
            : [src]"r"(src),                [a]"r"(a),
              [b]"r"(b),
              [ff_pw_4]"r"(ff_pw_4),        [ff_pw_3]"r"(ff_pw_3)
            : "memory"
        );

        tmp += 16;
        src += sstride;
    }
#else
    int a = 8 - mx, b = mx;
    int c = 8 - my, d = my;
    int x, y;
    uint8_t tmp_array[528];
    uint8_t *tmp = tmp_array;

    for (y = 0; y < h + 1; y++) {
        for (x = 0; x < 16; x++)
            tmp[x] = (a * src[x] + b * src[x + 1] + 4) >> 3;
        tmp += 16;
        src += sstride;
    }
#endif

    tmp = tmp_array;

#if OK
    for (y = 0; y < h; y++) {
        /*
        dst[0] = (c * tmp[0] + d * tmp[16] + 4) >> 3;
        dst[1] = (c * tmp[1] + d * tmp[17] + 4) >> 3;
        dst[2] = (c * tmp[2] + d * tmp[18] + 4) >> 3;
        dst[3] = (c * tmp[3] + d * tmp[19] + 4) >> 3;
        dst[4] = (c * tmp[4] + d * tmp[20] + 4) >> 3;
        dst[5] = (c * tmp[5] + d * tmp[21] + 4) >> 3;
        dst[6] = (c * tmp[6] + d * tmp[22] + 4) >> 3;
        dst[7] = (c * tmp[7] + d * tmp[23] + 4) >> 3;

        dst[ 8] = (c * tmp[ 8] + d * tmp[24] + 4) >> 3;
        dst[ 9] = (c * tmp[ 9] + d * tmp[25] + 4) >> 3;
        dst[10] = (c * tmp[10] + d * tmp[26] + 4) >> 3;
        dst[11] = (c * tmp[11] + d * tmp[27] + 4) >> 3;
        dst[12] = (c * tmp[12] + d * tmp[28] + 4) >> 3;
        dst[13] = (c * tmp[13] + d * tmp[29] + 4) >> 3;
        dst[14] = (c * tmp[14] + d * tmp[30] + 4) >> 3;
        dst[15] = (c * tmp[15] + d * tmp[31] + 4) >> 3;
        */
        __asm__ volatile (
            "xor        %[ftmp0],   %[ftmp0],       %[ftmp0]            \n\t"
            "gsldlc1    %[ftmp1],   0x07(%[tmp])                        \n\t"
            "gsldrc1    %[ftmp1],   0x00(%[tmp])                        \n\t"
            "gsldlc1    %[ftmp2],   0x0f(%[tmp])                        \n\t"
            "gsldrc1    %[ftmp2],   0x08(%[tmp])                        \n\t"
            "gsldlc1    %[ftmp3],   0x17(%[tmp])                        \n\t"
            "mtc1       %[c],       %[ftmp13]                           \n\t"
            "gsldrc1    %[ftmp3],   0x10(%[tmp])                        \n\t"
            "gsldlc1    %[ftmp4],   0x1f(%[tmp])                        \n\t"
            "mtc1       %[d],       %[ftmp14]                           \n\t"
            "gsldrc1    %[ftmp4],   0x18(%[tmp])                        \n\t"
            "pshufh     %[ftmp13],  %[ftmp13],      %[ftmp0]            \n\t"
            "pshufh     %[ftmp14],  %[ftmp14],      %[ftmp0]            \n\t"
            "punpcklbh  %[ftmp5],   %[ftmp1],       %[ftmp0]            \n\t"
            "punpckhbh  %[ftmp6],   %[ftmp1],       %[ftmp0]            \n\t"
            "punpcklbh  %[ftmp7],   %[ftmp2],       %[ftmp0]            \n\t"
            "punpckhbh  %[ftmp8],   %[ftmp2],       %[ftmp0]            \n\t"
            "punpcklbh  %[ftmp9],   %[ftmp3],       %[ftmp0]            \n\t"
            "punpckhbh  %[ftmp10],  %[ftmp3],       %[ftmp0]            \n\t"
            "punpcklbh  %[ftmp11],  %[ftmp4],       %[ftmp0]            \n\t"
            "punpckhbh  %[ftmp12],  %[ftmp4],       %[ftmp0]            \n\t"
            "pmullh     %[ftmp5],   %[ftmp5],       %[ftmp13]           \n\t"
            "pmullh     %[ftmp6],   %[ftmp6],       %[ftmp13]           \n\t"
            "pmullh     %[ftmp7],   %[ftmp7],       %[ftmp13]           \n\t"
            "pmullh     %[ftmp8],   %[ftmp8],       %[ftmp13]           \n\t"
            "pmullh     %[ftmp9],   %[ftmp9],       %[ftmp14]           \n\t"
            "pmullh     %[ftmp10],  %[ftmp10],      %[ftmp14]           \n\t"
            "pmullh     %[ftmp11],  %[ftmp11],      %[ftmp14]           \n\t"
            "pmullh     %[ftmp12],  %[ftmp12],      %[ftmp14]           \n\t"
            "paddsh     %[ftmp5],   %[ftmp5],       %[ftmp9]            \n\t"
            "paddsh     %[ftmp6],   %[ftmp6],       %[ftmp10]           \n\t"
            "paddsh     %[ftmp7],   %[ftmp7],       %[ftmp11]           \n\t"
            "paddsh     %[ftmp8],   %[ftmp8],       %[ftmp12]           \n\t"
            "dmtc1      %[ff_pw_4], %[ftmp14]                           \n\t"
            "dmtc1      %[ff_pw_3], %[ftmp13]                           \n\t"
            "paddsh     %[ftmp5],   %[ftmp5],       %[ftmp14]           \n\t"
            "paddsh     %[ftmp6],   %[ftmp6],       %[ftmp14]           \n\t"
            "paddsh     %[ftmp7],   %[ftmp7],       %[ftmp14]           \n\t"
            "paddsh     %[ftmp8],   %[ftmp8],       %[ftmp14]           \n\t"
            "psrlh      %[ftmp5],   %[ftmp5],       %[ftmp13]           \n\t"
            "psrlh      %[ftmp6],   %[ftmp6],       %[ftmp13]           \n\t"
            "psrlh      %[ftmp7],   %[ftmp7],       %[ftmp13]           \n\t"
            "psrlh      %[ftmp8],   %[ftmp8],       %[ftmp13]           \n\t"
            "packushb   %[ftmp5],   %[ftmp5],       %[ftmp6]            \n\t"
            "packushb   %[ftmp7],   %[ftmp7],       %[ftmp8]            \n\t"
            "gssdlc1    %[ftmp5],   0x07(%[dst])                        \n\t"
            "gssdrc1    %[ftmp5],   0x00(%[dst])                        \n\t"
            "gssdlc1    %[ftmp7],   0x0f(%[dst])                        \n\t"
            "gssdrc1    %[ftmp7],   0x08(%[dst])                        \n\t"
            : [ftmp0]"=&f"(ftmp[0]),        [ftmp1]"=&f"(ftmp[1]),
              [ftmp2]"=&f"(ftmp[2]),        [ftmp3]"=&f"(ftmp[3]),
              [ftmp4]"=&f"(ftmp[4]),        [ftmp5]"=&f"(ftmp[5]),
              [ftmp6]"=&f"(ftmp[6]),        [ftmp7]"=&f"(ftmp[7]),
              [ftmp8]"=&f"(ftmp[8]),        [ftmp9]"=&f"(ftmp[9]),
              [ftmp10]"=&f"(ftmp[10]),      [ftmp11]"=&f"(ftmp[11]),
              [ftmp12]"=&f"(ftmp[12]),      [ftmp13]"=&f"(ftmp[13]),
              [ftmp14]"=&f"(ftmp[14]),
              [dst]"+&r"(dst)
            : [tmp]"r"(tmp),                [c]"r"(c),
              [d]"r"(d),
              [ff_pw_4]"r"(ff_pw_4),        [ff_pw_3]"r"(ff_pw_3)
            : "memory"
        );

        dst += dstride;
        tmp += 16;
    }
#else
    for (y = 0; y < h; y++) {
        for (x = 0; x < 16; x++)
            dst[x] = (c * tmp[x] + d * tmp[x + 16] + 4) >> 3;
        dst += dstride;
        tmp += 16;
    }
#endif
}

void ff_put_vp8_bilinear8_h_mmi(uint8_t *dst, ptrdiff_t dstride, uint8_t *src,
        ptrdiff_t sstride, int h, int mx, int my)
{
#if OK
    int a = 8 - mx, b = mx;
    int y;
    double ftmp[9];

    for (y = 0; y < h; y++) {
        /*
        dst[0] = (a * src[0] + b * src[1] + 4) >> 3;
        dst[1] = (a * src[1] + b * src[2] + 4) >> 3;
        dst[2] = (a * src[2] + b * src[3] + 4) >> 3;
        dst[3] = (a * src[3] + b * src[4] + 4) >> 3;
        dst[4] = (a * src[4] + b * src[5] + 4) >> 3;
        dst[5] = (a * src[5] + b * src[6] + 4) >> 3;
        dst[6] = (a * src[6] + b * src[7] + 4) >> 3;
        dst[7] = (a * src[7] + b * src[8] + 4) >> 3;
        */
        __asm__ volatile (
            "xor        %[ftmp0],   %[ftmp0],       %[ftmp0]            \n\t"
            "gsldlc1    %[ftmp1],   0x07(%[src])                        \n\t"
            "mtc1       %[a],       %[ftmp3]                            \n\t"
            "gsldrc1    %[ftmp1],   0x00(%[src])                        \n\t"
            "gsldlc1    %[ftmp2],   0x08(%[src])                        \n\t"
            "mtc1       %[b],       %[ftmp4]                            \n\t"
            "gsldrc1    %[ftmp2],   0x01(%[src])                        \n\t"
            "pshufh     %[ftmp3],   %[ftmp3],       %[ftmp0]            \n\t"
            "pshufh     %[ftmp4],   %[ftmp4],       %[ftmp0]            \n\t"
            "punpcklbh  %[ftmp5],   %[ftmp1],       %[ftmp0]            \n\t"
            "punpckhbh  %[ftmp6],   %[ftmp1],       %[ftmp0]            \n\t"
            "punpcklbh  %[ftmp7],   %[ftmp2],       %[ftmp0]            \n\t"
            "punpckhbh  %[ftmp8],   %[ftmp2],       %[ftmp0]            \n\t"
            "pmullh     %[ftmp5],   %[ftmp5],       %[ftmp3]            \n\t"
            "pmullh     %[ftmp6],   %[ftmp6],       %[ftmp3]            \n\t"
            "pmullh     %[ftmp7],   %[ftmp7],       %[ftmp4]            \n\t"
            "pmullh     %[ftmp8],   %[ftmp8],       %[ftmp4]            \n\t"
            "paddsh     %[ftmp5],   %[ftmp5],       %[ftmp7]            \n\t"
            "paddsh     %[ftmp6],   %[ftmp6],       %[ftmp8]            \n\t"
            "paddsh     %[ftmp5],   %[ftmp5],       %[ff_pw_4]          \n\t"
            "paddsh     %[ftmp6],   %[ftmp6],       %[ff_pw_4]          \n\t"
            "psrlh      %[ftmp5],   %[ftmp5],       %[ff_pw_3]          \n\t"
            "psrlh      %[ftmp6],   %[ftmp6],       %[ff_pw_3]          \n\t"
            "packushb   %[ftmp5],   %[ftmp5],       %[ftmp6]            \n\t"
            "gssdlc1    %[ftmp5],   0x07(%[dst])                        \n\t"
            "gssdrc1    %[ftmp5],   0x00(%[dst])                        \n\t"
            : [ftmp0]"=&f"(ftmp[0]),        [ftmp1]"=&f"(ftmp[1]),
              [ftmp2]"=&f"(ftmp[2]),        [ftmp3]"=&f"(ftmp[3]),
              [ftmp4]"=&f"(ftmp[4]),        [ftmp5]"=&f"(ftmp[5]),
              [ftmp6]"=&f"(ftmp[6]),        [ftmp7]"=&f"(ftmp[7]),
              [ftmp8]"=&f"(ftmp[8]),
              [dst]"+&r"(dst)
            : [src]"r"(src),                [a]"r"(a),
              [b]"r"(b),
              [ff_pw_4]"f"(ff_pw_4),        [ff_pw_3]"f"(ff_pw_3)
            : "memory"
        );

        dst += dstride;
        src += sstride;
    }
#else
    int a = 8 - mx, b = mx;
    int x, y;

    for (y = 0; y < h; y++) {
        for (x = 0; x < 8; x++)
            dst[x] = (a * src[x] + b * src[x + 1] + 4) >> 3;
        dst += dstride;
        src += sstride;
    }
#endif
}

void ff_put_vp8_bilinear8_v_mmi(uint8_t *dst, ptrdiff_t dstride, uint8_t *src,
        ptrdiff_t sstride, int h, int mx, int my)
{
#if OK
    int c = 8 - my, d = my;
    int y;
    double ftmp[9];

    for (y = 0; y < h; y++) {
        /*
        dst[0] = (c * src[0] + d * src[    sstride] + 4) >> 3;
        dst[1] = (c * src[1] + d * src[1 + sstride] + 4) >> 3;
        dst[2] = (c * src[2] + d * src[2 + sstride] + 4) >> 3;
        dst[3] = (c * src[3] + d * src[3 + sstride] + 4) >> 3;
        dst[4] = (c * src[4] + d * src[4 + sstride] + 4) >> 3;
        dst[5] = (c * src[5] + d * src[5 + sstride] + 4) >> 3;
        dst[6] = (c * src[6] + d * src[6 + sstride] + 4) >> 3;
        dst[7] = (c * src[7] + d * src[7 + sstride] + 4) >> 3;
        */
        __asm__ volatile (
            "xor        %[ftmp0],   %[ftmp0],       %[ftmp0]            \n\t"
            "gsldlc1    %[ftmp1],   0x07(%[src1])                       \n\t"
            "mtc1       %[c],       %[ftmp3]                            \n\t"
            "gsldrc1    %[ftmp1],   0x00(%[src1])                       \n\t"
            "gsldlc1    %[ftmp2],   0x07(%[src2])                       \n\t"
            "mtc1       %[d],       %[ftmp4]                            \n\t"
            "gsldrc1    %[ftmp2],   0x00(%[src2])                       \n\t"
            "pshufh     %[ftmp3],   %[ftmp3],       %[ftmp0]            \n\t"
            "pshufh     %[ftmp4],   %[ftmp4],       %[ftmp0]            \n\t"
            "punpcklbh  %[ftmp5],   %[ftmp1],       %[ftmp0]            \n\t"
            "punpckhbh  %[ftmp6],   %[ftmp1],       %[ftmp0]            \n\t"
            "punpcklbh  %[ftmp7],   %[ftmp2],       %[ftmp0]            \n\t"
            "punpckhbh  %[ftmp8],   %[ftmp2],       %[ftmp0]            \n\t"
            "pmullh     %[ftmp5],   %[ftmp5],       %[ftmp3]            \n\t"
            "pmullh     %[ftmp6],   %[ftmp6],       %[ftmp3]            \n\t"
            "pmullh     %[ftmp7],   %[ftmp7],       %[ftmp4]            \n\t"
            "pmullh     %[ftmp8],   %[ftmp8],       %[ftmp4]            \n\t"
            "paddsh     %[ftmp5],   %[ftmp5],       %[ftmp7]            \n\t"
            "paddsh     %[ftmp6],   %[ftmp6],       %[ftmp8]            \n\t"
            "paddsh     %[ftmp5],   %[ftmp5],       %[ff_pw_4]          \n\t"
            "paddsh     %[ftmp6],   %[ftmp6],       %[ff_pw_4]          \n\t"
            "psrlh      %[ftmp5],   %[ftmp5],       %[ff_pw_3]          \n\t"
            "psrlh      %[ftmp6],   %[ftmp6],       %[ff_pw_3]          \n\t"
            "packushb   %[ftmp5],   %[ftmp5],       %[ftmp6]            \n\t"
            "gssdlc1    %[ftmp5],   0x07(%[dst])                        \n\t"
            "gssdrc1    %[ftmp5],   0x00(%[dst])                        \n\t"
            : [ftmp0]"=&f"(ftmp[0]),        [ftmp1]"=&f"(ftmp[1]),
              [ftmp2]"=&f"(ftmp[2]),        [ftmp3]"=&f"(ftmp[3]),
              [ftmp4]"=&f"(ftmp[4]),        [ftmp5]"=&f"(ftmp[5]),
              [ftmp6]"=&f"(ftmp[6]),        [ftmp7]"=&f"(ftmp[7]),
              [ftmp8]"=&f"(ftmp[8]),
              [dst]"+&r"(dst)
            : [src1]"r"(src),               [src2]"r"(src+sstride),
              [c]"r"(c),                    [d]"r"(d),
              [ff_pw_4]"f"(ff_pw_4),        [ff_pw_3]"f"(ff_pw_3)
            : "memory"
        );

        dst += dstride;
        src += sstride;
    }
#else
    int c = 8 - my, d = my;
    int x, y;

    for (y = 0; y < h; y++) {
        for (x = 0; x < 8; x++)
            dst[x] = (c * src[x] + d * src[x + sstride] + 4) >> 3;
        dst += dstride;
        src += sstride;
    }
#endif
}

void ff_put_vp8_bilinear8_hv_mmi(uint8_t *dst, ptrdiff_t dstride, uint8_t *src,
        ptrdiff_t sstride, int h, int mx, int my)
{
#if OK
    int a = 8 - mx, b = mx;
    int c = 8 - my, d = my;
    int y;
    uint8_t tmp_array[136];
    uint8_t *tmp = tmp_array;
    double ftmp[9];

    for (y = 0; y < h + 1; y++) {
        /*
        tmp[0] = (a * src[0] + b * src[1] + 4) >> 3;
        tmp[1] = (a * src[1] + b * src[2] + 4) >> 3;
        tmp[2] = (a * src[2] + b * src[3] + 4) >> 3;
        tmp[3] = (a * src[3] + b * src[4] + 4) >> 3;
        tmp[4] = (a * src[4] + b * src[5] + 4) >> 3;
        tmp[5] = (a * src[5] + b * src[6] + 4) >> 3;
        tmp[6] = (a * src[6] + b * src[7] + 4) >> 3;
        tmp[7] = (a * src[7] + b * src[8] + 4) >> 3;
        */
        __asm__ volatile (
            "xor        %[ftmp0],   %[ftmp0],       %[ftmp0]            \n\t"
            "gsldlc1    %[ftmp1],   0x07(%[src])                        \n\t"
            "mtc1       %[a],       %[ftmp3]                            \n\t"
            "gsldrc1    %[ftmp1],   0x00(%[src])                        \n\t"
            "gsldlc1    %[ftmp2],   0x08(%[src])                        \n\t"
            "mtc1       %[b],       %[ftmp4]                            \n\t"
            "gsldrc1    %[ftmp2],   0x01(%[src])                        \n\t"
            "pshufh     %[ftmp3],   %[ftmp3],       %[ftmp0]            \n\t"
            "pshufh     %[ftmp4],   %[ftmp4],       %[ftmp0]            \n\t"
            "punpcklbh  %[ftmp5],   %[ftmp1],       %[ftmp0]            \n\t"
            "punpckhbh  %[ftmp6],   %[ftmp1],       %[ftmp0]            \n\t"
            "punpcklbh  %[ftmp7],   %[ftmp2],       %[ftmp0]            \n\t"
            "punpckhbh  %[ftmp8],   %[ftmp2],       %[ftmp0]            \n\t"
            "pmullh     %[ftmp5],   %[ftmp5],       %[ftmp3]            \n\t"
            "pmullh     %[ftmp6],   %[ftmp6],       %[ftmp3]            \n\t"
            "pmullh     %[ftmp7],   %[ftmp7],       %[ftmp4]            \n\t"
            "pmullh     %[ftmp8],   %[ftmp8],       %[ftmp4]            \n\t"
            "paddsh     %[ftmp5],   %[ftmp5],       %[ftmp7]            \n\t"
            "paddsh     %[ftmp6],   %[ftmp6],       %[ftmp8]            \n\t"
            "paddsh     %[ftmp5],   %[ftmp5],       %[ff_pw_4]          \n\t"
            "paddsh     %[ftmp6],   %[ftmp6],       %[ff_pw_4]          \n\t"
            "psrlh      %[ftmp5],   %[ftmp5],       %[ff_pw_3]          \n\t"
            "psrlh      %[ftmp6],   %[ftmp6],       %[ff_pw_3]          \n\t"
            "packushb   %[ftmp5],   %[ftmp5],       %[ftmp6]            \n\t"
            "gssdlc1    %[ftmp5],   0x07(%[tmp])                        \n\t"
            "gssdrc1    %[ftmp5],   0x00(%[tmp])                        \n\t"
            : [ftmp0]"=&f"(ftmp[0]),        [ftmp1]"=&f"(ftmp[1]),
              [ftmp2]"=&f"(ftmp[2]),        [ftmp3]"=&f"(ftmp[3]),
              [ftmp4]"=&f"(ftmp[4]),        [ftmp5]"=&f"(ftmp[5]),
              [ftmp6]"=&f"(ftmp[6]),        [ftmp7]"=&f"(ftmp[7]),
              [ftmp8]"=&f"(ftmp[8]),
              [tmp]"+&r"(tmp)
            : [src]"r"(src),                [a]"r"(a),
              [b]"r"(b),
              [ff_pw_4]"f"(ff_pw_4),        [ff_pw_3]"f"(ff_pw_3)
            : "memory"
        );
        tmp += 8;
        src += sstride;
    }
#else
    int a = 8 - mx, b = mx;
    int c = 8 - my, d = my;
    int x, y;
    uint8_t tmp_array[136];
    uint8_t *tmp = tmp_array;

    for (y = 0; y < h + 1; y++) {
        for (x = 0; x < 8; x++)
            tmp[x] = (a * src[x] + b * src[x + 1] + 4) >> 3;
        tmp += 8;
        src += sstride;
    }
#endif

    tmp = tmp_array;

#if OK
    for (y = 0; y < h; y++) {
        /*
        dst[0] = (c * tmp[0] + d * tmp[ 8] + 4) >> 3;
        dst[1] = (c * tmp[1] + d * tmp[ 9] + 4) >> 3;
        dst[2] = (c * tmp[2] + d * tmp[10] + 4) >> 3;
        dst[3] = (c * tmp[3] + d * tmp[11] + 4) >> 3;
        dst[4] = (c * tmp[4] + d * tmp[12] + 4) >> 3;
        dst[5] = (c * tmp[5] + d * tmp[13] + 4) >> 3;
        dst[6] = (c * tmp[6] + d * tmp[14] + 4) >> 3;
        dst[7] = (c * tmp[7] + d * tmp[15] + 4) >> 3;
        */
        __asm__ volatile (
            "xor        %[ftmp0],   %[ftmp0],       %[ftmp0]            \n\t"
            "gsldlc1    %[ftmp1],   0x07(%[tmp])                        \n\t"
            "mtc1       %[c],       %[ftmp3]                            \n\t"
            "gsldrc1    %[ftmp1],   0x00(%[tmp])                        \n\t"
            "gsldlc1    %[ftmp2],   0x0f(%[tmp])                        \n\t"
            "mtc1       %[d],       %[ftmp4]                            \n\t"
            "gsldrc1    %[ftmp2],   0x08(%[tmp])                        \n\t"
            "pshufh     %[ftmp3],   %[ftmp3],       %[ftmp0]            \n\t"
            "pshufh     %[ftmp4],   %[ftmp4],       %[ftmp0]            \n\t"
            "punpcklbh  %[ftmp5],   %[ftmp1],       %[ftmp0]            \n\t"
            "punpckhbh  %[ftmp6],   %[ftmp1],       %[ftmp0]            \n\t"
            "punpcklbh  %[ftmp7],   %[ftmp2],       %[ftmp0]            \n\t"
            "punpckhbh  %[ftmp8],   %[ftmp2],       %[ftmp0]            \n\t"
            "pmullh     %[ftmp5],   %[ftmp5],       %[ftmp3]            \n\t"
            "pmullh     %[ftmp6],   %[ftmp6],       %[ftmp3]            \n\t"
            "pmullh     %[ftmp7],   %[ftmp7],       %[ftmp4]            \n\t"
            "pmullh     %[ftmp8],   %[ftmp8],       %[ftmp4]            \n\t"
            "paddsh     %[ftmp5],   %[ftmp5],       %[ftmp7]            \n\t"
            "paddsh     %[ftmp6],   %[ftmp6],       %[ftmp8]            \n\t"
            "paddsh     %[ftmp5],   %[ftmp5],       %[ff_pw_4]          \n\t"
            "paddsh     %[ftmp6],   %[ftmp6],       %[ff_pw_4]          \n\t"
            "psrlh      %[ftmp5],   %[ftmp5],       %[ff_pw_3]          \n\t"
            "psrlh      %[ftmp6],   %[ftmp6],       %[ff_pw_3]          \n\t"
            "packushb   %[ftmp5],   %[ftmp5],       %[ftmp6]            \n\t"
            "gssdlc1    %[ftmp5],   0x07(%[dst])                        \n\t"
            "gssdrc1    %[ftmp5],   0x00(%[dst])                        \n\t"
            : [ftmp0]"=&f"(ftmp[0]),        [ftmp1]"=&f"(ftmp[1]),
              [ftmp2]"=&f"(ftmp[2]),        [ftmp3]"=&f"(ftmp[3]),
              [ftmp4]"=&f"(ftmp[4]),        [ftmp5]"=&f"(ftmp[5]),
              [ftmp6]"=&f"(ftmp[6]),        [ftmp7]"=&f"(ftmp[7]),
              [ftmp8]"=&f"(ftmp[8]),
              [dst]"+&r"(dst)
            : [tmp]"r"(tmp),                [c]"r"(c),
              [d]"r"(d),
              [ff_pw_4]"f"(ff_pw_4),        [ff_pw_3]"f"(ff_pw_3)
            : "memory"
        );

        dst += dstride;
        tmp += 8;
    }
#else
    for (y = 0; y < h; y++) {
        for (x = 0; x < 8; x++)
            dst[x] = (c * tmp[x] + d * tmp[x + 8] + 4) >> 3;
        dst += dstride;
        tmp += 8;
    }
#endif
}

//no test
void ff_put_vp8_bilinear4_h_mmi(uint8_t *dst, ptrdiff_t dstride, uint8_t *src,
        ptrdiff_t sstride, int h, int mx, int my)
{
#if OK
    int a = 8 - mx, b = mx;
    int y;
    double ftmp[5];

    for (y = 0; y < h; y++) {
        /*
        dst[0] = (a * src[0] + b * src[1] + 4) >> 3;
        dst[1] = (a * src[1] + b * src[2] + 4) >> 3;
        dst[2] = (a * src[2] + b * src[3] + 4) >> 3;
        dst[3] = (a * src[3] + b * src[4] + 4) >> 3;
        */
        __asm__ volatile (
            "xor        %[ftmp0],   %[ftmp0],       %[ftmp0]            \n\t"
            "gslwlc1    %[ftmp1],   0x03(%[src])                        \n\t"
            "mtc1       %[a],       %[ftmp3]                            \n\t"
            "gslwrc1    %[ftmp1],   0x00(%[src])                        \n\t"
            "gslwlc1    %[ftmp2],   0x04(%[src])                        \n\t"
            "mtc1       %[b],       %[ftmp4]                            \n\t"
            "gslwrc1    %[ftmp2],   0x01(%[src])                        \n\t"
            "pshufh     %[ftmp3],   %[ftmp3],       %[ftmp0]            \n\t"
            "pshufh     %[ftmp4],   %[ftmp4],       %[ftmp0]            \n\t"
            "punpcklbh  %[ftmp1],   %[ftmp1],       %[ftmp0]            \n\t"
            "punpcklbh  %[ftmp2],   %[ftmp2],       %[ftmp0]            \n\t"
            "pmullh     %[ftmp1],   %[ftmp1],       %[ftmp3]            \n\t"
            "pmullh     %[ftmp2],   %[ftmp2],       %[ftmp4]            \n\t"
            "paddsh     %[ftmp1],   %[ftmp1],       %[ftmp2]            \n\t"
            "paddsh     %[ftmp1],   %[ftmp1],       %[ff_pw_4]          \n\t"
            "psrlh      %[ftmp1],   %[ftmp1],       %[ff_pw_3]          \n\t"
            "packushb   %[ftmp1],   %[ftmp1],       %[ftmp0]            \n\t"
            "gssdlc1    %[ftmp1],   0x07(%[dst])                        \n\t"
            "gssdrc1    %[ftmp1],   0x00(%[dst])                        \n\t"
            : [ftmp0]"=&f"(ftmp[0]),        [ftmp1]"=&f"(ftmp[1]),
              [ftmp2]"=&f"(ftmp[2]),        [ftmp3]"=&f"(ftmp[3]),
              [ftmp4]"=&f"(ftmp[4]),
              [dst]"+&r"(dst)
            : [src]"r"(src),                [a]"r"(a),
              [b]"r"(b),
              [ff_pw_4]"f"(ff_pw_4),        [ff_pw_3]"f"(ff_pw_3)
            : "memory"
        );

        dst += dstride;
        src += sstride;
    }
#else
    int a = 8 - mx, b = mx;
    int x, y;

    for (y = 0; y < h; y++) {
        for (x = 0; x < 4; x++)
            dst[x] = (a * src[x] + b * src[x + 1] + 4) >> 3;
        dst += dstride;
        src += sstride;
    }
#endif
}

void ff_put_vp8_bilinear4_v_mmi(uint8_t *dst, ptrdiff_t dstride, uint8_t *src,
        ptrdiff_t sstride, int h, int mx, int my)
{
#if OK
    int c = 8 - my, d = my;
    int y;
    double ftmp[5];

    for (y = 0; y < h; y++) {
        /*
        dst[0] = (c * src[0] + d * src[    sstride] + 4) >> 3;
        dst[1] = (c * src[1] + d * src[1 + sstride] + 4) >> 3;
        dst[2] = (c * src[2] + d * src[2 + sstride] + 4) >> 3;
        dst[3] = (c * src[3] + d * src[3 + sstride] + 4) >> 3;
        */
        __asm__ volatile (
            "xor        %[ftmp0],   %[ftmp0],       %[ftmp0]            \n\t"
            "gslwlc1    %[ftmp1],   0x03(%[src1])                       \n\t"
            "mtc1       %[c],       %[ftmp3]                            \n\t"
            "gslwrc1    %[ftmp1],   0x00(%[src1])                       \n\t"
            "gslwlc1    %[ftmp2],   0x03(%[src2])                       \n\t"
            "mtc1       %[d],       %[ftmp4]                            \n\t"
            "gslwrc1    %[ftmp2],   0x00(%[src2])                       \n\t"
            "pshufh     %[ftmp3],   %[ftmp3],       %[ftmp0]            \n\t"
            "pshufh     %[ftmp4],   %[ftmp4],       %[ftmp0]            \n\t"
            "punpcklbh  %[ftmp1],   %[ftmp1],       %[ftmp0]            \n\t"
            "punpcklbh  %[ftmp2],   %[ftmp2],       %[ftmp0]            \n\t"
            "pmullh     %[ftmp1],   %[ftmp1],       %[ftmp3]            \n\t"
            "pmullh     %[ftmp2],   %[ftmp2],       %[ftmp4]            \n\t"
            "paddsh     %[ftmp1],   %[ftmp1],       %[ftmp2]            \n\t"
            "paddsh     %[ftmp1],   %[ftmp1],       %[ff_pw_4]          \n\t"
            "psrlh      %[ftmp1],   %[ftmp1],       %[ff_pw_3]          \n\t"
            "packushb   %[ftmp1],   %[ftmp1],       %[ftmp0]            \n\t"
            "gssdlc1    %[ftmp1],   0x07(%[dst])                        \n\t"
            "gssdrc1    %[ftmp1],   0x00(%[dst])                        \n\t"
            : [ftmp0]"=&f"(ftmp[0]),        [ftmp1]"=&f"(ftmp[1]),
              [ftmp2]"=&f"(ftmp[2]),        [ftmp3]"=&f"(ftmp[3]),
              [ftmp4]"=&f"(ftmp[4]),
              [dst]"+&r"(dst)
            : [src1]"r"(src),               [src2]"r"(src+sstride),
              [c]"r"(c),                    [d]"r"(d),
              [ff_pw_4]"f"(ff_pw_4),        [ff_pw_3]"f"(ff_pw_3)
            : "memory"
        );

        dst += dstride;
        src += sstride;
    }
#else
    int c = 8 - my, d = my;
    int x, y;

    for (y = 0; y < h; y++) {
        for (x = 0; x < 4; x++)
            dst[x] = (c * src[x] + d * src[x + sstride] + 4) >> 3;
        dst += dstride;
        src += sstride;
    }
#endif
}

void ff_put_vp8_bilinear4_hv_mmi(uint8_t *dst, ptrdiff_t dstride, uint8_t *src,
        ptrdiff_t sstride, int h, int mx, int my)
{
#if OK
    int a = 8 - mx, b = mx;
    int c = 8 - my, d = my;
    int y;
    uint8_t tmp_array[36];
    uint8_t *tmp = tmp_array;
    double ftmp[5];

    for (y = 0; y < h + 1; y++) {
        /*
        tmp[0] = (a * src[0] + b * src[1] + 4) >> 3;
        tmp[1] = (a * src[1] + b * src[2] + 4) >> 3;
        tmp[2] = (a * src[2] + b * src[3] + 4) >> 3;
        tmp[3] = (a * src[3] + b * src[4] + 4) >> 3;
        */
        __asm__ volatile (
            "xor        %[ftmp0],   %[ftmp0],       %[ftmp0]            \n\t"
            "gslwlc1    %[ftmp1],   0x03(%[src])                        \n\t"
            "mtc1       %[a],       %[ftmp3]                            \n\t"
            "gslwrc1    %[ftmp1],   0x00(%[src])                        \n\t"
            "gslwlc1    %[ftmp2],   0x04(%[src])                        \n\t"
            "mtc1       %[b],       %[ftmp4]                            \n\t"
            "gslwrc1    %[ftmp2],   0x01(%[src])                        \n\t"
            "pshufh     %[ftmp3],   %[ftmp3],       %[ftmp0]            \n\t"
            "pshufh     %[ftmp4],   %[ftmp4],       %[ftmp0]            \n\t"
            "punpcklbh  %[ftmp1],   %[ftmp1],       %[ftmp0]            \n\t"
            "punpcklbh  %[ftmp2],   %[ftmp2],       %[ftmp0]            \n\t"
            "pmullh     %[ftmp1],   %[ftmp1],       %[ftmp3]            \n\t"
            "pmullh     %[ftmp2],   %[ftmp2],       %[ftmp4]            \n\t"
            "paddsh     %[ftmp1],   %[ftmp1],       %[ftmp2]            \n\t"
            "paddsh     %[ftmp1],   %[ftmp1],       %[ff_pw_4]          \n\t"
            "psrlh      %[ftmp1],   %[ftmp1],       %[ff_pw_3]          \n\t"
            "packushb   %[ftmp1],   %[ftmp1],       %[ftmp0]            \n\t"
            "gssdlc1    %[ftmp1],   0x07(%[tmp])                        \n\t"
            "gssdrc1    %[ftmp1],   0x00(%[tmp])                        \n\t"
            : [ftmp0]"=&f"(ftmp[0]),        [ftmp1]"=&f"(ftmp[1]),
              [ftmp2]"=&f"(ftmp[2]),        [ftmp3]"=&f"(ftmp[3]),
              [ftmp4]"=&f"(ftmp[4]),
              [tmp]"+&r"(tmp)
            : [src]"r"(src),                [a]"r"(a),
              [b]"r"(b),
              [ff_pw_4]"f"(ff_pw_4),        [ff_pw_3]"f"(ff_pw_3)
            : "memory"
        );

        tmp += 4;
        src += sstride;
    }
#else
    int a = 8 - mx, b = mx;
    int c = 8 - my, d = my;
    int x, y;
    uint8_t tmp_array[36];
    uint8_t *tmp = tmp_array;

    for (y = 0; y < h + 1; y++) {
        for (x = 0; x < 4; x++)
            tmp[x] = (a * src[x] + b * src[x + 1] + 4) >> 3;
        tmp += 4;
        src += sstride;
    }
#endif

    tmp = tmp_array;

#if OK
    for (y = 0; y < h; y++) {
        /*
        dst[0] = (c * tmp[0] + d * tmp[4] + 4) >> 3;
        dst[1] = (c * tmp[1] + d * tmp[5] + 4) >> 3;
        dst[2] = (c * tmp[2] + d * tmp[6] + 4) >> 3;
        dst[3] = (c * tmp[3] + d * tmp[7] + 4) >> 3;
        */
        __asm__ volatile (
            "xor        %[ftmp0],   %[ftmp0],       %[ftmp0]            \n\t"
            "mtc1       %[c],       %[ftmp3]                            \n\t"
            "gsldlc1    %[ftmp1],   0x07(%[tmp])                        \n\t"
            "mtc1       %[d],       %[ftmp4]                            \n\t"
            "gsldrc1    %[ftmp1],   0x00(%[tmp])                        \n\t"
            "pshufh     %[ftmp3],   %[ftmp3],       %[ftmp0]            \n\t"
            "pshufh     %[ftmp4],   %[ftmp4],       %[ftmp0]            \n\t"
            "punpckhbh  %[ftmp2],   %[ftmp1],       %[ftmp0]            \n\t"
            "punpcklbh  %[ftmp1],   %[ftmp1],       %[ftmp0]            \n\t"
            "pmullh     %[ftmp1],   %[ftmp1],       %[ftmp3]            \n\t"
            "pmullh     %[ftmp2],   %[ftmp2],       %[ftmp4]            \n\t"
            "paddsh     %[ftmp1],   %[ftmp1],       %[ftmp2]            \n\t"
            "paddsh     %[ftmp1],   %[ftmp1],       %[ff_pw_4]          \n\t"
            "psrlh      %[ftmp1],   %[ftmp1],       %[ff_pw_3]          \n\t"
            "packushb   %[ftmp1],   %[ftmp1],       %[ftmp0]            \n\t"
            "gssdlc1    %[ftmp1],   0x07(%[dst])                        \n\t"
            "gssdrc1    %[ftmp1],   0x00(%[dst])                        \n\t"
            : [ftmp0]"=&f"(ftmp[0]),        [ftmp1]"=&f"(ftmp[1]),
              [ftmp2]"=&f"(ftmp[2]),        [ftmp3]"=&f"(ftmp[3]),
              [ftmp4]"=&f"(ftmp[4]),
              [dst]"+&r"(dst)
            : [tmp]"r"(tmp),                [c]"r"(c),
              [d]"r"(d),
              [ff_pw_4]"f"(ff_pw_4),        [ff_pw_3]"f"(ff_pw_3)
            : "memory"
        );

        dst += dstride;
        tmp += 4;
    }
#else
    for (y = 0; y < h; y++) {
        for (x = 0; x < 4; x++)
            dst[x] = (c * tmp[x] + d * tmp[x + 4] + 4) >> 3;
        dst += dstride;
        tmp += 4;
    }
#endif
}
