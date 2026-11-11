/* In-place raster conversion for the fixed 40×30 vision grid. */
.arch armv8-a
.include "src/common.inc"
.text
.balign 64
.global d1_vision_quad_reduce
.type d1_vision_quad_reduce,%function
/* x0: full rows, w1: row width (multiple of four), x2: optional full detail. */
d1_vision_quad_reduce:
    ubfiz x3,x1,#2,#32
    mov x4,#40
    mul x4,x4,x3
    mov x5,x0                 /* top-left source row */
    mov x6,x0                 /* compact destination row */
    sub x7,x2,x0             /* detail address relative to source */
    mov w8,#15
    fmov v16.4s,#0.25
.Lreduce_y:
    mov w9,#20
.Lreduce_x:
    add x10,x5,x3
    add x11,x5,x4
    add x12,x11,x3
    mov x13,#0
.Lreduce_channels:
    ldr q0,[x5,x13]
    ldr q1,[x10,x13]
    ldr q2,[x11,x13]
    ldr q3,[x12,x13]
    fadd v4.4s,v0.4s,v1.4s
    fadd v4.4s,v4.4s,v2.4s
    fadd v4.4s,v4.4s,v3.4s
    fmul v4.4s,v4.4s,v16.4s
    str q4,[x6,x13]
    cbz x2,.Lreduce_next
    add x14,x7,x13
    fsub v0.4s,v0.4s,v4.4s
    fsub v1.4s,v1.4s,v4.4s
    fsub v2.4s,v2.4s,v4.4s
    fsub v3.4s,v3.4s,v4.4s
    str q0,[x5,x14]
    str q1,[x10,x14]
    str q2,[x11,x14]
    str q3,[x12,x14]
.Lreduce_next:
    add x13,x13,#16
    cmp x13,x3
    b.lo .Lreduce_channels
    add x5,x5,x3,lsl #1
    add x6,x6,x3
    subs w9,w9,#1
    b.ne .Lreduce_x
    add x5,x5,x4
    subs w8,w8,#1
    b.ne .Lreduce_y
    ret
.size d1_vision_quad_reduce,.-d1_vision_quad_reduce

.balign 64
.global d1_vision_quad_expand
.type d1_vision_quad_expand,%function
/* x0: compact rows with full capacity, w1: width, x2: optional detail. */
d1_vision_quad_expand:
    ubfiz x3,x1,#2,#32
    mov x4,#40
    mul x4,x4,x3
    mov x5,#1158             /* raster row of the last tile's top-left */
    madd x5,x5,x3,x0
    mov x6,#299
    madd x6,x6,x3,x0
    sub x7,x2,x0
    mov w8,#15
.Lexpand_y:
    mov w9,#20
.Lexpand_x:
    add x10,x5,x3
    add x11,x5,x4
    add x12,x11,x3
    mov x13,#0
.Lexpand_channels:
    ldr q4,[x6,x13]
    cbnz x2,.Lexpand_detail
    str q4,[x5,x13]
    str q4,[x10,x13]
    str q4,[x11,x13]
    str q4,[x12,x13]
    b .Lexpand_next
.Lexpand_detail:
    add x14,x7,x13
    ldr q0,[x5,x14]
    ldr q1,[x10,x14]
    ldr q2,[x11,x14]
    ldr q3,[x12,x14]
    fadd v0.4s,v4.4s,v0.4s
    fadd v1.4s,v4.4s,v1.4s
    fadd v2.4s,v4.4s,v2.4s
    fadd v3.4s,v4.4s,v3.4s
    str q0,[x5,x13]
    str q1,[x10,x13]
    str q2,[x11,x13]
    str q3,[x12,x13]
.Lexpand_next:
    add x13,x13,#16
    cmp x13,x3
    b.lo .Lexpand_channels
    sub x5,x5,x3,lsl #1
    sub x6,x6,x3
    subs w9,w9,#1
    b.ne .Lexpand_x
    sub x5,x5,x4
    subs w8,w8,#1
    b.ne .Lexpand_y
    ret
.size d1_vision_quad_expand,.-d1_vision_quad_expand

/* Early convolution groups use records indexed by the first raster row.
 * Each 16-byte record stores a u16 count, up to four u16 member rows, and
 * the FP32 anchor norm used only during planning. */
.balign 64
.global d1_vision_conv_group_plan
.type d1_vision_conv_group_plan,%function
d1_vision_conv_group_plan: /* full residual, records, u32 raster-to-group map, s0 cosine */
    ENTER
    sub sp,sp,#32
    mov x19,x0
    mov x20,x1
    mov x21,x2
    fmov s11,s0
    mov x0,x20
    mov w1,#0
    mov w2,#19200
    bl memset
    mov w22,#0
    mov w23,#20
.Lconv_group_tile:
    add w0,w22,#1
    add w1,w22,#40
    add w2,w22,#41
    stp w22,w0,[sp]
    stp w1,w2,[sp,#8]
    mov w24,#0
    mov w25,#0
.Lconv_group_row:
    ldr w26,[sp,x24,lsl #2]
    add x0,x19,x26,lsl #12
    mov x1,x0
    mov w2,#1024
    bl lfm_dot_native
    fsqrt s8,s0
    fmov s9,s11
    mov w28,w26
    mov w27,#0
.Lconv_group_compare:
    cmp w27,w25
    b.hs .Lconv_group_assign
    add x0,sp,#16
    ldr w0,[x0,x27,lsl #2]
    add x1,x20,x0,lsl #4
    ldr s0,[x1,#12]
    fmul s10,s8,s0
    fcmp s10,#0.0
    b.le .Lconv_group_compare_next
    add x1,x19,x0,lsl #12
    add x0,x19,x26,lsl #12
    mov w2,#1024
    bl lfm_dot_native
    fdiv s0,s0,s10
    fcmp s0,s9
    b.lt .Lconv_group_compare_next
    fmov s9,s0
    add x0,sp,#16
    ldr w28,[x0,x27,lsl #2]
.Lconv_group_compare_next:
    add w27,w27,#1
    b .Lconv_group_compare
.Lconv_group_assign:
    add x0,x20,x28,lsl #4
    cmp w28,w26
    b.ne .Lconv_group_append
    str s8,[x0,#12]
    add x1,sp,#16
    str w26,[x1,x25,lsl #2]
    add w25,w25,#1
.Lconv_group_append:
    ldrh w1,[x0]
    add w1,w1,#1
    strh w26,[x0,x1,lsl #1]
    strh w1,[x0]
    add w24,w24,#1
    cmp w24,#4
    b.lo .Lconv_group_row
    add w22,w22,#2
    subs w23,w23,#1
    b.ne .Lconv_group_tile
    add w22,w22,#40
    mov w23,#20
    cmp w22,#1200
    b.lo .Lconv_group_tile

    mov w0,#0
    mov w1,#0
.Lconv_group_map:
    add x2,x20,x1,lsl #4
    ldrh w3,[x2],#2
    cbz w3,.Lconv_group_map_next
.Lconv_group_member:
    ldrh w4,[x2],#2
    str w0,[x21,x4,lsl #2]
    subs w3,w3,#1
    b.ne .Lconv_group_member
    add w0,w0,#1
.Lconv_group_map_next:
    add w1,w1,#1
    cmp w1,#1200
    b.lo .Lconv_group_map
    add sp,sp,#32
    LEAVE
.size d1_vision_conv_group_plan,.-d1_vision_conv_group_plan

.macro CONV_GROUP_MEAN count
    .if \count > 1
    ldrh w10,[x5,#4]
    add x10,x0,x10,lsl #12
    .endif
    .if \count > 2
    ldrh w11,[x5,#6]
    add x11,x0,x11,lsl #12
    .endif
    .if \count > 3
    ldrh w12,[x5,#8]
    add x12,x0,x12,lsl #12
    .endif
    mov x7,#0
.Lconv_mean_vector\@:
    ldr q0,[x9,x7]
    .if \count == 1
    str q0,[x8,x7]
    .else
    ldr q1,[x10,x7]
    fadd v4.4s,v0.4s,v1.4s
    .if \count > 2
    ldr q2,[x11,x7]
    fadd v4.4s,v4.4s,v2.4s
    .endif
    .if \count > 3
    ldr q3,[x12,x7]
    fadd v4.4s,v4.4s,v3.4s
    .endif
    .if \count == 2
    fmul v4.4s,v4.4s,v16.4s
    .elseif \count == 3
    fdiv v4.4s,v4.4s,v17.4s
    .else
    fmul v4.4s,v4.4s,v18.4s
    .endif
    str q4,[x8,x7]
    cbz x2,.Lconv_mean_next\@
    add x14,x13,x7
    fsub v0.4s,v0.4s,v4.4s
    fsub v1.4s,v1.4s,v4.4s
    str q0,[x9,x14]
    str q1,[x10,x14]
    .if \count > 2
    fsub v2.4s,v2.4s,v4.4s
    str q2,[x11,x14]
    .endif
    .if \count > 3
    fsub v3.4s,v3.4s,v4.4s
    str q3,[x12,x14]
    .endif
    .endif
.Lconv_mean_next\@:
    add x7,x7,#16
    cmp x7,#4096
    b.lo .Lconv_mean_vector\@
.endm

.balign 64
.global d1_vision_conv_group_reduce
.type d1_vision_conv_group_reduce,%function
d1_vision_conv_group_reduce: /* in-place 1024-channel rows, records, optional detail */
    mov w3,#0
    mov w4,#0
    sub x13,x2,x0
    fmov v16.4s,#0.5
    fmov v17.4s,#3.0
    fmov v18.4s,#0.25
.Lconv_mean_record:
    add x5,x1,x3,lsl #4
    ldrh w6,[x5]
    cbz w6,.Lconv_mean_next_record
    add x8,x0,x4,lsl #12
    add x9,x0,x3,lsl #12
    cmp w6,#1
    b.ne .Lconv_mean_two
    cmp x8,x9
    b.eq .Lconv_mean_kept
    CONV_GROUP_MEAN 1
    b .Lconv_mean_kept
.Lconv_mean_two:
    cmp w6,#2
    b.ne .Lconv_mean_three
    CONV_GROUP_MEAN 2
    b .Lconv_mean_kept
.Lconv_mean_three:
    cmp w6,#3
    b.ne .Lconv_mean_four
    CONV_GROUP_MEAN 3
    b .Lconv_mean_kept
.Lconv_mean_four:
    CONV_GROUP_MEAN 4
.Lconv_mean_kept:
    add w4,w4,#1
.Lconv_mean_next_record:
    add w3,w3,#1
    cmp w3,#1200
    b.lo .Lconv_mean_record
    ret
.size d1_vision_conv_group_reduce,.-d1_vision_conv_group_reduce

.balign 64
.global d1_vision_conv_group_expand
.type d1_vision_conv_group_expand,%function
d1_vision_conv_group_expand: /* in-place compact normalized rows, raster-to-group map */
    mov w2,#1199
.Lconv_expand_row:
    ldr w3,[x1,x2,lsl #2]
    cmp w3,w2
    b.eq .Lconv_expand_next
    add x3,x0,x3,lsl #12
    add x4,x0,x2,lsl #12
    add x5,x4,#4096
.Lconv_expand_vector:
    ldp q0,q1,[x3],#32
    stp q0,q1,[x4],#32
    cmp x4,x5
    b.lo .Lconv_expand_vector
.Lconv_expand_next:
    subs w2,w2,#1
    b.ge .Lconv_expand_row
    ret
.size d1_vision_conv_group_expand,.-d1_vision_conv_group_expand

.balign 64
.global d1_vision_conv_group_finish
.type d1_vision_conv_group_finish,%function
/* Expand the operator residual, then add the original row's FFN output
 * and saved detail in that order. A compact ID cannot exceed its first
 * raster row, so backward traversal can expand in place. */
d1_vision_conv_group_finish: /* hidden, group map, FFN output, FFN map, detail */
    mov w5,#1199
.Lconv_finish_row:
    ldr w6,[x1,x5,lsl #2]
    ldr w7,[x3,x5,lsl #2]
    add x6,x0,x6,lsl #12
    add x7,x2,x7,lsl #12
    add x8,x4,x5,lsl #12
    add x9,x0,x5,lsl #12
    add x10,x9,#4096
.Lconv_finish_vector:
    ldr q0,[x6],#16
    ldr q1,[x7],#16
    ldr q2,[x8],#16
    fadd v0.4s,v0.4s,v1.4s
    fadd v0.4s,v0.4s,v2.4s
    str q0,[x9],#16
    cmp x9,x10
    b.lo .Lconv_finish_vector
    subs w5,w5,#1
    b.ge .Lconv_finish_row
    ret
.size d1_vision_conv_group_finish,.-d1_vision_conv_group_finish

.ifdef D1_VISION_QUERY_GROUP
/* Score all six pairs once. A pass contributes 1 + cosine to the
 * orientation, so a matched pair beats any unmatched pair. */
.macro D1_QUERY_PAIR a,b,slot
    ldr w0,[sp,#4*\a]
    ldr w1,[sp,#4*\b]
    add x0,x19,x0,lsl #12
    add x1,x19,x1,lsl #12
    mov w2,#1024
    bl lfm_dot_native
    ldr s1,[sp,#16+4*\a]
    ldr s2,[sp,#16+4*\b]
    fmul s1,s1,s2
    fcmp s1,#0.0
    b.le .Lquery_pair_reject\@
    fdiv s0,s0,s1
    fcmp s0,s8
    b.lt .Lquery_pair_reject\@
    orr w25,w25,#(1<<\slot)
    fmov s1,#1.0
    fadd s0,s0,s1
    b .Lquery_pair_weight\@
.Lquery_pair_reject\@:
    fmov s0,wzr
.Lquery_pair_weight\@:
    str s0,[sp,#32+4*\slot]
.endm

.macro D1_QUERY_RECORD a,b,slot
    ldr w0,[sp,#4*\a]
    ldr w1,[sp,#4*\b]
    add x2,x20,x0,lsl #4
    mov w3,#1
    strh w0,[x2,#2]
    tbnz w25,#\slot,.Lquery_record_pair\@
    add x4,x20,x1,lsl #4
    strh w3,[x4]
    strh w1,[x4,#2]
    b .Lquery_record_first\@
.Lquery_record_pair\@:
    mov w3,#2
    strh w1,[x2,#4]
.Lquery_record_first\@:
    strh w3,[x2]
.endm

.balign 64
.global d1_vision_query_group_plan
.type d1_vision_query_group_plan,%function
d1_vision_query_group_plan: /* post-RoPE query rows, records, raster map, s0 cosine */
    ENTER
    sub sp,sp,#64
    mov x19,x0
    mov x20,x1
    mov x21,x2
    fmov s8,s0
    mov x0,x20
    mov w1,#0
    mov w2,#19200
    bl memset
    mov w22,#0
    mov w23,#20
.Lquery_group_tile:
    add w0,w22,#1
    add w1,w22,#40
    add w2,w22,#41
    stp w22,w0,[sp]
    stp w1,w2,[sp,#8]
    mov w24,#0
.Lquery_group_norm:
    ldr w0,[sp,x24,lsl #2]
    add x0,x19,x0,lsl #12
    mov x1,x0
    mov w2,#1024
    bl lfm_dot_native
    fsqrt s0,s0
    add x0,sp,#16
    str s0,[x0,x24,lsl #2]
    add w24,w24,#1
    cmp w24,#4
    b.lo .Lquery_group_norm
    mov w25,#0
    D1_QUERY_PAIR 0,1,0
    D1_QUERY_PAIR 0,2,1
    D1_QUERY_PAIR 0,3,2
    D1_QUERY_PAIR 1,2,3
    D1_QUERY_PAIR 1,3,4
    D1_QUERY_PAIR 2,3,5
    cmp w25,#63
    b.ne .Lquery_group_pairs
    add x0,x20,x22,lsl #4
    mov w1,#4
    strh w1,[x0]
    ldp w1,w2,[sp]
    ldp w3,w4,[sp,#8]
    strh w1,[x0,#2]
    strh w2,[x0,#4]
    strh w3,[x0,#6]
    strh w4,[x0,#8]
    b .Lquery_group_next
.Lquery_group_pairs:
    ldp s0,s1,[sp,#32] /* 0/1 and 0/2 */
    ldp s2,s3,[sp,#48] /* 1/3 and 2/3 */
    fadd s0,s0,s3
    fadd s1,s1,s2
    fcmp s1,s0
    b.gt .Lquery_group_vertical
    D1_QUERY_RECORD 0,1,0
    D1_QUERY_RECORD 2,3,5
    b .Lquery_group_next
.Lquery_group_vertical:
    D1_QUERY_RECORD 0,2,1
    D1_QUERY_RECORD 1,3,4
.Lquery_group_next:
    add w22,w22,#2
    subs w23,w23,#1
    b.ne .Lquery_group_tile
    add w22,w22,#40
    mov w23,#20
    cmp w22,#1200
    b.lo .Lquery_group_tile

    mov w0,#0
    mov w1,#0
.Lquery_group_map:
    add x2,x20,x1,lsl #4
    ldrh w3,[x2],#2
    cbz w3,.Lquery_group_map_next
.Lquery_group_member:
    ldrh w4,[x2],#2
    str w0,[x21,x4,lsl #2]
    subs w3,w3,#1
    b.ne .Lquery_group_member
    add w0,w0,#1
.Lquery_group_map_next:
    add w1,w1,#1
    cmp w1,#1200
    b.lo .Lquery_group_map
    add sp,sp,#64
    LEAVE
.size d1_vision_query_group_plan,.-d1_vision_query_group_plan
.endif
.section .note.GNU-stack,"",%progbits
