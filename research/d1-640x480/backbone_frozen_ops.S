/* Prepare once from frozen residuals. Reuse only projections of eligible
 * image rows in the same 4x4 tile. All attention and tap positions remain. */
#ifdef D1_SPARSE_VISUAL
.arch armv8-a
.include "src/common.inc"
.text
.balign 64
.global d1_frozen_projection_plan
.type d1_frozen_projection_plan,%function
/* x0 hidden, x1 skip mask, w2 first image row, w3 positions, x4 owner, s0 cosine. */
d1_frozen_projection_plan:
    ENTER
    sub sp,sp,#144
    mov x19,x0
    mov x20,x1
    mov w21,w2
    add w22,w2,#300
    mov x23,x4
    fmov s10,s0
    mov w0,#0
.Lidentity:
    str w0,[x23,x0,lsl #2]
    add w0,w0,#1
    cmp w0,w3
    b.lo .Lidentity
    mov w24,#0
.Ltile:
    mov w0,#5
    udiv w1,w24,w0
    msub w2,w1,w0,w24
    mov w0,#80
    madd w1,w1,w0,w21
    add w1,w1,w2,lsl #2
    str w1,[sp,#128]
    mov w25,#0
    mov w26,#0
.Lrow:
    ldr w0,[sp,#128]
    lsr w1,w25,#2
    mov w2,#20
    madd w27,w1,w2,w0
    and w0,w25,#3
    add w27,w27,w0
    cmp w27,w22
    b.hs .Ltile_next
    ldrb w0,[x20,x27]
    cbnz w0,.Lrow_next
    add x0,x19,x27,lsl #13
    mov x1,x0
    mov w2,#2048
    bl lfm_dot_native
    fsqrt s8,s0
    fmov s9,s10
    mov w28,#0
.Lcompare:
    cmp w28,w26
    b.hs .Lkeep
    ldr s1,[sp,x28,lsl #2]
    fmul s1,s1,s8
    fcmp s1,#0.0
    b.le .Lcompare_next
    add x2,sp,#64
    ldr w1,[x2,x28,lsl #2]
    add x0,x19,x27,lsl #13
    add x1,x19,x1,lsl #13
    mov w2,#2048
    bl lfm_dot_native
    ldr s1,[sp,x28,lsl #2]
    fmul s1,s1,s8
    fdiv s0,s0,s1
    fcmp s0,s9
    b.lt .Lcompare_next
    fmov s9,s0
    add x0,sp,#64
    ldr w1,[x0,x28,lsl #2]
    str w1,[x23,x27,lsl #2]
.Lcompare_next:
    add w28,w28,#1
    b .Lcompare
.Lkeep:
    ldr w0,[x23,x27,lsl #2]
    cmp w0,w27
    b.ne .Lrow_next
    str s8,[sp,x26,lsl #2]
    add x0,sp,#64
    str w27,[x0,x26,lsl #2]
    add w26,w26,#1
.Lrow_next:
    add w25,w25,#1
    cmp w25,#16
    b.lo .Lrow
.Ltile_next:
    add w24,w24,#1
    cmp w24,#20
    b.lo .Ltile
    add sp,sp,#144
    LEAVE
.size d1_frozen_projection_plan,.-d1_frozen_projection_plan

.global d1_frozen_projection_list
.type d1_frozen_projection_list,%function
/* x0 owner, x1 optional original row IDs, w2 count, x3 filtered row IDs. */
d1_frozen_projection_list:
    mov w4,#0
    mov w5,#0
.Llist_row:
    mov w6,w4
    cbz x1,.Llist_owner
    ldr w6,[x1,x4,lsl #2]
.Llist_owner:
    ldr w7,[x0,x6,lsl #2]
    cmp w6,w7
    b.ne .Llist_next
    str w6,[x3,x5,lsl #2]
    add w5,w5,#1
.Llist_next:
    add w4,w4,#1
    cmp w4,w2
    b.lo .Llist_row
    mov w0,w5
    ret
.size d1_frozen_projection_list,.-d1_frozen_projection_list

.global d1_frozen_projection_scatter
.type d1_frozen_projection_scatter,%function
/* x0 output, x1 owner, w2 first image row, x3 stride, x4 output row bytes. */
d1_frozen_projection_scatter:
    ENTER
    mov x19,x0
    mov x20,x1
    mov w21,w2
    add w22,w2,#300
    mov x23,x3
    mov x24,x4
.Lscatter_row:
    ldr w0,[x20,x21,lsl #2]
    cmp w0,w21
    b.eq .Lscatter_next
    madd x1,x0,x23,x19
    madd x0,x21,x23,x19
    mov x2,x24
    bl memcpy
.Lscatter_next:
    add w21,w21,#1
    cmp w21,w22
    b.lo .Lscatter_row
    LEAVE
.size d1_frozen_projection_scatter,.-d1_frozen_projection_scatter

.global d1_scaled_projection_scatter
.type d1_scaled_projection_scatter,%function
/* x0 input, x1 compact output, x2 full output, x3 owners, w4 rows, x5 row bytes,
 * s0 blend from least-squares gain towards the input norm ratio.
 * Every owner precedes its duplicates in raster order. Overwrite the entries
 * of representatives with compact indices as their full rows are written. */
d1_scaled_projection_scatter:
    ENTER
    mov x19,x0
    mov x20,x1
    mov x21,x2
    mov x22,x3
    mov w23,w4
    mov x24,x5
    fmov s9,s0
    mov w25,#0
    mov w26,#0
.Lscaled_row:
    ldr w27,[x22,x25,lsl #2]
    cmp w27,w25
    b.ne .Lscaled_duplicate
    str w26,[x22,x25,lsl #2]
    madd x0,x25,x24,x21
    madd x1,x26,x24,x20
    mov x2,x24
    bl memcpy
    add w26,w26,#1
    b .Lscaled_next
.Lscaled_duplicate:
    add x28,x19,x27,lsl #13
    mov x0,x28
    mov x1,x28
    mov w2,#2048
    bl lfm_dot_native
    fmov s8,s0
    add x0,x19,x25,lsl #13
    mov x1,x28
    mov w2,#2048
    bl lfm_dot_native
    fdiv s0,s0,s8
    fcmp s9,#0.0
    b.eq .Lscaled_gain_ready
    fmov s10,s0
    add x0,x19,x25,lsl #13
    mov x1,x0
    mov w2,#2048
    bl lfm_dot_native
    fdiv s0,s0,s8
    fsqrt s0,s0
    fsub s0,s0,s10
    fmadd s0,s9,s0,s10
.Lscaled_gain_ready:
    dup v0.4s,v0.s[0]
    ldr w0,[x22,x27,lsl #2]
    madd x1,x0,x24,x20
    madd x0,x25,x24,x21
    add x2,x0,x24
.Lscaled_vector:
    ldp q1,q2,[x1],#32
    fmul v1.4s,v1.4s,v0.4s
    fmul v2.4s,v2.4s,v0.4s
    stp q1,q2,[x0],#32
    cmp x0,x2
    b.lo .Lscaled_vector
.Lscaled_next:
    add w25,w25,#1
    cmp w25,w23
    b.lo .Lscaled_row
    LEAVE
.size d1_scaled_projection_scatter,.-d1_scaled_projection_scatter
#endif
.section .note.GNU-stack,"",%progbits
