/* Model-specific seven-tap causal convolution. Both linear projections
 * are batched; the filter preserves H42's operation order per channel. */
.if CONV != 7 || DIM != 2048
.error "d1 batched convolution requires seven taps and 2048 channels"
.endif
.text

d1_read_conv_flag:
    ENTER
    ADDR x0,d1_conv_env
    bl getenv
    cbz x0,1f
    ldrb w0,[x0]
    cmp w0,#'1'
    cset w0,eq
    ADDR x1,d1_conv_enabled
    str w0,[x1]
1:  LEAVE

.type d1_conv_batch,%function
d1_conv_batch: /* descriptor, count, starting row, last-output-only flag, layer */
    ENTER
    mov x19,x0
    mov w20,w1
    ADDR x21,d1_hidden
    add x21,x21,x2,lsl #13
    ADDR x22,d1_batch_norm
    mov w24,w2
    mov w25,w3
    .ifdef D1_SPARSE_WORK
    ADDR x0,d1_early_conv_enabled
    ldr w26,[x0]
    cmp w4,#2
    csel w26,w26,wzr,lo
    ADDR x27,d1_early_conv_counts
    add x27,x27,x4,lsl #3
    .endif
    mov w23,#0
.Lc_norm:
    mov x0,x19
    add x1,x21,x23,lsl #13
    add x2,x22,x23,lsl #13
    mov w3,#DIM
    bl norm
    add w23,w23,#1
    cmp w23,w20
    b.lo .Lc_norm
    .ifdef D1_SPARSE_WORK
    ADDR x0,d1_sparse_live
    ldr w0,[x0]
    cbz w0,.Lc_dense_input
    add x0,x19,#7*32
    bl d1_sparse_conv_input
    b .Lc_input_ready
.Lc_dense_input:
    cbz w26,.Lc_full_input
    add x0,x19,#7*32
    mov x1,x22
    ADDR x2,d1_batch_gate
    mov w3,w20
    ADDR x4,d1_batch_up /* not written until the stencil */
    mov x5,x27
    bl d1_early_conv_project
    b .Lc_input_ready
.Lc_full_input:
    .endif
    add x0,x19,#7*32
    mov x1,x22
    ADDR x2,d1_batch_gate
    mov w3,w20
    bl d1_matrix_batch
.Lc_input_ready:

    /* The GGUF stores seven taps adjacent within each channel. */
    add x0,x19,#9*32
    bl tensor_ptr
    ADDR x1,d1_conv_taps
    mov w2,#DIM
.Lc_transpose_channel:
    mov x3,x1
    mov w4,#7
.Lc_transpose_tap:
    ldr s0,[x0],#4
    str s0,[x3]
    add x3,x3,#2,lsl #12
    subs w4,w4,#1
    b.ne .Lc_transpose_tap
    add x1,x1,#4
    subs w2,w2,#1
    b.ne .Lc_transpose_channel

    ADDR x0,d1_batch_gate
    mov x1,x22
    mov w2,w20
    bl d1_conv_make_gate
    ADDR x0,d1_batch_gate
    ADDR x1,d1_conv_gated
    ADDR x2,d1_conv_taps
    ADDR x3,d1_batch_up
    mov w4,w20
    bl d1_conv_stencil
    .ifdef D1_SPARSE_WORK
    ADDR x0,d1_sparse_live
    ldr w0,[x0]
    cbz w0,.Lc_dense_output
    add x0,x19,#8*32
    ADDR x1,d1_batch_up
    ADDR x2,d1_batch_gate
    ADDR x3,d1_sparse_active
    ADDR x4,d1_sparse_sizes
    ldr w4,[x4]
    mov x5,#0
    mov x6,#0
    bl d1_matrix_batch_indexed
    LEAVE
.Lc_dense_output:
    cbz w26,.Lc_full_output
    add x0,x19,#8*32
    ADDR x1,d1_batch_up
    ADDR x2,d1_batch_gate
    mov w3,w20
    ADDR x4,d1_sparse_output
    add x5,x27,#4
    bl d1_early_conv_project
    LEAVE
.Lc_full_output:
    .endif
    ADDR x1,d1_batch_up
    ADDR x2,d1_batch_gate
    add x2,x2,x24,lsl #13
    mov w3,w20
    cbz w25,.Lc_out
    sub w0,w20,#1
    add x1,x1,x0,lsl #13
    add x2,x2,x0,lsl #13
    mov w3,#1
.Lc_out:
    add x0,x19,#8*32
    bl d1_matrix_batch
    LEAVE
.size d1_conv_batch,.-d1_conv_batch

d1_conv_make_gate: /* B/C/X, B*X output after zero prefix, positions */
.Lc_gate_pos:
    mov x3,x0
    add x4,x0,#4,lsl #12
    mov w5,#DIM/4
.Lc_gate_lane:
    ldr q0,[x3],#16
    ldr q1,[x4],#16
    fmul v0.4s,v0.4s,v1.4s
    str q0,[x1],#16
    subs w5,w5,#1
    b.ne .Lc_gate_lane
    add x0,x0,#6,lsl #12
    subs w2,w2,#1
    b.ne .Lc_gate_pos
    ret

.macro D1_CAUSAL_TAP k
    ldr q1,[x6,#(\k)*DIM*4]
    ldr q2,[x8,#(\k)*DIM*4]
    fmla v0.4s,v1.4s,v2.4s
.endm

.balign 64
d1_conv_stencil: /* B/C/X, padded B*X, tap-major weights, output, positions */
    add x0,x0,#2,lsl #12 /* C is the middle third of the projection */
.Lc_conv_pos:
    mov x6,x1
    mov x7,x0
    mov x8,x2
    mov w5,#DIM/4
.Lc_conv_lane:
    movi v0.4s,#0
    .irp tap,0,1,2,3,4,5,6
    D1_CAUSAL_TAP \tap
    .endr
    ldr q1,[x7],#16
    fmul v0.4s,v0.4s,v1.4s
    str q0,[x3],#16
    add x6,x6,#16
    add x8,x8,#16
    subs w5,w5,#1
    b.ne .Lc_conv_lane
    add x0,x0,#6,lsl #12
    add x1,x1,#2,lsl #12
    subs w4,w4,#1
    b.ne .Lc_conv_pos
    ret

.section .rodata
d1_conv_env: .asciz "D1_CONV_BATCH"
.bss
.balign 64
d1_conv_taps: .skip CONV*DIM*4
d1_conv_enabled: .skip 4
.text
