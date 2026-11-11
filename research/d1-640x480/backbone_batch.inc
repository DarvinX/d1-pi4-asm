/* D1-only scheduler. Hidden and projection matrices are position-major.
 * The main thread starts one pool job per projection; each worker owns a
 * contiguous slice of weight rows across all positions. */
.text
d1_read_batch_flags:
    ENTER
    ADDR x0,d1_q6_env
    bl getenv
    cbz x0,1f
    ldrb w0,[x0]
    cmp w0,#'1'
    cset w0,eq
    ADDR x1,d1_q6_enabled
    str w0,[x1]
1:  ADDR x0,d1_q6_acts_env
    bl getenv
    cbz x0,2f
    ldrb w0,[x0]
    cmp w0,#'1'
    cset w0,eq
    ADDR x1,d1_q6_acts_enabled
    str w0,[x1]
2:  ADDR x0,d1_panel_env
    bl getenv
    cbz x0,3f
    ldrb w0,[x0]
    cmp w0,#'1'
    cset w0,eq
    ADDR x1,d1_panel_enabled
    str w0,[x1]
3:  ADDR x0,d1_q4_across_env
    bl getenv
    cbz x0,4f
    ldrb w0,[x0]
    cmp w0,#'1'
    cset w0,eq
    ADDR x1,d1_q4_across_enabled
    str w0,[x1]
4:  LEAVE

.ifdef D1_Q4_GATE_UP_REUSE
.type d1_matrix_batch_reuse_q4,%function
d1_matrix_batch_reuse_q4: /* consecutive FFN up projection after gate */
    ADDR x5,d1_q4_gate_up_reuse_enabled
    ldr w5,[x5]
    cbz w5,d1_matrix_batch
    ldr w5,[x0,#16]
    cmp w5,#12
    b.ne d1_matrix_batch
    ADDR x5,d1_batch_job
    ldr x6,[x5,#d1_batch_source-d1_batch_job]
    cmp x6,x1
    b.ne d1_matrix_batch
    ldp w6,w7,[x5,#40]
    ldr w8,[x0,#8]
    cmp w6,w8
    ccmp w7,#12,#0,eq
    b.ne d1_matrix_batch
    ldr w6,[x5,#56]
    cmp w6,w3
    b.ne d1_matrix_batch
    ADDR x5,d1_fmla_this_job
    ldr w5,[x5]
    cmp w5,#3
    b.ne d1_matrix_batch
    mov x4,#0
    mov w5,#3
    b .Lbatch_enter
.size d1_matrix_batch_reuse_q4,.-d1_matrix_batch_reuse_q4
.endif

.type d1_matrix_batch,%function
d1_matrix_batch: /* descriptor, input, output, position count */
    mov x4,#0
d1_matrix_batch_ptr: /* x4 overrides the weight pointer when nonzero */
    .ifdef D1_VISION_WORK
    mov w5,#0
    b .Lbatch_enter

d1_matrix_batch_reuse_ptr: /* caller guarantees the preceding input is unchanged */
    ADDR x6,d1_vis_pack_reuse_enabled
    ldr w6,[x6]
    cbz w6,d1_matrix_batch_ptr
    ADDR x6,d1_q8_res_count
    ldr w6,[x6]
    .ifdef D1_Q8_RES_REUSE
    mov w5,#1
    cbz w6,.Lbatch_reuse_nonresident
    ADDR x6,d1_fmla_this_job
    ldr w6,[x6]
    sub w6,w6,#6
    .ifdef D1_Q8_CERTIFIED_6_CRC
    cmp w6,#3
    .else
    .ifdef D1_Q8_HALF_6_CRC
    cmp w6,#2
    .else
    cmp w6,#1
    .endif
    .endif
    b.hi d1_matrix_batch_ptr
    mov w5,#2
    b .Lbatch_reuse_source
.Lbatch_reuse_nonresident:
    .else
    cbnz w6,d1_matrix_batch_ptr
    .endif
    cmp w3,#300
    b.ne .Lbatch_reuse_source
    /* A matching source can switch between ordinary and packed Q8_0 jobs. */
    ADDR x6,d1_q8_i8_index
    ldr x6,[x6]
    cbnz x6,d1_matrix_batch_ptr
    ADDR x6,d1_q8_ilp_count
    ldr w6,[x6]
    cbnz w6,d1_matrix_batch_ptr
.Lbatch_reuse_source:
    ADDR x6,d1_batch_job
    ldr x7,[x6,#d1_batch_source-d1_batch_job]
    cmp x1,x7
    b.ne d1_matrix_batch_ptr
    ldr w7,[x6,#56]
    cmp w3,w7
    b.ne d1_matrix_batch_ptr
    ldp w7,w8,[x6,#40]
    cmp w8,#8
    b.ne d1_matrix_batch_ptr
    ldr w8,[x0,#16]
    cmp w8,#8
    b.ne d1_matrix_batch_ptr
    ldp w8,w9,[x0,#8]
    cmp w7,w8
    b.ne d1_matrix_batch_ptr
    ldr w7,[x6,#52]
    orr w7,w7,w9
    tst w7,#31 /* keep FMLA eligibility unchanged */
    b.ne d1_matrix_batch_ptr
    .ifndef D1_Q8_RES_REUSE
    mov w5,#1
    .endif
.Lbatch_enter:
    .endif
    ENTER
    mov x19,x0
    mov x20,x1
    mov x21,x2
    mov w22,w3
    ADDR x23,d1_batch_job
    .ifdef D1_VISION_WORK
    str x20,[x23,#d1_batch_source-d1_batch_job]
    mov w28,w5
    .endif
    cbnz x4,.Lbatch_explicit
    bl tensor_ptr
    b .Lbatch_weight_ready
.Lbatch_explicit:
    mov x0,x4
.Lbatch_weight_ready:
    str x0,[x23]
    str x21,[x23,#16]
    ldp w24,w25,[x19,#8] /* input width, output width */
    ldp w26,w27,[x19,#16]
    stp w24,w26,[x23,#40]
    stp w27,w25,[x23,#48]
    stp w22,wzr,[x23,#56]
    lsl x25,x25,#2
    str x25,[x23,#32]
    .ifdef D1_VISION_WORK
    .ifdef D1_Q4_GATE_UP_REUSE
    cmp w28,#3
    b.eq .Lbatch_reuse_q4_pack
    .endif
    cbnz w28,.Lbatch_reuse_vision_pack
    .endif
    lsl x27,x24,#2
    cbz w26,.Lbatch_input_ready
    ADDR x19,lfm_q8_pack_h24
    .ifdef D1_VISION_WORK
    cmp w26,#8
    .ifdef D1_Q4_0_WEIGHTS
    b.eq .Lbatch_pack_q8_0
    cmp w26,#2
    .endif
    b.ne .Lbatch_pack_k
.Lbatch_pack_q8_0:
    lsr w25,w24,#5
    mov w26,#34
    mul w26,w25,w26
    ADDR x19,d1_pack_q8_0
    b .Lbatch_pack_setup
.Lbatch_pack_k:
    .endif
    cmp w26,#12
    b.ne .Lbatch_pack_k_stride
    ADDR x0,d1_q6_acts_enabled
    ldr w0,[x0]
    cbz w0,.Lbatch_pack_k_stride
    ADDR x19,lfm_q8_pack_h24_q6
.Lbatch_pack_k_stride:
    lsr w25,w24,#8
    mov w26,#320
    mul w26,w25,w26
.Lbatch_pack_setup:
    ADDR x25,d1_batch_pack
    .ifdef D1_PARALLEL_PACK
    ADDR x0,d1_parallel_pack_enabled
    ldr w0,[x0]
    cbz w0,.Lbatch_serial_pack
    cmp w22,#4
    b.lo .Lbatch_serial_pack
    mul w0,w22,w24
    cmp w0,#16,lsl #12 /* amortize the worker wakeups */
    b.lo .Lbatch_serial_pack
    mov x0,x20
    mov x1,x25
    mov w2,w24
    mov x3,x27
    mov x4,x26
    mov x5,x19
    mov w6,w22
    bl d1_dispatch_raw_pack
    b .Lbatch_packed
.Lbatch_serial_pack:
    .endif
    mov x21,x25
    mov w28,w22
.Lbatch_pack:
    mov x0,x20
    mov x1,x21
    mov w2,w24
    blr x19
    add x20,x20,x27
    add x21,x21,x26
    subs w28,w28,#1
    b.ne .Lbatch_pack
    .ifdef D1_PARALLEL_PACK
.Lbatch_packed:
    .endif
    mov x20,x25
    mov x27,x26
.Lbatch_input_ready:
    str x20,[x23,#8]
    str x27,[x23,#24]
    .ifdef D1_VISION_WORK
    ADDR x0,d1_vis_fmla_enabled
    ldr w0,[x0]
    .ifdef D1_ASYNC_BACKBONE_PANELS
    ADDR x1,d1_async_loader_active
    ldr w1,[x1]
    cbz w1,.Lbatch_check_backbone_panels
    /* Only the vision encoder dispatches while the loader owns the Q4
     * and Q6 metadata. Its resident panels are already verified. */
    ADDR x1,d1_q8_res_count
    ldr w1,[x1]
    orr w0,w0,w1
    b .Lbatch_check_other_vision_panels
.Lbatch_check_backbone_panels:
    .endif
    ADDR x1,d1_q6_i16_enabled
    ldr w1,[x1]
    orr w0,w0,w1
    ADDR x1,d1_q4_i8_min
    ldr w1,[x1]
    orr w0,w0,w1
    .ifdef D1_ASYNC_BACKBONE_PANELS
.Lbatch_check_other_vision_panels:
    .endif
    ADDR x1,d1_q8_i8_index
    ldr x1,[x1]
    orr x0,x0,x1
    ADDR x1,d1_q8_ilp_count
    ldr w1,[x1]
    orr x0,x0,x1
    cbz x0,.Lbatch_dispatch_matrix
    mov x0,x23
    bl d1_fmla_prepare_batch
.Lbatch_dispatch_matrix:
    .ifdef D1_DEFER_MODEL_CRC
    ADDR x0,d1_fmla_this_job
    ldr w0,[x0]
    sub w0,w0,#2 /* Q6 compact and Q4 panels already checked at map time. */
    cmp w0,#1
    b.ls .Lbatch_raw_ready
    .ifdef D1_DEFER_VISION_CRC
    sub w0,w0,#4 /* Resident vision kernel modes 6 through 9. */
    cmp w0,#3
    b.ls .Lbatch_raw_ready
    .endif
    ldr x0,[x23]
    bl d1_verify_raw_tensor
.Lbatch_raw_ready:
    .endif
    .endif
    mov w0,#1
    bl d1_batch_dispatch
    LEAVE
    .ifdef D1_VISION_WORK
    .ifdef D1_Q4_GATE_UP_REUSE
.Lbatch_reuse_q4_pack:
    /* If the new panel is ineligible, the regular Q4 path uses the raw pack. */
    ADDR x0,d1_fmla_this_job
    str wzr,[x0]
    mov x0,x23
    bl d1_q4_i8_prepare_reused
    b .Lbatch_dispatch_matrix
    .endif
.Lbatch_reuse_vision_pack:
    /* The packed pointer and stride are already in the job. */
    .ifdef D1_Q8_RES_REUSE
    cmp w28,#2
    b.eq .Lbatch_reuse_resident_pack
    .endif
    ADDR x0,d1_vis_pack_reuse_counts
    ldr w1,[x0]
    add w1,w1,#1
    str w1,[x0]
    ADDR x1,d1_fmla_this_job
    ldr w1,[x1]
    cbz w1,.Lbatch_dispatch_matrix
    ldr w1,[x0,#4]
    add w1,w1,#1
    str w1,[x0,#4]
    ADDR x0,d1_fmla_positions
    ldr w0,[x0]
    sub w0,w0,w22
    ADDR x1,d1_fmla_calls
    ldp w2,w3,[x1]
    add w2,w2,#1
    add w3,w3,w0
    stp w2,w3,[x1]
    b .Lbatch_dispatch_matrix
    .ifdef D1_Q8_RES_REUSE
.Lbatch_reuse_resident_pack:
    ADDR x0,d1_vis_pack_reuse_counts
    ldr w1,[x0]
    add w1,w1,#1
    str w1,[x0]
    mov x0,x23
    bl d1_q8_res_prepare_reused
    cbz w0,.Lbatch_reuse_resident_fallback
    ADDR x0,d1_q8_res_reuse_count
    ldr w1,[x0]
    add w1,w1,#1
    str w1,[x0]
    b .Lbatch_dispatch_matrix
.Lbatch_reuse_resident_fallback:
    mov x0,x23
    bl d1_fmla_prepare_batch
    b .Lbatch_dispatch_matrix
    .endif
    .endif
.size d1_matrix_batch,.-d1_matrix_batch

.type d1_batch_dispatch,%function
d1_batch_dispatch: /* publish one job of the requested type */
    ENTER
    ADDR x21,d1_batch_active
    str w0,[x21]
    ADDR x0,pool_size
    ldr w22,[x0]
    mov w20,#1
.Lbatch_post:
    cmp w20,w22
    b.hs .Lbatch_own
    ADDR x0,worker_semaphores
    add x0,x0,x20,lsl #6
    bl sem_post
    add w20,w20,#1
    b .Lbatch_post
.Lbatch_own:
    bl matrix_work
    mov w20,#1
.Lbatch_wait:
    cmp w20,w22
    b.hs .Lbatch_done
    ADDR x0,done_semaphore
    bl sem_wait_retry
    add w20,w20,#1
    b .Lbatch_wait
.Lbatch_done:
    str wzr,[x21]
    LEAVE
.size d1_batch_dispatch,.-d1_batch_dispatch

.ifdef D1_PARALLEL_PACK
.type d1_read_parallel_pack,%function
d1_read_parallel_pack:
    ENTER
    ADDR x0,d1_parallel_pack_env
    bl getenv
    cbz x0,.Lraw_pack_config_done
    ldrb w1,[x0]
    cmp w1,#'1'
    b.ne .Lraw_pack_config_done
    ldrb w1,[x0,#1]
    cbnz w1,.Lraw_pack_config_done
    ADDR x0,pool_size
    ldr w0,[x0]
    cmp w0,#4
    b.ne .Lraw_pack_config_done
    ADDR x0,d1_parallel_pack_enabled
    mov w1,#1
    str w1,[x0]
.Lraw_pack_config_done:
    LEAVE
.size d1_read_parallel_pack,.-d1_read_parallel_pack

.type d1_dispatch_raw_pack,%function
d1_dispatch_raw_pack: /* input, output, width, input stride, output stride, quantizer, rows */
    ADDR x7,d1_raw_pack_job
    stp x0,x1,[x7]
    stp x3,x4,[x7,#16]
    str x5,[x7,#32]
    stp w2,w6,[x7,#40]
    str wzr,[x7,#48]
    ADDR x7,d1_raw_pack_counts
    ldp w3,w4,[x7]
    add w3,w3,#1
    madd w4,w2,w6,w4
    stp w3,w4,[x7]
    mov w0,#7
    b d1_batch_dispatch
.size d1_dispatch_raw_pack,.-d1_dispatch_raw_pack

.type d1_raw_pack_worker,%function
d1_raw_pack_worker:
    ENTER
    ADDR x0,d1_raw_pack_job
    add x1,x0,#48
1:  ldaxr w2,[x1]
    add w3,w2,#1
    stlxr w4,w3,[x1]
    cbnz w4,1b
    ldp x19,x20,[x0]
    ldp x21,x22,[x0,#16]
    ldr x23,[x0,#32]
    ldp w24,w25,[x0,#40]
    mul w2,w2,w25
    mul w3,w3,w25
    lsr w2,w2,#2
    lsr w3,w3,#2
    sub w25,w3,w2
    madd x19,x2,x21,x19
    madd x20,x2,x22,x20
.Lraw_pack_row:
    mov x0,x19
    mov x1,x20
    mov w2,w24
    blr x23
    add x19,x19,x21
    add x20,x20,x22
    subs w25,w25,#1
    b.ne .Lraw_pack_row
    LEAVE
.size d1_raw_pack_worker,.-d1_raw_pack_worker

.section .rodata
d1_parallel_pack_env: .asciz "D1_PARALLEL_PACK"
d1_parallel_pack_stats: .asciz "d1_parallel_pack jobs=%u elements=%u\n"
.bss
.balign 64
d1_raw_pack_job: .skip 52
d1_raw_pack_counts: .skip 8
d1_parallel_pack_enabled: .skip 4
.text
.endif

.ifdef D1_PARALLEL_ACTIVATIONS
.type d1_read_parallel_activations,%function
d1_read_parallel_activations:
    ENTER
    ADDR x0,d1_parallel_activations_env
    bl getenv
    cbz x0,.Lactivation_config_done
    ldrb w1,[x0]
    cmp w1,#'1'
    b.ne .Lactivation_config_done
    ldrb w1,[x0,#1]
    cbnz w1,.Lactivation_config_done
    ADDR x0,pool_size
    ldr w0,[x0]
    cmp w0,#4
    b.ne .Lactivation_config_done
    ADDR x0,d1_parallel_activations_enabled
    mov w1,#1
    str w1,[x0]
.Lactivation_config_done:
    LEAVE
.size d1_read_parallel_activations,.-d1_read_parallel_activations

.type d1_dispatch_activations,%function
d1_dispatch_activations: /* gate, up (NULL for GELU), elements divisible by 16 */
    ADDR x3,d1_activation_job
    stp x0,x1,[x3]
    lsr w4,w2,#2
    stp w4,wzr,[x3,#16]
    ADDR x3,d1_activation_counts
    ldp w4,w5,[x3]
    ldr w6,[x3,#8]
    cmp x1,#0
    cinc w4,w4,ne
    cinc w5,w5,eq
    add w6,w6,w2
    stp w4,w5,[x3]
    str w6,[x3,#8]
    mov w0,#6
    b d1_batch_dispatch
.size d1_dispatch_activations,.-d1_dispatch_activations

.type d1_activation_worker,%function
d1_activation_worker:
    ADDR x3,d1_activation_job
    add x5,x3,#20
1:  ldaxr w4,[x5]
    add w6,w4,#1
    stlxr w7,w6,[x5]
    cbnz w7,1b
    ldp x0,x1,[x3]
    ldr w2,[x3,#16]
    mul w4,w4,w2
    add x0,x0,x4,lsl #2
    cbz x1,.Lactivation_gelu
    add x1,x1,x4,lsl #2
    b lfm_swiglu_native
.Lactivation_gelu:
    mov w1,w2
    b d1_vision_gelu_erf
.size d1_activation_worker,.-d1_activation_worker

.section .rodata
d1_parallel_activations_env: .asciz "D1_PARALLEL_ACTIVATIONS"
d1_parallel_activations_stats: .asciz "d1_parallel_activations swiglu=%u gelu=%u elements=%u\n"
.bss
.balign 64
d1_activation_job: .skip 24
d1_parallel_activations_enabled: .skip 4
d1_activation_counts: .skip 12
.text
.endif

.macro D1_BATCH_SCALAR pos
    ldp x0,x3,[sp,#32]
    ldr x1,[sp,#16]
    .if \pos
    mov x4,#\pos
    madd x1,x4,x23,x1
    .endif
    mov w2,w28
    blr x20
    ldr x3,[sp,#40]
    .if \pos
    mov x4,#\pos
    madd x3,x4,x24,x3
    .endif
    str s0,[x3]
.endm

.type d1_batch_work,%function
d1_batch_work:
    .ifdef D1_VISION_WORK
    ADDR x0,d1_fmla_this_job
    ldr w0,[x0]
    cmp w0,#2
    b.eq d1_fmla_compact_batch_work
    cmp w0,#3
    b.eq d1_q4_i8_batch_work
    cmp w0,#4
    b.eq d1_q8_i8_batch_work
    cmp w0,#5
    b.eq d1_q8_ilp_batch_work
    cmp w0,#6
    b.eq d1_q8_res_batch_work
    .ifdef D1_Q8_CONV_6_CRC
    cmp w0,#7
    b.eq d1_q8_res_narrow_batch_work
    .ifdef D1_Q8_HALF_6_CRC
    cmp w0,#8
    b.eq d1_q8_res_half_batch_work
    .ifdef D1_Q8_CERTIFIED_6_CRC
    cmp w0,#9
    b.eq d1_q8_res_certified_batch_work
    .endif
    .endif
    .endif
    cbnz w0,d1_fmla_batch_work
    .endif
    ENTER
    ADDR x19,d1_batch_job
    add x0,x19,#60
1:  ldaxr w1,[x0]
    add w2,w1,#1
    stlxr w3,w2,[x0]
    cbnz w3,1b
    ADDR x0,pool_size
    ldr w3,[x0]
    ldr w4,[x19,#52]
    mul w1,w1,w4
    mul w2,w2,w4
    udiv w1,w1,w3
    udiv w2,w2,w3
    subs w26,w2,w1
    b.eq .Lworker_exit
    ldp x21,x0,[x19]
    ldp x22,x23,[x19,#16]
    ldr x24,[x19,#32]
    ldr w25,[x19,#48]
    ldr w27,[x19,#56]
    ldr w28,[x19,#40]
    madd x21,x1,x25,x21
    add x22,x22,x1,lsl #2
    sub sp,sp,#96
    stp x0,x22,[sp,#16]
    str wzr,[sp,#84] /* no rows after the current panel by default */
    ldr w0,[x19,#44]
    .ifdef D1_VISION_WORK
    cmp w0,#8
    b.ne 2f
    ADDR x0,d1_dot_q8_0
    b 3f
2:
    .endif
    bl select_dot
3:
    mov x20,x0
    ldr w0,[x19,#44]
    .ifdef D1_VISION_WORK
    cmp w0,#8
    b.eq .Lworker_q8_panel
    .endif
    cmp w0,#12
    b.ne .Lworker_tile
    ADDR x0,d1_panel_enabled
    ldr w0,[x0]
    cbz w0,.Lworker_tile
    mov w2,#16
    b .Lworker_setup_panel
    .ifdef D1_VISION_WORK
.Lworker_q8_panel:
    ADDR x0,d1_vis_q8_panel_enabled
    ldr w0,[x0]
    cbz w0,.Lworker_tile
    /* A panel plus four packed activations stays below 28 KiB. */
    mov w2,#2
    mov w3,#4
    cmp w28,#2560
    csel w2,w3,w2,ls
    mov w3,#16
    cmp w28,#1024
    csel w2,w3,w2,ls
    .endif
.Lworker_setup_panel:
    cmp w26,w2
    b.ls .Lworker_tile
    cmp w27,#4
    b.lo .Lworker_tile
    ldp x0,x1,[sp,#16]
    stp x0,x1,[sp,#64] /* activation base, panel output base */
    sub w0,w26,w2
    stp w27,w0,[sp,#80] /* positions, remaining rows */
    str w2,[sp,#88]
    mov w26,w2
.Lworker_tile:
    cmp w27,#4
    b.lo .Lworker_tail
    ldr x22,[sp,#24]
    stp x21,x22,[sp,#32]
    str w26,[sp,#48]
    ldr w0,[x19,#44]
    .ifdef D1_VISION_WORK
    cbnz w0,4f
    ADDR x0,d1_dot4_f32
    b .Lworker_tile_selected
4:  cmp w0,#8
    b.ne 5f
    ADDR x0,d1_dot4_q8_0_ilp
    ADDR x1,d1_vis_q8_ilp_enabled
    ldr w1,[x1]
    cbnz w1,.Lworker_tile_selected
    ADDR x0,d1_dot4_q8_0
    ADDR x1,d1_vis_pair_enabled
    ldr w1,[x1]
    cbz w1,.Lworker_tile_selected
    ADDR x0,d1_dot4_q8_0_pair
    b .Lworker_tile_selected
5:
    .ifdef D1_Q4_0_WEIGHTS
    cmp w0,#2
    b.ne 6f
    ADDR x0,d1_dot4_q4_0
    b .Lworker_tile_selected
6:
    .endif
    .endif
    cmp w0,#12
    b.ne .Lworker_q6
    ADDR x0,d1_dot4_q4_k
    ADDR x1,d1_q4_across_enabled
    ldr w1,[x1]
    cbz w1,.Lworker_tile_selected
    ADDR x0,d1_dot4_q4_k_across
    b .Lworker_tile_selected
.Lworker_q6:
    cmp w0,#14
    b.ne .Lworker_other_rows
    ADDR x0,d1_q6_enabled
    ldr w0,[x0]
    cbz w0,.Lworker_other_rows
    ADDR x0,d1_dot4_q6_k
.Lworker_tile_selected:
    str x0,[sp,#56]
.Lworker_tiled_rows:
    ldr x0,[sp,#32]
    ldr x1,[sp,#16]
    mov x2,x23
    mov x3,sp
    mov w4,w28
    ldr x5,[sp,#56]
    blr x5
    ldr x3,[sp,#40]
    ldp s0,s1,[sp]
    ldp s2,s3,[sp,#8]
    str s0,[x3]
    add x3,x3,x24
    str s1,[x3]
    add x3,x3,x24
    str s2,[x3]
    add x3,x3,x24
    str s3,[x3]
    ldp x0,x3,[sp,#32]
    add x0,x0,x25
    add x3,x3,#4
    stp x0,x3,[sp,#32]
    ldr w0,[sp,#48]
    subs w0,w0,#1
    str w0,[sp,#48]
    b.ne .Lworker_tiled_rows
    b .Lworker_advance
.Lworker_other_rows:
    D1_BATCH_SCALAR 0
    D1_BATCH_SCALAR 1
    D1_BATCH_SCALAR 2
    D1_BATCH_SCALAR 3
    ldp x0,x3,[sp,#32]
    add x0,x0,x25
    add x3,x3,#4
    stp x0,x3,[sp,#32]
    ldr w0,[sp,#48]
    subs w0,w0,#1
    str w0,[sp,#48]
    b.ne .Lworker_other_rows
.Lworker_advance:
    ldp x0,x1,[sp,#16]
    add x0,x0,x23,lsl #2
    add x1,x1,x24,lsl #2
    stp x0,x1,[sp,#16]
    sub w27,w27,#4
    b .Lworker_tile
.Lworker_tail:
    cbz w27,.Lworker_finish
    ldr x22,[sp,#24]
    stp x21,x22,[sp,#32]
    str w26,[sp,#48]
.Lworker_tail_rows:
    D1_BATCH_SCALAR 0
    ldp x0,x3,[sp,#32]
    add x0,x0,x25
    add x3,x3,#4
    stp x0,x3,[sp,#32]
    ldr w0,[sp,#48]
    subs w0,w0,#1
    str w0,[sp,#48]
    b.ne .Lworker_tail_rows
    ldp x0,x1,[sp,#16]
    add x0,x0,x23
    add x1,x1,x24
    stp x0,x1,[sp,#16]
    subs w27,w27,#1
    b.ne .Lworker_tail
.Lworker_finish:
    ldr w2,[sp,#84]
    cbz w2,.Lworker_unwind
    ldp x0,x1,[sp,#64]
    madd x21,x26,x25,x21
    add x1,x1,x26,lsl #2
    str x1,[sp,#72]
    stp x0,x1,[sp,#16]
    ldr w27,[sp,#80]
    ldr w3,[sp,#88]
    cmp w2,w3
    csel w26,w2,w3,lo
    sub w2,w2,w26
    str w2,[sp,#84]
    b .Lworker_tile
.Lworker_unwind:
    add sp,sp,#96
.Lworker_exit:
    LEAVE
.size d1_batch_work,.-d1_batch_work

.type d1_ffn_batch,%function
d1_ffn_batch: /* descriptor, count, starting row, layer (sparse build) */
    ENTER
    mov x19,x0
    mov w20,w1
    ADDR x21,d1_hidden
    add x21,x21,x2,lsl #13
    ADDR x22,d1_batch_norm
    ADDR x23,d1_batch_gate
    ADDR x24,d1_batch_up
    mov w25,#0
    .ifdef D1_SPARSE_WORK
    mov w26,w3
    ADDR x0,d1_sparse_live
    ldr w0,[x0]
    cbz w0,.Lffn_norm
    .ifdef D1_BACKBONE_TRANSITION_FFN
    cmp w26,#2
    b.ne .Lffn_indexed_setup
    ADDR x0,d1_transition_ffn_enabled
    ldr w0,[x0]
    cmp w0,#4
    b.eq .Lffn_norm
.Lffn_indexed_setup:
    .endif
    ADDR x26,d1_sparse_active
    ADDR x0,d1_sparse_sizes
    ldr w27,[x0]
    ADDR x28,d1_ffn_row_map
.Lffn_norm_indexed:
    ldr w0,[x26,x25,lsl #2]
    add x1,x21,x0,lsl #13
    add x0,x19,#32
    add x2,x22,x25,lsl #13
    mov w3,#DIM
    bl norm
    add w25,w25,#1
    cmp w25,w27
    b.lo .Lffn_norm_indexed
    ADDR x0,d1_late_ffn_enabled
    ldr w6,[x0]
    cbz w6,.Lffn_gate
    mov x1,x26
    mov w2,w27
    ADDR x3,d1_late_ffn_reps
    ADDR x4,d1_image_first
    ldr w4,[x4]
    mov x5,x28
    ADDR x0,d1_late_ffn_cutoff
    ldr s0,[x0]
    mov x0,x22
    cmp w6,#2
    b.eq .Lffn_share_weighted
    ADDR x7,d1_sparse_small
    ldr w7,[x7]
    cbnz w7,.Lffn_share_eight
    bl d1_share_late_ffn
    b .Lffn_share_done
.Lffn_share_eight:
    .ifdef D1_BACKBONE_REDUCED_ROWS
    cmp w7,#8
    b.eq .Lffn_share_four
    .endif
    bl d1_share_late_ffn8
    b .Lffn_share_done
    .ifdef D1_BACKBONE_REDUCED_ROWS
.Lffn_share_four:
    bl d1_share_late_ffn4
    b .Lffn_share_done
    .endif
.Lffn_share_weighted:
    bl d1_share_late_ffn_weighted
.Lffn_share_done:
    sub w1,w27,w0
    mov w27,w0
    ADDR x0,d1_late_ffn_layers
    ldp w2,w3,[x0]
    add w2,w2,#1
    add w3,w3,w1
    stp w2,w3,[x0]
    b .Lffn_gate
    .endif
.Lffn_norm:
    add x0,x19,#32
    add x1,x21,x25,lsl #13
    add x2,x22,x25,lsl #13
    mov w3,#DIM
    bl norm
    add w25,w25,#1
    cmp w25,w20
    b.lo .Lffn_norm
    mov w27,w20
    mov x28,xzr
    .ifdef D1_VISION_WORK
    .ifdef D1_SPARSE_WORK
    cmp w26,#2
    b.ls .Lffn_maybe_reuse
    ADDR x0,d1_sparse_enabled
    ldr w0,[x0]
    cbnz w0,.Lffn_gate
.Lffn_maybe_reuse:
    .endif
    ADDR x0,d1_ffn_reuse_enabled
    ldr w26,[x0]
    cbz w26,.Lffn_gate
    ADDR x0,d1_ffn_rows
    ldr w1,[x0]
    add w1,w1,w20
    str w1,[x0]
    ADDR x0,d1_image_first
    ldr w2,[x0]
    add w0,w2,#300
    cmp w0,w20
    b.hi .Lffn_gate
    ADDR x28,d1_ffn_row_map
    mov x0,x22
    mov x1,x28
    mov w3,w20
    cmp w26,#4
    b.ne .Lffn_compact_pair
    bl d1_compact_ffn_centroids
    b .Lffn_compact_done
.Lffn_compact_pair:
    bl d1_compact_ffn_rows
.Lffn_compact_done:
    mov w27,w0
    .ifdef D1_BACKBONE_TRANSITION_FFN
    ADDR x0,d1_transition_ffn_budget
    ldr w0,[x0]
    cbz w0,.Lffn_reuse_count
    /* The original centroid map is complete before filtering its outputs. */
    mov x0,x22
    mov x1,x28
    ADDR x2,d1_sparse_mask
    mov w3,w20
    mov w4,w27
    ADDR x5,d1_transition_ffn_map
    bl d1_ffn_compact_keep
    mov w27,w0
.Lffn_reuse_count:
    .endif
    ADDR x0,d1_ffn_reused_rows
    ldr w1,[x0]
    sub w2,w20,w27
    add w1,w1,w2
    str w1,[x0]
    .endif
.Lffn_gate:
    add x0,x19,#(4*32)
    mov x1,x22
    mov x2,x23
    mov w3,w27
    bl d1_matrix_batch
    add x0,x19,#(5*32)
    mov x1,x22
    mov x2,x24
    mov w3,w27
    .ifdef D1_Q4_GATE_UP_REUSE
    bl d1_matrix_batch_reuse_q4
    .else
    bl d1_matrix_batch
    .endif
    mov w25,#0
    mov x26,#FF*4
    .ifdef D1_PARALLEL_ACTIVATIONS
    ADDR x0,d1_parallel_activations_enabled
    ldr w0,[x0]
    cbz w0,.Lffn_swiglu
    mov x0,x23
    mov x1,x24
    mov w2,#FF
    mul w2,w2,w27
    bl d1_dispatch_activations
    b .Lffn_swiglu_done
    .endif
.Lffn_swiglu:
    madd x0,x25,x26,x23
    madd x1,x25,x26,x24
    mov w2,#FF
    bl lfm_swiglu_native
    add w25,w25,#1
    cmp w25,w27
    b.lo .Lffn_swiglu
    .ifdef D1_PARALLEL_ACTIVATIONS
.Lffn_swiglu_done:
    .endif
    .ifdef D1_FFN_FIXTURE_WORK
    mov w0,#1
    mov x1,x23
    mul x2,x20,x26
    bl write_all
    LEAVE
    .endif
    add x0,x19,#(6*32)
    mov x1,x23
    mov x2,x22
    mov w3,w27
    bl d1_matrix_batch
    mov w25,#0
.Lffn_add:
    mov w0,w25
    cbz x28,.Lffn_output_row
    ldr w0,[x28,x25,lsl #2]
    .ifdef D1_SPARSE_WORK
    tbnz w0,#31,.Lffn_add_next
    .endif
.Lffn_output_row:
    add x26,x22,x0,lsl #13
    ldr x0,[x19,#(3*32)]
    cbz x0,.Lffn_add_row
    add x0,x19,#(3*32)
    mov x1,x26
    cbz x28,.Lffn_post_norm
    ADDR x26,operator_out /* shared source must survive normalization */
.Lffn_post_norm:
    mov x2,x26
    mov w3,#DIM
    bl norm
.Lffn_add_row:
    add x0,x21,x25,lsl #13
    mov x1,x26
    mov w2,#DIM
    bl lfm_add
.Lffn_add_next:
    add w25,w25,#1
    cmp w25,w20
    b.lo .Lffn_add
    LEAVE
.size d1_ffn_batch,.-d1_ffn_batch

.ifdef D1_VISION_WORK
.include "research/d1-640x480/ffn_compact.inc"
.endif

.ifdef D1_FFN_FIXTURE_WORK
d1_ffn_fixture:
    ENTER
    ADDR x19,layer_tensors
    ADDR x0,d1_count
    ldr w20,[x0]
    mov x0,x19
    mov w1,w20
    mov w2,#0
    mov w3,#0
    bl d1_conv_batch
    mov w21,#0
.Lfixture_residual:
    ADDR x0,operator_out
    ADDR x1,d1_batch_gate
    add x1,x1,x21,lsl #13
    mov x2,#DIM*4
    bl memcpy
    mov x0,x19
    ADDR x1,d1_hidden
    add x1,x1,x21,lsl #13
    bl d1_operator_residual
    add w21,w21,#1
    cmp w21,w20
    b.lo .Lfixture_residual
    mov x0,x19
    mov w1,w20
    mov w2,#0
    bl d1_ffn_batch
    LEAVE
.endif

.section .rodata
d1_batch_env: .asciz "D1_FFN_BATCH"
d1_q6_env: .asciz "D1_Q6_BATCH"
d1_q6_acts_env: .asciz "D1_Q6_ACT"
d1_panel_env: .asciz "D1_Q4_PANEL"
d1_q4_across_env: .asciz "D1_Q4_ACROSS"
.bss
.balign 64
d1_batch_job: .skip 64
.ifdef D1_VISION_WORK
d1_batch_source: .skip 8
.endif
d1_batch_active: .skip 4
d1_ffn_enabled: .skip 4
d1_q6_enabled: .skip 4
d1_q6_acts_enabled: .skip 4
d1_panel_enabled: .skip 4
d1_q4_across_enabled: .skip 4
.balign 64
d1_conv_gated: .skip (CONV-1)*DIM*4 /* permanent zero prefix for the causal stencil */
d1_batch_norm: .skip CTX*DIM*4
d1_batch_gate: .skip CTX*FF*4
d1_batch_up: .skip CTX*FF*4
d1_batch_pack:
.ifdef D1_VISION_WORK
    .if CTX*(FF/256)*320 < 1200*(2560/32)*34
    .skip 1200*(2560/32)*34
    .else
    .skip CTX*(FF/256)*320
    .endif
.else
    .skip CTX*(FF/256)*320
.endif
d1_batch_pack_end:
.text
