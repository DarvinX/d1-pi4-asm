/* Four-worker projection schedules for the pinned vision and backbone graphs. */
.equ D1_FMLA_KC,128
.equ D1_FMLA_MR,12
.equ D1_FMLA_NR,8
.equ D1_FMLA_MC,120
.equ D1_FMLA_MC_MAX,240
.equ D1_FMLA_NC,96
.equ D1_FMLA_MAX_K,4096
.equ D1_FMLA_MAX_POS,1200
.equ D1_FMLA_ACT_FLOATS,1200*2560
.equ D1_FMLA_WEIGHT_BYTES,D1_FMLA_NC*D1_FMLA_MAX_K*4
.equ D1_FMLA_OUTPUT_BYTES,D1_FMLA_MC_MAX*D1_FMLA_NC*4
.equ D1_Q6_I16_KC,256
.equ D1_Q6_I16_TILE_BYTES,4112
.equ D1_Q6_I16_ACT_BYTES,6192
.equ D1_Q6_I16_ACT_MAX_POS,((D1_FMLA_ACT_FLOATS*4)/((7680/D1_Q6_I16_KC)*D1_Q6_I16_ACT_BYTES))*D1_FMLA_MR

.text
.type d1_read_fmla_mc,%function
d1_read_fmla_mc:
    ENTER
    ADDR x0,d1_fmla_mc_env
    bl getenv
    cbz x0,.Lfmla_mc_done
    sub sp,sp,#16
    mov x1,sp
    mov w2,#10
    bl strtol
    ldr x1,[sp]
    add sp,sp,#16
    ldrb w1,[x1]
    cbnz w1,.Lfmla_mc_invalid
    cmp x0,#36
    b.eq .Lfmla_mc_store
    cmp x0,#D1_FMLA_MC
    b.eq .Lfmla_mc_store
    cmp x0,#D1_FMLA_MC_MAX
    b.ne .Lfmla_mc_invalid
.Lfmla_mc_store:
    ADDR x1,d1_fmla_mc
    str w0,[x1]
.Lfmla_mc_done:
    LEAVE
.Lfmla_mc_invalid:
    ADDR x0,d1_fmla_mc_error
    b fatal
.size d1_read_fmla_mc,.-d1_read_fmla_mc

.type d1_fmla_mc_report,%function
d1_fmla_mc_report:
    ADDR x0,d1_fmla_mc
    ldr w2,[x0]
    cmp w2,#D1_FMLA_MC
    b.eq .Lfmla_mc_unreported
    ADDR x0,d1_fmla_calls
    ldr w0,[x0]
    cbz w0,.Lfmla_mc_unreported
    mov w0,#2
    ADDR x1,d1_fmla_mc_stats
    b dprintf
.Lfmla_mc_unreported:
    ret
.size d1_fmla_mc_report,.-d1_fmla_mc_report

.macro D1_FMLA_ROWS_PER_PANEL reg,compact
    .if \compact
    mov \reg,#D1_FMLA_MC
    .else
    ADDR x0,d1_fmla_mc
    ldr \reg,[x0]
    .endif
.endm

.type d1_fmla_prepare_batch,%function
d1_fmla_prepare_batch: /* x0 matrix job, called before publishing to the pool */
    ENTER
    mov x19,x0
    ADDR x0,d1_fmla_this_job
    str wzr,[x0]
    ADDR x0,pool_size
    ldr w0,[x0]
    cmp w0,#4
    b.ne .Lfmla_prepare_done
    ldr w0,[x19,#44]
    cmp w0,#14
    b.eq .Lfmla_prepare_q6
    cmp w0,#12
    b.eq .Lfmla_prepare_q4
    cmp w0,#8
    b.ne .Lfmla_prepare_done
    .ifdef D1_Q8_ALL_6_CRC
    ADDR x0,d1_q8_res_all
    ldr w0,[x0]
    cbz w0,.Lfmla_prepare_ilp
    mov x0,x19
    bl d1_q8_res_prepare
    cbnz w0,.Lfmla_prepare_done
.Lfmla_prepare_ilp:
    .endif
    mov x0,x19
    bl d1_q8_ilp_prepare
    cbnz w0,.Lfmla_prepare_done
    mov x0,x19
    bl d1_q8_i8_prepare
    cbnz w0,.Lfmla_prepare_done
    mov w26,#1
    ADDR x0,d1_vis_fmla_enabled
    ldr w0,[x0]
    cbz w0,.Lfmla_prepare_done
    cmp w0,#1
    b.eq .Lfmla_stage_ready
    ADDR x1,d1_fmla_stage
    ldr w1,[x1]
    cmp w0,w1
    b.ne .Lfmla_prepare_done
.Lfmla_stage_ready:
    ldr w21,[x19,#40]
    cbz w21,.Lfmla_prepare_done
    cmp w21,#D1_FMLA_MAX_K
    b.hi .Lfmla_prepare_done
    tst w21,#D1_FMLA_KC-1
    b.ne .Lfmla_prepare_done
    ldp w0,w20,[x19,#52]
    cbz w0,.Lfmla_prepare_done
    tst w0,#4*D1_FMLA_NR-1
    b.ne .Lfmla_prepare_done
    mov x0,x19
    bl d1_q8_res_prepare
    cbnz w0,.Lfmla_prepare_done
    b .Lfmla_prepare_positions
.Lfmla_prepare_q4:
    mov x0,x19
    bl d1_q4_i8_prepare
    b .Lfmla_prepare_done
.Lfmla_prepare_q6:
    ADDR x0,d1_q6_i16_enabled
    ldr w0,[x0]
    cbz w0,.Lfmla_prepare_done
    ldr w21,[x19,#40]
    mov w0,#7680
    cmp w21,w0
    b.ne .Lfmla_prepare_done
    ldp w0,w20,[x19,#52]
    cmp w0,#2048
    b.ne .Lfmla_prepare_done
    .ifdef D1_SHORT_PACKED_ROWS
    ADDR x0,d1_q6_i16_min
    ldr w0,[x0]
    cmp w20,w0
    .else
    cmp w20,#48
    .endif
    b.lo .Lfmla_q6_small
    ldr x0,[x19]
    bl d1_q6_i16_find
    cbz x0,.Lfmla_prepare_done
    ADDR x1,d1_q6_i16_for_job
    str x0,[x1]
    mov w26,#2
    b .Lfmla_prepare_positions
.Lfmla_q6_small:
    ADDR x0,d1_q6_i16_fallback
    ldp w1,w2,[x0]
    add w1,w1,#1
    add w2,w2,w20
    stp w1,w2,[x0]
    b .Lfmla_prepare_done
.Lfmla_prepare_positions:
    cbz w20,.Lfmla_prepare_done
    cmp w20,#D1_FMLA_MAX_POS
    b.hi .Lfmla_prepare_done
    add w0,w20,#D1_FMLA_MR-1
    mov w1,#D1_FMLA_MR
    udiv w22,w0,w1
    mul w22,w22,w1
    cmp w26,#2
    b.eq .Lfmla_check_q6_capacity
    mul w0,w22,w21
    ldr w1,=D1_FMLA_ACT_FLOATS
    cmp w0,w1
    b.hi .Lfmla_prepare_done
    b .Lfmla_check_raw_capacity
.Lfmla_check_q6_capacity:
    /* The compact path stores one 6192-byte tile per 12 rows and 256 columns. */
    cmp w22,#D1_Q6_I16_ACT_MAX_POS
    b.hi .Lfmla_prepare_done
.Lfmla_check_raw_capacity:
    ldr x23,[x19,#8]
    ldr x24,[x19,#24]
    mul x0,x22,x24
    ldr x1,=d1_batch_pack_end-d1_batch_pack
    cmp x0,x1
    b.hi .Lfmla_prepare_done

    subs w25,w22,w20
    b.eq .Lfmla_prepare_pack
    madd x0,x20,x24,x23
    mov w1,#0
    mul x2,x25,x24
    bl memset
.Lfmla_prepare_pack:
    mov x0,x23
    ADDR x1,d1_fmla_activations
    mov w2,w22
    mov w3,w21
    mov w4,w24
    mov w5,#D1_FMLA_MR
    cmp w26,#2
    b.eq .Lfmla_pack_q6
    bl d1_fmla_pack_panel
    ADDR x0,d1_fmla_calls
    ldp w1,w2,[x0]
    add w1,w1,#1
    add w2,w2,w25
    stp w1,w2,[x0]
    b .Lfmla_publish
.Lfmla_pack_q6:
    mov w3,w4
    bl d1_q6_i16_pack_activations
    ADDR x0,d1_q6_i16_batches
    ldr w1,[x0]
    add w1,w1,#1
    str w1,[x0]
.Lfmla_publish:
    ADDR x0,d1_fmla_positions
    str w22,[x0]
    ADDR x0,d1_fmla_this_job
    str w26,[x0]
.Lfmla_prepare_done:
    LEAVE
.size d1_fmla_prepare_batch,.-d1_fmla_prepare_batch

.macro D1_FMLA_PACK_FOUR lane,kc,packfn
    add w0,w28,#\lane
    madd x0,x0,x23,x26
    .if \kc == D1_Q6_I16_KC
    add x1,x27,x28,lsl #10
    .else
    add x1,x27,x28,lsl #9
    .endif
    add x1,x1,#\lane*4
    mov x2,x23
    lsl x3,x24,#2
    mov w4,#\kc
    bl \packfn
.endm

/* x0 quantized rows, x1 FP32 panels, w2 rows, w3 channels,
 * w4 source row stride in bytes, w5 rows per tile (8 or 12).
 * Layout is [K block][row tile][channel][row lane]. */
.macro D1_FMLA_PACK_PANEL name,kc,group_shift,group_bytes,packfn
.type \name,%function
\name:
    ENTER
    mov x19,x0
    mov x20,x1
    mov w21,w2
    mov w22,w3
    mov w23,w4
    mov w24,w5
    mov w25,#0
.Lfmla_pack_k\@:
    lsr w0,w25,#\group_shift
    mov w1,#\group_bytes
    madd x26,x0,x1,x19
    mul w0,w25,w21
    add x27,x20,x0,lsl #2
    mov w28,#0
.Lfmla_pack_rows\@:
    D1_FMLA_PACK_FOUR 0,\kc,\packfn
    D1_FMLA_PACK_FOUR 4,\kc,\packfn
    cmp w24,#D1_FMLA_NR
    b.eq .Lfmla_pack_next_tile\@
    D1_FMLA_PACK_FOUR 8,\kc,\packfn
.Lfmla_pack_next_tile\@:
    add w28,w28,w24
    cmp w28,w21
    b.lo .Lfmla_pack_rows\@
    add w25,w25,#\kc
    cmp w25,w22
    b.lo .Lfmla_pack_k\@
    LEAVE
.size \name,.-\name
.endm

D1_FMLA_PACK_PANEL d1_fmla_pack_panel,D1_FMLA_KC,5,34,d1_pack_fmla_q8_4

.macro D1_FMLA_BATCH_WORK name,kc,compact=0
.type \name,%function
\name:
    ENTER
    ADDR x19,d1_batch_job
    add x0,x19,#60
1:  ldaxr w1,[x0]
    add w2,w1,#1
    stlxr w3,w2,[x0]
    cbnz w3,1b
    ldr w3,[x19,#52]
    lsr w3,w3,#2
    mul w20,w1,w3
    add w21,w20,w3
    .if !\compact
    ADDR x0,d1_fmla_weights
    mov x2,#D1_FMLA_WEIGHT_BYTES
    madd x23,x1,x2,x0
    .endif
    ADDR x0,d1_fmla_output
    ldr x2,=D1_FMLA_OUTPUT_BYTES
    madd x24,x1,x2,x0

.Lfmla_worker_n\@:
    sub w22,w21,w20
    mov w0,#D1_FMLA_NC
    cmp w22,w0
    csel w22,w22,w0,lo
    .if \compact
    ADDR x0,d1_q6_i16_for_job
    ldr x23,[x0]
    lsr w1,w20,#3
    ldr w0,=(7680/D1_Q6_I16_KC)*D1_Q6_I16_TILE_BYTES
    madd x23,x1,x0,x23
    .else
    ldr x0,[x19]
    mov x1,x23
    mov w2,w22
    ldr w3,[x19,#40]
    ldr w4,[x19,#48]
    mov w5,#D1_FMLA_NR
    madd x0,x20,x4,x0
    bl d1_fmla_pack_panel
    .endif
    mov w25,#0

.Lfmla_worker_p\@:
    ADDR x0,d1_fmla_positions
    ldr w28,[x0]
    sub w28,w28,w25
    D1_FMLA_ROWS_PER_PANEL w0,\compact
    cmp w28,w0
    csel w28,w28,w0,lo
    mov w0,#D1_FMLA_MR
    udiv w28,w28,w0
    mov w26,#0
.Lfmla_worker_k\@:
    mov w27,#0
.Lfmla_worker_tile\@:
    ADDR x0,d1_fmla_positions
    ldr w0,[x0]
    .if \compact
    lsr w1,w26,#8
    madd w1,w1,w0,w25
    mov w2,#D1_FMLA_MR
    udiv w1,w1,w2
    mov w2,#D1_Q6_I16_ACT_BYTES
    ADDR x0,d1_fmla_activations
    madd x0,x1,x2,x0
    .else
    lsl w1,w25,#7
    madd w1,w26,w0,w1
    ADDR x0,d1_fmla_activations
    add x0,x0,x1,lsl #2
    .endif
    .if \compact
    lsr w1,w26,#8
    madd w1,w1,w22,w27
    lsr w1,w1,#3
    mov w2,#D1_Q6_I16_TILE_BYTES
    madd x1,x1,x2,x23
    .else
    lsl w1,w27,#7
    madd w1,w26,w22,w1
    add x1,x23,x1,lsl #2
    .endif
    add x2,x24,x27,lsl #2
    lsl x3,x22,#2
    mov w4,#\kc
    cmp w26,#0
    cset w5,ne
    mov w6,w28
    .if \compact
    bl d1_q6_i16_exact_12x8
    .else
    bl d1_fmla_12x8
    .endif
    add w27,w27,#D1_FMLA_NR
    cmp w27,w22
    b.lo .Lfmla_worker_tile\@
    add w26,w26,#\kc
    ldr w0,[x19,#40]
    cmp w26,w0
    b.lo .Lfmla_worker_k\@

    /* The rounded tile is private. Scatter only the real output rows. */
    ldr w26,[x19,#56]
    sub w26,w26,w25
    D1_FMLA_ROWS_PER_PANEL w0,\compact
    cmp w26,w0
    csel w26,w26,w0,lo
    mov w27,#0
.Lfmla_worker_scatter\@:
    ldr x0,[x19,#16]
    add x0,x0,x20,lsl #2
    ldr x1,[x19,#32]
    add w2,w25,w27
    madd x0,x2,x1,x0
    lsl x2,x22,#2
    madd x1,x27,x2,x24
    bl memcpy
    add w27,w27,#1
    cmp w27,w26
    b.lo .Lfmla_worker_scatter\@
    D1_FMLA_ROWS_PER_PANEL w0,\compact
    add w25,w25,w0
    ldr w0,[x19,#56]
    cmp w25,w0
    b.lo .Lfmla_worker_p\@
    add w20,w20,w22
    cmp w20,w21
    b.lo .Lfmla_worker_n\@
    LEAVE
.size \name,.-\name
.endm

D1_FMLA_BATCH_WORK d1_fmla_batch_work,D1_FMLA_KC
D1_FMLA_BATCH_WORK d1_fmla_compact_batch_work,D1_Q6_I16_KC,1
.ltorg

.section .rodata
d1_vis_fmla_env: .asciz "D1_VISION_FMLA"
d1_fmla_mc_env: .asciz "D1_VISION_FMLA_MC"
d1_fmla_mc_error: .asciz "Vision FMLA panel height must be 36, 120 or 240.\n"
d1_fmla_mc_stats: .asciz "d1_vision_fmla_mc=%u\n"
.data
.balign 4
d1_fmla_mc: .word D1_FMLA_MC
.bss
.balign 64
d1_vis_fmla_enabled: .skip 4
d1_fmla_stage: .skip 4
d1_fmla_this_job: .skip 4
d1_fmla_positions: .skip 4
d1_fmla_calls: .skip 4
d1_fmla_pad_rows: .skip 4
.balign 64
d1_fmla_activations: .skip D1_FMLA_ACT_FLOATS*4
d1_fmla_weights: .skip 4*D1_FMLA_WEIGHT_BYTES
d1_fmla_output: .skip 4*D1_FMLA_OUTPUT_BYTES
.text
