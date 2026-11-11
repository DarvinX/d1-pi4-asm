/* Experimental layer-major d1 backbone from preassembled FP32 embeddings. */
#ifdef D1_BATCH
.equ D1_BATCH_WORK,1
#endif
#ifdef D1_VISION
.equ D1_VISION_WORK,1
#endif
#if defined(D1_VISION) && defined(D1_SPARSE_VISUAL)
.equ D1_SPARSE_WORK,1
#endif
#ifdef D1_VISION_PROFILE
.equ D1_VISION_PROFILE_WORK,1
#endif
#ifdef D1_FFN_FIXTURE
.equ D1_FFN_FIXTURE_WORK,1
#endif
#define H34_DRIVER_MAIN 1
#include "experiments/runtime-h42.S"
#undef main

.if DIM != 2048 || HEADS != 16 || HD != 128 || KV != 4 || KW != 512
.error "unexpected d1 backbone shape"
.endif

.text
.balign 64
.global main
.type main,%function
/* MODEL.gguf EMBEDDINGS.f32 COUNT -> one complete FP32 logit vector on stdout. */
main:
    ENTER
#ifdef D1_VISION
    cmp w0,#7
    b.ne 1f
    mov x19,x1
    ldr x0,[x19,#8]
    ADDR x1,d1_decision_flag
    bl strcmp
    cbnz w0,.Lusage
    mov x0,x19
    bl d1_decision_driver
    LEAVE
1:
#endif
    cmp w0,#4
    b.ne .Lusage
    mov x19,x1
#ifdef D1_VISION
    ldr x0,[x19,#8]
    ADDR x1,d1_vis_flag
    bl strcmp
    cbnz w0,1f
    mov x0,x19
    bl d1_vision_driver
    LEAVE
1:
#endif
    ldr x0,[x19,#24]
    mov x1,#0
    mov w2,#10
    bl strtol
    cmp x0,#1
    b.lt .Lusage
    cmp x0,#CTX
    b.hi .Lusage
    mov w20,w0
    ADDR x1,d1_count
    str w20,[x1]
    ldr x0,[x19,#16]
    ADDR x1,d1_read_mode
    bl fopen
    cbz x0,.Lusage
    mov x21,x0
    ADDR x0,d1_hidden
    mov x1,#DIM*4
    mov w2,w20
    mov x3,x21
    bl fread
    cmp x0,x20
    b.ne .Lbad_input
    mov x0,x21
    bl fgetc
    cmn w0,#1
    b.ne .Lbad_input
    mov x0,x21
    bl fclose
    bl hardware_check
    ldr x0,[x19,#8]
    bl map_model
    ADDR x0,use_q8
    mov w1,#1
    str w1,[x0]
#ifdef D1_BATCH
    ADDR x0,d1_batch_env
    bl getenv
    cbz x0,1f
    ldrb w0,[x0]
    cmp w0,#'1'
    cset w0,eq
    ADDR x1,d1_ffn_enabled
    str w0,[x1]
1:
    bl d1_read_conv_flag
    bl d1_read_attn_flag
    bl d1_read_tail_flag
    bl d1_read_batch_flags
#endif
    bl init_pool
#ifdef D1_FFN_FIXTURE
    bl d1_ffn_fixture
    mov w0,#0
    LEAVE
#endif
#ifdef D1_TIMING
    ADDR x0,d1_start
    bl clock_stamp
#endif
    bl d1_forward
#ifdef D1_TIMING
    ADDR x0,d1_end
    bl clock_stamp
    ADDR x0,d1_start
    ADDR x1,d1_end
    bl seconds_between
    mov w0,#2
    ADDR x1,d1_timing_format
    bl dprintf
#endif
    mov w0,#1
    ADDR x1,logits
    ldr x2,=VOCAB*4
    bl write_all
    mov w0,#0
    LEAVE
.Lbad_input:
    mov x0,x21
    bl fclose
.Lusage:
    mov w0,#2
    ADDR x1,d1_usage
    mov x2,#d1_usage_size
    bl write_all
    mov w0,#2
    LEAVE
.size main,.-main

/* Iterate all positions at one layer. Dense attention must project all
 * keys and values before producing any result for that layer. */
.type d1_forward,%function
d1_forward:
    ENTER
    ADDR x0,d1_count
    ldr w23,[x0]
#ifdef D1_VISION
    ADDR x0,d1_decision_profile_enabled
    ldr w28,[x0]
#endif
#ifdef D1_BATCH
    ADDR x0,d1_tail_enabled
    ldr w26,[x0]
    ADDR x0,d1_attn_enabled
    ldr w0,[x0]
    cbz w0,1f
    mov w0,w23
    bl d1_prepare_rope
1:
#endif
    mov w21,#0
.Llayer:
    .ifdef D1_SPARSE_WORK
    cmp w21,#2
    b.ne .Lsparse_check_tail
    ADDR x0,d1_sparse_enabled
    ldr w1,[x0]
    ADDR x0,d1_sparse_capture
    str w1,[x0]
.Lsparse_check_tail:
    cmp w21,#LAYERS-2
    b.ne .Lsparse_layer_ready
    ADDR x0,d1_sparse_live
    str wzr,[x0]
.Lsparse_layer_ready:
    .endif
#ifdef D1_VISION
    cbz w28,.Lbackbone_layer_start
    ADDR x0,d1_backbone_part_start
    bl clock_stamp
.Lbackbone_layer_start:
#endif
    ADDR x19,layer_tensors
    mov x0,#576
    madd x19,x21,x0,x19
    ADDR x0,layer_types
    ldr w22,[x0,w21,uxtw #2]
    mov w20,#0
#ifdef D1_BATCH
    mov w25,#0
    cbz w26,2f
    cmp w21,#LAYERS-2
    b.lo 2f
    subs w0,w23,#CONV
    csel w25,w0,wzr,hi
    mov w20,w25
2:
    mov w24,#0
    cbnz w22,.Lmaybe_attn_batch
    ADDR x0,d1_conv_enabled
    ldr w0,[x0]
    cbz w0,.Lapply
    mov x0,x19
    sub w1,w23,w25
    mov w2,w25
    mov w3,#0
    cmp w21,#LAYERS-1
    csel w3,w26,w3,eq
    cbz w3,3f
    sub w20,w23,#1
3:
    mov w4,w21
    bl d1_conv_batch
    mov w24,#1
    b .Lapply
.Lmaybe_attn_batch:
    ADDR x0,d1_attn_enabled
    ldr w0,[x0]
    cbz w0,.Lproject_all
    mov x0,x19
    mov w1,w23
    mov w2,w22
    mov w3,w25
    bl d1_attention_batch
    mov w24,#1
    b .Lapply
#else
    cbz w22,.Lapply
#endif
.Lproject_all:
    mov x0,x19
    mov w1,w20
    mov w2,w22
    bl d1_project
    add w20,w20,#1
    cmp w20,w23
    b.lo .Lproject_all
    mov w20,#0
.Lapply:
    .ifdef D1_SPARSE_WORK
    ADDR x0,d1_sparse_live
    ldr w0,[x0]
    cbz w0,.Lapply_live
    ADDR x0,d1_sparse_mask
    ldrb w0,[x0,x20]
    cbz w0,.Lnext_pos
.Lapply_live:
    .endif
#ifdef D1_BATCH
    cbz w24,.Lscalar_operator
    ADDR x0,operator_out
    ADDR x1,d1_batch_gate
    add x1,x1,x20,lsl #13
    mov x2,#DIM*4
    bl memcpy
    b .Lfinish
.Lscalar_operator:
#endif
    cbnz w22,.Lattend
    ADDR x1,d1_hidden
    add x1,x1,x20,lsl #13
    mov x0,x19
    ADDR x2,normalized
    mov w3,#DIM
    bl norm
    mov x0,x19
    mov w1,w21
    mov w2,w20
    bl convolution
    b .Lfinish
.Lattend:
    mov x0,x19
    mov w1,w20
    mov w2,w22
    mov w3,w23
    bl d1_reduce
.Lfinish:
    /* The pinned graph only runs the last FFN for requested output rows. */
    cmp w21,#LAYERS-1
    b.ne .Lfinish_one
    add w0,w20,#1
    cmp w0,w23
    b.ne .Lnext_pos
.Lfinish_one:
#ifdef D1_BATCH
    cmp w21,#LAYERS-1
    b.eq .Lscalar_finish
    ADDR x0,d1_ffn_enabled
    ldr w0,[x0]
    cbz w0,.Lscalar_finish
    mov x0,x19
    ADDR x1,d1_hidden
    add x1,x1,x20,lsl #13
    bl d1_operator_residual
    b .Lnext_pos
.Lscalar_finish:
#endif
    mov x0,x19
    ADDR x1,d1_hidden
    add x1,x1,x20,lsl #13
    bl d1_finish
.Lnext_pos:
    add w20,w20,#1
    cmp w20,w23
    b.lo .Lapply
#ifdef D1_BATCH
    cmp w21,#LAYERS-1
    b.eq .Lbackbone_scalar_part
    ADDR x0,d1_ffn_enabled
    ldr w0,[x0]
    cbz w0,.Lbackbone_scalar_part
#ifdef D1_VISION
    cbz w28,.Lbackbone_ffn
    cmp w22,#0
    cset w0,ne /* convolution = 0, attention = 1 */
    bl d1_charge_backbone_part
.Lbackbone_ffn:
#endif
    .ifdef D1_SPARSE_WORK
    .ifdef D1_BACKBONE_TRANSITION_FFN
    cmp w21,#2
    b.ne .Lbackbone_ffn_ready
    ADDR x0,d1_sparse_capture
    ldr w0,[x0]
    cbz w0,.Lbackbone_ffn_ready
    ADDR x0,d1_transition_ffn_enabled
    ldr w0,[x0]
    cbz w0,.Lbackbone_ffn_ready
    /* The full attention residual is ready and rank inputs are complete. */
    ADDR x1,d1_transition_ffn_budget
    .ifdef D1_BACKBONE_REDUCED_ROWS
    ADDR x2,d1_sparse_small
    ldr w2,[x2]
    cmp w0,#4
    .else
    cmp w0,#4
    mov w2,#16
    .endif
    csel w0,w2,w0,eq
    str w0,[x1]
    mov w0,w23
    bl d1_sparse_finish_capture
.Lbackbone_ffn_ready:
    .endif
    .endif
    mov x0,x19
    sub w1,w23,w25
    mov w2,w25
    .ifdef D1_SPARSE_WORK
    mov w3,w21
    .endif
    bl d1_ffn_batch
#ifdef D1_VISION
    cbz w28,.Lfinished_layer
    mov w0,#2
    bl d1_charge_backbone_part
#endif
    b .Lfinished_layer
.Lbackbone_scalar_part:
#ifdef D1_VISION
    cbz w28,.Lfinished_layer
    mov w0,#3
    bl d1_charge_backbone_part
#endif
.Lfinished_layer:
#endif
    .ifdef D1_SPARSE_WORK
    cmp w21,#2
    b.ne .Lsparse_after_capture
    .ifdef D1_BACKBONE_TRANSITION_FFN
    ADDR x0,d1_transition_ffn_budget
    ldr w1,[x0]
    cbz w1,.Lsparse_capture_dense
    str wzr,[x0]
    mov w0,w23
    bl d1_sparse_replan
    b .Lsparse_after_capture
.Lsparse_capture_dense:
    .endif
    ADDR x0,d1_sparse_capture
    ldr w0,[x0]
    cbz w0,.Lsparse_after_capture
    mov w0,w23
    bl d1_sparse_finish_capture
.Lsparse_after_capture:
    .ifdef D1_BACKBONE_DELAY_TEXT
    mov w0,w23
    mov w1,w21
    bl d1_sparse_update_budgets
    .else
    .ifdef D1_BACKBONE_DROP8
    ADDR x0,d1_sparse_drop_after
    ldr w0,[x0]
    cbz w0,.Lsparse_drop_done
    cmp w0,w21
    b.ne .Lsparse_drop_done
    ADDR x0,d1_sparse_small
    mov w1,#8
    str w1,[x0]
    mov w0,w23
    bl d1_sparse_replan
.Lsparse_drop_done:
    .endif
    .endif
    .ifdef D1_FROZEN_BASIS_FIXTURE
    cmp w21,#2
    b.ne .Lfrozen_basis_fixture_done
    mov w0,w23
    bl d1_write_frozen_basis_fixture
.Lfrozen_basis_fixture_done:
    .endif
    .endif
#ifdef D1_TRACE
    mov w0,#3
    ADDR x1,d1_hidden
    sub w2,w23,#1
    add x1,x1,x2,lsl #13
    mov x2,#DIM*4
    bl write_all
#endif
    add w21,w21,#1
    cmp w21,#LAYERS
    b.lo .Llayer

    ADDR x0,final_norm
    ADDR x1,d1_hidden
    sub w20,w23,#1
    add x1,x1,x20,lsl #13
    ADDR x2,normalized
    mov w3,#DIM
    bl norm
#ifdef D1_VISION
    ADDR x0,d1_decision_active
    ldr w0,[x0]
    cbnz w0,.Ldecision_done
#endif
    ADDR x0,output_tensor
    ADDR x1,normalized
    ADDR x2,logits
    bl matrix
#ifdef D1_VISION
.Ldecision_done:
#endif
    LEAVE
.size d1_forward,.-d1_forward

.ifdef D1_FROZEN_BASIS_FIXTURE
.type d1_write_frozen_basis_fixture,%function
/* Header u32: positions, first image row, active rows, support rows,
 * compact K/V rows, compact convolution rows. Then fixed raster data. */
d1_write_frozen_basis_fixture:
    ENTER
    sub sp,sp,#32
    ADDR x1,d1_image_first
    ldr w19,[x1]
    stp w0,w19,[sp]
    ADDR x1,d1_sparse_sizes
    ldp w2,w20,[x1]
    stp w2,w20,[sp,#8]
    ADDR x1,d1_frozen_ops_sizes
    ldp w2,w3,[x1]
    stp w2,w3,[sp,#16]
    mov w0,#3
    mov x1,sp
    mov x2,#24
    bl write_all
    mov w0,#3
    ADDR x1,d1_hidden
    add x1,x1,x19,lsl #13
    ldr x2,=300*2048*4
    bl write_all
    mov w0,#3
    ADDR x1,d1_sparse_mask
    add x1,x1,x19
    mov x2,#300
    bl write_all
    mov w0,#3
    ADDR x1,d1_frozen_kv_owner
    add x1,x1,x19,lsl #2
    mov x2,#300*4
    bl write_all
    mov w0,#3
    ADDR x1,d1_frozen_conv_owner
    add x1,x1,x19,lsl #2
    mov x2,#300*4
    bl write_all
    mov w0,#3
    ADDR x1,d1_sparse_support
    lsl x2,x20,#2
    bl write_all
    add sp,sp,#32
    LEAVE
.size d1_write_frozen_basis_fixture,.-d1_write_frozen_basis_fixture
.endif

.macro D1_SAVE src, dst, stride
    ADDR x0,\dst
    add x0,x0,x20,lsl #\stride
    ADDR x1,\src
    mov x2,#(1 << \stride)
    bl memcpy
.endm

.type d1_project,%function
d1_project: /* layer descriptor, position, 1=full or 2=sliding */
    ENTER
    mov x19,x0
    mov w20,w1
    mov w21,w2
    ADDR x1,d1_hidden
    add x1,x1,x20,lsl #13
    ADDR x2,normalized
    mov w3,#DIM
    bl norm
    MAT 10,normalized,query
    MAT 11,normalized,key
    MAT 12,normalized,value
    MAT 16,normalized,head_gates
    mov w22,#0
.Lnorm_query:
    add x0,x19,#448
    ADDR x1,query
    add x1,x1,x22,lsl #9
    mov x2,x1
    mov w3,#HD
    bl norm
    add w22,w22,#1
    cmp w22,#HEADS
    b.lo .Lnorm_query
    mov w22,#0
.Lnorm_key:
    add x0,x19,#480
    ADDR x1,key
    add x1,x1,x22,lsl #9
    mov x2,x1
    mov w3,#HD
    bl norm
    add w22,w22,#1
    cmp w22,#KV
    b.lo .Lnorm_key

    mov w0,w20
    bl make_rope
    ADDR x22,cos_global
    ADDR x23,sin_global
    cmp w21,#2
    b.ne .Lrotate
    ADDR x22,cos_window
    ADDR x23,sin_window
.Lrotate:
    ADDR x0,query
    mov x1,x22
    mov x2,x23
    mov w3,#HEADS
    bl rotate
    ADDR x0,key
    mov x1,x22
    mov x2,x23
    mov w3,#KV
    bl rotate
    D1_SAVE query,d1_queries,13
    D1_SAVE key,d1_keys,11
    D1_SAVE value,d1_values,11
    D1_SAVE head_gates,d1_gates,6
    LEAVE
.size d1_project,.-d1_project

.type d1_reduce,%function
d1_reduce: /* layer descriptor, query position, attention type, total positions */
    ENTER
    mov x19,x0
    mov w0,w1
    mov w1,w2
    mov w2,w3
    ADDR x3,attn_out
    ADDR x4,d1_scores
    bl d1_attention_row
    MAT 13,attn_out,operator_out
    LEAVE
.size d1_reduce,.-d1_reduce

.type d1_attention_row,%function
d1_attention_row: /* query position, attention type, positions, out, score scratch */
    ENTER
    mov w20,w0
    mov w21,#0 /* key start */
    mov w22,w2 /* key count */
    mov x27,x3
    mov x26,x4
    cmp w1,#2
    b.ne .Lkey_span
    add w22,w20,#1
    subs w21,w22,#WINDOW
    csel w21,w21,wzr,ge
    sub w22,w22,w21
.Lkey_span:
    .ifdef D1_SPARSE_WORK
    mov x28,xzr
    ADDR x0,d1_sparse_capture
    ldr w0,[x0]
    cbz w0,.Lsparse_no_mass
    ADDR x0,d1_image_first
    ldr w0,[x0]
    add w0,w0,#300
    cmp w20,w0
    b.lo .Lsparse_no_mass
    ADDR x28,d1_sparse_query_mass
    mov w0,#1200
    madd x28,x20,x0,x28
.Lsparse_no_mass:
    .endif
    mov w23,#0 /* query head */
.Lhead:
    ADDR x0,d1_queries
    add x0,x0,x20,lsl #13
    add x0,x0,x23,lsl #9
    ADDR x1,d1_keys
    ADDR x2,d1_values
    add x1,x1,x21,lsl #11
    add x2,x2,x21,lsl #11
    lsr w24,w23,#2
    add x1,x1,x24,lsl #9
    add x2,x2,x24,lsl #9
    add x25,x27,x23,lsl #9
    mov x3,x25
    mov x4,x26
    mov w5,w22
    mov x6,#KW*4
    bl d1_attention_head
    .ifdef D1_SPARSE_WORK
    cbz x28,.Lsparse_mass_done
    ADDR x0,d1_image_first
    ldr w1,[x0]
    add x1,x26,x1,lsl #2
    mov x0,x28
    mov w2,#300
    bl lfm_add
.Lsparse_mass_done:
    .endif

    ADDR x0,d1_gates
    add x0,x0,x20,lsl #6
    ldr s0,[x0,x23,lsl #2]
    fneg s0,s0
    bl expf
    fmov s1,#1.0
    fadd s0,s0,s1
    fdiv s0,s1,s0
    dup v16.4s,v0.s[0]
    mov w1,#HD
.Lsigmoid:
    ldr q0,[x25]
    fmul v0.4s,v0.4s,v16.4s
    str q0,[x25],#16
    subs w1,w1,#4
    b.ne .Lsigmoid
    add w23,w23,#1
    cmp w23,#HEADS
    b.lo .Lhead
    LEAVE
.size d1_attention_row,.-d1_attention_row

.type d1_operator_residual,%function
d1_operator_residual: /* layer descriptor, hidden row */
    ENTER
    mov x19,x0
    mov x20,x1
    ldr x0,[x19,#64]
    cbz x0,.Loperator_residual
    NORM 2,operator_out,operator_out,DIM
.Loperator_residual:
    mov x0,x20
    ADDR x1,operator_out
    mov w2,#DIM
    bl lfm_add
    LEAVE
.size d1_operator_residual,.-d1_operator_residual

.type d1_finish,%function
d1_finish: /* layer descriptor, hidden row */
    ENTER
    mov x19,x0
    mov x20,x1
    bl d1_operator_residual
    add x0,x19,#32
    mov x1,x20
    ADDR x2,normalized
    mov w3,#DIM
    bl norm
    MAT 4,normalized,ff_gate
    MAT 5,normalized,ff_up
    ADDR x0,ff_gate
    ADDR x1,ff_up
    mov w2,#FF
    bl lfm_swiglu_native
    MAT 6,ff_gate,operator_out
    ldr x0,[x19,#96]
    cbz x0,.Lff_residual
    NORM 3,operator_out,operator_out,DIM
.Lff_residual:
    mov x0,x20
    ADDR x1,operator_out
    mov w2,#DIM
    bl lfm_add
    LEAVE
.size d1_finish,.-d1_finish

#ifdef D1_BATCH
.include "research/d1-640x480/backbone_batch.inc"
.include "research/d1-640x480/backbone_conv_batch.inc"
.include "research/d1-640x480/backbone_attn_batch.inc"
#endif
    .ifdef D1_SPARSE_WORK
.include "research/d1-640x480/backbone_sparse.inc"
    .endif
#ifdef D1_VISION
#include D1_VISION_INDEX
.include "research/d1-640x480/vision_encode.inc"
.include "research/d1-640x480/vision_fmla_batch.inc"
.include "research/d1-640x480/backbone_panel_map.inc"
.include "research/d1-640x480/embedding_crc.inc"
.include "research/d1-640x480/deferred_model_crc.inc"
.include "research/d1-640x480/vision_q8_i8_job.inc"
.include "research/d1-640x480/vision_q8_ilp_job.inc"
.include "research/d1-640x480/vision_q8_resident_job.inc"
.include "research/d1-640x480/backbone_q6_i16.inc"
.include "research/d1-640x480/backbone_q4_i8.inc"
.include "research/d1-640x480/async_backbone_panels.inc"
.include "research/d1-640x480/decision.inc"
#endif

.section .rodata
d1_read_mode: .asciz "rb"
#ifdef D1_VISION
d1_usage: .ascii "Usage: d1-vision-asm --vision PROJECTOR.gguf IMAGE.rgb\n"
          .ascii "   or: d1-vision-asm --decision MODEL.gguf PROJECTOR.gguf IMAGE.rgb PROMPT.txt choice|choice2|noul|@LABELS.bin\n"
          .ascii "   or: d1-vision-asm MODEL.gguf EMBEDDINGS.f32 COUNT\n"
#else
d1_usage: .ascii "Usage: d1-backbone MODEL.gguf EMBEDDINGS.f32 COUNT\n"
#endif
.equ d1_usage_size,.-d1_usage
#ifdef D1_TIMING
d1_timing_format: .asciz "d1_forward_s=%.9f\n"
#endif
.bss
.balign 64
d1_hidden: .skip CTX*DIM*4
d1_queries: .skip CTX*DIM*4
d1_keys: .skip CTX*KW*4
d1_values: .skip CTX*KW*4
d1_gates: .skip CTX*HEADS*4
d1_scores: .skip ((CTX+3)&~3)*4
d1_count: .skip 4
#ifdef D1_TIMING
.balign 8
d1_start: .skip 16
d1_end: .skip 16
#endif
.section .note.GNU-stack,"",%progbits
