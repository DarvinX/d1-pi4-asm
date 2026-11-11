/* Per-query attention mass and indexed matrix batches for D85. */
.text
d1_sparse_config:
    ENTER
    ADDR x0,d1_sparse_env
    bl getenv
    cbz x0,.Lsparse_config_done
    ldrb w0,[x0]
    cmp w0,#'1'
    b.ne .Lsparse_config_done
    ADDR x0,d1_tail_enabled /* implies all three batched stages */
    ldr w0,[x0]
    cbz w0,d1_decision_invalid
    ADDR x0,d1_ffn_reuse_enabled
    ldr w0,[x0]
    cbz w0,.Lsparse_config_enable
    cmp w0,#4
    b.ne d1_decision_invalid
    ADDR x0,d1_sparse_ffn_cutoff_env
    ADDR x1,d1_sparse_ffn_cutoff
    bl d1_read_cosine
.Lsparse_config_enable:
    mov w0,#1
    ADDR x1,d1_sparse_enabled
    str w0,[x1]
    ADDR x0,d1_frozen_ops_env
    bl getenv
    cbz x0,.Lsparse_late_ffn_config
    mov w0,#1
    ADDR x1,d1_frozen_ops_enabled
    str w0,[x1]
    ADDR x0,d1_frozen_ops_env
    ADDR x1,d1_frozen_ops_cosine
    bl d1_read_cosine
.Lsparse_late_ffn_config:
    ADDR x0,d1_late_ffn_env
    bl getenv
    cbz x0,.Lsparse_budget_config
    ldrb w0,[x0]
    sub w0,w0,#'0'
    sub w1,w0,#1
    cmp w1,#1
    b.hi .Lsparse_budget_config
    ADDR x1,d1_late_ffn_enabled
    str w0,[x1]
    ADDR x0,d1_late_ffn_cutoff_env
    ADDR x1,d1_late_ffn_cutoff
    bl d1_read_cosine
.Lsparse_budget_config:
    ADDR x0,d1_sparse_budget_env
    bl getenv
    cbz x0,.Lsparse_config_done
    .ifdef D1_BACKBONE_REDUCED_ROWS
    sub sp,sp,#16
    mov x1,sp
    mov w2,#10
    bl strtoul
    ldr x1,[sp]
    add sp,sp,#16
    ldrb w1,[x1]
    cbnz w1,d1_decision_invalid
    cmp x0,#8
    b.eq .Lsparse_budget_valid
    cmp x0,#12
    b.eq .Lsparse_budget_valid
    cmp x0,#16
    b.ne d1_decision_invalid
.Lsparse_budget_valid:
    mov w19,w0
    .else
    ADDR x1,d1_sparse_budget16
    bl strcmp
    cbnz w0,d1_decision_invalid
    .endif
    ADDR x0,d1_late_ffn_enabled
    ldr w0,[x0]
    cmp w0,#1
    b.ne d1_decision_invalid
    ADDR x1,d1_sparse_small
    .ifdef D1_BACKBONE_REDUCED_ROWS
    str w19,[x1]
    .else
    str w0,[x1]
    .endif
.Lsparse_config_done:
    ADDR x0,d1_early_conv_env
    bl getenv
    cbz x0,.Learly_conv_config_done
    ldrb w0,[x0]
    cmp w0,#'1'
    b.ne .Learly_conv_config_done
    ADDR x0,d1_sparse_enabled
    ldr w1,[x0]
    cbz w1,d1_decision_invalid
    ADDR x0,d1_early_conv_enabled
    str w1,[x0]
    ADDR x0,d1_early_conv_cosine_env
    ADDR x1,d1_early_conv_cosine
    bl d1_read_cosine
    ADDR x0,d1_early_conv_blend_env
    ADDR x1,d1_early_conv_blend
    bl d1_read_cosine
.Learly_conv_config_done:
    .ifdef D1_BACKBONE_TRANSITION_FFN
    ADDR x0,d1_transition_ffn_env
    bl getenv
    cbz x0,.Ltransition_ffn_config_done
    ldrb w0,[x0]
    sub w0,w0,#'1'
    cmp w0,#3
    b.hi .Ltransition_ffn_config_done
    ADDR x1,d1_sparse_small
    ldr w1,[x1]
    cbz w1,d1_decision_invalid
    cmp w0,#3
    b.ne .Ltransition_ffn_ranked
    ADDR x1,d1_ffn_reuse_enabled
    ldr w1,[x1]
    cmp w1,#4
    b.ne d1_decision_invalid
    mov w0,#4
    b .Ltransition_ffn_store
.Ltransition_ffn_ranked:
    lsl w0,w0,#3
    add w0,w0,#16
.Ltransition_ffn_store:
    ADDR x1,d1_transition_ffn_enabled
    str w0,[x1]
.Ltransition_ffn_config_done:
    .endif
    .ifdef D1_BACKBONE_DROP8
    ADDR x0,d1_sparse_drop_env
    bl getenv
    cbz x0,.Lsparse_drop_config_done
    sub sp,sp,#16
    mov x1,sp
    mov w2,#10
    bl strtoul
    ldr x1,[sp]
    add sp,sp,#16
    ldrb w1,[x1]
    cbnz w1,d1_decision_invalid
    cmp x0,#6
    ccmp x0,#10,#4,ne
    b.ne d1_decision_invalid
    ADDR x1,d1_sparse_small
    ldr w1,[x1]
    cmp w1,#16
    ccmp w1,#12,#4,ne
    b.ne d1_decision_invalid
    ADDR x1,d1_sparse_drop_after
    str w0,[x1]
.Lsparse_drop_config_done:
    .endif
    .ifdef D1_BACKBONE_DELAY_TEXT
    ADDR x0,d1_sparse_text_env
    bl getenv
    cbz x0,.Lsparse_text_config_done
    ADDR x1,d1_sparse_enabled
    ldr w1,[x1]
    cbz w1,d1_decision_invalid
    sub sp,sp,#16
    mov x1,sp
    mov w2,#10
    bl strtoul
    ldr x1,[sp]
    ldrb w2,[x1]
    cmp w2,#':'
    b.ne d1_decision_invalid
    sub x2,x0,#3
    cmp x2,#17
    b.hi d1_decision_invalid
    mov w20,w0
    add x0,x1,#1
    mov x1,sp
    mov w2,#10
    bl strtoul
    ldr x1,[sp]
    add sp,sp,#16
    ldrb w1,[x1]
    cbnz w1,d1_decision_invalid
    sub x2,x0,#7
    cmp x2,#57
    b.hi d1_decision_invalid
    ADDR x1,d1_sparse_text_config
    stp w20,w0,[x1]
.Lsparse_text_config_done:
    .endif
    LEAVE

.ifdef D1_BACKBONE_DELAY_TEXT
.type d1_sparse_update_budgets,%function
d1_sparse_update_budgets: /* w0 positions, w1 completed layer */
    mov w4,#0
    ADDR x2,d1_sparse_text_config
    ldp w3,w5,[x2]
    cmp w1,w3
    b.ne .Lsparse_update_image_budget
    cbz w3,.Lsparse_update_image_budget
    ADDR x2,d1_sparse_text_live_tail
    str w5,[x2]
    mov w4,#1
.Lsparse_update_image_budget:
    .ifdef D1_BACKBONE_DROP8
    ADDR x2,d1_sparse_drop_after
    ldr w3,[x2]
    cmp w1,w3
    b.ne .Lsparse_update_finish
    cbz w3,.Lsparse_update_finish
    ADDR x2,d1_sparse_small
    mov w3,#8
    str w3,[x2]
    mov w4,#1
    .endif
.Lsparse_update_finish:
    cbnz w4,d1_sparse_replan
    ret
.size d1_sparse_update_budgets,.-d1_sparse_update_budgets
.endif

.ifdef D1_BACKBONE_TRANSITION_FFN
.type d1_sparse_replan,%function
d1_sparse_replan: /* w0 prompt positions, ranking mass already summed */
    ENTER
    mov w19,w0
    ADDR x0,d1_image_first
    ldr w20,[x0]
    ADDR x22,d1_sparse_mass
    b .Lsparse_rank_mass
.size d1_sparse_replan,.-d1_sparse_replan
.endif

.type d1_sparse_finish_capture,%function
d1_sparse_finish_capture: /* w0 prompt positions */
    ENTER
    mov w19,w0
    ADDR x0,d1_sparse_capture
    str wzr,[x0]
    ADDR x0,d1_image_first
    ldr w20,[x0]
    add w21,w20,#300
    cmp w21,w19
    b.hs d1_decision_invalid
    ADDR x22,d1_sparse_mass
    ADDR x23,d1_sparse_query_mass
    mov w24,#1200
.Lsparse_sum_query:
    mov x0,x22
    madd x1,x21,x24,x23
    mov w2,#300
    bl lfm_add
    add w21,w21,#1
    cmp w21,w19
    b.lo .Lsparse_sum_query
.Lsparse_rank_mass:
    mov x0,x22
    mov w1,w20
    mov w2,w19
    ADDR x3,d1_sparse_mask
    ADDR x4,d1_sparse_active
    ADDR x5,d1_sparse_support
    ADDR x6,d1_sparse_sizes
    ADDR x7,d1_sparse_small
    ldr w8,[x7]
    cbnz w8,.Lsparse_plan_small
    ADDR x7,d1_late_ffn_enabled
    ldr w8,[x7]
    cbz w8,.Lsparse_plan_plain
    ADDR x7,d1_sparse_ranked
    cmp w8,#2
    b.eq .Lsparse_plan_ranked12
    bl d1_sparse_plan_ranked
    b .Lsparse_plan_done
.Lsparse_plan_ranked12:
    bl d1_sparse_plan_ranked12
    b .Lsparse_plan_done
.Lsparse_plan_small:
    ADDR x7,d1_sparse_ranked
    .ifdef D1_BACKBONE_REDUCED_ROWS
    mov w9,w8
    cmp w8,#8
    mov w8,#8
    mov w10,#4
    csel w8,w10,w8,eq
    .ifdef D1_BACKBONE_TRANSITION_FFN
    ADDR x10,d1_transition_ffn_budget
    ldr w10,[x10]
    cmp w10,#0
    csel w9,w10,w9,ne
    .endif
    bl d1_sparse_plan_budget_anchors
    b .Lsparse_plan_done
    .else
    .ifdef D1_BACKBONE_TRANSITION_FFN
    ADDR x9,d1_transition_ffn_budget
    ldr w9,[x9]
    cbz w9,.Lsparse_plan16
    bl d1_sparse_plan_budget8
    b .Lsparse_plan_done
.Lsparse_plan16:
    .endif
    bl d1_sparse_plan16
    b .Lsparse_plan_done
    .endif
.Lsparse_plan_plain:
    bl d1_sparse_plan
.Lsparse_plan_done:
    ADDR x22,d1_ffn_row_map
    mov x0,x22
    mov w1,#255
    lsl w2,w19,#2
    bl memset
    ADDR x23,d1_sparse_active
    ADDR x0,d1_sparse_sizes
    ldr w24,[x0]
    .ifdef D1_BACKBONE_TRANSITION_FFN
    ADDR x0,d1_transition_ffn_budget
    ldr w0,[x0]
    cbz w0,.Lsparse_transition_rows_done
    ADDR x0,d1_transition_ffn_rows
    str w24,[x0]
.Lsparse_transition_rows_done:
    .endif
    mov w0,#0
.Lsparse_map:
    ldr w1,[x23,x0,lsl #2]
    str w0,[x22,x1,lsl #2]
    add w0,w0,#1
    cmp w0,w24
    b.lo .Lsparse_map
    ADDR x0,d1_late_ffn_enabled
    ldr w5,[x0]
    cbz w5,.Lsparse_activate
    ADDR x0,d1_sparse_ranked
    ADDR x1,d1_late_ffn_reps
    mov w2,#0
    mov w3,#0
.Llate_rep:
    ldrb w4,[x0,x3]
    cbz w4,.Llate_rep_next
    add w4,w20,w3
    ldr w4,[x22,x4,lsl #2]
    str w4,[x1,x2,lsl #2]
    add w2,w2,#1
.Llate_rep_next:
    add w3,w3,#1
    cmp w3,#300
    b.lo .Llate_rep
    mov w3,#20
    mov w4,#12
    cmp w5,#2
    csel w3,w4,w3,eq
    ADDR x5,d1_sparse_small
    ldr w5,[x5]
    mov w4,#8
    .ifdef D1_BACKBONE_REDUCED_ROWS
    cmp w5,#8
    mov w6,#4
    csel w4,w6,w4,eq
    .endif
    cmp w5,#0
    csel w3,w4,w3,ne
    cmp w2,w3
    b.ne d1_decision_invalid
.Lsparse_activate:
    .ifdef D1_BACKBONE_TRANSITION_FFN
    ADDR x0,d1_transition_ffn_budget
    ldr w0,[x0]
    cbnz w0,.Lsparse_ready
    .endif
    ADDR x0,d1_frozen_ops_enabled
    ldr w0,[x0]
    cbz w0,.Lsparse_ready
    mov w0,w19
    mov w1,w20
    bl d1_prepare_frozen_ops
.Lsparse_ready:
    mov w0,#1
    ADDR x1,d1_sparse_live
    str w0,[x1]
    LEAVE
.size d1_sparse_finish_capture,.-d1_sparse_finish_capture

.type d1_prepare_frozen_ops,%function
d1_prepare_frozen_ops: /* positions, first image row */
    ENTER
    mov w19,w0
    mov w20,w1
    ADDR x21,d1_frozen_ops_sizes
    ADDR x22,d1_sparse_mask
    ADDR x23,d1_frozen_conv_skip
    ADDR x24,d1_sparse_support
    ADDR x0,d1_sparse_sizes
    ldr w25,[x0,#4]

    ADDR x0,d1_hidden
    mov x1,x22
    mov w2,w20
    mov w3,w19
    ADDR x4,d1_frozen_kv_owner
    ADDR x5,d1_frozen_ops_cosine
    ldr s0,[x5]
    bl d1_frozen_projection_plan
    ADDR x0,d1_frozen_kv_owner
    mov x1,xzr
    mov w2,w19
    ADDR x3,d1_frozen_kv_list
    bl d1_frozen_projection_list
    str w0,[x21]

    mov x0,x23
    mov w1,#1
    mov w2,w19
    bl memset
    mov w0,#0
.Lfrozen_conv_support:
    ldr w1,[x24,x0,lsl #2]
    ldrb w2,[x22,x1]
    strb w2,[x23,x1]
    add w0,w0,#1
    cmp w0,w25
    b.lo .Lfrozen_conv_support
    ADDR x0,d1_hidden
    mov x1,x23
    mov w2,w20
    mov w3,w19
    ADDR x4,d1_frozen_conv_owner
    ADDR x5,d1_frozen_ops_cosine
    ldr s0,[x5]
    bl d1_frozen_projection_plan
    ADDR x0,d1_frozen_conv_owner
    mov x1,x24
    mov w2,w25
    ADDR x3,d1_frozen_conv_list
    bl d1_frozen_projection_list
    str w0,[x21,#4]
    LEAVE
.size d1_prepare_frozen_ops,.-d1_prepare_frozen_ops

.type d1_matrix_batch_indexed,%function
d1_matrix_batch_indexed:
    /* desc, full input/output, row IDs, count, output stride or 0, weight ptr or 0 */
    ENTER
    sub sp,sp,#16
    str x6,[sp]
    mov x19,x0
    mov x20,x1
    mov x21,x2
    mov x22,x3
    mov w23,w4
    ldp w24,w25,[x19,#8]
    lsl x24,x24,#2
    lsl x25,x25,#2
    cmp x5,#0
    csel x26,x5,x25,ne
    ADDR x28,d1_sparse_input
    mov w27,#0
.Lsparse_gather:
    ldr w0,[x22,x27,lsl #2]
    madd x1,x0,x24,x20
    madd x0,x27,x24,x28
    mov x2,x24
    bl memcpy
    add w27,w27,#1
    cmp w27,w23
    b.lo .Lsparse_gather
    mov x0,x19
    mov x1,x28
    ADDR x28,d1_sparse_output
    mov x2,x28
    mov w3,w23
    ldr x4,[sp]
    bl d1_matrix_batch_ptr
    mov w27,#0
.Lsparse_scatter:
    ldr w0,[x22,x27,lsl #2]
    madd x0,x0,x26,x21
    madd x1,x27,x25,x28
    mov x2,x25
    bl memcpy
    add w27,w27,#1
    cmp w27,w23
    b.lo .Lsparse_scatter
    add sp,sp,#16
    LEAVE
.size d1_matrix_batch_indexed,.-d1_matrix_batch_indexed

.type d1_early_conv_project,%function
d1_early_conv_project: /* descriptor, input, output, count, scratch output, counter */
    ENTER
    mov x19,x0
    mov x20,x1
    mov x21,x2
    mov w22,w3
    mov x23,x4
    mov x24,x5
    ADDR x25,d1_early_conv_owner
    ADDR x26,d1_early_conv_rows
    mov x0,x20
    ADDR x1,d1_early_conv_skip /* all image rows are eligible */
    ADDR x2,d1_image_first
    ldr w2,[x2]
    mov w3,w22
    mov x4,x25
    ADDR x5,d1_early_conv_cosine
    ldr s0,[x5]
    bl d1_frozen_projection_plan
    mov x0,x25
    mov x1,xzr
    mov w2,w22
    mov x3,x26
    bl d1_frozen_projection_list
    mov w27,w0
    subs w0,w22,w27
    str w0,[x24]
    b.eq .Learly_conv_dense

    ADDR x28,d1_sparse_input
    mov w24,#0
.Learly_conv_gather:
    ldr w0,[x26,x24,lsl #2]
    add x1,x20,x0,lsl #13
    add x0,x28,x24,lsl #13
    mov x2,#DIM*4
    bl memcpy
    add w24,w24,#1
    cmp w24,w27
    b.lo .Learly_conv_gather
    mov x0,x19
    mov x1,x28
    mov x2,x23
    mov w3,w27
    bl d1_matrix_batch
    mov x0,x20
    mov x1,x23
    mov x2,x21
    mov x3,x25
    mov w4,w22
    ldr w5,[x19,#12]
    lsl x5,x5,#2
    ADDR x6,d1_early_conv_blend
    ldr s0,[x6]
    bl d1_scaled_projection_scatter
    LEAVE
.Learly_conv_dense:
    mov x0,x19
    mov x1,x20
    mov x2,x21
    mov w3,w22
    bl d1_matrix_batch
    LEAVE
.size d1_early_conv_project,.-d1_early_conv_project

.type d1_sparse_conv_input,%function
d1_sparse_conv_input: /* split B/C/X into 2048 output rows */
    ENTER
    sub sp,sp,#32
    mov x19,x0
    ldp x1,x2,[x19]
    ldp x3,x4,[x19,#16]
    stp x1,x2,[sp]
    stp x3,x4,[sp,#16]
    mov w1,#DIM
    str w1,[sp,#12]
    bl tensor_ptr
    mov x20,x0
    ldr w21,[x19,#20]
    lsl x21,x21,#11 /* packed weight bytes in one third */
    ADDR x22,d1_batch_gate
    ADDR x23,d1_sparse_sizes
    mov w24,#0
.Lsparse_conv_third:
    mov x0,sp
    ADDR x1,d1_batch_norm
    mov x2,x22
    mov w25,#0
    cmp w24,#1
    b.eq .Lsparse_conv_c
    ADDR x7,d1_frozen_ops_enabled
    ldr w25,[x7]
    cbnz w25,.Lsparse_conv_shared
    ADDR x3,d1_sparse_support
    ldr w4,[x23,#4]
    b .Lsparse_conv_run
.Lsparse_conv_shared:
    ADDR x3,d1_frozen_conv_list
    ADDR x4,d1_frozen_ops_sizes
    ldr w4,[x4,#4]
    b .Lsparse_conv_run
.Lsparse_conv_c:
    ADDR x3,d1_sparse_active
    ldr w4,[x23]
.Lsparse_conv_run:
    mov x5,#3*DIM*4
    mov x6,x20
    bl d1_matrix_batch_indexed
    cbz w25,.Lsparse_conv_next
    mov x0,x22
    ADDR x1,d1_frozen_conv_owner
    ADDR x2,d1_image_first
    ldr w2,[x2]
    mov w3,#3*DIM*4
    mov w4,#DIM*4
    bl d1_frozen_projection_scatter
.Lsparse_conv_next:
    add x20,x20,x21
    add x22,x22,#DIM*4
    add w24,w24,#1
    cmp w24,#3
    b.lo .Lsparse_conv_third
    add sp,sp,#32
    LEAVE
.size d1_sparse_conv_input,.-d1_sparse_conv_input

.section .rodata
d1_sparse_env: .asciz "D1_BACKBONE_FREEZE"
d1_sparse_budget_env: .asciz "D1_BACKBONE_ROW_BUDGET"
d1_sparse_budget16: .asciz "16"
.ifdef D1_BACKBONE_DELAY_TEXT
d1_sparse_text_env: .asciz "D1_BACKBONE_DELAY_TEXT"
d1_sparse_text_stats: .asciz "d1_sparse_text after=%u tail=%u frozen=%u\n"
.endif
.ifdef D1_BACKBONE_DROP8
d1_sparse_drop_env: .asciz "D1_BACKBONE_DROP8_AFTER"
.endif
d1_sparse_ffn_cutoff_env: .asciz "D1_EARLY_FFN_COSINE"
d1_late_ffn_env: .asciz "D1_LATE_FFN_REUSE"
d1_late_ffn_cutoff_env: .asciz "D1_LATE_FFN_COSINE"
d1_frozen_ops_env: .asciz "D1_BACKBONE_FROZEN_OPS"
d1_early_conv_env: .asciz "D1_EARLY_CONV_PROJECTION"
d1_early_conv_cosine_env: .asciz "D1_EARLY_CONV_COSINE"
d1_early_conv_blend_env: .asciz "D1_EARLY_CONV_NORM_BLEND"
.ifdef D1_BACKBONE_TRANSITION_FFN
d1_transition_ffn_env: .asciz "D1_BACKBONE_TRANSITION_FFN"
.endif
d1_sparse_counts: .asciz "d1_sparse_rows active=%u support=%u\n"
.data
.balign 4
d1_sparse_ffn_cutoff: .float 0.38
d1_late_ffn_cutoff: .float 0.35
d1_frozen_ops_cosine: .float 0.45
d1_early_conv_cosine: .float 0.75
d1_early_conv_blend: .float 0.0
.bss
.balign 64
d1_sparse_query_mass: .skip CTX*300*4
d1_sparse_mass: .skip 300*4
d1_sparse_input: .skip CTX*DIM*4
d1_sparse_output: .skip CTX*DIM*4
d1_sparse_mask: .skip CTX
d1_sparse_ranked: .skip 300
.balign 4
d1_late_ffn_reps: .skip 20*4
d1_late_ffn_enabled: .skip 4
d1_late_ffn_layers: .skip 4
d1_late_ffn_reused: .skip 4
d1_sparse_active: .skip CTX*4
d1_sparse_support: .skip CTX*4
d1_sparse_sizes: .skip 8
d1_sparse_small: .skip 4
.ifdef D1_BACKBONE_DELAY_TEXT
d1_sparse_text_config: .skip 8
.global d1_sparse_text_live_tail
d1_sparse_text_live_tail: .skip 4
.endif
.ifdef D1_BACKBONE_DROP8
d1_sparse_drop_after: .skip 4
.endif
.ifdef D1_BACKBONE_TRANSITION_FFN
d1_transition_ffn_enabled: .skip 4
d1_transition_ffn_budget: .skip 4
d1_transition_ffn_rows: .skip 4
d1_transition_ffn_map: .skip CTX*4
.endif
d1_frozen_ops_enabled: .skip 4
d1_frozen_ops_sizes: .skip 8
d1_frozen_kv_owner: .skip CTX*4
d1_frozen_conv_owner: .skip CTX*4
d1_frozen_kv_list: .skip CTX*4
d1_frozen_conv_list: .skip CTX*4
d1_frozen_conv_skip: .skip CTX
.balign 16
d1_early_conv_enabled: .skip 4
d1_early_conv_counts: .skip 16
d1_early_conv_owner: .skip CTX*4
d1_early_conv_rows: .skip CTX*4
d1_early_conv_skip: .skip CTX
d1_sparse_enabled: .skip 4
d1_sparse_capture: .skip 4
d1_sparse_live: .skip 4
.text
