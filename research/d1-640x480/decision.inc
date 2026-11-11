/* d1 640×480 decision interface.
 * The caller renders the prompt with one media marker. The runtime
 * tokenizes both text spans, encodes the pixels, and scores the labels. */
.equ D1_DECISION_MAX_OPTIONS,255
.equ D1_DECISION_MAX_LABELS,3*D1_DECISION_MAX_OPTIONS
.equ D1_DECISION_PLAN_BYTES,16+4*(D1_DECISION_MAX_OPTIONS+D1_DECISION_MAX_LABELS)
.text
.type d1_decision_driver,%function
d1_decision_driver:
    ENTER
    mov x19,x0
    ldr x0,[x19,#48]
    ldrb w1,[x0]
    cmp w1,#'@'
    b.eq .Ldecision_plan
    ADDR x1,d1_choice
    bl strcmp
    cbz w0,.Lchoice_three
    ldr x0,[x19,#48]
    ADDR x1,d1_choice_two
    bl strcmp
    cbnz w0,.Ltry_noul
    mov w20,#2
    mov w21,#4
    b .Lchoice_labels
.Lchoice_three:
    mov w20,#3
    mov w21,#6
.Lchoice_labels:
    ADDR x22,d1_choice_labels
    ADDR x23,d1_choice_counts
    b .Ldecision_load
.Ltry_noul:
    ldr x0,[x19,#48]
    ADDR x1,d1_noul
    bl strcmp
    cbnz w0,d1_decision_invalid
    mov w20,#2
    mov w21,#5
    ADDR x22,d1_noul_labels
    ADDR x23,d1_noul_counts
    b .Ldecision_load
.Ldecision_plan:
    add x0,x0,#1
    bl d1_load_label_plan
    mov x23,x0
    mov x22,x1
    mov w20,w2
    mov w21,w3
.Ldecision_load:
    ldr x0,[x19,#32]
    bl d1_load_rgb
    ldr x0,[x19,#40]
    bl d1_decision_load_prompt
    bl hardware_check
    .ifdef D1_PARALLEL_OTHER_PANEL_CRC
    bl d1_read_other_panel_crc
    .endif
    .ifdef D1_MODEL_EMBED_ROW_CRC
    bl d1_read_embedding_crc
    .endif
    .ifdef D1_DEFER_MODEL_CRC
    bl d1_read_deferred_crc
    .endif
    .ifdef D1_DEFER_VISION_CRC
    bl d1_read_deferred_vision_crc
    .endif
    .ifdef D1_RETAIN_VERIFIED_PAGES
    bl d1_read_retained_pages
    .endif
    ldr x0,[x19,#16]
    bl map_model
    .ifdef D1_ASYNC_BACKBONE_PANELS
    bl d1_begin_backbone_panels
    .else
    bl d1_q6_i16_map
    bl d1_q4_i8_map
    .endif
    ldr x0,[x19,#24]
    bl d1_vision_map
    mov w1,#1
    ADDR x0,use_q8
    str w1,[x0]
    ADDR x0,d1_ffn_enabled
    str w1,[x0]
    ADDR x0,d1_decision_active
    str w1,[x0]
    bl d1_read_conv_flag
    bl d1_read_attn_flag
    bl d1_read_tail_flag
    bl d1_read_batch_flags
    ADDR x0,d1_ffn_reuse_env
    bl getenv
    cbz x0,.Ldecision_init_pool
    ldrb w0,[x0]
    cmp w0,#'1'
    cset w1,eq
    cmp w0,#'4'
    mov w0,#4
    csel w0,w0,w1,eq
    ADDR x1,d1_ffn_reuse_enabled
    str w0,[x1]
    cmp w0,#0
    cset w0,ne
    ADDR x1,d1_ffn_rows
    str w0,[x1] /* the final FFN has one scalar output row */
.Ldecision_init_pool:
    .ifdef D1_SPARSE_WORK
    bl d1_sparse_config
    .endif
    ADDR x0,d1_decision_profile_env
    bl getenv
    mov w24,#0
    cbz x0,.Ldecision_config_vision
    ldrb w0,[x0]
    cmp w0,#'1'
    cset w24,eq
.Ldecision_config_vision:
    ADDR x0,d1_decision_profile_enabled
    str w24,[x0]
    bl d1_vision_config
    bl init_pool
    .ifdef D1_PARALLEL_ACTIVATIONS
    bl d1_read_parallel_activations
    .endif
    .ifdef D1_PARALLEL_PACK
    bl d1_read_parallel_pack
    .endif

    ADDR x0,d1_start
    bl clock_stamp
    ADDR x0,d1_vis_rgb
    bl d1_vision_encode
    cbz w24,.Ldecision_prompt
    ADDR x0,d1_decision_vision_end
    bl clock_stamp
.Ldecision_prompt:
    ADDR x0,d1_decision_prompt
    bl d1_decision_embed
    cbz w24,.Ldecision_forward
    ADDR x0,d1_decision_prompt_end
    bl clock_stamp
.Ldecision_forward:
    .ifdef D1_ASYNC_BACKBONE_PANELS
    bl d1_join_backbone_panels
    .endif
    bl d1_forward
    ADDR x0,embedding
    bl tensor_ptr
    .ifdef D1_MODEL_EMBED_ROW_CRC
    mov x1,x22
    mov w2,w21
    bl d1_verify_label_rows
    .endif
    mov x1,x0
    ADDR x0,normalized
    mov x2,x22
    ADDR x3,d1_decision_logits
    mov w4,w21
    ADDR x5,activation_q8
    bl d1_label_logits
    ADDR x0,d1_decision_logits
    mov x1,x23
    mov w2,w20
    ADDR x3,d1_decision_probs
    bl d1_decision_softmax
    ADDR x0,d1_end
    bl clock_stamp
    ADDR x0,d1_start
    ADDR x1,d1_end
    bl seconds_between
    mov w0,#2
    ADDR x1,d1_decision_time
    ADDR x2,d1_count
    ldr w2,[x2]
    bl dprintf
    ADDR x0,d1_ffn_reuse_enabled
    ldr w0,[x0]
    cbz w0,.Ldecision_vision_counts
    mov w0,#2
    ADDR x1,d1_ffn_reuse_counts
    ADDR x2,d1_ffn_rows
    ldp w2,w3,[x2]
    bl dprintf
.Ldecision_vision_counts:
    ADDR x0,d1_vis_ffn_reuse_enabled
    ldr w0,[x0]
    cbz w0,.Ldecision_keys
    mov w0,#2
    ADDR x1,d1_vis_ffn_reuse_counts
    mov w2,#(14*1200)
    ADDR x3,d1_vis_ffn_reused
    ldr w3,[x3]
    bl dprintf
.Ldecision_keys:
    ADDR x0,d1_vis_adaptive_keys_enabled
    ldr w0,[x0]
    cbz w0,.Ldecision_profile
    mov w0,#2
    ADDR x1,d1_decision_key_counts
    ADDR x2,d1_vis_attended_keys
    ldr w2,[x2]
    bl dprintf
.Ldecision_profile:
    cbz w24,.Ldecision_output
    bl d1_report_decision_phases
.Ldecision_output:
    .ifdef D1_SPARSE_WORK
    ADDR x0,d1_sparse_enabled
    ldr w0,[x0]
    cbz w0,.Ldecision_sparse_done
    mov w0,#2
    ADDR x1,d1_sparse_counts
    ADDR x2,d1_sparse_sizes
    ldp w2,w3,[x2]
    bl dprintf
.Ldecision_sparse_done:
    .endif
    ADDR x0,d1_vis_group_enabled
    ldr w0,[x0]
    cbz w0,.Ldecision_write
    mov w0,#2
    ADDR x1,d1_decision_vision_groups
    ADDR x2,d1_vis_group_blocks
    ldp w2,w4,[x2]
    VIS_ROWS w3
    bl dprintf
.Ldecision_write:
    ADDR x0,d1_vis_early_conv_enabled
    ldr w0,[x0]
    cbz w0,.Ldecision_late_ffn
    mov w0,#2
    ADDR x1,d1_decision_early_groups
    ADDR x2,d1_vis_early_group_counts
    ldr w4,[x2,#8]
    ldp w2,w3,[x2]
    bl dprintf
.Ldecision_late_ffn:
    .ifdef D1_SPARSE_WORK
    ADDR x0,d1_late_ffn_enabled
    ldr w0,[x0]
    cbz w0,.Ldecision_frozen_ops
    mov w0,#2
    ADDR x1,d1_decision_late_ffn
    ADDR x2,d1_late_ffn_layers
    ldp w2,w3,[x2]
    bl dprintf
.Ldecision_frozen_ops:
    ADDR x0,d1_frozen_ops_enabled
    ldr w0,[x0]
    cbz w0,.Ldecision_early_conv
    mov w0,#2
    ADDR x1,d1_decision_frozen_ops
    ADDR x2,d1_count
    ldr w2,[x2]
    ADDR x3,d1_sparse_sizes
    ldr w3,[x3,#4]
    ADDR x4,d1_frozen_ops_sizes
    ldp w4,w5,[x4]
    sub w2,w2,w4
    sub w3,w3,w5
    bl dprintf
.Ldecision_early_conv:
    ADDR x0,d1_early_conv_enabled
    ldr w0,[x0]
    cbz w0,.Ldecision_emit
    mov w0,#2
    ADDR x1,d1_decision_early_conv
    ADDR x2,d1_early_conv_counts
    ldp w4,w5,[x2,#8]
    ldp w2,w3,[x2]
    bl dprintf
.Ldecision_emit:
    .endif
    ADDR x0,d1_vis_fmla_enabled
    ldr w0,[x0]
    cbz w0,.Ldecision_vision_reuse
    mov w0,#2
    ADDR x1,d1_decision_fmla
    ADDR x2,d1_fmla_calls
    ldp w2,w3,[x2]
    bl dprintf
.Ldecision_vision_reuse:
    ADDR x0,d1_vis_pack_reuse_enabled
    ldr w0,[x0]
    cbz w0,.Ldecision_write_probs
    mov w0,#2
    ADDR x1,d1_decision_vision_reuse
    ADDR x2,d1_vis_pack_reuse_counts
    ldp w2,w3,[x2]
    bl dprintf
.Ldecision_write_probs:
    bl d1_report_value12
    bl d1_q6_i16_report
    bl d1_q4_i8_report
    bl d1_q8_i8_report
    bl d1_q8_ilp_report
    bl d1_q8_res_report
    bl d1_q4_i8_slice_report
    ADDR x0,d1_vis_stem_group_count
    ldr w2,[x0]
    cbz w2,.Ldecision_probability_bytes
    mov w0,#2
    ADDR x1,d1_decision_stem_groups
    bl dprintf
.Ldecision_probability_bytes:
    bl d1_vis_route_report
    bl d1_vis_score12_report
    bl d1_fmla_mc_report
    .ifdef D1_SPARSE_WORK
    .ifdef D1_BACKBONE_TRANSITION_FFN
    ADDR x0,d1_transition_ffn_rows
    ldr w2,[x0]
    cbz w2,.Ldecision_write_final
    mov w0,#2
    ADDR x1,d1_decision_transition_ffn
    bl dprintf
.Ldecision_write_final:
    .endif
    .endif
    .ifdef D1_VISION_COMPACT_DW
    ADDR x0,d1_vis_dw_quad_layers
    ldr w2,[x0]
    cbz w2,.Ldecision_emit_probs
    mov w0,#2
    ADDR x1,d1_vis_dw_quad_stats
    bl dprintf
.Ldecision_emit_probs:
    .endif
    .ifdef D1_VISION_QUERY_GROUP
    bl d1_vis_report_query_groups
    .endif
    .ifdef D1_VISION_FFN_MASKED
    ADDR x0,d1_vis_ffn_mask
    ldr w2,[x0]
    cbz w2,.Ldecision_unmasked_ffn
    mov w0,#2
    ADDR x1,d1_vis_ffn_mask_stats
    ADDR x3,d1_vis_ffn_lower_cosine
    ldr s0,[x3]
    fcvt d0,s0
    bl dprintf
.Ldecision_unmasked_ffn:
    .endif
    .ifdef D1_Q8_RES_REUSE
    ADDR x0,d1_q8_res_reuse_count
    ldr w2,[x0]
    cbz w2,.Ldecision_no_res_reuse
    mov w0,#2
    ADDR x1,d1_q8_res_reuse_stats
    bl dprintf
.Ldecision_no_res_reuse:
    .endif
    .ifdef D1_VISION_LATE_FFN
    ADDR x0,d1_vis_late_ffn_counts
    ldp w2,w3,[x0]
    cbz w2,.Ldecision_no_late_vision_ffn
    mov w0,#2
    ADDR x1,d1_vis_late_ffn_stats
    bl dprintf
.Ldecision_no_late_vision_ffn:
    .endif
    .ifdef D1_VISION_EARLY_KV
    ADDR x0,d1_vis_early_kv_mode
    ldr w0,[x0]
    cbz w0,.Ldecision_no_early_kv
    ADDR x0,d1_vis_early_kv_counts
    ldp w2,w3,[x0]
    ldr w4,[x0,#8]
    mov w0,#2
    ADDR x1,d1_vis_early_kv_stats
    bl dprintf
.Ldecision_no_early_kv:
    .endif
    .ifdef D1_PARALLEL_ACTIVATIONS
    ADDR x0,d1_activation_counts
    ldp w2,w3,[x0]
    cbz w2,.Ldecision_no_parallel_activations
    ldr w4,[x0,#8]
    mov w0,#2
    ADDR x1,d1_parallel_activations_stats
    bl dprintf
.Ldecision_no_parallel_activations:
    .endif
    .ifdef D1_PARALLEL_PACK
    ADDR x0,d1_raw_pack_counts
    ldp w2,w3,[x0]
    cbz w2,.Ldecision_no_parallel_pack
    mov w0,#2
    ADDR x1,d1_parallel_pack_stats
    bl dprintf
.Ldecision_no_parallel_pack:
    .endif
    .ifdef D1_BACKBONE_DELAY_TEXT
    ADDR x0,d1_sparse_text_live_tail
    ldr w3,[x0]
    cbz w3,.Ldecision_no_text_freeze
    ADDR x0,d1_sparse_text_config
    ldr w2,[x0]
    ADDR x0,d1_image_first
    ldr w1,[x0]
    ADDR x0,d1_count
    ldr w4,[x0]
    sub w4,w4,w1
    sub w4,w4,#300
    subs w4,w4,w3
    csel w4,w4,wzr,hi
    mov w0,#2
    ADDR x1,d1_sparse_text_stats
    bl dprintf
.Ldecision_no_text_freeze:
    .endif
    .ifdef D1_PARALLEL_Q4_PANEL_CRC
    ADDR x0,d1_q4_panel_shards
    ldr w2,[x0]
    cbz w2,.Ldecision_no_q4_panel_shards
    mov w0,#2
    ADDR x1,d1_q4_panel_shards_stats
    bl dprintf
.Ldecision_no_q4_panel_shards:
    .endif
    .ifdef D1_PARALLEL_OTHER_PANEL_CRC
    ADDR x0,d1_other_crc_shards
    ldr w2,[x0]
    .ifdef D1_ASYNC_BACKBONE_PANELS
    ADDR x0,d1_q6_crc_shards
    ldr w3,[x0]
    add w2,w2,w3
    .endif
    cbz w2,.Ldecision_no_other_panel_shards
    mov w0,#2
    ADDR x1,d1_other_crc_stats
    bl dprintf
.Ldecision_no_other_panel_shards:
    .endif
    .ifdef D1_MODEL_EMBED_ROW_CRC
    ADDR x0,d1_embed_crc_rows
    ldr w2,[x0]
    cbz w2,.Ldecision_no_embed_crc
    mov w0,#2
    ADDR x1,d1_embed_crc_stats
    bl dprintf
.Ldecision_no_embed_crc:
    .endif
    .ifdef D1_Q4_SEED_PRODUCT
    ADDR x0,d1_q4_seed_jobs
    ldr w2,[x0]
    cbz w2,.Ldecision_no_q4_seed_jobs
    mov w0,#2
    ADDR x1,d1_q4_seed_stats
    bl dprintf
.Ldecision_no_q4_seed_jobs:
    .endif
    .ifdef D1_Q4_GATE_UP_REUSE
    ADDR x0,d1_q4_gate_up_reuse_counts
    ldp w2,w3,[x0]
    cbz w2,.Ldecision_no_q4_gate_up
    mov w0,#2
    ADDR x1,d1_q4_gate_up_reuse_stats
    bl dprintf
.Ldecision_no_q4_gate_up:
    .endif
    .ifdef D1_DEFER_MODEL_CRC
    ADDR x0,d1_defer_enabled
    ldr w0,[x0]
    cbz w0,.Ldecision_no_deferred_crc
    ADDR x0,d1_defer_counts
    ldp w2,w3,[x0]
    mov w0,#2
    ADDR x1,d1_defer_stats
    bl dprintf
.Ldecision_no_deferred_crc:
    .endif
    .ifdef D1_DEFER_VISION_CRC
    ADDR x0,d1_defer_vis_enabled
    ldr w0,[x0]
    cbz w0,.Ldecision_no_deferred_vision_crc
    ADDR x0,d1_defer_vis_counts
    ldp w2,w3,[x0]
    mov w0,#2
    ADDR x1,d1_defer_vis_stats
    bl dprintf
.Ldecision_no_deferred_vision_crc:
    .endif
    .ifdef D1_RETAIN_VERIFIED_PAGES
    ADDR x0,d1_retain_pages_enabled
    ldr w0,[x0]
    cbz w0,.Ldecision_no_retained_pages
    ADDR x0,d1_retain_pages_counts
    ldp w2,w3,[x0]
    mov w0,#2
    ADDR x1,d1_retain_pages_stats
    bl dprintf
.Ldecision_no_retained_pages:
    .endif
    .ifdef D1_VISION_LATE_FFN_TUNE
    ADDR x0,d1_vis_late_mask_used
    ldr w2,[x0]
    cbz w2,.Ldecision_no_late_ffn_mask
    mov w0,#2
    ADDR x1,d1_vis_late_mask_stats
    bl dprintf
.Ldecision_no_late_ffn_mask:
    .endif
    .ifdef D1_ASYNC_BACKBONE_PANELS
    ADDR x0,d1_async_loader_joined
    ldr w0,[x0]
    cbz w0,.Ldecision_no_async_backbone
    ADDR x0,d1_async_loader_nice
    ldr w2,[x0]
    ADDR x0,d1_async_join_seconds
    ldr d0,[x0]
    mov w0,#2
    ADDR x1,d1_async_loader_stats
    bl dprintf
.Ldecision_no_async_backbone:
    .endif
    mov w0,#1
    ADDR x1,d1_decision_probs
    lsl x2,x20,#3
    bl write_all
    mov w0,#0
    LEAVE
.size d1_decision_driver,.-d1_decision_driver

.type d1_report_decision_phases,%function
d1_report_decision_phases:
    ENTER
    ADDR x0,d1_start
    ADDR x1,d1_decision_vision_end
    bl seconds_between
    fmov d8,d0
    ADDR x0,d1_decision_vision_end
    ADDR x1,d1_decision_prompt_end
    bl seconds_between
    fmov d9,d0
    ADDR x0,d1_decision_prompt_end
    ADDR x1,d1_end
    bl seconds_between
    fmov d2,d0
    fmov d0,d8
    fmov d1,d9
    mov w0,#2
    ADDR x1,d1_decision_phase_times
    bl dprintf
    ADDR x0,d1_backbone_part_totals
    ldp d0,d1,[x0]
    ldp d2,d3,[x0,#16]
    mov w0,#2
    ADDR x1,d1_decision_backbone_times
    bl dprintf
    LEAVE
.size d1_report_decision_phases,.-d1_report_decision_phases

.type d1_charge_backbone_part,%function
d1_charge_backbone_part:
    ENTER
    mov w19,w0
    ADDR x0,d1_backbone_part_end
    bl clock_stamp
    ADDR x0,d1_backbone_part_start
    ADDR x1,d1_backbone_part_end
    bl seconds_between
    ADDR x0,d1_backbone_part_totals
    ldr d1,[x0,x19,lsl #3]
    fadd d0,d0,d1
    str d0,[x0,x19,lsl #3]
    ADDR x0,d1_backbone_part_end
    ldp x1,x2,[x0]
    ADDR x0,d1_backbone_part_start
    stp x1,x2,[x0]
    LEAVE
.size d1_charge_backbone_part,.-d1_charge_backbone_part

.type d1_decision_load_prompt,%function
d1_decision_load_prompt:
    ENTER
    ADDR x1,d1_read_mode
    bl fopen
    cbz x0,file_error
    mov x19,x0
    ADDR x20,d1_decision_prompt
    mov x0,x20
    mov x1,#1
    mov x2,#65536
    mov x3,x19
    bl fread
    cbz x0,d1_decision_invalid
    cmp x0,#65536
    b.hs d1_decision_invalid
    mov x21,x0
    mov x0,x19
    bl ferror
    cbnz w0,file_error
    mov x0,x19
    bl fclose
    mov x0,x20
    mov w1,#0
    mov x2,x21
    bl memchr
    cbnz x0,d1_decision_invalid
    strb wzr,[x20,x21]
    LEAVE
.size d1_decision_load_prompt,.-d1_decision_load_prompt

.type d1_load_label_plan,%function
d1_load_label_plan: /* D1LABEL1, u32 option/label counts, group sizes, token IDs */
    ENTER
    ADDR x1,d1_read_mode
    bl fopen
    cbz x0,file_error
    mov x19,x0
    ADDR x20,d1_label_plan
    mov x0,x20
    mov x1,#1
    mov x2,#D1_DECISION_PLAN_BYTES+1
    mov x3,x19
    bl fread
    mov x21,x0
    mov x0,x19
    bl ferror
    cbnz w0,file_error
    mov x0,x19
    bl fclose
    cmp x21,#16
    b.lo d1_decision_invalid
    ldr x0,[x20]
    ADDR x1,d1_label_plan_magic
    ldr x1,[x1]
    cmp x0,x1
    b.ne d1_decision_invalid
    ldp w22,w23,[x20,#8]
    sub w0,w22,#1
    cmp w0,#D1_DECISION_MAX_OPTIONS-1
    b.hi d1_decision_invalid
    add w0,w22,w22,lsl #1
    cmp w23,w0
    b.hi d1_decision_invalid
    add x0,x22,x23
    lsl x0,x0,#2
    add x0,x0,#16
    cmp x0,x21
    b.ne d1_decision_invalid
    add x0,x20,#16
    mov x1,x0
    mov w4,w22
    mov w5,#0
.Llabel_plan_group:
    ldr w6,[x1],#4
    sub w7,w6,#1
    cmp w7,#2
    b.hi d1_decision_invalid
    add w5,w5,w6
    subs w4,w4,#1
    b.ne .Llabel_plan_group
    cmp w5,w23
    b.ne d1_decision_invalid
    mov x4,x1
    mov w5,w23
    ldr w6,=VOCAB
.Llabel_plan_token:
    ldr w7,[x4],#4
    cmp w7,w6
    b.hs d1_decision_invalid
    subs w5,w5,#1
    b.ne .Llabel_plan_token
    mov w2,w22
    mov w3,w23
    LEAVE
.size d1_load_label_plan,.-d1_load_label_plan

.type d1_decision_embed,%function
d1_decision_embed: /* mutable prompt with exactly one <__media__> */
    ENTER
    mov x19,x0
    ADDR x1,d1_media_marker
    bl strstr
    cbz x0,d1_decision_invalid
    mov x20,x0
    add x21,x0,#11
    mov x0,x21
    ADDR x1,d1_media_marker
    bl strstr
    cbnz x0,d1_decision_invalid
    strb wzr,[x20]
    mov x0,x19
    bl tokenize
    mov w1,w0
    ADDR x0,input_ids
    mov w2,#0
    bl d1_append_embeddings
    mov w2,w0
    ADDR x0,d1_vision_start_id
    mov w1,#1
    bl d1_append_embeddings
    mov w22,w0
    ADDR x0,d1_image_first
    str w22,[x0]
    add w23,w22,#300
    cmp w23,#CTX-1
    b.hs context_error
    ADDR x0,d1_hidden
    add x0,x0,x22,lsl #13
    ADDR x1,d1_vis_output
    ldr x2,=300*DIM*4
    bl memcpy
    ADDR x0,d1_vision_end_id
    mov w1,#1
    mov w2,w23
    bl d1_append_embeddings
    mov w22,w0
    mov x0,x21
    bl tokenize
    mov w1,w0
    ADDR x0,input_ids
    mov w2,w22
    bl d1_append_embeddings
    ADDR x1,d1_count
    str w0,[x1]
    LEAVE
.size d1_decision_embed,.-d1_decision_embed

.type d1_append_embeddings,%function
d1_append_embeddings: /* ids, count, initial position -> next position */
    ENTER
    mov x19,x0
    mov w20,w1
    add w21,w1,w2
    cmp w21,#CTX
    b.hi context_error
    cbz w20,.Lappend_done
    ADDR x22,d1_hidden
    add x22,x22,x2,lsl #13
    ADDR x0,embedding
    bl tensor_ptr
    mov x23,x0
    mov x24,#1680 /* tied Q6_K row */
.Lappend_emb:
    ldr w0,[x19],#4
    .ifdef D1_MODEL_EMBED_ROW_CRC
    mov w1,w0
    .endif
    madd x0,x0,x24,x23
    .ifdef D1_MODEL_EMBED_ROW_CRC
    bl d1_verify_embedding_row
    .endif
    mov x1,x22
    mov w2,#DIM
    bl lfm_q6_unpack
    add x22,x22,#DIM*4
    subs w20,w20,#1
    b.ne .Lappend_emb
.Lappend_done:
    mov w0,w21
    LEAVE
.size d1_append_embeddings,.-d1_append_embeddings

.type d1_decision_softmax,%function
d1_decision_softmax: /* label logits, count per option, number of options, probs */
    ENTER
    mov x19,x3
    mov w20,w2
    mov w21,#0
    mov w5,#0xff800000
    fmov s8,w5
.Loption:
    ldr w4,[x1],#4
    fmov s0,w5
.Loption_max:
    ldr s1,[x0],#4
    fmax s0,s0,s1
    subs w4,w4,#1
    b.ne .Loption_max
    fmax s8,s8,s0
    fcvt d0,s0
    str d0,[x19,x21,lsl #3]
    add w21,w21,#1
    cmp w21,w20
    b.lo .Loption
    fcvt d8,s8
    ldr d9,.Lcalibration
    fmov d10,xzr
    mov w21,#0
.Lprob_exp:
    ldr d0,[x19,x21,lsl #3]
    fsub d0,d0,d8
    fdiv d0,d0,d9
    bl exp
    fadd d10,d10,d0
    str d0,[x19,x21,lsl #3]
    add w21,w21,#1
    cmp w21,w20
    b.lo .Lprob_exp
    mov w21,#0
.Lprob_div:
    ldr d0,[x19,x21,lsl #3]
    fdiv d0,d0,d10
    str d0,[x19,x21,lsl #3]
    add w21,w21,#1
    cmp w21,w20
    b.lo .Lprob_div
    LEAVE
.size d1_decision_softmax,.-d1_decision_softmax

d1_decision_invalid:
    ADDR x0,d1_decision_error
    b fatal
.balign 8
.Lcalibration: .double 1.3373984098434448
.ltorg
.section .rodata
d1_decision_flag: .asciz "--decision"
d1_choice: .asciz "choice"
d1_choice_two: .asciz "choice2"
d1_noul: .asciz "noul"
d1_media_marker: .asciz "<__media__>"
d1_decision_time: .asciz "d1_decision_s=%.9f positions=%u\n"
d1_decision_profile_env: .asciz "D1_DECISION_PROFILE"
d1_decision_phase_times: .asciz "d1_phases_s vision=%.9f prompt=%.9f backbone_score=%.9f\n"
d1_decision_backbone_times: .asciz "d1_backbone_s conv=%.9f attention=%.9f ffn=%.9f tail=%.9f\n"
d1_ffn_reuse_env: .asciz "D1_FFN_REUSE"
d1_ffn_reuse_counts: .asciz "d1_ffn_rows=%u reused=%u\n"
d1_vis_ffn_reuse_counts: .asciz "d1_vision_ffn_rows=%u reused=%u\n"
d1_decision_key_counts: .asciz "d1_vision_keys=7200 attended=%u\n"
d1_decision_vision_groups: .asciz "d1_vision_groups=%u rows=%u late_queries=%u\n"
d1_decision_early_groups: .asciz "d1_vision_early_groups=%u,%u,%u\n"
d1_decision_late_ffn: .asciz "d1_late_ffn_layers=%u reused=%u\n"
d1_decision_frozen_ops: .asciz "d1_frozen_ops kv=%u conv=%u\n"
d1_decision_stem_groups: .asciz "d1_vision_stem_groups=%u\n"
d1_decision_early_conv: .asciz "d1_early_conv_projection=%u,%u,%u,%u\n"
d1_decision_fmla: .asciz "d1_vision_fmla batches=%u pad_rows=%u\n"
d1_decision_vision_reuse: .asciz "d1_vision_pack_reuse q8=%u fmla=%u\n"
.ifdef D1_BACKBONE_TRANSITION_FFN
d1_decision_transition_ffn: .asciz "d1_transition_ffn_rows=%u\n"
.endif
d1_decision_error: .asciz "Invalid d1 prompt or label plan. Expected choice, choice2, noul, or @LABELS.bin.\n"
.balign 8
d1_label_plan_magic: .ascii "D1LABEL1"
.balign 4
d1_vision_start_id: .word 125009
d1_vision_end_id: .word 125010
d1_choice_labels: .word 41,334,42,378,43,340 /* A, B, C with/without a space */
d1_choice_counts: .word 2,2,2
d1_noul_labels: .word 11683,12447,2243,4547,19598 /* yes/Yes, no/No/NO */
d1_noul_counts: .word 2,3
.bss
.balign 8
d1_decision_vision_end: .skip 16
d1_decision_prompt_end: .skip 16
d1_backbone_part_start: .skip 16
d1_backbone_part_end: .skip 16
d1_backbone_part_totals: .skip 32
d1_decision_logits: .skip D1_DECISION_MAX_LABELS*4
.balign 8
d1_decision_probs: .skip D1_DECISION_MAX_OPTIONS*8
d1_label_plan: .skip D1_DECISION_PLAN_BYTES+1
.balign 4
d1_decision_active: .skip 4
d1_decision_profile_enabled: .skip 4
d1_ffn_reuse_enabled: .skip 4
d1_image_first: .skip 4
d1_ffn_rows: .skip 4
d1_ffn_reused_rows: .skip 4
d1_ffn_row_map: .skip CTX*4
d1_decision_prompt: .skip 65536
.text
