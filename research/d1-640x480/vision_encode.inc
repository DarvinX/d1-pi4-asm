/* LFM3-VL still-image encoder, fixed at the original 640×480 geometry.
 * All graph dispatch and inference math are ARM64. libc supplies file I/O
 * and the worker primitives shared with the d1 backbone. */
.text
.type d1_vision_driver,%function
d1_vision_driver: /* argv: PROGRAM --vision PROJECTOR.gguf IMAGE.rgb */
    ENTER
    mov x19,x0
    ldr x0,[x19,#24]
    bl d1_load_rgb
    bl hardware_check
    ldr x0,[x19,#16]
    bl d1_vision_map
    bl d1_vision_config
    bl init_pool
    ADDR x0,d1_start
    bl clock_stamp
    ADDR x0,d1_vis_rgb
    bl d1_vision_encode
    ADDR x0,d1_end
    bl clock_stamp
    ADDR x0,d1_start
    ADDR x1,d1_end
    bl seconds_between
    mov w0,#2
    ADDR x1,d1_vis_timing
    bl dprintf
    .ifdef D1_VISION_PROFILE_WORK
    bl d1_vision_report
    .endif
    bl d1_report_value12
    bl d1_q8_i8_report
    bl d1_q8_ilp_report
    bl d1_q8_res_report
    bl d1_vis_route_report
    bl d1_vis_score12_report
    bl d1_fmla_mc_report
    .ifdef D1_VISION_COMPACT_DW
    ADDR x0,d1_vis_dw_quad_layers
    ldr w2,[x0]
    cbz w2,.Lvis_driver_write
    mov w0,#2
    ADDR x1,d1_vis_dw_quad_stats
    bl dprintf
.Lvis_driver_write:
    .endif
    mov w0,#1
    ADDR x1,d1_vis_output
    ldr x2,=300*2048*4
    bl write_all
    mov w0,#0
    LEAVE
.size d1_vision_driver,.-d1_vision_driver

.type d1_load_rgb,%function
d1_load_rgb:
    ENTER
    ADDR x1,d1_read_mode
    bl fopen
    cbz x0,file_error
    mov x20,x0
    ADDR x0,d1_vis_rgb
    mov x1,#1
    ldr x2,=640*480*3
    mov x3,x20
    bl fread
    ldr x1,=640*480*3
    cmp x0,x1
    b.ne file_error
    mov x0,x20
    bl fgetc
    cmn w0,#1
    b.ne file_error
    mov x0,x20
    bl fclose
    LEAVE
.size d1_load_rgb,.-d1_load_rgb

d1_vision_config:
    ENTER
    ADDR x0,d1_vis_attn4_env
    bl getenv
    cbz x0,1f
    ldrb w0,[x0]
    cmp w0,#'1'
    cset w0,eq
    ADDR x1,d1_vis_attn4_enabled
    str w0,[x1]
1:  ADDR x0,d1_vis_pair_env
    bl getenv
    cbz x0,2f
    ldrb w0,[x0]
    cmp w0,#'1'
    cset w0,eq
    ADDR x1,d1_vis_pair_enabled
    str w0,[x1]
2:  ADDR x0,d1_vis_ffn_reuse_env
    bl getenv
    cbz x0,3f
    ldrb w0,[x0]
    cmp w0,#'1'
    cset w0,eq
    ADDR x1,d1_vis_ffn_reuse_enabled
    str w0,[x1]
3:  ADDR x0,d1_vis_q8_panel_env
    bl getenv
    cbz x0,4f
    ldrb w0,[x0]
    cmp w0,#'1'
    cset w0,eq
    ADDR x1,d1_vis_q8_panel_enabled
    str w0,[x1]
4:  ADDR x0,d1_vis_score_packed_env
    bl getenv
    cbz x0,5f
    ldrb w0,[x0]
    cmp w0,#'1'
    cset w0,eq
    ADDR x1,d1_vis_score_packed_enabled
    str w0,[x1]
5:  ADDR x0,d1_vis_adaptive_keys_env
    bl getenv
    cbz x0,6f
    ldrb w0,[x0]
    cmp w0,#'1'
    cset w2,eq
    cmp w0,#'2'
    mov w0,#2
    csel w0,w0,w2,eq
    cbz w0,6f
    ADDR x1,d1_vis_adaptive_keys_enabled
    str w0,[x1]
    mov w0,#1
    ADDR x1,d1_vis_attn4_enabled
    str w0,[x1]
6:  ADDR x0,d1_vis_group_env
    bl getenv
    cbz x0,.Lvis_config_done
    ldrb w0,[x0]
    cmp w0,#'1'
    b.ne .Lvis_config_done
    ADDR x0,d1_vis_adaptive_keys_enabled
    ldr w0,[x0]
    cmp w0,#2
    b.ne .Lvis_group_bad_flags
    ADDR x0,d1_vis_ffn_reuse_enabled
    ldr w0,[x0]
    cbz w0,.Lvis_group_bad_flags
    ADDR x0,d1_vis_group_enabled
    mov w1,#1
    str w1,[x0]
.Lvis_config_done:
    ADDR x0,d1_vis_early_conv_env
    bl getenv
    cbz x0,.Lvis_q8_config
    ldrb w0,[x0]
    cmp w0,#'1'
    b.ne .Lvis_q8_config
    ADDR x0,d1_vis_group_enabled
    ldr w0,[x0]
    cbz w0,.Lvis_group_bad_flags
    ADDR x1,d1_vis_early_conv_enabled
    str w0,[x1]
    ADDR x0,d1_vis_early_cosine_env
    ADDR x1,d1_vis_early_cosine
    bl d1_read_cosine
.Lvis_q8_config:
    ADDR x0,d1_vis_stem_group_env
    ADDR x1,d1_vis_stem_group_cosine
    bl d1_read_cosine
    ADDR x0,d1_vis_q8_ilp_env
    bl getenv
    cbz x0,.Lvis_flags_done
    ldrb w0,[x0]
    cmp w0,#'1'
    cset w0,eq
    ADDR x1,d1_vis_q8_ilp_enabled
    str w0,[x1]
.Lvis_flags_done:
    ADDR x0,d1_vis_fmla_env
    bl getenv
    cbz x0,.Lvis_fmla_done
    ldrb w0,[x0]
    sub w0,w0,#'0'
    cmp w0,#3
    csel w0,w0,wzr,ls
    ADDR x1,d1_vis_fmla_enabled
    str w0,[x1]
.Lvis_fmla_done:
    bl d1_read_fmla_mc
    ADDR x0,d1_vis_pack_reuse_env
    bl getenv
    cbz x0,.Lvis_pack_reuse_done
    ldrb w0,[x0]
    cmp w0,#'1'
    cset w0,eq
    ADDR x1,d1_vis_pack_reuse_enabled
    str w0,[x1]
.Lvis_pack_reuse_done:
    ADDR x0,d1_vis_value12_env
    bl getenv
    cbz x0,.Lvis_value12_done
    ldrb w0,[x0]
    cmp w0,#'1'
    cset w0,eq
    ADDR x1,d1_vis_value12_enabled
    str w0,[x1]
.Lvis_value12_done:
    ADDR x0,d1_vis_route_env
    bl getenv
    cbz x0,.Lvis_route_config_done
    ldrb w0,[x0]
    cmp w0,#'1'
    cset w1,eq
    cmp w0,#'6'
    mov w0,#6
    csel w0,w0,w1,eq
    ADDR x1,d1_vis_route_enabled
    str w0,[x1]
.Lvis_route_config_done:
    ADDR x0,d1_vis_score12_env
    bl getenv
    cbz x0,.Lvis_score12_config_done
    ldrb w0,[x0]
    cmp w0,#'1'
    cset w0,eq
    ADDR x1,d1_vis_score12_enabled
    str w0,[x1]
.Lvis_score12_config_done:
    .ifdef D1_VISION_COMPACT_DW
    ADDR x0,d1_vis_dw_quad_env
    bl getenv
    cbz x0,.Lvis_dw_quad_config_done
    ldrb w0,[x0]
    cmp w0,#'1'
    cset w0,eq
    ADDR x1,d1_vis_dw_quad_enabled
    str w0,[x1]
.Lvis_dw_quad_config_done:
    .endif
    .ifdef D1_VISION_QUERY_GROUP
    ADDR x0,d1_vis_query_group_env
    ADDR x1,d1_vis_query_cutoff
    bl d1_read_cosine
    .endif
    .ifdef D1_VISION_FFN_MASKED
    ADDR x0,d1_vis_ffn_base_cosine_env
    ADDR x1,d1_vis_ffn_base_cosine
    bl d1_read_cosine
    ADDR x0,d1_vis_ffn_mask_env
    bl getenv
    cbz x0,.Lvis_ffn_mask_done
    ldrb w1,[x0]
    cbz w1,.Lvis_ffn_mask_invalid
    ldrb w0,[x0,#1]
    cbnz w0,.Lvis_ffn_mask_invalid
    sub w1,w1,#'0'
    .ifdef D1_VISION_FFN_MASK_ADDITIONS
    cmp w1,#6
    .else
    cmp w1,#4
    .endif
    b.hi .Lvis_ffn_mask_invalid
    ADDR x0,d1_vis_ffn_masks
    ldr w1,[x0,x1,lsl #2]
    ADDR x0,d1_vis_ffn_mask
    str w1,[x0]
    ADDR x0,d1_vis_ffn_mask_cosine_env
    ADDR x1,d1_vis_ffn_lower_cosine
    bl d1_read_cosine
.Lvis_ffn_mask_done:
    .endif
    .ifdef D1_VISION_GROUP_FROM7
    ADDR x0,d1_vis_group_from7_env
    bl getenv
    cbz x0,.Lvis_group_from7_done
    ldrb w0,[x0]
    cmp w0,#'1'
    b.ne .Lvis_group_from7_done
    ADDR x0,d1_vis_group_enabled
    ldr w1,[x0]
    cbz w1,.Lvis_group_bad_flags
    ADDR x0,d1_vis_group_from7
    str w1,[x0]
.Lvis_group_from7_done:
    .endif
    .ifdef D1_VISION_LATE_FFN
    ADDR x0,d1_vis_late_ffn_env
    bl getenv
    cbz x0,.Lvis_late_ffn_config_done
    ldrb w0,[x0]
    cmp w0,#'1'
    b.ne .Lvis_late_ffn_config_done
    ADDR x0,d1_vis_group_enabled
    ldr w1,[x0]
    cbz w1,.Lvis_group_bad_flags
    ADDR x0,d1_vis_late_ffn_enabled
    str w1,[x0]
    .ifdef D1_VISION_LATE_FFN_TUNE
    ADDR x0,d1_vis_late_mask_env
    bl getenv
    cbz x0,.Lvis_late_ffn_config_done
    sub sp,sp,#16
    mov x1,sp
    mov w2,#0
    bl strtoul
    ldr x1,[sp]
    add sp,sp,#16
    ldrb w1,[x1]
    cbnz w1,.Lvis_group_bad_flags
    mov x2,#0x3f00
    bic x2,x0,x2
    cbnz x2,.Lvis_group_bad_flags
    ADDR x1,d1_vis_late_mask
    str w0,[x1]
    ADDR x0,d1_vis_late_cosine_env
    ADDR x1,d1_vis_late_override_cosine
    bl d1_read_cosine
    .endif
.Lvis_late_ffn_config_done:
    .endif
    .ifdef D1_VISION_EARLY_KV
    ADDR x0,d1_vis_early_kv_env
    bl getenv
    cbz x0,.Lvis_early_kv_config_done
    ldrb w1,[x0]
    sub w1,w1,#'0'
    cmp w1,#2
    b.hi .Lvis_group_bad_flags
    ADDR x0,d1_vis_early_kv_mode
    str w1,[x0]
.Lvis_early_kv_config_done:
    .endif
    .ifdef D1_VISION_POOLED12
    ADDR x0,d1_vis_pooled12_env
    bl getenv
    cbz x0,.Lvis_pooled12_done
    ldrb w0,[x0]
    cmp w0,#'1'
    b.ne .Lvis_pooled12_done
    ADDR x0,d1_vis_value12_enabled
    ldr w0,[x0]
    cbz w0,.Lvis_group_bad_flags
    ADDR x0,d1_vis_score12_enabled
    ldr w0,[x0]
    cbz w0,.Lvis_group_bad_flags
    ADDR x0,d1_vis_score_packed_enabled
    ldr w0,[x0]
    cbnz w0,.Lvis_group_bad_flags
    ADDR x0,d1_vis_pooled12_enabled
    mov w1,#1
    str w1,[x0]
.Lvis_pooled12_done:
    .endif
    LEAVE
.ifdef D1_VISION_FFN_MASKED
.Lvis_ffn_mask_invalid:
    ADDR x0,d1_vis_ffn_mask_error
    b fatal
.endif
.Lvis_group_bad_flags:
    ADDR x0,d1_vis_group_error
    b fatal

.type d1_read_cosine,%function
d1_read_cosine: /* environment name, destination initialized to default */
    ENTER
    mov x20,x1
    bl getenv
    cbz x0,.Lcosine_done
    mov x19,x0
    sub sp,sp,#16
    mov x1,sp
    bl strtof
    ldr x1,[sp]
    add sp,sp,#16
    cmp x19,x1
    b.eq .Lcosine_invalid
    ldrb w1,[x1]
    cbnz w1,.Lcosine_invalid
    fcmp s0,#0.0
    b.mi .Lcosine_invalid
    b.vs .Lcosine_invalid
    fmov s1,#1.0
    fcmp s0,s1
    b.gt .Lcosine_invalid
    str s0,[x20]
.Lcosine_done:
    LEAVE
.Lcosine_invalid:
    ADDR x0,d1_cosine_error
    b fatal
.size d1_read_cosine,.-d1_read_cosine

.type d1_vision_map,%function
d1_vision_map:
    ENTER
    mov w1,#0
    bl open
    .ifdef D1_PARALLEL_ACTIVATIONS
    cmp w0,#0
    b.lt file_error
    .else
    tbnz w0,#31,file_error
    .endif
    mov w19,w0
    mov x1,#0
    mov w2,#2 /* SEEK_END */
    bl lseek
    ldr x20,=D1_VISION_BYTES
    cmp x0,x20
    b.ne file_error
    mov x0,#0
    mov x1,x20
    mov w2,#1 /* PROT_READ */
    mov w3,#2 /* MAP_PRIVATE */
    mov w4,w19
    mov x5,#0
    bl mmap
    cmn x0,#1
    b.eq file_error
    mov x21,x0
    mov w0,w19
    bl close
    .ifdef D1_DEFER_VISION_CRC
    ADDR x0,d1_defer_vis_enabled
    ldr w0,[x0]
    cbz w0,.Lvis_full_crc
    mov x0,x21
    bl d1_verify_eager_vision_segments
    b .Lvis_crc_verified
.Lvis_full_crc:
    .endif
    .ifdef D1_PARALLEL_CHECKPOINT_CRC
    ADDR x0,d1_parallel_crc_enabled
    ldr w0,[x0]
    cbz w0,.Lvis_crc_serial
    mov x0,x21
    mov x1,x20
    ldr w2,=0x38ed840b
    ldr w3,=0xe4eeb6d9
    bl d1_verify_checkpoint_shards
    b .Lvis_crc_verified
.Lvis_crc_serial:
    .endif
    mov x0,x21
    mov w2,#-1
    .if D1_VISION_BYTES % 8
    .error "expected the pinned vision checkpoint size to be a multiple of 8"
    .endif
.Lvis_crc:
    ldr x3,[x0],#8
    crc32x w2,w2,x3
    subs x20,x20,#8
    b.ne .Lvis_crc
    mvn w2,w2
    ldr w3,=D1_VISION_CRC
    cmp w2,w3
    b.ne checksum_error
    .ifdef D1_PARALLEL_CHECKPOINT_CRC
.Lvis_crc_verified:
    .endif
    ADDR x0,d1_vis_base
    str x21,[x0]
    bl d1_q8_i8_map
    bl d1_q8_ilp_map
    bl d1_q8_res_map
    LEAVE
.size d1_vision_map,.-d1_vision_map

d1_vision_ptr: /* descriptor -> mapped projector data */
    ADDR x4,d1_vis_base
    ldr x4,[x4]
    ldr x5,[x0]
    add x4,x4,x5
    ret

d1_vision_mm: /* descriptor, input, output, positions */
    ADDR x4,d1_vis_base
    ldr x4,[x4]
    ldr x5,[x0]
    add x4,x4,x5
    b d1_matrix_batch_ptr

d1_vision_mm_reuse:
    ADDR x4,d1_vis_base
    ldr x4,[x4]
    ldr x5,[x0]
    add x4,x4,x5
    b d1_matrix_batch_reuse_ptr

.macro VIS_ROWS reg
    ADDR x9,d1_vis_rows_this_layer
    ldr \reg,[x9]
.endm

.macro VIS_MM slot,input,output,positions,reuse=0
    add x0,x19,#(\slot)*32
    ADDR x1,\input
    ADDR x2,\output
    .ifb \positions
    VIS_ROWS w3
    .else
    mov w3,\positions
    .endif
    .if \reuse
    bl d1_vision_mm_reuse
    .else
    bl d1_vision_mm
    .endif
.endm

.macro VIS_RMS slot,input,output,width,row_shift
    add x0,x19,#(\slot)*32
    bl d1_vision_ptr
    mov x2,x4
    ADDR x0,\output
    ADDR x1,\input
    mov w3,#\width
    VIS_ROWS w4
    .ifnb \row_shift
    lsl w4,w4,#\row_shift
    .endif
    bl d1_vision_rms_rows
.endm

.macro VIS_RESIDUAL
    ADDR x0,d1_vis_hidden
    ADDR x1,d1_vis_op
    VIS_ROWS w2
    lsl w2,w2,#10
    bl lfm_add
.endm

.macro VIS_QUAD function,buffer,width
    VIS_ROWS w0
    cmp w0,#300
    b.ne .Lquad_done\@
    ADDR x0,\buffer
    mov w1,#\width
    mov x2,xzr
    bl d1_vision_quad_\function
.Lquad_done\@:
.endm

.macro VIS_MARK
    .ifdef D1_VISION_PROFILE_WORK
    ADDR x0,d1_vis_prof_start
    bl clock_stamp
    .endif
.endm

.macro VIS_ACCUMULATE group
    .ifdef D1_VISION_PROFILE_WORK
    mov w0,#\group
    bl d1_vision_charge
    .endif
.endm

.type d1_vision_encode,%function
d1_vision_encode: /* x0 fresh RGB8 bytes -> d1_vis_output */
    ENTER
    mov x19,x0
    VIS_MARK
    mov x0,x19
    ADDR x1,d1_vis_patches
    ADDR x2,d1_vis_half
    bl d1_stem_prepare
    ADDR x0,d1_vis_stem_desc
    ADDR x1,d1_vis_patches
    ADDR x2,d1_vis_hidden
    mov w3,#1200
    ADDR x4,d1_vis_stem_asset
    bl d1_matrix_batch_ptr
    ADDR x0,d1_vis_hidden
    mov x1,x0
    ADDR x2,d1_vis_stem_asset
    add x2,x2,#768,lsl #12 /* 768 * 1024 * 4 bytes */
    mov w3,#1024
    mov w4,#1200
    bl d1_vision_rms_rows
    VIS_ACCUMULATE 0

    ADDR x19,d1_vis_block_desc
    ADDR x20,d1_vis_dw_asset
    mov w21,#14
    ADDR x0,d1_vis_rows_this_layer
    mov w1,#1200
    str w1,[x0]
.Lvis_layer:
    /* Modes 2 and 3 select opposite sides of the block-8 boundary. */
    ADDR x0,d1_fmla_stage
    mov w1,#2
    cmp w21,#6
    cinc w1,w1,ls
    str w1,[x0]
    .ifdef D1_VISION_GROUP_FROM7
    cmp w21,#7
    b.ne .Lvis_keep_early_group
    ADDR x0,d1_vis_group_from7
    ldr w0,[x0]
    cbz w0,.Lvis_keep_early_group
    ADDR x0,d1_vis_early_group_counts
    mov w1,#300
    str w1,[x0,#8]
    b .Lvis_force_late_group
.Lvis_keep_early_group:
    .endif
    ADDR x0,d1_vis_early_conv_enabled
    ldr w0,[x0]
    cbz w0,.Lvis_late_group_check
    cmp w21,#13 /* optional grouping in vision block 1 */
    b.eq .Lvis_stem_group_check
    cmp w21,#11 /* early convolution blocks 3, 5 and 7 */
    b.hi .Lvis_late_group_check
    cmp w21,#7
    b.lo .Lvis_late_group_check
    tbz w21,#0,.Lvis_late_group_check
    mov w1,#11
    sub w1,w1,w21
    lsr w1,w1,#1
    ADDR x28,d1_vis_early_group_counts
    add x28,x28,x1,lsl #2
    ADDR x3,d1_vis_early_cosine
    ldr s0,[x3]
    b .Lvis_plan_group
.Lvis_stem_group_check:
    ADDR x3,d1_vis_stem_group_cosine
    ldr s0,[x3]
    fcmp s0,#0.0
    b.mi .Lvis_late_group_check
    ADDR x28,d1_vis_stem_group_count
.Lvis_plan_group:
    ADDR x0,d1_vis_hidden
    ADDR x1,d1_vis_pool_records /* only the later attention blocks use these records */
    ADDR x2,d1_vis_early_map
    bl d1_vision_conv_group_plan
    ADDR x1,d1_vis_rows_this_layer
    str w0,[x1]
    str w0,[x28]
    ADDR x1,d1_vis_early_conv_active
    mov w0,#1
    str w0,[x1]
    ADDR x0,d1_vis_half
    mov w1,#0
    mov x2,#(1200*1024*4)
    bl memset
    ADDR x0,d1_vis_hidden
    ADDR x1,d1_vis_pool_records
    ADDR x2,d1_vis_half
    bl d1_vision_conv_group_reduce
    b .Lvis_operator
.Lvis_late_group_check:
    cmp w21,#6 /* grouping starts at vision block 8 */
    b.ne .Lvis_operator
    .ifdef D1_VISION_GROUP_FROM7
    ADDR x0,d1_vis_group_from7
    ldr w0,[x0]
    cbnz w0,.Lvis_operator
.Lvis_force_late_group:
    .endif
    ADDR x0,d1_vis_group_enabled
    ldr w0,[x0]
    cbz w0,.Lvis_operator
    ADDR x0,d1_vis_hidden
    mov w1,#1024
    ADDR x2,d1_vis_half /* patchifier workspace is dead */
    bl d1_vision_quad_reduce
    ADDR x0,d1_vis_rows_this_layer
    mov w1,#300
    str w1,[x0]
.Lvis_operator:
    VIS_RMS 0,d1_vis_hidden,d1_vis_norm,1024
    ldr x0,[x19,#5*32]
    cbz x0,.Lvis_attn
    VIS_MARK
    VIS_MM 5,d1_vis_norm,d1_vis_wide0
    ADDR x0,d1_vis_early_conv_active
    ldr w0,[x0]
    cbnz w0,.Lvis_conv_mapped
    .ifdef D1_VISION_COMPACT_DW
    VIS_ROWS w0
    cmp w0,#300
    b.ne .Lvis_conv_dense
    ADDR x0,d1_vis_dw_quad_enabled
    ldr w0,[x0]
    cbz w0,.Lvis_conv_dense
    ADDR x0,d1_vis_wide0
    ADDR x1,d1_vis_halo
    bl d1_vision_dw_stage_quad
    ADDR x0,d1_vis_wide0
    ADDR x1,d1_vis_halo
    mov x2,x20
    ADDR x3,d1_vis_work
    bl d1_vision_dw_apply_quad
    ADDR x0,d1_vis_dw_quad_layers
    ldr w1,[x0]
    add w1,w1,#1
    str w1,[x0]
    b .Lvis_conv_project
.Lvis_conv_dense:
    .endif
    VIS_QUAD expand,d1_vis_wide0,3072
    ADDR x0,d1_vis_wide0
    ADDR x1,d1_vis_halo
    bl d1_vision_dw_stage
    ADDR x0,d1_vis_wide0
    ADDR x1,d1_vis_halo
    mov x2,x20
    ADDR x3,d1_vis_work
    bl d1_vision_dw_apply
    VIS_QUAD reduce,d1_vis_work,1024
    b .Lvis_conv_project
.Lvis_conv_mapped:
    ADDR x0,d1_vis_wide0
    ADDR x1,d1_vis_halo
    ADDR x2,d1_vis_early_map
    bl d1_vision_dw_stage_mapped
    ADDR x0,d1_vis_wide0
    ADDR x1,d1_vis_halo
    mov x2,x20
    ADDR x3,d1_vis_work
    ADDR x4,d1_vis_early_map
    bl d1_vision_dw_apply_mapped
    ADDR x0,d1_vis_work
    ADDR x1,d1_vis_pool_records
    mov x2,xzr
    bl d1_vision_conv_group_reduce
.Lvis_conv_project:
    add x20,x20,#9,lsl #12
    VIS_MM 6,d1_vis_work,d1_vis_op
    VIS_ACCUMULATE 1
    b .Lvis_ffn
.Lvis_attn:
    VIS_MARK
    VIS_MM 7,d1_vis_norm,d1_vis_work
    VIS_MM 8,d1_vis_norm,d1_vis_keys,reuse=1
    VIS_MM 9,d1_vis_norm,d1_vis_values,reuse=1
    VIS_RMS 11,d1_vis_work,d1_vis_work,64,4
    VIS_RMS 12,d1_vis_keys,d1_vis_keys,64,3
    VIS_QUAD expand,d1_vis_work,1024
    VIS_QUAD expand,d1_vis_keys,512
    VIS_QUAD expand,d1_vis_values,512
    ADDR x0,d1_vis_work
    ADDR x1,d1_vis_rope_asset
    mov w2,#16
    bl d1_vision_rope_rows
    ADDR x0,d1_vis_keys
    ADDR x1,d1_vis_rope_asset
    mov w2,#8
    bl d1_vision_rope_rows
    VIS_QUAD reduce,d1_vis_work,1024
    .ifdef D1_VISION_ROUTE_FIXTURE_WORK
    cmp w21,#6
    b.ls .Lvis_route_inputs_done
    mov w0,#1
    ADDR x1,d1_vis_work
    mov x2,#1200*1024*4
    bl write_all
    mov w0,#1
    ADDR x1,d1_vis_keys
    ldr x2,=1200*512*4
    bl write_all
    mov w0,#1
    ADDR x1,d1_vis_values
    ldr x2,=1200*512*4
    bl write_all
.Lvis_route_inputs_done:
    .endif
    VIS_ACCUMULATE 2
    VIS_MARK
    ADDR x0,d1_vis_attn4_enabled
    ldr w0,[x0]
    cbz w0,.Lvis_head_job
    mov w26,#1200
    ADDR x2,d1_vis_pool_this_layer
    str wzr,[x2]
    ADDR x0,d1_vis_adaptive_keys_enabled
    ldr w0,[x0]
    cbz w0,.Lvis_pack_keys
    fmov s0,#0.5
    cmp w0,#2
    b.ne .Lvis_pool_keys
    cmp w21,#6 /* late attention layers 8, 10, 12 of 14 */
    .ifdef D1_VISION_EARLY_KV
    b.ls .Lvis_standard_late_keys
    ADDR x0,d1_vis_early_kv_mode
    ldr w0,[x0]
    cbz w0,.Lvis_count_keys
    cmp w21,#8 /* block 6 */
    b.eq .Lvis_pool_last_early
    cmp w21,#10 /* block 4 */
    ccmp w0,#2,#0,eq
    b.ne .Lvis_count_keys
.Lvis_pool_last_early:
    ADDR x1,d1_vis_early_kv_cutoffs
    ldr s0,[x1,x0,lsl #2]
    b .Lvis_pool_keys
.Lvis_standard_late_keys:
    .else
    b.hi .Lvis_count_keys
    .endif
    ADDR x0,d1_vis_late_cutoff
    ldr s0,[x0]
.Lvis_pool_keys:
    mov w0,#1
    str w0,[x2]
    ADDR x0,d1_vis_keys
    ADDR x1,d1_vis_values
    ADDR x2,d1_vis_pool_records
    bl d1_vision_pool_plan_cutoff
    mov w26,w0
    ADDR x0,d1_vis_keys
    ADDR x1,d1_vis_pool_records
    ADDR x2,d1_vis_pool_bias
    bl d1_vision_pool_apply
    ADDR x0,d1_vis_values
    ADDR x1,d1_vis_pool_records
    mov x2,xzr
    bl d1_vision_pool_apply
.Lvis_count_keys:
    .ifdef D1_VISION_EARLY_KV
    cmp w21,#6
    b.ls .Lvis_count_all_keys
    mov w0,#12
    sub w0,w0,w21
    lsr w0,w0,#1
    ADDR x1,d1_vis_early_kv_counts
    str w26,[x1,x0,lsl #2]
.Lvis_count_all_keys:
    .endif
    ADDR x0,d1_vis_attended_keys
    ldr w1,[x0]
    add w1,w1,w26
    str w1,[x0]
.Lvis_pack_keys:
    .ifdef D1_VISION_POOLED12
    ADDR x0,d1_vis_pool_this_layer
    ldr w0,[x0]
    cbz w0,.Lvis_pack_unpadded_keys
    ADDR x0,d1_vis_pooled12_enabled
    ldr w0,[x0]
    cbz w0,.Lvis_pack_unpadded_keys
    mov w0,w26
    bl d1_vis_pad_pooled_keys
    mov w26,w0
.Lvis_pack_unpadded_keys:
    .endif
    ADDR x0,d1_vis_keys_this_layer
    str w26,[x0]
    /* The normalized projection input is dead until the FFN. Its
     * 1200 × 1024 buffer fits both packed K and V arrays exactly. */
    ADDR x0,d1_vis_keys
    ADDR x1,d1_vis_norm
    mov w2,w26
    bl d1_vision_pack_heads
    ADDR x0,d1_vis_values
    ADDR x1,d1_vis_norm
    add x1,x1,#600,lsl #12
    mov w2,w26
    bl d1_vision_pack_heads
.Lvis_head_job:
    .ifdef D1_VISION_QUERY_GROUP
    mov w0,w21
    bl d1_vis_compact_query_layer
    .endif
    VIS_ROWS w1
    cmp w1,#300
    b.ne .Lvis_dispatch_head
    ADDR x0,d1_vis_group_queries
    ldr w2,[x0]
    add w2,w2,w1
    str w2,[x0]
.Lvis_dispatch_head:
    ADDR x0,d1_vis_attn_head
    str wzr,[x0]
    ADDR x0,d1_vis_route_enabled
    ldr w0,[x0]
    cbz w0,.Lvis_route_budget
    cmp w0,#1
    ccmp w21,#10,#0,eq /* route block 4 only in the two-block mode */
    mov w0,#120
    b.eq .Lvis_route_budget
    cmp w21,#8 /* vision block 6 */
    mov w0,#75
    csel w0,w0,wzr,eq
.Lvis_route_budget:
    ADDR x1,d1_vis_route_budget
    str w0,[x1]
    bl d1_vision_dispatch_attention
    .ifdef D1_VISION_ROUTE_FIXTURE_WORK
    cmp w21,#6
    b.ls .Lvis_route_context_done
    mov w0,#1
    ADDR x1,d1_vis_work
    mov x2,#1200*1024*4
    bl write_all
.Lvis_route_context_done:
    .endif
    VIS_ACCUMULATE 3
    VIS_MARK
    VIS_MM 10,d1_vis_work,d1_vis_op
    .ifdef D1_VISION_QUERY_GROUP
    bl d1_vis_expand_query_layer
    .endif
    VIS_ACCUMULATE 2
.Lvis_ffn:
    VIS_RESIDUAL
    VIS_RMS 1,d1_vis_hidden,d1_vis_norm,1024
    VIS_MARK
    VIS_ROWS w27
    mov x28,xzr
    ADDR x0,d1_vis_early_conv_active
    ldr w0,[x0]
    cbnz w0,.Lvis_ffn_expand
    cmp w27,#300
    .ifdef D1_VISION_LATE_FFN
    b.ne .Lvis_ffn_full_grid
    cmp w21,#6 /* Share FFN inputs in original vision blocks 8 through 13. */
    b.hi .Lvis_ffn_count
    ADDR x0,d1_vis_late_ffn_enabled
    ldr w0,[x0]
    cbz w0,.Lvis_ffn_count
    ADDR x28,d1_vis_ffn_row_map
    ADDR x0,d1_vis_norm
    mov x1,x28
    mov w2,#0
    mov w3,#300
    .ifdef D1_VISION_LATE_FFN_TUNE
    ADDR x4,d1_vis_late_base_cosine
    ldr s0,[x4]
    ADDR x4,d1_vis_late_mask
    ldr w4,[x4]
    mov w5,#14
    sub w5,w5,w21
    mov w6,#1
    lsl w6,w6,w5
    tst w4,w6
    b.eq .Lvis_late_cutoff_ready
    ADDR x4,d1_vis_late_mask_used
    ldr w5,[x4]
    orr w5,w5,w6
    str w5,[x4]
    ADDR x4,d1_vis_late_override_cosine
    ldr s0,[x4]
.Lvis_late_cutoff_ready:
    .endif
    bl d1_compact_vision_late_ffn
    mov w27,w0
    ADDR x0,d1_vis_late_ffn_counts
    ldp w1,w2,[x0]
    add w1,w1,#1
    add w2,w2,#300
    sub w2,w2,w27
    stp w1,w2,[x0]
    b .Lvis_ffn_count
.Lvis_ffn_full_grid:
    .else
    b.eq .Lvis_ffn_count
    .endif
    ADDR x0,d1_vis_ffn_reuse_enabled
    ldr w0,[x0]
    cbz w0,.Lvis_ffn_gate
    b .Lvis_ffn_compact
.Lvis_ffn_expand:
    ADDR x0,d1_vis_norm
    ADDR x1,d1_vis_early_map
    bl d1_vision_conv_group_expand
.Lvis_ffn_compact:
    .ifdef D1_VISION_FFN_FIXTURE
    cmp w21,#6
    b.ls .Lvis_ffn_fixture_input_done
    mov w0,#3
    ADDR x1,d1_vis_norm
    mov x2,#1200*1024*4
    bl write_all
.Lvis_ffn_fixture_input_done:
    .endif
    ADDR x28,d1_vis_ffn_row_map
    ADDR x0,d1_vis_norm
    mov x1,x28
    mov w2,#0
    mov w3,#1200
    .ifdef D1_VISION_FFN_MASKED
    mov w4,#14
    sub w4,w4,w21
    ADDR x5,d1_vis_ffn_mask
    ldr w5,[x5]
    lsr w5,w5,w4
    tst w5,#1
    ADDR x5,d1_vis_ffn_base_cosine
    ldr s0,[x5]
    ADDR x5,d1_vis_ffn_lower_cosine
    ldr s1,[x5]
    fcsel s0,s0,s1,eq
    .endif
    bl d1_compact_vision_ffn
    mov w27,w0
    .ifdef D1_VISION_FFN_FIXTURE
    cmp w21,#6
    b.ls .Lvis_ffn_fixture_map_done
    mov w0,#3
    mov x1,x28
    mov x2,#1200*4
    bl write_all
.Lvis_ffn_fixture_map_done:
    .endif
.Lvis_ffn_count:
    ADDR x0,d1_vis_ffn_reused
    ldr w1,[x0]
    mov w2,#1200
    sub w2,w2,w27
    add w1,w1,w2
    str w1,[x0]
.Lvis_ffn_gate:
    VIS_MM 2,d1_vis_norm,d1_vis_wide0,w27
    VIS_MM 3,d1_vis_norm,d1_vis_wide1,w27,reuse=1
    ADDR x22,d1_vis_wide0
    ADDR x23,d1_vis_wide1
    mov w24,w27
    mov x25,#2560*4
    .ifdef D1_PARALLEL_ACTIVATIONS
    ADDR x0,d1_parallel_activations_enabled
    ldr w0,[x0]
    cbz w0,.Lvis_swiglu
    mov x0,x22
    mov x1,x23
    mov w2,#2560
    mul w2,w2,w27
    bl d1_dispatch_activations
    b .Lvis_swiglu_done
    .endif
.Lvis_swiglu:
    mov x0,x22
    mov x1,x23
    mov w2,#2560
    bl lfm_swiglu_native
    add x22,x22,x25
    add x23,x23,x25
    subs w24,w24,#1
    b.ne .Lvis_swiglu
    .ifdef D1_PARALLEL_ACTIVATIONS
.Lvis_swiglu_done:
    .endif
    VIS_MM 4,d1_vis_wide0,d1_vis_op,w27
    VIS_ACCUMULATE 4
    ADDR x0,d1_vis_early_conv_active
    ldr w0,[x0]
    cbnz w0,.Lvis_finish_early_conv
    cbnz x28,.Lvis_ffn_scatter
    VIS_RESIDUAL
    b .Lvis_next_layer
.Lvis_ffn_scatter:
    ADDR x22,d1_vis_hidden
    ADDR x23,d1_vis_op
    .ifdef D1_VISION_LATE_FFN
    VIS_ROWS w24
    .else
    mov w24,#1200
    .endif
.Lvis_ffn_add:
    ldr w1,[x28],#4
    add x1,x23,x1,lsl #12
    mov x0,x22
    mov w2,#1024
    bl lfm_add
    add x22,x22,#4096
    subs w24,w24,#1
    b.ne .Lvis_ffn_add
    b .Lvis_next_layer
.Lvis_finish_early_conv:
    ADDR x0,d1_vis_hidden
    ADDR x1,d1_vis_early_map
    ADDR x2,d1_vis_op
    mov x3,x28
    ADDR x4,d1_vis_half
    bl d1_vision_conv_group_finish
    ADDR x0,d1_vis_early_conv_active
    str wzr,[x0]
    ADDR x0,d1_vis_rows_this_layer
    mov w1,#1200
    str w1,[x0]
.Lvis_next_layer:
    VIS_ROWS w0
    cmp w0,#300
    b.ne .Lvis_advance
    ADDR x0,d1_vis_group_blocks
    ldr w1,[x0]
    add w1,w1,#1
    str w1,[x0]
.Lvis_advance:
    add x19,x19,#13*32
    subs w21,w21,#1
    b.ne .Lvis_layer

    VIS_MARK
    VIS_ROWS w0
    cmp w0,#300
    b.ne .Lvis_merge
    ADDR x0,d1_vis_hidden
    mov w1,#1024
    ADDR x2,d1_vis_half
    bl d1_vision_quad_expand
.Lvis_merge:
    ADDR x0,d1_vis_gamma_desc
    bl d1_vision_ptr
    mov x1,x4
    ADDR x0,d1_vis_beta_desc
    bl d1_vision_ptr
    mov x2,x4
    ADDR x0,d1_vis_hidden
    ADDR x3,d1_vis_wide0
    bl d1_vision_norm_merge
    ADDR x0,d1_vis_up_desc
    ADDR x1,d1_vis_wide0
    ADDR x2,d1_vis_wide1
    mov w3,#300
    bl d1_vision_mm
    ADDR x0,d1_vis_up_bias_desc
    ADDR x1,d1_vis_wide1
    mov w2,#4096
    bl d1_vis_bias_rows
    .ifdef D1_PARALLEL_ACTIVATIONS
    ADDR x0,d1_parallel_activations_enabled
    ldr w0,[x0]
    cbz w0,.Lvis_serial_gelu
    ADDR x0,d1_vis_wide1
    mov x1,#0
    ldr w2,=300*4096
    bl d1_dispatch_activations
    b .Lvis_gelu_done
.Lvis_serial_gelu:
    .endif
    ADDR x0,d1_vis_wide1
    ldr w1,=300*4096
    bl d1_vision_gelu_erf
    .ifdef D1_PARALLEL_ACTIVATIONS
.Lvis_gelu_done:
    .endif
    ADDR x0,d1_vis_down_desc
    ADDR x1,d1_vis_wide1
    ADDR x2,d1_vis_output
    mov w3,#300
    bl d1_vision_mm
    ADDR x0,d1_vis_down_bias_desc
    ADDR x1,d1_vis_output
    mov w2,#2048
    bl d1_vis_bias_rows
    VIS_ACCUMULATE 5
    LEAVE
.size d1_vision_encode,.-d1_vision_encode

.ifdef D1_VISION_POOLED12
.type d1_vis_pad_pooled_keys,%function
d1_vis_pad_pooled_keys: /* w0 real keys, returns count rounded up to eight */
    ENTER
    mov w19,w0
    add w20,w19,#7
    and w20,w20,#~7
    subs w21,w20,w19
    b.eq .Lvis_keys_padded
    ADDR x0,d1_vis_keys
    add x0,x0,x19,lsl #11
    mov w1,#0
    lsl x2,x21,#11
    bl memset
    ADDR x0,d1_vis_values
    add x0,x0,x19,lsl #11
    mov w1,#0
    lsl x2,x21,#11
    bl memset
    ADDR x0,d1_vis_pool_bias
    add x0,x0,x19,lsl #2
    mov w1,#0xff800000
.Lvis_pad_key_bias:
    str w1,[x0],#4
    subs w21,w21,#1
    b.ne .Lvis_pad_key_bias
.Lvis_keys_padded:
    mov w0,w20
    LEAVE
.size d1_vis_pad_pooled_keys,.-d1_vis_pad_pooled_keys
.endif

.type d1_vis_bias_rows,%function
d1_vis_bias_rows: /* bias descriptor, output, width; 300 merged positions */
    ENTER
    mov x19,x1
    mov w20,w2
    bl d1_vision_ptr
    mov x21,x4
    mov w22,#300
.Lvis_bias:
    mov x0,x19
    mov x1,x21
    mov w2,w20
    bl lfm_add
    add x19,x19,x20,lsl #2
    subs w22,w22,#1
    b.ne .Lvis_bias
    LEAVE
.size d1_vis_bias_rows,.-d1_vis_bias_rows

.type d1_vision_attn_job,%function
d1_vision_attn_job: /* Each worker owns one head at a time. */
    ADDR x0,d1_vis_attn4_enabled
    ldr w0,[x0]
    cbnz w0,d1_vision_attn_job4
    ENTER
    sub sp,sp,#1,lsl #12
    sub sp,sp,#704 /* 1200 FP32 softmax entries */
    ADDR x19,d1_vis_attn_head
.Lclaim_head:
    ldaxr w20,[x19]
    add w0,w20,#1
    stlxr w1,w0,[x19]
    cbnz w1,.Lclaim_head
    cmp w20,#16
    b.hs .Lheads_done
    ADDR x21,d1_vis_work
    add x21,x21,x20,lsl #8
    lsr w0,w20,#1
    ADDR x22,d1_vis_keys
    ADDR x23,d1_vis_values
    add x22,x22,x0,lsl #8
    add x23,x23,x0,lsl #8
    mov w24,#1200
.Lquery_rows:
    mov x0,x21
    mov x1,x22
    mov x2,x23
    mov x3,x21 /* In-place; only this query is read by the head. */
    mov x4,sp
    mov w5,#1200
    mov x6,#2048
    bl d1_vision_attention_head
    add x21,x21,#1,lsl #12
    subs w24,w24,#1
    b.ne .Lquery_rows
    b .Lclaim_head
.Lheads_done:
    add sp,sp,#1,lsl #12
    add sp,sp,#704
    LEAVE
.size d1_vision_attn_job,.-d1_vision_attn_job

.type d1_vision_pack_heads,%function
d1_vision_pack_heads: /* [positions][8][64] -> [8][positions][64] */
    mov w5,#8
.Lpack_head:
    mov x3,x0
    mov w4,w2
.Lpack_kv_pos:
    .rept 4
    ld1 {v0.4s-v3.4s},[x3],#64
    st1 {v0.4s-v3.4s},[x1],#64
    .endr
    add x3,x3,#(2048-256)
    subs w4,w4,#1
    b.ne .Lpack_kv_pos
    add x0,x0,#256
    subs w5,w5,#1
    b.ne .Lpack_head
    ret
.size d1_vision_pack_heads,.-d1_vision_pack_heads

.type d1_vision_attn_job4,%function
d1_vision_attn_job4: /* One worker owns both query heads of a KV group. */
    ENTER
    sub sp,sp,#4,lsl #12
    sub sp,sp,#2816 /* four 1200-float score buffers */
    ADDR x19,d1_vis_attn_head
    ADDR x0,d1_vis_keys_this_layer
    ldr w27,[x0]
    ADDR x28,d1_vision_attention4
    ADDR x0,d1_vis_pool_this_layer
    ldr w0,[x0]
    cbz w0,.Lmaybe_packed_score
    ADDR x28,d1_vision_attention4_pooled
    b .Lclaim_kv
.Lmaybe_packed_score:
    ADDR x0,d1_vis_score_packed_enabled
    ldr w0,[x0]
    cbz w0,.Lclaim_kv
    ADDR x28,d1_vision_attention4_packed
.Lclaim_kv:
    ldaxr w20,[x19]
    add w0,w20,#1
    stlxr w1,w0,[x19]
    cbnz w1,.Lclaim_kv
    cmp w20,#8
    b.hs .Lkv_done
    ADDR x22,d1_vis_norm
    lsl x0,x27,#8
    madd x22,x20,x0,x22
    add x23,x22,#600,lsl #12
    ADDR x25,d1_vis_work
    add x25,x25,x20,lsl #9
    mov w26,#2
.Lgroup_head:
    mov x21,x25
    VIS_ROWS w24
    lsr w24,w24,#2
.Lquery_tiles:
    mov x0,x21
    mov x1,x22
    mov x2,x23
    mov x3,x21
    mov x4,sp
    mov w5,w27
    mov x6,#4096
    mov x7,#256
    ADDR x8,d1_vis_pool_bias
    blr x28
    add x21,x21,#4,lsl #12
    subs w24,w24,#1
    b.ne .Lquery_tiles
    add x25,x25,#256
    subs w26,w26,#1
    b.ne .Lgroup_head
    b .Lclaim_kv
.Lkv_done:
    add sp,sp,#4,lsl #12
    add sp,sp,#2816
    LEAVE
.size d1_vision_attn_job4,.-d1_vision_attn_job4

.include "research/d1-640x480/vision_value_job.inc"
.include "research/d1-640x480/vision_route_job.inc"

.ifdef D1_VISION_QUERY_GROUP
.type d1_vis_compact_query_layer,%function
d1_vis_compact_query_layer: /* reverse block number, dense early keys and score12 */
    cmp w0,#6
    b.ls .Lquery_layer_done
    ADDR x1,d1_vis_query_cutoff
    ldr s0,[x1]
    fcmp s0,#0.0
    b.le .Lquery_layer_done
    ADDR x1,d1_vis_score12_enabled
    ldr w1,[x1]
    cbz w1,.Lquery_layer_done
    .ifdef D1_VISION_EARLY_KV
    ADDR x1,d1_vis_early_kv_mode
    ldr w1,[x1]
    cbnz w1,.Lquery_plan
    .endif
    ADDR x1,d1_vis_keys_this_layer
    ldr w1,[x1]
    cmp w1,#1200
    b.ne .Lquery_layer_done
    ADDR x1,d1_vis_pool_this_layer
    ldr w1,[x1]
    cbnz w1,.Lquery_layer_done
    .ifdef D1_VISION_EARLY_KV
.Lquery_plan:
    /* The K/V plan has already materialized both packed arrays and the
     * log multiplicities. The pooled four-query worker accepts this padding. */
    .endif
    ENTER
    mov w19,#12
    sub w19,w19,w0
    lsr w19,w19,#1
    ADDR x0,d1_vis_work
    ADDR x1,d1_vis_pool_records
    ADDR x2,d1_vis_early_map
    bl d1_vision_query_group_plan
    ADDR x1,d1_vis_query_counts
    str w0,[x1,x19,lsl #2]
    cmp w0,#1200
    b.eq .Lquery_layer_leave
    mov w20,w0
    mov w1,#12
    add w2,w20,#11
    udiv w2,w2,w1
    mul w21,w2,w1
    ADDR x1,d1_vis_rows_this_layer
    str w21,[x1]
    ADDR x1,d1_vis_query_active
    mov w2,#1
    str w2,[x1]
    ADDR x0,d1_vis_work
    ADDR x1,d1_vis_pool_records
    mov x2,xzr
    bl d1_vision_conv_group_reduce
    subs w2,w21,w20
    b.eq .Lquery_layer_leave
    ADDR x0,d1_vis_work
    add x0,x0,x20,lsl #12
    lsl x2,x2,#12
    mov w1,#0
    bl memset
.Lquery_layer_leave:
    LEAVE
.Lquery_layer_done:
    ret
.size d1_vis_compact_query_layer,.-d1_vis_compact_query_layer

.type d1_vis_expand_query_layer,%function
d1_vis_expand_query_layer:
    ADDR x0,d1_vis_query_active
    ldr w1,[x0]
    cbz w1,.Lquery_expand_done
    str wzr,[x0]
    ADDR x0,d1_vis_rows_this_layer
    mov w1,#1200
    str w1,[x0]
    ADDR x0,d1_vis_op
    ADDR x1,d1_vis_early_map
    b d1_vision_conv_group_expand
.Lquery_expand_done:
    ret
.size d1_vis_expand_query_layer,.-d1_vis_expand_query_layer

.type d1_vis_report_query_groups,%function
d1_vis_report_query_groups:
    ADDR x0,d1_vis_query_counts
    ldr w2,[x0]
    cbz w2,.Lquery_report_done
    ldp w3,w4,[x0,#4]
    ADDR x1,d1_vis_query_group_stats
    mov w0,#2
    b dprintf
.Lquery_report_done:
    ret
.size d1_vis_report_query_groups,.-d1_vis_report_query_groups
.endif

.ifdef D1_VISION_PROFILE_WORK
.type d1_vision_charge,%function
d1_vision_charge:
    ENTER
    mov w19,w0
    ADDR x0,d1_vis_prof_end
    bl clock_stamp
    ADDR x0,d1_vis_prof_start
    ADDR x1,d1_vis_prof_end
    bl seconds_between
    ADDR x0,d1_vis_prof_totals
    ldr d1,[x0,x19,lsl #3]
    fadd d0,d0,d1
    str d0,[x0,x19,lsl #3]
    LEAVE
.size d1_vision_charge,.-d1_vision_charge

.type d1_vision_report,%function
d1_vision_report:
    ENTER
    ADDR x19,d1_vis_prof_totals
    ADDR x20,d1_vis_prof_names
    mov w21,#6
.Lvis_report:
    ldr d0,[x19],#8
    ldrsw x2,[x20],#4
    ADDR x3,d1_vis_prof_names
    add x2,x3,x2
    mov w0,#2
    ADDR x1,d1_vis_prof_format
    bl dprintf
    subs w21,w21,#1
    b.ne .Lvis_report
    LEAVE
.size d1_vision_report,.-d1_vision_report
.section .rodata
.balign 4
d1_vis_prof_names:
    .word .Lstem-d1_vis_prof_names,.Lconv-d1_vis_prof_names,.Lqkv-d1_vis_prof_names
    .word .Lattn-d1_vis_prof_names,.Lffn-d1_vis_prof_names,.Lmerge-d1_vis_prof_names
.Lstem: .asciz "stem"
.Lconv: .asciz "convolution"
.Lqkv: .asciz "attention_projections"
.Lattn: .asciz "attention_core"
.Lffn: .asciz "ffn"
.Lmerge: .asciz "merger"
d1_vis_prof_format: .asciz "d1_profile_%s_s=%.9f\n"
.bss
.balign 8
d1_vis_prof_start: .skip 16
d1_vis_prof_end: .skip 16
d1_vis_prof_totals: .skip 6*8
.endif

.section .rodata
d1_vis_flag: .asciz "--vision"
d1_vis_attn4_env: .asciz "D1_VISION_ATTN4"
d1_vis_pair_env: .asciz "D1_VISION_Q8_PAIR"
d1_vis_q8_ilp_env: .asciz "D1_VISION_Q8_ILP"
d1_vis_ffn_reuse_env: .asciz "D1_VISION_FFN_REUSE"
d1_vis_q8_panel_env: .asciz "D1_VISION_Q8_PANEL"
d1_vis_score_packed_env: .asciz "D1_VISION_SCORE_PACKED"
d1_vis_adaptive_keys_env: .asciz "D1_VISION_ADAPTIVE_KEYS"
d1_vis_group_env: .asciz "D1_VISION_GROUPED"
.ifdef D1_VISION_POOLED12
d1_vis_pooled12_env: .asciz "D1_VISION_POOLED12"
.endif
.ifdef D1_VISION_EARLY_KV
d1_vis_early_kv_env: .asciz "D1_VISION_EARLY_KV"
d1_vis_early_kv_stats: .asciz "d1_vision_early_kv=%u,%u,%u\n"
.balign 4
d1_vis_early_kv_cutoffs: .float 0,0.70,0.75
.endif
.ifdef D1_VISION_LATE_FFN
d1_vis_late_ffn_env: .asciz "D1_VISION_LATE_FFN"
d1_vis_late_ffn_stats: .asciz "d1_vision_late_ffn layers=%u reused=%u\n"
.ifdef D1_VISION_LATE_FFN_TUNE
d1_vis_late_mask_env: .asciz "D1_VISION_LATE_FFN_MASK"
d1_vis_late_cosine_env: .asciz "D1_VISION_LATE_FFN_COSINE"
d1_vis_late_mask_stats: .asciz "d1_vision_late_ffn_mask=%u\n"
.balign 4
d1_vis_late_base_cosine: .float 0.80
.endif
.endif
.ifdef D1_VISION_GROUP_FROM7
d1_vis_group_from7_env: .asciz "D1_VISION_GROUP_FROM7"
.endif
d1_vis_early_conv_env: .asciz "D1_VISION_EARLY_CONV"
d1_vis_early_cosine_env: .asciz "D1_VISION_EARLY_COSINE"
d1_vis_stem_group_env: .asciz "D1_VISION_STEM_GROUP_COSINE"
d1_vis_pack_reuse_env: .asciz "D1_VISION_PACK_REUSE"
.ifdef D1_VISION_FFN_MASKED
d1_vis_ffn_base_cosine_env: .asciz "D1_VISION_FFN_COSINE"
d1_vis_ffn_mask_env: .asciz "D1_VISION_FFN_MASKED_030"
d1_vis_ffn_mask_cosine_env: .asciz "D1_VISION_FFN_MASK_COSINE"
    .ifdef D1_VISION_FFN_MASK_ADDITIONS
d1_vis_ffn_mask_error: .asciz "Expected an FFN layer mask selector in 0..6.\n"
    .else
d1_vis_ffn_mask_error: .asciz "Expected an FFN layer mask selector in 0..4.\n"
    .endif
d1_vis_ffn_mask_stats: .asciz "d1_vision_ffn_mask=%u cutoff=%.4f\n"
.balign 4
d1_vis_ffn_masks: .word 0,0x0f,0xf0,0xab,0x54
    .ifdef D1_VISION_FFN_MASK_ADDITIONS
    .word 0xf1,0xf8
    .endif
.endif
.ifdef D1_VISION_QUERY_GROUP
d1_vis_query_group_env: .asciz "D1_VISION_QUERY_COSINE"
d1_vis_query_group_stats: .asciz "d1_vision_early_queries=%u,%u,%u\n"
.endif
.ifdef D1_VISION_COMPACT_DW
d1_vis_dw_quad_env: .asciz "D1_VISION_DW_QUAD"
d1_vis_dw_quad_stats: .asciz "d1_vision_dw_quad layers=%u\n"
.endif
d1_cosine_error: .asciz "Expected cosine cutoff between 0 and 1.\n"
d1_vis_group_error: .asciz "Grouped vision requires late key pooling and vision FFN reuse.\n"
d1_vis_timing: .asciz "d1_vision_s=%.9f\n"
.balign 4
d1_vis_late_cutoff: .float 0.65
.data
.balign 4
.ifdef D1_VISION_FFN_MASKED
d1_vis_ffn_base_cosine: .float 0.50
d1_vis_ffn_lower_cosine: .float 0.30
.endif
d1_vis_early_cosine: .float 0.50
d1_vis_stem_group_cosine: .float -1.0
.ifdef D1_VISION_LATE_FFN_TUNE
d1_vis_late_override_cosine: .float 0.80
.endif
.ifdef D1_VISION_QUERY_GROUP
d1_vis_query_cutoff: .float 0.0
.endif
.bss
.balign 64
d1_vis_base: .skip 8
d1_vis_attn_head: .skip 4
d1_vis_attn4_enabled: .skip 4
d1_vis_pair_enabled: .skip 4
d1_vis_q8_ilp_enabled: .skip 4
d1_vis_ffn_reuse_enabled: .skip 4
d1_vis_q8_panel_enabled: .skip 4
d1_vis_score_packed_enabled: .skip 4
d1_vis_adaptive_keys_enabled: .skip 4
d1_vis_pool_this_layer: .skip 4
d1_vis_keys_this_layer: .skip 4
d1_vis_attended_keys: .skip 4
d1_vis_group_enabled: .skip 4
.ifdef D1_VISION_POOLED12
d1_vis_pooled12_enabled: .skip 4
.endif
.ifdef D1_VISION_EARLY_KV
d1_vis_early_kv_mode: .skip 4
d1_vis_early_kv_counts: .skip 12
.endif
.ifdef D1_VISION_LATE_FFN
d1_vis_late_ffn_enabled: .skip 4
d1_vis_late_ffn_counts: .skip 8
.ifdef D1_VISION_LATE_FFN_TUNE
d1_vis_late_mask: .skip 4
d1_vis_late_mask_used: .skip 4
.endif
.endif
.ifdef D1_VISION_GROUP_FROM7
d1_vis_group_from7: .skip 4
.endif
d1_vis_rows_this_layer: .skip 4
d1_vis_group_blocks: .skip 4
d1_vis_group_queries: .skip 4
d1_vis_early_conv_enabled: .skip 4
d1_vis_early_conv_active: .skip 4
d1_vis_early_group_counts: .skip 3*4
d1_vis_stem_group_count: .skip 4
d1_vis_early_map: .skip 1200*4
d1_vis_ffn_reused: .skip 4
d1_vis_pack_reuse_enabled: .skip 4
d1_vis_pack_reuse_counts: .skip 8
.ifdef D1_VISION_FFN_MASKED
d1_vis_ffn_mask: .skip 4
.endif
.ifdef D1_VISION_QUERY_GROUP
d1_vis_query_active: .skip 4
d1_vis_query_counts: .skip 12
.endif
.ifdef D1_VISION_COMPACT_DW
d1_vis_dw_quad_enabled: .skip 4
d1_vis_dw_quad_layers: .skip 4
.endif
d1_vis_ffn_row_map: .skip 1200*4
.balign 64
d1_vis_pool_records: .skip 1200*16
d1_vis_pool_bias: .skip 1200*4
.balign 64
d1_vis_rgb: .skip 640*480*3
d1_vis_half: .skip 1200*768*2
d1_vis_patches: .skip 1200*768*4
d1_vis_hidden: .skip 1200*1024*4
.if (d1_vis_hidden-d1_vis_half) < 1200*1024*4
.error "vision patchifier workspace cannot hold saved residual detail"
.endif
d1_vis_norm: .skip 1200*1024*4
d1_vis_op: .skip 1200*1024*4
d1_vis_work: .skip 1200*1024*4
d1_vis_keys: .skip 1200*512*4
d1_vis_values: .skip 1200*512*4
d1_vis_wide0: .skip 1200*3072*4
d1_vis_wide1: .skip 1200*2560*4
d1_vis_halo: .skip 32*42*1024*4
d1_vis_output: .skip 300*2048*4
.text
.ltorg
