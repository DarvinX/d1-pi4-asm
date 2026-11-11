CC = gcc
.DEFAULT_GOAL := all

.PHONY: all
all: build/d1-vision-asm-sparse

empty :=
space := $(empty) $(empty)
comma := ,
DEFINES := D1_Q8_LONG_PAIRS=4 D1_Q8_CONV_6_CRC=0x3a03a480 D1_VISION_COMPACT_DW=1 D1_BACKBONE_TRANSITION_FFN=1 \
  D1_VISION_QUERY_GROUP=1 D1_Q8_EARLY_6_CRC=0xa23af264 D1_VISION_FFN_MASKED=1 D1_Q8_ALL_6_CRC=0x971ca1ed \
  D1_Q8_RES_REUSE=1 D1_VISION_GROUP_FROM7=1 D1_BACKBONE_REDUCED_ROWS=1 D1_Q8_HALF_LAYOUT=1 \
  D1_Q8_HALF_6_CRC=0x2d02c399 D1_SHORT_PACKED_ROWS=1 D1_PARALLEL_CHECKPOINT_CRC=1 \
  D1_VISION_FFN_MASK_ADDITIONS=1 D1_BACKBONE_DROP8=1 D1_VISION_LATE_FFN=1 D1_Q8_CERTIFIED_LAYOUT=1 \
  D1_Q8_CERTIFIED_6_CRC=0x7c30b13d D1_VISION_EARLY_KV=1 D1_Q8_MERGER_CRC=0x9e1c3c18 \
  D1_PARALLEL_ACTIVATIONS=1 D1_VISION_POOLED12=1 D1_PARALLEL_PACK=1 D1_BACKBONE_DELAY_TEXT=1 \
  D1_PARALLEL_Q4_PANEL_CRC=1 D1_PARALLEL_OTHER_PANEL_CRC=1 D1_MODEL_EMBED_ROW_CRC=1 D1_Q4_SEED_PRODUCT=1 \
  D1_Q4_GATE_UP_REUSE=1 D1_DEFER_MODEL_CRC=1 D1_DEFER_VISION_CRC=1 D1_RETAIN_VERIFIED_PAGES=1 \
  D1_VISION_LATE_FFN_TUNE=1 D1_ASYNC_BACKBONE_PANELS=1
ASM_FLAGS := -Wa,--defsym,$(subst $(space),$(comma)--defsym$(comma),$(strip $(DEFINES)))
SOURCES := research/d1-640x480/backbone.S research/d1-640x480/attention.S research/d1-640x480/backbone_sparse_plan.S \
  research/d1-640x480/backbone_ffn_share.S research/d1-640x480/backbone_frozen_ops.S \
  research/d1-640x480/q4_batch.S research/d1-640x480/q6_batch.S src/kernels.S src/q8-kernels.S \
  experiments/h19-q6-half-reuse.S experiments/h20-q4-pair-reuse.S experiments/h24-q8-sums.S \
  experiments/h32-dual-dot.S experiments/h34-coarse-q4-linked.S experiments/h42-q4-lanes.S \
  research/d1-640x480/vision_rgb.S research/d1-640x480/vision_stem.S research/d1-640x480/vision_depthwise.S \
  research/d1-640x480/vision_attention.S research/d1-640x480/vision_pool.S \
  research/d1-640x480/vision_groups.S research/d1-640x480/vision_projector.S \
  research/d1-640x480/vision_q8.S research/d1-640x480/vision_fmla.S research/d1-640x480/vision_value_tile.S \
  research/d1-640x480/vision_route.S research/d1-640x480/vision_score_fmla.S \
  research/d1-640x480/label_logits.S research/d1-640x480/q4_zero.S research/d1-640x480/q4_i8_tile.S \
  research/d1-640x480/q8_i8_tile.S research/d1-640x480/q8_ilp_prepack.S \
  research/d1-640x480/q8_i8_resident.S
INCLUDES := experiments/forward-h34.inc experiments/h33-probe.inc experiments/h33-tails.inc experiments/h34-batch.inc \
  experiments/h34-controller-v2.inc experiments/h36-generation.inc experiments/h36-latency.inc \
  experiments/runtime-h42.S experiments/workers-h34.inc research/d1-640x480/async_backbone_panels.inc \
  research/d1-640x480/backbone_attn_batch.inc research/d1-640x480/backbone_batch.inc \
  research/d1-640x480/backbone_conv_batch.inc research/d1-640x480/backbone_panel_map.inc \
  research/d1-640x480/backbone_q4_i8.inc research/d1-640x480/backbone_q6_i16.inc \
  research/d1-640x480/backbone_sparse.inc research/d1-640x480/decision.inc \
  research/d1-640x480/deferred_model_crc.inc research/d1-640x480/embedding_crc.inc \
  research/d1-640x480/ffn_compact.inc research/d1-640x480/vision_encode.inc \
  research/d1-640x480/vision_fmla_batch.inc research/d1-640x480/vision_q8_i8_job.inc \
  research/d1-640x480/vision_q8_ilp_job.inc research/d1-640x480/vision_q8_resident_job.inc \
  research/d1-640x480/vision_route_job.inc research/d1-640x480/vision_value_job.inc src/common.inc \
  src/native-math.inc src/tokenizer.inc
ASSETS := build/d1-backbone-assets/bytes.bin build/d1-backbone-assets/categories.bin \
  build/d1-backbone-assets/lookup.bin build/d1-backbone-assets/merges.bin \
  build/d1-backbone-assets/model.inc build/d1-backbone-assets/text.bin build/d1-backbone-assets/vocab.bin \
  build/d1-deferred-crc/projector.inc build/d1-deferred-crc/segments.inc build/d1-embed-crc/rows.bin \
  build/d1-embed-crc/rows.inc build/d1-vision-assets/depthwise.bin build/d1-vision-assets/rope.bin \
  build/d1-vision-assets/stem.bin build/d1-vision-assets/vision.inc

build/d1-vision-asm-sparse: $(SOURCES) $(INCLUDES) $(ASSETS)
	$(CC) -I. -O3 -mcpu=cortex-a72 $(ASM_FLAGS) -DCONTEXT=512 -DD1_BATCH -DD1_TIMING \
	  -DD1_LINK_EXISTING_NATIVE_MATH -DD1_VISION -DD1_SPARSE_VISUAL \
	  -DMODEL_INDEX='"build/d1-backbone-assets/model.inc"' \
	  -DD1_VISION_INDEX='"build/d1-vision-assets/vision.inc"' \
	  $(SOURCES) -static -lm -pthread -o $@
