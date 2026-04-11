#!/bin/bash
# Convert Manim-rendered MP4s to high-quality optimized GIFs
# Uses ffmpeg two-pass palette generation — much better than Manim's built-in GIF
# Output: 854x480 GIFs at 12fps with exact color palette
# Note: uses /bin/bash compatible syntax (no associative arrays)

OUT="/Users/akashsingh/Documents/akashicMarga.github.io/assets/images/codec-animations"
MEDIA="/Users/akashsingh/Documents/akashicMarga.github.io/assets/animations/media/videos"

# Format: "slug:ClassName"
SCENES=(
  "waveform_to_spectrogram:WaveformToSpectrogram"
  "mel_filterbank:MelFilterbank"
  "source_filter_model:SourceFilterModel"
  "rvq_deep_dive:RVQDeepDive"
  "vector_quantization:VectorQuantization"
  "rvq_residuals:RVQResiduals"
  "codebook_ema:CodebookEMA"
  "snac_multiscale:SNACMultiscale"
  "mimi_streaming:MimiStreaming"
  "gan_decoder:GANDecoder"
)

mkdir -p "$OUT"

for entry in "${SCENES[@]}"; do
  slug="${entry%%:*}"
  cls="${entry##*:}"

  mp4="$MEDIA/$slug/480p15/${cls}_ManimCE_v0.19.0.mp4"

  # Fallback: some Manim versions omit the _ManimCE_vX.Y.Z suffix
  if [ ! -f "$mp4" ]; then
    mp4="$MEDIA/$slug/480p15/${cls}.mp4"
  fi

  gif="$OUT/$slug.gif"

  if [ ! -f "$mp4" ]; then
    echo "MISSING: $MEDIA/$slug/480p15/${cls}[_ManimCE_v0.19.0].mp4 — skipping"
    continue
  fi

  echo "Converting $cls -> $gif ..."

  # Two-pass: first generate optimal palette, then apply it
  PALETTE="/tmp/${slug}_palette.png"
  ffmpeg -y -i "$mp4" \
    -vf "fps=12,scale=854:-1:flags=lanczos,palettegen=max_colors=256:stats_mode=diff" \
    "$PALETTE" 2>/dev/null

  ffmpeg -y -i "$mp4" -i "$PALETTE" \
    -lavfi "fps=12,scale=854:-1:flags=lanczos [x]; [x][1:v] paletteuse=dither=bayer:bayer_scale=5:diff_mode=rectangle" \
    "$gif" 2>/dev/null

  rm -f "$PALETTE"
  SIZE=$(du -sh "$gif" | cut -f1)
  echo "  -> $gif  ($SIZE)"
done

echo ""
echo "All GIFs written to $OUT"
ls -lh "$OUT"/*.gif
