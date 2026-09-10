# The CNN Blueprint — distilled memory

Source: `Notes/The_CNN_Blueprint.pdf` (NotebookLM visual primer, 14 scanned pages).
Full slide-by-slide OCR: [`The_CNN_Blueprint_OCR.md`](The_CNN_Blueprint_OCR.md).

## Core story

1. **Demystifying CNNs** — how computers learn to “see”; images become digital data.
2. **Image as spreadsheet** — humans see shapes/colors/depth; computers split **RGB** channels into matrices of intensity numbers; goal is meaning in that sea of numbers.
3. **Parameter explosion** — fully connected nets connect every pixel to every neuron (28×28 → thousands of connections; 200×200 → ~120k instantly): too heavy, slow, destroys spatial layout. CNNs use **local connectivity** — fewer connections, preserves layout.
4. **Inspired by visual cortex** — biological neurons fire for specific edges (vertical/horizontal/diagonal). CNNs: specialized filters sweep locally and fire when their feature appears. Pipeline sketch: input → filters/feature maps → pooling → fully connected.
5. **Four pillars:** Convolution → ReLU → Pooling → Fully Connected (classification).

## Convolution (“matching game”)

- Small **filters/kernels** slide over the image asking whether each patch matches the sought feature.
- Example bipolar encoding: black=1, white=-1.
- Match math (primer style): line up → element-wise multiply → **average** (sum/9); perfect match score **1**; mismatch lowers score.
- Output = **feature map** (spatial heatmap of where that feature fired). Many filters → stacked deck of heatmaps.

## ReLU

- \(f(x)=\max(0,x)\): snap negatives to 0.
- Prevents positive/negative canceling later; isolates where features are present.

## Pooling (max)

- Typical: **2×2 window, stride 2**, keep max.
- Shrinks maps for speed; keeps strongest clues; more **translation-tolerant** (presence over exact pixel location).

## Depth / stacking

- Repeat Conv→ReLU→Pool.
- Early: edges/lines → Middle: corners/circles → Deep: objects (eyes, wheels, characters).

## Fully connected + decision

- Flatten stacked maps to a **1D vector**, then classic dense net → class scores (e.g. X / triangle / square).
- **Final verdict:** compare flattened clues to stored ideal patterns per class; higher match score wins (primer example: X score 0.91 vs O 0.51 → predict X).

## Anatomy summary (page 14)

| Step | Metaphor | Math action | Output |
|------|----------|-------------|---------|
| Convolution | Magnifying glass | Sliding filters match local features | Feature maps (heatmaps) |
| ReLU | Noise filter | Negatives → 0 | Rectified maps |
| Pooling | Shrink ray | Max over moving window | Compressed dominant maps |
| Fully connected | Judge | Flatten + tally scores | Final category prediction |

## Tie-in to this project

These ideas underwrite the DR pipeline: retinal photo → tensor → EfficientNet (stacked conv features) → dense head → 5-class softmax, with Grad-CAM inspecting which spatial clues drove the grade.
