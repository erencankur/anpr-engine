# Detection architecture and cited influences

The detector is a project-owned, project-trained PyTorch implementation. It is
a one-class, anchor-free, multi-scale detector with stride 8/16/32 features,
FPN/PAN-style fusion, decoupled prediction heads, focal-style objectness loss,
GIoU box regression, dynamic positive assignment, and non-maximum suppression.

The implementation is informed by established work and is not presented as
independent of that work:

- Redmon et al., [You Only Look Once](https://arxiv.org/abs/1506.02640).
- Ge et al., [YOLOX](https://arxiv.org/abs/2107.08430), including the SimOTA
  assignment direction.
- Lin et al., [Feature Pyramid Networks](https://arxiv.org/abs/1612.03144).
- Liu et al., [Path Aggregation Network](https://arxiv.org/abs/1803.01534).
- Lin et al., [Focal Loss](https://arxiv.org/abs/1708.02002).
- Rezatofighi et al., [Generalized IoU](https://arxiv.org/abs/1902.09630).

Public source includes the architecture, loss, training, evaluation, checkpoint,
and inference contracts. Meaningful training and evaluation require a lawful
user-supplied dataset; meaningful project inference requires the approved
private detector weight. Public documentation reports only approved percentages,
not exact private populations or record-level evidence.

## Quadrilateral v2 candidate boundary

The repository also contains an experimental `yolox_multiscale_quad_v2` implementation. It keeps
the V1 backbone and FPN/PAN parameter names, adds an eight-value ordered-corner branch to each
decoupled head, and retains the box task for positive assignment and NMS. Its checkpoint format,
annotation manifest, geometry validation, result contracts, and crop policy are separately
versioned; a V2 checkpoint never overwrites or masquerades as V1.

V2 is not a released model and has no public accuracy claim. Official training, candidate
selection, and rollout require approved true-corner annotations and private acceptance evidence.
Invalid predicted geometry fails closed as unresolved and is not silently replaced with the V1
box-plus-heuristic crop while being described as a detector-predicted boundary.

## External provider boundary

Candidate providers do not become dependencies of the active service runtime. Provider-native
bbox arrays are mapped through a strict, provider-independent contract that requires `[N, 4]`
XYXY boxes, one finite confidence and integer class per box, non-degenerate geometry, and
confidence values in `[0, 1]`. Geometry is preserved exactly—out-of-frame coordinates are not
silently clipped—and malformed output fails closed.

The MMDetection adapter accepts the public `DetDataSample.pred_instances` shape behind a structural
interface, so importing the ANPR service does not import MMDetection. Class/confidence filtering is
separate from coordinate mapping and must be explicitly configured. This boundary is candidate
infrastructure only; it does not change the active project-owned detector.
