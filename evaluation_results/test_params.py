# optimizer hyperparams version 1
optimizer = ElasticChainOptimizer(
    sub_tablet_text=sub_tablet_text_aligned,
    detection_heatmap=sub_tablet_detection.heatmap,
    detection_boxes=sub_tablet_detection.to_detection_list(),
    scale_factor=scale_factor,
    lambda_data=10000.0,
    lambda_iou=20000.0,
    lambda_seq=1.0,
    lambda_smooth=0.15,
    lambda_anchor=0.05,
    alpha_geo=0.0,            # 20% geometric (existence), 80% semantic (per-class)
    prior_aspect_ratio=avg_width / avg_height,
    device=device
)

# optimizer hyperparams version 2 (after search, and tune manually)

optimizer = ElasticChainOptimizer(
    sub_tablet_text=sub_tablet_text_aligned,
    detection_heatmap=sub_tablet_detection.heatmap,
    detection_boxes=sub_tablet_detection.to_detection_list(),
    scale_factor=scale_factor,
    lambda_data=50000.0,
    lambda_iou=20000.0,
    lambda_seq=0.10,
    lambda_smooth=0.05,
    lambda_anchor=0.1,
    alpha_geo=0.0,            # 20% geometric (existence), 80% semantic (per-class)
    prior_aspect_ratio=avg_width / avg_height,
    device=device
)