graph [
  comment "Euclidean Gradient Field — Conservative Semantic Progression Mapping"
  directed 1
  id 1301
  label "Euclidean_Gradient_Space"

  node [
    id 1
    label "Base_Premise"
    x -1.0 y -1.0 z 0.0
    curve_id "gamma_flat_1"
    parameter_t 0.0
    arc_length 0.0
    curvature 0.0
    scalar_potential 0.20
    static_weight 0.20
  ]

  node [
    id 2
    label "Derived_Inference"
    x 1.0 y 1.0 z 0.0
    curve_id "gamma_flat_1"
    parameter_t 1.0
    arc_length 2.8284
    curvature 0.0
    scalar_potential 0.95
    static_weight 0.95
  ]

  edge [
    source 1 target 2
    label "steepest_ascent_line"
    verb "@requires"
    weight 1.0
    curve_id "gamma_flat_1"
    t0 0.0 t1 1.0
    field_kind "euclidean_gradient"
    vx 2.0 vy 2.0 vz 0.0
    curl_x 0.0 curl_y 0.0 curl_z 0.0
    potential_delta 0.75
    loss_cost 0.0
  ]
]
