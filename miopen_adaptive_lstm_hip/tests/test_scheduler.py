from miopen_adaptive_lstm_hip.scheduler import (
    DynamicAlgoConfig,
    build_adaptive_plan,
    check_dynamic_algo_selection,
    choose_hidden_update_launch,
    hidden_prop_segments,
    masked_pow2_range,
)


def test_dynamic_selector_allows_project_inference_adaptation():
    assert check_dynamic_algo_selection(
        DynamicAlgoConfig(algo_mode="rounded_dynamic", fwd_mode="inference")
    )


def test_dynamic_selector_can_match_miopen_inference_rejection():
    assert not check_dynamic_algo_selection(
        DynamicAlgoConfig(
            algo_mode="rounded_dynamic",
            fwd_mode="inference",
            allow_inference_adaptation=False,
        )
    )


def test_hidden_launch_matches_miopen_formula_for_h128():
    launch = choose_hidden_update_launch(
        max_compute_units=120,
        wavefront_width=64,
        max_batch=512,
        hidden_size=128,
    )
    assert launch.max_active_threads == 245760
    assert launch.total_work == 65536
    assert launch.read_block == 1
    assert launch.items_per_group == 256


def test_hidden_launch_uses_read_block_four_when_work_is_large():
    launch = choose_hidden_update_launch(
        max_compute_units=120,
        wavefront_width=64,
        max_batch=8192,
        hidden_size=128,
    )
    assert launch.read_block == 4


def test_pow2_segmentation():
    assert masked_pow2_range(13) == (8, 4, 1)
    assert hidden_prop_segments(13) == hidden_prop_segments(13)
    assert [segment.size for segment in hidden_prop_segments(13)] == [8, 4, 1]


def test_plan_selects_mfma_candidate_for_h128():
    plan = build_adaptive_plan(
        batch_size=512,
        seq_len=1000,
        input_size=7,
        hidden_size=128,
        num_layers=2,
    )
    assert plan.use_dynamic_algo
    assert plan.recurrent_algo == "adaptive_tiled_mfma_candidate"

