from verl.workers.rollout.hf_rollout import _merge_eos_token_ids


def test_merge_eos_token_ids_appends_and_deduplicates_completion_tokens():
    assert _merge_eos_token_ids(2, [7, 9]) == [2, 7, 9]
    assert _merge_eos_token_ids([2, 7], [7, 9]) == [2, 7, 9]


def test_merge_eos_token_ids_preserves_original_type_without_completion_tokens():
    assert _merge_eos_token_ids(2, []) == 2
    assert _merge_eos_token_ids([2, 3], []) == [2, 3]
