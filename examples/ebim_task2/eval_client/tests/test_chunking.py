from eval_client import chunking
import numpy as np
import pytest


def _chunk(length=50, dim=20, start=0.0):
    """Chunk whose row i is filled with start + i (easy to identify)."""
    return np.arange(start, start + length, dtype=np.float32)[:, None] * np.ones(dim, dtype=np.float32)


# Dim 3 is a left-arm joint (interpolated by blends); dims 0 (base twist)
# and 17 (left gripper) are never interpolated and take the fresh value.
ARM = 3
BASE = 0
GRIP = 17


class TestChunkExecutor:
    def test_needs_infer_initially(self):
        assert chunking.ChunkExecutor(replan_steps=25).needs_infer

    def test_load_keeps_replan_steps_and_20_dims(self):
        executor = chunking.ChunkExecutor(replan_steps=25)
        executor.load(_chunk(50, 32))
        for i in range(25):
            action = executor.next_action()
            assert action.shape == (20,)
            assert action[0] == i
        assert executor.needs_infer

    def test_rejects_bad_shapes(self):
        executor = chunking.ChunkExecutor(replan_steps=25)
        with pytest.raises(ValueError, match="Bad action chunk shape"):
            executor.load(np.zeros(50))
        with pytest.raises(ValueError, match="Bad action chunk shape"):
            executor.load(np.zeros((50, 7)))


class TestMergeStrategies:
    def test_registry_names(self):
        assert set(chunking.CHUNK_MERGE_STRATEGIES) == {
            "replace",
            "blend_uniform",
            "blend_favor_new",
            "blend_favor_old",
            "offset_decay",
            "crossfade",
        }

    def _buffer(self, merge, horizon=25, blend_steps=10):
        return chunking.AsyncChunkBuffer(
            execute_horizon=horizon, trigger_threshold=0.5, merge=merge, blend_steps=blend_steps
        )

    def test_replace_discards_old_queue(self):
        buffer = self._buffer("replace")
        buffer.on_chunk(0, _chunk(50, start=0.0), next_step=0)
        buffer.on_chunk(5, _chunk(50, start=100.0), next_step=5)
        action = buffer.pop()
        assert action[ARM] == 100.0  # fresh chunk row 0 (obs at step 5, no stale rows)

    def test_blend_uniform_averages_overlap(self):
        buffer = self._buffer("blend_uniform", horizon=4)
        buffer.on_chunk(0, _chunk(4, start=0.0), next_step=0)  # queue: 0,1,2,3
        buffer.on_chunk(0, _chunk(4, start=10.0), next_step=0)  # fresh: 10,11,12,13
        assert buffer.pop()[ARM] == pytest.approx(5.0)  # 0.5*0 + 0.5*10
        assert buffer.pop()[ARM] == pytest.approx(6.0)

    def test_blend_weights(self):
        for merge, expected in (("blend_favor_new", 7.0), ("blend_favor_old", 3.0)):
            buffer = self._buffer(merge, horizon=4)
            buffer.on_chunk(0, _chunk(4, start=0.0), next_step=0)
            buffer.on_chunk(0, _chunk(4, start=10.0), next_step=0)
            assert buffer.pop()[ARM] == pytest.approx(expected)  # w_old*0 + w_new*10

    def test_blend_never_interpolates_base_or_gripper(self):
        buffer = self._buffer("blend_uniform", horizon=4)
        buffer.on_chunk(0, _chunk(4, start=0.0), next_step=0)
        buffer.on_chunk(0, _chunk(4, start=10.0), next_step=0)
        action = buffer.pop()
        assert action[ARM] == pytest.approx(5.0)
        assert action[19] == pytest.approx(5.0)  # spine is interpolated too
        for dim in (0, 1, 2, 17, 18):
            assert action[dim] == 10.0  # fresh value, no averaging

    def test_blend_keeps_longer_tail(self):
        # Old queue longer than fresh chunk: fresh never shortens the plan.
        buffer = self._buffer("blend_uniform", horizon=6)
        buffer.on_chunk(0, _chunk(6, start=0.0), next_step=0)  # queue: 0..5
        buffer.pop()  # consume row 0; queue: 1..5 (len 5)
        # Fresh chunk observed at step 1 arriving at next_step=3 -> 2 stale rows
        # skipped, usable rows start at 12 with len 4 < queue len 5.
        buffer.on_chunk(1, _chunk(6, start=10.0), next_step=3)
        values = [buffer.pop()[ARM] for _ in range(5)]
        assert values[:4] == [pytest.approx(0.5 * (1 + i) + 0.5 * (12 + i)) for i in range(4)]
        assert values[4] == 5.0  # old tail preserved

    def test_offset_decay_bleeds_off_disagreement(self):
        buffer = self._buffer("offset_decay", horizon=6, blend_steps=4)
        buffer.on_chunk(0, _chunk(6, start=0.0), next_step=0)  # queue: 0..5
        buffer.pop()  # queue: 1..5, anchor for the merge is row 1
        # obs at step 1, arrives at step 3 -> rows 10,11 stale; usable 12..17
        buffer.on_chunk(1, _chunk(8, start=10.0), next_step=3)
        assert len(buffer) == 6  # old tail dropped: the fresh plan is the plan
        rows = [buffer.pop() for _ in range(6)]
        # delta = anchor - fresh[0] = 1 - 12 = -11, weights 1, .75, .5, .25, then 0
        assert [r[ARM] for r in rows] == [pytest.approx(v) for v in (1.0, 4.75, 8.5, 12.25, 16.0, 17.0)]
        assert [r[BASE] for r in rows] == [12.0, 13.0, 14.0, 15.0, 16.0, 17.0]
        assert [r[GRIP] for r in rows] == [12.0, 13.0, 14.0, 15.0, 16.0, 17.0]

    def test_offset_decay_anchors_on_last_action_when_dry(self):
        buffer = chunking.AsyncChunkBuffer(
            execute_horizon=2, trigger_threshold=0.0, merge="offset_decay", blend_steps=2
        )
        buffer.on_chunk(0, _chunk(2), next_step=0)
        buffer.pop()
        buffer.pop()  # last_action = row 1
        buffer.pop()  # dry: holds row 1
        buffer.on_chunk(3, _chunk(4, start=20.0), next_step=3)  # usable 20, 21
        first = buffer.pop()
        assert first[ARM] == pytest.approx(1.0)  # continuous with the held action
        assert first[BASE] == 20.0
        assert buffer.pop()[ARM] == pytest.approx(11.5)  # 21 + 0.5 * (1 - 20)

    def test_offset_decay_first_chunk_is_pass_through(self):
        buffer = self._buffer("offset_decay", horizon=4)
        buffer.on_chunk(0, _chunk(4, start=5.0), next_step=0)
        assert [buffer.pop()[ARM] for _ in range(4)] == [5.0, 6.0, 7.0, 8.0]

    def test_crossfade_ramps_into_fresh(self):
        buffer = self._buffer("crossfade", horizon=4, blend_steps=2)
        buffer.on_chunk(0, _chunk(4, start=0.0), next_step=0)  # queue: 0..3
        buffer.on_chunk(0, _chunk(4, start=10.0), next_step=0)  # fresh: 10..13
        rows = [buffer.pop() for _ in range(4)]
        # alpha = 1/3, 2/3 over the 2-row window, then pure fresh
        assert [r[ARM] for r in rows] == [pytest.approx(v) for v in (10 / 3, 1 / 3 + 22 / 3, 12.0, 13.0)]
        assert [r[BASE] for r in rows] == [10.0, 11.0, 12.0, 13.0]
        assert [r[GRIP] for r in rows] == [10.0, 11.0, 12.0, 13.0]

    def test_crossfade_window_clamped_to_overlap(self):
        buffer = self._buffer("crossfade", horizon=4, blend_steps=10)
        buffer.on_chunk(0, _chunk(4, start=0.0), next_step=0)
        buffer.on_chunk(0, _chunk(4, start=10.0), next_step=0)
        # alpha = .2, .4, .6, .8 across the whole 4-row overlap
        expected = [0.8 * 0 + 0.2 * 10, 0.6 * 1 + 0.4 * 11, 0.4 * 2 + 0.6 * 12, 0.2 * 3 + 0.8 * 13]
        assert [buffer.pop()[ARM] for _ in range(4)] == [pytest.approx(v) for v in expected]

    def test_crossfade_keeps_longer_tail(self):
        buffer = self._buffer("crossfade", horizon=6, blend_steps=2)
        buffer.on_chunk(0, _chunk(6, start=0.0), next_step=0)  # queue: 0..5
        buffer.pop()  # queue: 1..5 (len 5)
        buffer.on_chunk(1, _chunk(6, start=10.0), next_step=3)  # usable 12..15 (len 4)
        values = [buffer.pop()[ARM] for _ in range(5)]
        expected = [2 / 3 * 1 + 1 / 3 * 12, 1 / 3 * 2 + 2 / 3 * 13, 14.0, 15.0, 5.0]
        assert values == [pytest.approx(v) for v in expected]


class TestAsyncChunkBuffer:
    def test_validates_params(self):
        with pytest.raises(ValueError, match="execute_horizon"):
            chunking.AsyncChunkBuffer(execute_horizon=0, trigger_threshold=0.5)
        with pytest.raises(ValueError, match="trigger_threshold"):
            chunking.AsyncChunkBuffer(execute_horizon=25, trigger_threshold=1.0)
        with pytest.raises(ValueError, match="merge strategy"):
            chunking.AsyncChunkBuffer(execute_horizon=25, trigger_threshold=0.5, merge="nope")
        with pytest.raises(ValueError, match="blend_steps"):
            chunking.AsyncChunkBuffer(execute_horizon=25, trigger_threshold=0.5, blend_steps=0)

    def test_trigger_boundary(self):
        buffer = chunking.AsyncChunkBuffer(execute_horizon=25, trigger_threshold=0.5)
        assert buffer.trigger_len == 13
        buffer.on_chunk(0, _chunk(50), next_step=0)  # 25 queued
        assert not buffer.should_request()
        for _ in range(12):
            buffer.pop()
        assert len(buffer) == 13
        assert buffer.should_request()

    def test_zero_threshold_requests_before_dry(self):
        buffer = chunking.AsyncChunkBuffer(execute_horizon=25, trigger_threshold=0.0)
        assert buffer.trigger_len == 1
        buffer.on_chunk(0, _chunk(50), next_step=0)
        for _ in range(24):
            buffer.pop()
        assert len(buffer) == 1
        assert buffer.should_request()

    def test_no_request_while_in_flight(self):
        buffer = chunking.AsyncChunkBuffer(execute_horizon=25, trigger_threshold=0.5)
        assert buffer.should_request()
        buffer.on_request_sent()
        assert not buffer.should_request()

    def test_stale_prefix_skipped(self):
        buffer = chunking.AsyncChunkBuffer(execute_horizon=25, trigger_threshold=0.5)
        buffer.on_chunk(obs_step=10, chunk=_chunk(50), next_step=14)
        assert buffer.pop()[0] == 4.0  # rows 0-3 stale
        assert buffer.stale_skipped == 4

    def test_fully_stale_chunk_dropped_and_rerequestable(self):
        buffer = chunking.AsyncChunkBuffer(execute_horizon=25, trigger_threshold=0.5)
        buffer.on_request_sent()
        buffer.on_chunk(obs_step=0, chunk=_chunk(50), next_step=60)
        assert buffer.chunks_dropped == 1
        assert buffer.should_request()
        assert buffer.pop() is None  # nothing ever queued

    def test_dry_pop_holds_last_action_and_counts(self):
        buffer = chunking.AsyncChunkBuffer(execute_horizon=2, trigger_threshold=0.0)
        buffer.on_chunk(0, _chunk(2), next_step=0)
        buffer.pop()
        last = buffer.pop()
        assert buffer.consecutive_dry == 0
        held = buffer.pop()
        np.testing.assert_array_equal(held, last)
        assert buffer.consecutive_dry == 1
        buffer.pop()
        assert buffer.consecutive_dry == 2
        assert buffer.dry_events == 1
        buffer.on_chunk(4, _chunk(2, start=50.0), next_step=4)
        buffer.pop()
        assert buffer.consecutive_dry == 0
        assert buffer.dry_steps == 2

    def test_switch_and_step_jump_stats(self):
        buffer = chunking.AsyncChunkBuffer(execute_horizon=25, trigger_threshold=0.5)
        buffer.on_chunk(0, _chunk(50), next_step=0)
        buffer.pop()  # row 0: nothing to compare against yet
        buffer.pop()  # row 1: step jump 1
        fresh = _chunk(50, start=100.0)
        fresh[:, BASE] += 1000.0  # base / gripper dims never count as a jump
        fresh[:, GRIP] += 1000.0
        buffer.on_chunk(2, fresh, next_step=2)  # replace -> row 100 next
        buffer.pop()  # switch: 100 - 1 = 99
        buffer.pop()  # row 101: step jump 1
        stats = buffer.stats()
        assert stats["switch_jump_max"] == pytest.approx(99.0)
        assert stats["switch_jump_mean"] == pytest.approx(99.0)
        assert stats["step_jump_max"] == pytest.approx(99.0)
        assert stats["step_jump_mean"] == pytest.approx((1 + 99 + 1) / 3, abs=1e-4)

    def test_dry_pops_do_not_count_as_jumps(self):
        buffer = chunking.AsyncChunkBuffer(execute_horizon=2, trigger_threshold=0.0)
        buffer.on_chunk(0, _chunk(2), next_step=0)
        buffer.pop()
        buffer.pop()  # step jump 1
        buffer.pop()  # dry hold: not a jump
        assert buffer.stats()["step_jump_mean"] == pytest.approx(1.0)

    def test_stats_dict(self):
        buffer = chunking.AsyncChunkBuffer(execute_horizon=25, trigger_threshold=0.5)
        buffer.on_chunk(0, _chunk(50), next_step=3)
        buffer.pop()
        stats = buffer.stats()
        assert stats["chunk_latency_steps_mean"] == 3.0
        assert stats["chunk_latency_steps_max"] == 3
        assert stats["stale_steps_skipped"] == 3
        assert stats["queue_dry_steps"] == 0
        assert stats["queue_dry_events"] == 0
        assert stats["queue_depth_at_pop_mean"] == 25.0
        assert stats["queue_depth_at_pop_min"] == 25
        assert stats["chunks_dropped"] == 0

    def test_stats_empty(self):
        stats = chunking.AsyncChunkBuffer(execute_horizon=25, trigger_threshold=0.5).stats()
        assert stats["chunk_latency_steps_mean"] is None
        assert stats["queue_depth_at_pop_mean"] is None
        assert stats["switch_jump_max"] is None
        assert stats["switch_jump_mean"] is None
        assert stats["step_jump_max"] is None
        assert stats["step_jump_mean"] is None
