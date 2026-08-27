"""Autonomous episodic eval: N episodes with scene resets between them.

Videos and official IoU scoring come from the separately running
eval_recorder service (benchmark repo, services/eval_recording/): it
auto-scores the ending episode on each scene-reset *request* and segments
episodes on each reset *event*. This client only publishes actions,
requests resets, and logs timing/abort metadata to result.jsonl.

Usage (inside a ROS 2 Jazzy environment with openpi-client installed):
  python3 -m eval_client.main --host <policy-server> --port 8000
"""

import logging

import rclpy

from eval_client import episode
from eval_client import results
from eval_client.config import EvalConfig
from eval_client.inference import InferenceWorker
from eval_client.inference import PolicyClient
from eval_client.ros_node import EvalClientNode
from eval_client.ros_node import spin_in_thread


def _final_reset(node) -> None:
    """One last reset request after the final episode so the eval_recorder
    scores it (the recorder evaluates on each reset request)."""
    if not episode.wait_for(node.reset_request_has_subscriber, timeout_s=10.0):
        logging.warning("final reset request skipped: no subscriber on the reset topic")
        return
    before = node.reset_count()
    node.request_reset()
    if not episode.wait_for(lambda: node.reset_count() > before, timeout_s=15.0):
        logging.warning("final scene reset not confirmed")


def main() -> None:
    cfg = EvalConfig.from_cli()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    rclpy.init()
    node = EvalClientNode()
    spin_thread = spin_in_thread(node)

    logging.info("Waiting for policy server at %s:%s", cfg.host, cfg.port)
    policy_client = PolicyClient(cfg.host, cfg.port, cfg.api_key, cfg.max_reconnects)
    server_metadata = policy_client.metadata()
    logging.info("Server metadata: %s", server_metadata)

    worker = None
    if cfg.async_inference:
        # The worker owns the websocket client from here on; the main
        # loop must not call policy_client.infer itself.
        worker = InferenceWorker(policy_client)
        worker.start()
        logging.info(
            "Async inference on: horizon=%s threshold=%.2f merge=%s",
            cfg.execute_horizon,
            cfg.chunk_size_threshold,
            cfg.chunk_merge,
        )

    layout = results.run_layout(cfg.out_dir, cfg.run_name)
    # redacted: host/port/api_key stay out of the shareable eval artifact
    results.dump_run_config(layout.config_path, cfg.redacted_dict(), server_metadata)
    logging.info("Run %s: writing results to %s", layout.run_dir.name, layout.results_path)

    try:
        handled_resets = node.reset_count()
        for index in range(cfg.num_episodes):
            # Streams first: forces DDS discovery to settle and proves the
            # sim is alive before a reset is requested.
            if not episode.wait_for(lambda: not node.missing_streams(), timeout_s=30.0):
                raise RuntimeError(
                    f"observation streams not live: {node.missing_streams()} (is the sim running with --record?)"
                )
            logging.info("Episode %d/%d: resetting scene", index + 1, cfg.num_episodes)
            handled_resets = episode.ensure_fresh_scene(node, handled_resets)
            if not episode.wait_for(lambda: node.streams_fresh_since(node.last_reset_event_wall()), timeout_s=30.0):
                raise RuntimeError(f"streams stale after scene reset: {node.missing_streams()}")

            record = {"episode": index, "aborted": False}
            try:
                if worker is not None:
                    record.update(episode.run_episode_async(node, worker, cfg, generation=index))
                else:
                    record.update(episode.run_episode_sync(node, policy_client, cfg))
            except (episode.EpisodeAbortedError, RuntimeError) as exc:
                node.publish_stop()
                logging.warning("Episode %d aborted: %s", index, exc)
                record.update({"aborted": True, "abort_reason": str(exc)})
            results.append_result(layout.results_path, record)
            logging.info("Episode %d result: %s", index, record)
            if worker is not None and not worker.is_alive():
                raise RuntimeError(f"inference worker died: {worker.error}")

        if cfg.final_reset:
            _final_reset(node)
        logging.info("Run summary: %s", results.summarize(layout.results_path))
    except KeyboardInterrupt:
        logging.info("Interrupted; stopping robot")
    finally:
        node.publish_stop()
        if worker is not None:
            worker.stop()
        rclpy.shutdown()
        spin_thread.join(timeout=2.0)


if __name__ == "__main__":
    main()
