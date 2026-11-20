import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HARNESS = Path(os.environ.get("GRID_HARNESS", ROOT / "third_party" / "AeroVLA")).resolve()
os.chdir(HARNESS)
sys.path[:0] = [str(HARNESS), str(HARNESS / "src" / "vlnce_src"), str(ROOT), str(ROOT / "third_party")]

import torch
import tqdm

from src.common.param import args, model_args
from src.vlnce_src.assist import Assist
from src.vlnce_src.closeloop_util import (
    BatchIterator, CheckPort, EvalBatchState, initialize_env_eval, is_dist_avail_and_initialized, setup,
)
from utils.logger import logger

from grid.actions import direction_hint, navigation_prompt, object_description, parse_action
from grid.policy import GRIDPolicy, to_pil


class GRIDWrapper:
    def __init__(self, weights_dir):
        # GRID_BASE_MODEL lets offline clusters point at a local OpenVLA-7B copy instead of the Hub.
        base_model = os.environ.get("GRID_BASE_MODEL")
        self.policy = GRIDPolicy(weights_dir, base_model=base_model) if base_model else GRIDPolicy(weights_dir)

    def eval(self):
        self.policy.model.eval()

    def prepare_inputs(self, episodes, target_positions, instructions=None):
        batch = []
        for episode, target, instruction in zip(episodes, target_positions, instructions):
            state = episode[-1]["sensors"]["state"]
            frames = episode[-1]["rgb"]
            hint = direction_hint(state["position"], state["orientation"], target)
            prompt = navigation_prompt(hint, object_description(instruction))
            front, down = to_pil(frames[0], bgr=True), to_pil(frames[4], bgr=True)
            batch.append(self.policy.prepare(front, down, prompt))
        return batch, None

    def run(self, inputs, episodes=None, rot_to_targets=None):
        actions, stops = [], []
        for item in inputs:
            action = parse_action(self.policy.generate(item))
            actions.append({"fwd": action["fwd"], "down": action["down"], "yaw": action["yaw"]})
            stops.append(action["land"])
        return actions, stops


def evaluate(wrapper, assist, env):
    wrapper.eval()
    with torch.no_grad():
        total = len(BatchIterator(env))
        progress = tqdm.tqdm(total=total)
        while True:
            batch = env.next_minibatch()
            if batch is None:
                break
            state = EvalBatchState(batch_size=env.batch_size, env_batchs=batch, env=env, assist=assist)
            progress.update(n=env.batch_size)
            inputs, _ = wrapper.prepare_inputs(state.episodes, state.target_positions, instructions=state.instructions)
            for step in range(int(args.maxWaypoints) + 1):
                logger.info(f"Step: {step} \t Completed: {int(env.index_data) - int(env.batch_size)} / {total}")
                if state.check_batch_termination(step):
                    break
                actions, stops = wrapper.run(inputs=inputs, episodes=state.episodes)
                env.makeActions(actions)
                time.sleep(0.01)
                state.update_from_env_output(env.get_obs())
                state.predict_dones = stops
                state.update_metric()
                inputs, _ = wrapper.prepare_inputs(state.episodes, state.target_positions,
                                                   instructions=state.instructions)
        progress.close()


def main():
    os.makedirs(args.eval_save_path, exist_ok=True)
    setup()
    if not CheckPort():
        raise SystemExit(f"simulator server is not reachable on port {args.simulator_tool_port}")
    env = initialize_env_eval(dataset_path=args.dataset_path, save_path=args.eval_save_path,
                              eval_json_path=args.eval_json_path)
    if is_dist_avail_and_initialized():
        torch.distributed.destroy_process_group()
    args.DistributedDataParallel = False
    wrapper = GRIDWrapper(model_args.model_path)
    evaluate(wrapper, Assist(always_help=False, use_gt=False), env)
    env.delete_VectorEnvUtil()


if __name__ == "__main__":
    main()
