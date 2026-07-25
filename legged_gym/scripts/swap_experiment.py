"""
Policy-switching demo, built on legged_gym/control/ (SimAdapter, PolicySupervisor,
SafetyGovernor, ControlService — see that package's README for the full
design writeup).

This script is the "supervised from a web UI, in simulation" corner of the
control architecture. The viser GUI buttons below call ControlService
methods directly, in-process — the same methods an autonomous Selector loop
or, eventually, a networked bridge to a real robot would call. Nothing here
is simulator-specific except the SimAdapter construction; everything from
"which policy is active" down is backend-agnostic.

Usage:
    python legged_gym/scripts/swap_experiment.py \
        --policy stable:/path/to/unitree_rl_gym/deploy/pre_train/g1/motion.pt \
        --policy cautious:logs/g1_cautious/<run>/exported/policy_lstm_1.pt \
        --active stable
"""
import argparse
import os
import time
from pathlib import Path

from fastapi.staticfiles import StaticFiles

from legged_gym import *
from legged_gym.envs import *
from legged_gym.utils import task_registry
from legged_gym.utils.viser_viewer import create_viser_viewer

from legged_gym.control import (
    SimAdapter, PolicySupervisor, SafetyGovernor, ControlService,
    load_policy, damping_policy,
)
from legged_gym.control.transport import ControlServer


def parse_policy_args(policy_args):
    """--policy name:/path/to/file.pt, repeatable."""
    policies = {}
    for spec in policy_args:
        name, path = spec.split(":", 1)
        policies[name] = path
    return policies


def main():
    parser = argparse.ArgumentParser(description="Policy-switching demo on G1 (see legged_gym/control/)")
    parser.add_argument('--policy', action='append', required=True, dest='policy_specs',
                         help="name:/path/to/policy_lstm_1.pt — repeatable, first one is the default active")
    parser.add_argument('--active', type=str, default=None, help="which --policy name starts active (default: first one given)")
    parser.add_argument('--ramp_ticks', type=int, default=15, help="control ticks to cross-fade over on a switch")
    parser.add_argument('--headless', action='store_true', default=False,
                         help="no viewer at all — runs a scripted smoke test (switch once, then exit)")
    parser.add_argument('--viser_port', type=int, default=9006,
                         help="Genesis's native viewer has a rasterizer indexing bug on this Mac/asset "
                              "combo — viser (web viewer) is the reliable way to actually watch this run.")
    parser.add_argument('--speed', type=float, default=0.35,
                         help="playback speed multiplier (1.0 = real-time 50Hz control rate)")
    parser.add_argument('--docs_port', type=int, default=None,
                         help="DEPRECATED — no-op if --control_port is set (the unified control web at "
                              "--control_port now serves docs/ itself; see web/index.html's Docs tab). "
                              "Only takes effect standalone, without --control_port: adds a link in the "
                              "viser panel to a docs server assumed running at http://localhost:<docs_port>/ "
                              "(e.g. `python -m http.server <docs_port>` run from the docs/ directory).")
    parser.add_argument('--control_port', type=int, default=None,
                         help="if set, starts a networked ControlServer (JSON-over-WebSocket at /ws, see "
                              "legged_gym/control/transport.py) on this port, exposing request_switch/"
                              "status/pause/resume/estop to external clients — the same five calls the "
                              "viser GUI buttons already make in-process. Unless --headless, this port "
                              "also serves the unified control web (web/index.html: Docs/Simulator tabs "
                              "+ persistent controls panel + keyboard shortcuts) at http://localhost:"
                              "<control_port>/, superseding --docs_port.")
    cli = parser.parse_args()

    policy_paths = parse_policy_args(cli.policy_specs)
    active_name = cli.active or next(iter(policy_paths))

    # unitree_rl_gym's own pretrained checkpoints store hidden_state/cell_state as a fixed
    # (1, 1, 64) buffer -> batch size must be exactly 1 to use them, so num_envs=1 throughout.
    args = argparse.Namespace(
        task="g1", headless=True, cpu=True, num_envs=1, max_iterations=None,
        resume=False, sync_wandb=False, export_onnx=False, debug=False, load_run=None,
        ckpt=-1, use_joystick=False, joystick_type='xbox', follow_robot=False,
        viewer='viser', viser_port=cli.viser_port, motion_file=None, motion_out_dir=None,
        num_student=None,
    )

    if SIMULATOR == "genesis":
        backend = os.environ.get("GENESIS_BACKEND", "cpu").lower()
        if backend == "cuda":
            try:
                gs.init(backend=gs.cuda, logging_level='warning')
                print("Genesis initialised with CUDA backend.")
            except Exception as e:
                print(f"Warning: CUDA backend failed ({e}), falling back to CPU.")
                gs.init(backend=gs.cpu, logging_level='warning')
        else:
            gs.init(backend=gs.cpu, logging_level='warning')

    env, env_cfg = task_registry.make_env(name=args.task, args=args)
    adapter = SimAdapter(env)

    hidden_size = 64  # matches G1RoughCfgPPO.policy.rnn_hidden_size
    print("Loading policies:")
    policies = {}
    for name, path in policy_paths.items():
        policies[name] = load_policy(name, path, num_obs=env_cfg.env.num_observations,
                                      hidden_size=hidden_size, num_envs=env.num_envs)
        print(f"  '{name}' <- {path}")
    policies["damping"] = damping_policy(env.num_envs, env_cfg.env.num_actions)

    supervisor = PolicySupervisor(policies, active=active_name, ramp_ticks=cli.ramp_ticks)
    safety = SafetyGovernor(supervisor, damping_policy_name="damping")
    service = ControlService(adapter, supervisor, safety, selector=None)

    control_server = None
    if cli.control_port is not None:
        control_server = ControlServer(service, port=cli.control_port)

    viser_viewer = None
    policy_label = None

    if not cli.headless:
        viser_viewer = create_viser_viewer(env, port=cli.viser_port)
        print(f"Viser web viewer started at http://localhost:{cli.viser_port}")

        policy_label = viser_viewer.server.gui.add_markdown("### Active policy: —", order=-100)

        if control_server is not None:
            # Mount the unified control web (Docs/Simulator tabs + controls
            # panel + keyboard shortcuts — web/index.html) onto the SAME
            # FastAPI app/port as the /ws transport, per HANDOFF_control_web.md
            # §3-B: one process, one port, same-origin WS (no CORS). This
            # retires --docs_port's separate http.server for anyone using
            # --control_port. Routes must be added before serve_in_thread().
            repo_root = Path(__file__).resolve().parents[2]

            @control_server.app.get("/config")
            def _web_config():
                return {"viser_port": cli.viser_port}

            control_server.app.mount(
                "/docs", StaticFiles(directory=str(repo_root / "docs"), html=True), name="docs",
            )
            control_server.app.mount(
                "/", StaticFiles(directory=str(repo_root / "web"), html=True), name="web",
            )
            viser_viewer.server.gui.add_markdown(
                f"[🖥 Open unified control web](http://localhost:{cli.control_port}/)", order=-99,
            )
        elif cli.docs_port is not None:
            viser_viewer.server.gui.add_markdown(
                f"[📖 Read the docs](http://localhost:{cli.docs_port}/)", order=-99,
            )

        restart_button = viser_viewer.server.gui.add_button(
            "Restart",
            hint="Reset the simulation and hold the current policy.",
        )
        restart_requested = {"flag": False}

        @restart_button.on_click
        def _(_) -> None:
            print("\n>>> Restart requested from the web UI <<<\n")
            restart_requested["flag"] = True

        pause_button = viser_viewer.server.gui.add_button(
            "Pause", hint="Freeze the sim in place. Click again to resume.",
        )

        @pause_button.on_click
        def _(_) -> None:
            if service.paused:
                service.resume()
                pause_button.label = "Pause"
                print("\n>>> Resumed from the web UI <<<\n")
            else:
                service.pause()
                pause_button.label = "Resume"
                print("\n>>> Paused from the web UI <<<\n")

        with viser_viewer.server.gui.add_folder("Policies"):
            for name in policy_paths:  # one button per loaded skill; "damping" isn't user-selectable
                switch_button = viser_viewer.server.gui.add_button(f"Switch to: {name}")

                def _make_handler(target_name):
                    def _handler(_):
                        ok = service.request_switch(target_name)
                        print(f"\n>>> Switch to '{target_name}' requested from the web UI "
                              f"({'queued' if ok else 'no-op, already active'}) <<<\n")
                    return _handler

                switch_button.on_click(_make_handler(name))
    else:
        restart_requested = {"flag": False}

    if control_server is not None:
        # Routes/mounts (if any — see the `if not cli.headless` block above)
        # must already be on control_server.app before this call.
        control_server.serve_in_thread()
        listening_at = f"ControlServer listening at ws://localhost:{cli.control_port}/ws"
        if not cli.headless:
            listening_at += f" — unified control web at http://localhost:{cli.control_port}/"
        print(listening_at)

    frame_dt = (1 / 60.0) / max(cli.speed, 0.01)

    def update_label():
        if policy_label is None:
            return
        status = service.status()
        color = "🟢" if not status["ramping"] else "🟡"
        if status["safety_tripped"]:
            color = "🔴"
        text = f"### {color} Active: **{status['active']}**"
        if status["pending"]:
            text += f" → switching to **{status['pending']}**"
        if status["paused"]:
            text += " (paused)"
        policy_label.content = text

    def run_headless_smoke_test():
        """No web UI at all: request one switch partway through, purely to
        prove the mechanism works end-to-end without a browser attached —
        this is the shape an autonomous on-robot process would drive."""
        other = next((n for n in policy_paths if n != active_name), None)
        if other is None:
            print(f"Only one policy loaded ('{active_name}') — running without a switch.")

        obs = adapter.get_observations()
        switched = False
        for i in range(80):
            if control_server is not None:
                control_server.drain_commands()
            if i == 40 and not switched and other is not None:
                print(f"[autonomous] requesting switch to '{other}' at step {i}")
                service.request_switch(other)
                switched = True
            action = service.tick(obs)
            adapter.send_action(action)
            obs = adapter.get_observations()
            if control_server is not None:
                control_server.publish_status(service.status())
            if i % 20 == 0:
                print(f"step {i:3d} | {service.status()}")
        print("Headless smoke test done.")

    if cli.headless:
        run_headless_smoke_test()
        return

    print(f"\nOpen http://localhost:{cli.viser_port} — use the Policies buttons to switch live.")
    obs = adapter.get_observations()
    while True:
        t_start = time.perf_counter()

        if control_server is not None:
            control_server.drain_commands()

        if restart_requested["flag"]:
            restart_requested["flag"] = False
            adapter.reset()
            obs = adapter.get_observations()
            safety.reset()

        action = service.tick(obs)
        if action is not None:
            adapter.send_action(action)
            obs = adapter.get_observations()
            if viser_viewer is not None:
                viser_viewer.update_from_simulator(env, 0)

        if control_server is not None:
            control_server.publish_status(service.status())

        update_label()

        elapsed = time.perf_counter() - t_start
        remaining = frame_dt - elapsed
        if remaining > 0:
            time.sleep(remaining)


if __name__ == '__main__':
    main()
