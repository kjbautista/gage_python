"""
Control Newport XPS motion stage via command line or external script.
Developed by Kathlyne Jayne B. Bautista (2026-04-08)

Connection parameters (IP, username, password, timeouts) are taken from
``run_gage.config.load_gage_config()`` and can be overridden via environment
variables (``GAGE_XPS_IP``, ``GAGE_XPS_USER``, ``GAGE_XPS_PASSWORD``) or
``run_gage/config.local.json``. CLI flags still take precedence when provided.

Usage:
    python xps_control_main.py [action] [options]

Command Line Examples:
    python xps_control_main.py check_status
    python xps_control_main.py init
    python xps_control_main.py rel_move Group1 10.0
    python xps_control_main.py abs_move Group2 50.0
    python xps_control_main.py get_positions
    python xps_control_main.py get_limits
    python xps_control_main.py disconnect

MATLAB Examples (using pyrunfile):
    % Note: When using pyrunfile, variables passed as name-value pairs
    % are injected into the global scope. The script checks for these.

    % 1. Connect and get the XPS object (run once)
    [out, xps] = pyrunfile('xps_control_main.py', ["out", "xps"]);

    % 2. Check status using the existing connection
    out = pyrunfile('xps_control_main.py', "out", xps=xps, action='check_status');

    % 3. Move a stage (relative move of 10mm for Group1)
    out = pyrunfile('xps_control_main.py', "out", xps=xps, action='rel_move', group='Group1', distance=10.0);

    % 4. Move a stage (absolute move to 50mm for Group2)
    out = pyrunfile('xps_control_main.py', "out", xps=xps, action='abs_move', group='Group2', position=50.0);

    % 5. Get current positions
    out = pyrunfile('xps_control_main.py', "out", xps=xps, action='get_positions');
    positions = double(out['positions']); % Convert numpy array to MATLAB double
    stage_names = string(out['stage_names']); % Convert list to string array

    % 6. Get hardware limits
    out = pyrunfile('xps_control_main.py', "out", xps=xps, action='get_limits');
    limits = double(out['limits']); % Convert numpy array to MATLAB double
    stage_names = string(out['stage_names']); % Convert list to string array
"""
import argparse
import logging
import time
import sys
import numpy as np
from run_gage.newportxps import NewportXPS

from run_gage.config import load_gage_config
from run_gage.models import StageInitializationError


LOGGER = logging.getLogger(__name__)


def check_status(xps):
    """Check init/home status."""
    gstat = xps.get_group_status()
    istat = [1 if gstat[g] != 'Not initialized state' else 0 for g in xps.groups]
    hstat = [1 if gstat[g] not in ('Not initialized state', 'Not referenced state') else 0 for g in xps.groups]
    return istat, hstat


def _wait_for_group_state(xps, gname, forbidden_state, *, timeout_s: float, poll_s: float) -> str:
    """Poll ``xps.get_group_status()`` until ``gname`` leaves ``forbidden_state``.

    Raises:
        StageInitializationError: if the group stays in ``forbidden_state`` for
            longer than ``timeout_s`` seconds. The error carries the group
            name, last-observed state, and elapsed wait time.
    """

    deadline = time.monotonic() + float(timeout_s)
    last_state = xps.get_group_status()[gname]
    while last_state == forbidden_state:
        if time.monotonic() >= deadline:
            elapsed = time.monotonic() - (deadline - float(timeout_s))
            raise StageInitializationError(
                f"{gname} remained in '{forbidden_state}' for {elapsed:.1f}s "
                f"(timeout {timeout_s:.1f}s)."
            )
        time.sleep(poll_s)
        last_state = xps.get_group_status()[gname]
    return last_state


def initialize_stage(xps):
    """Init and home all groups with bounded per-phase timeouts.

    Raises ``StageInitializationError`` if any group fails to leave its initial
    "Not initialized" or "Not referenced" state within the configured window.
    """
    cfg = load_gage_config().xps
    for gname in xps.groups:
        if xps.get_group_status()[gname] == 'Not initialized state':
            print(f"Initializing {gname}...")
            xps.initialize_group(gname)
            _wait_for_group_state(
                xps, gname, 'Not initialized state',
                timeout_s=cfg.initialize_timeout_s,
                poll_s=cfg.initialize_poll_s,
            )

        if xps.get_group_status()[gname] == 'Not referenced state':
            print(f"Homing {gname}...")
            xps.home_group(gname)
            _wait_for_group_state(
                xps, gname, 'Not referenced state',
                timeout_s=cfg.initialize_timeout_s,
                poll_s=cfg.initialize_poll_s,
            )
    return check_status(xps)


def move_stage(xps, group, val, relative=True):
    """Move stage relative or absolute."""
    if group not in xps.groups: raise ValueError(f"Group '{group}' not found.")
    mtype = "relative" if relative else "absolute"
    print(f"Moving {group} {mtype} to {val} mm...")
    LOGGER.debug('move_stage: group=%s type=%s val=%.6g', group, mtype, float(val))
    xps.move_stage(f'{group}.Pos', val, relative=relative)
    pos = xps.get_stage_position(f'{group}.Pos')
    print(f"New position: {pos} mm")
    return pos


def get_positions(xps):
    """Get all group positions.
    Returns:
        stage_names (list): List of stage names.
        positions (list): List of positions matching the stage names.
    """
    stage_names = []
    positions = []
    for gname in xps.groups:
        for pname in xps.groups[gname]['positioners']:
            stage = f'{gname}.{pname}'
            stage_names.append(stage)
            positions.append(xps.get_stage_position(stage))

    return stage_names, np.array(positions)

def get_limits(xps):
    """Get hardware travel limits for each positioner.
    Returns:
        stage_names (list): List of stage names.
        limits (list): List of [min, max] lists matching the stage names.
    """
    stage_names = []
    limits = []
    for gname in xps.groups:
        for pname in xps.groups[gname]['positioners']:
            stage_name = f"{gname}.{pname}"
            stage_names.append(stage_name)
            if stage_name in xps.stages:
                # newportxps populates these during connection
                low = xps.stages[stage_name].get('low_limit', float('nan'))
                high = xps.stages[stage_name].get('high_limit', float('nan'))
                limits.append([low, high])
            else:
                limits.append([float('nan'), float('nan')])

    # Convert limits to a NumPy array for easier MATLAB conversion
    # MATLAB: limits = double(out.limits)
    return stage_names, np.array(limits)

def set_velocity(xps, group, velocity, acceleration=None):
    """Set change velocity and acceleration for a stage."""
    # Handle single positioner groups simply by appending .Pos if needed
    full_stage_name = f'{group}.Pos'
    if group not in xps.groups:
         # Check if user passed full stage name
         if group in xps.stages:
             full_stage_name = group
         else:
             raise ValueError(f"Group '{group}' not found.")

    print(f"Setting velocity for {full_stage_name} to {velocity} (Accel: {acceleration})...")
    xps.set_velocity(full_stage_name, velocity, accl=acceleration)
    return xps.stages[full_stage_name].get('max_velo'), xps.stages[full_stage_name].get('max_accel')

def reset_velocities(xps):
    """Reset all stages to max velocity and acceleration."""
    results = {}
    stage_names = []
    max_velos = []
    max_accels = []

    for stage_name, stage_info in xps.stages.items():
        max_v = stage_info.get('max_velo')
        max_a = stage_info.get('max_accel')
        if max_v is not None and max_a is not None:
             print(f"Resetting {stage_name} -> V={max_v}, A={max_a}")
             xps.set_velocity(stage_name, max_v, accl=max_a)
             stage_names.append(stage_name)
             max_velos.append(max_v)
             max_accels.append(max_a)

    return stage_names, np.array(max_velos), np.array(max_accels)


def get_xps_object(ip: str | None = None, user: str | None = None, pwd: str | None = None):
    """Connect and return an XPS object using the merged config as default credentials."""
    cfg = load_gage_config().xps
    ip = ip if ip is not None else cfg.ip
    user = user if user is not None else cfg.username
    pwd = pwd if pwd is not None else cfg.password
    print(f"Connecting to {ip}...")
    LOGGER.info('Opening XPS connection to %s (user=%s)', ip, user)
    return NewportXPS(ip, username=user, password=pwd)


def main(xps=None, action=None, group=None, distance=None, position=None, velocity=None, acceleration=None, ip=None, user=None, pwd=None):
    cfg = load_gage_config().xps
    if __name__ == "__main__" and action is None:
        p = argparse.ArgumentParser(description='Newport XPS Controller')
        p.add_argument('--ip', default=cfg.ip); p.add_argument('--user', default=cfg.username); p.add_argument('--pwd', default=cfg.password)
        sp = p.add_subparsers(dest='action', required=False) # action not required if just connecting
        sp.add_parser('check_status'); sp.add_parser('init'); sp.add_parser('get_positions'); sp.add_parser('disconnect'); sp.add_parser('get_limits')
        sp.add_parser('reset_velocities')

        op = sp.add_parser('rel_move'); op.add_argument('group'); op.add_argument('distance', type=float)
        ap = sp.add_parser('abs_move'); ap.add_argument('group'); ap.add_argument('position', type=float)

        vp = sp.add_parser('set_velocity'); vp.add_argument('group'); vp.add_argument('velocity', type=float); vp.add_argument('--accel', type=float, default=None)

        args = p.parse_args()
        action, ip, user, pwd = args.action, args.ip, args.user, args.pwd
        group = getattr(args, 'group', None)
        distance = getattr(args, 'distance', None)
        position = getattr(args, 'position', None)
        velocity = getattr(args, 'velocity', None)
        acceleration = getattr(args, 'accel', None)

    out = {}
    try:
        if xps is None: xps = get_xps_object(ip, user, pwd)

        if action == 'check_status':
            out['init_status'], out['home_status'] = check_status(xps)
            print("\n--- Status ---")
            for i, g in enumerate(xps.groups):
                print(f"  {g}: Init={'Yes' if out['init_status'][i] else 'No'}, Home={'Yes' if out['home_status'][i] else 'No'}")

        elif action == 'init':
            out['init_status'], out['home_status'] = initialize_stage(xps)
            print("\n--- Init/Home Complete ---")

        elif action == 'rel_move':  out['position'] = move_stage(xps, group, distance, True)
        elif action == 'abs_move':  out['position'] = move_stage(xps, group, position, False)

        elif action == 'set_velocity':
            out['max_velo'], out['max_accel'] = set_velocity(xps, group, velocity, acceleration)
            print("\n--- Velocity Set ---")

        elif action == 'reset_velocities':
            out['stage_names'], out['max_velos'], out['max_accels'] = reset_velocities(xps)
            print("\n--- Velocities Reset ---")

        elif action == 'get_positions':
            out['stage_names'], out['positions'] = get_positions(xps)
            print("\n--- Positions ---")
            for name, pos in zip(out['stage_names'], out['positions']):
                 print(f"  {name}: {pos} mm")

        elif action == 'get_limits':
            out['stage_names'], out['limits'] = get_limits(xps)
            print("\n--- Hardware Limits ---")
            for name, lim in zip(out['stage_names'], out['limits']):
                 print(f"  {name}: Min={lim[0]} mm, Max={lim[1]} mm")

        elif action == 'disconnect':
            print("Disconnecting..."); xps.disconnect(); xps = None

    except Exception as e:
        print(f"Error: {e}")
        if xps and __name__ == "__main__":
            try: xps.disconnect()
            except Exception:
                LOGGER.warning('XPS disconnect failed during error recovery', exc_info=True)
        raise

    out['xps'] = xps
    return out

if __name__ == "__main__":
    # Check if running from MATLAB/environment with injected variables
    if 'action' in globals():
        # Collect known arguments from global scope
        kwargs = {k: globals()[k] for k in ['xps', 'action', 'group', 'distance', 'position', 'velocity', 'acceleration', 'ip', 'user', 'pwd'] if k in globals()}
        out = main(**kwargs)
    else:
        # Standard CLI execution
        main()
