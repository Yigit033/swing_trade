"""Deterministic daily-bar accounting. No I/O, settings, providers or clock.

Daily OHLC is usable from a known session-open entry only. Intraday entries
and intraday policy edits need finer data; never attribute pre-entry extremes
to the position. Stop-first is a conservative convention, not observed order.
"""
import copy
import hashlib
import json
import math
from datetime import datetime, time, timezone
from zoneinfo import ZoneInfo

import pandas as pd

NY = ZoneInfo("America/New_York")
VERSION = 1
TRAIL_ATR_MULT = 3.0
TRAIL_ACTIVATE_ATR = 1.5


def timestamp(value):
    text = str(value)
    if len(text) <= 10:
        raise ValueError("ENTRY_TIME_UNKNOWN")
    dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
    return (dt.replace(tzinfo=NY) if dt.tzinfo is None else dt).astimezone(NY)


def event_time(day, at_open=False):
    return datetime.combine(day, time(9, 30) if at_open else time(16), NY).astimezone(
        timezone.utc).isoformat().replace("+00:00", "Z")


def mark_pnl(trade, price):
    """Total marked profit = realized partial + remaining inventory profit."""
    size = int(trade['position_size'])
    sold = int(size * (float(trade.get('partial_exit_pct') or 0) / 100))
    state = json.loads(trade['execution_state']) if trade.get('execution_state') else None
    if state:
        sold = state['sold']  # executed integer quantity, not re-rounded settings
    if not 0 <= sold <= size:
        raise ValueError("INVALID_PARTIAL_QUANTITY")
    entry = float(trade['entry_price'])
    partial = float(trade.get('partial_exit_price') or entry)
    if state and state['partial']:
        partial = state['partial_price']
    pnl = sold * (partial - entry) + (size - sold) * (float(price) - entry)
    return pnl, 100 * pnl / (size * entry) if size * entry else 0


def evaluate(trade, history, fraction):
    """Return an atomic update plus tuple-compatible exit result.

    Persisted state is a checkpoint, including processed OHLC fingerprints.
    Revised or incomplete history is quarantined rather than changing fills.
    Legacy closed rows and partial rows without a checkpoint are never replayed.
    """
    def review(reason):
        return {}, ('OPEN', 0., '', 'REVIEW_REQUIRED: ' + reason,
                    trade.get('trailing_stop') or trade['stop_loss'])

    if trade.get('status', 'OPEN') not in ('OPEN',):
        return {}, (trade.get('status'), trade.get('exit_price') or 0,
                    trade.get('exit_date') or '', '', trade.get('trailing_stop') or trade['stop_loss'])
    try:
        entry_dt = timestamp(trade['entry_date'])
        if entry_dt.time().replace(tzinfo=None) != time(9, 30):
            return review('INTRADAY_ENTRY_REQUIRES_INTRADAY_BARS')
        persisted = trade.get('execution_state')
        state = json.loads(persisted) if persisted else None
        if state and state.get('version') != VERSION:
            return review('UNKNOWN_STATE_VERSION')
        if not state:
            if trade.get('partial_exit_price'):
                return review('LEGACY_PARTIAL_TIME_OR_ORIGINAL_RISK_UNKNOWN')
            initial = float(trade.get('initial_stop') or trade['stop_loss'])
            if not 0 < initial < float(trade['entry_price']):
                return review('ORIGINAL_STOP_UNKNOWN')
            if trade['stop_loss'] != initial and not trade.get('policy_effective_at'):
                return review('LEGACY_STOP_EDIT_TIME_UNKNOWN')
            state = dict(version=VERSION, processed={}, events=[], stop=initial,
                         peak=float(trade['entry_price']), partial=False, sold=0,
                         partial_price=0., partial_date=None, fraction=float(fraction),
                         target=float(trade['target']), target_2=float(trade.get('target_2') or trade['target']),
                         hold=int(trade.get('max_hold_days', 7)), bars=0,
                         identity=[trade['entry_date'], float(trade['entry_price']),
                                   int(trade['position_size']), initial, float(trade.get('atr') or 0)])
        expected = [trade['entry_date'], float(trade['entry_price']), int(trade['position_size']),
                    float(trade.get('initial_stop') or trade['stop_loss']), float(trade.get('atr') or 0)]
        if state['identity'] != expected:
            return review('ENTRY_OR_RISK_CHANGED')
        if not 0 < state['fraction'] <= 1 or expected[2] < 1:
            return review('INVALID_QUANTITY_OR_FRACTION')
        frame = history.copy()
        frame['Date'] = pd.to_datetime(frame['Date']).dt.date
        frame = frame[frame.Date >= entry_dt.date()]
        if frame.empty:
            return review('ENTRY_BAR_MISSING')
        if frame.Date.duplicated().any() or not frame.Date.is_monotonic_increasing:
            return review('DUPLICATE_OR_UNSORTED_BARS')
        from ..utils.market_calendar import next_trading_day
        expected_day = entry_dt.date()
        fingerprints = {}
        for _, row in frame.iterrows():
            if row.Date != expected_day:
                return review('MISSING_SESSION')
            expected_day = next_trading_day(row.Date)
            values = [float(row[c]) for c in ('Open', 'High', 'Low', 'Close')]
            if not all(math.isfinite(v) and v > 0 for v in values):
                return review('INVALID_OHLC')
            o, h, low, cl = values
            if not low <= min(o, cl) <= max(o, cl) <= h:
                return review('INVALID_OHLC_ENVELOPE')
            fingerprints[row.Date.isoformat()] = hashlib.sha256(json.dumps(values).encode()).hexdigest()
        if any(fingerprints.get(d) != h for d, h in state['processed'].items()):
            return review('PROCESSED_HISTORY_CHANGED_OR_MISSING')
    except (ValueError, TypeError, KeyError):
        return review('INVALID_OR_UNKNOWN_EXECUTION_INPUT')

    state = copy.deepcopy(state)
    entry, size, initial, atr = expected[1:]
    status, exit_price, exit_date, reason = 'OPEN', 0., '', ''
    policy_at = trade.get('policy_effective_at')
    for _, row in frame.iterrows():
        day = row.Date
        key = day.isoformat()
        if key in state['processed']:
            continue
        o, high, low, close = (float(row[c]) for c in ('Open', 'High', 'Low', 'Close'))
        if policy_at and state.get('policy_applied') != policy_at:
            effective = timestamp(policy_at)
            opening = timestamp(event_time(day, True))
            closing = timestamp(event_time(day))
            if opening < effective < closing:
                return review('INTRADAY_POLICY_EDIT_REQUIRES_INTRADAY_BARS')
            if effective <= opening:
                state['stop'] = max(state['stop'], float(trade['stop_loss']))
                state['manual_stop'] = float(trade['stop_loss'])
                state['target'] = float(trade['target'])
                state['hold'] = int(trade['max_hold_days'])
                state['policy_applied'] = policy_at
        state['bars'] += 1
        state['processed'][key] = fingerprints[key]
        stop = state['stop']
        # Prior-bar stop has priority over EVERY target on an ambiguous bar.
        if low <= stop:
            status = 'TRAILED' if stop > initial and stop != state.get('manual_stop') else 'STOPPED'
            exit_price = round(min(o, stop), 2)
            exit_date = event_time(day, o <= stop)
            reason = ('Trailing stop; ' if status == 'TRAILED' else 'Stop; ') + 'stop-first daily OHLC convention; gap filled at Open'
        else:
            if not state['partial'] and high >= state['target']:
                if state['target_2'] > state['target'] * 1.01:
                    # Integer shares: at least one executable share, capped at
                    # the inventory (also covers a one-share full T1 exit).
                    state['sold'] = min(size, max(1, int(size * state['fraction'])))
                    state['partial'] = True
                    state['partial_price'] = state['target']
                    state['partial_date'] = event_time(day, o >= state['target'])
                    state['events'].append(dict(kind='T1', date=state['partial_date'],
                                                shares=state['sold'], price=state['target']))
                    state['stop'] = max(state['stop'], entry)
                    state['target'] = state['target_2']
                    if state['sold'] == size:
                        status, exit_price = 'TARGET', state['partial_price']
                    # Unknown low-vs-T1 order: conservative BE before T2.
                    elif low <= state['stop']:
                        status, exit_price = 'TRAILED', state['stop']
                        reason = 'Ambiguous T1/BE bar: conservative BE-first, not observed sequence'
                else:
                    status, exit_price = 'TARGET', state['target']
            if status == 'OPEN' and state['partial'] and high >= state['target']:
                status, exit_price = 'TARGET', state['target']
            state['peak'] = max(state['peak'], high)
            if status == 'OPEN' and atr > 0 and (state['peak'] - entry) / atr >= TRAIL_ACTIVATE_ATR:
                state['stop'] = max(state['stop'], round(state['peak'] - TRAIL_ATR_MULT * atr, 2))
            if status == 'OPEN' and state['bars'] >= state['hold']:
                status, exit_price = 'TIMEOUT', close
            if status != 'OPEN':
                exit_date = event_time(day)
                reason = reason or 'Daily OHLC target/timeout proxy; intraday time unknown'
        if status != 'OPEN':
            state['events'].append(dict(kind=status, date=exit_date, shares=size-state['sold'],
                                        price=exit_price))
            break

    updates = dict(execution_state=json.dumps(state, sort_keys=True), trailing_stop=state['stop'])
    marked = dict(trade)
    if state['partial']:
        updates.update(partial_exit_price=state['partial_price'], partial_exit_pct=state['fraction'] * 100,
                       partial_exit_date=state['partial_date'], stop_loss=max(entry, trade['stop_loss']),
                       target=state['target'])
        marked.update(updates)
    price = exit_price if status != 'OPEN' else float(frame.Close.iloc[-1])
    pnl, pct = mark_pnl(marked, price)
    updates.update(current_price=round(price, 4), unrealized_pnl=round(pnl, 2),
                   unrealized_pnl_pct=round(pct, 2))
    if status != 'OPEN':
        risk = trade.get('initial_risk')
        updates.update(status=status, exit_price=exit_price, exit_date=exit_date, notes=reason,
                       realized_pnl=round(pnl, 2), realized_pnl_pct=round(pct, 2),
                       r_multiple=pnl / risk if risk and risk > 0 else None,
                       unrealized_pnl=0., unrealized_pnl_pct=0.)
    return updates, (status, exit_price, exit_date, reason, state['stop'])
