"""Fixed candidate comparison; select on 2025 validation before scoring January.

Install .[train,benchmark]. Uses two CPU threads and preserves the deployed model.
No extra weather fields, actual future weather, or lagged power are used.
"""
import argparse
import copy
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import time

os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('MKL_NUM_THREADS', '2')
import numpy as np
import pandas as pd
import torch
from torch import nn
import catboost
from catboost import CatBoostRegressor

from windpower.features import BASE_FEATURES, make_features
from windpower.model import Predictor
from windpower.train import split_masks, metrics


CANDIDATES = [
    dict(name='mlp_mse_cpu_control', family='mlp', loss='mse', widths=[64, 32], seed=42),
    dict(name='mlp_mae', family='mlp', loss='mae', widths=[64, 32], seed=42),
    dict(name='mlp_huber', family='mlp', loss='huber', widths=[64, 32], seed=42),
    dict(name='mlp_mae_larger', family='mlp', loss='mae', widths=[128, 64], seed=42),
    dict(name='mlp_mae_seed43', family='mlp', loss='mae', widths=[64, 32], seed=43),
    dict(name='catboost_rmse', family='catboost', loss='RMSE', depth=6, separate=False, seed=42),
    dict(name='catboost_mae', family='catboost', loss='MAE', depth=6, separate=False, seed=42),
    dict(name='catboost_mae_depth8', family='catboost', loss='MAE', depth=8, separate=False, seed=42),
    dict(name='catboost_mae_separate', family='catboost', loss='MAE', depth=6, separate=True, seed=42),
]


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def fit_mlp(spec, x_train, y_train, x_val, y_val, output, metadata, turbines):
    torch.manual_seed(spec['seed'])
    np.random.seed(spec['seed'])
    mean, scale = x_train.mean(axis=0), x_train.std(axis=0)
    scale = np.where(scale < 1e-6, 1, scale)
    xt, xv = [torch.tensor((x - mean) / scale) for x in (x_train, x_val)]
    yt, yv = [torch.tensor(y, dtype=torch.float32).reshape(-1, 1) for y in (y_train, y_val)]
    layers = []
    previous = x_train.shape[1]
    for width in spec['widths']:
        layers += [nn.Linear(previous, width), nn.ReLU()]
        previous = width
    model = nn.Sequential(*layers, nn.Linear(previous, 1), nn.Sigmoid())
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=0.001)
    loss_fn = {'mse': nn.MSELoss(), 'mae': nn.L1Loss(), 'huber': nn.SmoothL1Loss(beta=0.1)}[spec['loss']]
    best, stale, state, best_epoch = float('inf'), 0, None, 0
    history = []
    for epoch in range(1, 81):
        model.train()
        for ids in torch.randperm(len(xt)).split(512):
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(xt[ids]), yt[ids])
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.inference_mode():
            score = (model(xv) - yv).abs().mean().item()
        history.append(dict(epoch=epoch, validation_mae=score))
        if score < best - 1e-6:
            best, stale, best_epoch, state = score, 0, epoch, copy.deepcopy(model.state_dict())
        else:
            stale += 1
        if epoch % 10 == 0:
            print(json.dumps(dict(candidate=spec['name'], epoch=epoch, best_validation_mae=best)), flush=True)
        if stale >= 10:
            break
    model.load_state_dict(state)
    exported = [dict(weight=layer.weight.detach().tolist(), bias=layer.bias.detach().tolist()) for layer in model if isinstance(layer, nn.Linear)]
    version = 'mlp-' + hashlib.sha256(json.dumps(exported, sort_keys=True).encode()).hexdigest()[:12]
    bundle = dict(schema_version=1, architecture='mlp_relu_sigmoid', model_version=version,
        created_at_utc=dt.datetime.now(dt.timezone.utc).isoformat(), weather_model=metadata['weather_model'],
        turbines=turbines, feature_names=BASE_FEATURES + ['turbine:' + t for t in turbines],
        mean=mean.tolist(), scale=scale.tolist(), layers=exported, dataset_metadata=metadata,
        training_seed=spec['seed'], best_epoch=best_epoch, experiment=spec)
    write_json(output / 'model.json', bundle)
    with torch.inference_mode():
        pred = model(xv).numpy()[:, 0]
    return dict(best_epoch=best_epoch, epochs_run=len(history), history=history), pred


def fit_catboost(spec, x_train, y_train, x_val, y_val, train_ids, val_ids, output, turbines):
    prediction = np.zeros(len(x_val))
    details = {}
    for turbine in turbines if spec['separate'] else ['shared']:
        tm = train_ids == turbine if turbine != 'shared' else np.ones(len(y_train), dtype=bool)
        vm = val_ids == turbine if turbine != 'shared' else np.ones(len(y_val), dtype=bool)
        model = CatBoostRegressor(iterations=1000, depth=spec['depth'], learning_rate=0.05,
            loss_function=spec['loss'], eval_metric='MAE', l2_leaf_reg=3, random_seed=spec['seed'],
            thread_count=2, allow_writing_files=False, verbose=False, task_type='CPU')
        model.fit(x_train[tm], y_train[tm], eval_set=(x_val[vm], y_val[vm]),
            early_stopping_rounds=80, use_best_model=True)
        model.save_model(str(output / (turbine + '.cbm')))
        prediction[vm] = np.clip(model.predict(x_val[vm]), 0, 1)
        details[turbine] = dict(best_iteration=model.get_best_iteration(), tree_count=model.tree_count_)
    return details, prediction


def evaluate_saved(spec, directory, frame, x, turbines, weather_model):
    if spec['family'] == 'mlp':
        return Predictor(directory / 'model.json', allow_provisional=True).predict(frame, weather_model)
    pred = np.zeros(len(frame))
    for turbine in turbines if spec['separate'] else ['shared']:
        selected = frame.turbine_id.eq(turbine).to_numpy() if turbine != 'shared' else np.ones(len(frame), dtype=bool)
        model = CatBoostRegressor()
        model.load_model(str(directory / (turbine + '.cbm')))
        pred[selected] = np.clip(model.predict(x[selected], thread_count=2), 0, 1)
    return pred


def paired_day_bootstrap(test, original, candidate):
    # Resample days jointly across turbines and offsets, not correlated rows.
    daily = pd.DataFrame(dict(day=pd.to_datetime(test.valid_time_utc, utc=True).dt.floor('D'),
        gain=np.abs(original - test.target_power.to_numpy()) - np.abs(candidate - test.target_power.to_numpy())))
    grouped = daily.groupby('day').gain.agg(['sum', 'count']).to_numpy()
    rng = np.random.default_rng(20260923)
    draws = rng.integers(len(grouped), size=(4000, len(grouped)))
    gains = grouped[draws, 0].sum(axis=1) / grouped[draws, 1].sum(axis=1)
    return dict(mean_mae_reduction=float(daily.gain.mean()),
        day_bootstrap_95_percent_interval=np.quantile(gains, [0.025, 0.975]).tolist(),
        sampled_days=len(grouped), caveat='Approximate within-January uncertainty; days may still be serially correlated. January was previously inspected.')


def main(args):
    torch.set_num_threads(2)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    dataset = Path(args.dataset)
    metadata = json.loads(dataset.with_suffix('.metadata.json').read_text())
    digest = hashlib.sha256(dataset.read_bytes()).hexdigest()
    if digest != metadata['dataset_sha256']:
        raise ValueError('Dataset checksum mismatch')
    frame = pd.read_csv(dataset)
    if frame.duplicated(['turbine_id', 'valid_time_utc', 'provider_offset_days']).any():
        raise ValueError('Duplicate examples')
    if set(frame.model) != {metadata['weather_model']}:
        raise ValueError('Weather provider mismatch')
    masks = split_masks(frame, '2025-10-01', '2026-01-01', '2026-02-01')
    train, val = [frame.loc[masks[s]].reset_index(drop=True) for s in ('train', 'validation')]
    turbines = sorted(train.turbine_id.unique())
    baseline = Predictor(args.baseline, allow_provisional=True)
    if baseline.bundle['dataset_metadata']['dataset_sha256'] != digest:
        raise ValueError('Baseline uses a different dataset')
    protocol = dict(created_at_utc=dt.datetime.now(dt.timezone.utc).isoformat(), dataset_sha256=digest,
        train_end_exclusive='2025-10-01', validation_start='2025-10-01', validation_end_exclusive='2026-01-01',
        test_start='2026-01-01', test_end_exclusive='2026-02-01', selection_metric='validation MAE',
        candidates=CANDIDATES, fixed_ensembles=['mean of the two small MAE networks', 'equal average of existing network and validation-best CatBoost'],
        validation_used_for='Early stopping and candidate selection, not final accuracy claim',
        test_note='Previously inspected January; excluded from fitting and selection in this experiment. No February labels.',
        threads=2, device='cpu', versions={'torch':torch.__version__,'catboost':catboost.__version__,'numpy':np.__version__,'pandas':pd.__version__})
    write_json(output / 'protocol.json', protocol)
    x_train, x_val = [make_features(part, turbines) for part in (train, val)]
    y_train, y_val = [part.target_power.to_numpy(dtype=np.float32) for part in (train, val)]
    val_preds = {'current_gpu_mlp': baseline.predict(val, metadata['weather_model'])}
    details = {'current_gpu_mlp': {'artifact':baseline.bundle['model_version']}}
    for spec in CANDIDATES:
        started = time.monotonic()
        directory = output / spec['name']; directory.mkdir()
        print('Starting ' + spec['name'], flush=True)
        if spec['family'] == 'mlp':
            info, predicted = fit_mlp(spec, x_train, y_train, x_val, y_val, directory, metadata, turbines)
        else:
            info, predicted = fit_catboost(spec, x_train, y_train, x_val, y_val,
                train.turbine_id.to_numpy(), val.turbine_id.to_numpy(), directory, turbines)
        # Reload saved artifacts to verify exports before comparing candidates.
        reloaded = evaluate_saved(spec, directory, val, x_val, turbines, metadata['weather_model'])
        np.testing.assert_allclose(reloaded, predicted, atol=2e-6, rtol=1e-5)
        val_preds[spec['name']] = reloaded
        details[spec['name']] = dict(fit=info, duration_seconds=time.monotonic()-started)
        write_json(directory / 'training.json', details[spec['name']])
        np.save(directory / 'validation_predictions.npy', reloaded)
        print(json.dumps(dict(candidate=spec['name'], validation=metrics(y_val, reloaded), seconds=details[spec['name']]['duration_seconds'])), flush=True)
    best_tree = min([s['name'] for s in CANDIDATES if s['family'] == 'catboost'], key=lambda name:metrics(y_val,val_preds[name])['mae'])
    ensemble_members = {'mlp_mae_ensemble':['mlp_mae','mlp_mae_seed43'], 'mlp_tree_ensemble':['current_gpu_mlp',best_tree]}
    for name, members in ensemble_members.items():
        val_preds[name] = np.mean([val_preds[m] for m in members], axis=0)
        details[name] = dict(members=members, weights=[0.5,0.5])
    validation = {name:metrics(y_val,pred) for name,pred in val_preds.items()}
    selected = min(validation, key=lambda name:validation[name]['mae'])
    # Persist the decision before reading test targets or generating test scores.
    write_json(output / 'selection.json', dict(selected=selected, validation=validation, ensembles=ensemble_members,
        selected_at_utc=dt.datetime.now(dt.timezone.utc).isoformat()))
    print('Validation selection locked: ' + selected, flush=True)
    test = frame.loc[masks['test']].reset_index(drop=True)
    x_test = make_features(test, turbines)
    y_test = test.target_power.to_numpy()
    test_preds = {'current_gpu_mlp':baseline.predict(test, metadata['weather_model'])}
    for spec in CANDIDATES:
        test_preds[spec['name']] = evaluate_saved(spec, output/spec['name'], test, x_test, turbines, metadata['weather_model'])
    for name, members in ensemble_members.items():
        test_preds[name] = np.mean([test_preds[m] for m in members], axis=0)
    result = dict(protocol=protocol, selected_by_validation=selected, candidates={}, selected_vs_current=paired_day_bootstrap(test,test_preds['current_gpu_mlp'],test_preds[selected]))
    for name,pred in test_preds.items():
        if not np.isfinite(pred).all() or ((pred<0)|(pred>1)).any():
            raise ValueError('Invalid prediction range')
        result['candidates'][name] = dict(validation=validation[name], january=metrics(y_test,pred),
            by_turbine={t:metrics(y_test[test.turbine_id.eq(t)],pred[test.turbine_id.eq(t)]) for t in turbines},
            by_offset={str(o):metrics(y_test[test.provider_offset_days.eq(o)],pred[test.provider_offset_days.eq(o)]) for o in (1,2)},
            training=details[name])
    predictions = test[['turbine_id','valid_time_utc','provider_offset_days','target_power']].copy()
    for name,pred in test_preds.items(): predictions[name]=pred
    predictions.to_csv(output/'january_predictions.csv',index=False)
    write_json(output/'comparison.json',result)
    table = pd.DataFrame([dict(candidate=name,validation_mae=r['validation']['mae'],january_mae=r['january']['mae'],january_rmse=r['january']['rmse'],january_bias=r['january']['bias'],turbine_1_mae=r['by_turbine']['turbine_1']['mae'],turbine_2_mae=r['by_turbine']['turbine_2']['mae']) for name,r in result['candidates'].items()])
    table.to_csv(output/'comparison.csv',index=False)
    print(table.to_string(index=False),flush=True)
    print(json.dumps(result['selected_vs_current']),flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',required=True)
    p.add_argument('--baseline',required=True)
    p.add_argument('--output',required=True)
    main(p.parse_args())
