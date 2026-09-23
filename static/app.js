'use strict';
const $ = (selector) => document.querySelector(selector);
let activePanel = 'forecast';
let currentRun = null;
let currentWeather = [];
let turbines = [];
let busy = false;
const fmt = (value, digits = 1) => Number(value).toLocaleString('ru-RU', {maximumFractionDigits: digits});
const timeLabel = (value) => new Date(value).toLocaleString('ru-RU', {timeZone: 'UTC', day: '2-digit', month: '2-digit', hour: '2-digit', minute: '2-digit'});

function notice(message = '', error = false) {
  $('#notice').textContent = message;
  $('#notice').hidden = !message;
  $('#notice').classList.toggle('error', error);
}

async function api(path, options = {}) {
  const csrf = document.querySelector('[name=csrfmiddlewaretoken]').value;
  let response;
  try {
    response = await fetch(path, {credentials: 'same-origin', ...options, headers: {'Content-Type': 'application/json', 'X-CSRFToken': csrf, ...options.headers}});
  } catch (_) { throw new Error('Не удалось соединиться с сервером. Проверьте подключение и повторите запрос.'); }
  let data;
  try { data = await response.json(); } catch (_) { throw new Error(`Сервер вернул ошибку ${response.status}. Обновите страницу и повторите запрос.`); }
  if (!response.ok) {
    if (data.id) renderRun(data);
    throw new Error(data.error?.message || `Ошибка ${response.status}`);
  }
  return data;
}

function link(selector, href) {
  const element = $(selector);
  if (href) element.href = href; else element.removeAttribute('href');
  element.classList.toggle('disabled', !href);
  element.setAttribute('aria-disabled', String(!href));
}

function switchPanel(name) {
  activePanel = name;
  document.querySelectorAll('.tab').forEach(button => {
    button.classList.toggle('active', button.dataset.panel === name);
    if (button.dataset.panel === name) button.setAttribute('aria-current', 'page'); else button.removeAttribute('aria-current');
  });
  $('#workspace').hidden = !['forecast', 'weather'].includes(name);
  ['forecast', 'weather', 'history', 'integration'].forEach(panel => $(`#${panel}-panel`).hidden = panel !== name);
  $('#submit').textContent = name === 'weather' ? 'Получить погоду ↗' : 'Построить прогноз ↗';
  if (name === 'forecast' && currentRun?.status === 'completed') {drawChart(currentRun.records); updateHour();}
  if (name === 'history') loadHistory();
}

function setBusy(value) {
  busy = value;
  $('#submit').disabled = value;
  $('#refresh').disabled = value || !currentRun;
  $('#forecast-form').setAttribute('aria-busy', String(value));
  if (value) $('#submit').textContent = 'Получаем данные…';
  else $('#submit').textContent = activePanel === 'weather' ? 'Получить погоду ↗' : 'Построить прогноз ↗';
}

function requestData() {
  return {turbine_id: $('#turbine').value, as_of: $('#as-of').value + ':00Z',
    start_time: $('#start-time').value + ':00Z', provider: $('#provider').value, weather_model: $('#weather-model').value.trim()};
}

function renderWeather(records, provenance = {}, snapshotId) {
  currentWeather = records;
  $('#weather-meta').textContent = provenance.notice || 'Данные проверены. Время указано в UTC.';
  const fragment = document.createDocumentFragment();
  records.forEach(row => {
    const tr = document.createElement('tr');
    [timeLabel(row.target_time), fmt(row.wind_speed), fmt(row.wind_direction), fmt(row.temperature), fmt(row.pressure), fmt(row.forecast_age_hours)].forEach(value => {
      const td = document.createElement('td'); td.textContent = value; tr.append(td);
    });
    fragment.append(tr);
  });
  $('#weather-rows').replaceChildren(fragment);
  link('#weather-json', snapshotId ? `/api/v1/weather/snapshots/${snapshotId}/` : null);
}

function renderRun(run) {
  currentRun = run;
  if (run.request) {
    $('#turbine').value = run.request.turbine_id;
    $('#as-of').value = run.request.as_of.slice(0, 16);
    $('#start-time').value = run.request.start_time.slice(0, 16);
    $('#provider').value = run.request.provider;
    $('#weather-model').value = run.request.weather_model;
    updateSourceNote();
    $('#turbine').dispatchEvent(new Event('change'));
  }
  $('#run-id').textContent = run.id.slice(0, 8);
  const completed = run.status === 'completed';
  $('#result-badge').textContent = !completed ? 'Ошибка расчёта' : run.is_demo ? 'Демонстрационный результат' : 'Прогноз готов';
  $('#result-badge').classList.toggle('demo', run.is_demo);
  $('#mean').textContent = completed ? fmt(run.analysis.mean_normalized_power * 100) + '%' : '—';
  $('#peak').textContent = completed ? fmt(run.analysis.peak_normalized_power * 100) + '%' : '—';
  $('#wind').textContent = run.weather.length ? fmt(run.weather.reduce((sum, r) => sum + r.wind_speed, 0) / run.weather.length) + ' м/с' : '—';
  $('#chart-empty').hidden = completed;
  $('#chart-wrap').hidden = !completed;
  $('#period').textContent = completed ? `${timeLabel(run.records[0].target_time)} — ${timeLabel(run.records[47].target_time)} UTC` : 'Нет завершённого прогноза';
  link('#csv', completed ? `/api/v1/forecasts/${run.id}/export.csv` : null);
  link('#json', `/api/v1/forecasts/${run.id}/`);
  $('#refresh').disabled = busy;
  const fragment = document.createDocumentFragment();
  run.trace.forEach(entry => {
    const li = document.createElement('li');
    const text = document.createElement('span'); text.textContent = entry.message;
    const clock = document.createElement('time'); clock.textContent = `${fmt(entry.elapsed_ms / 1000, 2)} с`;
    li.append(text, clock); fragment.append(li);
  });
  $('#trace').replaceChildren(fragment);
  renderWeather(run.weather, run.provenance, run.snapshot_id);
  if (completed) {
    $('#hour').value = 0;
    drawChart(run.records);
    updateHour();
    notice(run.analysis.warnings.join(' '));
  }
}

function svgNode(tag, attrs, content) {
  const node = document.createElementNS('http://www.w3.org/2000/svg', tag);
  Object.entries(attrs).forEach(([key, value]) => node.setAttribute(key, value));
  if (content !== undefined) node.textContent = content;
  return node;
}

function drawChart(records) {
  const svg = $('#power-chart'); svg.replaceChildren();
  const width = Math.max(300, svg.clientWidth || 900);
  svg.setAttribute('viewBox', `0 0 ${width} 260`);
  const x = i => 42 + i * (width - 64) / 47, y = p => 211 - p * 180;
  [0, .25, .5, .75, 1].forEach(p => {
    svg.append(svgNode('line', {x1:42, x2:width-22, y1:y(p), y2:y(p), stroke:'#e5e9dc', 'stroke-dasharray':'3 5'}));
    svg.append(svgNode('text', {x:34, y:y(p)+4, 'text-anchor':'end', fill:'#878d7e', 'font-size':10}, String(p*100)));
  });
  const points = records.map((row, i) => `${x(i)},${y(row.predicted_normalized_power)}`).join(' ');
  svg.append(svgNode('polygon', {points:`42,211 ${points} ${width-22},211`, fill:'#dce7bd', opacity:'.55'}));
  svg.append(svgNode('polyline', {points, fill:'none', stroke:'#7b964b', 'stroke-width':2.5, 'stroke-linejoin':'round'}));
  (width < 600 ? [0, 24, 47] : [0, 8, 16, 24, 32, 40, 47]).forEach(i => svg.append(svgNode('text', {x:x(i), y:240, 'text-anchor':i === 0 ? 'start' : i === 47 ? 'end' : 'middle', fill:'#878d7e', 'font-size':10}, timeLabel(records[i].target_time))));
  svg.append(svgNode('line', {id:'chart-marker', x1:48, x2:48, y1:25, y2:212, stroke:'#5d6a45', 'stroke-dasharray':'4 4'}));
  svg.append(svgNode('circle', {id:'chart-point', cx:48, cy:y(records[0].predicted_normalized_power), r:4, fill:'#384b24', stroke:'white', 'stroke-width':2}));
}

function updateHour() {
  if (!currentRun || currentRun.status !== 'completed') return;
  const width = Math.max(300, $('#power-chart').clientWidth || 900);
  const i = Number($('#hour').value), row = currentRun.records[i], x = 42 + i * (width - 64) / 47;
  $('#hour-value').textContent = `${timeLabel(row.target_time)} UTC · ${fmt(row.predicted_normalized_power * 100)}%`;
  $('#chart-marker').setAttribute('x1', x); $('#chart-marker').setAttribute('x2', x);
  $('#chart-point').setAttribute('cx', x); $('#chart-point').setAttribute('cy', 211 - row.predicted_normalized_power * 180);
}

async function loadHistory() {
  try {
    const {forecasts} = await api('/api/v1/forecasts/');
    const fragment = document.createDocumentFragment();
    forecasts.forEach(run => {
      const tr = document.createElement('tr');
      [run.turbine_id, timeLabel(run.as_of), run.model_version || '—', run.status === 'completed' ? (run.is_demo ? 'Демо · готово' : 'Готово') : run.status === 'failed' ? 'Ошибка' : 'Выполняется'].forEach(value => {const td = document.createElement('td'); td.textContent = value; tr.append(td);});
      const td = document.createElement('td'), button = document.createElement('button'); button.className = 'secondary'; button.textContent = 'Открыть ↗';
      button.addEventListener('click', async () => {
        try { const result = await api(`/api/v1/forecasts/${run.id}/`); switchPanel('forecast'); renderRun(result); notice(result.error.message || result.analysis.warnings?.join(' ') || '', result.status === 'failed'); }
        catch (error) {notice(error.message, true);}
      });
      td.append(button); tr.append(td); fragment.append(tr);
    });
    if (!forecasts.length) {const tr = document.createElement('tr'), td = document.createElement('td');td.colSpan=5;td.className='empty-cell';td.textContent='Пока нет запусков. Создайте первый прогноз.';tr.append(td);fragment.append(tr);}
    $('#history-rows').replaceChildren(fragment);
  } catch (error) {notice(error.message, true);}
}

function updateSourceNote() {
  const source = $('#provider').value;
  const stub = document.body.dataset.mlBackend === 'demo' ? ' ML пока работает в демонстрационном режиме.' : '';
  $('#source-note').textContent = source === 'demo' ? 'Синтетическая погода для проверки интеграции. Не использовать для оценки точности.' + stub : source === 'archive' ? 'Точные импортированные выпуски: время публикации проверяется относительно as_of.' + stub : 'Архивные прогнозы Open-Meteo. Время выпуска оценочное, с запасом 8 часов на публикацию.' + stub;
  $('#weather-model').disabled = source === 'demo';
}

document.querySelectorAll('.tab').forEach(button => button.addEventListener('click', () => switchPanel(button.dataset.panel)));
$('#hour').addEventListener('input', updateHour);
window.addEventListener('resize', () => {
  if (currentRun?.status === 'completed' && activePanel === 'forecast') {drawChart(currentRun.records); updateHour();}
});
$('#provider').addEventListener('change', updateSourceNote);
$('#reload-history').addEventListener('click', loadHistory);
$('#turbine').addEventListener('change', () => {
  const turbine = turbines.find(t => t.id === $('#turbine').value);
  $('#coordinates').textContent = turbine?.latitude != null ? `${turbine.latitude.toFixed(6)} N · ${turbine.longitude.toFixed(6)} E` : 'Координаты не настроены';
});
$('#forecast-form').addEventListener('submit', async event => {
  event.preventDefault(); if (busy) return;
  setBusy(true); notice('Получаем погоду и проверяем данные…');
  const mode = activePanel;
  try {
    const data = requestData();
    const result = await api(mode === 'weather' ? '/api/v1/weather/forecast/' : '/api/v1/forecasts/', {method:'POST', body:JSON.stringify(data)});
    if (mode === 'weather') {renderWeather(result.records, result.provenance, result.snapshot_id);notice(result.provenance.notice);}
    else renderRun(result);
  } catch (error) {notice(error.message, true);}
  finally {setBusy(false);}
});
$('#refresh').addEventListener('click', async () => {
  if (!currentRun || busy) return;
  setBusy(true); notice('Проверяем изменения погодного снимка…');
  try {renderRun(await api(`/api/v1/forecasts/${currentRun.id}/refresh/`, {method:'POST', body:'{}'}));}
  catch (error) {notice(error.message, true);}
  finally {setBusy(false);}
});

(async () => {
  updateSourceNote();
  try {
    const [health, data] = await Promise.all([api('/api/v1/health/'), api('/api/v1/turbines/')]);
    $('#connection').textContent = health.ml_backend === 'demo' ? 'ML · демо-режим' : 'ML · внешний сервис';
    turbines = data.turbines;
    $('#turbine').replaceChildren(...turbines.map(t => new Option(t.name, t.id)));
    $('#turbine').dispatchEvent(new Event('change'));
    if (!turbines.length) notice('Сначала выполните python manage.py seed_demo, чтобы добавить турбины.', true);
  } catch (error) {$('#connection').textContent = 'Нет соединения';notice(error.message, true);}
})();
