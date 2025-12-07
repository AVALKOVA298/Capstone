// app.js
// Глобальные переменные
let model = null;
let config = null;
let edaData = null;

/
 * Инициализация после загрузки страницы
 */
document.addEventListener("DOMContentLoaded", async () => {
  setupTabs();
  setupPredictionHandler();

  // Загружаем конфиг и EDA
  try {
    config = await fetch("frontend_config.json").then(r => r.json());
    console.log("Config loaded:", config);

    edaData = await fetch("eda_data.json").then(r => r.json());
    console.log("EDA data loaded:", edaData);

    renderEdaSummary(edaData);
  } catch (err) {
    console.error("Error loading config or EDA data:", err);
    const edaDiv = document.getElementById("eda-summary");
    if (edaDiv) {
      edaDiv.innerHTML = `<p style="color:red;">Failed to load EDA/config data. Check console.</p>`;
    }
  }

  // Загружаем модель
  try {
    model = await tf.loadLayersModel("model.json");
    console.log("Model loaded");
  } catch (err) {
    console.error("Error loading model:", err);
    const resDiv = document.getElementById("prediction-result");
    if (resDiv) {
      resDiv.innerHTML = `<p style="color:red;">Failed to load model. Check console.</p>`;
    }
  }
});

/
 * Переключение вкладок (Prediction / EDA)
 */
function setupTabs() {
  const tabButtons = document.querySelectorAll(".tab-button");
  const tabContents = document.querySelectorAll(".tab-content");

  tabButtons.forEach(btn => {
    btn.addEventListener("click", () => {
      const tab = btn.dataset.tab;

      tabButtons.forEach(b => b.classList.remove("active"));
      tabContents.forEach(c => c.classList.remove("active"));

      btn.classList.add("active");
      document.getElementById(tab-${tab}).classList.add("active");
    });
  });
}

/
 * Настройка логики кнопки Predict
 */
function setupPredictionHandler() {
  const btn = document.getElementById("predict-btn");
  const textarea = document.getElementById("job-text");
  const resultDiv = document.getElementById("prediction-result");

  if (!btn || !textarea || !resultDiv) return;

  btn.addEventListener("click", async () => {
    if (!config || !model) {
      resultDiv.textContent = "Model or config is still loading. Please wait...";
      return;
    }

    const text = textarea.value.trim();
    if (!text) {
      resultDiv.textContent = "Please paste a job description.";
      return;
    }

    try {
      const input = preprocessText(text, config);
      const prediction = model.predict(input);
      const probArr = await prediction.data();
      const prob = probArr[0];

      prediction.dispose();
      input.dispose();

      const threshold = config.threshold || 0.5;
      const label = prob >= threshold ? "Fake / Risky" : "Real / Likely legitimate";

      resultDiv.innerHTML = `
        <p><b>Prediction:</b> ${label}</p>
        <p>Score: ${(prob * 100).toFixed(2)} % (threshold = ${(threshold * 100).toFixed(1)} %)</p>
      `;
    } catch (err) {
      console.error("Prediction error:", err);
      resultDiv.innerHTML = `<p style="color:red;">Prediction failed. Check console.</p>`;
    }
  });
}

/
 * Предобработка текста под BiLSTM-модель
 * @param {string} text
 * @param {Object} cfg - объект из frontend_config.json
 * @returns {tf.Tensor2D} [1, max_len]
 */
function preprocessText(text, cfg) {
  const maxLen = cfg.max_len;
  const oovIdx = cfg.oov_index;
  const wordIndex = cfg.word_index;

  // очень простая токенизация: нижний регистр + удаляем всё кроме букв/цифр/пробелов
  const tokens = text
    .toLowerCase()
    .replace(/[^a-z0-9\s]/g, " ")
    .split(/\s+/)
    .filter(t => t.length > 0);
  let sequence = tokens.map(w => wordIndex[w]  oovIdx);

  // усечение / паддинг
  if (sequence.length > maxLen) {
    sequence = sequence.slice(0, maxLen);
  } else if (sequence.length < maxLen) {
    const pad = new Array(maxLen - sequence.length).fill(0);
    sequence = sequence.concat(pad);
  }

  return tf.tensor2d([sequence], [1, maxLen]);
}

/**
 * Отображение EDA-результатов
 * @param {Object} eda
 */
function renderEdaSummary(eda) {
  const div = document.getElementById("eda-summary");
  if (!div) return;

  const classCounts = eda.class_counts  {};
  const lengths = eda.lengths  {};
  const metrics = eda.metrics  {};
  const cm = eda.confusion_matrix  {};
  const missing = eda.missing  {};

  const totalReal = classCounts.Real ?? 0;
  const totalFake = classCounts.Fake ?? 0;
  const total = totalReal + totalFake;

  const acc = metrics.accuracy
    ? (metrics.accuracy * 100).toFixed(2)
    : "–";

  // Метрики по классам, если есть
  const realMetrics = metrics.real  {};
  const fakeMetrics = metrics.fake  {};

  // Конфьюжн-матрица
  let cmHtml = "";
  if (cm.matrix && cm.labels_true && cm.labels_pred) {
    const m = cm.matrix;
    cmHtml += `
      <table class="cm-table">
        <thead>
          <tr>
            <th></th>
            <th>${cm.labels_pred[0]  "Pred 0"}</th>
            <th>${cm.labels_pred[1]  "Pred 1"}</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <th>${cm.labels_true[0]  "True 0"}</th>
            <td>${m[0][0]}</td>
            <td>${m[0][1]}</td>
          </tr>
          <tr>
            <th>${cm.labels_true[1]  "True 1"}</th>
            <td>${m[1][0]}</td>
            <td>${m[1][1]}</td>
          </tr>
        </tbody>
      </table>
    `;
  }

  // Пропуски по столбцам
  let missingHtml = "";
  if (missing.columns && missing.real && missing.fake) {
    missingHtml += `<table class="missing-table">
      <thead>
        <tr>
          <th>Column</th>
          <th>Missing (Real, %)</th>
          <th>Missing (Fake, %)</th>
        </tr>
      </thead>
      <tbody>
    `;

    for (let i = 0; i < missing.columns.length; i++) {
      const col = missing.columns[i];
      const mr = (missing.real[i] * 100).toFixed(1);
      const mf = (missing.fake[i] * 100).toFixed(1);
      missingHtml += `
        <tr>
          <td>${col}</td>
          <td>${mr}</td>
          <td>${mf}</td>
        </tr>
      `;
    }
    missingHtml += </tbody></table>;
  }

  div.innerHTML = `
    <h3>Class distribution</h3>
    <p>Total samples: <b>${total}</b></p>
    <p>Real: <b>${totalReal}</b> (${total ? ((totalReal / total) * 100).toFixed(1) : "–"} %),
       Fake: <b>${totalFake}</b> (${total ? ((totalFake / total) * 100).toFixed(1) : "–"} %)</p>

    <h3>Model metrics (validation/test)</h3>
    <p><b>Accuracy:</b> ${acc} % (threshold = ${(metrics.threshold * 100).toFixed(1)} %)</p>

    <details open>
      <summary><b>Per-class metrics</b></summary>
      <p><b>Real</b><br>
         Precision: ${(realMetrics.precision * 100).toFixed(2)} %<br>
         Recall: ${(realMetrics.recall * 100).toFixed(2)} %<br>
         F1: ${(realMetrics.f1 * 100).toFixed(2)} %<br>
         Support: ${realMetrics.support}
      </p>
      <p><b>Fake</b><br>
         Precision: ${(fakeMetrics.precision * 100).toFixed(2)} %<br>
         Recall: ${(fakeMetrics.recall * 100).toFixed(2)} %<br>
         F1: ${(fakeMetrics.f1 * 100).toFixed(2)} %<br>
         Support: ${fakeMetrics.support}
      </p>
    </details>

    <h3>Confusion matrix</h3>
    ${cmHtml  "<p>No confusion matrix data.</p>"}

    <h3>Missing values by column</h3>
    ${missingHtml  "<p>No missing data info.</p>"}
  `;
}
