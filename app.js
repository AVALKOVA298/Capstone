// Path to TFJS model folder in your repo
const MODEL_URL = 'model.json';

// Max sequence length used in training (300 tokens)
const MAX_LEN = 300;

let model;
let tokenIndex = {}; // will be loaded from vocab JSON if you export it

const predictBtn = document.getElementById('predict-btn');
const form = document.getElementById('job-form');
const scoreCard = document.getElementById('score-card');
const scoreValueEl = document.getElementById('score-value');
const scoreLabelEl = document.getElementById('score-label');
const scoreTextEl = document.getElementById('score-text');

async function loadModel() {
  predictBtn.disabled = true;
  predictBtn.textContent = 'Loading model...';
  model = await tf.loadLayersModel(MODEL_URL);
  predictBtn.disabled = false;
  predictBtn.textContent = 'Get fraud score';
}

// Very simple tokenizer: lowercase + split on spaces.
// Для продакшна лучше выгрузить из TF TextVectorization словарь и
// собрать здесь такой же токенайзер.
function simpleTokenize(text) {
  return text
    .toLowerCase()
    .replace(/[^a-z0-9\s]/g, ' ')
    .split(/\s+/)
    .filter(t => t.length > 0);
}

// Паддинг/обрезка до MAX_LEN
function vectorize(text) {
  const tokens = simpleTokenize(text);
  const seq = new Array(MAX_LEN).fill(0);

  for (let i = 0; i < Math.min(tokens.length, MAX_LEN); i++) {
    const t = tokens[i];
    // если есть словарь tokenIndex — используем его, иначе простая хеш-индексация
    const idx = tokenIndex[t] || (Math.abs(hashString(t)) % 29999) + 1;
    seq[i] = idx;
  }
  return seq;
}

// Простейший детерминированный хеш
function hashString(str) {
  let h = 0;
  for (let i = 0; i < str.length; i++) {
    h = (h * 31 + str.charCodeAt(i)) | 0;
  }
  return h;
}

function buildFullText() {
  const title = document.getElementById('title').value || '';
  const company = document.getElementById('company').value || '';
  const description = document.getElementById('description').value || '';
  const requirements = document.getElementById('requirements').value || '';
  const benefits = document.getElementById('benefits').value || '';
  const location = document.getElementById('location').value || '';
  const salary = document.getElementById('salary').value || '';
  const employment = document.getElementById('employment').value || '';
  const industry = document.getElementById('industry').value || '';

  // тот же concat‑порядок, который был в ноутбуке (title, companyprofile, description, requirements, benefits, location, salary_range, employment_type, industry) [file:34]
  return [
    title,
    company,
    description,
    requirements,
    benefits,
    location,
    salary,
    employment,
    industry
  ].join(' ').trim();
}

function renderScore(prob) {
  const score = Number(prob.toFixed(3));
  scoreValueEl.textContent = score.toString();

  let label = 'Low risk';
  let cls = 'pill pill-safe';
  let explanation =
    'The model assigns a low probability of fraud for this job posting.';

  if (score >= 0.7) {
    label = 'High risk';
    cls = 'pill pill-risk';
    explanation =
      'The model assigns a high probability of fraud. Treat this job posting with caution.';
  } else if (score >= 0.4) {
    label = 'Medium risk';
    cls = 'pill pill-med';
    explanation =
      'The model sees mixed signals. Review this posting carefully before applying.';
  }

  scoreLabelEl.textContent = label;
  scoreLabelEl.className = cls;
  scoreTextEl.textContent =
    explanation +
    ' Score is the predicted probability that the posting is fraudulent (0–1).';

  scoreCard.style.display = 'block';
}

form.addEventListener('submit', async (e) => {
  e.preventDefault();
  if (!model) return;

  const fullText = buildFullText();
  if (!fullText) {
    alert('Please enter at least some job information.');
    return;
  }

  predictBtn.disabled = true;
  predictBtn.textContent = 'Scoring...';

  const seq = vectorize(fullText);
  const input = tf.tensor2d([seq], [1, MAX_LEN], 'int32');
  const pred = model.predict(input);
  const prob = (await pred.data())[0];
  tf.dispose([input, pred]);

  renderScore(prob);

  predictBtn.disabled = false;
  predictBtn.textContent = 'Get fraud score';
});

// Автозагрузка модели при открытии страницы
loadModel();
