const FIELD_DEFINITIONS = [
  {
    id: "age",
    label: "Age",
    placeholder: "e.g. 54",
    helper: "Age in years",
  },
  {
    id: "sex",
    label: "Sex",
    placeholder: "0 = male, 1 = female",
    helper: "Encoded as 0 or 1",
  },
  {
    id: "chest_pain",
    label: "Chest Pain Type",
    placeholder: "0–3",
    helper: "Ordinal encoding of chest pain type",
  },
  {
    id: "blood_pressure",
    label: "Resting Blood Pressure",
    placeholder: "e.g. 130",
    helper: "Resting BP (mm Hg)",
  },
  {
    id: "cholesterol",
    label: "Cholesterol",
    placeholder: "e.g. 245",
    helper: "Serum cholestoral (mg/dl)",
  },
  {
    id: "max_hr",
    label: "Max Heart Rate",
    placeholder: "e.g. 160",
    helper: "Maximum heart rate achieved",
  },
  {
    id: "st_depression",
    label: "ST Depression",
    placeholder: "e.g. 1.4",
    helper: "ST depression induced by exercise",
  },
];

/**
 * Render the structured feature input form using semantic labels and helpers.
 */
function buildForm() {
  const form = document.getElementById("form");
  const grid = document.createElement("div");
  grid.className = "grid-two-col";

  FIELD_DEFINITIONS.forEach((field) => {
    const wrapper = document.createElement("div");
    wrapper.className = "field";

    const label = document.createElement("label");
    label.setAttribute("for", field.id);
    label.textContent = field.label;

    const small = document.createElement("small");
    small.textContent = field.helper;

    const input = document.createElement("input");
    input.id = field.id;
    input.type = "number";
    input.placeholder = field.placeholder;
    input.inputMode = "decimal";
    input.step = "any";

    wrapper.appendChild(label);
    wrapper.appendChild(input);
    wrapper.appendChild(small);
    grid.appendChild(wrapper);
  });

  form.appendChild(grid);
}

/**
 * Read and validate feature values from the form.
 */
function collectPayload() {
  const payload = {};
  let hasMissing = false;

  FIELD_DEFINITIONS.forEach((field) => {
    const input = document.getElementById(field.id);
    const rawValue = input.value.trim();
    if (rawValue === "") {
      hasMissing = true;
    }
    payload[field.id] = rawValue === "" ? null : Number(rawValue);
  });

  return { payload, hasMissing };
}

/**
 * Display an error banner with the given message.
 */
function showError(message) {
  const banner = document.getElementById("error-banner");
  banner.textContent = "";
  banner.className = "error-banner error-banner--visible";

  const strong = document.createElement("strong");
  strong.textContent = "Error";

  const span = document.createElement("span");
  span.textContent = " · ";

  const text = document.createElement("span");
  text.textContent = message;

  banner.appendChild(strong);
  banner.appendChild(span);
  banner.appendChild(text);
}

/**
 * Clear any error banner.
 */
function clearError() {
  const banner = document.getElementById("error-banner");
  banner.className = "error-banner";
  banner.textContent = "";
}

/**
 * Render the prediction result card based on the API response.
 */
function renderResult(result) {
  const container = document.getElementById("result");
  container.className = "result-card result-card--visible";

  const probabilityPercent = (result.probability * 100).toFixed(2);
  const isPositive = result.prediction === 1;

  const riskLevel = result.risk_level || "Unknown";
  const riskKey = riskLevel.toLowerCase();

  let pillClass = "result-pill";
  if (riskKey.includes("low")) {
    pillClass += " result-pill--low";
  } else if (riskKey.includes("moderate")) {
    pillClass += " result-pill--moderate";
  } else if (riskKey.includes("high")) {
    pillClass += " result-pill--high";
  }

  const statusText = isPositive ? "Indicative of heart disease" : "No strong indication of heart disease";

  container.innerHTML = `
    <div class="result-card__header">
      <div>
        <div class="result-card__title">${statusText}</div>
        <div class="result-card__sub">Risk classification: ${riskLevel}</div>
      </div>
      <div class="${pillClass}">
        ${riskLevel}
      </div>
    </div>
    <div class="result-card__value-row">
      <div class="result-card__probability">${probabilityPercent}%</div>
      <div class="result-card__sub">Estimated probability of heart disease (positive class).</div>
    </div>
    <div class="result-card__meta">
      <span>Interpretation tip:</span> this probability comes from a logistic regression model trained and
      calibrated on the curated heart‑disease dataset. Use it as decision support alongside clinical judgement.
    </div>
  `;
}

/**
 * Perform the prediction request and update the UI.
 */
async function submitForm() {
  clearError();

  const { payload, hasMissing } = collectPayload();
  if (hasMissing) {
    showError("Please provide a value for each feature before running the analysis.");
    return;
  }

  try {
    const response = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    if (!response.ok) {
      const errorText = await response.text();
      showError(`The server responded with status ${response.status}. Details: ${errorText}`);
      return;
    }

    const result = await response.json();
    renderResult(result);
  } catch (error) {
    showError("Unable to reach the prediction API. Please confirm that the backend is running.");
  }
}

/**
 * Wire up initial UI behaviour.
 */
function initialize() {
  buildForm();
  const button = document.getElementById("analyze-button");
  if (button) {
    button.addEventListener("click", submitForm);
  }
}

document.addEventListener("DOMContentLoaded", initialize);
