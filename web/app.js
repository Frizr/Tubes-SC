const samplePayload = {
  age: 55,
  blood_pressure: 80,
  specific_gravity: 1.02,
  albumin: 0,
  sugar: 0,
  red_blood_cells: "normal",
  pus_cell: "normal",
  pus_cell_clumps: "notpresent",
  bacteria: "notpresent",
  blood_glucose_random: 120,
  blood_urea: 40,
  serum_creatinine: 1.2,
  sodium: 137,
  potassium: 4.5,
  hemoglobin: 14.5,
  packed_cell_volume: 45,
  white_blood_cell_count: 8000,
  red_blood_cell_count: 5.2,
  hypertension: "no",
  diabetes_mellitus: "no",
  coronary_artery_disease: "no",
  appetite: "good",
  pedal_edema: "no",
  anemia: "no",
};

const sections = [
  {
    title: "Vitals and Urine Chemistry",
    fields: [
      { key: "age", label: "Age", type: "number", min: 0, max: 120, step: 1 },
      { key: "blood_pressure", label: "Blood Pressure", type: "number", min: 30, max: 220, step: 1 },
      { key: "specific_gravity", label: "Specific Gravity", type: "number", min: 1, max: 1.04, step: 0.001 },
      { key: "albumin", label: "Albumin", type: "number", min: 0, max: 5, step: 1 },
      { key: "sugar", label: "Sugar", type: "number", min: 0, max: 5, step: 1 },
    ],
  },
  {
    title: "Urinalysis Findings",
    fields: [
      { key: "red_blood_cells", label: "Red Blood Cells", type: "select", options: ["normal", "abnormal"] },
      { key: "pus_cell", label: "Pus Cell", type: "select", options: ["normal", "abnormal"] },
      { key: "pus_cell_clumps", label: "Pus Cell Clumps", type: "select", options: ["notpresent", "present"] },
      { key: "bacteria", label: "Bacteria", type: "select", options: ["notpresent", "present"] },
    ],
  },
  {
    title: "Blood Chemistry",
    fields: [
      { key: "blood_glucose_random", label: "Blood Glucose Random", type: "number", min: 20, max: 600, step: 1 },
      { key: "blood_urea", label: "Blood Urea", type: "number", min: 1, max: 300, step: 0.1 },
      { key: "serum_creatinine", label: "Serum Creatinine", type: "number", min: 0.1, max: 30, step: 0.1 },
      { key: "sodium", label: "Sodium", type: "number", min: 90, max: 180, step: 0.1 },
      { key: "potassium", label: "Potassium", type: "number", min: 1, max: 10, step: 0.1 },
    ],
  },
  {
    title: "Hematology",
    fields: [
      { key: "hemoglobin", label: "Hemoglobin", type: "number", min: 3, max: 25, step: 0.1 },
      { key: "packed_cell_volume", label: "Packed Cell Volume", type: "number", min: 10, max: 70, step: 1 },
      { key: "white_blood_cell_count", label: "White Blood Cell Count", type: "number", min: 1000, max: 50000, step: 1 },
      { key: "red_blood_cell_count", label: "Red Blood Cell Count", type: "number", min: 1, max: 9, step: 0.1 },
    ],
  },
  {
    title: "Medical History",
    fields: [
      { key: "hypertension", label: "Hypertension", type: "select", options: ["no", "yes"] },
      { key: "diabetes_mellitus", label: "Diabetes Mellitus", type: "select", options: ["no", "yes"] },
      { key: "coronary_artery_disease", label: "Coronary Artery Disease", type: "select", options: ["no", "yes"] },
      { key: "appetite", label: "Appetite", type: "select", options: ["good", "poor"] },
      { key: "pedal_edema", label: "Pedal Edema", type: "select", options: ["no", "yes"] },
      { key: "anemia", label: "Anemia", type: "select", options: ["no", "yes"] },
    ],
  },
];

function createInput(field) {
  const wrapper = document.createElement("label");
  wrapper.className = "field";
  wrapper.htmlFor = field.key;

  const label = document.createElement("span");
  label.textContent = field.label;
  wrapper.appendChild(label);

  if (field.type === "select") {
    const select = document.createElement("select");
    select.name = field.key;
    select.id = field.key;
    select.required = true;
    field.options.forEach((optionValue) => {
      const option = document.createElement("option");
      option.value = optionValue;
      option.textContent = optionValue;
      select.appendChild(option);
    });
    wrapper.appendChild(select);
    return wrapper;
  }

  const input = document.createElement("input");
  input.type = "number";
  input.name = field.key;
  input.id = field.key;
  input.min = field.min;
  input.max = field.max;
  input.step = field.step;
  input.required = true;
  wrapper.appendChild(input);
  return wrapper;
}

function renderFields() {
  const container = document.querySelector("#field-sections");
  sections.forEach((section) => {
    const group = document.createElement("fieldset");
    group.className = "section-group";

    const legend = document.createElement("legend");
    legend.textContent = section.title;
    group.appendChild(legend);

    const grid = document.createElement("div");
    grid.className = "field-grid";
    section.fields.forEach((field) => grid.appendChild(createInput(field)));
    group.appendChild(grid);
    container.appendChild(group);
  });
}

function fillExample() {
  Object.entries(samplePayload).forEach(([key, value]) => {
    const control = document.querySelector(`[name="${key}"]`);
    if (control) {
      control.value = value;
    }
  });
}

function collectPayload(form) {
  const formData = new FormData(form);
  const payload = {};
  sections.flatMap((section) => section.fields).forEach((field) => {
    const value = formData.get(field.key);
    payload[field.key] = field.type === "number" ? Number(value) : value;
  });
  return payload;
}

function setStatus(ok) {
  const status = document.querySelector("#api-status");
  status.textContent = ok ? "API online" : "API unavailable";
  status.classList.toggle("ok", ok);
}

function renderResult(payload) {
  document.querySelector("#prediction-value").textContent = payload.prediction.toUpperCase();
  document.querySelector("#probability-value").textContent = `${Math.round(payload.probability * 100)}% model confidence`;
  document.querySelector("#disclaimer").textContent = payload.disclaimer;

  const list = document.querySelector("#feature-list");
  list.innerHTML = "";
  payload.top_features.forEach((item) => {
    const li = document.createElement("li");
    const score = Math.max(0, item.importance).toFixed(3);
    li.textContent = `${item.feature.replaceAll("_", " ")}: ${score}`;
    list.appendChild(li);
  });
}

function renderError(message) {
  document.querySelector("#prediction-value").textContent = "Request failed";
  document.querySelector("#probability-value").textContent = message;
  document.querySelector("#feature-list").innerHTML = "<li>No explanation available.</li>";
}

async function loadModelInfo() {
  try {
    const response = await fetch("/api/v1/model-info");
    if (!response.ok) throw new Error("Model info unavailable");
    const payload = await response.json();
    document.querySelector("#model-name").textContent = payload.model_name;
    document.querySelector("#feature-count").textContent = String(payload.feature_count);
  } catch (error) {
    document.querySelector("#model-name").textContent = "Unavailable";
    document.querySelector("#feature-count").textContent = "Unavailable";
  }
}

async function checkHealth() {
  try {
    const response = await fetch("/api/v1/health");
    setStatus(response.ok);
  } catch (error) {
    setStatus(false);
  }
}

async function submitScreening(event) {
  event.preventDefault();
  const payload = collectPayload(event.currentTarget);

  try {
    const response = await fetch("/api/v1/screen", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (!response.ok) {
      const detail = await response.json().catch(() => ({}));
      throw new Error(detail.detail ? "Validation error" : "Prediction service error");
    }
    renderResult(await response.json());
  } catch (error) {
    renderError(error.message);
  }
}

document.addEventListener("DOMContentLoaded", () => {
  renderFields();
  fillExample();
  checkHealth();
  loadModelInfo();

  document.querySelector("#example-button").addEventListener("click", fillExample);
  document.querySelector("#screening-form").addEventListener("submit", submitScreening);
  document.querySelector("#screening-form").addEventListener("reset", () => {
    document.querySelector("#prediction-value").textContent = "Awaiting input";
    document.querySelector("#probability-value").textContent = "Probability will appear after screening.";
    document.querySelector("#feature-list").innerHTML = "<li>No explanation available yet.</li>";
  });
});
