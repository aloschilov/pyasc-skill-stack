const {
  layout: { roles, width, sections, positions },
  nodes,
  transfers,
} = JSON.parse(document.getElementById("workflow-data").textContent);
const by = (id) => document.getElementById(id),
  svg = document.querySelector("svg"),
  sheet = by("sheet"),
  nav = by("nav"),
  buttons = [];
let current = 0,
  selected;
for (const role of roles) {
  const b = document.createElement("b");
  b.textContent = role;
  by("heads").append(b);
}
function sectionOf(k) {
  return sections.findIndex(
    (s) => positions[k] >= s.top && positions[k] < s.bottom,
  );
}
function select(k) {
  const n = nodes[k];
  if (!n) return;
  selected?.classList.remove("chosen");
  selected = svg.querySelector('[data-element-id="' + k + '"]');
  selected?.classList.add("chosen");
  by("example").textContent = n.role + " · " + n.name + " — " + n.example;
}
function jump(k) {
  show(sectionOf(k));
  const s = sections[current],
    scale = svg.clientWidth / width;
  sheet.scrollTop = Math.max(
    0,
    (positions[k] - s.top) * scale - sheet.clientHeight / 3,
  );
  select(k);
}
function show(i) {
  current = i;
  const s = sections[i],
    h = s.bottom - s.top;
  svg.setAttribute("viewBox", "0 " + s.top + " " + width + " " + h);
  svg.setAttribute("width", width);
  svg.setAttribute("height", h);
  buttons.forEach((b, j) => b.setAttribute("aria-current", i === j));
  by("phase").textContent = s.name;
  sheet.scrollTop = 0;
  sheet.scrollLeft = 0;
  selected?.classList.remove("chosen");
  by("example").textContent =
    "Select a node for details. Blue arrows cross role boundaries.";
  const ts = transfers.filter((t) => sectionOf(t.a) === i);
  by("count").textContent = "Role handoffs from this phase (" + ts.length + ")";
  by("handoffs").replaceChildren();
  for (const t of ts) {
    const li = document.createElement("li");
    for (const [j, k] of [t.a, t.b].entries()) {
      if (j) li.append(" → ");
      const link = document.createElement("button");
      link.textContent = nodes[k].role + " · " + nodes[k].name;
      link.onclick = () => jump(k);
      li.append(link);
    }
    if (t.condition) li.append(" (" + t.condition + ")");
    by("handoffs").append(li);
  }
}
sections.forEach((s, i) => {
  const b = document.createElement("button");
  b.textContent = "Phase " + i;
  b.onclick = () => {
    show(i);
    history.replaceState(null, "", "#phase-" + i);
  };
  nav.insertBefore(b, nav.querySelector("label"));
  buttons.push(b);
});
for (const g of svg.querySelectorAll("[data-element-id]")) {
  const k = g.dataset.elementId;
  if (!nodes[k]) continue;
  g.setAttribute("tabindex", "0");
  g.setAttribute("role", "button");
  g.setAttribute("aria-label", nodes[k].role + " · " + nodes[k].name);
  g.onclick = (ev) => {
    ev.stopPropagation();
    select(k);
  };
  g.onkeydown = (ev) => {
    if (ev.key === "Enter" || ev.key === " ") {
      ev.preventDefault();
      select(k);
    }
  };
}
by("zoom").onchange = (e) =>
  (by("paper").style.width = Number(e.target.value)
    ? e.target.value + "px"
    : "100%");
show(
  Math.min(
    sections.length - 1,
    Number((location.hash.match(/^#phase-([0-4])$/) || [])[1] || 0),
  ),
);
