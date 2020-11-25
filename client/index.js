let controls = [
  { name: "root_angle", min: -10, max: 10 },
  { name: "head_tilt", min: -10, max: 10 },
  { name: "head_look", min: -10, max: 10 },
  { name: "head_nod", min: -10, max: 10 },
  { name: "left_eye_x", min: -10, max: 10 },
  { name: "left_eye_y", min: -10, max: 10 },
  { name: "right_eye_y", min: -10, max: 10 },
  { name: "right_eye_x", min: -10, max: 10 },
];

window.onload = () => {
  let object = document.getElementById("svg");
  let svg = object.contentDocument.getElementsByTagName("svg")[0];
  document.getElementById("canvas").appendChild(svg);
  object.remove();

  let controlsDiv = document.getElementById("controls");

  controls.forEach(({ name, min, max }) => {
    let label = document.createElement("label");
    label.textContent = name;

    let elem = document.createElement("input");
    elem.type = "range";
    elem.id = name;
    elem.min = min;
    elem.max = max;
    elem.value = 0;

    let container = document.createElement("div");
    container.appendChild(label);
    container.appendChild(elem);

    controlsDiv.appendChild(container);
  });

  window.setInterval(update, 100);
};

const getSliderParams = () =>
  controls.reduce(
    (acc, { name }) => ({
      ...acc,
      [name]: document.getElementById(name).value,
    }),
    {}
  );

function update() {
  let {
    root_angle,
    head_tilt,
    head_look,
    head_nod,
    left_eye_x,
    left_eye_y,
    right_eye_x,
    right_eye_y,
  } = window.pose || getSliderParams();

  anime({
    targets: "#root",
    rotate: {
      value: `${root_angle * 0.2}`,
    },
  });

  anime({
    targets: "#head",
    rotate: {
      value: `${head_tilt}`,
    },
  });

  [
    ["left_eye", 0.4],
    ["right_eye", 0.4],
    ["nose", 0.6],
    ["mouth", 0.4],
    ["face", 0.2],
    ["hair_back", 0.2],
    ["left_ear", 0.1],
    ["right_ear", 0.1],
    ["hair_mid", 0.1],
    ["hair_front", 0.2],
    ["hair_under", 0.2],
  ].forEach(([id, parallax]) =>
    anime({
      targets: `#${id}`,
      translateX: { value: `${head_look * parallax}px` },
      translateY: { value: `${head_nod * parallax}px` },
    })
  );

  // anime({
  //   targets: "#nose",
  //   scaleX: { value: head_look >= 0 ? 1 : -1 },
  // });

  anime({
    targets: "#left_iris",
    translateX: { value: `${left_eye_x * 0.45}px` },
    translateY: { value: `${left_eye_y * 0.1}px` },
  });
  anime({
    targets: "#right_iris",
    translateX: { value: `${right_eye_x * 0.45}px` },
    translateY: { value: `${right_eye_y * 0.1}px` },
  });
}
