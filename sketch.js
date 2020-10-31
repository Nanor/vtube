let video;
let poseNet;
let faceApi;

let pose;
let face;

const smoothing = 0.3;

let parts = {};

function preload() {
  parts = {
    body: loadImage("assets/avatar/body.PNG"),
    nose: loadImage("assets/avatar/nose.PNG"),
    leftEye: loadImage("assets/avatar/leftEye.PNG"),
    rightEye: loadImage("assets/avatar/rightEye.PNG"),
    background: loadImage("assets/avatar/background.PNG"),
    head: loadImage("assets/avatar/head.PNG"),
    mouth: loadImage("assets/avatar/mouth.PNG"),
  };
}

function setup() {
  createCanvas(640, 480);
  video = createCapture(VIDEO);
  video.hide();

  poseNet = ml5.poseNet(video, poseNetReady);
  poseNet.on("pose", gotPoses);

  const faceOptions = {
    withLandmarks: true,
    withExpressions: false,
    withDescriptors: false,
  };

  faceApi = ml5.faceApi(video, faceOptions, faceApiReady);
}

function poseNetReady() {
  console.log("poseNet ready");
}

function faceApiReady() {
  console.log("faceApi ready");

  faceApi.detectSingle(gotFace);
}

function gotPoses(poses) {
  if (poses.length > 0) {
    if (!pose) {
      console.log(poses[0].pose);
    }

    const points = poses[0].pose;

    const rootX = lerp(points.leftShoulder.x, points.rightShoulder.x, 0.5);
    const rootY = lerp(points.leftShoulder.y, points.rightShoulder.y, 0.5);
    const shoulderDist = dist(
      points.leftShoulder.x,
      points.leftShoulder.y,
      points.rightShoulder.x,
      points.rightShoulder.y
    );

    const headX = lerp(points.leftEye.x, points.rightEye.x, 0.5);
    const headY = lerp(points.leftEye.y, points.rightEye.y, 0.5);
    const eyeDist = dist(
      points.leftEye.x,
      points.leftEye.y,
      points.rightEye.x,
      points.rightEye.y
    );

    const newPose = {
      root: {
        pos: {
          x: rootX,
          y: rootY,
          z: shoulderDist,
        },
        rot: { z: (points.leftShoulder.y - points.rightShoulder.y) / 200 },
      },
      head: {
        pos: {
          x: headX,
          y: headY,
          z: eyeDist,
        },
        rot: {
          x: points.nose.y - headY - eyeDist,
          y: points.nose.x - headX,
          z: (points.leftEye.y - points.rightEye.y) / 50,
        },
      },
    };

    if (pose) {
      pose = smoothPose(pose, newPose, smoothing);
    } else {
      pose = newPose;
    }
  }
}

function smoothPose(pose, newPose, smoothing) {
  if (!pose) return newPose;

  const lerpObj = (obj1, obj2, s) => {
    const newObj = {};

    Object.keys(obj1).forEach((key) => {
      if (typeof obj1[key] === "number") {
        newObj[key] = lerp(obj1[key], obj2[key], s);
      } else {
        newObj[key] = lerpObj(obj1[key], obj2[key], s);
      }
    });

    return newObj;
  };

  return lerpObj(pose, newPose, smoothing);
}

function gotFace(error, result) {
  if (error) {
    console.log(error);
    faceApi.detectSingle(gotFace);
    return;
  }

  if (!face) console.log(result);

  const partSize = (part) => {
    const partSum = part.reduce(([x, y], { _x, _y }) => [x + _x, y + _y], [
      0,
      0,
    ]);

    const partMiddle = [partSum[0] / part.length, partSum[1] / part.length];

    const partDists = part.map(({ _x, _y }) => dist(_x, _y, ...partMiddle));

    return Math.min(...partDists) / Math.max(...partDists);
  };

  const mouthOpen = partSize(result.parts.mouth);
  const leftEyeOpen = partSize(result.parts.rightEye);
  const rightEyeOpen = partSize(result.parts.leftEye);

  face = smoothPose(
    face,
    {
      mouth: mouthOpen,
      leftEye: leftEyeOpen,
      rightEye: rightEyeOpen,
    },
    smoothing
  );

  faceApi.detectSingle(gotFace);
}

function draw() {
  image(parts.background, 0, 0);

  translate(width / 2, height / 2);
  // scale(-1, 1);
  translate(-width / 2, -height / 2);

  if (pose && face) {
    stroke(0);
    fill(255);

    push();
    translate(0, 50);

    translate(pose.root.pos.x, pose.root.pos.y);
    rotate(pose.root.rot.z);

    scale(0.6);

    const offset = {
      x: -parts.body.width / 2 - 50,
      y: -parts.body.height / 2 + 50,
    };

    image(parts.body, offset.x, offset.y - 100);

    rotate(pose.head.rot.z);
    translate(0, -150);

    image(parts.head, offset.x, offset.y + 50);

    translate(
      offset.x + pose.head.rot.y * 2,
      offset.y + 60 + pose.head.rot.x * 2
    );

    image(parts.nose, 0, 0);

    push();
    translate(0, 410);
    scale(1, face.leftEye * 2);
    translate(0, -410);

    image(parts.leftEye, 0, 0);
    pop();

    push();
    translate(0, 410);
    scale(1, face.rightEye * 2);
    translate(0, -410);

    image(parts.rightEye, 0, 0);
    pop();
    translate(0, 500);
    scale(1, face.mouth * 1.2 + 0.2);
    translate(0, -500);

    image(parts.mouth, 0, 0);

    pop();
  }
}
