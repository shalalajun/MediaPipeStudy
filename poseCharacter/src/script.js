import * as THREE from 'three'
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js'
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js'
import { VertexNormalsHelper } from 'three/examples/jsm/helpers/VertexNormalsHelper.js'

import * as dat from 'lil-gui'

import {
    PoseLandmarker,
    FilesetResolver,
    DrawingUtils
  } from '@mediapipe/tasks-vision/vision_bundle'



let model;
let init = false;
let body; 
let poseLandmarker;
let runningMode = "IMAGE";
let enableWebcamButton;
let webcamRunning = false;
const videoHeight = "360px";
const videoWidth = "480px";

const demosSection = document.getElementById("demos");

const createPoseLandmarker = async () => {
    const vision = await FilesetResolver.forVisionTasks(
      "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0/wasm"
    );
    poseLandmarker = await PoseLandmarker.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath: `https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task`,
        delegate: "GPU"
      },
      runningMode: runningMode,
      numPoses: 2
    });
  //demosSection.classList.remove("invisible");
  };
  createPoseLandmarker();

const video = document.getElementById("webcam");
const canvasElement = document.getElementById(
    "output_canvas"
  );
const canvasCtx = canvasElement.getContext("2d");
const drawingUtils = new DrawingUtils(canvasCtx);

const hasGetUserMedia = () => !!navigator.mediaDevices?.getUserMedia;
if (hasGetUserMedia()) {
    enableWebcamButton = document.getElementById("webcamButton");
    enableWebcamButton.addEventListener("click", enableCam);
  } else {
    console.warn("getUserMedia() is not supported by your browser");
  }
function enableCam(event) {
if (!poseLandmarker) {
    console.log("Wait! poseLandmaker not loaded yet.");
    return;
}

if (webcamRunning === true) {
    webcamRunning = false;
    enableWebcamButton.innerText = "ENABLE PREDICTIONS";
} else {
    webcamRunning = true;
    enableWebcamButton.innerText = "DISABLE PREDICTIONS";
}

const constraints = {
    video: true
  };

  // Activate the webcam stream.
  navigator.mediaDevices.getUserMedia(constraints).then((stream) => {
    video.srcObject = stream;
    video.addEventListener("loadeddata", predictWebcam);
  });
}

let lastVideoTime = -1;
async function predictWebcam() {
  canvasElement.style.height = videoHeight;
  video.style.height = videoHeight;
  canvasElement.style.width = videoWidth;
  video.style.width = videoWidth;
  // Now let's start detecting the stream.
  if (runningMode === "IMAGE") {
    runningMode = "VIDEO";
    await poseLandmarker.setOptions({ runningMode: "VIDEO" });
  }
  let startTimeMs = performance.now();
  if (lastVideoTime !== video.currentTime) {
    lastVideoTime = video.currentTime;
    poseLandmarker.detectForVideo(video, startTimeMs, (result) => {
      canvasCtx.save();
      canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
      let body_poses = []; 
      for (const landmark of result.landmarks) {
       // const { x, y, z } = landmark; // x, y, z 값을 추출
        body_poses.push(landmark);
        drawingUtils.drawLandmarks(landmark, {
            radius: (data) => {
                if(data && data.from && data.from.z) {
                    return DrawingUtils.lerp(data.from.z, -0.15, 0.1, 5, 1);
                }
            }
        });

      //  
      update_data(body_poses);  

        drawingUtils.drawConnectors(landmark, PoseLandmarker.POSE_CONNECTIONS);
      }
      canvasCtx.restore();
    });
  }
  if (webcamRunning === true) {
    window.requestAnimationFrame(predictWebcam);
  }
}



/**
 * Base
 */
// Debug
const gui = new dat.GUI()

// Canvas
const canvas = document.querySelector('canvas.webgl')

// Scene
const scene = new THREE.Scene()

/**
 * Textures
 */
const textureLoader = new THREE.TextureLoader()

/**
 * Test mesh
 */
// Geometry
const geometry = new THREE.PlaneGeometry(1, 1, 32, 32)

// Material
const material = new THREE.MeshBasicMaterial()

// Mesh
const MODEL_PATH = 'stacy_lightweight.glb';
let stacy_txt = new THREE.TextureLoader().load('https://s3-us-west-2.amazonaws.com/s.cdpn.io/1376484/stacy.jpg');
stacy_txt.flipY = false;

const stacy_mtl = new THREE.MeshPhongMaterial({
    map: stacy_txt,
    color: 0xffffff,
    skinning: true
});

var model_loader = new GLTFLoader();

let neck, head, right_shoulder, left_shoulder, right_arm, right_fore_arm, left_arm, left_fore_arm, right_hand, left_hand;

let right_hand_index_4, right_thumb_4, left_thumb_4;

var helper_axes = new THREE.AxesHelper(80);

let right_leg, spine_1, spine_2, spine, hips;


model_loader.load(
    MODEL_PATH,
    (gltf) => {
        model = gltf.scene;
       
        model.traverse(o => {

            if (o.isMesh) {
                o.castShadow = true;
                o.receiveShadow = true;
                o.material = stacy_mtl;
            }
            if (o.isBone && o.name === 'mixamorigNeck') {
                neck = o;
            }
           
            else if (o.isBone && o.name === 'mixamorigHead') {
                head = o;
            }
            else if (o.isBone && o.name === 'mixamorigRightShoulder') {
                right_shoulder = o;
            }
            else if (o.isBone && o.name === 'mixamorigLeftShoulder') {
                left_shoulder = o;
            }

            else if (o.isBone && o.name === 'mixamorigRightArm') {
                right_arm = o;
            }
            else if (o.isBone && o.name === 'mixamorigRightForeArm') {
                right_fore_arm = o;
            }
            else if (o.isBone && o.name === 'mixamorigLeftArm') {
                left_arm = o;
            }
            else if (o.isBone && o.name === 'mixamorigLeftForeArm') {
                left_fore_arm = o;
            }

            else if (o.isBone && o.name === 'mixamorigRightHand') {
                right_hand = o;
            }
            else if (o.isBone && o.name === 'mixamorigLeftHand') {
                left_hand = o;
            }
            else if (o.isBone && o.name === 'mixamorigRightHandIndex4') {
                right_hand_index_4 = o;
            }
            else if (o.isBone && o.name === 'mixamorigRightThumb4') {
                right_thumb_4 = o;
                right_thumb_4.removeFromParent();
                if(right_hand)
                {
                    right_thumb_4.attach(right_hand);
                    console.log("right_thumb_4 attached to right_hand");
                }
               
            }
            else if (o.isBone && o.name === 'mixamorigLeftThumb4') {
                left_thumb_4 = o;
                left_thumb_4.removeFromParent();
                if(left_hand)
                {
                    left_thumb_4.attach(right_hand);
                    console.log("left_thumb_4 attached to right_hand");
                }

            }

            else if (o.isBone && o.name === 'mixamorigRightLeg') {
                right_leg = o;
            }
            else if (o.isBone && o.name === 'mixamorigSpine1') {
                spine_1 = o;
            }
            else if (o.isBone && o.name === 'mixamorigSpine2') {
                spine_2 = o;
            }
            else if (o.isBone && o.name === 'mixamorigSpine') {
                spine = o;
                spine.add(helper_axes);
            }
            else if (o.isBone && o.name === 'mixamorigHips') {
                hips = o;
            }
        });

        model.scale.set(4, 7, 4);
        model.position.x = 0;
        model.position.y = -2; // -11
        model.position.z = 0;

        scene.add(model);

        const skeletonHelper = new THREE.SkeletonHelper(model);
        scene.add(skeletonHelper);
        // model.visible = false;
    },
    undefined, // We don't need function
    function (error) {
        console.error(error);
    }
);




/**
 * Sizes
 */
const sizes = {
    width: window.innerWidth,
    height: window.innerHeight
}

window.addEventListener('resize', () =>
{
    // Update sizes
    sizes.width = window.innerWidth
    sizes.height = window.innerHeight

    // Update camera
    camera.aspect = sizes.width / sizes.height
    camera.updateProjectionMatrix()

    // Update renderer
    renderer.setSize(sizes.width, sizes.height)
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
})

/**
 * Camera
 */
// Base camera
const camera = new THREE.PerspectiveCamera(75, sizes.width / sizes.height, 0.1, 100)
camera.position.set(0.0, 5.0, 10)
scene.add(camera)


/**
 * light
 */
let hemiLight = new THREE.HemisphereLight(0xffffff, 0xffffff, 0.61);
hemiLight.position.set(0, 50, 0);
// Add hemisphere light to scene
scene.add(hemiLight);


let d = 8.25;
let dirLight = new THREE.DirectionalLight(0xffffff, 0.54);
dirLight.position.set(-8, 12, 8);
dirLight.castShadow = true;
dirLight.shadow.mapSize = new THREE.Vector2(1024, 1024);
dirLight.shadow.camera.near = 0.1;
dirLight.shadow.camera.far = 1500;
dirLight.shadow.camera.left = d * -1;
dirLight.shadow.camera.right = d;
dirLight.shadow.camera.top = d;
dirLight.shadow.camera.bottom = d * -1;
// Add directional Light to scene
scene.add(dirLight);




// Controls
const controls = new OrbitControls(camera, canvas)
controls.enableDamping = true

/**
 * Renderer
 */
const renderer = new THREE.WebGLRenderer({
    canvas: canvas
})
renderer.setSize(sizes.width, sizes.height)
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))

/**
 * Animate
 */
const clock = new THREE.Clock()

const tick = () =>
{
    const elapsedTime = clock.getElapsedTime()

    // Update controls
    controls.update()
    if (model) {
        model.updateMatrixWorld();
      }

    // Render
    renderer.render(scene, camera)

    // Call tick again on the next frame
    window.requestAnimationFrame(tick)
}
tick()

let body_pose;
function update_data(bodyPose)
{
    body_pose = bodyPose;

    
    if(neck)
    {
       
    }
    if (right_arm) {
        poseAngles(right_arm);
    }
    if (right_fore_arm) {
        poseAngles(right_fore_arm);
    }
    if (left_arm) {
        poseAngles(left_arm);
    }
    if (left_fore_arm) {
        poseAngles(left_fore_arm);
    }
    if (right_hand) {
        poseAngles(right_hand);
    }
    // if (right_shoulder) {
    //     poseAngles(right_shoulder);
    // }

    // if (left_shoulder) {
    //     poseAngles(left_shoulder);
    // }


    // let neck, head, right_shoulder, left_shoulder, right_arm, right_fore_arm, left_arm, left_fore_arm, right_hand, left_hand;

    // let right_hand_index_4, right_thumb_4, left_thumb_4;
    
    // var helper_axes = new THREE.AxesHelper(80);
    
    // let right_leg, spine_1, spine_2, spine, hips;
}

// slice는 어레이에서 잘라서 새로운 배열을 만든다. 즉 body_pose[0][11].x 는 왼쪽 어깨의 x,y,z,w 에서 x를 가르킨다.
// function poseAngles(joint) {
//     if (body_pose[0].length === 0) return;

   

//     // const pose_left_shoulder = new THREE.Vector3(body_pose[0][0][11].slice()[0], -body_pose[0][0][11].slice()[1], -body_pose[0][0][11].slice()[2]);

//     const pose_left_shoulder = new THREE.Vector3(body_pose[0][0][11].x, -body_pose[0][0][11].y, -body_pose[0][0][11].z);
   
//     const pose_right_shoulder = body_pose[0][12] ? new THREE.Vector3(body_pose[0][12].x, -body_pose[0][12].y, -body_pose[0][12].z);

//     const pose_left_elbow = body_pose[0][13] ? new THREE.Vector3(body_pose[0][13].x, -body_pose[0][13].y, -body_pose[0][13].z);
//     const pose_right_elbow = body_pose[0][14] ? new THREE.Vector3(body_pose[0][14].x, -body_pose[0][14].y, -body_pose[0][14].z);
//     const pose_left_hand = body_pose[0][15] ? new THREE.Vector3(body_pose[0][15].x, -body_pose[0][15].y, -body_pose[0][15].z);
//     const pose_right_hand = body_pose[0][16] ? new THREE.Vector3(body_pose[0][16].x, -body_pose[0][16].y, -body_pose[0][16].z);
//     const pose_left_hand_thumb_4 = body_pose[0][21] ? new THREE.Vector3(body_pose[0][21].x, -body_pose[0][21].y, -body_pose[0][21].z);
//     const pose_right_hand_thumb_4 = body_pose[0][22] ? new THREE.Vector3(body_pose[0][22].x, -body_pose[0][22].y, -body_pose[0][22].z);
//     const pose_left_hip = body_pose[0][23] ? new THREE.Vector3(body_pose[0][23].x, -body_pose[0][23].y, -body_pose[0][23].z);
//     const pose_right_hip = body_pose[0][24] ? new THREE.Vector3(body_pose[0][24].x, -body_pose[0][24].y, -body_pose[0][24].z);
  
//     const pose_hips = pose_left_hip && pose_right_hip ? ((new THREE.Vector3).copy(pose_left_hip)).add(pose_right_hip).multiplyScalar(0.5);
//     const pose_spine_2 = pose_right_shoulder && pose_left_shoulder ? ((new THREE.Vector3).copy(pose_right_shoulder)).add(pose_left_shoulder).multiplyScalar(0.5);
  
//     let point_parent, point_articulation, point_child;
  
//     if (joint === neck) {
//       point_parent = pose_hips;
//       point_articulation = pose_spine_2;
//       point_child = pose_right_elbow;
  
//       const vec_parent = point_articulation && point_parent ? (new THREE.Vector3).subVectors(point_articulation, point_parent).multiplyScalar(0.375);
//       const vec_bone = point_child && point_articulation ? (new THREE.Vector3).subVectors(point_child, point_articulation);
  
//       if (vec_parent && vec_bone) {
//         setJointAnglesFromVects(joint, vec_bone, vec_parent);
//       }
//     } else if (joint === right_arm) {
//       point_parent = pose_spine_2;
//       point_articulation = pose_right_shoulder;
//       point_child = pose_right_elbow;
//     } else if (joint === left_arm) {
//       point_parent = pose_spine_2;
//       point_articulation = pose_left_shoulder;
//       point_child = pose_left_elbow;
//     } else if (joint === right_fore_arm) {
//       point_parent = pose_right_shoulder;
//       point_articulation = pose_right_elbow;
//       point_child = pose_right_hand;
//     } else if (joint === left_fore_arm) {
//       point_parent = pose_left_shoulder;
//       point_articulation = pose_left_elbow;
//       point_child = pose_left_hand;
//     } else if (joint === right_hand) {
//       point_parent = pose_right_elbow;
//       point_articulation = pose_right_hand;
//       point_child = pose_right_hand_thumb_4;
//     } else if (joint === left_hand) {
//       point_parent = pose_left_elbow;
//       point_articulation = pose_left_hand;
//       point_child = pose_left_hand_thumb_4;
//     }
  
//     const vec_parent = point_articulation && point_parent ? (new THREE.Vector3).subVectors(point_articulation, point_parent);
//     const vec_bone = point_child && point_articulation ? (new THREE.Vector3).subVectors(point_child, point_articulation);
  
//     if (vec_parent && vec_bone) {
//       setJointAnglesFromVects(joint, vec_parent, vec_bone);
//     }
//   }

  function poseAngles(joint) {
    if (body_pose[0].length === 0) return;

    // const pose_left_shoulder = new THREE.Vector3(body_pose[0][0][11].slice()[0], -body_pose[0][0][11].slice()[1], -body_pose[0][0][11].slice()[2]);

    const pose_left_shoulder = new THREE.Vector3(body_pose[0][11].x, -body_pose[0][11].y, -body_pose[0][11].z);

    const pose_right_shoulder = new THREE.Vector3(body_pose[0][12].x, -body_pose[0][12].y, -body_pose[0][12].z);
    

    const pose_left_elbow = new THREE.Vector3(body_pose[0][13].x, -body_pose[0][13].y, -body_pose[0][13].z);

    const pose_right_elbow = new THREE.Vector3(body_pose[0][14].x, -body_pose[0][14].y, -body_pose[0][14].z);
    const pose_left_hand = new THREE.Vector3(body_pose[0][15].x, -body_pose[0][15].y, -body_pose[0][15].z);
    const pose_right_hand = new THREE.Vector3(body_pose[0][16].x, -body_pose[0][16].y, -body_pose[0][16].z);
    const pose_left_hand_thumb_4 = new THREE.Vector3(body_pose[0][21].x, -body_pose[0][21].y, -body_pose[0][21].z);
    const pose_right_hand_thumb_4 = new THREE.Vector3(body_pose[0][22].x, -body_pose[0][22].y, -body_pose[0][22].z);
    const pose_left_hip = new THREE.Vector3(body_pose[0][23].x, -body_pose[0][23].y, -body_pose[0][23].z);
    const pose_right_hip = new THREE.Vector3(body_pose[0][24].x, -body_pose[0][24].y, -body_pose[0][24].z);
  
    const pose_hips = ((new THREE.Vector3).copy(pose_left_hip)).add(pose_right_hip).multiplyScalar(0.5); 
 
    const pose_spine_2 = ((new THREE.Vector3).copy(pose_right_shoulder)).add(pose_left_shoulder).multiplyScalar(0.5);
   //.multiplyScalar(0.728);
      
       var point_parent;
       var point_articulation;
       var point_child;
       if (joint == neck) {
           var point_parent = pose_hips;
           var point_articulation = pose_spine_2;
           var point_arm = pose_right_elbow;
           
           const vec_parent = (new THREE.Vector3).subVectors(point_articulation, point_parent).multiplyScalar(0.375);
           const vec_bone = (new THREE.Vector3).subVectors(point_arm, point_articulation);
 
           setJointAnglesFromVects(joint, vec_bone, vec_parent);
       }
       else if (joint == right_arm) {
           point_parent = pose_spine_2;
           point_articulation = pose_right_shoulder;
           point_child = pose_right_elbow;
       }
      
       else if (joint == left_arm) {
           point_parent = pose_spine_2;
           point_articulation = pose_left_shoulder;
           point_child = pose_left_elbow;
       }
       else if (joint == right_fore_arm) {
           point_parent = pose_right_shoulder;
           point_articulation = pose_right_elbow;
           point_child = pose_right_hand;
       }
       else if (joint == left_fore_arm) {
           point_parent = pose_left_shoulder;
           point_articulation = pose_left_elbow;
           point_child = pose_left_hand;
       }
       else if (joint == right_hand) {
           point_parent = pose_right_elbow;
           point_articulation = pose_right_hand;
           point_child = pose_right_hand_thumb_4;
       }
       else if (joint == left_hand) {
           point_parent = pose_left_elbow;
           point_articulation = pose_left_hand;
           point_child = pose_left_hand_thumb_4;
       }
       const vec_parent = (new THREE.Vector3).subVectors(point_articulation, point_parent);
       const vec_bone = (new THREE.Vector3).subVectors(point_child, point_articulation);
       setJointAnglesFromVects(joint, vec_parent, vec_bone);
  }


function setJointAnglesFromVects(joint, vec_parent_world, vec_child_world)
{
   const vec_child_local = joint.parent.clone().worldToLocal(vec_child_world.clone());
   const vec_parent_local = joint.parent.clone().worldToLocal(vec_parent_world.clone());
   var quat_pose_rot = new THREE.Quaternion();
   quat_pose_rot.setFromUnitVectors(vec_parent_local.clone().normalize(), vec_child_local.clone().normalize());
   joint.quaternion.rotateTowards(quat_pose_rot.clone(), 0.05);
}

