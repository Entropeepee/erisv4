import React, { useEffect, useRef, useState } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { useGLTF } from '@react-three/drei';
import * as THREE from 'three';

type Avatar3DProps = {
  isTalking: boolean;
  vitals: { dCdX: number; coherence: number };
  analyser?: AnalyserNode | null;
};

const GLBAvatar = ({ isTalking, vitals, analyser }: Avatar3DProps) => {
  const { scene } = useGLTF('/models/avatar.glb');
  const timeRef = useRef(0);
  const blinkRef = useRef({ timeUntilNext: 2.0, duration: 0.1, isBlinking: false });
  const headMeshRef = useRef<THREE.SkinnedMesh | null>(null);

  useEffect(() => {
    if (scene) {
      // Fix orientation if needed
      scene.rotation.y = 0;
      scene.position.y = -1.2;

      // Find the mesh with morph targets (usually Wolf3D_Head or similar in RPM)
      scene.traverse((node) => {
        const mesh = node as THREE.SkinnedMesh;
        if (mesh.isMesh && mesh.morphTargetDictionary) {
          if (mesh.name.includes('Head') || mesh.morphTargetDictionary['eyeBlinkLeft'] !== undefined) {
            headMeshRef.current = mesh;
          }
        }
        // Attempt to find arm bones for lotus pose
        if (node.isBone) {
          if (node.name.includes('LeftArm')) node.rotation.z = 1.0;
          if (node.name.includes('RightArm')) node.rotation.z = -1.0;
        }
      });
    }
  }, [scene]);

  useFrame((state, delta) => {
    if (!scene) return;
    timeRef.current += delta;
    const t = timeRef.current;

    // 1. Procedural Hover (Lotus Breathing)
    const speedMultiplier = 1.0 + (vitals.dCdX * 0.5);
    scene.position.y = -1.2 + Math.sin(t * 2.0 * speedMultiplier) * 0.05;

    // Handle morph targets if head mesh is found
    if (headMeshRef.current && headMeshRef.current.morphTargetInfluences && headMeshRef.current.morphTargetDictionary) {
      const dict = headMeshRef.current.morphTargetDictionary;
      const influences = headMeshRef.current.morphTargetInfluences;

      // 2. Procedural Blinking (ARKit: eyeBlinkLeft / eyeBlinkRight)
      const blink = blinkRef.current;
      let blinkValue = 0.0;
      if (!blink.isBlinking) {
        blink.timeUntilNext -= delta;
        if (blink.timeUntilNext <= 0) {
          blink.isBlinking = true;
          blink.duration = 0.1;
          blinkValue = 1.0;
        }
      } else {
        blink.duration -= delta;
        if (blink.duration <= 0) {
          blink.isBlinking = false;
          blink.timeUntilNext = 2.0 + Math.random() * 3.0;
          blinkValue = 0.0;
        } else {
            blinkValue = 1.0;
        }
      }
      
      if (dict['eyeBlinkLeft'] !== undefined) influences[dict['eyeBlinkLeft']] = blinkValue;
      if (dict['eyeBlinkRight'] !== undefined) influences[dict['eyeBlinkRight']] = blinkValue;

      // 3. Procedural Lip Sync via Web Audio API
      if (isTalking && analyser) {
        const dataArray = new Uint8Array(analyser.frequencyBinCount);
        analyser.getByteFrequencyData(dataArray);

        const lowerHalf = dataArray.slice(0, dataArray.length / 2);
        const upperHalf = dataArray.slice(dataArray.length / 2);
        const lowerAvg = lowerHalf.reduce((a, b) => a + b, 0) / lowerHalf.length;
        const upperAvg = upperHalf.reduce((a, b) => a + b, 0) / upperHalf.length;
        const totalAvg = dataArray.reduce((a, b) => a + b, 0) / dataArray.length;

        const volume = Math.min(totalAvg / 128.0, 1.0);
        const lowF = Math.min(lowerAvg / 128.0, 1.0);
        const highF = Math.min(upperAvg / 128.0, 1.0);

        const lerpFactor = 0.3;
        
        const jawOpenIdx = dict['jawOpen'] !== undefined ? dict['jawOpen'] : dict['mouthOpen'];
        if (jawOpenIdx !== undefined) influences[jawOpenIdx] = THREE.MathUtils.lerp(influences[jawOpenIdx], volume * 0.8, lerpFactor);
        
        if (dict['mouthFunnel'] !== undefined) influences[dict['mouthFunnel']] = THREE.MathUtils.lerp(influences[dict['mouthFunnel']], lowF * 0.5, lerpFactor);
        if (dict['mouthPucker'] !== undefined) influences[dict['mouthPucker']] = THREE.MathUtils.lerp(influences[dict['mouthPucker']], highF * 0.5, lerpFactor);
        if (dict['mouthSmile'] !== undefined) influences[dict['mouthSmile']] = THREE.MathUtils.lerp(influences[dict['mouthSmile']], volume * 0.3, lerpFactor);
      } else {
        const lerpFactor = 0.1;
        ['jawOpen', 'mouthOpen', 'mouthFunnel', 'mouthPucker', 'mouthSmile'].forEach(key => {
          if (dict[key] !== undefined) influences[dict[key]] = THREE.MathUtils.lerp(influences[dict[key]], 0.0, lerpFactor);
        });
      }
    }
  });

  return <primitive object={scene} />;
};

const Avatar3D = ({ isTalking, vitals, analyser }: Avatar3DProps) => {
  return (
    <div style={{ width: '100%', height: '300px', borderRadius: '12px', overflow: 'hidden', background: 'radial-gradient(circle at center, #1a1a2e 0%, #0a0a12 100%)', boxShadow: '0 0 20px rgba(0, 255, 255, 0.1)' }}>
      <Canvas camera={{ position: [0, 0, 1.5], fov: 40 }}>
        <ambientLight intensity={0.5} />
        <directionalLight position={[1, 1, 1]} intensity={1.0} color="#e0ffff" />
        <directionalLight position={[-1, 0.5, -1]} intensity={2.0} color="#ff00ff" />
        <GLBAvatar isTalking={isTalking} vitals={vitals} analyser={analyser} />
      </Canvas>
    </div>
  );
};
export default Avatar3D;
