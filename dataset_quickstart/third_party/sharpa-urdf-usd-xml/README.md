# Sharpa Wave Hand URDF USD and XML Files

## Model Description

1. URDF Files:
   - `src/left_sharpa_wave/left_sharpa_wave.urdf` for left hand.
   - `src/right_sharpa_wave/right_sharpa_wave.urdf` for right hand.
   - `src/dual_sharpa_wave.urdf` for dual hand.
2. XML Files:
   - `src/left_sharpa_wave/left_sharpa_wave.xml` for left hand.
   - `src/right_sharpa_wave/right_sharpa_wave.xml` for right hand.
   - `src/dual_sharpa_wave.xml` for dual hand.
3. USD Files:
  - `src/left_sharpa_wave/left_sharpa_wave.usda` for left hand.
  - `src/right_sharpa_wave/right_sharpa_wave.usda` for right hand.
  - `src/dual_sharpa_wave.usda` for dual hand.
4. Wrist/Flange Variants:
  - `with_wrist`: includes the wrist structure in front of the hand base. Use this version when your setup requires wrist kinematics, wrist collision geometry, or a direct wrist-level interface in simulation/control.
  - `with_flange`: uses a flange-level mounting interface as the hand root and does not include the wrist structure. Use this version when the hand is mounted directly to an external arm/end-effector flange.
  - Both variants are provided for left and right hands in `urdf`, `xml`, and `usd` formats, for example:
    - `src/right_sharpa_wave/right_sharpa_wave_with_wrist.urdf`
    - `src/right_sharpa_wave/right_sharpa_wave_with_flange.urdf`
    - `src/right_sharpa_wave/right_sharpa_wave_with_wrist.xml`
    - `src/right_sharpa_wave/right_sharpa_wave_with_flange.xml`
    - `src/right_sharpa_wave/right_sharpa_wave_with_wrist.usda`
    - `src/right_sharpa_wave/right_sharpa_wave_with_flange.usda`
5. Float base / Dual-hand setup:
  - `float base`: adds 6 DoF in front of the wrist base link, which makes it easier to directly control the wrist pose.
  - `dual_sharpa_wave`: combines the left and right hands into one model, and adds 6 DoF in front of each hand base link, which is convenient for bimanual use cases.
6. Each fingertip has three coordinate frames:  
   - `XXX_DP` frame: satisfies the MDH convention,  
   - `XXX_elastomer` frame: used for tactile sensors,  
   - `XXX_fingertip` frame: used for IK fingertip position annotation.  
   Please choose according to your needs.

## Dynamics Parameter Description

### Isaac Lab Dynamics
1. The dynamics parameters for Isaac Lab have already been configured in the USD files. You can directly use `IdealPDActuator` with both `stiffness` and `damping` set to `None`. By default, these dynamics parameters are tuned to match `position mode`. If you use other control modes, additional retuning may be required.
3. `XXX_MITmode.usda` corresponds to the hardware MIT mode and is closer to PD control behavior in simulation. When using this mode, a control frequency higher than `80 Hz` is recommended.

### XML Parameter Description
1. Currently, `armature`, `frictionloss`, `actuatorfrcrange`, and `damping` are set based on the calibration results from IsaacLab, which have been verified to be quite close to those of the real robot.   
2. For rigid body collision parameters, everything is set to default except for the fingertip elastomer, which has a modified `solref` to make it softer and more elastic. Please contact lei.su@sharpa.com if you notice any issues.  

## RViz Instructions
1. Install ROS2 and RViz2.  
2. In the `SharpaWave_URDF_XML_USD` folder, run:  

```bash
colcon build
source install/setup.bash
ros2 launch left_sharpa_wave display.launch.py  # View left hand URDF in RViz
ros2 launch right_sharpa_wave display.launch.py # View right hand URDF in RViz
```


